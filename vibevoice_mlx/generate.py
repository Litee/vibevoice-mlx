"""VibeVoice generation pipeline — autoregressive TTS with DPM-Solver++ diffusion."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from tqdm import tqdm

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .model import KVCache, VibeVoiceConfig, VibeVoiceModel, compute_rope
from .streaming_vae import StreamingVAEDecoder
from .fast_forward import FastLM, FastDiffusionHead, PreparedDiffusionConditioning


# ---------------------------------------------------------------------------
# DPM-Solver++ schedule (precomputed in float64)
# ---------------------------------------------------------------------------

DDPM_STEPS = 1000
VAE_DIM = 64

_AC64 = np.cos((np.arange(DDPM_STEPS + 1, dtype=np.float64) / DDPM_STEPS + 0.008) / 1.008 * np.pi / 2) ** 2
# Discretize cosine intervals before taking the cumulative product. The beta
# cap keeps the final training timestep at finite, nonzero signal strength.
_BETAS_NP = np.minimum(1.0 - _AC64[1:] / _AC64[:-1], 0.999)
_AC64 = np.cumprod(1.0 - _BETAS_NP)
_ALPHA_NP = np.sqrt(_AC64)
_SIGMA_NP = np.sqrt(1.0 - _AC64)
_LAMBDA_NP = np.log(_ALPHA_NP / np.maximum(_SIGMA_NP, 1e-10))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class GenerationOptions:
    solver: str = "dpm"            # "dpm" or "sde"
    diffusion_steps: int = 10      # DPM-Solver++ steps
    cfg_scale: float = 1.3         # Classifier-free guidance
    max_speech_tokens: int = 200   # Safety limit
    silence_detection: bool = False  # Boost speech_end logit on sustained silence
    trim_trailing_silence: bool | None = None  # Post-gen trim (None = follow silence_detection)
    silence_threshold: float = 0.05  # RMS threshold for silence detection
    silence_min_duration_ms: int = 1500  # Forward scan: min silence gap to cut
    silence_pad_ms: int = 300      # Padding after detected speech end
    seed: int = 42


GenerationStopReason = Literal[
    "eos",
    "speech_end",
    "max_speech_tokens",
    "max_generation_tokens",
]


@dataclass
class GenerationMetrics:
    name: str = ""
    timings: dict = field(default_factory=dict)
    total_time: float = 0.0
    num_speech_tokens: int = 0
    num_text_tokens: int = 0
    audio_samples: int = 0
    stop_reason: GenerationStopReason | None = None

    def record(self, component: str, ms: float):
        if component not in self.timings:
            self.timings[component] = []
        self.timings[component].append(ms)

    def summary(self) -> dict:
        result = {"name": self.name}
        for k, v in self.timings.items():
            result[f"{k}_total_ms"] = sum(v)
            result[f"{k}_mean_ms"] = sum(v) / len(v) if v else 0
            result[f"{k}_count"] = len(v)
        result["total_ms"] = self.total_time
        result["speech_tokens"] = self.num_speech_tokens
        result["text_tokens"] = self.num_text_tokens
        result["audio_samples"] = self.audio_samples
        result["stop_reason"] = self.stop_reason
        result["audio_seconds"] = self.audio_samples / 24000
        if self.audio_samples > 0 and self.total_time > 0:
            audio_ms = self.audio_samples / 24000 * 1000
            result["rtf"] = audio_ms / self.total_time
        load_ms = result.get("load_total_ms", 0)
        gen_ms = self.total_time - load_ms
        result["gen_ms"] = gen_ms
        if self.audio_samples > 0 and gen_ms > 0:
            audio_ms = self.audio_samples / 24000 * 1000
            result["gen_rtf"] = audio_ms / gen_ms
        return result


# ---------------------------------------------------------------------------
# DPM-Solver++ 2M — ODE and SDE variants (all MLX, batched CFG)
# ---------------------------------------------------------------------------

def _validate_cfg_scale(cfg_scale: float) -> None:
    """Reject non-finite guidance without coercing valid scalar values."""
    try:
        finite = isinstance(
            cfg_scale, (int, float, np.integer, np.floating, np.bool_)
        ) and np.isfinite(cfg_scale)
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(
            f"cfg_scale must be a finite real number; received {cfg_scale!r}."
        )


def _validate_diffusion_steps(num_steps: int) -> int:
    """Keep rounded training timesteps distinct before the zero-noise endpoint."""
    if (
        not isinstance(num_steps, (int, np.integer))
        or isinstance(num_steps, bool)
        or not 1 <= num_steps < DDPM_STEPS
    ):
        raise ValueError(
            f"diffusion steps must be an integer between 1 and {DDPM_STEPS - 1}; "
            f"received {num_steps!r}."
        )
    return int(num_steps)


def _prepare_diffusion_conditioning(
    diff_head: Callable,
    condition: mx.array,
    timesteps: np.ndarray,
    dtype: mx.Dtype,
) -> PreparedDiffusionConditioning | None:
    # Subclasses and arbitrary callbacks may override the denoiser's behavior.
    # Optimize only the concrete built-in head; preserve every other call path.
    if type(diff_head) is not FastDiffusionHead:
        return None
    return diff_head.prepare_conditioning(
        condition, mx.array(timesteps).astype(dtype), dtype=dtype,
    )


def _dpm_denoise_step(
    diff_head: Callable,
    sample: mx.array,
    batched_cond: mx.array,
    s: int,
    cfg_scale: float,
    dtype: mx.Dtype,
    *,
    prepared: PreparedDiffusionConditioning | None = None,
    step: int = 0,
) -> mx.array:
    """Run diffusion head with batched CFG, return x0 prediction.

    No mx.eval — relies on MLX lazy evaluation to batch the entire
    diffusion solve into fewer GPU submissions.
    """
    batched_sample = mx.concatenate([sample, sample], axis=0).astype(dtype)
    if prepared is None:
        ts_mx = mx.array([float(s)]).astype(dtype)
        v_batched = diff_head(batched_sample, ts_mx, batched_cond)
    else:
        v_batched = diff_head.forward_prepared(batched_sample, prepared, step)

    v_cond = v_batched[0:1].astype(mx.float32)
    v_uncond = v_batched[1:2].astype(mx.float32)
    v = v_uncond + cfg_scale * (v_cond - v_uncond)

    alpha_s = float(_ALPHA_NP[s])
    sigma_s = float(_SIGMA_NP[s])
    return alpha_s * sample - sigma_s * v


def dpm_solver_2m(
    diff_head,
    condition: mx.array,
    neg_condition: mx.array,
    cfg_scale: float,
    num_steps: int = 10,
    seed: int = 0,
    dtype=mx.float16,
) -> mx.array:
    """ODE DPM-Solver++ 2M with batched CFG.

    Returns sample of shape (1, VAE_DIM) in float32.
    """
    _validate_cfg_scale(cfg_scale)
    num_steps = _validate_diffusion_steps(num_steps)
    t_schedule = np.round(
        np.linspace(0, DDPM_STEPS - 1, num_steps + 1)
    ).astype(np.int64)[::-1]

    key = mx.random.key(seed)
    sample = mx.random.normal(shape=(1, VAE_DIM), key=key).astype(mx.float32)

    batched_cond = mx.concatenate([
        condition.astype(dtype), neg_condition.astype(dtype)
    ], axis=0)
    prepared = _prepare_diffusion_conditioning(
        diff_head, batched_cond, t_schedule[:-1], dtype,
    )

    x0_list = []

    for i in range(num_steps):
        s = int(t_schedule[i])
        t = int(t_schedule[i + 1])

        x0 = _dpm_denoise_step(
            diff_head, sample, batched_cond, s, cfg_scale, dtype,
            prepared=prepared, step=i,
        )
        # The inference endpoint is zero noise, distinct from training index 0.
        # Its first-order update returns the latest clean prediction exactly.
        if i == num_steps - 1:
            return x0
        x0_list.append(x0)

        sigma_s = float(_SIGMA_NP[s])
        lam_s = float(_LAMBDA_NP[s])
        lam_t = float(_LAMBDA_NP[max(t, 0)])
        h = lam_t - lam_s

        # All updates after the initial one remain second-order, including
        # the penultimate update for short schedules.
        use_first_order = len(x0_list) < 2

        if use_first_order:
            D = x0_list[-1]
        else:
            s_prev = int(t_schedule[i - 1])
            lam_s_prev = float(_LAMBDA_NP[s_prev])
            h_prev = lam_s - lam_s_prev
            r = h_prev / h
            D = x0_list[-1] + 0.5 / r * (x0_list[-1] - x0_list[-2])

        sigma_t = float(_SIGMA_NP[t])
        alpha_t = float(_ALPHA_NP[t])
        sample = (sigma_t / sigma_s) * sample - alpha_t * float(np.expm1(-h)) * D

    return sample


def dpm_solver_sde_2m(
    diff_head,
    condition: mx.array,
    neg_condition: mx.array,
    cfg_scale: float,
    num_steps: int = 20,
    seed: int = 0,
    dtype=mx.float16,
) -> mx.array:
    """SDE DPM-Solver++ 2M (stochastic, midpoint) with batched CFG.

    Matches the sde-dpmsolver++ algorithm from HuggingFace Diffusers:
    - final_sigmas_type="zero": last step targets sigma=0 (perfect denoising)
    - lower_order_final: last step uses first-order update
    - Noise injected at every step (coefficient=0 at last step due to sigma_t=0)

    Returns sample of shape (1, VAE_DIM) in float32.
    """
    # Timestep schedule: N timesteps from 999→~50 (matching diffusers linspace)
    _validate_cfg_scale(cfg_scale)
    num_steps = _validate_diffusion_steps(num_steps)
    timesteps = np.round(
        np.linspace(0, DDPM_STEPS - 1, num_steps + 1)
    ).astype(np.int64)[::-1][:-1]  # (num_steps,) from high to low

    # Build sigmas array with final sigma=0 (diffusers final_sigmas_type="zero")
    all_sigmas = ((1.0 - _AC64) / _AC64) ** 0.5  # ratio-form sigmas
    sigmas = np.interp(timesteps, np.arange(len(all_sigmas)), all_sigmas)
    sigmas = np.append(sigmas, 0.0)  # (num_steps + 1,) — last entry is 0

    def _sig_to_alpha_sigma(sig):
        a = 1.0 / np.sqrt(sig ** 2 + 1.0)
        s = sig * a
        return float(a), float(s)

    key = mx.random.key(seed)
    # Pre-generate all noise vectors upfront (avoids per-step random split overhead)
    noise_keys = mx.random.split(key, num_steps + 1)
    sample = mx.random.normal(shape=(1, VAE_DIM), key=noise_keys[0]).astype(mx.float32)
    noise_vecs = mx.random.normal(shape=(num_steps, VAE_DIM), key=noise_keys[1])
    mx.eval(sample, noise_vecs)

    # Precompute schedule values (all in numpy, no per-step recomputation)
    alphas = np.array([_sig_to_alpha_sigma(s)[0] for s in sigmas])
    sigma_vals = np.array([_sig_to_alpha_sigma(s)[1] for s in sigmas])
    lambdas = np.log(np.maximum(alphas, 1e-10)) - np.log(np.maximum(sigma_vals, 1e-10))
    lambdas[-1] = np.inf  # sigma=0 at last position

    batched_cond = mx.concatenate([
        condition.astype(dtype), neg_condition.astype(dtype)
    ], axis=0)
    prepared = _prepare_diffusion_conditioning(
        diff_head, batched_cond, timesteps, dtype,
    )

    x0_list = []

    for i in range(num_steps):
        s_ts = int(timesteps[i])

        x0 = _dpm_denoise_step(
            diff_head, sample, batched_cond, s_ts, cfg_scale, dtype,
            prepared=prepared, step=i,
        )
        x0_list.append(x0)

        h = float(lambdas[i + 1] - lambdas[i])
        is_last = (i == num_steps - 1)
        use_first_order = len(x0_list) < 2 or is_last

        if use_first_order:
            D = x0_list[-1]
        else:
            h_prev = float(lambdas[i] - lambdas[i - 1])
            r = h_prev / h
            D = x0_list[-1] + 0.5 / r * (x0_list[-1] - x0_list[-2])

        # SDE update
        if np.isinf(h):
            sample = D
        else:
            exp_neg_h = float(np.exp(-h))
            exp_neg_2h = float(np.exp(-2.0 * h))
            s_t = float(sigma_vals[i + 1])
            s_s = float(sigma_vals[i])
            a_t = float(alphas[i + 1])
            sample = (s_t / s_s * exp_neg_h) * sample + a_t * (1.0 - exp_neg_2h) * D
            noise = noise_vecs[i:i + 1].astype(mx.float32)
            sample = sample + s_t * float(np.sqrt(max(0.0, 1.0 - exp_neg_2h))) * noise

    return sample


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

@dataclass
class MLXSemanticCallback:
    """Opt into device-native feedback while retaining the NumPy callback API.

    encode_mlx receives a flat float32 audio chunk and returns an MLX embedding.
    Calling the adapter directly still accepts and returns NumPy arrays.
    """

    encode_mlx: Callable[[mx.array], mx.array]

    def __call__(self, audio_chunk: np.ndarray) -> np.ndarray:
        return np.array(self.encode_mlx(mx.array(audio_chunk, dtype=mx.float32)))


def generate(
    model: VibeVoiceModel,
    input_ids: list[int],
    opts: GenerationOptions | None = None,
    semantic_encoder_fn: Optional[Callable] = None,
    semantic_reset_fn: Optional[Callable] = None,
    voice_embeds: Optional[dict[int, mx.array]] = None,
    estimated_total: Optional[int] = None,
) -> tuple[np.ndarray, GenerationMetrics]:
    """Full autoregressive TTS generation.

    Args:
        model: Loaded VibeVoiceModel
        input_ids: Tokenized input sequence (includes special tokens)
        opts: Generation options
        semantic_encoder_fn: Optional callback for semantic encoder.
            Signature: fn(audio_chunk: np.ndarray) -> np.ndarray of shape (1, 1, hidden_size)
            MLXSemanticCallback opts into MLX arrays without host round trips.
        voice_embeds: Optional dict mapping position -> embedding for voice cloning.
            Each value is an mx.array of shape (1, hidden_size).
        estimated_total: Estimated total speech tokens for progress bar.
            If None, falls back to n_prefill as a rough guess.

    Returns:
        (audio_array, metrics)
    """
    if opts is None:
        opts = GenerationOptions()

    solver_fn = {"dpm": dpm_solver_2m, "sde": dpm_solver_sde_2m}.get(opts.solver)
    if solver_fn is None:
        raise ValueError(f"Unsupported solver {opts.solver!r}; choose 'dpm' or 'sde'.")
    _validate_cfg_scale(opts.cfg_scale)
    diffusion_steps = _validate_diffusion_steps(opts.diffusion_steps)

    config = model.config
    dtype = mx.float16
    metrics = GenerationMetrics(
        name=f"MLX ({opts.solver}-{opts.diffusion_steps}s)"
    )

    t0_total = time.perf_counter()

    # Build fast-path LM and diffusion head (raw matmul, no nn.Module dispatch)
    # Cache on model to avoid re-extracting weight references each call.
    if not hasattr(model, "_fast_lm"):
        model._fast_lm = FastLM(model, config)
        model._fast_diff = FastDiffusionHead(model, config)
    fast_lm = model._fast_lm
    fast_diff = model._fast_diff
    embed_table = fast_lm.embed_w
    NL = config.num_hidden_layers

    # Each invocation owns independent recurrent state, including semantic history.
    if semantic_encoder_fn is not None and semantic_reset_fn is not None:
        semantic_reset_fn()

    use_evolving_cfg = opts.cfg_scale > 1.0
    prepared_neg_hidden = None
    provisional_neg_hidden = None

    # The negative branch consumes the previous input only when diffusion runs.
    # Only allocate when CFG is active to save memory.
    if use_evolving_cfg:
        pending_neg_embed = embed_table[config.speech_start_id].reshape(1, 1, config.hidden_size)
        neg_cache = KVCache(NL)
        neg_position = 0
    else:
        # Static zero condition for CFG=1.0 (no guidance)
        neg_condition = mx.zeros((1, config.hidden_size), dtype=dtype)
        neg_cache = None
        neg_position = 0

    # Prefill (use fast_lm.prefill for batched tokens)
    t0 = time.perf_counter()
    n_prefill = len(input_ids)

    if voice_embeds:
        embeds_list = []
        for pos, tok_id in enumerate(input_ids):
            if pos in voice_embeds:
                embeds_list.append(voice_embeds[pos].reshape(1, config.hidden_size))
            else:
                embeds_list.append(embed_table[tok_id].reshape(1, config.hidden_size))
        prefill_embeds = mx.stack(embeds_list, axis=0).reshape(1, n_prefill, config.hidden_size)
    else:
        ids_mx = mx.array(input_ids)
        prefill_embeds = embed_table[ids_mx].reshape(1, n_prefill, config.hidden_size)

    positions = mx.arange(n_prefill, dtype=mx.float32)
    cos_prefill, sin_prefill = compute_rope(positions, config.head_dim, config.rope_theta)
    causal_mask = mx.triu(mx.full((n_prefill, n_prefill), float("-inf"), dtype=dtype), k=1)

    cache = KVCache(NL)
    hidden = fast_lm.prefill(prefill_embeds, cos_prefill, sin_prefill, causal_mask, cache)
    mx.eval(hidden, *cache.keys, *cache.values)
    hidden = hidden[:, -1:, :]

    metrics.record("prefill", (time.perf_counter() - t0) * 1000)
    metrics.num_text_tokens = n_prefill

    # Constrain to speech-structural tokens only to prevent text token
    # hallucination and premature EOS. Matches original PyTorch VibeVoice
    # VibeVoiceTokenConstraintProcessor behavior (valid_tokens = speech_start,
    # speech_end, speech_diffusion, eos). The speaker labels are in the input,
    # not the output — the model only generates speech tokens.
    # First token
    next_token = fast_lm.select_token(hidden, speech_only=True)

    # Semantic feedback must hear the same continuous audio we return. Keep
    # causal decoder history for this invocation, including across speech_end
    # boundaries, just as a batch decode of all generated latents would.
    streaming_decoder = (
        StreamingVAEDecoder(model.vae_decoder)
        if semantic_encoder_fn is not None else None
    )

    # Autoregressive generation
    audio_chunks: list[np.ndarray | mx.array] = []
    native_semantic = isinstance(semantic_encoder_fn, MLXSemanticCallback)
    all_latents = []
    silent_run = 0
    rng = np.random.RandomState(opts.seed)
    position = n_prefill

    # Setup progress bar — use estimated total if provided, else rough guess
    pbar = tqdm(
        total=estimated_total if estimated_total is not None else n_prefill,
        desc="Generating",
        unit="tok",
    )

    for step in range(opts.max_speech_tokens * 3):
        if next_token == config.eos_id:
            metrics.stop_reason = "eos"
            break
        if config.single_segment and next_token == config.speech_end_id:
            metrics.stop_reason = "speech_end"
            break
        if metrics.num_speech_tokens >= opts.max_speech_tokens:
            metrics.stop_reason = "max_speech_tokens"
            break

        negative_lm_ms = 0.0
        if next_token == config.speech_diffusion_id:
            metrics.num_speech_tokens += 1

            # Update progress bar
            pbar.update(1)

            if use_evolving_cfg:
                if prepared_neg_hidden is None:
                    t0 = time.perf_counter()
                    neg_pos = mx.array([float(neg_position)], dtype=mx.float32)
                    neg_cos, neg_sin = compute_rope(neg_pos, config.head_dim, config.rope_theta)
                    neg_hidden = fast_lm.forward(pending_neg_embed, neg_cos, neg_sin, neg_cache)
                    mx.eval(neg_hidden, *neg_cache.keys, *neg_cache.values)
                    neg_position += 1
                    negative_lm_ms = (time.perf_counter() - t0) * 1000
                else:
                    neg_hidden = prepared_neg_hidden
                    prepared_neg_hidden = None
                neg_condition = neg_hidden[:, 0:1, :].reshape(1, config.hidden_size)

            # Diffusion (fast path — no nn.Module dispatch)
            t0 = time.perf_counter()
            condition = hidden[:, 0:1, :].reshape(1, config.hidden_size)
            sample = solver_fn(
                fast_diff, condition, neg_condition, opts.cfg_scale,
                num_steps=diffusion_steps,
                seed=rng.randint(0, 2**31),
                dtype=dtype,
            )
            metrics.record("diffusion", (time.perf_counter() - t0) * 1000)

            # Keep latents for silence detection and the no-semantic batch path.
            latent = (sample / config.speech_scaling_factor - config.speech_bias_factor)
            all_latents.append(latent)

            # Decode once with causal history, then reuse for feedback and output.
            if streaming_decoder is not None:
                t0 = time.perf_counter()
                latent_frame = latent[:, :, None].astype(dtype)
                audio = streaming_decoder(latent_frame)
                mx.eval(audio, *streaming_decoder.caches.values())
                if native_semantic:
                    audio_chunks.append(audio.reshape(-1).astype(mx.float32))
                else:
                    audio_chunks.append(np.array(audio).squeeze().astype(np.float32))
                metrics.record("vae", (time.perf_counter() - t0) * 1000)

            # Connectors: acoustic + optional semantic feedback
            t0 = time.perf_counter()
            acoustic_embed = model.acoustic_connector(sample[:, None, :].astype(dtype))

            if semantic_encoder_fn is not None and audio_chunks:
                if native_semantic:
                    sem_embed = semantic_encoder_fn.encode_mlx(
                        audio_chunks[-1][:3200]
                    ).astype(dtype)
                else:
                    chunk = audio_chunks[-1][:3200].astype(np.float32)
                    sem_embed_np = semantic_encoder_fn(chunk)
                    sem_embed = mx.array(sem_embed_np).astype(dtype)
                if sem_embed.ndim == 3:
                    next_embed = acoustic_embed + sem_embed
                else:
                    next_embed = acoustic_embed + sem_embed.reshape(1, 1, config.hidden_size)
            else:
                next_embed = acoustic_embed
            metrics.record("connector", (time.perf_counter() - t0) * 1000)
        else:
            if next_token == config.speech_end_id and semantic_reset_fn is not None:
                semantic_reset_fn()
            next_embed = embed_table[next_token].reshape(1, 1, config.hidden_size)

        # Positive LM history includes every generated control/audio embedding.
        t0 = time.perf_counter()
        pos = mx.array([float(position)], dtype=mx.float32)
        cos, sin = compute_rope(pos, config.head_dim, config.rope_theta)

        if use_evolving_cfg and next_token == config.speech_start_id:
            # Discard previous segment history when a new segment starts.
            neg_cache.reset()
            neg_position = 0

        if use_evolving_cfg:
            pending_neg_embed = next_embed
        provisional_neg_hidden = None
        if (
            use_evolving_cfg
            and next_token == config.speech_diffusion_id
            and metrics.num_speech_tokens < opts.max_speech_tokens
            and step + 1 < opts.max_speech_tokens * 3
        ):
            neg_pos = mx.array([float(neg_position)], dtype=mx.float32)
            neg_cos, neg_sin = compute_rope(neg_pos, config.head_dim, config.rope_theta)
            hidden, provisional_neg_hidden = fast_lm.forward_dual(
                next_embed, cos, sin, cache,
                next_embed, neg_cos, neg_sin, neg_cache,
            )
        else:
            hidden = fast_lm.forward(next_embed, cos, sin, cache)

        boost = 0.0
        # Silence-aware stop: boost speech_end when generating silence
        if opts.silence_detection and config.single_segment and all_latents:
            lat_rms = float(mx.sqrt(mx.mean(all_latents[-1] ** 2)))
            if lat_rms < 1.0:
                silent_run += 1
            else:
                silent_run = 0
            if silent_run >= 3:
                boost = min((silent_run - 2) * 5.0, 20.0)
        next_token = fast_lm.select_token(hidden, speech_only=True, stop_boost=boost)
        if provisional_neg_hidden is not None:
            if next_token == config.speech_diffusion_id:
                prepared_neg_hidden = provisional_neg_hidden
                neg_position += 1
            else:
                neg_cache.truncate(neg_position)
        # Fused negative work is charged with this positive LM step. Remaining
        # lazy operations evaluate with their consumers; do not add a sync here.
        metrics.record("lm_step", (time.perf_counter() - t0) * 1000 + negative_lm_ms)
        position += 1

        # Free MLX Metal buffer pool every 10 steps to prevent unbounded growth.
        if step % 10 == 0:
            mx.clear_cache()
    else:
        metrics.stop_reason = "max_generation_tokens"

    # Close progress bar
    pbar.close()

    # Release autoregressive state before final VAE materialization. Any latent
    # graphs that depend on these buffers retain the values they need.
    cache.reset()
    if neg_cache is not None:
        neg_cache.reset()
    hidden = None
    prepared_neg_hidden = None
    provisional_neg_hidden = None

    # Reuse continuous feedback audio; batch-decode only without semantic feedback.
    if all_latents:
        t0 = time.perf_counter()
        if streaming_decoder is not None:
            if native_semantic:
                audio_out = np.array(mx.concatenate(audio_chunks))
            else:
                audio_out = np.concatenate(audio_chunks)
        else:
            full_latent = mx.concatenate(all_latents, axis=0).T[None, :, :].astype(dtype)
            full_audio = model.vae_decoder(full_latent)
            mx.eval(full_audio)
            audio_out = np.array(full_audio).squeeze().astype(np.float32)
        do_trim = opts.trim_trailing_silence if opts.trim_trailing_silence is not None else opts.silence_detection
        if do_trim:
            audio_out = _trim_trailing_silence(
                audio_out,
                threshold=opts.silence_threshold,
                long_silence_ms=opts.silence_min_duration_ms,
                pad_ms=opts.silence_pad_ms,
            )
        metrics.record("vae_final", (time.perf_counter() - t0) * 1000)
    else:
        audio_out = np.zeros(0, dtype=np.float32)

    metrics.total_time = (time.perf_counter() - t0_total) * 1000
    metrics.audio_samples = len(audio_out)

    # Free MLX metal buffer pool so repeated calls don't grow unbounded.
    mx.clear_cache()

    return audio_out, metrics


def _trim_trailing_silence(audio: np.ndarray, sr: int = 24000,
                           threshold: float = 0.05,
                           long_silence_ms: int = 1500,
                           pad_ms: int = 300) -> np.ndarray:
    """Trim audio after speech ends.

    Two-pass approach:
    1. Forward scan: if a long silence gap (>= long_silence_ms) follows
       speech, cut there — this catches model repetition after a pause.
    2. Backward scan: trim trailing silence/noise from the end.
    """
    window = int(sr * 0.05)  # 50ms windows
    pad = int(sr * pad_ms / 1000)
    long_silent_windows = max(1, int(long_silence_ms / 50))

    n_windows = len(audio) // window
    if n_windows == 0:
        return audio

    rms = np.array([
        np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])

    # Forward: find first long silence gap after speech starts
    found_speech = False
    silent_count = 0
    for i in range(n_windows):
        if rms[i] >= threshold:
            found_speech = True
            silent_count = 0
        elif found_speech:
            silent_count += 1
            if silent_count >= long_silent_windows:
                cut = (i - silent_count + 1) * window + pad
                audio = audio[:min(cut, len(audio))]
                break

    # Backward: trim trailing silence/noise
    n_windows = len(audio) // window
    if n_windows > 2:
        rms = np.array([
            np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))
            for i in range(n_windows)
        ])
        for i in range(n_windows - 1, 2, -1):
            if rms[i] >= threshold and rms[i - 1] >= threshold and rms[i - 2] >= threshold:
                end = min((i + 1) * window + pad, len(audio))
                return audio[:end]

    return audio
