"""Fast-path forward functions for VibeVoice generation.

Bypasses nn.Module dispatch by extracting weights into flat dicts
and using raw mx.quantized_matmul / matmul calls. ~30% faster than
nn.Module for the autoregressive loop.

Used by generate.py for the hot path (LM step + diffusion).
Model loading still uses nn.Module for clean weight mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .model import KVCache, VibeVoiceModel, VibeVoiceConfig, apply_rope


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def _extract_linear(mod) -> dict:
    """Extract weight data from nn.Linear or nn.QuantizedLinear."""
    d = {}
    if hasattr(mod, "scales"):  # quantized
        d["w"] = mod.weight
        d["s"] = mod.scales
        d["b"] = mod.biases
        d["gs"] = mod.group_size
        d["bits"] = mod.bits
        d["q"] = True
    else:
        d["w"] = mod.weight
        d["q"] = False
    if hasattr(mod, "bias") and mod.bias is not None:
        d["bias"] = mod.bias
    return d


def _mm(x, d):
    """x @ w.T — dispatches to quantized or plain matmul."""
    if d["q"]:
        return mx.quantized_matmul(
            x, d["w"], d["s"], d["b"],
            transpose=True, group_size=d["gs"], bits=d["bits"],
        )
    return x @ d["w"].T


class FastLM:
    """Flat-dict LM for fast autoregressive decode."""

    def __init__(self, model: VibeVoiceModel, config: VibeVoiceConfig):
        self.H = config.hidden_size
        self.NH = config.num_attention_heads
        self.NKV = config.num_key_value_heads
        self.HD = config.head_dim
        self.NL = config.num_hidden_layers
        self.scale = config.head_dim ** -0.5
        self.eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta

        # Extract layer weights
        self.layers = []
        for layer in model.model.layers:
            sa, ml = layer.self_attn, layer.mlp
            d = {
                "iln": layer.input_layernorm.weight,
                "pln": layer.post_attention_layernorm.weight,
                "q": _extract_linear(sa.q_proj),
                "k": _extract_linear(sa.k_proj),
                "v": _extract_linear(sa.v_proj),
                "o": _extract_linear(sa.o_proj),
                "g": _extract_linear(ml.gate_proj),
                "u": _extract_linear(ml.up_proj),
                "d": _extract_linear(ml.down_proj),
            }
            self.layers.append(d)

        self.norm_w = model.model.norm.weight
        self.embed_w = model.model.embed_tokens.weight

        # LM head
        if model.lm_head is not None:
            self.lm_head_w = model.lm_head.weight
        else:
            self.lm_head_w = None  # tied

        # Ascending IDs preserve full-vocabulary argmax tie-breaking.
        self.speech_token_ids = tuple(
            sorted(
                {
                    config.speech_start_id,
                    config.speech_end_id,
                    config.speech_diffusion_id,
                    config.eos_id,
                }
            )
        )
        self._stop_indices = (
            self.speech_token_ids.index(config.speech_end_id),
            self.speech_token_ids.index(config.eos_id),
        )
        self._speech_head_w: mx.array | None = None

    def forward(
        self, h: mx.array, cos: mx.array, sin: mx.array, cache: KVCache
    ) -> mx.array:
        """Single-step LM forward with KV cache.

        h: (1, Q, H)
        cos, sin: (Q, HD) from compute_rope
        cache: layer KV buffers, updated with this token.
        Returns: hidden (1, Q, H).
        """
        return self._forward_inner(h, cos, sin, cache)

    def forward_dual(
        self, h_main: mx.array, cos_main: mx.array, sin_main: mx.array, cache: KVCache,
        h_neg: mx.array, cos_neg: mx.array, sin_neg: mx.array, neg_cache: KVCache,
    ) -> tuple[mx.array, mx.array]:
        """Batched main+neg LM forward — reads weights once for both.

        Returns: (hidden_main, hidden_neg), updates both KV caches.
        """
        NH, NKV, HD, H = self.NH, self.NKV, self.HD, self.H
        scale = self.scale
        eps = self.eps

        hm, hn_input = h_main, h_neg

        for li, d in enumerate(self.layers):
            # Batch projections: concat (1,1,H) inputs → (2,1,H), single matmul
            h_cat = mx.concatenate([hm, hn_input], axis=0)  # (2, 1, H)
            res_cat = h_cat

            hn_cat = mx.fast.rms_norm(h_cat, d["iln"], eps)

            q_cat = _mm(hn_cat, d["q"])
            if "bias" in d["q"]:
                q_cat = q_cat + d["q"]["bias"]
            k_cat = _mm(hn_cat, d["k"])
            if "bias" in d["k"]:
                k_cat = k_cat + d["k"]["bias"]
            v_cat = _mm(hn_cat, d["v"])
            if "bias" in d["v"]:
                v_cat = v_cat + d["v"]["bias"]

            # Split for attention (different KV caches)
            q_m, q_n = q_cat[0:1], q_cat[1:2]
            k_m, k_n = k_cat[0:1], k_cat[1:2]
            v_m, v_n = v_cat[0:1], v_cat[1:2]

            # Main attention
            q_m = q_m.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k_m = k_m.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v_m = v_m.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            q_m = apply_rope(q_m, cos_main, sin_main)
            k_m = apply_rope(k_m, cos_main, sin_main)
            keys, values = cache.update(li, k_m, v_m)
            out_m = mx.fast.scaled_dot_product_attention(
                q_m, keys, values, scale=scale,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            # Neg attention
            q_n = q_n.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k_n = k_n.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v_n = v_n.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            q_n = apply_rope(q_n, cos_neg, sin_neg)
            k_n = apply_rope(k_n, cos_neg, sin_neg)
            neg_keys, neg_values = neg_cache.update(li, k_n, v_n)
            out_n = mx.fast.scaled_dot_product_attention(
                q_n, neg_keys, neg_values, scale=scale,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            # Batch o_proj + MLP
            out_cat = mx.concatenate([out_m, out_n], axis=0)
            h_cat = res_cat + _mm(out_cat, d["o"])

            res_cat = h_cat
            hn_cat = mx.fast.rms_norm(h_cat, d["pln"], eps)
            h_cat = res_cat + _mm(nn.silu(_mm(hn_cat, d["g"])) * _mm(hn_cat, d["u"]), d["d"])

            hm, hn_input = h_cat[0:1], h_cat[1:2]

        hm = mx.fast.rms_norm(hm, self.norm_w, eps)
        hn_input = mx.fast.rms_norm(hn_input, self.norm_w, eps)
        return hm, hn_input

    def _forward_inner(
        self, h: mx.array, cos: mx.array, sin: mx.array, cache: KVCache
    ) -> mx.array:
        NH, NKV, HD, H = self.NH, self.NKV, self.HD, self.H
        scale = self.scale
        eps = self.eps

        for li, d in enumerate(self.layers):
            res = h
            hn = mx.fast.rms_norm(h, d["iln"], eps)

            q = _mm(hn, d["q"])
            if "bias" in d["q"]:
                q = q + d["q"]["bias"]
            k = _mm(hn, d["k"])
            if "bias" in d["k"]:
                k = k + d["k"]["bias"]
            v = _mm(hn, d["v"])
            if "bias" in d["v"]:
                v = v + d["v"]["bias"]

            q = q.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k = k.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v = v.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            keys, values = cache.update(li, k, v)

            out = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=scale,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            h = res + _mm(out, d["o"])

            res = h
            hn = mx.fast.rms_norm(h, d["pln"], eps)
            h = res + _mm(nn.silu(_mm(hn, d["g"])) * _mm(hn, d["u"]), d["d"])

        return mx.fast.rms_norm(h, self.norm_w, eps)

    def logits(self, h: mx.array, *, speech_only: bool = False) -> mx.array:
        """Compute float32-accumulated logits, cast back to the input dtype.

        With speech_only, the last axis follows speech_token_ids instead of
        vocabulary IDs. Cache only those rows to avoid scanning the full head.
        """
        weight = self.lm_head_w if self.lm_head_w is not None else self.embed_w
        if speech_only:
            if self._speech_head_w is None:
                self._speech_head_w = weight[mx.array(self.speech_token_ids)].astype(
                    mx.float32
                )
            weight = self._speech_head_w
        return (h.astype(mx.float32) @ weight.astype(mx.float32).T).astype(h.dtype)

    def select_token(
        self, h: mx.array, *, speech_only: bool = False, stop_boost: float = 0.0
    ) -> int:
        """Greedy selection for one hidden state, optionally boosting speech stops."""
        logits = self.logits(h, speech_only=speech_only)
        if speech_only:
            # The former full-vocabulary mask promoted scores to float32 before
            # silence boosts. Preserve that rounding behavior after projection.
            logits = logits.astype(mx.float32)
            if stop_boost:
                for index in self._stop_indices:
                    logits[0, 0, index] += stop_boost
        index = int(mx.argmax(logits[0, 0]).item())
        return self.speech_token_ids[index] if speech_only else index

    def prefill(
        self, embeds: mx.array, cos: mx.array, sin: mx.array,
        mask: mx.array, cache: KVCache,
    ) -> mx.array:
        """Batched prefill (Q>1) with causal mask. Same as forward but with mask."""
        NH, NKV, HD, H = self.NH, self.NKV, self.HD, self.H
        scale = self.scale
        eps = self.eps
        h = embeds
        cache.reset()

        for li, d in enumerate(self.layers):
            res = h
            hn = mx.fast.rms_norm(h, d["iln"], eps)

            q = _mm(hn, d["q"])
            if "bias" in d["q"]:
                q = q + d["q"]["bias"]
            k = _mm(hn, d["k"])
            if "bias" in d["k"]:
                k = k + d["k"]["bias"]
            v = _mm(hn, d["v"])
            if "bias" in d["v"]:
                v = v + d["v"]["bias"]

            q = q.reshape(1, -1, NH, HD).transpose(0, 2, 1, 3)
            k = k.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)
            v = v.reshape(1, -1, NKV, HD).transpose(0, 2, 1, 3)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            cache.update(li, k, v)

            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask,
            ).transpose(0, 2, 1, 3).reshape(1, -1, H)

            h = res + _mm(out, d["o"])

            res = h
            hn = mx.fast.rms_norm(h, d["pln"], eps)
            h = res + _mm(nn.silu(_mm(hn, d["g"])) * _mm(hn, d["u"]), d["d"])

        return mx.fast.rms_norm(h, self.norm_w, eps)


@dataclass(frozen=True)
class PreparedDiffusionConditioning:
    """Per-solve modulation arrays, indexed by timestep then CFG branch."""

    layers: tuple[mx.array, ...]
    final: mx.array


class FastDiffusionHead:
    """Flat-dict diffusion head for fast DPM solver.

    Handles both fp16 and quantized (INT8) diffusion head weights.
    """

    def __init__(self, model: VibeVoiceModel, config: VibeVoiceConfig):
        dh = model.diffusion_head
        self.noisy = _extract_linear(dh.noisy_images_proj)
        self.cond = _extract_linear(dh.cond_proj)
        self.t0 = _extract_linear(dh.t_embedder.mlp[0])
        self.t2 = _extract_linear(dh.t_embedder.mlp[2])
        self.freq_dim = dh.t_embedder.freq_dim

        self.layers = []
        for layer in dh.layers:
            self.layers.append((
                _extract_linear(layer.adaLN_modulation[1]),
                layer.norm.weight,
                _extract_linear(layer.ffn.gate_proj),
                _extract_linear(layer.ffn.up_proj),
                _extract_linear(layer.ffn.down_proj),
            ))

        self.final_adaln = _extract_linear(dh.final_layer.adaLN_modulation[1])
        self.final_linear = _extract_linear(dh.final_layer.linear)
        self.H = config.hidden_size

    def prepare_conditioning(
        self, condition: mx.array, timesteps: mx.array, *, dtype: mx.Dtype,
    ) -> PreparedDiffusionConditioning:
        """Prepare sample-independent work for one complete denoising schedule.

        Keep the condition projection, timestep MLP, addition, SiLU and
        modulation projections in their original order and precision. Only
        the independent timestep rows are batched. Nothing is cached across
        solves, so changing conditions, schedules or dtypes cannot reuse state.
        """
        weight = self.noisy["s"] if self.noisy["q"] else self.noisy["w"]
        # Match matmul's promotion without evaluating a sample projection.
        activation_dtype = (
            mx.zeros((), dtype=dtype) + mx.zeros((), dtype=weight.dtype)
        ).dtype
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = timesteps[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(
            activation_dtype
        )
        t = _mm(nn.silu(_mm(emb, self.t0)), self.t2)
        c = _mm(condition, self.cond)[None] + t[:, None, :]
        # One 2D matmul per projection shares its weights across all timesteps
        # and CFG branches instead of dispatching separate tiny batches.
        activated = nn.silu(c).reshape(-1, self.H)
        layers = tuple(
            _mm(activated, adaln).reshape(*c.shape[:-1], 3 * self.H)
            for adaln, *_ in self.layers
        )
        final = _mm(activated, self.final_adaln).reshape(*c.shape[:-1], 2 * self.H)
        return PreparedDiffusionConditioning(layers, final)

    def forward_prepared(
        self, noisy: mx.array, conditioning: PreparedDiffusionConditioning, step: int,
    ) -> mx.array:
        """Evaluate one sample using this timestep's prepared modulations."""
        H = self.H
        x = _mm(noisy, self.noisy)
        for (_, norm_w, gate, up, down), prepared in zip(
            self.layers, conditioning.layers, strict=True
        ):
            mods = prepared[step]
            shift, scale, g = mods[..., :H], mods[..., H:2*H], mods[..., 2*H:3*H]
            h = mx.fast.rms_norm(x, norm_w, 1e-5) * (1 + scale) + shift
            x = x + g * _mm(nn.silu(_mm(h, gate)) * _mm(h, up), down)
        mods = conditioning.final[step]
        shift, scale = mods[..., :H], mods[..., H:]
        h = mx.fast.rms_norm(x, mx.ones(H, dtype=x.dtype), 1e-5) * (1 + scale) + shift
        return _mm(h, self.final_linear)

    def __call__(self, noisy, timestep, condition):
        H = self.H
        x = _mm(noisy, self.noisy)

        # Timestep embedding
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = timestep[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(x.dtype)
        t = _mm(nn.silu(_mm(emb, self.t0)), self.t2)

        c = _mm(condition, self.cond) + t

        for adaln, norm_w, gate, up, down in self.layers:
            mods = _mm(nn.silu(c), adaln)
            shift, scale, g = mods[..., :H], mods[..., H:2*H], mods[..., 2*H:3*H]
            h = mx.fast.rms_norm(x, norm_w, 1e-5) * (1 + scale) + shift
            x = x + g * _mm(nn.silu(_mm(h, gate)) * _mm(h, up), down)

        mods = _mm(nn.silu(c), self.final_adaln)
        shift, scale = mods[..., :H], mods[..., H:]
        h = mx.fast.rms_norm(x, mx.ones(H, dtype=x.dtype), 1e-5) * (1 + scale) + shift
        return _mm(h, self.final_linear)
