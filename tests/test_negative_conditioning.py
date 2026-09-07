"""CFG uses the current speech segment as its unconditional context."""

from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import pytest
from test_semantic_audio_context import tiny_decoder

from vibevoice_mlx.fast_forward import FastDiffusionHead, FastLM
from vibevoice_mlx.generate import (
    GenerationOptions,
    dpm_solver_2m,
    dpm_solver_sde_2m,
    generate,
)
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel, compute_rope


@pytest.fixture
def model() -> Iterator[VibeVoiceModel]:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(17)
        model = VibeVoiceModel(
            VibeVoiceConfig(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                intermediate_size=16,
                vocab_size=8,
                diffusion_layers=1,
                speech_start_id=0,
                speech_end_id=1,
                speech_diffusion_id=2,
                eos_id=3,
                speech_scaling_factor=1.0,
                speech_bias_factor=0.0,
            )
        )
        model.vae_decoder = tiny_decoder()
        yield model
    finally:
        mx.set_default_device(previous)


def reference_audio(
    model: VibeVoiceModel,
    tokens: list[int],
    opts: GenerationOptions,
    semantic: bool,
) -> np.ndarray:
    """Recompute whole contexts without KV caches, following refresh_negative=True.

    The positive sequence includes every control token and audio embedding.
    Each diffusion request appends the preceding iteration's embedding to the
    negative sequence. Speech starts clear that sequence. No negative LM call
    occurs for other selected tokens.
    """
    config = model.config
    embedding = model.model.embed_tokens
    positive = embedding(mx.array([[4, 0]]))
    start = embedding(mx.array([[0]]))
    negative_inputs = []
    pending = start
    head = FastDiffusionHead(model, config)
    solver = dpm_solver_2m if opts.solver == "dpm" else dpm_solver_sde_2m
    rng = np.random.RandomState(opts.seed)
    latents = []

    def condition(sequence: mx.array) -> mx.array:
        length = sequence.shape[1]
        cos, sin = compute_rope(mx.arange(length), config.head_dim, config.rope_theta)
        mask = mx.triu(mx.full((length, length), float("-inf")), k=1)
        return model.model(sequence, cos, sin, mask)[:, -1, :]

    for token in tokens:
        if token == config.eos_id:
            break
        if token == config.speech_diffusion_id:
            negative_inputs.append(pending)
            sample = solver(
                head,
                condition(positive),
                condition(mx.concatenate(negative_inputs, axis=1))
                if opts.cfg_scale > 1
                else mx.zeros((1, config.hidden_size)),
                opts.cfg_scale,
                num_steps=opts.diffusion_steps,
                seed=rng.randint(0, 2**31),
            )
            latents.append(sample)
            next_embed = model.acoustic_connector(sample[:, None].astype(mx.float16))
            if semantic:
                next_embed += mx.array(semantic_embedding()).astype(mx.float16)
        else:
            next_embed = embedding(mx.array([[token]]))
            if token == config.speech_start_id:
                negative_inputs = []
        pending = next_embed
        positive = mx.concatenate([positive, next_embed], axis=1)
    latent = mx.concatenate(latents, axis=0).T[None].astype(mx.float16)
    return np.array(model.vae_decoder(latent)).reshape(-1)


def semantic_embedding() -> np.ndarray:
    # A fixed external callback result isolates context policy from tiny
    # floating-point differences between streaming and batch decoder outputs.
    return np.array(
        [0.2, -0.1, 0.3, 0.1, -0.2, 0.4, -0.3, 0.2], dtype=np.float32
    ).reshape(1, 1, 8)


@pytest.mark.parametrize("cfg_scale", [1.0, 2.0])
@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("solver", ["dpm", "sde"])
@pytest.mark.parametrize(
    ("tokens", "speech_tokens", "speech_ends"),
    [
        ([2, 2, 1, 0, 2, 2, 3], 4, 1),
        ([0, 2, 2, 1, 0, 2, 2, 3], 4, 1),
        ([2, 1, 2, 3], 2, 1),
        ([2, 2, 1, 2, 3], 3, 1),
        ([1, 1, 2, 3], 1, 2),
        ([0, 1, 2, 3], 1, 1),
        ([0, 0, 2, 3], 1, 0),
        ([2, 0, 0, 2, 3], 2, 0),
    ],
    ids=[
        "segments",
        "generated-start",
        "audio-end-audio",
        "two-audio-end-audio",
        "leading-ends",
        "start-end-audio",
        "repeated-starts",
        "repeated-mid-starts",
    ],
)
def test_generation_matches_deferred_negative_context(
    model: VibeVoiceModel,
    monkeypatch: pytest.MonkeyPatch,
    cfg_scale: float,
    semantic: bool,
    solver: str,
    tokens: list[int],
    speech_tokens: int,
    speech_ends: int,
) -> None:
    opts = GenerationOptions(
        solver=solver, cfg_scale=cfg_scale, diffusion_steps=2, max_speech_tokens=5
    )
    expected = reference_audio(model, tokens, opts, semantic)
    for _ in range(2):
        selected = iter(tokens)
        monkeypatch.setattr(
            FastLM,
            "select_token",
            lambda *args, selected=selected, **kwargs: next(selected),
        )
        chunks = []
        resets = []

        def feedback(
            chunk: np.ndarray, chunks: list[np.ndarray] = chunks
        ) -> np.ndarray:
            chunks.append(chunk.copy())
            return semantic_embedding()

        audio, metrics = generate(
            model,
            [4, 0],
            opts,
            semantic_encoder_fn=feedback if semantic else None,
            semantic_reset_fn=lambda resets=resets: resets.append(True),
        )

        assert metrics.num_speech_tokens == speech_tokens
        assert len(metrics.timings["diffusion"]) == speech_tokens
        assert len(metrics.timings["lm_step"]) == len(tokens) - 1
        assert audio.shape == (3200 * speech_tokens,)
        assert resets == [True] * speech_ends
        np.testing.assert_allclose(audio, expected, atol=2e-5, rtol=2e-3)
        if semantic:
            np.testing.assert_array_equal(np.concatenate(chunks), audio)
