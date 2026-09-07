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
    preserve_control_history: bool = False,
) -> np.ndarray:
    """Recompute whole contexts without KV caches, following refresh_negative=True.

    The positive sequence includes every control token and audio embedding.
    The negative sequence contains one start plus only this segment's audio.
    preserve_control_history checks the established local behavior for irregular
    control sequences; it is not an upstream fidelity oracle for those sequences.
    """
    config = model.config
    embedding = model.model.embed_tokens
    positive = embedding(mx.array([[4, 0]]))
    start = embedding(mx.array([[0]]))
    negative = start
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
            sample = solver(
                head,
                condition(positive),
                condition(negative)
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
            negative = mx.concatenate([negative, next_embed], axis=1)
        else:
            next_embed = embedding(mx.array([[token]]))
            if token == config.speech_start_id:
                negative = start
            elif preserve_control_history:
                negative = mx.concatenate([negative, next_embed], axis=1)
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
@pytest.mark.parametrize("generated_start", [False, True])
@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("solver", ["dpm", "sde"])
def test_generation_refreshes_negative_context_for_each_segment(
    model: VibeVoiceModel,
    monkeypatch: pytest.MonkeyPatch,
    cfg_scale: float,
    generated_start: bool,
    semantic: bool,
    solver: str,
) -> None:
    tokens = [2, 2, 1, 0, 2, 2, 3]
    if generated_start:
        tokens.insert(0, 0)
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

        assert metrics.num_speech_tokens == 4
        assert audio.shape == (12800,)
        assert resets == [True]
        np.testing.assert_allclose(audio, expected, atol=2e-5, rtol=2e-3)
        if semantic:
            np.testing.assert_array_equal(np.concatenate(chunks), audio)


def test_generation_preserves_control_history_without_a_new_speech_start(
    model: VibeVoiceModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Preserve existing behavior for irregular controls; only speech_start
    # changes the negative context policy in this fix.
    tokens = [2, 1, 2, 3]
    opts = GenerationOptions(cfg_scale=2.0, diffusion_steps=2, max_speech_tokens=3)
    expected = reference_audio(
        model,
        tokens,
        opts,
        semantic=False,
        preserve_control_history=True,
    )
    selected = iter(tokens)
    monkeypatch.setattr(FastLM, "select_token", lambda *args, **kwargs: next(selected))

    audio, metrics = generate(model, [4, 0], opts)

    assert metrics.num_speech_tokens == 2
    np.testing.assert_allclose(audio, expected, atol=2e-5, rtol=2e-3)
