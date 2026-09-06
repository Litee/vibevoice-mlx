"""Generation feedback must be the same continuous audio that is returned."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.model import VAEDecoder
from vibevoice_mlx.streaming_vae import DEPTHS, RATIOS

generation = importlib.import_module("vibevoice_mlx.generate")


def tiny_decoder() -> VAEDecoder:
    """Real causal decoder architecture, with small channels and seeded weights."""
    rng = np.random.RandomState(7)

    def weight(*shape: int) -> mx.array:
        return mx.array(rng.normal(0, 0.15, shape).astype(np.float16))

    def zeros(n: int) -> mx.array:
        return mx.zeros((n,), dtype=mx.float16)

    def ones(n: int) -> mx.array:
        return mx.ones((n,), dtype=mx.float16)

    vae = VAEDecoder()
    c = 2
    vae.init_conv_w, vae.init_conv_b = weight(c, 64, 3), zeros(c)
    for depth in DEPTHS:
        vae.stages.append(
            [
                {
                    "norm_w": ones(c),
                    "conv_w": weight(c, 1, 3),
                    "conv_b": zeros(c),
                    "gamma": ones(c),
                    "ffn_norm_w": ones(c),
                    "ffn_l1_w": weight(4, c),
                    "ffn_l1_b": zeros(4),
                    "ffn_l2_w": weight(c, 4),
                    "ffn_l2_b": zeros(c),
                    "ffn_gamma": ones(c),
                }
                for _ in range(depth)
            ]
        )
    vae.upsample_convs = [(weight(c, c, 2 * r), zeros(c), r) for r in RATIOS]
    vae.head_w, vae.head_b = weight(1, c, 3), zeros(1)
    return vae


class FakeLM(FastLM):
    def __init__(self, tokens: list[int]):
        self.tokens = iter(tokens)
        self.embed_w = mx.zeros((4, 2), dtype=mx.float16)
        self.speech_token_ids = (0, 1, 2, 3)
        self._stop_indices = (1, 3)

    def prefill(self, *args: object) -> mx.array:
        return mx.zeros((1, 1, 2), dtype=mx.float16)

    forward = prefill

    def logits(self, hidden: mx.array, *, speech_only: bool = False) -> mx.array:
        logits = mx.full((1, 1, 4), -10.0)
        logits[0, 0, next(self.tokens)] = 10.0
        return logits


@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("limit", [False, True])
def test_generation_preserves_audio_history(semantic: bool, limit: bool) -> None:
    vae = tiny_decoder()
    config = SimpleNamespace(
        hidden_size=2,
        num_hidden_layers=0,
        head_dim=2,
        rope_theta=10000,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
        vocab_size=4,
        single_segment=False,
        speech_scaling_factor=1.0,
        speech_bias_factor=0.0,
    )
    model = SimpleNamespace(
        config=config,
        vae_decoder=vae,
        _fast_diff=None,
        acoustic_connector=lambda sample: mx.zeros((1, 1, 2), dtype=mx.float16),
    )
    samples = mx.array(
        np.random.RandomState(11).normal(size=(3, 64)).astype(np.float32)
    )
    expected = np.array(vae(samples.T[None].astype(mx.float16))).reshape(-1)
    # Two calls on the same model also check that decoder state is invocation-local.
    for _ in range(2):
        model._fast_lm = FakeLM([2, 2, 1, 0, 2, 3])
        chunks = []
        resets = []

        def feedback(chunk: np.ndarray, chunks: list = chunks) -> np.ndarray:
            chunks.append(chunk.copy())
            return np.zeros((1, 1, 2), dtype=np.float32)

        with (
            patch.object(
                VAEDecoder, "__call__", autospec=True, side_effect=VAEDecoder.__call__
            ) as batch_decode,
            patch.object(
                generation,
                "dpm_solver_2m",
                side_effect=[samples[i : i + 1] for i in range(3)],
            ),
        ):
            audio, metrics = generation.generate(
                model,
                [0],
                generation.GenerationOptions(
                    cfg_scale=1, max_speech_tokens=3 if limit else 5
                ),
                semantic_encoder_fn=feedback if semantic else None,
                semantic_reset_fn=lambda resets=resets: resets.append(True),
            )
        assert batch_decode.call_count == (0 if semantic else 1)
        assert metrics.num_speech_tokens == 3
        assert resets == [True]
        np.testing.assert_allclose(audio, expected, atol=2e-3, rtol=2e-3)
        if semantic:
            assert len(chunks) == 3
            assert all(chunk.shape == (3200,) for chunk in chunks)
            np.testing.assert_allclose(
                np.concatenate(chunks), expected, atol=2e-3, rtol=2e-3
            )
            np.testing.assert_array_equal(np.concatenate(chunks), audio)
