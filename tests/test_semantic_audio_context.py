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
        self.inputs: list[np.ndarray] = []

    def prefill(self, *args: object) -> mx.array:
        return mx.zeros((1, 1, 2), dtype=mx.float16)

    def forward(self, embedding: mx.array, *args: object) -> mx.array:
        self.inputs.append(np.array(embedding))
        return self.prefill()

    def logits(self, hidden: mx.array, *, speech_only: bool = False) -> mx.array:
        logits = mx.full((1, 1, 4), -10.0)
        logits[0, 0, next(self.tokens)] = 10.0
        return logits


@pytest.mark.parametrize("semantic", ["numpy", "mlx", None])
@pytest.mark.parametrize("limit", [False, True])
@pytest.mark.parametrize("solver", ["dpm", "sde"])
def test_generation_preserves_audio_history(
    semantic: str | None, limit: bool, solver: str
) -> None:
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
            assert isinstance(chunk, np.ndarray)
            chunks.append(chunk.copy())
            return chunk[:2].reshape(1, 1, 2)

        def feedback_mlx(chunk: mx.array, chunks: list = chunks) -> mx.array:
            assert isinstance(chunk, mx.array)
            assert chunk.dtype == mx.float32
            chunks.append(np.array(chunk))
            return chunk[:2].reshape(1, 1, 2)

        callback = feedback if semantic == "numpy" else None
        if semantic == "mlx":

            class NativeOnlyFeedback(generation.MLXSemanticCallback):
                def __call__(self, chunk: np.ndarray) -> np.ndarray:
                    raise AssertionError("Generation must use native MLX feedback")

            callback = NativeOnlyFeedback(feedback_mlx)

        with (
            patch.object(
                VAEDecoder, "__call__", autospec=True, side_effect=VAEDecoder.__call__
            ) as batch_decode,
            patch.object(
                generation,
                "dpm_solver_2m" if solver == "dpm" else "dpm_solver_sde_2m",
                side_effect=[samples[i : i + 1] for i in range(3)],
            ) as selected_solver,
            patch.object(
                generation,
                "dpm_solver_sde_2m" if solver == "dpm" else "dpm_solver_2m",
                side_effect=AssertionError("Wrong solver selected"),
            ) as other_solver,
        ):
            audio, metrics = generation.generate(
                model,
                [0],
                generation.GenerationOptions(
                    solver=solver, cfg_scale=1, max_speech_tokens=3 if limit else 5
                ),
                semantic_encoder_fn=callback,
                semantic_reset_fn=lambda resets=resets: resets.append(True),
            )
        assert batch_decode.call_count == (0 if semantic else 1)
        assert selected_solver.call_count == 3
        other_solver.assert_not_called()
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
            for chunk, position in zip(chunks, [0, 1, 4]):
                np.testing.assert_array_equal(
                    model._fast_lm.inputs[position],
                    chunk[:2].reshape(1, 1, 2).astype(np.float16),
                )


def test_mlx_callback_preserves_numpy_contract() -> None:
    def encode(audio: mx.array) -> mx.array:
        assert isinstance(audio, mx.array)
        assert audio.dtype == mx.float32
        return audio[:2].reshape(1, 1, 2).astype(mx.float16)

    callback = generation.MLXSemanticCallback(encode)
    embedding = callback(np.array([0.25, -0.5, 1.0], dtype=np.float16))
    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float16
    np.testing.assert_array_equal(embedding, [[[0.25, -0.5]]])


def test_stateful_native_feedback_preserves_lazy_history_and_reset() -> None:
    class DeferredLM(FakeLM):
        def __init__(self) -> None:
            super().__init__([2, 2, 1, 0, 2, 3])
            self.deferred_inputs: list[mx.array] = []

        def forward(self, embedding: mx.array, *args: object) -> mx.array:
            self.deferred_inputs.append(embedding)
            return embedding

        def select_token(self, *args: object, **kwargs: object) -> int:
            # Token decisions are scripted; do not evaluate feedback through argmax.
            return next(self.tokens)

    class StatefulFeedback(generation.MLXSemanticCallback):
        def __init__(self) -> None:
            super().__init__(self.encode)
            self.history = mx.zeros((1, 1, 2), dtype=mx.float32)
            self.chunks: list[mx.array] = []
            self.embeddings: list[mx.array] = []
            self.reset_offsets: list[int] = []

        def __call__(self, chunk: np.ndarray) -> np.ndarray:
            raise AssertionError("Native generation must not invoke the NumPy adapter")

        def encode(self, chunk: mx.array) -> mx.array:
            assert isinstance(chunk, mx.array)
            assert chunk.dtype == mx.float32
            self.chunks.append(chunk)
            self.history = self.history + chunk[:2].reshape(1, 1, 2) + 0.25
            self.embeddings.append(self.history)
            return self.history

        def reset(self) -> None:
            self.reset_offsets.append(len(self.chunks))
            self.history = mx.zeros((1, 1, 2), dtype=mx.float32)

    config = SimpleNamespace(
        hidden_size=2,
        num_hidden_layers=0,
        head_dim=2,
        rope_theta=10000,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
        single_segment=False,
        speech_scaling_factor=1.0,
        speech_bias_factor=0.0,
    )
    lm = DeferredLM()
    model = SimpleNamespace(
        config=config,
        vae_decoder=tiny_decoder(),
        _fast_lm=lm,
        _fast_diff=None,
        acoustic_connector=lambda sample: mx.zeros((1, 1, 2), dtype=mx.float16),
    )
    feedback = StatefulFeedback()
    samples = mx.arange(3 * 64, dtype=mx.float32).reshape(3, 64) / 64
    with patch.object(
        generation, "dpm_solver_2m", side_effect=[samples[i : i + 1] for i in range(3)]
    ):
        audio, metrics = generation.generate(
            model,
            [0],
            generation.GenerationOptions(cfg_scale=1, max_speech_tokens=5),
            semantic_encoder_fn=feedback,
            semantic_reset_fn=feedback.reset,
        )

    # Callback and LM retained only MLX arrays; synchronize for assertions here.
    assert metrics.num_speech_tokens == 3
    assert feedback.reset_offsets == [2]
    assert len(feedback.chunks) == len(feedback.embeddings) == 3
    assert len(lm.deferred_inputs) == 5
    chunks = [np.array(chunk) for chunk in feedback.chunks]
    np.testing.assert_array_equal(np.concatenate(chunks), audio)
    expected_history = np.zeros((1, 1, 2), dtype=np.float32)
    for index, lm_position in enumerate([0, 1, 4]):
        if index == 2:
            expected_history = np.zeros_like(expected_history)
        expected_history = expected_history + chunks[index][:2].reshape(1, 1, 2) + 0.25
        np.testing.assert_array_equal(
            np.array(feedback.embeddings[index]), expected_history
        )
        np.testing.assert_array_equal(
            np.array(lm.deferred_inputs[lm_position]),
            expected_history.astype(np.float16),
        )
