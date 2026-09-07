"""The MLX loader preserves frame preparation and the NumPy callback contract."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx import e2e_pipeline, load_weights, semantic_encoder
from vibevoice_mlx.generate import MLXSemanticCallback


@pytest.mark.parametrize(
    "length", [0, 3, 3200, 3207], ids=["empty", "short", "exact", "long"]
)
def test_loaded_mlx_callback_prepares_frames_and_resets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, length: int
) -> None:
    class RecordingEncoder:
        def __init__(self) -> None:
            self.inputs: list[mx.array] = []
            self.caches = [mx.zeros((1,), dtype=mx.float32)]
            self.resets = 0

        def __call__(self, audio: mx.array) -> mx.array:
            assert isinstance(audio, mx.array)
            assert audio.shape == (1, 1, 3200)
            assert audio.dtype == mx.float32
            self.inputs.append(audio)
            features = mx.arange(128, dtype=mx.float32).reshape(1, 128, 1)
            features = features + audio[..., :1] + audio[..., -1:] + self.caches[0]
            self.caches = [self.caches[0] + 1]
            return features

        def reset_caches(self) -> None:
            self.resets += 1
            self.caches = [mx.zeros((1,), dtype=mx.float32)]

    encoder = RecordingEncoder()
    connector_inputs: list[mx.array] = []

    def connector(features: mx.array) -> mx.array:
        assert isinstance(features, mx.array)
        assert features.shape == (1, 1, 128)
        assert features.dtype == mx.float16
        connector_inputs.append(features)
        return mx.concatenate([features[..., :1], features[..., -1:]], axis=-1)

    weights = {"semantic_encoder.fixture": mx.zeros((1,))}
    monkeypatch.setattr(load_weights, "resolve_model_path", lambda _: tmp_path)
    monkeypatch.setattr(load_weights, "_load_safetensors", lambda _: weights)
    monkeypatch.setattr(semantic_encoder, "load_semantic_encoder", lambda _: encoder)
    result = e2e_pipeline._try_mlx_semantic(
        SimpleNamespace(semantic_connector=connector), SimpleNamespace(), str(tmp_path)
    )
    assert result is not None
    callback, reset = result
    assert isinstance(callback, MLXSemanticCallback)
    assert encoder.resets == 1  # The loader clears warm-up history.
    assert len(encoder.inputs) == 1
    np.testing.assert_array_equal(np.array(encoder.inputs[0]), np.zeros((1, 1, 3200)))

    chunk = (np.arange(length, dtype=np.float32) + 1) / 8
    # Float16 input also verifies preparation restores the encoder's float32 input.
    chunk = chunk.astype(np.float16)
    first = callback.encode_mlx(mx.array(chunk))
    second = callback.encode_mlx(mx.array(chunk))
    assert isinstance(first, mx.array)
    assert isinstance(second, mx.array)
    reset()
    compatible = callback(chunk)
    assert encoder.resets == 2
    assert len(encoder.inputs) == 4
    assert len(connector_inputs) == 3
    assert isinstance(compatible, np.ndarray)
    assert compatible.dtype == np.float16

    expected_audio = np.zeros((1, 1, 3200), dtype=np.float32)
    count = min(length, 3200)
    expected_audio[0, 0, :count] = chunk[:count]
    for audio in encoder.inputs[1:]:
        np.testing.assert_array_equal(np.array(audio), expected_audio)
    endpoint_sum = expected_audio[0, 0, 0] + expected_audio[0, 0, -1]
    for features, history in zip(connector_inputs, [0, 1, 0]):
        expected = (np.arange(128, dtype=np.float32) + endpoint_sum + history).astype(
            np.float16
        )
        np.testing.assert_array_equal(np.array(features), expected.reshape(1, 1, 128))
    expected_first = np.array([endpoint_sum, endpoint_sum + 127], dtype=np.float16)
    expected_second = np.array([endpoint_sum + 1, endpoint_sum + 128], dtype=np.float16)
    np.testing.assert_array_equal(np.array(first), expected_first.reshape(1, 1, 2))
    np.testing.assert_array_equal(np.array(second), expected_second.reshape(1, 1, 2))
    np.testing.assert_array_equal(compatible, np.array(first))
