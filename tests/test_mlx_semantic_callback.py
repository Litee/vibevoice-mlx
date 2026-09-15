"""The MLX loader preserves frame preparation and the NumPy callback contract."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx import e2e_pipeline, load_weights, semantic_encoder
from vibevoice_mlx.generate import MLXSemanticCallback


class StubSemanticEncoder:
    def __init__(self) -> None:
        self.caches: list[mx.array] = []

    def __call__(self, _: mx.array) -> mx.array:
        return mx.zeros((1, 128, 1), dtype=mx.float32)

    def reset_caches(self) -> None:
        pass


def semantic_model() -> SimpleNamespace:
    return SimpleNamespace(semantic_connector=lambda features: features)


def run_mlx_semantic(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> tuple[MLXSemanticCallback, Callable[[], None]] | None:
    monkeypatch.setattr(load_weights, "resolve_model_path", lambda _: model_path)
    return e2e_pipeline._try_mlx_semantic(
        semantic_model(), SimpleNamespace(), str(model_path)
    )


def capture_encoder_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, mx.array]]:
    loaded: list[dict[str, mx.array]] = []

    def load_encoder(weights: dict[str, mx.array]) -> StubSemanticEncoder:
        loaded.append(weights)
        return StubSemanticEncoder()

    monkeypatch.setattr(semantic_encoder, "load_semantic_encoder", load_encoder)
    return loaded


def test_mlx_semantic_loading_requests_only_encoder_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    semantic_weights = {
        "model.semantic_tokenizer.encoder.hf": mx.zeros((1,)),
        "semantic_encoder.converted": mx.ones((1,)),
    }
    unrelated = {
        "model.semantic_tokenizer.decoder.unrelated",
        "model.semantic_connector.unrelated",
        "language_model.unrelated",
    }
    mx.save_safetensors(str(tmp_path / "semantic.safetensors"), semantic_weights)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **dict.fromkeys(semantic_weights, "semantic.safetensors"),
                    **dict.fromkeys(unrelated, "missing-unrelated.safetensors"),
                }
            }
        )
    )

    selected: dict[str, mx.array] | None = None
    loaded = capture_encoder_weights(monkeypatch)
    load_safetensors = load_weights._load_safetensors

    def load_filtered(
        path: Path, tensor_filter: Callable[[str], bool]
    ) -> dict[str, mx.array]:
        nonlocal selected
        selected = load_safetensors(path, tensor_filter)
        return selected

    monkeypatch.setattr(load_weights, "_load_safetensors", load_filtered)

    result = run_mlx_semantic(monkeypatch, tmp_path)

    assert result is not None
    assert selected is not None
    assert selected.keys() == semantic_weights.keys()
    assert loaded[0] is selected


def test_mlx_semantic_loading_filters_unindexed_shards_after_opening_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    semantic_weights = {
        "model.semantic_tokenizer.encoder.hf": mx.zeros((1,)),
        "semantic_encoder.converted": mx.ones((1,)),
    }
    mx.save_safetensors(str(tmp_path / "semantic.safetensors"), semantic_weights)
    mx.save_safetensors(
        str(tmp_path / "unrelated.safetensors"),
        {"model.language_model.unrelated": mx.zeros((1,))},
    )
    opened: list[str] = []
    loaded = capture_encoder_weights(monkeypatch)
    mlx_load = load_weights.mx.load

    def record_load(path: str) -> dict[str, mx.array]:
        opened.append(Path(path).name)
        return mlx_load(path)

    monkeypatch.setattr(load_weights.mx, "load", record_load)

    result = run_mlx_semantic(monkeypatch, tmp_path)

    assert result is not None
    assert opened == ["semantic.safetensors", "unrelated.safetensors"]
    assert loaded[0].keys() == semantic_weights.keys()


@pytest.mark.parametrize("missing", ["shard", "tensor"])
def test_mlx_semantic_loading_handles_missing_selected_index_content(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    missing: str,
) -> None:
    tensor_name = "semantic_encoder.required"
    shard_name = "semantic.safetensors"
    if missing == "tensor":
        mx.save_safetensors(str(tmp_path / shard_name), {"unrelated": mx.zeros((1,))})
    else:
        shard_name = "missing.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}})
    )

    result = run_mlx_semantic(monkeypatch, tmp_path)

    assert result is None
    assert "Could not load semantic encoder" in caplog.text
    assert shard_name in caplog.text


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
    monkeypatch.setattr(load_weights, "_load_safetensors", lambda *_: weights)
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
