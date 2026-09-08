"""Short voice references preserve the full padded encoder's prefix embeddings."""

from collections.abc import Iterator
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from vibevoice_mlx import load_weights, vae_encoder
from vibevoice_mlx.e2e_pipeline import (
    VOICE_CLONE_SAMPLES,
    encode_voice_reference,
)
from vibevoice_mlx.model import Connector


@pytest.fixture(params=["cpu", "metal"])
def device(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.param == "metal" and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    with mx.stream(mx.cpu if request.param == "cpu" else mx.gpu):
        yield


@pytest.fixture
def voice_model(device: None) -> tuple[SimpleNamespace, SimpleNamespace]:
    rng = np.random.RandomState(17)

    def weight(*shape: int) -> mx.array:
        return mx.array(rng.normal(0, 0.1, shape).astype(np.float16))

    weights = {
        "stem_w": weight(2, 1, 7),
        "stem_b": weight(2),
        "head_w": weight(4, 2, 7),
        "head_b": weight(4),
        "blocks": [],
        "ds": [],
    }
    for depth in vae_encoder.DEPTHS:
        weights["blocks"].append(
            [
                {
                    "norm_w": mx.ones((2,), dtype=mx.float16),
                    "conv_w": weight(2, 1, 7),
                    "conv_b": weight(2),
                    "gamma": weight(2),
                    "ffn_norm_w": mx.ones((2,), dtype=mx.float16),
                    "ffn_l1_w": weight(4, 2),
                    "ffn_l1_b": weight(4),
                    "ffn_l2_w": weight(2, 4),
                    "ffn_l2_b": weight(2),
                    "ffn_gamma": weight(2),
                }
                for _ in range(depth)
            ]
        )
    weights["ds"] = [
        {"w": weight(2, 2, 2 * ratio), "b": weight(2)} for ratio in vae_encoder.RATIOS
    ]
    mx.random.seed(29)
    connector = Connector(4, 8)
    connector.update(
        tree_map(lambda value: value.astype(mx.float16), connector.parameters())
    )
    model = SimpleNamespace(_encoder_weights=weights, acoustic_connector=connector)
    config = SimpleNamespace(speech_bias_factor=-0.05, speech_scaling_factor=0.2)
    return model, config


def full_padded_embeddings(
    wav: np.ndarray, tokens: object, model: SimpleNamespace, config: SimpleNamespace
) -> np.ndarray:
    """Previous public behavior, using the real encoder and connector."""
    padded = np.zeros(VOICE_CLONE_SAMPLES, dtype=np.float32)
    length = min(len(wav), VOICE_CLONE_SAMPLES)
    padded[:length] = wav[:length]
    latent = vae_encoder.encode_audio(
        mx.array(padded).reshape(1, 1, -1), model._encoder_weights
    )
    latent = latent[:, : min(latent.shape[1], tokens), :]
    features = (
        (latent + config.speech_bias_factor) * config.speech_scaling_factor
    ).astype(mx.float16)
    return np.array(model.acoustic_connector(features)[0])


@pytest.mark.parametrize(
    ("samples", "tokens", "encoded_samples"),
    [
        (0, 1, 3200),
        (1, 1, 3200),
        (3200, 1, 3200),
        (3201, 2, 6400),
        (12000, 2, 6400),
        (3200, 75, 240000),
        (260000, 100, 240000),
    ],
)
def test_voice_encoding_limits_input_without_changing_embeddings(
    voice_model: tuple[SimpleNamespace, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    samples: int,
    tokens: int,
    encoded_samples: int,
) -> None:
    model, config = voice_model
    wav = np.random.RandomState(37).normal(0, 0.2, samples).astype(np.float32)
    expected = full_padded_embeddings(wav, tokens, model, config)
    input_shapes = []
    encode = vae_encoder.encode_audio

    def observe_input(audio: mx.array, weights: dict) -> mx.array:
        input_shapes.append(audio.shape)
        assert audio.dtype == mx.float32
        return encode(audio, weights)

    def unexpected_load(_: str) -> None:
        raise AssertionError(
            "Pre-extracted encoder weights must avoid checkpoint loading"
        )

    monkeypatch.setattr(vae_encoder, "encode_audio", observe_input)
    monkeypatch.setattr(load_weights, "resolve_model_path", unexpected_load)
    actual = encode_voice_reference(wav, tokens, model, config, "unused")
    assert input_shapes == [(1, 1, encoded_samples)]
    assert actual.shape == (min(tokens, 75), 8)
    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "tokens",
    [0, -1, -80, 2.5, 76.5, None, "2", np.int64(2), np.int64(2**63 - 1), True, False],
)
def test_voice_encoding_preserves_existing_token_count_boundaries(
    voice_model: tuple[SimpleNamespace, SimpleNamespace], tokens: object
) -> None:
    model, config = voice_model
    wav = np.ones((3201,), dtype=np.float32)
    try:
        expected = full_padded_embeddings(wav, tokens, model, config)
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            encode_voice_reference(wav, tokens, model, config, "unused")
    else:
        actual = encode_voice_reference(wav, tokens, model, config, "unused")
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_voice_encoding_preserves_cached_encoder_fallback(
    voice_model: tuple[SimpleNamespace, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = voice_model
    wav = np.ones((3201,), dtype=np.float32)
    expected = full_padded_embeddings(wav, 2, model, config)
    monkeypatch.setattr(
        encode_voice_reference,
        "_enc_cache",
        {"cached-voice": model._encoder_weights},
        raising=False,
    )
    model._encoder_weights = None

    def unexpected_load(_: str) -> None:
        raise AssertionError("Cached encoder weights must avoid checkpoint loading")

    monkeypatch.setattr(load_weights, "resolve_model_path", unexpected_load)
    actual = encode_voice_reference(wav, 2, model, config, "cached-voice")
    np.testing.assert_array_equal(actual, expected)


def test_voice_encoding_keeps_last_needed_sample_and_excludes_future(
    voice_model: tuple[SimpleNamespace, SimpleNamespace],
) -> None:
    model, _ = voice_model
    weights = model._encoder_weights
    weights["stem_w"] = mx.zeros((2, 1, 1), dtype=mx.float16)
    weights["stem_w"][0, 0, 0] = 1
    weights["stem_b"] = mx.zeros((2,), dtype=mx.float16)
    weights["head_w"] = mx.zeros((4, 2, 1), dtype=mx.float16)
    weights["head_w"][0, 0, 0] = 1
    weights["head_b"] = mx.zeros((4,), dtype=mx.float16)
    for stage in weights["blocks"]:
        for block in stage:
            block["gamma"] = mx.zeros_like(block["gamma"])
            block["ffn_gamma"] = mx.zeros_like(block["ffn_gamma"])
    for ratio, downsample in zip(vae_encoder.RATIOS, weights["ds"]):
        downsample["w"] = mx.zeros((2, 2, 2 * ratio), dtype=mx.float16)
        downsample["w"][0, 0, -1] = 1
        downsample["b"] = mx.zeros((2,), dtype=mx.float16)
    model.acoustic_connector = lambda features: features @ mx.eye(4, dtype=mx.float16)
    config = SimpleNamespace(speech_bias_factor=0.0, speech_scaling_factor=1.0)
    wav = np.zeros((6401,), dtype=np.float32)
    wav[3199], wav[6399], wav[6400] = 1, 2, 99
    actual = encode_voice_reference(wav, 2, model, config, "unused")
    np.testing.assert_array_equal(actual, [[1, 0, 0, 0], [2, 0, 0, 0]])
    np.testing.assert_array_equal(actual, full_padded_embeddings(wav, 2, model, config))
