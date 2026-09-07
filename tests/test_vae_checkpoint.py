"""Incomplete decoders must fail while loading, before audio generation."""

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx.utils import tree_flatten

from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


@pytest.fixture(params=["mlx", "hf"])
def checkpoint_format(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=["full", "prequantized", "runtime"])
def quantization_mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def checkpoint(tmp_path: Path, checkpoint_format: str, quantization_mode: str) -> Path:
    config = VibeVoiceConfig(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=8,
        vocab_size=8,
        diffusion_layers=1,
        vae_dim=1,
    )
    if quantization_mode == "prequantized":
        config.quantization = {"bits": 8, "group_size": 32}
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    weights = dict(tree_flatten(VibeVoiceModel(config).parameters()))
    weights.update(tiny_vae_weights())
    if checkpoint_format == "hf":
        prefixes = {
            "model.": "model.language_model.",
            "diffusion_head.": "model.prediction_head.",
            "acoustic_connector.": "model.acoustic_connector.",
            "semantic_connector.": "model.semantic_connector.",
            "vae_decoder.": "model.acoustic_tokenizer.decoder.",
        }
        weights = {
            replacement + key[len(prefix) :]: value
            for key, value in weights.items()
            for prefix, replacement in prefixes.items()
            if key.startswith(prefix)
        }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path


@pytest.mark.parametrize(
    "missing_suffix",
    [
        "stages.2.1.",
        "stages.0.7.ffn.linear2.bias",
        "stages.6.2.norm.weight",
        "upsample_layers.3.0.convtr.convtr.",
        "upsample_layers.6.0.convtr.convtr.weight",
        "upsample_layers.1.0.convtr.convtr.bias",
        "upsample_layers.0.0.conv.conv.",
        "upsample_layers.0.0.conv.conv.weight",
        "upsample_layers.0.0.conv.conv.bias",
        "head.conv.conv.",
        "head.conv.conv.weight",
        "head.conv.conv.bias",
        "",
    ],
    ids=[
        "whole-block",
        "block-bias",
        "block-norm",
        "whole-upsampler",
        "upsampler-weight",
        "upsampler-bias",
        "whole-init",
        "init-weight",
        "init-bias",
        "whole-head",
        "head-weight",
        "head-bias",
        "entire-decoder",
    ],
)
def test_load_model_rejects_incomplete_decoder(
    checkpoint: Path,
    checkpoint_format: str,
    quantization_mode: str,
    missing_suffix: str,
) -> None:
    weights_path = checkpoint / "model.safetensors"
    weights = mx.load(str(weights_path))
    prefix = (
        "model.acoustic_tokenizer.decoder."
        if checkpoint_format == "hf"
        else "vae_decoder."
    )
    omitted = [key for key in weights if key.startswith(prefix + missing_suffix)]
    assert omitted
    for key in omitted:
        del weights[key]
    mx.eval(weights)
    mx.save_safetensors(str(weights_path), weights)

    with pytest.raises(ValueError, match="VAE decoder") as error:
        load_model(
            str(checkpoint),
            quantize_bits=8 if quantization_mode == "runtime" else None,
        )
    for key in omitted:
        assert key in str(error.value)


def test_complete_decoder_loads_and_decodes_without_optional_encoders(
    checkpoint: Path,
    quantization_mode: str,
) -> None:
    model, config = load_model(
        str(checkpoint),
        quantize_bits=8 if quantization_mode == "runtime" else None,
    )
    audio = model.vae_decoder(mx.zeros((1, config.vae_dim, 1), dtype=mx.float16))
    assert audio.shape == (1, 1, 3200)
    np.testing.assert_array_equal(np.asarray(audio), 0.25)
