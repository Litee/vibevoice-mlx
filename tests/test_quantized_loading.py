"""Quantized loading rejects incomplete or incompatible checkpoints."""

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx import nn
from mlx.utils import tree_flatten

from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel, compute_rope


@pytest.fixture
def model() -> Iterator[VibeVoiceModel]:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(19)
        yield VibeVoiceModel(
            VibeVoiceConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=32,
                intermediate_size=64,
                vocab_size=16,
                diffusion_layers=0,
                speech_start_id=0,
                speech_end_id=1,
                speech_diffusion_id=2,
                eos_id=3,
            )
        )
    finally:
        mx.set_default_device(previous)


def checkpoint_weights(
    model: VibeVoiceModel,
    prequantized: bool,
    bits: int = 4,
) -> dict[str, mx.array]:
    if prequantized:
        nn.quantize(
            model.model,
            bits=bits,
            group_size=64 if bits == 4 else 32,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
        model.config.quantization = {
            "bits": bits,
            "group_size": 64 if bits == 4 else 32,
        }
    return dict(tree_flatten(model.parameters()))


def save_checkpoint(
    path: Path, model: VibeVoiceModel, weights: dict[str, mx.array]
) -> None:
    (path / "config.json").write_text(json.dumps(asdict(model.config)))
    mx.save_safetensors(
        str(path / "model.safetensors"),
        {
            **weights,
            **tiny_vae_weights(model.config.vae_dim),
        },
    )


@pytest.mark.parametrize("suffix", ["weight", "scales", "biases"])
def test_prequantized_checkpoint_requires_every_weight(
    model: VibeVoiceModel,
    tmp_path: Path,
    suffix: str,
) -> None:
    weights = checkpoint_weights(model, prequantized=True)
    name = f"model.layers.0.self_attn.q_proj.{suffix}"
    del weights[name]
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match=name):
        load_model(str(tmp_path))


@pytest.mark.parametrize("prequantized", [False, True])
def test_quantized_loading_requires_every_layer(
    model: VibeVoiceModel,
    tmp_path: Path,
    prequantized: bool,
) -> None:
    weights = checkpoint_weights(model, prequantized)
    weights = {
        name: value
        for name, value in weights.items()
        if not name.startswith("model.layers.1.")
    }
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match="model.layers.1"):
        load_model(str(tmp_path), quantize_bits=None if prequantized else 4)


@pytest.mark.parametrize("prequantized", [False, True])
def test_quantized_loading_rejects_wrong_shapes(
    model: VibeVoiceModel,
    tmp_path: Path,
    prequantized: bool,
) -> None:
    weights = checkpoint_weights(model, prequantized)
    name = "model.layers.0.self_attn.q_proj.weight"
    weights[name] = weights[name][:32]
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match=name):
        load_model(str(tmp_path), quantize_bits=None if prequantized else 4)


@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "model.norm.weight",
        "acoustic_connector.fc1.bias",
        "semantic_connector.fc2.weight",
    ],
)
def test_quantized_loading_requires_non_layer_parameters(
    model: VibeVoiceModel,
    tmp_path: Path,
    prequantized: bool,
    name: str,
) -> None:
    weights = checkpoint_weights(model, prequantized)
    del weights[name]
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match=name):
        load_model(str(tmp_path), quantize_bits=None if prequantized else 4)


@pytest.mark.parametrize("prequantized", [False, True])
def test_quantized_loading_rejects_unexpected_parameters(
    model: VibeVoiceModel,
    tmp_path: Path,
    prequantized: bool,
) -> None:
    weights = checkpoint_weights(model, prequantized)
    weights["model.layers.2.input_layernorm.weight"] = mx.ones((64,))
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match="model.layers.2.input_layernorm.weight"):
        load_model(str(tmp_path), quantize_bits=None if prequantized else 4)


@pytest.mark.parametrize("prequantized", [False, True])
def test_untied_quantized_checkpoint_requires_output_head(
    model: VibeVoiceModel,
    tmp_path: Path,
    prequantized: bool,
) -> None:
    model.config.tie_word_embeddings = False
    model.lm_head = nn.Linear(64, 16, bias=False)
    weights = checkpoint_weights(model, prequantized)
    del weights["lm_head.weight"]
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match="lm_head.weight"):
        load_model(str(tmp_path), quantize_bits=None if prequantized else 4)


@pytest.mark.parametrize(
    "metadata", [{"bits": 8, "group_size": 64}, {"bits": 4, "group_size": 32}]
)
def test_prequantized_loading_rejects_incompatible_metadata(
    model: VibeVoiceModel,
    tmp_path: Path,
    metadata: dict[str, int],
) -> None:
    weights = checkpoint_weights(model, prequantized=True)
    model.config.quantization = metadata
    save_checkpoint(tmp_path, model, weights)

    with pytest.raises(ValueError, match="Expected shape"):
        load_model(str(tmp_path))


@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_valid_quantized_loading_preserves_model_output(
    model: VibeVoiceModel,
    tmp_path: Path,
    prequantized: bool,
    bits: int,
    tied: bool,
) -> None:
    if not tied:
        model.config.tie_word_embeddings = False
        model.lm_head = nn.Linear(64, 16, bias=False)
    weights = checkpoint_weights(model, prequantized, bits)
    # These payloads are intentionally outside the nn.Module parameter schema.
    weights["semantic_encoder.unused.weight"] = mx.ones((1,))
    if tied:
        weights["lm_head.weight"] = mx.zeros((16, 64))
    save_checkpoint(tmp_path, model, weights)
    if not prequantized:
        nn.quantize(
            model.model,
            bits=bits,
            group_size=64 if bits == 4 else 32,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )

    loaded, config = load_model(
        str(tmp_path), quantize_bits=None if prequantized else bits
    )

    assert config.tie_word_embeddings == tied
    assert isinstance(loaded.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    cos, sin = compute_rope(mx.array([0]), 32, config.rope_theta)
    ids = mx.array([[4]])
    expected = model.get_logits(model.model(model.model.embed_tokens(ids), cos, sin))
    actual = loaded.get_logits(loaded.model(loaded.model.embed_tokens(ids), cos, sin))
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=1e-6, rtol=1e-6
    )
