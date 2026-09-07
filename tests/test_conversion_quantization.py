"""Re-exported quantized checkpoints retain their loading behavior."""

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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from convert import convert_model
from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel, compute_rope


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


PACKED_FORMATS = [(4, 32), (4, 64), (8, 32), (8, 64)]


@pytest.fixture(params=[None, *PACKED_FORMATS])
def quantization(request: pytest.FixtureRequest) -> dict[str, int] | None:
    if request.param is None:
        return None
    bits, group_size = request.param
    return {"bits": bits, "group_size": group_size}


@pytest.fixture(params=[True, False], ids=["tied", "untied"])
def checkpoint(
    tmp_path: Path, request: pytest.FixtureRequest, quantization: dict[str, int] | None
) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    config = VibeVoiceConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=64,
        vocab_size=8,
        diffusion_layers=0,
        tie_word_embeddings=request.param,
        quantization=quantization,
    )
    model = VibeVoiceModel(config)
    if quantization is not None:
        nn.quantize(
            model.model,
            **quantization,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
    weights = dict(tree_flatten(model.parameters()))
    weights.update(tiny_vae_weights(config.vae_dim))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    (directory / "config.json").write_text(json.dumps(asdict(config)))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>")
        ),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(str(directory))
    return directory


def logits(model: VibeVoiceModel, config: VibeVoiceConfig) -> np.ndarray:
    ids = mx.array([[1, 4, 2]])
    cos, sin = compute_rope(mx.arange(3), config.head_dim, config.rope_theta)
    return np.asarray(
        model.get_logits(model.model(model.model.embed_tokens(ids), cos, sin))
    )


def test_reexport_remains_loadable_with_identical_outputs(
    checkpoint: Path, tmp_path: Path, quantization: dict[str, int] | None
) -> None:
    original, original_config = load_model(str(checkpoint))
    expected = logits(original, original_config)
    output = tmp_path / "reexported"

    convert_model(str(checkpoint), output, tokenizer_id=str(checkpoint))
    restored, restored_config = load_model(str(output))

    assert restored_config.quantization == quantization
    assert restored_config.tie_word_embeddings == original_config.tie_word_embeddings
    np.testing.assert_array_equal(logits(restored, restored_config), expected)
    audio = restored.vae_decoder(
        mx.zeros((1, restored_config.vae_dim, 1), dtype=mx.float16)
    )
    assert audio.shape == (1, 1, 3200)
    np.testing.assert_array_equal(np.asarray(audio), 0.25)


@pytest.mark.parametrize("quantization", PACKED_FORMATS, indirect=True)
def test_matching_load_override_preserves_model_output(
    checkpoint: Path, quantization: dict[str, int]
) -> None:
    original, original_config = load_model(str(checkpoint))
    restored, restored_config = load_model(
        str(checkpoint), quantize_bits=quantization["bits"]
    )
    assert restored_config.quantization == quantization
    np.testing.assert_array_equal(
        logits(restored, restored_config), logits(original, original_config)
    )


@pytest.mark.parametrize("quantization", PACKED_FORMATS, indirect=True)
def test_matching_conversion_override_preserves_packed_tensors(
    checkpoint: Path, tmp_path: Path, quantization: dict[str, int]
) -> None:
    output = tmp_path / "matching-override"
    convert_model(
        str(checkpoint),
        output,
        tokenizer_id=str(checkpoint),
        quantize_bits=quantization["bits"],
    )
    restored, restored_config = load_model(str(output))
    original, original_config = load_model(str(checkpoint))
    assert restored_config.quantization == quantization
    np.testing.assert_array_equal(
        logits(restored, restored_config), logits(original, original_config)
    )
    source_weights = mx.load(str(checkpoint / "model.safetensors"))
    output_weights = mx.load(str(output / "model.safetensors"))
    assert source_weights.keys() == output_weights.keys()
    for name, value in source_weights.items():
        assert output_weights[name].dtype == value.dtype
        np.testing.assert_array_equal(
            np.asarray(output_weights[name]), np.asarray(value)
        )


@pytest.mark.parametrize("quantization", PACKED_FORMATS, indirect=True)
@pytest.mark.parametrize("operation", ["load", "convert"])
def test_different_bits_fail_before_reading_tensors_or_writing_output(
    checkpoint: Path, tmp_path: Path, quantization: dict[str, int], operation: str
) -> None:
    source_bits = quantization["bits"]
    requested_bits = 8 if source_bits == 4 else 4
    # An unreadable payload makes accidental tensor loading observable without
    # mocking either API's internal loaders.
    weights_path = checkpoint / "model.safetensors"
    weights_path.write_bytes(b"tensor loading must not be reached")
    output = tmp_path / "unsupported-requantization"
    with pytest.raises(
        ValueError, match=rf"INT{source_bits}.*INT{requested_bits}.*unsupported"
    ):
        if operation == "load":
            load_model(str(checkpoint), quantize_bits=requested_bits)
        else:
            convert_model(
                str(checkpoint),
                output,
                tokenizer_id=str(checkpoint),
                quantize_bits=requested_bits,
            )
    assert not output.exists()
    assert weights_path.read_bytes() == b"tensor loading must not be reached"


@pytest.mark.parametrize("quantization", [None], indirect=True)
@pytest.mark.parametrize("bits", [4, 8])
def test_full_precision_conversion_still_matches_runtime_quantization(
    checkpoint: Path, tmp_path: Path, bits: int
) -> None:
    weights_path = checkpoint / "model.safetensors"
    # Use the converter's output precision so this comparison isolates
    # quantization behavior from its existing float32-to-float16 cast.
    weights = {
        name: value.astype(mx.float16)
        for name, value in mx.load(str(weights_path)).items()
    }
    mx.eval(weights)
    mx.save_safetensors(str(weights_path), weights)
    runtime, runtime_config = load_model(str(checkpoint), quantize_bits=bits)
    output = tmp_path / "new-quantization"
    convert_model(
        str(checkpoint), output, tokenizer_id=str(checkpoint), quantize_bits=bits
    )
    converted, converted_config = load_model(str(output))
    assert converted_config.quantization == {
        "bits": bits,
        "group_size": 64 if bits == 4 else 32,
    }
    np.testing.assert_array_equal(
        logits(converted, converted_config), logits(runtime, runtime_config)
    )
