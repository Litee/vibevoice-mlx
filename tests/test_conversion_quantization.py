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


@pytest.fixture(params=[None, (4, 64), (8, 32), (8, 64)])
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
