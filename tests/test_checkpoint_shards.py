"""Checkpoint file selection is unambiguous at the public model-loading seam."""

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx.utils import tree_flatten

from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


@pytest.fixture
def checkpoint(tmp_path: Path) -> dict[str, mx.array]:
    config = VibeVoiceConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        vocab_size=16,
        diffusion_layers=0,
        vae_dim=1,
    )
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    return {
        **dict(tree_flatten(VibeVoiceModel(config).parameters())),
        **tiny_vae_weights(config.vae_dim),
    }


def test_index_ignores_unreferenced_safetensors(
    tmp_path: Path, checkpoint: dict[str, mx.array]
) -> None:
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), checkpoint)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(checkpoint, "weights.safetensors")})
    )
    # A second model/export in the same directory is not part of this checkpoint.
    mx.save_safetensors(
        str(tmp_path / "z-backup.safetensors"),
        {"model.norm.weight": mx.full((8,), 17.0)},
    )

    loaded, _ = load_model(str(tmp_path))

    np.testing.assert_array_equal(np.array(loaded.model.norm.weight), np.ones((8,)))


@pytest.mark.parametrize("missing", ["shard", "tensor"])
def test_index_reports_missing_referenced_content(
    tmp_path: Path, checkpoint: dict[str, mx.array], missing: str
) -> None:
    weight_map = dict.fromkeys(checkpoint, "weights.safetensors")
    if missing == "shard":
        weight_map["model.norm.weight"] = "missing.safetensors"
        expected_error = FileNotFoundError
        message = "missing.safetensors"
    else:
        del checkpoint["model.norm.weight"]
        expected_error = ValueError
        message = "model.norm.weight.*weights.safetensors"
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), checkpoint)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    with pytest.raises(expected_error, match=message):
        load_model(str(tmp_path))


def test_legacy_checkpoint_rejects_duplicate_tensors(
    tmp_path: Path, checkpoint: dict[str, mx.array]
) -> None:
    mx.save_safetensors(str(tmp_path / "first.safetensors"), checkpoint)
    mx.save_safetensors(
        str(tmp_path / "second.safetensors"),
        {"model.norm.weight": checkpoint["model.norm.weight"]},
    )

    with pytest.raises(
        ValueError, match="Duplicate tensor 'model.norm.weight'.*first.*second"
    ):
        load_model(str(tmp_path))


@pytest.mark.parametrize("indexed", [False, True])
def test_noncanonical_shard_names_preserve_all_model_parameters(
    tmp_path: Path, checkpoint: dict[str, mx.array], indexed: bool
) -> None:
    shards = {
        "language.safetensors": {
            name: value
            for name, value in checkpoint.items()
            if name.startswith("model.")
        },
        "audio.safetensors": {
            name: value
            for name, value in checkpoint.items()
            if not name.startswith("model.")
        },
    }
    for filename, weights in shards.items():
        mx.save_safetensors(str(tmp_path / filename), weights)
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        name: filename
                        for filename, weights in shards.items()
                        for name in weights
                    }
                }
            )
        )

    loaded, _ = load_model(str(tmp_path))

    actual = dict(tree_flatten(loaded.parameters()))
    expected = {
        name: value
        for name, value in checkpoint.items()
        if not name.startswith("vae_decoder.")
    }
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        np.testing.assert_array_equal(np.array(parameter), np.array(expected[name]))


@pytest.mark.parametrize(
    "index",
    [
        [],
        {},
        {"weight_map": []},
        {"weight_map": {}},
        {"weight_map": {"model.norm.weight": 5}},
    ],
)
def test_invalid_index_does_not_fall_back_to_directory_scan(
    tmp_path: Path, checkpoint: dict[str, mx.array], index: object
) -> None:
    mx.save_safetensors(str(tmp_path / "model.safetensors"), checkpoint)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match="model.safetensors.index.json.*weight_map"):
        load_model(str(tmp_path))


@pytest.mark.parametrize("filename", ["../weights.safetensors", "/weights.safetensors"])
def test_index_rejects_shard_paths_outside_checkpoint(
    tmp_path: Path, checkpoint: dict[str, mx.array], filename: str
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(checkpoint, filename)})
    )

    with pytest.raises(ValueError, match="Invalid checkpoint shard path"):
        load_model(str(tmp_path))
