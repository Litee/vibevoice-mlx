"""Repeated conversion produces one complete current checkpoint."""

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

import convert
from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


def save_source(path: Path, audio_bias: float) -> None:
    path.mkdir()
    config = VibeVoiceConfig(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=8,
        vocab_size=8,
        diffusion_layers=0,
    )
    weights = dict(tree_flatten(VibeVoiceModel(config).parameters()))
    weights.update(tiny_vae_weights(config.vae_dim))
    weights["vae_decoder.head.conv.conv.bias"] = mx.array([audio_bias])
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(asdict(config)))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(str(path))


def decoded_audio(path: Path) -> np.ndarray:
    model, config = load_model(str(path))
    return np.asarray(
        model.vae_decoder(mx.zeros((1, config.vae_dim, 1), dtype=mx.float16))
    )


@pytest.mark.parametrize(
    "first_limit,second_limit", [(10**9, 1024), (1024, 10**9), (512, 2048), (2048, 512)]
)
def test_replacing_shard_layout_loads_latest_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_limit: int,
    second_limit: int,
) -> None:
    with mx.stream(mx.cpu):
        first, second, output = (
            tmp_path / name for name in ("first", "second", "output")
        )
        save_source(first, 0.0)
        save_source(second, 0.5)
        monkeypatch.setattr(convert, "SHARD_SIZE", first_limit)
        convert.convert_model(str(first), output, tokenizer_id=str(first))
        np.testing.assert_array_equal(decoded_audio(output), 0.25)
        (output / "notes.txt").write_text("keep my notes")
        (output / "recordings").mkdir()
        (output / "recordings" / "sample.txt").write_text("keep my recording")

        monkeypatch.setattr(convert, "SHARD_SIZE", second_limit)
        convert.convert_model(str(second), output, tokenizer_id=str(second))

        np.testing.assert_array_equal(decoded_audio(output), 0.75)
        actual = {path.name for path in output.glob("*.safetensors")}
        if second_limit == 10**9:
            assert actual == {"model.safetensors"}
            assert not (output / "model.safetensors.index.json").exists()
        else:
            index = json.loads((output / "model.safetensors.index.json").read_text())
            assert actual == set(index["weight_map"].values())
        assert (output / "notes.txt").read_text() == "keep my notes"
        assert (output / "recordings" / "sample.txt").read_text() == "keep my recording"


def test_tokenizer_failure_preserves_existing_checkpoint(tmp_path: Path) -> None:
    with mx.stream(mx.cpu):
        first, second, output = (
            tmp_path / name for name in ("first", "second", "output")
        )
        save_source(first, 0.0)
        save_source(second, 0.5)
        convert.convert_model(str(first), output, tokenizer_id=str(first))
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        (second / "tokenizer.json").write_text("{broken")

        with pytest.raises(json.JSONDecodeError):
            convert.convert_model(str(second), output, tokenizer_id=str(second))

        assert {p.name: p.read_bytes() for p in output.iterdir()} == before
        np.testing.assert_array_equal(decoded_audio(output), 0.25)


@pytest.mark.parametrize("filename", ["voice.safetensors", "model-backup.safetensors"])
def test_unrelated_safetensors_are_preserved_and_rejected(
    tmp_path: Path, filename: str
) -> None:
    with mx.stream(mx.cpu):
        source, output = tmp_path / "source", tmp_path / "output"
        save_source(source, 0.0)
        output.mkdir()
        foreign = output / filename
        mx.save_safetensors(str(foreign), {"embeddings": mx.zeros((1, 4))})
        original = foreign.read_bytes()

        with pytest.raises(ValueError, match=filename):
            convert.convert_model(str(source), output, tokenizer_id=str(source))

        assert list(output.iterdir()) == [foreign]
        assert foreign.read_bytes() == original


@pytest.mark.parametrize("alias", [False, True])
def test_in_place_conversion_finishes_reads_before_replacing_files(
    tmp_path: Path, alias: bool
) -> None:
    with mx.stream(mx.cpu):
        source = tmp_path / "source"
        save_source(source, 0.5)
        output = source
        if alias:
            output = tmp_path / "alias"
            output.symlink_to(source, target_is_directory=True)

        convert.convert_model(str(source), output, tokenizer_id=str(source))

        np.testing.assert_array_equal(decoded_audio(source), 0.75)


def test_repeated_conversion_updates_nested_tokenizer_files(tmp_path: Path) -> None:
    with mx.stream(mx.cpu):
        source, output = tmp_path / "source", tmp_path / "output"
        save_source(source, 0.0)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(source))
        tokenizer.chat_template = {"default": "{{ messages }}", "tool_use": "old"}
        tokenizer.save_pretrained(str(source))
        convert.convert_model(str(source), output, tokenizer_id=str(source))
        note = next(output.rglob("tool_use.jinja")).parent / "notes.txt"
        note.write_text("keep this note")

        tokenizer.chat_template = {"default": "{{ messages }}", "tool_use": "new"}
        tokenizer.save_pretrained(str(source))
        convert.convert_model(str(source), output, tokenizer_id=str(source))

        restored = PreTrainedTokenizerFast.from_pretrained(str(output))
        assert restored.chat_template == tokenizer.chat_template
        assert note.read_text() == "keep this note"
        np.testing.assert_array_equal(decoded_audio(output), 0.25)


@pytest.mark.parametrize("template", [None, "{{ messages }}"])
def test_repeated_conversion_removes_obsolete_templates(
    tmp_path: Path,
    template: str | None,
) -> None:
    with mx.stream(mx.cpu):
        first, second, output = (
            tmp_path / name for name in ("first", "second", "output")
        )
        save_source(first, 0.0)
        save_source(second, 0.5)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(first))
        tokenizer.chat_template = {"default": "old default", "tool_use": "old tool"}
        tokenizer.save_pretrained(str(first))
        convert.convert_model(str(first), output, tokenizer_id=str(first))
        note = next(output.rglob("tool_use.jinja")).parent / "notes.txt"
        note.write_text("keep this note")
        replacement = PreTrainedTokenizerFast.from_pretrained(str(second))
        replacement.chat_template = template
        replacement.save_pretrained(str(second))

        convert.convert_model(str(second), output, tokenizer_id=str(second))

        assert (
            PreTrainedTokenizerFast.from_pretrained(str(output)).chat_template
            == template
        )
        assert note.read_text() == "keep this note"
        np.testing.assert_array_equal(decoded_audio(output), 0.75)
