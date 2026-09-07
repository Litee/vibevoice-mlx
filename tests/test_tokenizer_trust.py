"""Tokenizer loading never executes repository-supplied Python code."""

import json
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, dynamic_module_utils

import convert
from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.e2e_pipeline import tokenize_text
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


@pytest.fixture
def custom_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    directory = tmp_path / "custom_tokenizer"
    directory.mkdir()
    marker = tmp_path / "custom_code_executed"
    monkeypatch.setattr(
        dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules")
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "FixtureTokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_fixture.FixtureTokenizer", None]
                },
            }
        )
    )
    (directory / "tokenization_fixture.py").write_text(f"""from pathlib import Path
from transformers import PreTrainedTokenizer

Path({str(marker)!r}).write_text("executed")

class FixtureTokenizer(PreTrainedTokenizer):
    vocab_files_names = {{}}

    def __init__(self, **kwargs):
        kwargs.setdefault("unk_token", "<unk>")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return 1

    def get_vocab(self):
        return {{"<unk>": 0}}

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return 0

    def _convert_id_to_token(self, index):
        return "<unk>"

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()
""")
    return directory, marker


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Iterator[Path]:
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        directory = tmp_path / "tiny_checkpoint"
        directory.mkdir()
        config = VibeVoiceConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            intermediate_size=8,
            vocab_size=8,
            diffusion_layers=0,
            speech_start_id=0,
            speech_end_id=1,
            speech_diffusion_id=2,
            eos_id=3,
        )
        model = VibeVoiceModel(config)
        (directory / "config.json").write_text(json.dumps(asdict(config)))
        weights = dict(tree_flatten(model.parameters()))
        weights.update(tiny_vae_weights(config.vae_dim))
        mx.save_safetensors(str(directory / "model.safetensors"), weights)
        yield directory
    finally:
        mx.set_default_device(previous_device)


@pytest.mark.parametrize(
    "route", ["api", "conversion_api", "inference", "conversion", "builtin"]
)
def test_custom_tokenizer_code_is_rejected(
    custom_tokenizer: tuple[Path, Path],
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    directory, marker = custom_tokenizer
    with pytest.raises(ValueError, match="trust_remote_code"):
        _run_route(route, directory, tiny_checkpoint, tmp_path, monkeypatch)
    assert not marker.exists()


def _run_route(
    route: str,
    tokenizer_path: Path,
    checkpoint: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if route == "api":
        assert tokenize_text("Hello", str(tokenizer_path), VibeVoiceConfig())
    elif route == "conversion_api":
        saved = output / "converted_output"
        convert.convert_model(str(checkpoint), saved, str(tokenizer_path))
        assert tokenize_text("Hello", str(saved), VibeVoiceConfig())
    elif route == "inference":
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vibevoice-mlx",
                "--model",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer_path),
                "--text",
                "Hello",
                "--no-semantic",
                "--max-speech-tokens",
                "0",
            ],
        )
        e2e_pipeline.main()
    else:
        saved = output / "converted_output"
        if route == "builtin":
            monkeypatch.setitem(convert.MODEL_IDS, "1.5b", str(checkpoint))
            monkeypatch.setitem(convert.TOKENIZER_IDS, "1.5b", str(tokenizer_path))
            model_args = ["--models", "1.5b"]
            saved = saved / "vibevoice-1.5b-mlx"
        else:
            model_args = [
                "--model-id",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer_path),
            ]
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "convert.py",
                "--output-dir",
                str(output / "converted_output"),
                *model_args,
            ],
        )
        convert.main()
        assert (saved / "model.safetensors").exists()
        assert tokenize_text("Hello", str(saved), VibeVoiceConfig())


@pytest.mark.parametrize(
    "route", ["api", "conversion_api", "inference", "conversion", "builtin"]
)
def test_standard_tokenizer_loads_without_trust(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    directory = tmp_path / "standard_tokenizer"
    tokenizer.save_pretrained(directory)
    _run_route(route, directory, tiny_checkpoint, tmp_path, monkeypatch)


def test_passed_tokenizer_does_not_load_custom_code(
    custom_tokenizer: tuple[Path, Path],
) -> None:
    directory, marker = custom_tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    assert tokenize_text(
        "Hello", str(directory), VibeVoiceConfig(), tokenizer=tokenizer
    )
    assert not marker.exists()
