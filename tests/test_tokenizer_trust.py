"""Custom tokenizer code requires an explicit trust decision."""

import json
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
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
        mx.save_safetensors(
            str(directory / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        yield directory
    finally:
        mx.set_default_device(previous_device)


def test_tokenization_rejects_custom_code_by_default(
    custom_tokenizer: tuple[Path, Path],
) -> None:
    directory, marker = custom_tokenizer
    with pytest.raises(ValueError, match="trust_remote_code"):
        tokenize_text("Hello", str(directory), VibeVoiceConfig())
    assert not marker.exists()


def test_tokenization_allows_explicit_custom_code_opt_in(
    custom_tokenizer: tuple[Path, Path],
) -> None:
    directory, marker = custom_tokenizer
    tokens = tokenize_text(
        "Hello", str(directory), VibeVoiceConfig(), trust_remote_code=True
    )
    assert tokens
    assert marker.read_text() == "executed"


@pytest.mark.parametrize("trust", [False, True])
def test_inference_cli_requires_explicit_trust(
    custom_tokenizer: tuple[Path, Path],
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    trust: bool,
) -> None:
    directory, marker = custom_tokenizer
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--model",
            str(tiny_checkpoint),
            "--tokenizer",
            str(directory),
            "--text",
            "Hello",
            "--no-semantic",
            "--max-speech-tokens",
            "0",
            *(["--trust-remote-code"] if trust else []),
        ],
    )
    if trust:
        e2e_pipeline.main()
        assert marker.read_text() == "executed"
    else:
        with pytest.raises(ValueError, match="trust_remote_code"):
            e2e_pipeline.main()
        assert not marker.exists()


def test_conversion_rejects_custom_code_by_default(
    custom_tokenizer: tuple[Path, Path],
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    directory, marker = custom_tokenizer
    with pytest.raises(ValueError, match="trust_remote_code"):
        convert.convert_model(
            str(tiny_checkpoint), tmp_path / "converted", str(directory)
        )
    assert not marker.exists()


def test_conversion_allows_explicit_custom_code_opt_in(
    custom_tokenizer: tuple[Path, Path],
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    directory, marker = custom_tokenizer
    output = tmp_path / "converted"
    convert.convert_model(
        str(tiny_checkpoint), output, str(directory), trust_remote_code=True
    )
    assert marker.read_text() == "executed"
    assert (output / "model.safetensors").exists()
    assert (output / "tokenizer_config.json").exists()
    assert tokenize_text(
        "Hello", str(output), VibeVoiceConfig(), trust_remote_code=True
    )


@pytest.mark.parametrize("builtin", [False, True])
@pytest.mark.parametrize("trust", [False, True])
def test_conversion_cli_requires_explicit_trust(
    custom_tokenizer: tuple[Path, Path],
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builtin: bool,
    trust: bool,
) -> None:
    directory, marker = custom_tokenizer
    output = tmp_path / "converted_output"
    if builtin:
        monkeypatch.setitem(convert.MODEL_IDS, "1.5b", str(tiny_checkpoint))
        monkeypatch.setitem(convert.TOKENIZER_IDS, "1.5b", str(directory))
        model_args = ["--models", "1.5b"]
        saved = output / "vibevoice-1.5b-mlx"
    else:
        model_args = ["--model-id", str(tiny_checkpoint), "--tokenizer", str(directory)]
        saved = output
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert.py",
            "--output-dir",
            str(output),
            *model_args,
            *(["--trust-remote-code"] if trust else []),
        ],
    )
    if trust:
        convert.main()
        assert marker.read_text() == "executed"
        assert (saved / "model.safetensors").exists()
        assert (saved / "tokenizer_config.json").exists()
    else:
        with pytest.raises(ValueError, match="trust_remote_code"):
            convert.main()
        assert not marker.exists()


def test_standard_tokenizer_loads_without_trust(tmp_path: Path) -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)
    assert tokenize_text("Hello", str(tmp_path), VibeVoiceConfig())


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
