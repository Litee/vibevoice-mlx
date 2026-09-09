"""The synthesis CLI uses local bundle tokenizers before legacy Hub defaults."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import huggingface_hub
import numpy as np
import pytest
import test_tokenizer_trust
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import convert
from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.generate import GenerationMetrics
from vibevoice_mlx.model import VibeVoiceConfig

custom_tokenizer = test_tokenizer_trust.custom_tokenizer
tiny_checkpoint = test_tokenizer_trust.tiny_checkpoint


def tokenizer_with_hello(token_id: int) -> PreTrainedTokenizerFast:
    vocab = {
        "<unk>": 0,
        **{f"unused{i}": i for i in range(1, token_id)},
        "Hello": token_id,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")


@pytest.fixture
def generated_prompts(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    prompts: list[list[int]] = []

    def generate(
        *, input_ids: list[int], **kwargs: object
    ) -> tuple[np.ndarray, GenerationMetrics]:
        prompts.append(input_ids)
        return np.zeros(3200, dtype=np.float32), GenerationMetrics(
            num_text_tokens=len(input_ids), audio_samples=3200, total_time=1.0
        )

    monkeypatch.setattr(e2e_pipeline, "generate", generate)
    return prompts


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("source", ["local", "hub"])
def test_cli_uses_bundled_tokenizer_offline_unless_overridden(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_prompts: list[list[int]],
    override: bool,
    source: str,
) -> None:
    tokenizer_with_hello(5).save_pretrained(tiny_checkpoint)
    alternate = tmp_path / "alternate"
    tokenizer_with_hello(6).save_pretrained(alternate)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: str(tiny_checkpoint),
    )

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Local synthesis must not request Hub assets")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    output = tmp_path / "speech.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--model",
            str(tiny_checkpoint) if source == "local" else "example/bundled-model",
            "--text",
            "Hello",
            "--no-semantic",
            "--output",
            str(output),
            *(["--tokenizer", str(alternate)] if override else []),
        ],
    )

    e2e_pipeline.main()

    assert len(generated_prompts) == 1
    assert (6 if override else 5) in generated_prompts[0]
    assert (5 if override else 6) not in generated_prompts[0]
    assert output.exists()


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize("route", ["api", "builtin_cli"])
def test_reconversion_preserves_source_tokenizer_unless_overridden(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: bool,
    source: str,
    route: str,
) -> None:
    tokenizer = tokenizer_with_hello(5)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<voice>"]})
    tokenizer.chat_template = {"default": "{{ messages }}", "tool_use": "tools"}
    tokenizer.save_pretrained(tiny_checkpoint)
    alternate = tmp_path / "alternate"
    tokenizer_with_hello(6).save_pretrained(alternate)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: str(tiny_checkpoint),
    )

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Reconversion must use the resolved bundle tokenizer")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    output = tmp_path / "converted"

    model_id = str(tiny_checkpoint) if source == "local" else "example/bundled-model"
    if route == "api":
        convert.convert_model(
            model_id,
            output,
            tokenizer_id=str(alternate) if override else None,
        )
    else:
        monkeypatch.setitem(convert.MODEL_IDS, "1.5b", model_id)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "convert.py",
                "--models",
                "1.5b",
                "--output-dir",
                str(output),
                *(["--tokenizer", str(alternate)] if override else []),
            ],
        )
        convert.main()
        output = output / "vibevoice-1.5b-mlx"

    restored = AutoTokenizer.from_pretrained(output, trust_remote_code=False)
    assert restored.encode("Hello") == [6 if override else 5]
    if not override:
        assert restored.encode("<voice>") == [6]
        assert restored.chat_template == {
            "default": "{{ messages }}",
            "tool_use": "tools",
        }


@pytest.mark.parametrize("vocab_size", [151936, 152064])
@pytest.mark.parametrize("route", ["inference", "conversion"])
@pytest.mark.parametrize(
    "source",
    ["remote", "empty_directory", "config_only", "tokenizer_only", "vocab_only"],
)
def test_cli_preserves_legacy_tokenizer_fallback(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_prompts: list[list[int]],
    vocab_size: int,
    source: str,
    route: str,
) -> None:
    directory = tiny_checkpoint
    config_data = json.loads((directory / "config.json").read_text())
    config_data["vocab_size"] = vocab_size
    (directory / "config.json").write_text(json.dumps(config_data))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(directory)
    )
    model_id = "example/legacy-model"
    if source != "remote":
        if source == "config_only":
            (directory / "tokenizer_config.json").write_text("{}")
        elif source == "tokenizer_only":
            (directory / "tokenizer.json").write_text("{}")
        elif source == "vocab_only":
            (directory / "tokenizer_config.json").write_text("{}")
            (directory / "vocab.json").write_text("{}")
        model_id = str(directory)
    model = SimpleNamespace(config=VibeVoiceConfig(vocab_size=vocab_size))
    monkeypatch.setattr(
        e2e_pipeline, "load_model", lambda *args, **kwargs: (model, model.config)
    )
    requests: list[tuple[str, bool]] = []

    def load_tokenizer(
        name: str, *, trust_remote_code: bool
    ) -> PreTrainedTokenizerFast:
        requests.append((name, trust_remote_code))
        return tokenizer_with_hello(5)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--model",
            model_id,
            "--text",
            "Hello",
            "--no-semantic",
            "--output",
            str(tmp_path / "speech.wav"),
        ],
    )

    if route == "inference":
        e2e_pipeline.main()
        assert 5 in generated_prompts[0]
    else:
        convert.convert_model(model_id, tmp_path / "converted")

    expected = "Qwen/Qwen2.5-1.5B" if vocab_size <= 151936 else "Qwen/Qwen2.5-7B"
    assert requests == [(expected, False)]


def test_cli_uses_bundled_qwen_vocabulary_and_merges(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_prompts: list[list[int]],
) -> None:
    (tiny_checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"})
    )
    (tiny_checkpoint / "vocab.json").write_text(
        json.dumps(
            {"H": 0, "e": 1, "l": 2, "o": 3, "He": 4, "Hel": 5, "Hell": 6, "Hello": 7}
        )
    )
    (tiny_checkpoint / "merges.txt").write_text(
        "#version: 0.2\nH e\nHe l\nHel l\nHell o\n"
    )

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Local synthesis must not request Hub assets")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--model",
            str(tiny_checkpoint),
            "--text",
            "Hello",
            "--no-semantic",
            "--output",
            str(tmp_path / "speech.wav"),
        ],
    )

    e2e_pipeline.main()

    assert 7 in generated_prompts[0]


@pytest.mark.parametrize("route", ["inference", "conversion"])
@pytest.mark.parametrize("source", ["local", "hub"])
def test_bundled_tokenizer_still_rejects_custom_code(
    tiny_checkpoint: Path,
    custom_tokenizer: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
    source: str,
) -> None:
    custom, marker = custom_tokenizer
    tokenizer_with_hello(5).save_pretrained(tiny_checkpoint)
    for name in ("tokenizer_config.json", "tokenization_fixture.py"):
        (tiny_checkpoint / name).write_bytes((custom / name).read_bytes())
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: str(tiny_checkpoint),
    )
    model_id = str(tiny_checkpoint) if source == "local" else "example/custom-model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--model",
            model_id,
            "--text",
            "Hello",
            "--no-semantic",
        ],
    )

    with pytest.raises(ValueError, match="trust_remote_code"):
        if route == "inference":
            e2e_pipeline.main()
        else:
            convert.convert_model(model_id, tmp_path / "converted")

    assert not marker.exists()
