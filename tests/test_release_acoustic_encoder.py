"""One-shot setup releases acoustic weights only after preparing every voice."""

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from benchmarks import podcast
from vibevoice_mlx import e2e_pipeline as pipeline


class SetupComplete(Exception):
    """Stop at the generation boundary without running inference."""


@pytest.mark.parametrize("source", ["raw", "cached", "mixed", "none"])
@pytest.mark.parametrize("semantic", [True, False])
def test_cli_releases_encoder_after_all_voices_before_next_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str, semantic: bool
) -> None:
    model = SimpleNamespace(_encoder_weights=object())
    weights = model._encoder_weights
    encoded: list[int] = []
    speakers = [
        pipeline.SpeakerRef(
            index,
            np.array([index], dtype=np.float32),
            1,
            [index],
            np.full((1, 2), index, dtype=np.float16)
            if source == "cached" or (source == "mixed" and index == 0)
            else None,
        )
        for index in range(2)
    ]
    prompt = [0] if source == "none" else pipeline.VoiceCloneData([0, 1], speakers)
    args = ["vibevoice-mlx", "--model", str(tmp_path), "--text", "Hello"]
    if not semantic:
        args.append("--no-semantic")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(pipeline, "load_model", lambda *a, **kw: (model, object()))
    monkeypatch.setattr(pipeline, "detect_tokenizer", lambda *a: "unused")
    monkeypatch.setattr(pipeline, "tokenize_text", lambda *a, **kw: prompt)

    def encode(wav: np.ndarray, *args: Any) -> np.ndarray:
        assert model._encoder_weights is weights
        encoded.append(int(wav[0]))
        return np.full((1, 2), wav[0], dtype=np.float16)

    def next_stage(*args: Any, **kwargs: Any) -> None:
        assert model._encoder_weights is None
        assert encoded == {"raw": [0, 1], "mixed": [1]}.get(source, [])
        if not semantic:
            if source == "none":
                assert kwargs["voice_embeds"] is None
            else:
                for index in range(2):
                    np.testing.assert_array_equal(
                        kwargs["voice_embeds"][index],
                        np.full((1, 2), index, dtype=np.float16),
                    )
        raise SetupComplete

    monkeypatch.setattr(pipeline, "encode_voice_reference", encode)
    monkeypatch.setattr(pipeline, "_load_semantic_encoder", next_stage)
    monkeypatch.setattr(pipeline, "generate", next_stage)
    with pytest.raises(SetupComplete):
        pipeline.main()


@pytest.mark.parametrize("backend", ["mlx", "coreml", "ane"])
def test_podcast_releases_encoder_after_both_voices_before_semantic_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> None:
    text = tmp_path / "text.txt"
    text.write_text("Speaker 0: Hello.\nSpeaker 1: Hi.")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "podcast.py",
            "--model",
            str(tmp_path),
            "--text-file",
            str(text),
            "--ref-audio",
            "first.wav",
            "second.wav",
            "--backend",
            backend,
            "--output",
            str(tmp_path / "audio.wav"),
        ],
    )
    model = SimpleNamespace(_encoder_weights=object())
    weights = model._encoder_weights
    encoded: list[int] = []
    speakers = [
        pipeline.SpeakerRef(index, np.array([index]), 1, [index]) for index in range(2)
    ]
    config = SimpleNamespace(vocab_size=152064)
    monkeypatch.setattr(podcast, "load_model", lambda *a, **kw: (model, config))
    monkeypatch.setattr(
        podcast,
        "tokenize_text",
        lambda *a, **kw: pipeline.VoiceCloneData([0, 1], speakers),
    )

    def encode(wav: np.ndarray, *args: Any) -> np.ndarray:
        assert model._encoder_weights is weights
        encoded.append(int(wav[0]))
        return np.ones((1, 2), dtype=np.float16)

    def semantic_setup(*args: Any, **kwargs: Any) -> None:
        assert encoded == [0, 1]
        assert model._encoder_weights is None
        raise SetupComplete

    monkeypatch.setattr(podcast, "encode_voice_reference", encode)
    monkeypatch.setattr(podcast, "_try_mlx_semantic", semantic_setup)
    monkeypatch.setattr(podcast, "_try_coreml_semantic", semantic_setup)
    with pytest.raises(SetupComplete):
        podcast.main()


def test_release_preserves_generation_components_and_library_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = {
        name: object()
        for name in ("acoustic_connector", "semantic_connector", "vae_decoder")
    }
    model = SimpleNamespace(_encoder_weights=object(), **retained)
    cache = {"other-model": object()}
    monkeypatch.setattr(
        pipeline.encode_voice_reference, "_enc_cache", cache, raising=False
    )
    cleared: list[bool] = []

    def clear_cache() -> None:
        assert model._encoder_weights is None
        cleared.append(True)

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    pipeline._release_acoustic_encoder(model)
    pipeline._release_acoustic_encoder(model)
    pipeline._release_acoustic_encoder(object())

    assert cleared == [True]
    for name, component in retained.items():
        assert getattr(model, name) is component
    assert pipeline.encode_voice_reference._enc_cache is cache


def test_release_drops_materialized_mlx_weights_before_clearing_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Module()
    model._encoder_weights = {"stem_w": mx.ones((2, 1, 7))}
    mx.eval(model._encoder_weights)
    reference = weakref.ref(model._encoder_weights["stem_w"])

    def clear_cache() -> None:
        assert reference() is None

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    pipeline._release_acoustic_encoder(model)
    assert reference() is None
