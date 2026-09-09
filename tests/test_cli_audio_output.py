"""Synthesis succeeds only when the CLI writes generated audio."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.generate import GenerationMetrics
from vibevoice_mlx.model import VibeVoiceConfig


def _prepare_synthesis(
    monkeypatch: pytest.MonkeyPatch, output: Path, audio: np.ndarray
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--model",
            str(output.parent),
            "--text",
            "Hello",
            "--no-semantic",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        e2e_pipeline,
        "load_model",
        lambda *args, **kwargs: (object(), VibeVoiceConfig()),
    )
    monkeypatch.setattr(e2e_pipeline, "tokenize_text", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        e2e_pipeline,
        "generate",
        lambda **kwargs: (audio, GenerationMetrics(audio_samples=len(audio))),
    )


@pytest.mark.parametrize("existing_output", [False, True])
def test_empty_audio_fails_without_changing_output(
    existing_output: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "speech.wav"
    previous_contents = b"previous successful output"
    if existing_output:
        output.write_bytes(previous_contents)
    _prepare_synthesis(monkeypatch, output, np.array([], dtype=np.float32))

    with pytest.raises(SystemExit) as error:
        e2e_pipeline.main()

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "No audio generated" in captured.err
    assert "Saved to" not in captured.out
    if existing_output:
        assert output.read_bytes() == previous_contents
    else:
        assert not output.exists()


@pytest.mark.parametrize("existing_output", [False, True])
def test_nonempty_audio_is_saved_successfully(
    existing_output: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "speech.wav"
    if existing_output:
        output.write_bytes(b"previous successful output")
    audio = np.array([0.0, 0.25, -0.5, 0.75], dtype=np.float32)
    _prepare_synthesis(monkeypatch, output, audio)

    assert e2e_pipeline.main() is None

    saved_audio, sample_rate = sf.read(output)
    np.testing.assert_array_equal(saved_audio, audio)
    assert sample_rate == 24000
    captured = capsys.readouterr()
    assert f"Saved to {output}" in captured.out
    assert captured.err == ""
