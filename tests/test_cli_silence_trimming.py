"""Silence trimming preserves later speech in the CLI's saved waveform."""

import importlib
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf
from test_generation_stop_reason import fake_model

from vibevoice_mlx import e2e_pipeline

generation = importlib.import_module("vibevoice_mlx.generate")


def synthesize(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    audio: np.ndarray,
    flags: list[str],
) -> np.ndarray:
    """Run CLI parsing, generation postprocessing, and WAV writing with fake weights."""
    model = fake_model([2, 3])
    model.vae_decoder = lambda latent: mx.array(audio)[None, None, :]
    monkeypatch.setattr(
        e2e_pipeline, "load_model", lambda *a, **kw: (model, model.config)
    )
    monkeypatch.setattr(e2e_pipeline, "tokenize_text", lambda *a, **kw: [0])
    monkeypatch.setattr(generation, "dpm_solver_2m", lambda *a, **kw: mx.zeros((1, 64)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--text",
            "First sentence. Second sentence.",
            "--no-semantic",
            "--cfg-scale",
            "1",
            "--output",
            str(output),
            *flags,
        ],
    )

    e2e_pipeline.main()

    saved, sample_rate = sf.read(output)
    assert sample_rate == 24000
    return saved


@pytest.mark.parametrize(
    "flags",
    [
        ["--trim-trailing-silence"],
        ["--silence-detection"],
        ["--silence-detection", "--trim-trailing-silence"],
    ],
)
def test_cli_trims_only_terminal_silence_after_an_internal_long_pause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str]
) -> None:
    # 1 s speech, 2 s pause, 1 s speech, 2 s terminal silence.
    audio = np.zeros(144000, dtype=np.float32)
    audio[:24000] = 0.25
    audio[72000:96000] = -0.5

    saved = synthesize(monkeypatch, tmp_path / "speech.wav", audio, flags)

    # Keep all speech and exactly 300 ms of terminal padding.
    np.testing.assert_array_equal(saved, audio[:103200])


@pytest.mark.parametrize("speech_samples", [600, 1200, 2400])
def test_cli_preserves_brief_final_speech_after_a_pause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, speech_samples: int
) -> None:
    audio = np.zeros(120000, dtype=np.float32)
    audio[:24000] = 0.25
    audio[72000 : 72000 + speech_samples] = -0.5

    saved = synthesize(
        monkeypatch, tmp_path / "speech.wav", audio, ["--trim-trailing-silence"]
    )

    # Keep the last audible 50 ms window, including a partial speech window.
    expected_end = 80400 if speech_samples <= 1200 else 81600
    np.testing.assert_array_equal(saved, audio[:expected_end])


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--no-trim-trailing-silence"],
        ["--silence-detection", "--no-trim-trailing-silence"],
    ],
)
def test_cli_keeps_the_entire_waveform_when_trimming_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str]
) -> None:
    audio = np.zeros(144000, dtype=np.float32)
    audio[:24000] = 0.25
    audio[72000:96000] = -0.5

    saved = synthesize(monkeypatch, tmp_path / "speech.wav", audio, flags)

    np.testing.assert_array_equal(saved, audio)


@pytest.mark.parametrize("silence_only", [False, True])
def test_cli_preserves_final_partial_window_and_silence_only_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_only: bool
) -> None:
    audio = np.zeros(72600, dtype=np.float32)
    if not silence_only:
        audio[:24000] = 0.25
        audio[-600:] = -0.5

    saved = synthesize(
        monkeypatch, tmp_path / "speech.wav", audio, ["--trim-trailing-silence"]
    )

    np.testing.assert_array_equal(saved, audio)


def test_cli_trimming_uses_configured_threshold_and_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = np.zeros(48000, dtype=np.float32)
    audio[:24000] = 0.25
    audio[24000:] = 0.125

    saved = synthesize(
        monkeypatch,
        tmp_path / "speech.wav",
        audio,
        [
            "--trim-trailing-silence",
            "--silence-threshold",
            "0.2",
            "--silence-pad-ms",
            "100",
        ],
    )

    np.testing.assert_array_equal(saved, audio[:26400])
