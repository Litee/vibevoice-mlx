"""Synthesis succeeds only when the CLI writes generated audio."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf
from test_semantic_audio_context import FakeLM, generation

from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.generate import (
    GenerationMetrics,
    GenerationOptions,
    NonFiniteAudioError,
)
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


@pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("existing_output", [False, True])
def test_nonfinite_audio_fails_without_opening_output(
    sample: float,
    existing_output: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "speech.wav"
    previous_contents = b"previous successful output"
    if existing_output:
        output.write_bytes(previous_contents)
    _prepare_synthesis(monkeypatch, output, np.array([0.1, sample], dtype=np.float32))

    def reject_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid audio must be rejected before the output is opened")

    monkeypatch.setattr(sf, "write", reject_write)
    with pytest.raises(SystemExit) as error:
        e2e_pipeline.main()

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "non-finite" in captured.err
    assert "Saved to" not in captured.out
    if existing_output:
        assert output.read_bytes() == previous_contents
    else:
        assert not output.exists()


def test_finite_out_of_range_audio_preserves_existing_pcm_clipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "speech.wav"
    _prepare_synthesis(monkeypatch, output, np.array([-2.0, 2.0], dtype=np.float32))
    assert e2e_pipeline.main() is None
    saved_audio, sample_rate = sf.read(output)
    np.testing.assert_array_equal(saved_audio, [-1.0, 32767 / 32768])
    assert sample_rate == 24000


@pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "trim_flag", ["--trim-trailing-silence", "--silence-detection"]
)
@pytest.mark.parametrize("existing_output", [False, True])
def test_decoded_nonfinite_tail_fails_before_trimming(
    sample: float,
    trim_flag: str,
    existing_output: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "speech.wav"
    previous_contents = b"previous successful output"
    if existing_output:
        output.write_bytes(previous_contents)
    # 200 ms speech, 500 ms silence, then 200 ms invalid samples. Trimming
    # could hide the invalid tail beyond the default 300 ms speech padding.
    decoded = np.concatenate(
        [np.ones(4800), np.zeros(12000), np.full(4800, sample)]
    ).astype(np.float32)
    _prepare_synthesis(monkeypatch, output, decoded)
    monkeypatch.setattr(sys, "argv", [*sys.argv, trim_flag, "--cfg-scale=1"])
    config = VibeVoiceConfig(
        hidden_size=2,
        num_hidden_layers=0,
        head_dim=2,
        vocab_size=4,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
    )
    model = SimpleNamespace(
        config=config,
        _fast_lm=FakeLM([2, 3]),
        _fast_diff=None,
        acoustic_connector=lambda _: mx.zeros((1, 1, 2), dtype=mx.float16),
        vae_decoder=lambda _: mx.array(decoded).reshape(1, 1, -1),
    )
    monkeypatch.setattr(e2e_pipeline, "load_model", lambda *a, **kw: (model, config))
    monkeypatch.setattr(e2e_pipeline, "generate", generation.generate)
    monkeypatch.setattr(generation, "dpm_solver_2m", lambda *a, **kw: mx.zeros((1, 64)))

    with mx.stream(mx.cpu), pytest.raises(SystemExit) as error:
        e2e_pipeline.main()

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "non-finite" in captured.err
    assert "Saved to" not in captured.out
    if existing_output:
        assert output.read_bytes() == previous_contents
    else:
        assert not output.exists()


def test_nonfinite_generation_error_preserves_untrimmed_audio_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = np.concatenate(
        [np.ones(4800), np.zeros(12000), np.full(4800, np.nan)]
    ).astype(np.float32)
    config = VibeVoiceConfig(
        hidden_size=2,
        num_hidden_layers=0,
        head_dim=2,
        vocab_size=4,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
    )
    model = SimpleNamespace(
        config=config,
        _fast_lm=FakeLM([2, 3]),
        _fast_diff=None,
        acoustic_connector=lambda _: mx.zeros((1, 1, 2), dtype=mx.float16),
        vae_decoder=lambda _: mx.array(decoded).reshape(1, 1, -1),
    )
    monkeypatch.setattr(generation, "dpm_solver_2m", lambda *a, **kw: mx.zeros((1, 64)))

    with mx.stream(mx.cpu), pytest.raises(NonFiniteAudioError) as captured:
        generation.generate(
            model=model,
            input_ids=[0],
            opts=GenerationOptions(
                cfg_scale=1,
                max_speech_tokens=1,
                trim_trailing_silence=True,
            ),
        )

    np.testing.assert_array_equal(captured.value.audio, decoded)
    assert captured.value.metrics.audio_samples == decoded.size
    assert captured.value.metrics.num_speech_tokens == 1
    assert captured.value.metrics.stop_reason == "eos"
    assert captured.value.metrics.summary()["audio_seconds"] == decoded.size / 24000
