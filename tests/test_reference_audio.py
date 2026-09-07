"""Waveform references must be usable before voice prompts are constructed."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from test_speaker_labels import RecordingTokenizer

from vibevoice_mlx.e2e_pipeline import VoiceCloneData, tokenize_text
from vibevoice_mlx.model import VibeVoiceConfig


@pytest.mark.parametrize("sample_rate", [24000, 16000])
def test_empty_reference_audio_is_rejected(tmp_path: Path, sample_rate: int) -> None:
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), sample_rate)

    with pytest.raises(ValueError, match="empty") as error:
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)],
        )

    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    ("sample_rate", "input_samples", "output_samples", "voice_tokens"),
    [
        (24000, 3200, 3200, 1),
        (24000, 3201, 3201, 2),
        (16000, 3200, 4800, 2),
        (48000, 6400, 3200, 1),
    ],
)
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("silence", [False, True])
def test_finite_reference_audio_preserves_waveform_and_voice_tokens(
    tmp_path: Path,
    sample_rate: int,
    input_samples: int,
    output_samples: int,
    voice_tokens: int,
    channels: int,
    silence: bool,
) -> None:
    path = tmp_path / "finite.wav"
    samples = np.zeros((input_samples, channels), dtype=np.float32)
    if not silence:
        samples[:, 0] = 0.25
        if channels == 2:
            samples[:, 1] = 0.75
    sf.write(path, samples, sample_rate, subtype="FLOAT")
    config = VibeVoiceConfig()

    result = tokenize_text(
        "Hello",
        "unused",
        config,
        tokenizer=RecordingTokenizer(),
        ref_audio=[str(path)],
    )

    assert isinstance(result, VoiceCloneData)
    assert len(result.speakers) == 1
    speaker = result.speakers[0]
    assert speaker.ref_audio_np.shape == (output_samples,)
    assert speaker.ref_audio_np.dtype == np.float32
    assert np.isfinite(speaker.ref_audio_np).all()
    assert speaker.num_vae_tokens == voice_tokens
    assert len(speaker.speech_embed_positions) == voice_tokens
    assert all(
        result.input_ids[index] == config.speech_diffusion_id
        for index in speaker.speech_embed_positions
    )
    expected_level = 0.0 if silence else (0.25 if channels == 1 else 0.5)
    if silence or sample_rate == 24000:
        np.testing.assert_array_equal(speaker.ref_audio_np, expected_level)
    else:
        # Polyphase filter boundaries ring; the interior preserves a constant.
        np.testing.assert_allclose(
            speaker.ref_audio_np[32:-32], expected_level, atol=1e-3
        )


@pytest.mark.parametrize(("sample_rate", "channels"), [(24000, 2), (16000, 1)])
def test_reference_audio_that_overflows_during_conversion_is_rejected(
    tmp_path: Path,
    sample_rate: int,
    channels: int,
) -> None:
    path = tmp_path / "overflow.wav"
    samples = np.full((64, channels), np.finfo(np.float32).max, dtype=np.float32)
    sf.write(path, samples, sample_rate, subtype="FLOAT")

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="non-finite") as error,
    ):
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)],
        )

    assert str(path) in str(error.value)


@pytest.mark.parametrize("sample_rate", [24000, 16000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("invalid_sample", [np.nan, np.inf, -np.inf])
def test_non_finite_reference_audio_is_rejected(
    tmp_path: Path,
    sample_rate: int,
    channels: int,
    invalid_sample: float,
) -> None:
    path = tmp_path / "non_finite.wav"
    samples = np.zeros((16, channels), dtype=np.float32)
    samples[8, -1] = invalid_sample
    sf.write(path, samples, sample_rate, subtype="FLOAT")

    with pytest.raises(ValueError, match="non-finite") as error:
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)],
        )

    assert str(path) in str(error.value)
