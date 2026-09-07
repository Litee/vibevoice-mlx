"""Pre-encoded voice inputs must match the selected model and token count."""

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_speaker_labels import RecordingTokenizer

from vibevoice_mlx.e2e_pipeline import VoiceCloneData, tokenize_text
from vibevoice_mlx.model import VibeVoiceConfig


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("shape", [(), (3,), (1, 2, 3), (0, 3), (2, 0), (2, 4)])
def test_invalid_voice_embedding_shape_is_rejected(
    tmp_path: Path,
    saved: bool,
    shape: tuple[int, ...],
) -> None:
    embeddings = np.zeros(shape, dtype=np.float32)
    path = tmp_path / "invalid.safetensors"
    if saved:
        save_file({"embeddings": embeddings}, path)

    with pytest.raises(ValueError, match="embeddings") as error:
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(hidden_size=3),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)] if saved else None,
            speaker_embeds=None if saved else [(2, embeddings)],
        )

    assert str(path) in str(error.value) if saved else "Speaker 0" in str(error.value)


@pytest.mark.parametrize(
    "num_tokens", [0, -1, 1, 3, 2.0, True, np.bool_(True), None, "2"]
)
def test_supplied_voice_token_count_must_match_embedding_rows(
    num_tokens: object,
) -> None:
    with pytest.raises(ValueError, match="Speaker 0.*num_tokens"):
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(hidden_size=3),
            tokenizer=RecordingTokenizer(),
            speaker_embeds=[(num_tokens, np.zeros((2, 3), dtype=np.float32))],
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        None,
        [[0, 0, 0]],
        np.zeros((1, 3), dtype=bool),
        np.zeros((1, 3), dtype=complex),
        np.zeros((1, 3), dtype=object),
        np.zeros((1, 3), dtype="U1"),
    ],
)
def test_supplied_voice_requires_real_numeric_array(embeddings: object) -> None:
    with pytest.raises(ValueError, match="Speaker 0.*real numeric NumPy array"):
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(hidden_size=3),
            tokenizer=RecordingTokenizer(),
            speaker_embeds=[(1, embeddings)],
        )


def test_saved_boolean_embeddings_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "boolean.safetensors"
    save_file({"embeddings": np.zeros((1, 3), dtype=bool)}, path)
    with pytest.raises(ValueError, match="real numeric NumPy array") as error:
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(hidden_size=3),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)],
        )
    assert str(path) in str(error.value)


def test_saved_voice_requires_embeddings_tensor(tmp_path: Path) -> None:
    path = tmp_path / "missing.safetensors"
    save_file({"unrelated": np.zeros((1, 3), dtype=np.float32)}, path)
    with pytest.raises(ValueError, match="embeddings") as error:
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(hidden_size=3),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)],
        )
    assert str(path) in str(error.value)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int32, np.uint32]
)
def test_valid_voice_embeddings_preserve_values_dtype_and_position_order(
    tmp_path: Path,
    saved: bool,
    dtype: type,
) -> None:
    embeddings = [
        np.arange(6, dtype=dtype).reshape(2, 3),
        np.zeros((1, 3), dtype=dtype),
    ]
    paths = [tmp_path / f"speaker{i}.safetensors" for i in range(2)]
    for path, embedding in zip(paths, embeddings, strict=True):
        save_file({"embeddings": embedding}, path)
    config = VibeVoiceConfig(hidden_size=3)

    result = tokenize_text(
        "Speaker 0: Hello\nSpeaker 1: Hi",
        "unused",
        config,
        tokenizer=RecordingTokenizer(),
        # A valid supplied list must take precedence over even an invalid path.
        ref_audio=[str(path) for path in paths]
        if saved
        else [str(tmp_path / "missing.wav")],
        speaker_embeds=None
        if saved
        else [(np.int64(2), embeddings[0]), (1, embeddings[1])],
    )

    assert isinstance(result, VoiceCloneData)
    assert [speaker.speaker_id for speaker in result.speakers] == [0, 1]
    assert [speaker.num_vae_tokens for speaker in result.speakers] == [2, 1]
    previous_position = -1
    for speaker, embedding in zip(result.speakers, embeddings, strict=True):
        np.testing.assert_array_equal(speaker.cached_embeds, embedding)
        assert speaker.cached_embeds.dtype == embedding.dtype
        if not saved:
            assert speaker.cached_embeds is embedding
        assert speaker.ref_audio_np.size == 0
        assert len(speaker.speech_embed_positions) == speaker.num_vae_tokens
        assert speaker.speech_embed_positions[0] > previous_position
        for position in speaker.speech_embed_positions:
            assert result.input_ids[position] == config.speech_diffusion_id
        previous_position = speaker.speech_embed_positions[-1]


def test_empty_supplied_list_preserves_saved_voice_fallback(tmp_path: Path) -> None:
    path = tmp_path / "voice.safetensors"
    embeddings = np.zeros((1, 3), dtype=np.float32)
    save_file({"embeddings": embeddings}, path)

    result = tokenize_text(
        "Hello",
        "unused",
        VibeVoiceConfig(hidden_size=3),
        tokenizer=RecordingTokenizer(),
        speaker_embeds=[],
        ref_audio=[str(path)],
    )

    assert isinstance(result, VoiceCloneData)
    np.testing.assert_array_equal(result.speakers[0].cached_embeds, embeddings)


@pytest.mark.parametrize("count_type", [np.int8, np.uint8])
def test_narrow_numpy_counts_support_large_prompt_offsets(count_type: type) -> None:
    embeddings = np.zeros((2, 3), dtype=np.float32)
    # Each reference contributes another speaker prefix, taking the final
    # embedding positions beyond both int8 and uint8 limits.
    voices = [(count_type(2), embeddings)] * 10
    config = VibeVoiceConfig(hidden_size=3)

    result = tokenize_text(
        "Hello",
        "unused",
        config,
        tokenizer=RecordingTokenizer(),
        speaker_embeds=voices,
    )

    assert isinstance(result, VoiceCloneData)
    assert result.speakers[-1].speech_embed_positions[-1] > 255
    for speaker in result.speakers:
        assert speaker.cached_embeds is embeddings
        assert speaker.num_vae_tokens == 2
        assert len(speaker.speech_embed_positions) == 2
        assert all(
            result.input_ids[position] == config.speech_diffusion_id
            for position in speaker.speech_embed_positions
        )


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
def test_non_finite_voice_embeddings_are_rejected(
    tmp_path: Path,
    saved: bool,
    sample: float,
) -> None:
    embeddings = np.zeros((2, 3), dtype=np.float32)
    embeddings[1, 2] = sample
    path = tmp_path / "non_finite.safetensors"
    if saved:
        save_file({"embeddings": embeddings}, path)

    with pytest.raises(ValueError, match="non-finite") as error:
        tokenize_text(
            "Hello",
            "unused",
            VibeVoiceConfig(hidden_size=3),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(path)] if saved else None,
            speaker_embeds=None if saved else [(2, embeddings)],
        )

    assert str(path) in str(error.value) if saved else "Speaker 0" in str(error.value)
