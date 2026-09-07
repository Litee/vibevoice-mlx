"""Voice-cloning prompts preserve speaker identities and dialogue text."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vibevoice_mlx.e2e_pipeline import VoiceCloneData, tokenize_text
from vibevoice_mlx.model import VibeVoiceConfig


@pytest.mark.parametrize("single_segment", [False, True])
@pytest.mark.parametrize("file_references", [False, True])
@pytest.mark.parametrize(
    ("script", "missing"),
    [
        ("Speaker 0: Hello\nSpeaker 1: Hi\nSpeaker 2: Goodbye", 2),
        ("Speaker 1: Hello\nSpeaker 2: Hi\nSpeaker 3: Goodbye", 3),
        ("Speaker 0: Hello\nSpeaker 2: Goodbye", 2),
    ],
)
def test_speaker_without_reference_is_rejected(
    script: str,
    missing: int,
    single_segment: bool,
    file_references: bool,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "voice.wav"
    sf.write(audio_path, np.zeros(3200), 24000)
    with pytest.raises(ValueError, match=rf"Speaker {missing}.*2 reference voices"):
        tokenize_text(
            script,
            "unused",
            VibeVoiceConfig(single_segment=single_segment, hidden_size=3),
            tokenizer=RecordingTokenizer(),
            ref_audio=[str(audio_path)] * 2 if file_references else None,
            speaker_embeds=None if file_references else [(1, np.zeros((1, 3)))] * 2,
        )


class RecordingTokenizer:
    def __init__(self) -> None:
        self.text: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.text.append(text)
        return [ord(character) for character in text]


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("Speaker 0: Hello\nSpeaker 1: Hi", " Speaker 0: Hello\n Speaker 1: Hi\n"),
        ("Speaker 1: Hi\nSpeaker 0: Hello", " Speaker 1: Hi\n Speaker 0: Hello\n"),
        ("Speaker 1: Hello\nSpeaker 2: Hi", " Speaker 0: Hello\n Speaker 1: Hi\n"),
        ("Speaker 2: Hello", " Speaker 1: Hello\n"),
        (
            "Speaker 1: Ask Speaker 2: are you ready?\nSpeaker 2: Yes",
            " Speaker 0: Ask Speaker 2: are you ready?\n Speaker 1: Yes\n",
        ),
        (
            "Speaker 1: Mention Speaker 0\nSpeaker 2: Hi",
            " Speaker 0: Mention Speaker 0\n Speaker 1: Hi\n",
        ),
        (
            "  Speaker 1 : Hello  \n\tSpeaker 2:\tHi",
            " Speaker 0 : Hello\n Speaker 1:\tHi\n",
        ),
    ],
)
@pytest.mark.parametrize("single_segment", [False, True])
def test_voice_clone_speaker_labels(
    script: str, expected: str, single_segment: bool
) -> None:
    tokenizer = RecordingTokenizer()
    config = VibeVoiceConfig(single_segment=single_segment, hidden_size=3)
    embeddings = [np.zeros((2, 3)), np.ones((1, 3))]
    result = tokenize_text(
        script,
        "unused",
        config,
        tokenizer=tokenizer,
        speaker_embeds=[(len(embedding), embedding) for embedding in embeddings],
    )

    assert isinstance(result, VoiceCloneData)
    text_start = tokenizer.text.index(" Text input:\n") + 1
    assert "".join(tokenizer.text[text_start:-1]) == expected
    assert [speaker.speaker_id for speaker in result.speakers] == [0, 1]
    for speaker, embedding in zip(result.speakers, embeddings, strict=True):
        assert speaker.cached_embeds is embedding
        assert len(speaker.speech_embed_positions) == len(embedding)
        assert all(
            result.input_ids[position] == config.speech_diffusion_id
            for position in speaker.speech_embed_positions
        )


@pytest.mark.parametrize("single_segment", [False, True])
@pytest.mark.parametrize("with_voice", [False, True])
def test_unlabeled_text_keeps_existing_prefix_behavior(
    single_segment: bool, with_voice: bool
) -> None:
    tokenizer = RecordingTokenizer()
    tokenize_text(
        "Ask Speaker 2 to begin.",
        "unused",
        VibeVoiceConfig(single_segment=single_segment, hidden_size=3),
        tokenizer=tokenizer,
        speaker_embeds=[(1, np.zeros((1, 3)))] if with_voice else None,
    )
    expected = (
        " Speaker 0: Ask Speaker 2 to begin.\n"
        if single_segment
        else (" Ask Speaker 2 to begin.\n")
    )
    assert tokenizer.text[-2] == expected


@pytest.mark.parametrize("single_segment", [False, True])
@pytest.mark.parametrize("with_voice", [False, True])
def test_leading_speaker_mention_without_colon_is_dialogue(
    single_segment: bool,
    with_voice: bool,
) -> None:
    tokenizer = RecordingTokenizer()
    tokenize_text(
        "Speaker 2 should begin.",
        "unused",
        VibeVoiceConfig(single_segment=single_segment, hidden_size=3),
        tokenizer=tokenizer,
        speaker_embeds=[(1, np.zeros((1, 3)))] if with_voice else None,
    )
    prefix = " Speaker 0: " if single_segment else " "
    assert tokenizer.text[-2] == f"{prefix}Speaker 2 should begin.\n"


@pytest.mark.parametrize("single_segment", [False, True])
@pytest.mark.parametrize("colon_spacing", ["", "  "])
def test_labels_without_voice_references_keep_existing_behavior(
    single_segment: bool, colon_spacing: str
) -> None:
    tokenizer = RecordingTokenizer()
    lines = [f"Speaker 1{colon_spacing}: Hello", f"Speaker 2{colon_spacing}: Hi"]
    result = tokenize_text(
        "\n".join(lines),
        "unused",
        VibeVoiceConfig(single_segment=single_segment),
        tokenizer=tokenizer,
    )
    assert isinstance(result, list)
    assert tokenizer.text[-3:-1] == [f" {line}\n" for line in lines]
