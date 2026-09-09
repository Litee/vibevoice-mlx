"""Generation settings fail at public entry points before expensive work."""

import sys

import numpy as np
import pytest

from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.generate import GenerationOptions, generate


class ModelAccessReached(Exception):
    pass


class UnloadedModel:
    @property
    def config(self) -> None:
        raise ModelAccessReached


INVALID_OPTIONS = [
    *[
        ("max_speech_tokens", value)
        for value in [0, -1, True, np.bool_(True), 1.0, 1.5, None, "2", np.array(2)]
    ],
    *[
        ("seed", value)
        for value in [-1, 2**32, 1.0, 1.5, "42", np.bool_(True), [42], np.array(42)]
    ],
    *[
        ("silence_threshold", value)
        for value in [-0.1, np.nan, np.inf, -np.inf, None, "0.1", [0.1], 1j]
    ],
    *[
        ("silence_min_duration_ms", value)
        for value in [0, -1, np.nan, np.inf, None, "1500", [1500]]
    ],
    *[("silence_pad_ms", value) for value in [-1, np.nan, np.inf, None, "300", [300]]],
]


@pytest.mark.parametrize("name,value", INVALID_OPTIONS)
def test_api_rejects_mutated_invalid_options_before_model_access(
    name: str, value: object
) -> None:
    options = GenerationOptions(trim_trailing_silence=True)
    setattr(options, name, value)
    with pytest.raises(ValueError, match=name):
        generate(UnloadedModel(), [], options)


@pytest.mark.parametrize(
    "seed",
    [
        0,
        42,
        2**32 - 1,
        np.int64(42),
        np.uint32(2**32 - 1),
        np.uint64(42),
        True,
        False,
        None,
    ],
)
def test_api_preserves_valid_seed_inputs(seed: object) -> None:
    with pytest.raises(ModelAccessReached):
        generate(UnloadedModel(), [], GenerationOptions(seed=seed))


@pytest.mark.parametrize(
    "tokens", [1, 200, np.int8(100), np.int64(200), np.uint32(200)]
)
def test_api_accepts_positive_integral_limits(tokens: int) -> None:
    with pytest.raises(ModelAccessReached):
        generate(UnloadedModel(), [], GenerationOptions(max_speech_tokens=tokens))


@pytest.mark.parametrize(
    "detection,trim,validates",
    [
        (False, None, False),
        (True, None, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
        (True, False, False),
    ],
)
@pytest.mark.parametrize(
    "name,value",
    [("silence_threshold", -1), ("silence_min_duration_ms", 0), ("silence_pad_ms", -1)],
)
def test_api_validates_silence_settings_only_when_trimming(
    detection: bool, trim: bool | None, validates: bool, name: str, value: object
) -> None:
    options = GenerationOptions(silence_detection=detection, trim_trailing_silence=trim)
    setattr(options, name, value)
    with pytest.raises(ValueError if validates else ModelAccessReached):
        generate(UnloadedModel(), [], options)


@pytest.mark.parametrize(
    "threshold,duration,padding",
    [(0, 1, 0), (np.float32(0.05), np.int64(1500), np.int64(300)), (0.05, 0.5, 0.5)],
)
def test_api_accepts_valid_trimming_boundaries(
    threshold: float, duration: float, padding: float
) -> None:
    with pytest.raises(ModelAccessReached):
        generate(
            UnloadedModel(),
            [],
            GenerationOptions(
                trim_trailing_silence=True,
                silence_threshold=threshold,
                silence_min_duration_ms=duration,
                silence_pad_ms=padding,
            ),
        )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("max-speech-tokens", "0"),
        ("max-speech-tokens", "-1"),
        ("max-speech-tokens", "1.5"),
        ("seed", "-1"),
        ("seed", str(2**32)),
        ("seed", "1.5"),
        ("silence-threshold", "-0.1"),
        ("silence-threshold", "nan"),
        ("silence-threshold", "inf"),
        ("silence-min-duration-ms", "0"),
        ("silence-min-duration-ms", "-1"),
        ("silence-pad-ms", "-1"),
    ],
)
def test_cli_rejects_invalid_settings_before_loading(
    flag: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid generation options must fail before loading")

    monkeypatch.setattr(e2e_pipeline, "load_model", unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--text",
            "Hello",
            "--trim-trailing-silence",
            f"--{flag}={value}",
        ],
    )
    with pytest.raises(SystemExit) as error:
        e2e_pipeline.main()
    assert error.value.code == 2
    assert flag.replace("-", "_") in capsys.readouterr().err.replace("-", "_")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--max-speech-tokens=1", "--seed=0"],
        [f"--seed={2**32 - 1}"],
        [
            "--trim-trailing-silence",
            "--silence-threshold=0",
            "--silence-min-duration-ms=1",
            "--silence-pad-ms=0",
        ],
        [
            "--silence-threshold=nan",
            "--silence-min-duration-ms=0",
            "--silence-pad-ms=-1",
        ],
        [
            "--silence-detection",
            "--no-trim-trailing-silence",
            "--silence-threshold=nan",
            "--silence-min-duration-ms=0",
            "--silence-pad-ms=-1",
        ],
    ],
)
def test_cli_accepts_valid_or_unused_settings(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reached_load(*args: object, **kwargs: object) -> None:
        raise ModelAccessReached

    monkeypatch.setattr(e2e_pipeline, "load_model", reached_load)
    monkeypatch.setattr(sys, "argv", ["vibevoice-mlx", "--text", "Hello", *arguments])
    with pytest.raises(ModelAccessReached):
        e2e_pipeline.main()
