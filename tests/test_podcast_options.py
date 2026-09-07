"""Podcast CLI options reach warmup, measured generation, and the saved report."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "podcast.py"


@pytest.fixture
def run_podcast(
    tmp_path: Path,
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    # Execute the actual CLI; only model/backend work is replaced at startup.
    # GenerationOptions and input validators remain the production definitions.
    shutil.copyfile(
        Path(__file__).with_name("podcast_worker_stub.py"),
        tmp_path / "sitecustomize.py",
    )
    (tmp_path / "text.txt").write_text("Speaker 0: Hello.\nSpeaker 1: Hi.")
    for name in ("first.wav", "second.wav"):
        (tmp_path / name).write_bytes(b"reference fixture")

    def run(extra: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--model",
                "unused",
                "--text-file",
                str(tmp_path / "text.txt"),
                "--ref-audio",
                str(tmp_path / "first.wav"),
                str(tmp_path / "second.wav"),
                "--backend",
                "mlx",
                "--output",
                str(tmp_path / "audio.wav"),
                "--tokens",
                "2",
                *extra,
            ],
            cwd=SCRIPT.parents[1],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(SCRIPT.parents[1]))),
            },
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )

    return run


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], {"solver": "dpm", "diffusion_steps": 10, "cfg_scale": 1.3, "seed": 42}),
        (
            [
                "--solver",
                "sde",
                "--diffusion-steps",
                "20",
                "--cfg-scale",
                "0",
                "--seed",
                "17",
            ],
            {"solver": "sde", "diffusion_steps": 20, "cfg_scale": 0.0, "seed": 17},
        ),
    ],
)
def test_options_reach_warmup_generation_and_report(
    run_podcast: Callable[[list[str]], subprocess.CompletedProcess[str]],
    tmp_path: Path,
    extra: list[str],
    expected: dict,
) -> None:
    result = run_podcast(extra)
    assert result.returncode == 0, result.stderr
    events = [
        json.loads(line.removeprefix("PODCAST_EVENT:"))
        for line in result.stdout.splitlines()
        if line.startswith("PODCAST_EVENT:")
    ]
    calls = [event for event in events if event["name"] == "generate"]
    assert len(calls) == 2
    for call, tokens in zip(calls, (8, 2), strict=True):
        assert {
            key: call["options"][key]
            for key in (
                "solver",
                "diffusion_steps",
                "cfg_scale",
                "seed",
                "max_speech_tokens",
            )
        } == {
            **expected,
            "max_speech_tokens": tokens,
        }
        assert call["estimated_total"] == tokens
    report = json.loads((tmp_path / "audio.json").read_text())
    assert {key: report[key] for key in expected} == expected
    assert report["max_speech_tokens"] == 2
    assert report["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "extra,error",
    [
        (["--solver", "ddpm"], "invalid choice"),
        (["--diffusion-steps", "0"], "diffusion steps must"),
        (["--diffusion-steps", "1000"], "diffusion steps must"),
        (["--diffusion-steps", "1.5"], "invalid int"),
        (["--cfg-scale=nan"], "cfg_scale must"),
        (["--cfg-scale=inf"], "cfg_scale must"),
        (["--cfg-scale=-inf"], "cfg_scale must"),
    ],
)
def test_invalid_options_fail_before_model_loading(
    run_podcast: Callable[[list[str]], subprocess.CompletedProcess[str]],
    extra: list[str],
    error: str,
) -> None:
    result = run_podcast(extra)
    assert result.returncode == 2
    assert error in result.stderr
    assert "PODCAST_EVENT:" not in result.stdout


def test_report_proves_consumed_segment_transition(
    run_podcast: Callable[[list[str]], subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    result = run_podcast([])
    assert result.returncode == 0, result.stderr
    trace = json.loads((tmp_path / "audio.json").read_text())["segment_trace"]
    assert trace == {
        "consumed_diffusion_tokens": 2,
        "controls": [
            {"token_offset": 1, "speech_offset": 1, "control": "speech_end"},
            {"token_offset": 2, "speech_offset": 1, "control": "speech_start"},
        ],
        "end_start_diffusion_transitions": [
            {
                "end_token_offset": 1,
                "start_token_offset": 2,
                "diffusion_token_offset": 3,
                "speech_offset": 1,
            }
        ],
    }


@pytest.mark.parametrize(
    "selections,cap,controls",
    [
        ([3, 1], 1, []),  # Token cap: selected start is never consumed.
        (
            [3, 2, 4],
            2,
            [{"token_offset": 1, "speech_offset": 1, "control": "speech_end"}],
        ),
        ([3, 2], 2, []),  # Single-segment stop on speech_end before cap.
        (
            [1, 2, 1, 2, 3, 2, 1],
            2,
            [  # Six consumed tokens exhaust the loop.
                {"token_offset": 0, "speech_offset": 0, "control": "speech_start"},
                {"token_offset": 1, "speech_offset": 0, "control": "speech_end"},
                {"token_offset": 2, "speech_offset": 0, "control": "speech_start"},
                {"token_offset": 3, "speech_offset": 0, "control": "speech_end"},
                {"token_offset": 5, "speech_offset": 1, "control": "speech_end"},
            ],
        ),
    ],
)
def test_unconsumed_controls_do_not_prove_a_new_segment(
    run_podcast: Callable[[list[str]], subprocess.CompletedProcess[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selections: list[int],
    cap: int,
    controls: list[dict],
) -> None:
    monkeypatch.setenv("PODCAST_SELECTIONS", json.dumps(selections))
    result = run_podcast(["--tokens", str(cap)])
    if cap == 1:
        assert result.returncode == 0, result.stderr
    else:
        # Early stop/loop-limit runs write their report, then fail the existing
        # fixed-duration assertion. They still prove which controls were consumed.
        assert result.returncode != 0
        assert "expected 0.27s" in result.stderr
    trace = json.loads((tmp_path / "audio.json").read_text())["segment_trace"]
    assert trace == {
        "consumed_diffusion_tokens": 1,
        "controls": controls,
        "end_start_diffusion_transitions": [],
    }
    assert '"name": "restored", "value": true' in result.stdout


@pytest.mark.parametrize(
    "variable,error",
    [
        ("PODCAST_FAIL", "fixture generation failed"),
        ("PODCAST_METRIC_ERROR", "Consumed diffusion trace disagrees"),
    ],
)
def test_selector_is_restored_when_generation_or_trace_fails(
    run_podcast: Callable[[list[str]], subprocess.CompletedProcess[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variable: str,
    error: str,
) -> None:
    monkeypatch.setenv(variable, "1")
    result = run_podcast([])
    assert result.returncode != 0
    assert error in result.stderr
    assert '"name": "restored", "value": true' in result.stdout
    assert not (tmp_path / "audio.json").exists()
