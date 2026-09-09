"""Benchmark workers must receive user strings as data, never Python source."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "bench_compare", Path(__file__).resolve().parents[1] / "bench_compare.py"
)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

USER_STRINGS = [
    pytest.param(
        "quotes'\" and \\slashes\\\nsecond line 雪🙂", id="special-characters"
    ),
    pytest.param(
        '" + (__import__("builtins").print("INJECTED_STRING") or "") + "',
        id="python-expression",
    ),
]


@pytest.fixture
def workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[subprocess.CompletedProcess[str]]:
    # Run the actual child source and argv, but substitute all heavyweight
    # modules at interpreter startup. No MLX, model loading, or audio I/O runs.
    shutil.copyfile(
        Path(__file__).with_name("benchmark_worker_stub.py"),
        tmp_path / "sitecustomize.py",
    )
    original_run = subprocess.run
    completed = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = original_run(
            command,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            **kwargs,
        )
        completed.append(result)
        return result

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    return completed


def events(result: subprocess.CompletedProcess[str]) -> dict[str, dict[str, Any]]:
    return {
        value["name"]: value
        for line in result.stdout.splitlines()
        if line.startswith("WORKER_EVENT:")
        for value in [json.loads(line.removeprefix("WORKER_EVENT:"))]
    }


def test_voice_filename_cannot_execute_python(
    workers: list[subprocess.CompletedProcess[str]],
) -> None:
    ref_audio = (
        'voice" + (__import__("builtins").print("INJECTED_FILENAME") or "") + ".wav'
    )
    result = benchmark.pre_encode_voice("model", ref_audio, "saved.safetensors")

    assert result == "saved.safetensors"
    child = workers[-1]
    assert child.returncode == 0, child.stderr
    assert "INJECTED_FILENAME" not in child.stdout.splitlines()
    assert events(child)["load_audio"]["path"] == ref_audio


def test_benchmark_model_cannot_execute_python(
    workers: list[subprocess.CompletedProcess[str]], tmp_path: Path
) -> None:
    model = 'model" + (__import__("builtins").print("INJECTED_MODEL") or "") + "'
    args = Namespace(
        model=model, voice_arg="voice.safetensors", text="Hello", seed=17, max_tokens=9
    )
    result = benchmark.run_config("int8", {"quantize": 8}, args, audio_dir=tmp_path)

    assert result is not None
    assert result["audio_s"] == 2.0
    assert result["peak_mem_gb"] == 1.0
    assert result["speech_tokens"] == 2
    child = workers[-1]
    assert child.returncode == 0, child.stderr
    assert "INJECTED_MODEL" not in child.stdout.splitlines()
    observed = events(child)
    assert observed["load_model"] == {
        "name": "load_model",
        "model": model,
        "quantize_bits": 8,
    }
    assert observed["semantic"]["model"] == model
    assert observed["detect_tokenizer"]["model"] == model
    assert observed["load_voice"]["path"] == "voice.safetensors"
    assert observed["generate"] == {
        "name": "generate",
        "options": {
            "solver": "dpm",
            "diffusion_steps": 10,
            "cfg_scale": 1.3,
            "max_speech_tokens": 9,
            "seed": 17,
        },
        "semantic": True,
        "semantic_reset": True,
        "voice_positions": [0, 1],
    }
    assert observed["write_audio"]["path"] == str(tmp_path / "int8.wav")


@pytest.mark.parametrize("value", USER_STRINGS)
def test_voice_encoding_strings_round_trip(
    value: str, workers: list[subprocess.CompletedProcess[str]]
) -> None:
    model, reference, save = "model " + value, value + ".wav", value + ".safetensors"
    assert benchmark.pre_encode_voice(model, reference, save) == save

    child = workers[-1]
    assert child.returncode == 0, child.stderr
    assert "INJECTED_STRING" not in child.stdout.splitlines()
    observed = events(child)
    assert observed["load_model"]["model"] == model
    assert observed["load_model"]["quantize_bits"] is None
    assert observed["load_audio"]["path"] == reference
    assert observed["save_voice"]["path"] == save
    assert observed["encode_voice"] == {
        "name": "encode_voice",
        "audio_samples": 8,
        "tokens": 2,
        "model": model,
    }


@pytest.mark.parametrize("value", USER_STRINGS)
@pytest.mark.parametrize("voice_extension", [".wav", ".safetensors"])
def test_benchmark_strings_round_trip(
    value: str,
    voice_extension: str,
    workers: list[subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    args = Namespace(
        model="model " + value,
        voice_arg=value + voice_extension,
        text=value,
        seed=42,
        max_tokens=11,
    )
    audio_dir = tmp_path / value
    assert benchmark.run_config("int4", {"quantize": 4}, args, audio_dir) is not None

    child = workers[-1]
    assert child.returncode == 0, child.stderr
    assert "INJECTED_STRING" not in child.stdout.splitlines()
    observed = events(child)
    assert observed["load_model"]["model"] == args.model
    assert observed["load_model"]["quantize_bits"] == 4
    assert observed["semantic"]["model"] == args.model
    assert observed["detect_tokenizer"]["model"] == args.model
    assert observed["tokenize"] == {
        "name": "tokenize",
        "text": value,
        "tokenizer": "test-tokenizer",
        "ref_audio": [args.voice_arg],
    }
    assert observed["write_audio"]["path"] == str(audio_dir / "int4.wav")
    if voice_extension == ".safetensors":
        assert observed["load_voice"]["path"] == args.voice_arg
        assert "encode_voice" not in observed
    else:
        assert observed["encode_voice"]["model"] == args.model
        assert "load_voice" not in observed


def test_benchmark_can_disable_semantic_voice_and_audio_output(
    workers: list[subprocess.CompletedProcess[str]],
) -> None:
    args = Namespace(model="model", voice_arg=None, text="Hello", seed=0, max_tokens=1)
    assert (
        benchmark.run_config(
            "fp16", {"no_semantic": True, "coreml_semantic": True}, args
        )
        is not None
    )

    child = workers[-1]
    assert child.returncode == 0, child.stderr
    observed = events(child)
    assert observed["load_model"]["quantize_bits"] is None
    assert observed["tokenize"]["ref_audio"] is None
    assert observed["generate"]["semantic"] is False
    assert observed["generate"]["semantic_reset"] is False
    assert observed["generate"]["voice_positions"] == []
    assert (
        not {"semantic", "coreml_fallback", "load_voice", "encode_voice", "write_audio"}
        & observed.keys()
    )


def test_unavailable_coreml_is_not_benchmarked_as_mlx(
    workers: list[subprocess.CompletedProcess[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    args = Namespace(model="model", voice_arg=None, text="Hello", seed=0, max_tokens=1)
    result = benchmark.run_config(
        "fp16, coreml-semantic", {"coreml_semantic": True}, args
    )

    assert result is None
    assert "Requested coreml semantic backend is unavailable" in capsys.readouterr().out
    assert "generate" not in events(workers[-1])


def test_packed_checkpoint_is_not_benchmarked_as_fp16(
    workers: list[subprocess.CompletedProcess[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("BENCH_TEST_SOURCE_BITS", "4")
    args = Namespace(model="model", voice_arg=None, text="Hello", seed=0, max_tokens=1)
    assert benchmark.run_config("fp16", {}, args) is None
    assert "Requested fp16 quantization, but loaded int4" in capsys.readouterr().out
    assert "generate" not in events(workers[-1])


def test_successful_result_reports_requested_and_effective_configuration(
    workers: list[subprocess.CompletedProcess[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_TEST_SOURCE_BITS", "4")
    args = Namespace(
        model="model", voice_arg="voice.safetensors", text="Hello", seed=0, max_tokens=1
    )
    result = benchmark.run_config("int4", {"quantize": 4}, args)
    assert result is not None
    assert result["requested"] == {
        "quantization": "int4",
        "semantic_backend": "mlx",
        "voice_reference": "cached",
    }
    assert result["effective"] == {
        "quantization": "int4",
        "semantic_backend": "mlx",
        "voice_reference": "cached",
    }


def test_cli_stops_when_requested_voice_encoding_fails(
    workers: list[subprocess.CompletedProcess[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCH_TEST_VOICE_FAILURE", "1")
    child = benchmark.subprocess.run(
        [
            sys.executable,
            str(Path(benchmark.__file__).resolve()),
            "--model",
            "model",
            "--ref-audio",
            "voice.wav",
            "--save-audio",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert "Cannot benchmark the requested voice reference" in child.stderr
    assert "generate" not in events(child)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ("BENCH_TEST_MISSING_MLX", "Requested mlx semantic backend is unavailable"),
        (
            "BENCH_TEST_MISSING_VOICE",
            "Requested voice reference did not produce voice embeddings",
        ),
    ],
)
def test_missing_requested_conditioning_stops_before_generation(
    env: str,
    message: str,
    workers: list[subprocess.CompletedProcess[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(env, "1")
    args = Namespace(
        model="model", voice_arg="voice.wav", text="Hello", seed=0, max_tokens=1
    )
    assert benchmark.run_config("fp16", {}, args) is None
    assert message in capsys.readouterr().out
    assert "generate" not in events(workers[-1])


def test_coreml_and_unconditioned_metadata_are_reported(
    workers: list[subprocess.CompletedProcess[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_TEST_COREML_AVAILABLE", "1")
    args = Namespace(model="model", voice_arg=None, text="Hello", seed=0, max_tokens=1)
    result = benchmark.run_config(
        "fp16, coreml-semantic", {"coreml_semantic": True}, args
    )
    expected = {
        "quantization": "fp16",
        "semantic_backend": "coreml",
        "voice_reference": "none",
    }
    assert result is not None
    assert result["requested"] == expected
    assert result["effective"] == expected


def test_cli_persists_configuration_metadata(
    workers: list[subprocess.CompletedProcess[str]], tmp_path: Path
) -> None:
    child = benchmark.subprocess.run(
        [
            sys.executable,
            str(Path(benchmark.__file__).resolve()),
            "--model",
            "model",
            "--save-audio",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    rows = json.loads((tmp_path / "1.5b" / "results.json").read_text())
    assert len(rows) == 6  # Three unavailable CoreML configurations are excluded.
    assert rows[0][1]["effective"] == {
        "quantization": "fp16",
        "semantic_backend": "mlx",
        "voice_reference": "none",
    }
    assert rows[1][1]["effective"]["semantic_backend"] == "none"
    assert all(result["requested"] == result["effective"] for _, result in rows)
