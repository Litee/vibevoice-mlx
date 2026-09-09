"""Exercise the paired CLI with real, lightweight child interpreters."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict, Path, list]:
    spec = importlib.util.spec_from_file_location(
        "bench_paired", Path(__file__).resolve().parents[1] / "bench_paired.py"
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for side in ("A", "B"):
        checkout = tmp_path / side
        checkout.mkdir()
        shutil.copyfile(
            Path(__file__).parents[1] / "bench_compare.py",
            checkout / "bench_compare.py",
        )
    model = tmp_path / "snapshots" / ("a" * 40)
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    shutil.copyfile(
        Path(__file__).with_name("benchmark_worker_stub.py"), stubs / "sitecustomize.py"
    )
    original_run = subprocess.run
    launches = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/vm_stat":
            launches.append("memory")
            return subprocess.CompletedProcess(
                command,
                0,
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 262144.\nPages inactive: 262144.\nPages speculative: 262144.\n",
                "",
            )
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "b" * 40 + "\n", "")
        launches.append((Path(kwargs["cwd"]).name, json.loads(command[-1])["seed"]))
        return original_run(
            command,
            env={
                **os.environ,
                "PYTHONPATH": str(stubs),
                "PAIRED_SECRET": "do-not-publish-me",
            },
            **kwargs,
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(tmp_path))
    plan = {
        "A": str(tmp_path / "A"),
        "B": str(tmp_path / "B"),
        "model": str(model),
        "text": "Private prompt",
        "seeds": [11, 22],
        "repeats": 2,
    }
    return runner, plan, tmp_path / "output", launches


def test_cli_alternates_pairs_and_checks_memory_before_each_serial_worker(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, launches = paired
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 0
    assert launches == [
        "memory",
        ("A", 11),
        "memory",
        ("B", 11),
        "memory",
        ("B", 22),
        "memory",
        ("A", 22),
        "memory",
        ("A", 11),
        "memory",
        ("B", 11),
        "memory",
        ("B", 22),
        "memory",
        ("A", 22),
    ]
    manifest = json.loads((output / "trials.json").read_text())
    assert [trial["status"] for trial in manifest["trials"]] == ["ok"] * 8
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete_pairs"] == 4
    assert summary["successful_trials"] == 8
    saved = "".join(path.read_text() for path in output.iterdir())
    assert str(tmp_path) not in saved
    assert "Private prompt" not in saved
    assert "do-not-publish-me" not in saved


def test_trial_records_runtime_import_model_and_memory_provenance(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[5], repeats=1)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 0
    manifest = json.loads((output / "trials.json").read_text())
    first = manifest["trials"][0]
    assert first["source"]["commit"] == "b" * 40
    provenance = first["provenance"]
    assert provenance["python"]["version"]
    assert len(provenance["python"]["executable_sha256"]) == 64
    assert "mlx" in provenance["libraries"]
    assert provenance["modules"]["vibevoice_mlx.generate"]["origin"] == "unavailable"
    assert len(provenance["benchmark_source_sha256"]) == 64
    assert provenance["model"]["snapshot_revision"] == "a" * 40
    assert (
        provenance["model"]["config_sha256"]
        == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )
    assert first["metrics"]["mlx_peak_generation_bytes"] == 1_000_000_000
    assert first["metrics"]["process_lifetime_peak_rss_bytes"] > 0
    summary = json.loads((output / "summary.json").read_text())
    assert (
        "process lifetime"
        in summary["memory_labels"]["process_lifetime_peak_rss_bytes"]
    )


def test_failed_worker_keeps_provenance_and_cannot_contribute_a_pair(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=1)
    worker = Path(plan["A"]) / "bench_compare.py"
    with worker.open("a") as target:
        target.write(
            '\n_original_script = make_script\ndef make_script():\n    return _original_script() + "\\nprint(\\"do-not-publish-me\\")\\nraise SystemExit(7)\\n"\n'
        )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    trials = json.loads((output / "trials.json").read_text())["trials"]
    assert [trial["status"] for trial in trials] == ["worker_failed", "ok"]
    assert trials[0]["returncode"] == 7
    assert trials[0]["provenance"]["python"]["version"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete_pairs"] == 0
    assert summary["median_paired_B_over_A_gen_s"] is None
    assert summary["status_counts"] == {"ok": 1, "worker_failed": 1}
    assert "do-not-publish-me" not in (output / "trials.json").read_text()


def test_manifest_identifies_inputs_without_saving_paths_or_prompt(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, _ = paired
    voice = tmp_path / "private-speaker.safetensors"
    voice.write_bytes(b"voice data")
    plan.update(
        seeds=[1],
        repeats=1,
        voice_arg=str(voice),
        model_revision="c" * 40,
        quantize=8,
        sem_mode="none",
        max_tokens=12,
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 0
    manifest = json.loads((output / "trials.json").read_text())
    assert manifest["requested"] == {
        "quantize": 8,
        "sem_mode": "none",
        "max_tokens": 12,
        "timeout_seconds": 600,
    }
    assert len(manifest["inputs"]["text_sha256"]) == 64
    assert len(manifest["inputs"]["voice_sha256"]) == 64
    assert len(manifest["inputs"]["model_file_metadata_sha256"]) == 64
    assert manifest["inputs"]["declared_model_revision"] == "c" * 40
    serialized = (output / "trials.json").read_text()
    assert "private-speaker" not in serialized
    assert "Private prompt" not in serialized


@pytest.mark.parametrize(
    "memory_text",
    [
        "",
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 262143.\nPages inactive: 262144.\nPages speculative: 262144.\n",
    ],
)
def test_low_or_unknown_memory_prevents_every_worker(
    paired: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memory_text: str
) -> None:
    runner, plan, output, launches = paired
    original = runner.subprocess.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/vm_stat":
            return subprocess.CompletedProcess(command, 0, memory_text, "")
        return original(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", run)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    assert launches == []
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful_trials"] == 0
    assert summary["complete_pairs"] == 0


def test_held_generation_lock_prevents_launch(paired: tuple, tmp_path: Path) -> None:
    runner, plan, output, launches = paired
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    with (tmp_path / "vibevoice-generation.lock").open("w") as lock:
        runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        assert runner.main(["--plan", str(source), "--output", str(output)]) == 2
    assert launches == []
    assert not output.exists()


@pytest.mark.parametrize(
    "result",
    [
        {"gen_s": 1, "audio_s": 0, "rtf": 0, "speech_tokens": 0, "peak_mem_gb": 1},
        {"gen_s": 1, "audio_s": 2, "rtf": 2, "speech_tokens": 1.5, "peak_mem_gb": 1},
        {
            "gen_s": float("nan"),
            "audio_s": 2,
            "rtf": 2,
            "speech_tokens": 2,
            "peak_mem_gb": 1,
        },
    ],
)
def test_empty_or_malformed_generation_is_a_failed_trial(
    paired: tuple, tmp_path: Path, result: dict
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=1)
    code = "print(" + repr("BENCH_RESULT:" + json.dumps(result)) + ")"
    (Path(plan["A"]) / "bench_compare.py").write_text(
        "def make_script():\n    return " + repr(code)
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    trials = json.loads((output / "trials.json").read_text())["trials"]
    assert trials[0]["status"] == "invalid_result"
    assert trials[1]["status"] == "ok"


def test_timeout_finishes_before_the_next_worker_starts(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, launches = paired
    plan.update(seeds=[1], repeats=1, timeout=1)
    (Path(plan["A"]) / "bench_compare.py").write_text(
        'def make_script():\n    return "import time; time.sleep(30)"'
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    trials = json.loads((output / "trials.json").read_text())["trials"]
    assert [trial["status"] for trial in trials] == ["timeout", "ok"]
    assert launches == ["memory", ("A", 1), "memory", ("B", 1)]


def test_comparison_uses_matched_trials_and_labels_first_generation_scope(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=2)
    for side, seconds in (("A", 2), ("B", 1)):
        result = {
            "gen_s": seconds,
            "audio_s": 2,
            "rtf": 2 / seconds,
            "speech_tokens": 2,
            "peak_mem_gb": 1,
        }
        code = "print(" + repr("BENCH_RESULT:" + json.dumps(result)) + ")"
        (Path(plan[side]) / "bench_compare.py").write_text(
            "def make_script():\n    return " + repr(code)
        )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["median_paired_B_over_A_gen_s"] == 0.5
    assert summary["measurement_scope"]["warmup_calls"] == 0
    assert (
        summary["measurement_scope"]["generation_call"]
        == "first call in a fresh worker process"
    )
    assert "excludes model loading" in summary["measurement_scope"]["gen_s"]
    manifest = json.loads((output / "trials.json").read_text())
    assert manifest["measurement_scope"] == summary["measurement_scope"]


def test_actual_imported_files_are_identified_by_content_not_local_paths(
    paired: tuple, tmp_path: Path
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=1)
    (tmp_path / "stubs" / "sitecustomize.py").write_text("")
    for side in ("A", "B"):
        package = Path(plan[side]) / "vibevoice_mlx"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "generate.py").write_text("marker = " + repr(side))
        code = 'import vibevoice_mlx.generate\nprint(\'BENCH_RESULT:{"gen_s":1,"audio_s":2,"rtf":2,"speech_tokens":2,"peak_mem_gb":1}\')'
        (Path(plan[side]) / "bench_compare.py").write_text(
            "def make_script():\n    return " + repr(code)
        )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 0
    trials = json.loads((output / "trials.json").read_text())["trials"]
    for trial in trials:
        module = trial["provenance"]["modules"]["vibevoice_mlx.generate"]
        assert module == {
            "origin": "checkout",
            "sha256": hashlib.sha256(
                ("marker = " + repr(trial["side"])).encode()
            ).hexdigest(),
        }


@pytest.mark.parametrize(
    "updates",
    [
        {"seeds": [True]},
        {"repeats": 0},
        {"timeout": -1},
        {"model_revision": "/private/secret"},
        {"voice_arg": "/missing/voice.wav"},
    ],
)
def test_invalid_plan_never_launches_a_worker(
    paired: tuple, tmp_path: Path, updates: dict
) -> None:
    runner, plan, output, launches = paired
    plan.update(updates)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 2
    assert launches == []
    assert not output.exists()
