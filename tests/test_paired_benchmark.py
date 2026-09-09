"""Exercise the paired CLI with real, lightweight child interpreters."""

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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
    original_popen = subprocess.Popen
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
        return original_run(command, **kwargs)

    def popen(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        if command[1:2] != ["-c"] or command[2] != runner.WORKER:
            return original_popen(command, **kwargs)
        launches.append((Path(kwargs["cwd"]).name, json.loads(command[-1])["seed"]))
        return original_popen(
            command,
            env={
                **os.environ,
                "PYTHONPATH": str(stubs),
                "PAIRED_SECRET": "do-not-publish-me",
            },
            **kwargs,
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
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
    assert first["configuration_source"] == "executed_worker_state"
    assert first["effective"] == {
        "quantization": "fp16",
        "semantic_backend": "mlx",
        "voice_reference": "none",
    }
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
            "effective": {
                "quantization": "fp16",
                "semantic_backend": "mlx",
                "voice_reference": "none",
            },
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
        code = 'import vibevoice_mlx.generate\nprint(\'BENCH_RESULT:{"gen_s":1,"audio_s":2,"rtf":2,"speech_tokens":2,"peak_mem_gb":1,"effective":{"quantization":"fp16","semantic_backend":"mlx","voice_reference":"none"}}\')'
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


def wait_until(predicate: Any) -> None:
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Subprocess lifecycle condition did not become true")
        time.sleep(0.02)


def paused_runner(paired: tuple, tmp_path: Path) -> tuple[list[str], Path]:
    runner, plan, _, _ = paired
    plan.update(seeds=[1], repeats=1, timeout=30)
    ready = tmp_path / "ready.json"
    script = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(ready)!r}, 'w').write(str(os.getpid()) + ',' + str(child.pid))\n"
        "time.sleep(60)\n"
    )
    (Path(plan["A"]) / "bench_compare.py").write_text(
        "def make_script():\n    return " + repr(script)
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('bench_paired', {runner.__file__!r})\n"
        "runner = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(runner)\n"
        f"runner.tempfile.gettempdir = lambda: {str(tmp_path)!r}\n"
        "runner.available_memory = lambda: runner.MIN_AVAILABLE_BYTES\n"
        "raise SystemExit(runner.main())\n"
    )
    return [sys.executable, "-c", code, "--plan", str(source)], ready


def process_running(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) and not result.stdout.lstrip().startswith("Z")


def kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_worker_retains_generation_lock_after_parent_is_killed(
    paired: tuple, tmp_path: Path
) -> None:
    command, ready = paused_runner(paired, tmp_path)
    parent = subprocess.Popen(
        command + ["--output", str(tmp_path / "first")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = None
    try:
        wait_until(lambda: ready.exists() and "," in ready.read_text())
        worker, descendant = map(int, ready.read_text().split(","))
        parent.kill()
        parent.wait(timeout=10)
        assert process_running(worker)
        assert process_running(descendant)
        second = subprocess.run(
            command + ["--output", str(tmp_path / "second")],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert second.returncode == 2
        assert not (tmp_path / "second").exists()
    finally:
        if worker is not None:
            kill_group(worker)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)


@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_cancellation_kills_worker_and_descendant_before_releasing_lock(
    paired: tuple, tmp_path: Path, cancel_signal: int
) -> None:
    command, ready = paused_runner(paired, tmp_path)
    parent = subprocess.Popen(
        command + ["--output", str(tmp_path / "cancelled")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = None
    try:
        wait_until(lambda: ready.exists() and "," in ready.read_text())
        worker, descendant = map(int, ready.read_text().split(","))
        parent.send_signal(cancel_signal)
        assert parent.wait(timeout=10) == 130
        wait_until(
            lambda: not process_running(worker) and not process_running(descendant)
        )
        with (tmp_path / "vibevoice-generation.lock").open("a") as lock:
            paired[0].fcntl.flock(
                lock, paired[0].fcntl.LOCK_EX | paired[0].fcntl.LOCK_NB
            )
    finally:
        if worker is not None:
            kill_group(worker)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)


def test_timeout_kills_worker_descendants_before_next_trial(
    paired: tuple, tmp_path: Path
) -> None:
    _, ready = paused_runner(paired, tmp_path)
    runner, plan, output, _ = paired
    plan["timeout"] = 1
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    worker, descendant = map(int, ready.read_text().split(","))
    wait_until(lambda: not process_running(worker) and not process_running(descendant))
    trials = json.loads((output / "trials.json").read_text())["trials"]
    assert [trial["status"] for trial in trials] == ["timeout", "ok"]


@pytest.mark.parametrize(
    ("updates", "old", "new", "key", "actual"),
    [
        ({"sem_mode": "coreml"}, "", "", "semantic_backend", "mlx"),
        (
            {},
            "r = _try_mlx_semantic(model, config, model_id)",
            "r = None",
            "semantic_backend",
            "none",
        ),
        (
            {"quantize": 8},
            'quantize_bits=args["quantize"]',
            "quantize_bits=None",
            "quantization",
            "fp16",
        ),
        (
            {"voice_arg": True},
            "ref_audio=voice_list",
            "ref_audio=None",
            "voice_reference",
            "none",
        ),
    ],
)
def test_legacy_worker_configuration_is_observed_and_mismatches_fail_closed(
    paired: tuple,
    tmp_path: Path,
    updates: dict,
    old: str,
    new: str,
    key: str,
    actual: str,
) -> None:
    runner, plan, output, _ = paired
    if updates.get("voice_arg"):
        voice = tmp_path / "secret-voice.safetensors"
        voice.write_bytes(b"reference")
        updates = {**updates, "voice_arg": str(voice)}
    plan.update(seeds=[1], repeats=1, **updates)
    worker = Path(plan["A"]) / "bench_compare.py"
    worker.write_text(
        worker.read_text().replace(old, new) if old else worker.read_text()
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    manifest = json.loads((output / "trials.json").read_text())
    assert manifest["trials"][0]["status"] == "configuration_mismatch"
    assert manifest["trials"][0]["effective"][key] == actual
    assert manifest["trials"][0]["effective"] != manifest["trials"][0]["requested"]
    assert json.loads((output / "summary.json").read_text())["complete_pairs"] == 0
    assert str(tmp_path) not in (output / "trials.json").read_text()


@pytest.mark.parametrize("effective", [None, {}, {"quantization": "/private/secret"}])
def test_unknown_worker_configuration_is_excluded_without_leaking_strings(
    paired: tuple, tmp_path: Path, effective: dict | None
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=1)
    result = {"gen_s": 1, "audio_s": 2, "rtf": 2, "speech_tokens": 2, "peak_mem_gb": 1}
    if effective is not None:
        result["effective"] = effective
    script = "print(" + repr("BENCH_RESULT:" + json.dumps(result)) + ")"
    (Path(plan["A"]) / "bench_compare.py").write_text(
        "def make_script():\n    return " + repr(script)
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    serialized = (output / "trials.json").read_text()
    assert json.loads(serialized)["trials"][0]["status"] == "unknown_configuration"
    assert "/private/secret" not in serialized
    assert json.loads((output / "summary.json").read_text())["complete_pairs"] == 0


@pytest.mark.parametrize("reported_backend", ["mlx", "none"])
def test_explicit_worker_metadata_is_retained_and_checked_against_runtime(
    paired: tuple, tmp_path: Path, reported_backend: str
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=1)
    worker = Path(plan["A"]) / "bench_compare.py"
    effective = {
        "quantization": "fp16",
        "semantic_backend": reported_backend,
        "voice_reference": "none",
    }
    worker.write_text(
        worker.read_text().replace(
            '"gen_s": gen_s,', f'"effective": {effective!r}, "gen_s": gen_s,'
        )
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == (
        0 if reported_backend == "mlx" else 1
    )
    first = json.loads((output / "trials.json").read_text())["trials"][0]
    assert first["effective"] == effective
    assert first["configuration_source"] == "worker_report"
    assert first["observed_effective"]["semantic_backend"] == "mlx"


@pytest.mark.parametrize(
    "mutation",
    [
        "model = object()",
        "model.named_modules()[0][1].weight.dtype = 'float32'",
        "from types import SimpleNamespace; model.named_modules = lambda: [('unknown', SimpleNamespace(weight=0))]",
    ],
)
def test_unprovable_legacy_model_configuration_fails_closed(
    paired: tuple, tmp_path: Path, mutation: str
) -> None:
    runner, plan, output, _ = paired
    plan.update(seeds=[1], repeats=1)
    worker = Path(plan["A"]) / "bench_compare.py"
    worker.write_text(
        worker.read_text().replace(
            'sem_mode = args["sem_mode"]', mutation + '\nsem_mode = args["sem_mode"]'
        )
    )
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    assert runner.main(["--plan", str(source), "--output", str(output)]) == 1
    first = json.loads((output / "trials.json").read_text())["trials"][0]
    assert first["status"] == "unknown_configuration"
    assert first["effective"]["quantization"] is None
