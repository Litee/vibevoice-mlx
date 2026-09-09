"""Serial paired benchmarks across two checkouts, using their bench_compare worker.

Run: uv run python bench_paired.py --plan plan.json --output paired-results
Plan keys: A and B (checkout directories), model (local snapshot directory),
text, seeds (integer list), repeats (positive integer, default 1). Optional keys:
voice_arg, quantize (4/8/null), sem_mode (mlx/coreml/none), max_tokens (default
400), timeout (seconds, default 600), model_revision (declared revision).

Each repeat visits every seed; successive pairs alternate AB, BA. The same
interpreter runs both checkouts. Memory blocks and failed workers count as
unsuccessful trials; only complete pairs contribute to paired comparisons.
No worker output, environment, prompt, or absolute path is saved or printed.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MIN_AVAILABLE_BYTES = 12 * 1024**3
MEASUREMENT_SCOPE = {
    "generation_call": "first call in a fresh worker process",
    "warmup_calls": 0,
    "gen_s": "wall time of generate only; excludes model loading, semantic encoder setup, tokenization and voice preparation; includes lazy fast-path initialization",
    "cache_state": "fresh process; OS filesystem and external caches are not cleared",
}
WORKER = """
import hashlib, importlib.metadata, json, platform, re, resource, runpy, sys
from pathlib import Path
checkout = Path.cwd().resolve()
sys.path.insert(0, str(checkout))
payload = json.loads(sys.argv[1])
def file_hash(path: str | Path | None) -> str | None:
    try:
        with Path(path).open("rb") as source:
            result = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                result.update(block)
            return result.hexdigest()
    except (OSError, TypeError):
        return None
def safe_version(value: str) -> str | None:
    return value if re.fullmatch(r"[A-Za-z0-9.!+_-]{1,100}", value) else None
try:
    exec(runpy.run_path("bench_compare.py")["make_script"](), {})
finally:
    # Capture before provenance hashing adds its own temporary allocations.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    libraries = {}
    for name in ("mlx", "numpy", "transformers", "huggingface-hub", "safetensors", "scipy", "soundfile"):
        try:
            libraries[name] = safe_version(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    modules = {}
    for name, module in sorted(sys.modules.copy().items()):
        if name == "mlx.core" or name == "vibevoice_mlx" or name.startswith("vibevoice_mlx."):
            path = getattr(module, "__file__", None)
            origin = "unavailable"
            if path:
                origin = "checkout" if Path(path).resolve().is_relative_to(checkout) else "external"
            modules[name] = {"origin": origin, "sha256": file_hash(path)}
    model = Path(payload["model"])
    revision = model.name if model.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40,64}", model.name) else None
    print("PAIRED_PROVENANCE:" + json.dumps({
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(), "executable_sha256": file_hash(sys.executable)},
        "platform": {"system": platform.system(), "machine": platform.machine(), "release": safe_version(platform.release())},
        "libraries": libraries,
        "modules": modules,
        "benchmark_source_sha256": file_hash(checkout / "bench_compare.py"),
        "model": {"snapshot_revision": revision, "config_sha256": file_hash(model / "config.json")},
        "process_lifetime_peak_rss_bytes": rss_bytes,
    }))
"""


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def available_memory() -> int:
    """macOS free + inactive + speculative pages; fail closed if unavailable."""
    result = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=10, check=True
    )
    page_size = re.search(r"page size of (\d+) bytes", result.stdout)
    if page_size is None:
        raise ValueError("Missing page size")
    pages = []
    for name in ("free", "inactive", "speculative"):
        match = re.search(rf"^Pages {name}:\s+(\d+)\.", result.stdout, re.MULTILINE)
        if match is None:
            raise ValueError("Missing memory counter")
        pages.append(int(match[1]))
    return sum(pages) * int(page_size[1])


def validate_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise TypeError("Expected a plan object")
    for key in ("A", "B", "model"):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_dir():
            raise ValueError("Expected a local directory")
    for side in ("A", "B"):
        if not (Path(plan[side]) / "bench_compare.py").is_file():
            raise ValueError("Missing worker")
    if not isinstance(plan.get("text"), str) or not plan["text"].strip():
        raise ValueError("Expected text")
    if (
        not isinstance(plan.get("seeds"), list)
        or not plan["seeds"]
        or any(type(seed) is not int for seed in plan["seeds"])
    ):
        raise ValueError("Expected integer seeds")
    for key, default in (("repeats", 1), ("max_tokens", 400), ("timeout", 600)):
        value = plan.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError("Expected positive integer")
    if plan.get("quantize") not in (None, 4, 8):
        raise ValueError("Invalid quantization")
    if plan.get("sem_mode", "mlx") not in ("mlx", "coreml", "none"):
        raise ValueError("Invalid semantic mode")
    if plan.get("voice_arg") is not None and (
        not isinstance(plan["voice_arg"], str) or not Path(plan["voice_arg"]).is_file()
    ):
        raise ValueError("Expected local voice input")
    if plan.get("model_revision") is not None and (
        not isinstance(plan["model_revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", plan["model_revision"])
    ):
        raise ValueError("Expected a revision hash")


def input_provenance(plan: dict[str, Any]) -> dict[str, Any]:
    """Input identities, never input contents or paths.

    The metadata digest detects local model file-set changes; it is explicitly
    not a weight-content digest or verification of a declared remote revision.
    """
    model = Path(plan["model"])
    files = []
    for path in sorted(model.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append((str(path.relative_to(model)), stat.st_size, stat.st_mtime_ns))
    voice_hash = None
    if plan.get("voice_arg"):
        hasher = hashlib.sha256()
        with Path(plan["voice_arg"]).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        voice_hash = hasher.hexdigest()
    return {
        "text_sha256": digest(plan["text"].encode()),
        "voice_sha256": voice_hash,
        "model_file_metadata_sha256": digest(json.dumps(files).encode()),
        "declared_model_revision": plan.get("model_revision"),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_provenance(checkout: Path) -> dict[str, Any]:
    """Record revision and working-tree state without publishing git paths."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return {
            "commit": revision if re.fullmatch(r"[0-9a-f]{40,64}", revision) else None,
            "dirty": bool(status.strip()),
            "working_tree_status_sha256": digest(status.encode()),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "working_tree_status_sha256": None}


def worker_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "returncode": result.returncode,
        "stdout_sha256": digest(result.stdout.encode()),
        "stderr_sha256": digest(result.stderr.encode()),
    }
    provenance_lines = [
        line.removeprefix("PAIRED_PROVENANCE:")
        for line in result.stdout.splitlines()
        if line.startswith("PAIRED_PROVENANCE:")
    ]
    try:
        if len(provenance_lines) != 1:
            raise ValueError("Expected one provenance record")
        provenance = json.loads(provenance_lines[0])
        rss = provenance.pop("process_lifetime_peak_rss_bytes")
        record["provenance"] = provenance
    except (ValueError, KeyError, TypeError, AttributeError):
        return {
            **record,
            "status": "worker_failed" if result.returncode else "invalid_result",
        }
    if result.returncode:
        return {**record, "status": "worker_failed"}
    lines = [
        line.removeprefix("BENCH_RESULT:")
        for line in result.stdout.splitlines()
        if line.startswith("BENCH_RESULT:")
    ]
    try:
        if len(lines) != 1:
            raise ValueError("Expected one result")
        value = json.loads(lines[0])
        metrics = {
            key: value[key] for key in ("gen_s", "audio_s", "rtf", "speech_tokens")
        }
        metrics["mlx_peak_generation_bytes"] = value["peak_mem_gb"] * 1_000_000_000
        metrics["process_lifetime_peak_rss_bytes"] = rss
        if any(
            type(item) not in (int, float) or not math.isfinite(item) or item < 0
            for item in metrics.values()
        ):
            raise ValueError("Invalid metrics")
        if (
            metrics["gen_s"] <= 0
            or metrics["audio_s"] <= 0
            or type(metrics["speech_tokens"]) is not int
            or metrics["speech_tokens"] <= 0
        ):
            raise ValueError("Empty or invalid generation")
        return {**record, "status": "ok", "metrics": metrics, "provenance": provenance}
    except (ValueError, KeyError, TypeError):
        return {**record, "status": "invalid_result"}


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = []
    for start in range(0, len(trials), 2):
        pair = trials[start : start + 2]
        if len(pair) == 2 and all(trial["status"] == "ok" for trial in pair):
            sides = {trial["side"]: trial["metrics"] for trial in pair}
            ratios.append(sides["B"]["gen_s"] / sides["A"]["gen_s"])
    return {
        "measurement_scope": MEASUREMENT_SCOPE,
        "planned_trials": len(trials),
        "successful_trials": sum(trial["status"] == "ok" for trial in trials),
        "status_counts": {
            status: sum(trial["status"] == status for trial in trials)
            for status in sorted({trial["status"] for trial in trials})
        },
        "complete_pairs": len(ratios),
        "median_paired_B_over_A_gen_s": statistics.median(ratios) if ratios else None,
        "memory_labels": {
            "mlx_peak_generation_bytes": "MLX allocator peak since reset immediately before generate; includes live model allocations",
            "process_lifetime_peak_rss_bytes": "OS peak resident set over process lifetime through generation completion; not a generation-only delta",
        },
    }


def run_plan(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run paired subprocesses serially and persist every outcome."""
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    for repeat in range(plan.get("repeats", 1)):
        for seed in plan["seeds"]:
            pair_index = len(trials) // 2
            for side in ("A", "B") if pair_index % 2 == 0 else ("B", "A"):
                trials.append(
                    {
                        "pair": pair_index,
                        "repeat": repeat,
                        "seed": seed,
                        "side": side,
                        "status": "pending",
                    }
                )
    manifest = {
        "schema_version": 1,
        "measurement_scope": MEASUREMENT_SCOPE,
        "memory_gate_bytes": MIN_AVAILABLE_BYTES,
        "inputs": input_provenance(plan),
        "requested": {
            "quantize": plan.get("quantize"),
            "sem_mode": plan.get("sem_mode", "mlx"),
            "max_tokens": plan.get("max_tokens", 400),
            "timeout_seconds": plan.get("timeout", 600),
        },
        "trials": trials,
    }
    write_json(output / "trials.json", manifest)
    for trial in trials:
        trial["source"] = source_provenance(Path(plan[trial["side"]]).resolve())
        payload = {
            "model": str(Path(plan["model"]).resolve()),
            "text": plan["text"],
            "voice_arg": str(Path(plan["voice_arg"]).resolve())
            if plan.get("voice_arg")
            else None,
            "seed": trial["seed"],
            "max_tokens": plan.get("max_tokens", 400),
            "quantize": plan.get("quantize"),
            "sem_mode": plan.get("sem_mode", "mlx"),
            "audio_out": None,
        }
        try:
            trial["available_memory_bytes"] = available_memory()
            if trial["available_memory_bytes"] < MIN_AVAILABLE_BYTES:
                trial["status"] = "blocked_memory"
            else:
                trial.update(
                    worker_result(
                        subprocess.run(
                            [sys.executable, "-c", WORKER, json.dumps(payload)],
                            cwd=Path(plan[trial["side"]]).resolve(),
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=plan.get("timeout", 600),
                        )
                    )
                )
        except subprocess.TimeoutExpired:
            trial["status"] = "timeout"
        except (OSError, ValueError, subprocess.CalledProcessError):
            trial["status"] = "launch_or_memory_check_failed"
        write_json(output / "trials.json", manifest)
    summary = summarize(trials)
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        with (Path(tempfile.gettempdir()) / "vibevoice-generation.lock").open(
            "a"
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            summary = run_plan(plan, args.output)
        print(json.dumps(summary, allow_nan=False))
        return 0 if summary["successful_trials"] == summary["planned_trials"] else 1
    except (OSError, ValueError, TypeError):
        print(
            "Paired run could not start: invalid plan, unavailable output directory, or generation lock held.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
