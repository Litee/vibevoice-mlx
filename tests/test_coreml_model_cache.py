from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from vibevoice_mlx import coreml_semantic, e2e_pipeline


def _stub_coreml_setup(monkeypatch, source_root: Path, home: Path) -> None:
    coremltools = ModuleType("coremltools")
    coremltools.ComputeUnit = SimpleNamespace(
        CPU_AND_NE=SimpleNamespace(name="CPU_AND_NE"),
        CPU_AND_GPU=SimpleNamespace(name="CPU_AND_GPU"),
    )
    coremltools.models = SimpleNamespace(
        MLModel=lambda path, compute_units: SimpleNamespace(
            path=path, compute_units=compute_units
        )
    )
    monkeypatch.setitem(sys.modules, "coremltools", coremltools)

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda model_id: str(source_root)
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        coreml_semantic,
        "prepare_explicit_cache_model",
        lambda source, cache_dir: source,
    )

    class Encoder:
        def __init__(self, model: object) -> None:
            self.model = model

        def __call__(self, audio: np.ndarray) -> np.ndarray:
            return np.zeros((1, 128, 1), dtype=np.float32)

        def reset(self) -> None:
            pass

    monkeypatch.setattr(coreml_semantic, "ExplicitCacheEncoder", Encoder)


def _semantic_model() -> SimpleNamespace:
    return SimpleNamespace(semantic_connector=lambda features: features)


def test_interrupted_coreml_copy_does_not_leave_a_cached_package(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "download"
    source = source_root / "semantic_encoder_streaming.mlpackage"
    source.mkdir(parents=True)
    (source / "complete").write_text("yes")
    home = tmp_path / "home"
    _stub_coreml_setup(monkeypatch, source_root, home)
    monkeypatch.chdir(tmp_path)

    def interrupted_copy(source: str, destination: str, **kwargs: object) -> None:
        partial = Path(destination)
        partial.mkdir(parents=True)
        (partial / "partial").write_text("incomplete")
        raise OSError("copy interrupted")

    monkeypatch.setattr(shutil, "copytree", interrupted_copy)

    result = e2e_pipeline._try_coreml_semantic(_semantic_model(), SimpleNamespace())

    cache = home / ".cache" / "vibevoice-mlx" / "coreml"
    assert result is None
    assert not (cache / "semantic_encoder_streaming.mlpackage").exists()
    assert list(cache.iterdir()) == []


def test_concurrent_coreml_copies_converge_on_completed_package(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "download"
    source = source_root / "semantic_encoder_streaming.mlpackage"
    source.mkdir(parents=True)
    (source / "complete").write_text("yes")
    home = tmp_path / "home"
    _stub_coreml_setup(monkeypatch, source_root, home)
    monkeypatch.chdir(tmp_path)

    real_rename = Path.rename
    renames_started = threading.Barrier(2)
    rename_failures: list[OSError] = []

    def simultaneous_rename(source: Path, target: Path) -> Path:
        renames_started.wait(timeout=5)
        try:
            return real_rename(source, target)
        except OSError as error:
            rename_failures.append(error)
            raise

    monkeypatch.setattr(Path, "rename", simultaneous_rename)
    results: list[object] = []

    def load() -> None:
        results.append(
            e2e_pipeline._try_coreml_semantic(_semantic_model(), SimpleNamespace())
        )

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    cache = home / ".cache" / "vibevoice-mlx" / "coreml"
    package = cache / "semantic_encoder_streaming.mlpackage"
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert all(result is not None for result in results)
    assert len(rename_failures) == 1
    assert (package / "complete").read_text() == "yes"
    assert list(cache.iterdir()) == [package]
