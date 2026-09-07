"""Unsupported solver names fail before loading or evaluating a model."""

import importlib
import sys
from unittest.mock import patch

import pytest

from vibevoice_mlx import e2e_pipeline

generation = importlib.import_module("vibevoice_mlx.generate")


class ModelLoadReachedError(Exception):
    pass


@pytest.mark.parametrize("solver", ["ddpm", "unknown", ""])
def test_api_rejects_unsupported_solver_before_model_access(solver: str) -> None:
    with pytest.raises(ValueError, match="Unsupported solver"):
        generation.generate(object(), [], generation.GenerationOptions(solver=solver))


def test_api_validates_mutated_options() -> None:
    options = generation.GenerationOptions()
    options.solver = "ddpm"
    with pytest.raises(ValueError, match="Unsupported solver"):
        generation.generate(object(), [], options)


@pytest.mark.parametrize("solver", ["ddpm", "unknown"])
def test_cli_rejects_unsupported_solver_before_loading(
    solver: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["vibevoice-mlx", "--text", "Hello", "--solver", solver]
    )
    with (
        patch.object(
            e2e_pipeline, "load_model", side_effect=ModelLoadReachedError
        ) as load,
        pytest.raises(SystemExit) as error,
    ):
        e2e_pipeline.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    load.assert_not_called()


@pytest.mark.parametrize("arguments", [[], ["--solver", "dpm"], ["--solver", "sde"]])
def test_cli_accepts_supported_solvers(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["vibevoice-mlx", "--text", "Hello", *arguments])
    with (
        patch.object(
            e2e_pipeline, "load_model", side_effect=ModelLoadReachedError
        ) as load,
        pytest.raises(ModelLoadReachedError),
    ):
        e2e_pipeline.main()
    load.assert_called_once()
