"""Diffusion schedules reject invalid counts before doing inference work."""

import sys
from collections.abc import Callable
from itertools import pairwise

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.generate import (
    GenerationOptions,
    dpm_solver_2m,
    dpm_solver_sde_2m,
    generate,
)


def test_ode_rejects_zero_steps() -> None:
    with mx.stream(mx.cpu), pytest.raises(ValueError, match="diffusion steps"):
        condition = mx.zeros((1, 1))
        dpm_solver_2m(None, condition, condition, 1.3, num_steps=0)


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize(
    "num_steps", [0, -1, 1000, 1001, True, np.bool_(True), 1.5, 1.0]
)
def test_invalid_steps_fail_before_rng(
    solver: Callable[..., mx.array],
    num_steps: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_rng(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid steps must fail before RNG work")

    monkeypatch.setattr(mx.random, "key", unexpected_rng)
    with pytest.raises(ValueError, match="diffusion steps.*1.*999"):
        solver(None, None, None, 1.3, num_steps=num_steps)


@pytest.mark.parametrize("num_steps", [0, -1, 1000, 1001, True, 1.5])
def test_generate_validates_mutated_steps_before_model_access(
    num_steps: object,
) -> None:
    options = GenerationOptions()
    options.diffusion_steps = num_steps
    with pytest.raises(ValueError, match="diffusion steps.*1.*999"):
        generate(object(), [], options)


@pytest.mark.parametrize("value", ["0", "-1", "1000", "1001", "1.5"])
def test_cli_rejects_invalid_steps_before_loading(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid steps must fail before model loading")

    monkeypatch.setattr(e2e_pipeline, "load_model", unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--text",
            "Hello",
            "--diffusion-steps",
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        e2e_pipeline.main()
    assert error.value.code == 2
    assert "diffusion" in capsys.readouterr().err


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("num_steps", [1, 999, np.int8(127), np.uint8(255)])
def test_valid_steps_have_distinct_denoiser_timesteps(
    solver: Callable[..., mx.array],
    num_steps: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zero_noise(shape: tuple[int, ...], **kwargs: object) -> mx.array:
        return mx.zeros(shape, dtype=mx.float32)

    timesteps = []

    def head(sample: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        timesteps.append(int(timestep.item()))
        return mx.zeros_like(sample)

    monkeypatch.setattr(mx.random, "normal", zero_noise)
    with mx.stream(mx.cpu):
        condition = mx.zeros((1, 1))
        result = solver(head, condition, condition, 1.3, num_steps=num_steps)
        np.testing.assert_array_equal(np.asarray(result), 0.0)

    assert len(timesteps) == int(num_steps)
    assert all(left > right > 0 for left, right in pairwise(timesteps))
    assert timesteps[0] == 999
    if num_steps == 999:
        assert timesteps == list(range(999, 0, -1))


@pytest.mark.parametrize(
    "arguments", [[], ["--diffusion-steps", "1"], ["--diffusion-steps", "999"]]
)
def test_cli_accepts_supported_steps(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModelLoadReached(Exception):
        pass

    def reached_load(*args: object, **kwargs: object) -> None:
        raise ModelLoadReached

    monkeypatch.setattr(e2e_pipeline, "load_model", reached_load)
    monkeypatch.setattr(
        sys, "argv", ["vibevoice-mlx", "--model", ".", "--text", "Hello", *arguments]
    )
    with pytest.raises(ModelLoadReached):
        e2e_pipeline.main()
