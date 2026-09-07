"""CFG scales must be finite before inference starts."""

import sys
from collections.abc import Callable

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


def test_ode_rejects_nan_scale() -> None:
    def head(sample: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        return mx.ones_like(sample)

    with mx.stream(mx.cpu), pytest.raises(ValueError, match="cfg_scale.*finite"):
        condition = mx.zeros((1, 1))
        dpm_solver_2m(head, condition, condition, np.nan, num_steps=1)


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize(
    "scale",
    [
        np.nan,
        np.inf,
        -np.inf,
        np.float32(np.nan),
        None,
        "1.3",
        1 + 0j,
        [1.3],
        np.array([1.3]),
        np.array(1.3),
    ],
)
def test_invalid_scales_fail_before_rng(
    solver: Callable[..., mx.array],
    scale: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_rng(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid cfg_scale must fail before RNG work")

    monkeypatch.setattr(mx.random, "key", unexpected_rng)
    with pytest.raises(ValueError, match="cfg_scale.*finite real number"):
        solver(None, None, None, scale, num_steps=1)


@pytest.mark.parametrize(
    "scale", [np.nan, np.inf, -np.inf, np.float32(np.nan), [1.3], 1 + 0j]
)
def test_generate_validates_mutated_scale_before_model_access(scale: object) -> None:
    options = GenerationOptions()
    options.cfg_scale = scale
    with pytest.raises(ValueError, match="cfg_scale.*finite"):
        generate(object(), [], options)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite_scale_before_loading(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("Non-finite cfg_scale must fail before model loading")

    monkeypatch.setattr(e2e_pipeline, "load_model", unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibevoice-mlx",
            "--text",
            "Hello",
            f"--cfg-scale={value}",
        ],
    )
    with pytest.raises(SystemExit) as error:
        e2e_pipeline.main()
    assert error.value.code == 2
    assert "cfg_scale must be a finite real number" in capsys.readouterr().err


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize(
    "scale",
    [
        0,
        -1,
        0.5,
        1.0,
        2.0,
        np.float16(-0.5),
        np.float32(1.5),
        np.float64(2.0),
        np.int16(-1),
        True,
        np.bool_(False),
    ],
)
def test_finite_scales_preserve_guidance_without_clipping(
    solver: Callable[..., mx.array],
    scale: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zero_noise(shape: tuple[int, ...], **kwargs: object) -> mx.array:
        return mx.zeros(shape, dtype=mx.float32)

    def head(sample: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        return mx.broadcast_to(condition, sample.shape)

    monkeypatch.setattr(mx.random, "normal", zero_noise)
    with mx.stream(mx.cpu):

        def solve(value: float) -> np.ndarray:
            return np.asarray(
                solver(
                    head,
                    mx.array([[0.75]]),
                    mx.array([[-0.25]]),
                    value,
                    num_steps=1,
                    dtype=mx.float32,
                )
            )

        unconditional = solve(0.0)
        conditional = solve(1.0)
        actual = solve(scale)
    # With one step and a constant velocity head, CFG interpolates/extrapolates
    # between the unconditional and conditional clean predictions.
    expected = unconditional + scale * (conditional - unconditional)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "arguments", [[], ["--cfg-scale=0"], ["--cfg-scale=-1"], ["--cfg-scale=1.3"]]
)
def test_cli_accepts_finite_scales(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModelLoadReached(Exception):
        pass

    def reached_load(*args: object, **kwargs: object) -> None:
        raise ModelLoadReached

    monkeypatch.setattr(e2e_pipeline, "load_model", reached_load)
    monkeypatch.setattr(sys, "argv", ["vibevoice-mlx", "--text", "Hello", *arguments])
    with pytest.raises(ModelLoadReached):
        e2e_pipeline.main()
