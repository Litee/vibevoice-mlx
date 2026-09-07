"""Public solvers follow the reference cosine diffusion schedule."""

import math
from collections.abc import Callable, Iterator

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx.generate import dpm_solver_2m, dpm_solver_sde_2m


@pytest.fixture(autouse=True)
def cpu() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


def reference_parameters() -> tuple[list[float], list[float]]:
    # Independent scalar translation of Microsoft's betas_for_alpha_bar and
    # cumulative-product schedule, retaining float64 instead of Torch float32:
    # https://github.com/microsoft/VibeVoice/blob/main/vibevoice/schedule/dpm_solver.py
    def curve(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    cumulative = 1.0
    alphas, sigmas = [], []
    for index in range(1000):
        beta = min(1 - curve((index + 1) / 1000) / curve(index / 1000), 0.999)
        cumulative *= 1 - beta
        alphas.append(math.sqrt(cumulative))
        sigmas.append(math.sqrt(1 - cumulative))
    return alphas, sigmas


@pytest.mark.parametrize(
    "solver", [dpm_solver_2m, dpm_solver_sde_2m], ids=["ode", "sde"]
)
def test_one_step_uses_clamped_cosine_schedule_and_zero_endpoint(
    solver: Callable[..., mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    def noise(shape: tuple[int, ...], **kwargs: object) -> mx.array:
        return mx.full(shape, 0.125, dtype=mx.float32)

    observed_timesteps = []

    def head(sample: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        observed_timesteps.append(int(timestep.item()))
        return mx.full(sample.shape, 0.25, dtype=mx.float32)

    monkeypatch.setattr(mx.random, "normal", noise)
    condition = mx.zeros((1, 1), dtype=mx.float32)
    result = solver(
        head, condition, condition, 1.3, num_steps=1, seed=42, dtype=mx.float32
    )

    alphas, sigmas = reference_parameters()
    # At the explicit zero-noise endpoint, the result is the last x0 prediction.
    expected = alphas[999] * 0.125 - sigmas[999] * 0.25
    assert observed_timesteps == [999]
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-6, atol=1e-7)


def reference_trajectory(
    num_steps: int, stochastic: bool
) -> tuple[list[int], list[float], float]:
    alphas, sigmas = reference_parameters()
    # Follow the reference's ascending linspace, round, reverse, drop-zero order.
    timesteps = (
        np.linspace(0, 999, num_steps + 1).round().astype(int)[::-1][:-1].tolist()
    )
    lambdas = [
        math.log(alpha) - math.log(sigma) for alpha, sigma in zip(alphas, sigmas)
    ]
    sample = 0.125
    inputs, clean_predictions = [], []
    for index, source in enumerate(timesteps):
        inputs.append(sample)
        # The supplied denoiser is linear in the sample, timestep and condition.
        conditional_v = 0.2 * sample + source / 10000 + 0.125
        unconditional_v = 0.2 * sample + source / 10000 - 0.25
        velocity = unconditional_v + 1.3 * (conditional_v - unconditional_v)
        clean = alphas[source] * sample - sigmas[source] * velocity
        clean_predictions.append(clean)
        if index == len(timesteps) - 1:
            return timesteps, inputs, clean

        target = timesteps[index + 1]
        h = lambdas[target] - lambdas[source]
        derivative = 0.0
        if index:
            previous_h = lambdas[source] - lambdas[timesteps[index - 1]]
            derivative = (clean - clean_predictions[-2]) * h / previous_h

        # Separate midpoint terms from Microsoft's second-order update.
        if stochastic:
            decay = math.exp(-h)
            coefficient = alphas[target] * (1 - decay**2)
            sample = sigmas[target] / sigmas[source] * decay * sample
            sample += coefficient * clean + 0.5 * coefficient * derivative
            sample += sigmas[target] * math.sqrt(1 - decay**2) * 0.125
        else:
            coefficient = -alphas[target] * math.expm1(-h)
            sample = sigmas[target] / sigmas[source] * sample
            sample += coefficient * clean + 0.5 * coefficient * derivative
    raise AssertionError("Reference needs at least one step")


@pytest.mark.parametrize("num_steps", [2, 10, 14, 15, 20, 26, 30])
@pytest.mark.parametrize("stochastic", [False, True], ids=["ode", "sde"])
def test_solver_matches_reference_trajectory(
    num_steps: int, stochastic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def noise(shape: tuple[int, ...], **kwargs: object) -> mx.array:
        return mx.full(shape, 0.125, dtype=mx.float32)

    observed_timesteps, observed_inputs = [], []

    def head(sample: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        observed_timesteps.append(int(timestep.item()))
        observed_inputs.append(float(sample[0, 0].item()))
        return 0.2 * sample + timestep / 10000 + condition

    monkeypatch.setattr(mx.random, "normal", noise)
    solver = dpm_solver_sde_2m if stochastic else dpm_solver_2m
    result = solver(
        head,
        mx.array([[0.125]]),
        mx.array([[-0.25]]),
        1.3,
        num_steps=num_steps,
        seed=42,
        dtype=mx.float32,
    )
    timesteps, inputs, expected = reference_trajectory(num_steps, stochastic)

    # Also checks exact denoiser count and every callback's noise level, so
    # an unchanged endpoint cannot conceal an incorrect earlier trajectory.
    assert observed_timesteps == timesteps
    np.testing.assert_allclose(observed_inputs, inputs, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(np.array(result), expected, rtol=3e-5, atol=3e-6)
