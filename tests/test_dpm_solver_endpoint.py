"""ODE solver order must preserve the trajectory and zero-noise endpoint."""

import importlib
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

generation = importlib.import_module("vibevoice_mlx.generate")


@pytest.mark.parametrize("num_steps", [3, 10, 14, 15, 20])
def test_penultimate_step_uses_second_order(num_steps: int) -> None:
    condition = mx.zeros((1, 2), dtype=mx.float16)
    zero = mx.zeros((1, generation.VAE_DIM), dtype=mx.float32)
    penultimate_clean = mx.ones_like(zero)
    final_clean = mx.full(zero.shape, 0.05, dtype=mx.float32)
    predictions = [zero] * (num_steps - 2) + [penultimate_clean, final_clean]

    # Zero noise and earlier predictions isolate the penultimate update.
    with (
        patch.object(mx.random, "normal", return_value=zero),
        patch.object(
            generation, "_dpm_denoise_step", side_effect=predictions
        ) as denoise,
    ):
        result = generation.dpm_solver_2m(
            None, condition, condition, cfg_scale=1.3, num_steps=num_steps, seed=42
        )

    # Scalar DPM-Solver++ midpoint reference on the port's existing schedule:
    # sample=0, D0=1, D1=(1-0)/r. Microsoft keeps this second-order update
    # for solver_order=2 even when lower_order_second applies (<15 steps).
    schedule = np.round(
        np.linspace(generation.DDPM_STEPS - 1, 0, num_steps + 1)
    ).astype(np.int64)
    previous, source, target = schedule[-4:-1]
    h = generation._LAMBDA_NP[target] - generation._LAMBDA_NP[source]
    h_previous = generation._LAMBDA_NP[source] - generation._LAMBDA_NP[previous]
    coefficient = -generation._ALPHA_NP[target] * np.expm1(-h)
    expected_final_input = coefficient + 0.5 * coefficient * h / h_previous

    # The endpoint alone hides this bug: a controlled final clean prediction
    # overwrites the sample regardless of the order used one step earlier.
    final_input = denoise.call_args_list[-1].args[1]
    np.testing.assert_allclose(
        np.array(final_input), expected_final_input, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.array(result), np.array(final_clean), atol=1e-6)


@pytest.mark.parametrize("num_steps", [1, 10, 14, 15, 20, 50])
def test_zero_noise_endpoint_returns_final_clean_prediction(num_steps: int) -> None:
    condition = mx.zeros((1, 2), dtype=mx.float16)
    previous_clean = mx.full((1, generation.VAE_DIM), 0.1, dtype=mx.float32)
    final_clean = mx.full((1, generation.VAE_DIM), 0.05, dtype=mx.float32)
    predictions = [previous_clean] * (num_steps - 1) + [final_clean]

    # Control denoiser predictions while exercising the real solver updates.
    # The changed final prediction exposes invalid second-order extrapolation.
    with patch.object(generation, "_dpm_denoise_step", side_effect=predictions):
        result = generation.dpm_solver_2m(
            None, condition, condition, cfg_scale=1.3, num_steps=num_steps, seed=42
        )

    np.testing.assert_allclose(np.array(result), np.array(final_clean), atol=1e-6)
