"""Prepared conditioning preserves the public diffusion head and solver outputs."""

from collections.abc import Callable, Iterator

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_map

from vibevoice_mlx.fast_forward import FastDiffusionHead, PreparedDiffusionConditioning
from vibevoice_mlx.generate import dpm_solver_2m, dpm_solver_sde_2m
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


def make_head(
    dtype: mx.Dtype = mx.float16,
    bits: int | None = None,
    layers: int = 2,
) -> FastDiffusionHead:
    mx.random.seed(19)
    config = VibeVoiceConfig(
        hidden_size=64,
        num_hidden_layers=0,
        vocab_size=8,
        diffusion_layers=layers,
    )
    model = VibeVoiceModel(config)
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    if bits is not None:
        nn.quantize(
            model.diffusion_head,
            group_size=32,
            bits=bits,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
    return FastDiffusionHead(model, config)


@pytest.mark.parametrize("weight_dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_prepared_head_matches_individual_timestep_calls(
    weight_dtype: mx.Dtype,
    dtype: mx.Dtype,
    bits: int | None,
) -> None:
    head = make_head(weight_dtype, bits)
    timesteps = mx.array([999, 749, 500, 250], dtype=dtype)
    condition = mx.random.normal((2, 64)).astype(dtype)
    prepared = head.prepare_conditioning(condition, timesteps, dtype=dtype)
    for step in range(len(timesteps)):
        noisy = mx.random.normal((2, 64)).astype(dtype)
        expected = head(noisy, timesteps[step : step + 1], condition)
        actual = head.forward_prepared(noisy, prepared, step)
        assert actual.shape == expected.shape == (2, 64)
        assert actual.dtype == expected.dtype
        tolerance = 2e-3 if actual.dtype == mx.float16 else 1e-6
        np.testing.assert_allclose(
            np.array(actual),
            np.array(expected),
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("num_steps", [1, 2, 10, 20, 26])
@pytest.mark.parametrize("cfg_scale", [-0.5, 0.0, 1.0, 1.3])
@pytest.mark.parametrize("bits", [None, 8])
def test_solver_prepares_schedule_once_and_preserves_callback_fallback(
    monkeypatch: pytest.MonkeyPatch,
    solver: Callable[..., mx.array],
    num_steps: int,
    cfg_scale: float,
    bits: int | None,
) -> None:
    head = make_head(bits=bits)
    condition = mx.random.normal((1, 64)).astype(mx.float16)
    negative = -condition
    observed_timesteps: list[int] = []

    def unprepared(noisy: mx.array, timestep: mx.array, cond: mx.array) -> mx.array:
        observed_timesteps.append(int(timestep.item()))
        return head(noisy, timestep, cond)

    expected = solver(unprepared, condition, negative, cfg_scale, num_steps=num_steps)
    schedules: list[mx.array] = []
    prepare = head.prepare_conditioning

    def record_prepare(
        cond: mx.array,
        timesteps: mx.array,
        *,
        dtype: mx.Dtype,
    ) -> PreparedDiffusionConditioning:
        schedules.append(timesteps)
        return prepare(cond, timesteps, dtype=dtype)

    monkeypatch.setattr(head, "prepare_conditioning", record_prepare)
    actual = solver(head, condition, negative, cfg_scale, num_steps=num_steps)
    assert len(schedules) == 1
    assert np.array(schedules[0]).tolist() == observed_timesteps
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("layers", [0, 2])
def test_prepared_conditions_remain_independent_across_calls(layers: int) -> None:
    head = make_head(layers=layers)
    first = mx.ones((2, 64), dtype=mx.float16)
    second = -first
    noisy = mx.full((2, 64), 0.25, dtype=mx.float16)
    first_times = mx.array([999, 500], dtype=mx.float16)
    second_times = mx.array([750], dtype=mx.float32)
    prepared_first = head.prepare_conditioning(first, first_times, dtype=mx.float16)
    prepared_second = head.prepare_conditioning(
        second.astype(mx.float32),
        second_times,
        dtype=mx.float32,
    )
    expected_first = head(noisy, first_times[1:], first)
    expected_second = head(
        noisy.astype(mx.float32), second_times, second.astype(mx.float32)
    )
    # Evaluate after preparing both schedules, including the older preparation.
    np.testing.assert_allclose(
        np.array(head.forward_prepared(noisy, prepared_first, 1)),
        np.array(expected_first),
        atol=2e-3,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        np.array(head.forward_prepared(noisy.astype(mx.float32), prepared_second, 0)),
        np.array(expected_second),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
def test_solver_preserves_overridden_head_callbacks(
    solver: Callable[..., mx.array],
) -> None:
    class CustomHead(FastDiffusionHead):
        def __init__(self) -> None:
            self.timesteps: list[int] = []

        def __call__(
            self, noisy: mx.array, timestep: mx.array, condition: mx.array
        ) -> mx.array:
            self.timesteps.append(int(timestep.item()))
            return mx.full(noisy.shape, 0.125, dtype=noisy.dtype)

        def prepare_conditioning(
            self, *args: object, **kwargs: object
        ) -> PreparedDiffusionConditioning:
            raise AssertionError("Custom head must retain its __call__ behavior")

    head = CustomHead()
    condition = mx.zeros((1, 64), dtype=mx.float16)
    actual = solver(head, condition, condition, 1.3, num_steps=2)
    expected = solver(
        lambda noisy, timestep, condition: mx.full(
            noisy.shape, 0.125, dtype=noisy.dtype
        ),
        condition,
        condition,
        1.3,
        num_steps=2,
    )
    assert head.timesteps == [999, 500]
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is unavailable")
@pytest.mark.parametrize("bits", [None, 8])
@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
def test_prepared_solver_matches_unprepared_on_metal(
    bits: int | None,
    solver: Callable[..., mx.array],
) -> None:
    # Larger batched matmuls can choose different Metal kernels than the
    # original per-timestep calls; CPU coverage alone cannot exercise that.
    with mx.stream(mx.gpu):
        head = make_head(bits=bits)
        condition = mx.random.normal((1, 64)).astype(mx.float16)
        negative = -condition
        expected = solver(
            lambda noisy, timestep, cond: head(noisy, timestep, cond),
            condition,
            negative,
            1.3,
            num_steps=10,
        )
        actual = solver(head, condition, negative, 1.3, num_steps=10)
        np.testing.assert_allclose(
            np.array(actual),
            np.array(expected),
            atol=2e-3,
            rtol=2e-3,
        )
