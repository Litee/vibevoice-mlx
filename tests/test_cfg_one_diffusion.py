"""Unit guidance skips only the built-in head's unused negative branch."""

from collections.abc import Callable, Iterator
from unittest.mock import patch

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
    *,
    group_size: int = 32,
    fp32_condition_projection: bool = False,
) -> FastDiffusionHead:
    mx.random.seed(19)
    config = VibeVoiceConfig(
        hidden_size=64,
        num_hidden_layers=0,
        vocab_size=8,
        diffusion_layers=2,
    )
    model = VibeVoiceModel(config)
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    if fp32_condition_projection:
        projection = model.diffusion_head.cond_proj
        projection.weight = projection.weight.astype(mx.float32)
    if bits is not None:
        nn.quantize(
            model.diffusion_head,
            group_size=group_size,
            bits=bits,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
    return FastDiffusionHead(model, config)


def solve_two_branches(
    solver: Callable[..., mx.array],
    head: FastDiffusionHead,
    condition: mx.array,
    negative: mx.array,
    cfg_scale: float,
    *,
    num_steps: int,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    # Preserve prepared conditioning while disabling only branch elimination.
    with patch.object(head, "supports_single_branch", return_value=False):
        return solver(
            head, condition, negative, cfg_scale, num_steps=num_steps, dtype=dtype
        )


def record_prepared_branches(
    monkeypatch: pytest.MonkeyPatch, head: FastDiffusionHead
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    conditions: list[tuple[int, ...]] = []
    samples: list[tuple[int, ...]] = []
    prepare = head.prepare_conditioning
    forward = head.forward_prepared

    def record_prepare(
        condition: mx.array, timesteps: mx.array, *, dtype: mx.Dtype
    ) -> PreparedDiffusionConditioning:
        conditions.append(condition.shape)
        return prepare(condition, timesteps, dtype=dtype)

    def record_forward(
        noisy: mx.array, conditioning: PreparedDiffusionConditioning, step: int
    ) -> mx.array:
        samples.append(noisy.shape)
        assert conditioning.final[step].shape[0] == noisy.shape[0]
        assert all(layer[step].shape[0] == noisy.shape[0] for layer in conditioning.layers)
        return forward(noisy, conditioning, step)

    monkeypatch.setattr(head, "prepare_conditioning", record_prepare)
    monkeypatch.setattr(head, "forward_prepared", record_forward)
    return conditions, samples


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
def test_unit_guidance_evaluates_only_one_builtin_branch(
    monkeypatch: pytest.MonkeyPatch,
    solver: Callable[..., mx.array],
) -> None:
    head = make_head()
    condition = mx.random.normal((1, 64)).astype(mx.float16)
    conditions, samples = record_prepared_branches(monkeypatch, head)
    actual = solver(head, condition, -condition, 1.0, num_steps=2)
    assert conditions == [(1, 64)]
    assert samples == [(1, 64)] * 2
    assert actual.shape == (1, 64)
    assert actual.dtype == mx.float32
    assert np.isfinite(np.array(actual)).all()


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("num_steps", [1, 2, 10, 20, 26])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_unit_guidance_matches_two_branch_solver(
    solver: Callable[..., mx.array],
    num_steps: int,
    dtype: mx.Dtype,
    bits: int | None,
) -> None:
    head = make_head(dtype, bits)
    condition = mx.random.normal((1, 64)).astype(dtype)
    negative = -condition
    expected = solve_two_branches(
        solver,
        head,
        condition,
        negative,
        1.0,
        num_steps=num_steps,
        dtype=dtype,
    )
    actual = solver(head, condition, negative, 1.0, num_steps=num_steps, dtype=dtype)
    tolerance = 2e-3 if dtype == mx.float16 else 1e-6
    assert actual.shape == expected.shape == (1, 64)
    assert actual.dtype == expected.dtype == mx.float32
    assert np.isfinite(np.array(actual)).all()
    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_fp16_unit_guidance_is_independent_of_negative_condition(
    solver: Callable[..., mx.array],
    bits: int | None,
) -> None:
    dtype = mx.float16
    head = make_head(dtype, bits)
    condition = mx.ones((1, 64), dtype=dtype)
    expected = solver(head, condition, -condition, 1.0, num_steps=10, dtype=dtype)
    # An unused NaN must not leak through a nominally cancelled negative branch.
    for negative in [
        mx.zeros_like(condition),
        mx.full(condition.shape, float("nan"), dtype=dtype),
    ]:
        actual = solver(head, condition, negative, 1.0, num_steps=10, dtype=dtype)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize(
    "cfg_scale", [-0.5, 0.0, 0.5, 1.3, np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)]
)
def test_other_guidance_scales_retain_two_branches(
    monkeypatch: pytest.MonkeyPatch,
    solver: Callable[..., mx.array],
    cfg_scale: float,
) -> None:
    head = make_head()
    condition = mx.ones((1, 64), dtype=mx.float16)
    expected = solve_two_branches(
        solver,
        head,
        condition,
        -condition,
        cfg_scale,
        num_steps=2,
    )
    conditions, samples = record_prepared_branches(monkeypatch, head)
    actual = solver(head, condition, -condition, cfg_scale, num_steps=2)
    assert conditions == [(2, 64)]
    assert samples == [(2, 64)] * 2
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
def test_mixed_parameter_head_preserves_two_branches(
    monkeypatch: pytest.MonkeyPatch,
    solver: Callable[..., mx.array],
) -> None:
    # Only the condition projection is FP32; input and all other parameters
    # remain FP16. A guard checking just input or first-projection dtype fails.
    head = make_head(fp32_condition_projection=True)
    condition = mx.ones((1, 64), dtype=mx.float16)
    expected = solve_two_branches(
        solver,
        head,
        condition,
        -condition,
        1.0,
        num_steps=2,
    )
    conditions, samples = record_prepared_branches(monkeypatch, head)
    actual = solver(head, condition, -condition, 1.0, num_steps=2)
    assert conditions == [(2, 64)]
    assert samples == [(2, 64)] * 2
    assert np.isfinite(np.array(actual)).all()
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize(
    "device",
    [
        mx.cpu,
        pytest.param(
            mx.gpu,
            marks=pytest.mark.skipif(
                not mx.metal.is_available(), reason="Metal is unavailable"
            ),
        ),
    ],
    ids=["cpu", "metal"],
)
@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("bits", [4, 8])
def test_group_size_64_quantized_heads_use_single_branch(
    monkeypatch: pytest.MonkeyPatch,
    solver: Callable[..., mx.array],
    bits: int,
    device: mx.Device,
) -> None:
    with mx.stream(device):
        head = make_head(bits=bits, group_size=64)
        condition = mx.random.normal((1, 64)).astype(mx.float16)
        expected = solve_two_branches(
            solver,
            head,
            condition,
            -condition,
            1.0,
            num_steps=10,
        )
        conditions, samples = record_prepared_branches(monkeypatch, head)
        actual = solver(head, condition, -condition, 1.0, num_steps=10)
        assert conditions == [(1, 64)]
        assert samples == [(1, 64)] * 10
        assert np.isfinite(np.array(actual)).all()
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), atol=2e-3, rtol=2e-3
        )


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [(mx.float32, mx.float32), (mx.float16, mx.float32), (mx.float32, mx.float16)],
)
def test_fp32_and_mixed_precision_preserve_original_batching(
    monkeypatch: pytest.MonkeyPatch,
    solver: Callable[..., mx.array],
    bits: int | None,
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
) -> None:
    head = make_head(weight_dtype, bits)
    condition = mx.ones((1, 64), dtype=input_dtype)
    expected = solve_two_branches(
        solver,
        head,
        condition,
        -condition,
        1.0,
        num_steps=2,
        dtype=input_dtype,
    )
    conditions, samples = record_prepared_branches(monkeypatch, head)
    actual = solver(head, condition, -condition, 1.0, num_steps=2, dtype=input_dtype)
    assert conditions == [(2, 64)]
    assert samples == [(2, 64)] * 2
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("subclass", [False, True])
def test_custom_denoisers_keep_their_two_branch_contract(
    solver: Callable[..., mx.array],
    subclass: bool,
) -> None:
    shapes: list[tuple[int, ...]] = []

    def callback(noisy: mx.array, timestep: mx.array, condition: mx.array) -> mx.array:
        shapes.extend([noisy.shape, condition.shape])
        # Custom callbacks may intentionally couple the branches.
        return noisy + mx.mean(condition, axis=0, keepdims=True)

    class CustomHead(FastDiffusionHead):
        def __init__(self) -> None:
            pass

        def __call__(
            self, noisy: mx.array, timestep: mx.array, condition: mx.array
        ) -> mx.array:
            return callback(noisy, timestep, condition)

    head = CustomHead() if subclass else callback
    condition = mx.ones((1, 64), dtype=mx.float16)
    actual = solver(head, condition, -condition, 1.0, num_steps=2)
    assert shapes == [(2, 64)] * 4
    expected = solver(
        lambda noisy, timestep, cond: noisy, condition, -condition, 1.0, num_steps=2
    )
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is unavailable")
@pytest.mark.parametrize("solver", [dpm_solver_2m, dpm_solver_sde_2m])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_unit_guidance_matches_two_branches_on_metal(
    solver: Callable[..., mx.array],
    dtype: mx.Dtype,
    bits: int | None,
) -> None:
    with mx.stream(mx.gpu):
        head = make_head(dtype, bits)
        condition = mx.random.normal((1, 64)).astype(dtype)
        expected = solve_two_branches(
            solver,
            head,
            condition,
            -condition,
            1.0,
            num_steps=10,
            dtype=dtype,
        )
        actual = solver(head, condition, -condition, 1.0, num_steps=10, dtype=dtype)
        tolerance = 2e-3 if dtype == mx.float16 else 1e-6
        assert np.isfinite(np.array(actual)).all()
        np.testing.assert_allclose(
            np.array(actual),
            np.array(expected),
            atol=tolerance,
            rtol=tolerance,
        )
