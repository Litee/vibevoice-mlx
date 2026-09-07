"""Rotary embeddings must not widen the autoregressive model dtype."""

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_map

from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.model import (
    KVCache,
    VibeVoiceConfig,
    VibeVoiceModel,
    apply_rope,
    compute_rope,
)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_apply_rope_preserves_input_dtype(dtype: mx.Dtype) -> None:
    x = mx.arange(4 * 3 * 128).reshape(1, 4, 3, 128).astype(dtype) / 100
    positions = mx.array([0, 257, 32767], dtype=mx.float32)
    cos, sin = compute_rope(positions, head_dim=128, rope_theta=1_000_000.0)

    actual = apply_rope(x, cos, sin)

    assert cos.dtype == sin.dtype == mx.float32
    assert actual.dtype == dtype
    assert actual.shape == x.shape
    assert mx.all(mx.isfinite(actual)).item()


def test_float16_rope_rounds_the_float32_rotation_once() -> None:
    x = mx.random.normal((1, 4, 3, 128)).astype(mx.float16)
    positions = mx.array([0, 257, 32767], dtype=mx.float32)
    cos, sin = compute_rope(positions, head_dim=128, rope_theta=1_000_000.0)

    actual = apply_rope(x, cos, sin)
    x32 = x.astype(mx.float32)
    x1, x2 = x32[..., :64], x32[..., 64:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    expected = (x32 * cos[None, None, :, :] + rotated * sin[None, None, :, :]).astype(
        mx.float16
    )

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("quantized", [False, True], ids=["fp16", "int8"])
def test_float16_model_and_fast_path_keep_hidden_and_cache_dtypes(
    quantized: bool,
) -> None:
    mx.random.seed(7)
    config = VibeVoiceConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=64,
        diffusion_layers=0,
    )
    model = VibeVoiceModel(config)
    model.update(tree_map(lambda value: value.astype(mx.float16), model.parameters()))
    if quantized:
        nn.quantize(
            model.model,
            bits=8,
            group_size=32,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
    fast_lm = FastLM(model, config)
    model_cache = KVCache(config.num_hidden_layers, growth_step=4)
    fast_cache = KVCache(config.num_hidden_layers, growth_step=4)

    prompt = mx.random.normal((1, 3, config.hidden_size)).astype(mx.float16)
    positions = mx.arange(3, dtype=mx.float32)
    cos, sin = compute_rope(positions, config.head_dim, config.rope_theta)
    mask = mx.triu(mx.full((3, 3), float("-inf"), dtype=mx.float16), k=1)
    expected = model.model(prompt, cos, sin, mask, model_cache)
    actual = fast_lm.prefill(prompt, cos, sin, mask, fast_cache)
    mx.eval(
        expected,
        actual,
        *model_cache.keys,
        *model_cache.values,
        *fast_cache.keys,
        *fast_cache.values,
    )

    assert expected.dtype == actual.dtype == mx.float16
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3).item()

    for cache in (model_cache, fast_cache):
        assert all(
            value is not None and value.dtype == mx.float16 for value in cache.keys
        )
        assert all(
            value is not None and value.dtype == mx.float16 for value in cache.values
        )

    token = mx.random.normal((1, 1, config.hidden_size)).astype(mx.float16)
    cos, sin = compute_rope(mx.array([3]), config.head_dim, config.rope_theta)
    expected = model.model(token, cos, sin, cache=model_cache)
    actual = fast_lm.forward(token, cos, sin, fast_cache)
    mx.eval(expected, actual, *model_cache.keys, *fast_cache.keys)

    assert expected.dtype == actual.dtype == mx.float16
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3).item()

    negative_cache = KVCache(config.num_hidden_layers, growth_step=4)
    negative = fast_lm.forward(
        token,
        *compute_rope(mx.array([0]), config.head_dim, config.rope_theta),
        negative_cache,
    )
    main, negative = fast_lm.forward_dual(
        token,
        *compute_rope(mx.array([4]), config.head_dim, config.rope_theta),
        fast_cache,
        token,
        *compute_rope(mx.array([1]), config.head_dim, config.rope_theta),
        negative_cache,
    )
    mx.eval(main, negative, *fast_cache.keys, *negative_cache.keys)

    assert main.dtype == negative.dtype == mx.float16
    assert fast_cache.keys[0].shape[2] == 8
    for cache in (fast_cache, negative_cache):
        assert all(
            value is not None and value.dtype == mx.float16 for value in cache.keys
        )
        assert all(
            value is not None and value.dtype == mx.float16 for value in cache.values
        )
