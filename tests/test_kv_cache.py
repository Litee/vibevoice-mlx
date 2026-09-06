"""Capacity-managed caches must preserve causal attention history."""

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.model import KVCache, VibeVoiceConfig, VibeVoiceModel, compute_rope


@pytest.mark.parametrize(
    ("chunk_sizes", "capacities"),
    [
        ([3, 1, 1, 11, 1], [4, 4, 8, 16, 20]),
        ([1] * 13, [4] * 4 + [8] * 4 + [12] * 4 + [16]),
    ],
)
@pytest.mark.parametrize("evaluate_each_update", [False, True])
def test_cache_append_matches_full_history(
    chunk_sizes: list[int], capacities: list[int], evaluate_each_update: bool
) -> None:
    cache = KVCache(2, growth_step=4)
    histories = [[], []]
    snapshots = []
    for size, capacity in zip(chunk_sizes, capacities, strict=True):
        for layer in range(2):
            start = sum(chunk.shape[2] for chunk in histories[layer])
            key = mx.arange(start, start + size)[None, None, :, None].astype(mx.float16)
            histories[layer].append(key)
            keys, values = cache.update(layer, key, -key)
            expected = mx.concatenate(histories[layer], axis=2)
            assert keys.shape == expected.shape
            assert cache.keys[layer].shape[2] == capacity
            assert cache.values[layer].shape[2] == capacity
            if evaluate_each_update:
                mx.eval(keys, values)
            snapshots.append((keys, values, expected))
    # Previously returned views must remain valid after later writes/growth.
    for keys, values, expected in snapshots:
        assert mx.array_equal(keys, expected).item()
        assert mx.array_equal(values, -expected).item()


def test_cache_reset_discards_previous_history() -> None:
    cache = KVCache(1, growth_step=4)
    cache.update(0, mx.ones((1, 1, 3, 2)), mx.ones((1, 1, 3, 2)))
    cache.advance(3)
    cache.reset()
    keys, values = cache.update(0, mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 2)))
    assert cache.offset == 0
    assert keys.shape == (1, 1, 1, 2)
    assert mx.array_equal(keys, values).item()
    assert mx.all(keys == 0).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_attention_matches_contiguous_history_at_production_head_size(
    dtype: mx.Dtype,
) -> None:
    mx.random.seed(7)
    cache = KVCache(1)
    key_chunks, value_chunks = [], []
    q = mx.random.normal((1, 28, 1, 128)).astype(dtype)
    for size in [255, 1, 1]:
        k = mx.random.normal((1, 4, size, 128)).astype(dtype)
        v = mx.random.normal((1, 4, size, 128)).astype(dtype)
        key_chunks.append(k)
        value_chunks.append(v)
        keys, values = cache.update(0, k, v)
        actual = mx.fast.scaled_dot_product_attention(q, keys, values, scale=128**-0.5)
        expected = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate(key_chunks, axis=2),
            mx.concatenate(value_chunks, axis=2),
            scale=128**-0.5,
        )
        assert mx.allclose(actual, expected, atol=1e-3, rtol=1e-3).item()


class ConcatenatingCache(KVCache):
    """Original storage strategy as an independent attention-history oracle."""

    def update(
        self, layer_idx: int, k: mx.array, v: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx], self.values[layer_idx] = k, v
        else:
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx], k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], v], axis=2)
        return self.keys[layer_idx], self.values[layer_idx]


@pytest.mark.parametrize("dual", [False, True])
def test_decode_matches_concatenation_across_growth_boundaries(dual: bool) -> None:
    mx.random.seed(42)
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
    model.update(tree_map(lambda w: w.astype(mx.float16), model.parameters()))
    lm = FastLM(model, config)
    caches = [KVCache(2, growth_step=4), ConcatenatingCache(2)]
    negatives = [KVCache(2, growth_step=4), ConcatenatingCache(2)]

    def rope(position: int, count: int = 1) -> tuple[mx.array, mx.array]:
        return compute_rope(mx.arange(position, position + count), 8, config.rope_theta)

    for _ in range(2):  # Prefill must clear old state when reusing a cache.
        prompt = mx.random.normal((1, 3, 32)).astype(mx.float16)
        mask = mx.triu(mx.full((3, 3), float("-inf"), dtype=mx.float16), k=1)
        for cache in caches:
            hidden = lm.prefill(prompt, *rope(0, 3), mask, cache)
            mx.eval(hidden, *cache.keys, *cache.values)
        negative = mx.random.normal((1, 1, 32)).astype(mx.float16)
        for cache in negatives:
            cache.reset()
            hidden = lm.forward(negative, *rope(0), cache)
            mx.eval(hidden, *cache.keys, *cache.values)
        for step in range(10):
            token = mx.random.normal((1, 1, 32)).astype(mx.float16)
            if dual:
                outputs = [
                    lm.forward_dual(
                        token, *rope(3 + step), cache, token, *rope(1 + step), neg
                    )
                    for cache, neg in zip(caches, negatives)
                ]
                for actual, expected in zip(*outputs):
                    assert mx.allclose(actual, expected, atol=1e-3, rtol=1e-3).item()
            else:
                actual, expected = [
                    lm.forward(token, *rope(3 + step), c) for c in caches
                ]
                assert mx.allclose(actual, expected, atol=1e-3, rtol=1e-3).item()
