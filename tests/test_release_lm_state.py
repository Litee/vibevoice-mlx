"""Completed language-model state is released before final audio decoding."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.model import KVCache


class CacheRecordingLM:
    def __init__(self) -> None:
        self.embed_w = mx.zeros((4, 2), dtype=mx.float16)
        self.speech_token_ids = (0, 1, 2, 3)
        self._stop_indices = (1, 3)
        self.tokens = iter([2, 3])
        self.caches: list[KVCache] = []

    def _record(self, cache: KVCache) -> mx.array:
        if cache not in self.caches:
            self.caches.append(cache)
        value = mx.ones((1, 1, 1, 2), dtype=mx.float16)
        keys, _ = cache.update(0, value, value)
        return keys[:, :, -1, :]

    def prefill(self, *args: object) -> mx.array:
        return self._record(args[-1])

    def forward(self, *args: object) -> mx.array:
        return self._record(args[-1])

    def forward_dual(self, *args: object) -> tuple[mx.array, mx.array]:
        return self._record(args[3]), self._record(args[7])

    def select_token(self, *args: object, **kwargs: object) -> int:
        return next(self.tokens)


@pytest.mark.parametrize("cfg_scale", [1.0, 1.3])
def test_generate_releases_lm_caches_before_final_decode(cfg_scale: float) -> None:
    fast_lm = CacheRecordingLM()
    config = SimpleNamespace(
        hidden_size=2,
        num_hidden_layers=1,
        head_dim=2,
        rope_theta=10000,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
        vocab_size=4,
        single_segment=False,
        speech_scaling_factor=1.0,
        speech_bias_factor=0.0,
    )

    expected = np.arange(3200, dtype=np.float32)

    def decode(latent: mx.array) -> mx.array:
        assert len(fast_lm.caches) == (2 if cfg_scale > 1 else 1)
        for cache in fast_lm.caches:
            assert cache.offset == 0
            assert all(value is None for value in cache.keys)
            assert all(value is None for value in cache.values)
        np.testing.assert_array_equal(np.array(latent)[0, :2, 0], [1, 1])
        return mx.array(expected).reshape(1, 1, -1)

    model = SimpleNamespace(
        config=config,
        vae_decoder=decode,
        acoustic_connector=lambda sample: mx.zeros((1, 1, 2), dtype=mx.float16),
        _fast_lm=fast_lm,
        _fast_diff=None,
    )

    def solve(
        _: object, condition: mx.array, *args: object, **kwargs: object
    ) -> mx.array:
        return mx.pad(condition.reshape(1, 2), [(0, 0), (0, 62)])

    with patch("vibevoice_mlx.generate.dpm_solver_2m", side_effect=solve):
        audio, metrics = generate(
            model,
            [0],
            GenerationOptions(cfg_scale=cfg_scale, max_speech_tokens=2),
        )

    assert metrics.num_speech_tokens == 1
    np.testing.assert_array_equal(audio, expected)
