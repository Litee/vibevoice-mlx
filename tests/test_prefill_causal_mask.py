"""Prefill uses native causal attention without changing subsequent decoding."""

from collections.abc import Callable, Iterator

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_map
from test_semantic_audio_context import tiny_decoder

from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.model import (
    KVCache,
    VibeVoiceConfig,
    VibeVoiceModel,
    compute_rope,
)


def make_model(dtype: mx.Dtype, bits: int | None = None) -> VibeVoiceModel:
    mx.random.seed(37)
    config = VibeVoiceConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=512,
        vocab_size=16,
        diffusion_layers=0,
        eos_id=0,
        speech_start_id=1,
        speech_end_id=2,
        speech_diffusion_id=3,
    )
    model = VibeVoiceModel(config)
    model.model.update(
        tree_map(lambda value: value.astype(dtype), model.model.parameters())
    )
    if bits is not None:
        nn.quantize(
            model.model,
            bits=bits,
            group_size=64,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
    return model


def additive_mask(length: int, dtype: mx.Dtype) -> mx.array:
    return mx.triu(mx.full((length, length), float("-inf"), dtype=dtype), k=1)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "metal"])
@pytest.mark.parametrize("length", [1, 7, 129, 1025])
@pytest.mark.parametrize(
    ("dtype", "bits"),
    [(mx.float32, None), (mx.float16, None), (mx.float16, 4), (mx.float16, 8)],
    ids=["fp32", "fp16", "int4", "int8"],
)
def test_native_prefill_preserves_hidden_cache_and_following_tokens(
    device: mx.Device,
    length: int,
    dtype: mx.Dtype,
    bits: int | None,
    record_property: Callable[[str, object], None],
) -> None:
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    with mx.stream(device):
        model = make_model(dtype, bits)
        config = model.config
        lm = FastLM(model, config)
        expected_cache = KVCache(2, growth_step=16)
        actual_cache = KVCache(2, growth_step=16)
        prompt = mx.random.normal((1, length, config.hidden_size)).astype(dtype)
        cos, sin = compute_rope(mx.arange(length), config.head_dim, config.rope_theta)
        # Nonempty caches exercise prefill's public reset behavior as well.
        for cache in (expected_cache, actual_cache):
            lm.forward(prompt[:, :1], cos[:1], sin[:1], cache)
        expected = lm.prefill(
            prompt, cos, sin, additive_mask(length, dtype), expected_cache
        )
        actual = lm.prefill(prompt, cos, sin, "causal", actual_cache)
        tolerance = 3e-3 if dtype == mx.float16 else 2e-6
        max_difference = 0.0

        def assert_close(actual: mx.array, expected: mx.array) -> None:
            nonlocal max_difference
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype == dtype
            difference = mx.max(
                mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            ).item()
            max_difference = max(max_difference, difference)
            assert mx.allclose(
                actual, expected, atol=tolerance, rtol=tolerance
            ).item(), difference

        def assert_caches_match() -> None:
            for actual_values, expected_values in (
                (actual_cache.keys, expected_cache.keys),
                (actual_cache.values, expected_cache.values),
            ):
                for actual_value, expected_value in zip(actual_values, expected_values):
                    assert_close(actual_value, expected_value)

        assert_close(actual, expected)
        assert_caches_match()
        actual, expected = actual[:, -1:], expected[:, -1:]
        # Sixteen following positions also cross a cache allocation boundary.
        for position in range(length, length + 16):
            expected_token = lm.select_token(expected, speech_only=True)
            actual_token = lm.select_token(actual, speech_only=True)
            assert actual_token == expected_token
            token_embed = lm.embed_w[expected_token].reshape(1, 1, config.hidden_size)
            cos, sin = compute_rope(
                mx.array([position]), config.head_dim, config.rope_theta
            )
            expected = lm.forward(token_embed, cos, sin, expected_cache)
            actual = lm.forward(token_embed, cos, sin, actual_cache)
            assert_close(actual, expected)
        assert_caches_match()
        record_property("max_absolute_difference", max_difference)


@pytest.mark.parametrize("native", [True, False])
def test_prefill_propagates_unrelated_attention_type_errors(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    with mx.stream(mx.cpu):
        model = make_model(mx.float16)
        lm = FastLM(model, model.config)
        prompt = model.model.embed_tokens(mx.array([[4, 5, 6]]))
        cos, sin = compute_rope(mx.arange(3), 128, model.config.rope_theta)
        failure = TypeError("Unrelated attention implementation failure")
        masks = []

        def failing_attention(
            q: mx.array,
            k: mx.array,
            v: mx.array,
            *,
            scale: float,
            mask: mx.array | str,
        ) -> mx.array:
            masks.append(mask)
            raise failure

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", failing_attention)
        mask = "causal" if native else additive_mask(3, mx.float16)
        with pytest.raises(TypeError) as error:
            lm.prefill(prompt, cos, sin, mask, KVCache(2))
        assert error.value is failure
        assert len(masks) == 1
        assert masks[0] is mask


def test_generation_prefill_does_not_materialize_a_square_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = VibeVoiceConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=8,
        diffusion_layers=0,
        eos_id=0,
        speech_start_id=1,
        speech_end_id=2,
        speech_diffusion_id=3,
    )
    model = VibeVoiceModel(config)
    model.model.embed_tokens.weight = mx.zeros((8, 32), dtype=mx.float16)
    prefill = FastLM.prefill
    masks = []

    def checked_prefill(
        self: FastLM,
        embeds: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | str,
        cache: KVCache,
    ) -> mx.array:
        masks.append(mask)
        assert isinstance(mask, str) and mask == "causal"
        return prefill(self, embeds, cos, sin, mask, cache)

    monkeypatch.setattr(FastLM, "prefill", checked_prefill)
    audio, metrics = generate(
        model, [4] * 7, GenerationOptions(cfg_scale=1, max_speech_tokens=1)
    )

    assert masks == ["causal"]
    assert audio.size == 0
    assert metrics.num_text_tokens == 7
    assert metrics.num_speech_tokens == 0


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "metal"])
@pytest.mark.parametrize("bits", [None, 4, 8], ids=["fp16", "int4", "int8"])
def test_generation_with_voice_embeddings_matches_additive_prefill(
    monkeypatch: pytest.MonkeyPatch,
    device: mx.Device,
    bits: int | None,
    record_property: Callable[[str, object], None],
) -> None:
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    with mx.stream(device):
        model = make_model(mx.float16, bits)
        model.vae_decoder = tiny_decoder()
        prompt = [4, 5, 6] * 43
        voice_embeds = {
            1: mx.random.normal((1, 256)).astype(mx.float16),
            127: mx.random.normal((1, 256)).astype(mx.float16),
        }
        prefill = FastLM.prefill
        select_token = FastLM.select_token

        def additive_prefill(
            self: FastLM,
            embeds: mx.array,
            cos: mx.array,
            sin: mx.array,
            mask: mx.array | str,
            cache: KVCache,
        ) -> mx.array:
            return prefill(
                self,
                embeds,
                cos,
                sin,
                additive_mask(embeds.shape[1], mx.float16),
                cache,
            )

        results = []
        selections = []
        for implementation in (additive_prefill, prefill):
            monkeypatch.setattr(FastLM, "prefill", implementation)
            tokens = iter([3, 3, 0])
            predictions = []

            def select_for_audio(
                self: FastLM,
                hidden: mx.array,
                *,
                speech_only: bool = False,
                stop_boost: float = 0.0,
                predictions: list[int] = predictions,
                tokens: Iterator[int] = tokens,
            ) -> int:
                predictions.append(
                    select_token(
                        self, hidden, speech_only=speech_only, stop_boost=stop_boost
                    )
                )
                return next(tokens)

            monkeypatch.setattr(FastLM, "select_token", select_for_audio)
            results.append(
                generate(
                    model,
                    prompt,
                    GenerationOptions(
                        cfg_scale=2, diffusion_steps=2, max_speech_tokens=3
                    ),
                    voice_embeds=voice_embeds,
                )
            )
            selections.append(predictions)

        (expected, expected_metrics), (actual, actual_metrics) = results
        assert actual.shape == expected.shape == (6400,)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
        record_property(
            "max_absolute_difference", float(np.max(np.abs(actual - expected)))
        )
        assert selections[0] == selections[1]
        assert actual_metrics.num_text_tokens == expected_metrics.num_text_tokens == 129
        assert (
            actual_metrics.num_speech_tokens == expected_metrics.num_speech_tokens == 2
        )
