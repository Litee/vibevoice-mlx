"""Chunked prefill preserves full causal history and generation inputs."""

import importlib
from collections.abc import Callable

import mlx.core as mx
import pytest
from test_prefill_causal_mask import make_model

from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.model import KVCache, compute_rope


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "metal"])
@pytest.mark.parametrize(
    ("dtype", "bits"),
    [(mx.float32, None), (mx.float16, None), (mx.float16, 4), (mx.float16, 8)],
    ids=["fp32", "fp16", "int4", "int8"],
)
@pytest.mark.parametrize(
    ("length", "chunk_size"),
    [(1, 1), (7, 3), (129, 1), (129, 32), (129, 128), (513, 512)],
)
def test_chunks_preserve_hidden_cache_and_following_tokens(
    device: mx.Device,
    dtype: mx.Dtype,
    bits: int | None,
    chunk_size: int,
    length: int,
    record_property: Callable[[str, object], None],
) -> None:
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    with mx.stream(device):
        model = make_model(dtype, bits)
        lm = FastLM(model, model.config)
        embeds = mx.random.normal((1, length, lm.H)).astype(dtype)
        cos, sin = compute_rope(mx.arange(length + 8), lm.HD, lm.rope_theta)
        expected_cache, actual_cache = KVCache(lm.NL, 16), KVCache(lm.NL, 16)
        expected = lm.prefill(
            embeds, cos[:length], sin[:length], "causal", expected_cache
        )
        pieces = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            hidden = lm.prefill_chunk(
                embeds[:, start:end], cos[start:end], sin[start:end], actual_cache
            )
            mx.eval(hidden, *actual_cache.keys, *actual_cache.values)
            pieces.append(hidden)
        actual = mx.concatenate(pieces, axis=1)
        # Metal selects different matmul kernels for batched and single queries.
        tolerance = 4e-3 if dtype == mx.float16 or device == mx.gpu else 3e-6
        maximum = 0.0

        def assert_close(
            actual: mx.array, expected: mx.array, *, decode: bool = False
        ) -> None:
            nonlocal maximum
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype == dtype
            error = mx.max(
                mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            ).item()
            maximum = max(maximum, error)
            # Recurrent fp16 decode can accumulate several rounding steps.
            bound = 1e-2 if decode and dtype == mx.float16 else tolerance
            assert mx.allclose(actual, expected, atol=bound, rtol=bound).item(), error

        assert_close(actual, expected)
        actual, expected = actual[:, -1:], expected[:, -1:]
        for position in range(length, length + 8):
            expected_token = lm.select_token(expected, speech_only=True)
            assert lm.select_token(actual, speech_only=True) == expected_token
            embed = lm.embed_w[expected_token].reshape(1, 1, lm.H)
            expected = lm.forward(
                embed,
                cos[position : position + 1],
                sin[position : position + 1],
                expected_cache,
            )
            actual = lm.forward(
                embed,
                cos[position : position + 1],
                sin[position : position + 1],
                actual_cache,
            )
            assert_close(actual, expected, decode=True)
        for expected_values, actual_values in (
            (expected_cache.keys, actual_cache.keys),
            (expected_cache.values, actual_cache.values),
        ):
            for expected_value, actual_value in zip(
                expected_values, actual_values, strict=True
            ):
                assert_close(actual_value, expected_value)
        record_property("max_absolute_difference", maximum)


def test_generation_chunks_preserve_voice_positions_and_final_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("vibevoice_mlx.generate")
    with mx.stream(mx.cpu):
        model = make_model(mx.float16)
        lm = FastLM(model, model.config)
        ids = [4, 5, 6] * 5
        voices = {
            pos: mx.random.normal((1, lm.H)).astype(mx.float16)
            for pos in (0, 3, 4, 7, 8, 14)
        }
        full_embeds = mx.stack(
            [
                voices[pos].reshape(lm.H) if pos in voices else lm.embed_w[token]
                for pos, token in enumerate(ids)
            ]
        )[None]
        cos, sin = compute_rope(mx.arange(len(ids)), lm.HD, lm.rope_theta)
        expected_cache = KVCache(lm.NL)
        expected = lm.prefill(full_embeds, cos, sin, "causal", expected_cache)[:, -1:]
        actual_chunks = []
        prefill_chunk = FastLM.prefill_chunk

        def checked_chunk(
            self: FastLM, embeds: mx.array, cos: mx.array, sin: mx.array, cache: KVCache
        ) -> mx.array:
            actual_chunks.append(embeds)
            return prefill_chunk(self, embeds, cos, sin, cache)

        def stop_at_selection(self: FastLM, hidden: mx.array, **kwargs: object) -> int:
            assert mx.allclose(hidden, expected, atol=4e-3, rtol=4e-3).item()
            return model.config.eos_id

        monkeypatch.setattr(module, "_LM_PREFILL_CHUNK_TOKENS", 4)
        monkeypatch.setattr(FastLM, "prefill_chunk", checked_chunk)
        monkeypatch.setattr(FastLM, "select_token", stop_at_selection)
        audio, metrics = generate(
            model,
            ids,
            GenerationOptions(cfg_scale=1, max_speech_tokens=1),
            voice_embeds=voices,
        )
        assert [chunk.shape[1] for chunk in actual_chunks] == [4, 4, 4, 3]
        assert mx.array_equal(mx.concatenate(actual_chunks, axis=1), full_embeds).item()
        assert audio.size == 0
        assert metrics.num_text_tokens == len(ids)
        assert metrics.num_speech_tokens == 0
