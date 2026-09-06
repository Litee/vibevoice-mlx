"""Constrained projection must preserve full-vocabulary scores and decisions."""

import mlx.core as mx
import pytest

from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


def make_model(tied: bool) -> VibeVoiceModel:
    config = VibeVoiceConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=128,
        diffusion_layers=1,
        speech_start_id=91,
        speech_end_id=73,
        speech_diffusion_id=105,
        eos_id=12,
        tie_word_embeddings=tied,
        single_segment=True,
    )
    mx.random.seed(42)
    return VibeVoiceModel(config)


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_speech_logits_match_full_vocabulary(tied: bool, dtype: mx.Dtype) -> None:
    model = make_model(tied)
    lm = FastLM(model, model.config)
    hidden = mx.random.normal((1, 1, 32)).astype(dtype)
    expected = model.get_logits(hidden)[..., mx.array([12, 73, 91, 105])]

    actual = lm.logits(hidden, speech_only=True)

    assert actual.shape == (1, 1, 4)
    assert actual.dtype == dtype
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("speech_only", [True, False])
@pytest.mark.parametrize("boost", [0.0, 5.0, 20.0])
def test_token_selection_matches_masked_full_vocabulary(
    tied: bool, speech_only: bool, boost: float
) -> None:
    model = make_model(tied)
    lm = FastLM(model, model.config)
    hidden = mx.random.normal((1, 1, 32)).astype(mx.float16)
    full = model.get_logits(hidden)
    if speech_only:
        mask = mx.full(full.shape, float("-inf"), dtype=mx.float32)
        mask[..., mx.array([12, 73, 91, 105])] = 0
        full = full + mask
        full[0, 0, 73] += boost
        full[0, 0, 12] += boost
    expected = int(mx.argmax(full[0, 0]).item())

    assert (
        lm.select_token(hidden, speech_only=speech_only, stop_boost=boost) == expected
    )


def test_tied_scores_choose_lowest_allowed_token_id() -> None:
    model = make_model(True)
    lm = FastLM(model, model.config)
    assert lm.select_token(mx.zeros((1, 1, 32)), speech_only=True) == 12


def test_stop_boost_preserves_float32_precision_after_fp16_projection() -> None:
    model = make_model(True)
    weight = mx.zeros((128, 32), dtype=mx.float16)
    weight[73, 0] = 8192
    weight[105, 0] = 8200
    weight[0, 0] = 10000  # Disallowed even though its score is highest.
    model.model.embed_tokens.weight = weight
    lm = FastLM(model, model.config)
    hidden = mx.zeros((1, 1, 32), dtype=mx.float16)
    hidden[0, 0, 0] = 1

    # 8192 + 5 is below 8200 in float32; fp16 would round it to a tie.
    assert lm.select_token(hidden, speech_only=True, stop_boost=5) == 105
    assert lm.select_token(hidden, speech_only=True, stop_boost=20) == 73
    assert lm.select_token(hidden) == 0


@pytest.mark.parametrize("single_segment", [True, False])
def test_generation_maps_compact_prefill_selection_to_eos(single_segment: bool) -> None:
    model = make_model(True)
    model.config.single_segment = single_segment
    model.model.embed_tokens.weight = mx.zeros((128, 32), dtype=mx.float16)
    # Every output score ties, so constrained selection must choose EOS (12),
    # not vocabulary ID 0 or a compact score index interpreted as a token ID.
    audio, metrics = generate(
        model, [0], GenerationOptions(cfg_scale=1, max_speech_tokens=1)
    )
    assert audio.size == 0
    assert metrics.num_speech_tokens == 0
