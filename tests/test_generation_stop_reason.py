"""Generation reports why autoregressive decoding stopped."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from vibevoice_mlx import e2e_pipeline
from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.generate import GenerationMetrics, GenerationOptions, generate


class FakeLM(FastLM):
    def __init__(self, tokens: list[int]):
        self.tokens = iter(tokens)
        self.embed_w = mx.zeros((4, 2), dtype=mx.float16)
        self.speech_token_ids = (0, 1, 2, 3)
        self._stop_indices = (1, 3)

    def prefill(self, *args: object) -> mx.array:
        return mx.zeros((1, 1, 2), dtype=mx.float16)

    forward = prefill

    def logits(self, hidden: mx.array, *, speech_only: bool = False) -> mx.array:
        logits = mx.full((1, 1, 4), -10.0)
        logits[0, 0, next(self.tokens)] = 10.0
        return logits


def fake_model(tokens: list[int], *, single_segment: bool = False) -> SimpleNamespace:
    config = SimpleNamespace(
        hidden_size=2,
        num_hidden_layers=0,
        head_dim=2,
        rope_theta=10000,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
        vocab_size=4,
        single_segment=single_segment,
        speech_scaling_factor=1.0,
        speech_bias_factor=0.0,
    )
    return SimpleNamespace(
        config=config,
        vae_decoder=lambda latent: mx.zeros((1, 1, 3200), dtype=mx.float16),
        acoustic_connector=lambda sample: mx.zeros((1, 1, 2), dtype=mx.float16),
        _fast_lm=FakeLM(tokens),
        _fast_diff=None,
    )


@pytest.mark.parametrize(
    ("tokens", "single_segment", "expected"),
    [
        ([3], False, "eos"),
        ([1], True, "speech_end"),
        ([0, 0, 0, 0], False, "max_generation_tokens"),
    ],
)
def test_generation_reports_control_token_stop_reason(
    tokens: list[int], single_segment: bool, expected: str
) -> None:
    _, metrics = generate(
        fake_model(tokens, single_segment=single_segment),
        [0],
        GenerationOptions(max_speech_tokens=1),
    )

    assert metrics.stop_reason == expected
    assert metrics.summary()["stop_reason"] == expected


def test_generation_reports_speech_token_limit() -> None:
    sample = mx.zeros((1, 64), dtype=mx.float16)
    with patch("vibevoice_mlx.generate.dpm_solver_2m", return_value=sample):
        _, metrics = generate(
            fake_model([2, 2]),
            [0],
            GenerationOptions(max_speech_tokens=1),
        )

    assert metrics.stop_reason == "max_speech_tokens"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("eos", None),
        ("speech_end", None),
        ("max_speech_tokens", "speech token limit"),
        ("max_generation_tokens", "generation token limit"),
    ],
)
def test_cli_warning_describes_incomplete_generation(
    reason: str, expected: str | None
) -> None:
    warning = e2e_pipeline._generation_stop_warning(reason)

    if expected is None:
        assert warning is None
    else:
        assert expected in warning


@pytest.mark.parametrize(
    ("reason", "warns"),
    [("eos", False), ("max_speech_tokens", True)],
)
def test_cli_prints_incomplete_generation_warning(
    reason: str,
    warns: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = GenerationMetrics(stop_reason=reason)
    monkeypatch.setattr(
        e2e_pipeline,
        "load_model",
        lambda *args, **kwargs: (object(), SimpleNamespace(vocab_size=4)),
    )
    monkeypatch.setattr(e2e_pipeline, "tokenize_text", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        e2e_pipeline,
        "generate",
        lambda *args, **kwargs: (mx.zeros((0,), dtype=mx.float32), metrics),
    )
    monkeypatch.setattr(
        sys, "argv", ["vibevoice-mlx", "--text", "Hello", "--no-semantic"]
    )

    e2e_pipeline.main()

    assert ("saved audio may be incomplete" in capsys.readouterr().out) is warns
