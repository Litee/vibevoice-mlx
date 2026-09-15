"""Encode-only mode loads only the acoustic encoder and connector."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx.utils import tree_flatten, tree_map

from vibevoice_mlx import e2e_pipeline, load_weights, vae_encoder
from vibevoice_mlx.load_weights import load_model, load_voice_encoder
from vibevoice_mlx.model import Connector, VibeVoiceConfig, VibeVoiceModel


def acoustic_encoder_weights(prefix: str = "acoustic_encoder.") -> dict[str, mx.array]:
    """Return a complete small acoustic encoder checkpoint."""
    rng = np.random.RandomState(71)

    def weight(*shape: int) -> mx.array:
        return mx.array(rng.normal(0, 0.1, shape).astype(np.float16))

    weights = {
        prefix + "downsample_layers.0.0.conv.conv.weight": weight(2, 1, 7),
        prefix + "downsample_layers.0.0.conv.conv.bias": weight(2),
        prefix + "head.conv.conv.weight": weight(4, 2, 7),
        prefix + "head.conv.conv.bias": weight(4),
    }
    for stage, depth in enumerate(vae_encoder.DEPTHS):
        for block in range(depth):
            block_prefix = f"{prefix}stages.{stage}.{block}."
            weights.update(
                {
                    block_prefix + "norm.weight": mx.ones((2,), dtype=mx.float16),
                    block_prefix + "mixer.conv.conv.conv.weight": weight(2, 1, 7),
                    block_prefix + "mixer.conv.conv.conv.bias": weight(2),
                    block_prefix + "gamma": weight(2),
                    block_prefix + "ffn_norm.weight": mx.ones((2,), dtype=mx.float16),
                    block_prefix + "ffn.linear1.weight": weight(4, 2),
                    block_prefix + "ffn.linear1.bias": weight(4),
                    block_prefix + "ffn.linear2.weight": weight(2, 4),
                    block_prefix + "ffn.linear2.bias": weight(2),
                    block_prefix + "ffn_gamma": weight(2),
                }
            )
    for stage, ratio in enumerate(vae_encoder.RATIOS, start=1):
        weights[prefix + f"downsample_layers.{stage}.0.conv.conv.weight"] = weight(
            2, 2, 2 * ratio
        )
        weights[prefix + f"downsample_layers.{stage}.0.conv.conv.bias"] = weight(2)
    return weights


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    config = VibeVoiceConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=64,
        vocab_size=16,
        vae_dim=4,
        diffusion_layers=0,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
    )
    source = VibeVoiceModel(config)
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            **dict(tree_flatten(source.parameters())),
            **tiny_vae_weights(config.vae_dim),
            **acoustic_encoder_weights(),
        },
    )
    return tmp_path


def test_lightweight_loader_matches_full_loader_exactly(checkpoint: Path) -> None:
    full, full_config = load_model(str(checkpoint))
    lightweight, lightweight_config = load_voice_encoder(str(checkpoint))

    assert lightweight_config == full_config
    expected_connector = dict(tree_flatten(full.acoustic_connector.parameters()))
    actual_connector = dict(tree_flatten(lightweight.acoustic_connector.parameters()))
    assert actual_connector.keys() == expected_connector.keys()
    for name, expected in expected_connector.items():
        np.testing.assert_array_equal(
            np.array(actual_connector[name]), np.array(expected)
        )
    expected_encoder = dict(tree_flatten(full._encoder_weights))
    actual_encoder = dict(tree_flatten(lightweight._encoder_weights))
    assert actual_encoder.keys() == expected_encoder.keys()
    for name, expected in expected_encoder.items():
        np.testing.assert_array_equal(
            np.array(actual_encoder[name]), np.array(expected)
        )

    wav = np.random.RandomState(72).normal(0, 0.1, 3200).astype(np.float32)
    expected_embeds = e2e_pipeline.encode_voice_reference(
        wav, 1, full, full_config, str(checkpoint)
    )
    actual_embeds = e2e_pipeline.encode_voice_reference(
        wav, 1, lightweight, lightweight_config, str(checkpoint)
    )
    np.testing.assert_array_equal(actual_embeds, expected_embeds)


def test_lightweight_loader_does_not_construct_full_model(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_full_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("encode-only loading must not construct VibeVoiceModel")

    monkeypatch.setattr(load_weights, "VibeVoiceModel", reject_full_model)

    lightweight, _ = load_voice_encoder(str(checkpoint))

    assert lightweight._encoder_weights
    assert lightweight.acoustic_connector is not None


def test_hf_loader_preserves_scalars_and_connector_execution_dtype(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Connector(4, 64)
    source.update(
        tree_map(lambda value: value.astype(mx.bfloat16), source.parameters())
    )
    raw = {
        "model.language_model.norm.weight": mx.ones((64,)),
        "model.speech_scaling_factor": mx.array(0.25),
        "model.speech_bias_factor": mx.array(-0.125),
        **{
            "model.acoustic_connector." + name: value
            for name, value in tree_flatten(source.parameters())
        },
        **acoustic_encoder_weights("model.acoustic_tokenizer.encoder."),
    }
    monkeypatch.setattr(load_weights, "_load_safetensors", lambda *_: raw)

    model, config = load_voice_encoder(str(checkpoint))

    assert config.speech_scaling_factor == 0.25
    assert config.speech_bias_factor == -0.125
    for name, value in tree_flatten(model.acoustic_connector.parameters()):
        assert value.dtype == mx.float16, name


def test_indexed_loader_skips_shards_without_voice_tensors(
    checkpoint: Path,
) -> None:
    source = mx.load(str(checkpoint / "model.safetensors"))
    voice_prefixes = ("acoustic_connector.", "acoustic_encoder.")
    voice_weights = {
        name: value for name, value in source.items() if name.startswith(voice_prefixes)
    }
    mx.save_safetensors(str(checkpoint / "voice.safetensors"), voice_weights)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **dict.fromkeys(voice_weights, "voice.safetensors"),
                    "model.layers.0.self_attn.q_proj.weight": "missing.safetensors",
                }
            }
        )
    )

    model, _ = load_voice_encoder(str(checkpoint))

    assert model._encoder_weights


@pytest.mark.parametrize(
    "prefix",
    ["acoustic_encoder.", "model.acoustic_tokenizer.encoder."],
)
def test_voice_reference_fallback_loads_only_indexed_encoder_shards(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    raw_encoder = acoustic_encoder_weights(prefix)
    mx.save_safetensors(str(checkpoint / "voice.safetensors"), raw_encoder)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **dict.fromkeys(raw_encoder, "voice.safetensors"),
                    "model.layers.0.self_attn.q_proj.weight": "missing.safetensors",
                }
            }
        )
    )
    config = SimpleNamespace(speech_bias_factor=-0.05, speech_scaling_factor=0.2)
    connector = Connector(4, 64)
    direct_model = SimpleNamespace(
        _encoder_weights=vae_encoder.load_vae_encoder_weights(raw_encoder),
        acoustic_connector=connector,
    )
    fallback_model = SimpleNamespace(
        _encoder_weights=None,
        acoustic_connector=connector,
    )
    wav = np.random.RandomState(73).normal(0, 0.1, 3200).astype(np.float32)
    expected = e2e_pipeline.encode_voice_reference(
        wav, 1, direct_model, config, "unused"
    )
    monkeypatch.setattr(
        e2e_pipeline.encode_voice_reference, "_enc_cache", {}, raising=False
    )

    actual = e2e_pipeline.encode_voice_reference(
        wav, 1, fallback_model, config, str(checkpoint)
    )
    np.testing.assert_array_equal(actual, expected)

    (checkpoint / "voice.safetensors").unlink()
    cached = e2e_pipeline.encode_voice_reference(
        wav, 1, fallback_model, config, str(checkpoint)
    )
    np.testing.assert_array_equal(cached, expected)


def test_voice_reference_fallback_preserves_missing_encoder_error(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "missing.safetensors"
                }
            }
        )
    )
    model = SimpleNamespace(_encoder_weights=None)
    config = SimpleNamespace(speech_bias_factor=-0.05, speech_scaling_factor=0.2)
    monkeypatch.setattr(
        e2e_pipeline.encode_voice_reference, "_enc_cache", {}, raising=False
    )

    with pytest.raises(RuntimeError, match="No acoustic encoder weights found"):
        e2e_pipeline.encode_voice_reference(
            np.zeros(3200, dtype=np.float32),
            1,
            model,
            config,
            str(checkpoint),
        )


def test_loader_distinguishes_absent_from_partial_encoder(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = mx.load(str(checkpoint / "model.safetensors"))
    connector = {
        name: value
        for name, value in source.items()
        if name.startswith("acoustic_connector.")
    }
    monkeypatch.setattr(load_weights, "_load_safetensors", lambda *_: connector)
    with pytest.raises(RuntimeError, match="No acoustic encoder weights found"):
        load_voice_encoder(str(checkpoint))

    partial = {
        name: value
        for name, value in source.items()
        if name.startswith(("acoustic_connector.", "acoustic_encoder."))
    }
    missing = "acoustic_encoder.head.conv.conv.bias"
    partial.pop(missing)
    monkeypatch.setattr(load_weights, "_load_safetensors", lambda *_: partial)
    with pytest.raises(KeyError, match=missing):
        load_voice_encoder(str(checkpoint))


def test_cli_encode_only_uses_lightweight_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = object()
    config = object()
    calls: list[str] = []

    def load_voice(model_id: str) -> tuple[object, object]:
        calls.append(model_id)
        return model, config

    def reject_full_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("encode-only CLI must not call load_model")

    monkeypatch.setattr(e2e_pipeline, "load_voice_encoder", load_voice)
    monkeypatch.setattr(e2e_pipeline, "load_model", reject_full_loader)
    monkeypatch.setattr(
        e2e_pipeline, "_load_and_resample", lambda _: np.zeros(3200, dtype=np.float32)
    )
    monkeypatch.setattr(
        e2e_pipeline,
        "encode_voice_reference",
        lambda *args: np.zeros((1, 64), dtype=np.float16),
    )
    saved: list[str] = []
    monkeypatch.setattr(e2e_pipeline, "save_voice", lambda path, _: saved.append(path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "vibevoice-mlx",
            "--model",
            "local-model",
            "--ref-audio",
            "voice.wav",
            "--save-voice",
            str(tmp_path / "voice.safetensors"),
        ],
    )

    e2e_pipeline.main()

    assert calls == ["local-model"]
    assert saved == [str(tmp_path / "voice.safetensors")]
