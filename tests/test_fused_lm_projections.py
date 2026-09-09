"""Projection packing preserves the LM output, token and cache interfaces."""

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from vibevoice_mlx import fast_forward
from vibevoice_mlx.fast_forward import FastLM
from vibevoice_mlx.model import KVCache, VibeVoiceConfig, VibeVoiceModel, compute_rope


def make_model(bits: int | None = None) -> VibeVoiceModel:
    mx.random.seed(19)
    config = VibeVoiceConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        diffusion_layers=0,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
    )
    model = VibeVoiceModel(config)
    model.update(tree_map(lambda value: value.astype(mx.float16), model.parameters()))
    if bits is not None:
        nn.quantize(
            model.model,
            bits=bits,
            group_size=64 if bits == 4 else 32,
            class_predicate=lambda _, layer: isinstance(layer, nn.Linear),
        )
    return model


def assert_same(actual: mx.array, expected: mx.array) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.float16
    assert mx.allclose(actual, expected, atol=3e-3, rtol=3e-3).item()


@pytest.mark.parametrize("bits", [None, 4, 8], ids=["fp16", "int4", "int8"])
@pytest.mark.parametrize(
    "fuse_qkv,fuse_gate_up", [(True, False), (False, True), (True, True)]
)
def test_fusion_preserves_prefill_decode_tokens_and_caches(
    bits: int | None,
    fuse_qkv: bool,
    fuse_gate_up: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(bits)
    reference = make_model(bits)
    reference.load_weights(tree_flatten(model.parameters()))
    lm = FastLM(model, model.config, fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
    caches = [KVCache(2, growth_step=4), KVCache(2, growth_step=4)]
    calls: list[dict] = []
    original_mm = fast_forward._mm

    def record_mm(x: mx.array, weights: dict) -> mx.array:
        calls.append(weights)
        return original_mm(x, weights)

    monkeypatch.setattr(fast_forward, "_mm", record_mm)

    for position, count in [(0, 3), (3, 1), (4, 1)]:
        h = mx.random.normal((1, count, 64)).astype(mx.float16)
        cos, sin = compute_rope(mx.arange(position, position + count), 16, 1_000_000)
        mask = mx.triu(mx.full((count, count), -mx.inf, dtype=mx.float16), k=1)
        expected = reference.model(h, cos, sin, mask if count > 1 else None, caches[0])
        calls.clear()
        actual = (
            lm.prefill(h, cos, sin, mask, caches[1])
            if count > 1
            else lm.forward(h, cos, sin, caches[1])
        )
        # Output parity alone also passes if fusion silently stops activating.
        for layer in lm.layers:
            for key, names, enabled in (
                ("qkv", ("q", "k", "v"), fuse_qkv),
                ("gu", ("g", "u"), fuse_gate_up),
            ):
                assert (layer[key] is not None) == enabled
                if enabled:
                    assert sum(weights is layer[key] for weights in calls) == 1
                for name in names:
                    assert sum(weights is layer[name] for weights in calls) == (
                        0 if enabled else 1
                    )
        assert_same(actual, expected)
        assert lm.select_token(actual[:, -1:]) == lm.select_token(expected[:, -1:])
        assert lm.select_token(actual[:, -1:], speech_only=True) == lm.select_token(
            expected[:, -1:], speech_only=True
        )
        for source, result in zip(caches[0].keys, caches[1].keys, strict=True):
            assert_same(result, source)
        for source, result in zip(caches[0].values, caches[1].values, strict=True):
            assert_same(result, source)


@pytest.mark.parametrize("bits", [None, 4, 8], ids=["fp16", "int4", "int8"])
@pytest.mark.parametrize(
    "fuse_qkv,fuse_gate_up", [(True, False), (False, True), (True, True)]
)
def test_fused_parameters_share_backing_storage(
    bits: int | None, fuse_qkv: bool, fuse_gate_up: bool
) -> None:
    model = make_model(bits)
    lm = FastLM(model, model.config, fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
    fields = {"w": "weight"}
    if bits is not None:
        fields.update(s="scales", b="biases")

    for module, layer in zip(model.model.layers, lm.layers, strict=True):
        attention, mlp = module.self_attn, module.mlp
        for key, projections in (
            (
                "qkv",
                (
                    ("q", attention.q_proj),
                    ("k", attention.k_proj),
                    ("v", attention.v_proj),
                ),
            ),
            ("gu", (("g", mlp.gate_proj), ("u", mlp.up_proj))),
        ):
            if layer[key] is None:
                continue
            offset = 0
            for name, projection in projections:
                size = projection.weight.shape[0]
                for field, attribute in fields.items():
                    parameter = getattr(projection, attribute)
                    assert layer[name][field] is parameter
                    # NumPy exposes the evaluated MLX buffer without copying.
                    # Compare storage directly, avoiding allocator noise from
                    # other arrays or processes in a memory-counter assertion.
                    backing = np.asarray(layer[key][field])
                    actual = np.asarray(parameter)
                    expected = backing[offset : offset + size]
                    assert np.shares_memory(actual, backing)
                    assert actual.shape == expected.shape
                    assert actual.strides == expected.strides
                    assert actual.ctypes.data == expected.ctypes.data
                offset += size


@pytest.mark.parametrize("bits", [None, 4, 8], ids=["fp16", "int4", "int8"])
@pytest.mark.parametrize(
    "fuse_qkv,fuse_gate_up", [(True, False), (False, True), (True, True)]
)
def test_fusion_preserves_dual_decode_with_different_cache_lengths(
    bits: int | None, fuse_qkv: bool, fuse_gate_up: bool
) -> None:
    model = make_model(bits)
    reference = make_model(bits)
    reference.load_weights(tree_flatten(model.parameters()))
    fused = FastLM(model, model.config, fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
    separate = FastLM(reference, reference.config, fuse_qkv=False, fuse_gate_up=False)
    fused_caches = [KVCache(2, growth_step=4), KVCache(2, growth_step=4)]
    separate_caches = [KVCache(2, growth_step=4), KVCache(2, growth_step=4)]
    for index, count in enumerate((3, 1)):
        h = mx.random.normal((1, count, 64)).astype(mx.float16)
        cos, sin = compute_rope(mx.arange(count), 16, 1_000_000)
        fused.prefill(h, cos, sin, "causal", fused_caches[index])
        separate.prefill(h, cos, sin, "causal", separate_caches[index])
    for step in range(3):
        main, negative = [
            mx.random.normal((1, 1, 64)).astype(mx.float16) for _ in range(2)
        ]
        main_rope = compute_rope(mx.array([3 + step]), 16, 1_000_000)
        negative_rope = compute_rope(mx.array([1 + step]), 16, 1_000_000)
        actual = fused.forward_dual(
            main, *main_rope, fused_caches[0], negative, *negative_rope, fused_caches[1]
        )
        expected = separate.forward_dual(
            main,
            *main_rope,
            separate_caches[0],
            negative,
            *negative_rope,
            separate_caches[1],
        )
        for result, source in zip(actual, expected, strict=True):
            assert_same(result, source)
            assert fused.select_token(
                result, speech_only=True
            ) == separate.select_token(source, speech_only=True)
        for fused_cache, separate_cache in zip(
            fused_caches, separate_caches, strict=True
        ):
            for result, source in zip(
                fused_cache.keys, separate_cache.keys, strict=True
            ):
                assert_same(result, source)
            for result, source in zip(
                fused_cache.values, separate_cache.values, strict=True
            ):
                assert_same(result, source)


@pytest.mark.parametrize("bits", [None, 4, 8], ids=["fp16", "int4", "int8"])
def test_packed_model_preserves_checkpoint_schema_and_reload_output(
    bits: int | None,
) -> None:
    model = make_model(bits)
    reference = make_model(bits)
    original = dict(tree_flatten(reference.parameters()))
    FastLM(model, model.config)
    packed = dict(tree_flatten(model.parameters()))
    assert packed.keys() == original.keys()
    for name, value in packed.items():
        assert value.shape == original[name].shape
        assert value.dtype == original[name].dtype
        assert mx.array_equal(value, original[name]).item(), name

    # A fresh model must accept packed parameters through its unchanged schema.
    reference.load_weights(list(packed.items()))
    h = mx.random.normal((1, 3, 64)).astype(mx.float16)
    cos, sin = compute_rope(mx.arange(3), 16, 1_000_000)
    assert_same(
        model.model(h, cos, sin, "causal"), reference.model(h, cos, sin, "causal")
    )

    # Reloading the original model remains supported. As with any FastLM weight
    # snapshot, construct a new fast path after loading new model parameters.
    replacement = {
        name: value * 0.5 if value.dtype == mx.float16 else value
        for name, value in original.items()
    }
    model.load_weights(list(replacement.items()))
    reference.load_weights(list(replacement.items()))
    lm = FastLM(model, model.config)
    assert_same(
        lm.prefill(h, cos, sin, "causal", KVCache(2)),
        reference.model(h, cos, sin, "causal"),
    )


@pytest.mark.parametrize("mismatch", ["format", "bits", "group_size", "dtype", "bias"])
def test_mixed_projection_formats_and_biases_preserve_output(mismatch: str) -> None:
    models = [make_model(), make_model()]
    for model in models:
        attention = model.model.layers[0].self_attn
        mlp = model.model.layers[0].mlp
        if mismatch in ("format", "bits", "group_size"):
            attention.q_proj = nn.QuantizedLinear.from_linear(
                attention.q_proj, bits=4, group_size=64
            )
            mlp.gate_proj = nn.QuantizedLinear.from_linear(
                mlp.gate_proj, bits=4, group_size=64
            )
            if mismatch != "format":
                bits = 8 if mismatch == "bits" else 4
                group_size = 32 if mismatch == "group_size" else 64
                attention.k_proj = nn.QuantizedLinear.from_linear(
                    attention.k_proj, bits=bits, group_size=group_size
                )
                attention.v_proj = nn.QuantizedLinear.from_linear(
                    attention.v_proj, bits=bits, group_size=group_size
                )
                mlp.up_proj = nn.QuantizedLinear.from_linear(
                    mlp.up_proj, bits=bits, group_size=group_size
                )
        elif mismatch == "dtype":
            attention.q_proj.weight = attention.q_proj.weight.astype(mx.float32)
            mlp.gate_proj.weight = mlp.gate_proj.weight.astype(mx.float32)
        else:
            del attention.k_proj.bias
            attention.q_proj.bias = mx.full((64,), 0.125, dtype=mx.float16)
            attention.v_proj.bias = mx.full((32,), -0.25, dtype=mx.float16)
            mlp.gate_proj.bias = mx.full((128,), 0.25, dtype=mx.float16)
    lm = FastLM(models[0], models[0].config)
    h = mx.random.normal((1, 3, 64)).astype(mx.float16)
    cos, sin = compute_rope(mx.arange(3), 16, 1_000_000)
    actual = lm.prefill(h, cos, sin, "causal", KVCache(2))
    expected = models[1].model(h, cos, sin, "causal")
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=3e-3, rtol=3e-3).item()
