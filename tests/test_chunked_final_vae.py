"""Generation bounds final decoding while preserving the batch waveform."""

from collections.abc import Callable, Iterator
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_semantic_audio_context import FakeLM, generation, tiny_decoder

from vibevoice_mlx.model import VAEDecoder
from vibevoice_mlx.streaming_vae import StreamingVAEDecoder


@pytest.fixture(params=["cpu", "metal"])
def device(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.param == "metal" and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    with mx.stream(mx.cpu if request.param == "cpu" else mx.gpu):
        yield


def make_model() -> SimpleNamespace:
    config = SimpleNamespace(
        hidden_size=2,
        num_hidden_layers=0,
        head_dim=2,
        rope_theta=10000,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
        single_segment=False,
        speech_scaling_factor=1.0,
        speech_bias_factor=0.0,
    )
    return SimpleNamespace(
        config=config,
        vae_decoder=tiny_decoder(),
        _fast_diff=None,
        acoustic_connector=lambda sample: mx.zeros((1, 1, 2), dtype=mx.float16),
    )


def run_generation(
    model: SimpleNamespace,
    samples: mx.array,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop: str = "eos",
    trim: bool = False,
) -> tuple[np.ndarray, generation.GenerationMetrics]:
    frames = samples.shape[0]
    model.config.single_segment = stop == "single_segment"
    tokens = (
        [2] * frames + [1, 2, 3] if stop == "single_segment" else [2] * frames + [3]
    )
    if stop == "cap":
        tokens = [2] * (frames + 2) + [3]
    elif stop == "segments":
        tokens = [2] * 31 + [1, 0] + [2] * (frames - 31) + [3]
    model._fast_lm = FakeLM(tokens)
    sequence = iter(samples[i : i + 1] for i in range(frames))
    monkeypatch.setattr(generation, "dpm_solver_2m", lambda *a, **kw: next(sequence))
    return generation.generate(
        model,
        [0],
        generation.GenerationOptions(
            cfg_scale=1,
            max_speech_tokens=frames if stop == "cap" else frames + 1,
            trim_trailing_silence=trim,
        ),
    )


def test_long_generation_bounds_decoder_input(
    device: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model()
    samples = mx.array(np.random.RandomState(41).normal(size=(65, 64)), mx.float32)
    expected = np.array(model.vae_decoder(samples.T[None].astype(mx.float16))).reshape(
        -1
    )
    lengths: list[int] = []
    batch_decode = VAEDecoder.__call__
    stream_decode = StreamingVAEDecoder.__call__

    def batch(decoder: VAEDecoder, latent: mx.array) -> mx.array:
        lengths.append(latent.shape[-1])
        assert latent.shape[-1] <= 32, "Final decoding must bound activation memory"
        return batch_decode(decoder, latent)

    def stream(decoder: StreamingVAEDecoder, latent: mx.array) -> mx.array:
        lengths.append(latent.shape[-1])
        assert latent.shape[-1] <= 32, "Final decoding must bound activation memory"
        return stream_decode(decoder, latent)

    monkeypatch.setattr(VAEDecoder, "__call__", batch)
    monkeypatch.setattr(StreamingVAEDecoder, "__call__", stream)
    actual, metrics = run_generation(model, samples, monkeypatch)
    assert sum(lengths) == 65
    assert actual.shape == (65 * 3200,)
    assert actual.dtype == np.float32
    assert metrics.audio_samples == len(actual)
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("frames", [0, 1, 31, 32, 33, 65])
def test_generation_matches_batch_at_chunk_boundaries(
    device: None,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    frames: int,
) -> None:
    model = make_model()
    samples = mx.array(np.random.RandomState(43).normal(size=(frames, 64)), mx.float32)
    expected = (
        np.array(model.vae_decoder(samples.T[None].astype(mx.float16))).reshape(-1)
        if frames
        else np.zeros(0, dtype=np.float16)
    ).astype(np.float32)
    actual, metrics = run_generation(model, samples, monkeypatch)
    assert actual.shape == (frames * 3200,)
    assert actual.dtype == np.float32
    assert metrics.num_speech_tokens == frames
    assert metrics.audio_samples == len(actual)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
    record_property(
        "max_abs_error", float(np.max(np.abs(actual - expected), initial=0))
    )
    record_property("unequal_samples", int(np.count_nonzero(actual != expected)))


@pytest.mark.parametrize("stop", ["eos", "cap", "single_segment", "segments"])
def test_chunked_generation_preserves_stops_and_repeated_calls(
    device: None, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    model = make_model()
    samples = mx.array(np.random.RandomState(47).normal(size=(33, 64)), mx.float32)
    expected = np.array(model.vae_decoder(samples.T[None].astype(mx.float16))).reshape(
        -1
    )
    outputs = []
    for _ in range(2):
        actual, metrics = run_generation(model, samples, monkeypatch, stop=stop)
        assert metrics.num_speech_tokens == 33
        assert metrics.stop_reason == {
            "cap": "max_speech_tokens",
            "single_segment": "speech_end",
        }.get(stop, "eos")
        assert actual.shape == (33 * 3200,)
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
        outputs.append(actual)
    np.testing.assert_array_equal(*outputs)


def repeat_sample_decoder() -> VAEDecoder:
    """Exact path: repeat each latent's first channel for 3200 samples."""
    decoder = tiny_decoder()
    decoder.init_conv_w = mx.zeros((2, 64, 1), dtype=mx.float16)
    decoder.init_conv_w[0, 0, 0] = 1
    decoder.head_w = mx.zeros((1, 2, 1), dtype=mx.float16)
    decoder.head_w[0, 0, 0] = 1
    for stage in decoder.stages:
        for block in stage:
            block["gamma"] = mx.zeros_like(block["gamma"])
            block["ffn_gamma"] = mx.zeros_like(block["ffn_gamma"])
    for index, (weight, bias, stride) in enumerate(decoder.upsample_convs):
        weight = mx.zeros_like(weight)
        weight[0, 0, :stride] = 1
        decoder.upsample_convs[index] = (weight, bias, stride)
    return decoder


def test_chunked_generation_keeps_samples_on_both_sides_of_boundary(
    device: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model()
    model.vae_decoder = repeat_sample_decoder()
    samples = mx.zeros((33, 64), dtype=mx.float32)
    samples[31, 0], samples[32, 0] = 1, 2
    actual, _ = run_generation(model, samples, monkeypatch)
    expected = np.repeat(np.r_[np.zeros(31), 1, 2], 3200).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("trim", [False, True])
def test_chunked_generation_preserves_trailing_silence_trimming(
    device: None, monkeypatch: pytest.MonkeyPatch, trim: bool
) -> None:
    model = make_model()
    model.vae_decoder = repeat_sample_decoder()
    samples = mx.zeros((33, 64), dtype=mx.float32)
    samples[:9, 0] = 1
    actual, metrics = run_generation(model, samples, monkeypatch, trim=trim)
    # 1.2 seconds of speech followed by silence. Trimming retains 300 ms of pad.
    expected = np.zeros(36000 if trim else 33 * 3200, dtype=np.float32)
    expected[:28800] = 1
    np.testing.assert_array_equal(actual, expected)
    assert metrics.audio_samples == len(expected)


@pytest.mark.parametrize("subclass", [False, True])
def test_custom_final_decoders_keep_the_complete_sequence(
    device: None, monkeypatch: pytest.MonkeyPatch, subclass: bool
) -> None:
    calls: list[int] = []

    def decode(latent: mx.array) -> mx.array:
        calls.append(latent.shape[-1])
        # A custom decoder may intentionally couple all time positions.
        return mx.broadcast_to(mx.mean(latent), (1, 1, latent.shape[-1] * 3200))

    class CustomDecoder(VAEDecoder):
        def __call__(self, latent: mx.array) -> mx.array:
            return decode(latent)

    model = make_model()
    model.vae_decoder = CustomDecoder() if subclass else decode
    samples = mx.array(np.random.RandomState(53).normal(size=(33, 64)), mx.float32)
    expected = np.array(mx.mean(samples.astype(mx.float16))).item()
    actual, _ = run_generation(model, samples, monkeypatch)
    assert calls == [33]
    np.testing.assert_array_equal(
        actual, np.full(33 * 3200, expected, dtype=np.float32)
    )


@pytest.mark.parametrize(
    "topology",
    [
        "fewer_blocks",
        "extra_block",
        "fewer_stages",
        "extra_stage",
        "fewer_upsamplers",
        "extra_upsampler",
    ],
)
def test_custom_base_decoder_topology_keeps_batch_decoding(
    device: None, monkeypatch: pytest.MonkeyPatch, topology: str
) -> None:
    model = make_model()
    decoder = model.vae_decoder
    assert type(decoder) is VAEDecoder
    if topology == "fewer_blocks":
        decoder.stages[0].pop()
    elif topology == "extra_block":
        decoder.stages[0].append(decoder.stages[0][-1])
    elif topology == "fewer_stages":
        decoder.stages.pop()
    elif topology == "extra_stage":
        decoder.stages.append(decoder.stages[-1])
    elif topology == "fewer_upsamplers":
        decoder.upsample_convs.pop()
    else:
        decoder.upsample_convs.append(decoder.upsample_convs[-1])
    samples = mx.array(np.random.RandomState(59).normal(size=(33, 64)), mx.float32)
    expected = np.array(decoder(samples.T[None].astype(mx.float16))).reshape(-1)
    calls: list[int] = []
    batch_decode = VAEDecoder.__call__

    def batch(decoder: VAEDecoder, latent: mx.array) -> mx.array:
        calls.append(latent.shape[-1])
        return batch_decode(decoder, latent)

    monkeypatch.setattr(VAEDecoder, "__call__", batch)
    actual, metrics = run_generation(model, samples, monkeypatch)
    assert calls == [33]
    assert actual.dtype == np.float32
    assert metrics.audio_samples == expected.size
    np.testing.assert_array_equal(actual, expected)
