"""Streaming decode preserves causal audio with only the required input history."""

import mlx.core as mx
import numpy as np
import pytest
from test_semantic_audio_context import tiny_decoder

from vibevoice_mlx.model import VAEDecoder
from vibevoice_mlx.streaming_vae import RATIOS, StreamingVAEDecoder


@pytest.fixture(
    params=[(1, 0, 0), (1, 1, 1), (2, 0, 1), (2, 1, 2), (3, 0, 2)],
    ids=["k=s", "k=s+1", "k=2s", "k=2s+1", "k=3s"],
)
def decoder_case(request: pytest.FixtureRequest) -> tuple[VAEDecoder, int]:
    multiplier, extra, history = request.param
    vae = tiny_decoder()
    rng = np.random.RandomState(23)
    vae.upsample_convs = [
        (
            mx.array(
                rng.normal(0, 0.15, (2, 2, multiplier * stride + extra)).astype(
                    np.float16
                )
            ),
            mx.array(rng.normal(0, 0.1, (2,)).astype(np.float16)),
            stride,
        )
        for stride in RATIOS
    ]
    return vae, history


def test_transpose_caches_store_only_required_input_frames(
    decoder_case: tuple[VAEDecoder, int],
) -> None:
    vae, expected_history = decoder_case
    decoder = StreamingVAEDecoder(vae)
    for stage in range(len(RATIOS)):
        assert decoder.caches[f"up{stage}"].shape == (1, 2, expected_history)


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1), (2, 1, 2), (1, 3, 1)])
def test_streaming_matches_batch_for_kernel_sizes_chunking_and_reset(
    decoder_case: tuple[VAEDecoder, int], chunks: tuple[int, ...]
) -> None:
    vae, _ = decoder_case
    latent = mx.array(
        np.random.RandomState(31).normal(0, 0.5, (1, 64, 5)).astype(np.float16)
    )
    expected = np.array(vae(latent))
    decoder = StreamingVAEDecoder(vae)
    for _ in range(2):
        outputs = []
        offset = 0
        for frames in chunks:
            audio = decoder(latent[:, :, offset : offset + frames])
            mx.eval(audio, *decoder.caches.values())
            assert audio.shape == (1, 1, frames * 3200)
            outputs.append(np.array(audio))
            offset += frames
        np.testing.assert_allclose(
            np.concatenate(outputs, axis=2), expected, atol=2e-3, rtol=2e-3
        )
        decoder.reset()


def test_non_multiple_kernel_retains_oldest_required_impulse() -> None:
    # Make the decoder an exact delayed impulse path. K=2s+1 at the first
    # upsampler needs the input from two frames ago, including its last tap.
    vae = tiny_decoder()
    vae.init_conv_w = mx.zeros((2, 64, 1), dtype=mx.float16)
    vae.init_conv_w[0, 0, 0] = 1
    vae.head_w = mx.zeros((1, 2, 1), dtype=mx.float16)
    vae.head_w[0, 0, 0] = 1
    for stage in vae.stages:
        for block in stage:
            block["gamma"] = mx.zeros_like(block["gamma"])
            block["ffn_gamma"] = mx.zeros_like(block["ffn_gamma"])
    vae.upsample_convs = []
    for stage, stride in enumerate(RATIOS):
        kernel = 2 * stride + 1 if stage == 0 else stride
        weight = mx.zeros((2, 2, kernel), dtype=mx.float16)
        weight[0, 0, kernel - 1 if stage == 0 else 0] = 1
        vae.upsample_convs.append((weight, mx.zeros((2,), dtype=mx.float16), stride))
    latent = mx.zeros((1, 64, 4), dtype=mx.float16)
    latent[0, 0, 0] = 1
    expected = np.zeros((1, 1, 4 * 3200), dtype=np.float16)
    expected[0, 0, 2 * 3200] = 1
    np.testing.assert_array_equal(np.array(vae(latent)), expected)
    decoder = StreamingVAEDecoder(vae)
    output = []
    for frame in range(4):
        audio = decoder(latent[:, :, frame : frame + 1])
        mx.eval(audio, *decoder.caches.values())
        output.append(np.array(audio))
    np.testing.assert_array_equal(np.concatenate(output, axis=2), expected)
