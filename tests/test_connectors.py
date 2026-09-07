"""Acoustic and semantic projections must match SpeechConnector normalization."""

from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import pytest

from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


@pytest.fixture(params=[(mx.cpu, 1e-6, 1e-6), (mx.gpu, 1e-3, 5e-3)], ids=["cpu", "gpu"])
def connector_tolerance(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[float, float]]:
    device, rtol, atol = request.param
    # M5 float32 matmuls showed up to 0.004 absolute error, including
    # cancellation near zero. Bound that GPU rounding and keep the CPU
    # oracle strict; the old epsilon produces errors as large as 3.85.
    with mx.stream(device):
        yield rtol, atol


@pytest.mark.parametrize(
    ("connector_name", "input_dim"),
    [("acoustic_connector", 64), ("semantic_connector", 128)],
)
def test_connector_matches_reference_rms_normalization(
    connector_name: str, input_dim: int, connector_tolerance: tuple[float, float]
) -> None:
    config = VibeVoiceConfig(
        hidden_size=4,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=8,
        vocab_size=8,
        diffusion_layers=0,
    )
    model = VibeVoiceModel(config)
    connector = getattr(model, connector_name)

    fc1_weight = np.eye(4, input_dim, dtype=np.float32)
    fc1_bias = np.array([1, -2, 3, -4], dtype=np.float32) * 1e-4
    norm_weight = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    fc2_weight = np.array(
        [[1, 0, 0, 1], [0, 2, -1, 0], [1, 0, 3, 0], [0, 1, 0, 4]],
        dtype=np.float32,
    )
    fc2_bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    connector.fc1.weight = mx.array(fc1_weight)
    connector.fc1.bias = mx.array(fc1_bias)
    connector.norm.weight = mx.array(norm_weight)
    connector.fc2.weight = mx.array(fc2_weight)
    connector.fc2.bias = mx.array(fc2_bias)

    # Near-zero projected features make the reference epsilon observable.
    features = np.zeros((1, 2, input_dim), dtype=np.float32)
    features[0, :, :4] = [[1e-4, -2e-4, 3e-4, -4e-4], [-4e-4, 3e-4, -2e-4, 1e-4]]
    projected = features.astype(np.float64) @ fc1_weight.T + fc1_bias
    normalized = projected / np.sqrt(
        np.mean(projected**2, axis=-1, keepdims=True) + 1e-6
    )
    expected = (normalized * norm_weight) @ fc2_weight.T + fc2_bias

    actual = connector(mx.array(features))

    rtol, atol = connector_tolerance
    np.testing.assert_allclose(np.array(actual), expected, rtol=rtol, atol=atol)
