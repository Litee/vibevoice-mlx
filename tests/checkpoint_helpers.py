"""Small, complete checkpoint tensors for loader integration tests."""

import mlx.core as mx


def tiny_vae_weights(
    latent_size: int = 1,
    stage_channels: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1),
) -> dict[str, mx.array]:
    """A small decoder with all 26 blocks and six upsampling stages."""
    prefix = "vae_decoder."
    weights = {
        prefix + "upsample_layers.0.0.conv.conv.weight": mx.zeros(
            (stage_channels[0], latent_size, 1)
        ),
        prefix + "upsample_layers.0.0.conv.conv.bias": mx.full(
            (stage_channels[0],), 0.25
        ),
        prefix + "head.conv.conv.weight": mx.full(
            (1, stage_channels[-1], 1), 1 / stage_channels[-1]
        ),
        prefix + "head.conv.conv.bias": mx.zeros((1,)),
    }
    for stage, depth in enumerate([8, 3, 3, 3, 3, 3, 3]):
        channels = stage_channels[stage]
        for block in range(depth):
            ffn_width = channels + block
            block_prefix = f"{prefix}stages.{stage}.{block}."
            for name, value in {
                "norm.weight": mx.ones((channels,)),
                "mixer.conv.conv.conv.weight": mx.zeros((channels, 1, 1)),
                "mixer.conv.conv.conv.bias": mx.zeros((channels,)),
                "gamma": mx.zeros((channels,)),
                "ffn_norm.weight": mx.ones((channels,)),
                "ffn.linear1.weight": mx.zeros((ffn_width, channels)),
                "ffn.linear1.bias": mx.zeros((ffn_width,)),
                "ffn.linear2.weight": mx.zeros((channels, ffn_width)),
                "ffn.linear2.bias": mx.zeros((channels,)),
                "ffn_gamma": mx.zeros((channels,)),
            }.items():
                weights[block_prefix + name] = value
    for stage, stride in enumerate([8, 5, 5, 4, 2, 2], start=1):
        upsample_prefix = f"{prefix}upsample_layers.{stage}.0.convtr.convtr."
        previous, channels = stage_channels[stage - 1 : stage + 1]
        weights[upsample_prefix + "weight"] = mx.full(
            (previous, channels, stride), 1 / previous
        )
        weights[upsample_prefix + "bias"] = mx.zeros((channels,))
    return weights
