"""Small, complete checkpoint tensors for loader integration tests."""

import mlx.core as mx


def tiny_vae_weights(latent_size: int = 1) -> dict[str, mx.array]:
    """A one-channel decoder with all 26 blocks and six upsampling stages."""
    prefix = "vae_decoder."
    weights = {
        prefix + "upsample_layers.0.0.conv.conv.weight": mx.zeros((1, latent_size, 1)),
        prefix + "upsample_layers.0.0.conv.conv.bias": mx.array([0.25]),
        prefix + "head.conv.conv.weight": mx.ones((1, 1, 1)),
        prefix + "head.conv.conv.bias": mx.zeros((1,)),
    }
    for stage, depth in enumerate([8, 3, 3, 3, 3, 3, 3]):
        for block in range(depth):
            block_prefix = f"{prefix}stages.{stage}.{block}."
            for name, value in {
                "norm.weight": mx.ones((1,)),
                "mixer.conv.conv.conv.weight": mx.zeros((1, 1, 1)),
                "mixer.conv.conv.conv.bias": mx.zeros((1,)),
                "gamma": mx.zeros((1,)),
                "ffn_norm.weight": mx.ones((1,)),
                "ffn.linear1.weight": mx.zeros((1, 1)),
                "ffn.linear1.bias": mx.zeros((1,)),
                "ffn.linear2.weight": mx.zeros((1, 1)),
                "ffn.linear2.bias": mx.zeros((1,)),
                "ffn_gamma": mx.zeros((1,)),
            }.items():
                weights[block_prefix + name] = value
    for stage, stride in enumerate([8, 5, 5, 4, 2, 2], start=1):
        upsample_prefix = f"{prefix}upsample_layers.{stage}.0.convtr.convtr."
        weights[upsample_prefix + "weight"] = mx.ones((1, 1, stride))
        weights[upsample_prefix + "bias"] = mx.zeros((1,))
    return weights
