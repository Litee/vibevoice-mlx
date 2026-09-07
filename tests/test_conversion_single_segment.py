"""Local checkpoint conversion preserves the model's segmentation policy."""

import json
from collections.abc import Iterator
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import pytest
from checkpoint_helpers import tiny_vae_weights
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from convert import convert_model
from vibevoice_mlx.e2e_pipeline import tokenize_text
from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.model import VibeVoiceConfig, VibeVoiceModel


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize(
    "metadata,source_name,expected",
    [
        ({"single_segment": True}, "local-source", True),
        (
            {"single_segment": False, "model_type": "kugelaudio"},
            "kugelaudio-source",
            False,
        ),
        ({"model_type": "kugelaudio"}, "local-source", True),
        ({"base_model": "kugelaudio/kugelaudio-0-open"}, "local-source", True),
        ({"speech_scaling_factor": 0.1953125}, "local-source", True),
    ],
    ids=[
        "explicit-true",
        "explicit-false",
        "inferred-type",
        "inferred-base",
        "inferred-scaling",
    ],
)
def test_local_reexport_preserves_segmentation_policy(
    tmp_path: Path,
    metadata: dict[str, object],
    source_name: str,
    expected: bool,
) -> None:
    source = tmp_path / source_name
    source.mkdir()
    config = VibeVoiceConfig(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=8,
        vocab_size=8,
        diffusion_layers=0,
        tie_word_embeddings=False,
        speech_start_id=0,
        speech_end_id=1,
        speech_diffusion_id=2,
        eos_id=3,
    )
    source_config = asdict(config)
    del source_config["single_segment"]
    source_config.update(metadata)
    (source / "config.json").write_text(json.dumps(source_config))
    model = VibeVoiceModel(config)
    # Keep each LM residual equal to its embedding and select speech_end.
    # This exercises real generation termination without invoking diffusion.
    model.model.embed_tokens.weight = mx.ones((8, 4))
    model.model.layers[0].self_attn.o_proj.weight = mx.zeros((4, 4))
    model.model.layers[0].mlp.down_proj.weight = mx.zeros((4, 8))
    model.lm_head.weight = mx.zeros((8, 4))
    model.lm_head.weight[config.speech_end_id] = mx.ones((4,))
    weights = dict(tree_flatten(model.parameters()))
    weights.update(tiny_vae_weights(config.vae_dim))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "Speaker": 1, "0": 2, ":": 3, "Hello": 4, "again": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(str(source))
    original, original_config = load_model(str(source))
    assert original_config.single_segment is expected

    output = tmp_path / "local-reexport"
    convert_model(str(source), output, tokenizer_id=str(source))
    restored, restored_config = load_model(str(output))

    assert restored_config.single_segment is expected
    assert restored.config.single_segment is expected

    original_prompt = tokenize_text("Hello\nagain", str(source), original_config)
    restored_prompt = tokenize_text("Hello\nagain", str(output), restored_config)
    assert restored_prompt == original_prompt
    assert original_prompt.count(1) == (2 if expected else 0)

    for loaded in (original, restored):
        resets = []
        audio, metrics = generate(
            loaded,
            [4],
            GenerationOptions(cfg_scale=1.0, max_speech_tokens=1),
            semantic_reset_fn=partial(resets.append, "speech_end"),
        )
        assert resets == ([] if expected else ["speech_end"] * 3)
        assert audio.size == 0
        assert metrics.num_speech_tokens == 0
