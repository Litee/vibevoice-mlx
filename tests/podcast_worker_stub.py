"""Replace model work in podcast CLI subprocess tests, keeping real options validation."""

import atexit
import importlib
import json
import os
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np

generation = importlib.import_module("vibevoice_mlx.generate")
pipeline = importlib.import_module("vibevoice_mlx.e2e_pipeline")
weights = importlib.import_module("vibevoice_mlx.load_weights")
mx.set_default_device(mx.cpu)


def event(name: str, **values: Any) -> None:
    print("PODCAST_EVENT:" + json.dumps({"name": name, **values}), flush=True)


def load_model(*args: Any, **kwargs: Any) -> tuple[object, object]:
    event("load_model")
    model = SimpleNamespace()
    atexit.register(
        lambda: event("restored", value=model._fast_lm.select_token is original_select)
    )
    return model, SimpleNamespace(
        speech_start_id=1, speech_end_id=2, speech_diffusion_id=3, eos_id=4
    )


def original_select(hidden: int, **kwargs: Any) -> int:
    assert kwargs == {"speech_only": True, "stop_boost": 0.0}
    return hidden


def tokenize_text(*args: Any, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(input_ids=[1], speakers=[])


def semantic(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    return lambda value: value, lambda: None


def generate(
    model: object,
    input_ids: list[int],
    opts: generation.GenerationOptions,
    **kwargs: Any,
) -> tuple[np.ndarray, SimpleNamespace]:
    event("generate", options=asdict(opts), estimated_total=kwargs["estimated_total"])
    if not hasattr(model, "_fast_lm"):
        model._fast_lm = SimpleNamespace(select_token=original_select)
    selections = json.loads(os.environ.get("PODCAST_SELECTIONS", "[3,2,1,3,4]"))
    for token in selections:
        assert (
            model._fast_lm.select_token(token, speech_only=True, stop_boost=0.0)
            == token
        )
    if opts.max_speech_tokens != 8 and os.environ.get("PODCAST_FAIL"):
        raise RuntimeError("fixture generation failed")
    return np.full(
        selections[:-1].count(3) * 3200, 0.25, dtype=np.float32
    ), SimpleNamespace(
        summary=lambda: {"speech_tokens": selections[:-1].count(3)},
        num_speech_tokens=selections[:-1].count(3)
        + int(os.environ.get("PODCAST_METRIC_ERROR", "0")),
    )


weights.load_model = load_model
pipeline.tokenize_text = tokenize_text
pipeline._try_mlx_semantic = semantic
pipeline._try_coreml_semantic = semantic
generation.generate = generate
mx.synchronize = lambda: None
mx.reset_peak_memory = lambda: None
mx.get_peak_memory = lambda: 0
