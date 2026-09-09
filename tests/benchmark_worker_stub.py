"""CPU-only stand-ins installed by sitecustomize in benchmark worker tests."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any


def event(name: str, **values: Any) -> None:
    print("WORKER_EVENT:" + json.dumps({"name": name, **values}))


def module(name: str) -> ModuleType:
    value = ModuleType(name)
    sys.modules[name] = value
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, value)
    return value


class Array:
    shape = (2, 3)

    def __getitem__(self, key: Any) -> "Array":
        return self

    def astype(self, dtype: Any) -> "Array":
        return self


class VoiceCloneData:
    def __init__(self) -> None:
        self.input_ids = [1, 2]
        self.speakers = [
            SimpleNamespace(
                cached_embeds=None,
                num_vae_tokens=2,
                ref_audio_np=[0] * 8,
                speech_embed_positions=[0, 1],
            )
        ]


def load_model(model: str, quantize_bits: int | None = None) -> tuple[object, object]:
    event("load_model", model=model, quantize_bits=quantize_bits)
    source_bits = os.environ.get("BENCH_TEST_SOURCE_BITS")
    effective_bits = int(source_bits) if source_bits else quantize_bits
    layer = QuantizedLinear() if effective_bits else Linear()
    if effective_bits:
        layer.bits = effective_bits
    return SimpleNamespace(
        named_modules=lambda: [("projection", layer)]
    ), SimpleNamespace(
        quantization={"bits": int(source_bits)} if source_bits else None
    )


def load_audio(path: str) -> list[int]:
    event("load_audio", path=path)
    return [0] * 12


def encode_voice(
    audio: list[int], tokens: int, model: object, config: object, model_id: str
) -> Array:
    event("encode_voice", audio_samples=len(audio), tokens=tokens, model=model_id)
    if os.environ.get("BENCH_TEST_VOICE_FAILURE"):
        raise ValueError("Voice encoding failed")
    return Array()


def save_voice(path: str, embeddings: Array) -> None:
    event("save_voice", path=path)


def load_voice(path: str) -> Array:
    event("load_voice", path=path)
    return Array()


def detect_tokenizer(model: str, config: object) -> str:
    event("detect_tokenizer", model=model)
    return "test-tokenizer"


def tokenize_text(
    text: str, tokenizer: str, config: object, ref_audio: list[str] | None = None
) -> VoiceCloneData | list[int]:
    event("tokenize", text=text, tokenizer=tokenizer, ref_audio=ref_audio)
    if os.environ.get("BENCH_TEST_MISSING_VOICE"):
        return [1, 2]
    return VoiceCloneData() if ref_audio else [1, 2]


def semantic(
    model: object, config: object, model_id: str
) -> tuple[object, object] | None:
    event("semantic", model=model_id)
    if os.environ.get("BENCH_TEST_MISSING_MLX"):
        return None
    return object(), object()


def coreml(model: object, config: object) -> tuple[object, object] | None:
    event("coreml_fallback")
    if os.environ.get("BENCH_TEST_COREML_AVAILABLE"):
        return object(), object()
    return None


def generate(**kwargs: Any) -> tuple[list[float], SimpleNamespace]:
    event(
        "generate",
        options=vars(kwargs["opts"]),
        semantic=kwargs["semantic_encoder_fn"] is not None,
        semantic_reset=kwargs["semantic_reset_fn"] is not None,
        voice_positions=sorted(kwargs["voice_embeds"] or {}),
    )
    return [0.1, 0.2], SimpleNamespace(
        summary=lambda: {"audio_seconds": 2.0}, num_speech_tokens=2
    )


def write_audio(path: str, audio: list[float], sample_rate: int) -> None:
    event("write_audio", path=path, audio=audio, sample_rate=sample_rate)


module("mlx")
nn = module("mlx.nn")


class Linear:
    weight = SimpleNamespace(dtype="float16")


class QuantizedLinear:
    bits = 4


nn.Linear = nn.Embedding = Linear
nn.QuantizedLinear = nn.QuantizedEmbedding = QuantizedLinear
nn.RMSNorm = type("RMSNorm", (), {})
core = module("mlx.core")
core.array = lambda value: Array()
core.float16 = "float16"
core.reset_peak_memory = lambda: None
core.get_peak_memory = lambda: 1_000_000_000
module("vibevoice_mlx")
weights = module("vibevoice_mlx.load_weights")
weights.load_model = load_model
weights.resolve_model_path = lambda path: path
pipeline = module("vibevoice_mlx.e2e_pipeline")
pipeline._load_and_resample = load_audio
pipeline.encode_voice_reference = encode_voice
pipeline.save_voice = save_voice
pipeline.load_voice = load_voice
pipeline._detect_tokenizer = detect_tokenizer
pipeline.tokenize_text = tokenize_text
pipeline._try_mlx_semantic = semantic
pipeline._try_coreml_semantic = coreml
pipeline.VoiceCloneData = VoiceCloneData
pipeline.SAMPLE_RATE = 24000
pipeline.VOICE_CLONE_SAMPLES = 8
pipeline.SPEECH_TOK_COMPRESS_RATIO = 4
generation = module("vibevoice_mlx.generate")
generation.GenerationOptions = SimpleNamespace
generation.generate = generate
module("soundfile").write = write_audio
