"""Frozen benchmark worker from before explicit configuration reporting."""

import textwrap


def make_script() -> str:
    """Build the historical worker source used to test runtime inspection."""
    return textwrap.dedent("""\
import json, time, os, sys
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import mlx.core as mx
from vibevoice_mlx.load_weights import load_model
from vibevoice_mlx.generate import generate, GenerationOptions
from vibevoice_mlx.e2e_pipeline import (tokenize_text, VoiceCloneData, SAMPLE_RATE,
                          _detect_tokenizer, load_voice, encode_voice_reference)

args = json.loads(sys.argv[1])
model_id = args["model"]
model, config = load_model(model_id, quantize_bits=args["quantize"])

sem_mode = args["sem_mode"]
semantic_fn = None
semantic_reset = None

if sem_mode == "coreml":
    from vibevoice_mlx.e2e_pipeline import _try_coreml_semantic
    r = _try_coreml_semantic(model, config)
    if r is not None:
        semantic_fn, semantic_reset = r
    else:
        sem_mode = "mlx"

if sem_mode == "mlx":
    from vibevoice_mlx.e2e_pipeline import _try_mlx_semantic
    r = _try_mlx_semantic(model, config, model_id)
    if r is not None:
        semantic_fn, semantic_reset = r

tokenizer_name = _detect_tokenizer(model_id, config)
voice_arg = args["voice_arg"]
voice_list = [voice_arg] if voice_arg else None

text = args["text"]
result = tokenize_text(text, tokenizer_name, config, ref_audio=voice_list)

voice_embeds = None
if isinstance(result, VoiceCloneData):
    input_ids = result.input_ids
    voice_embeds = {}
    for spk in result.speakers:
        if voice_arg and voice_arg.endswith(".safetensors"):
            spk.cached_embeds = load_voice(voice_arg)[:spk.num_vae_tokens]
        else:
            spk.cached_embeds = encode_voice_reference(
                spk.ref_audio_np, spk.num_vae_tokens, model, config, model_id)
        embeds_mx = mx.array(spk.cached_embeds).astype(mx.float16)
        for i, pos in enumerate(spk.speech_embed_positions):
            if i < embeds_mx.shape[0]:
                voice_embeds[pos] = embeds_mx[i:i+1]
else:
    input_ids = result

opts = GenerationOptions(
    solver="dpm",
    diffusion_steps=10,
    cfg_scale=1.3,
    max_speech_tokens=args["max_tokens"],
    seed=args["seed"],
)

mx.reset_peak_memory()
t0 = time.perf_counter()
audio, metrics = generate(
    model=model,
    input_ids=input_ids,
    opts=opts,
    semantic_encoder_fn=semantic_fn,
    semantic_reset_fn=semantic_reset,
    voice_embeds=voice_embeds,
)
gen_s = time.perf_counter() - t0

summary = metrics.summary()
peak = mx.get_peak_memory() / 1e9
audio_s = summary.get("audio_seconds", 0)
rtf = audio_s / gen_s if gen_s > 0 else 0

audio_out = args["audio_out"]
if audio_out and len(audio) > 0:
    import soundfile as sf
    sf.write(audio_out, audio, SAMPLE_RATE)

print("BENCH_RESULT:" + json.dumps({
    "gen_s": gen_s,
    "audio_s": audio_s,
    "rtf": rtf,
    "peak_mem_gb": peak,
    "speech_tokens": metrics.num_speech_tokens,
}))
""")
