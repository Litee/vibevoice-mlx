# VibeVoice MLX

MLX inference for [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) text-to-speech on Apple Silicon.

Zero-shot voice cloning TTS: synthesize speech from text, optionally cloning one or more reference voices. Pure MLX — no PyTorch dependency at inference time.

## Quick start

```bash
# Install
git clone https://github.com/Litee/vibevoice-mlx && cd vibevoice-mlx
uv sync

# Basic synthesis (model downloads automatically)
uv run vibevoice-mlx --text "Hello, world!" --output hello.wav

# Voice cloning
uv run vibevoice-mlx \
  --ref-audio speaker.wav --text "Clone this voice" --output cloned.wav

# Encode a voice for reuse (one-time)
uv run vibevoice-mlx \
  --ref-audio speaker.wav --save-voice voice.safetensors

# Synthesize with saved voice
uv run vibevoice-mlx \
  --voice voice.safetensors --text "Hello again!" --output hello.wav

# Multi-speaker voice cloning
uv run vibevoice-mlx \
  --ref-audio spk1.wav spk2.wav \
  --text "Speaker 1: Hello.\nSpeaker 2: Hi there." --output dialogue.wav

# With quantization for faster generation
uv run vibevoice-mlx --quantize 8 --text "Hello, world!"

# CoreML semantic feedback (optional dependency)
uv run --extra coreml vibevoice-mlx \
  --coreml-semantic --text "Hello, world!" --output coreml.wav
```

Reference files map to speakers in command-line order. Speaker labels may be
zero-based (`Speaker 0:`, `Speaker 1:`) or one-based (`Speaker 1:`, `Speaker 2:`)
when every explicit label is positive. Mixed numbering is interpreted as
zero-based, and labels that do not map to a supplied voice are rejected. Raw
reference audio uses at most its first 10 seconds.

The default 200-speech-token budget produces at most about 26.7 seconds of
audio. Raise `--max-speech-tokens` for longer input; the CLI warns when the
budget truncates generation.

## Performance

These historical, hardware-specific results were measured subprocess-isolated
on Apple Silicon (M4 Max, 64 GB) with voice cloning and about 30 seconds of
audio. Current `main` contains later performance and correctness changes, so use
the benchmark tools below for current measurements on your machine.

**vibevoice-1.5b-mlx:**

| Config | RTF | Gen | Peak Mem |
|--------|-----|-----|----------|
| fp16 | 1.85x | 16.3s | 6.7 GB |
| fp16, no-semantic | 2.46x | 11.2s | 5.7 GB |
| fp16, coreml-semantic | 2.07x | 12.3s | 6.0 GB |
| int8 | 2.63x | 9.7s | 5.4 GB |
| int8, no-semantic | 4.03x | 6.9s | 4.5 GB |
| int8, coreml-semantic | 3.07x | 8.4s | 4.8 GB |
| int4 | 2.72x | 8.9s | 4.6 GB |
| int4, no-semantic | 4.30x | 5.2s | 3.8 GB |
| int4, coreml-semantic | 3.22x | 7.3s | 3.9 GB |

**vibevoice-7b-mlx:**

| Config | RTF | Gen | Peak Mem |
|--------|-----|-----|----------|
| fp16 | 0.53x | 53.0s | 21.7 GB |
| fp16, no-semantic | 0.59x | 51.7s | 20.3 GB |
| fp16, coreml-semantic | 0.56x | 57.2s | 21.0 GB |
| int8 | 1.06x | 29.6s | 14.9 GB |
| int8, no-semantic | 1.24x | 23.3s | 13.6 GB |
| int8, coreml-semantic | 1.12x | 25.5s | 14.2 GB |
| int4 | 1.16x | 25.8s | 11.2 GB |
| int4, no-semantic | 1.37x | 19.5s | 9.8 GB |
| int4, coreml-semantic | 1.24x | 22.0s | 10.5 GB |

RTF = audio duration / processing time (higher is faster; >1x means faster than real-time).

## Supported models

Pre-converted MLX weights with bundled tokenizer (downloaded automatically):

| Model | Size | Original |
|-------|------|----------|
| [gafiatulin/vibevoice-1.5b-mlx](https://huggingface.co/gafiatulin/vibevoice-1.5b-mlx) | 4.7 GB | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| [gafiatulin/vibevoice-7b-mlx](https://huggingface.co/gafiatulin/vibevoice-7b-mlx) | 18 GB | [vibevoice/VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) |

Optional CoreML semantic encoder (downloaded automatically when `--coreml-semantic` is used):

| Component | Size |
|-----------|------|
| [gafiatulin/vibevoice-semantic-encoder-mlpackage](https://huggingface.co/gafiatulin/vibevoice-semantic-encoder-mlpackage) | ~657 MB |

The semantic encoder provides acoustic feedback to the LLM during generation, improving speech quality. By default it runs as a pure MLX model on the GPU. The optional CoreML path requires the `coreml` extra and uses explicit convolution-cache tensors so feedback preserves history. `--coreml-semantic` selects CPU/GPU execution; `--ane-semantic` selects CPU/Neural Engine execution. The synthesis CLI falls back to MLX if CoreML cannot load, while the benchmark scripts reject an unavailable requested backend. ANE speed and numerical precision vary by device, so benchmark and listen before choosing it. Predictions remain synchronous because the next LLM step depends on the semantic feedback.

## Architecture

```
Text ──→ Qwen2.5 LLM backbone ──→ control tokens
                │
                └──→ Diffusion head (DDPM v-prediction, DPM-Solver++) ──→ VAE latents
                                                                              │
                                                          VAE decoder ──→ 24kHz audio
                                                              ▲
                                          Semantic encoder ───┘ (optional feedback)
```

## CLI options

```
--text TEXT              Text to synthesize (required except for voice encoding)
--model MODEL            Hugging Face model ID or local bundle path
                         (default: gafiatulin/vibevoice-1.5b-mlx)
--output FILE            Output WAV path (default: output.wav)
--ref-audio FILE [FILE]  Reference audio for voice cloning (one per speaker)
--voice FILE [FILE]      Pre-encoded voice (.safetensors) for voice cloning
--save-voice FILE        Save encoded voice embeddings for reuse
--quantize {4,8}         Quantize LLM backbone (int4 or int8)
--quantize-diffusion     Also quantize the diffusion head
--solver {dpm,sde}       DPM-Solver++ ODE or stochastic SDE (default: dpm)
--diffusion-steps N      DPM-Solver++ steps, 1-999 (default: 10)
--cfg-scale FLOAT        Finite classifier-free guidance scale (default: 1.3)
--max-speech-tokens N    Positive speech-token limit (default: 200)
--silence-detection      Boost speech-end selection on sustained silence
--trim-trailing-silence  Enable waveform trimming (default: follows silence detection)
--no-trim-trailing-silence
                         Disable trimming, including with silence detection
--silence-threshold N    RMS threshold used by waveform trimming (default: 0.05)
--silence-min-duration-ms N
                         Long silence gap used by forward trimming (default: 1500)
--silence-pad-ms N       Audio retained after a detected cut (default: 300)
--seed INT               Random seed (default: 42)
--no-semantic            Disable semantic encoder feedback
--coreml-semantic        Use CoreML CPU/GPU semantic encoder
--ane-semantic           Use CoreML CPU/Neural Engine semantic encoder (opt-in)
--tokenizer MODEL        Override tokenizer auto-detection
```

Complete tokenizer assets in the resolved model bundle take precedence.
Legacy bundles without them fall back to the Qwen tokenizer selected for the
model vocabulary. Explicit `--tokenizer` values always win.

When waveform trimming is enabled, the current implementation cuts at the
first qualifying long silence after speech begins and then removes terminal
silence. Use `--no-trim-trailing-silence` for dialogue where long internal
pauses must be preserved.

## Optimizations

- **DPM-Solver++ 2M**: Second-order multistep solver — 10 DPM steps > 100 DDPM steps quality
- **Streaming VAE decoder**: Causal conv caches for chunk-by-chunk decoding
- **Streaming semantic encoder**: 34-buffer causal CNN for real-time feedback
- **CoreML semantic encoder**: Explicit recurrent caches with CPU/GPU or opt-in CPU/Neural Engine execution
- **Selective quantization**: LLM backbone quantized (int4/int8), diffusion head stays full precision
- **Bounded long-form memory**: Chunked LM prefill and final VAE decode avoid retaining full-sequence intermediates
- **Selective logits**: Projects only the control-token logits used during speech generation
- **MLX-native RNG**: Seeded, on-device diffusion noise sampling
- **bf16→fp16 conversion**: 2x faster inference on Apple Silicon vs bfloat16

## Project structure

```
vibevoice_mlx/
├── e2e_pipeline.py     CLI entry point and voice cloning
├── model.py            Qwen2.5 backbone + diffusion head + VAE decoder
├── generate.py         Autoregressive generation with DPM-Solver++
├── load_weights.py     HuggingFace weight loading and key mapping
├── streaming_vae.py    Streaming VAE decoder with conv caches
├── vae_encoder.py      Voice-reference acoustic encoder
├── semantic_encoder.py Pure MLX streaming semantic encoder
├── coreml_semantic.py  Optional CoreML semantic encoder wrapper
└── fast_forward.py     Optimized LM and diffusion head forward

convert.py              Weight conversion and HuggingFace upload
bench_compare.py        Quantization and config benchmark suite
bench_paired.py         Serial paired benchmark runner
benchmarks/podcast.py   Long-form backend benchmark
```

## Requirements

- Python >= 3.10
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.24.0

```bash
uv sync
```

## License

This inference code is MIT licensed. See [LICENSE](LICENSE).

The model weights ([microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)) are under the [MIT License](https://huggingface.co/microsoft/VibeVoice-1.5B/blob/main/LICENSE).

## Reproducible five-minute benchmark

Install the optional backend with `uv sync --frozen --extra coreml`. Run each backend
in a separate process, with the same local model and voice references:

```bash
uv run --frozen --extra coreml python benchmarks/podcast.py \
  --model /path/to/vibevoice-7b-mlx \
  --text-file benchmarks/podcast.txt \
  --ref-audio /path/to/Alice.wav /path/to/Frank.wav \
  --backend mlx --mode fixed-duration --tokens 2250 \
  --output /path/to/baseline.wav

# Repeat with --backend ane and a different output path.
# --backend coreml measures the corrected CoreML CPU/GPU path.
```

Fixed-duration mode uses INT8 LLM weights, 10 ODE steps, guidance 1.3, seed 42,
a separate eight-token warm-up, and the requested speech-token budget. A budget
of 2,250 tokens is exactly 300 seconds, so the supplied text must be longer than
the timed excerpt. Fixed-duration success proves that the run reached its budget;
it is not evidence of natural completion.

Use `--mode completion` with a sufficiently large `--tokens` limit to require
non-empty audio with a natural stop (`eos`, or the valid single-segment speech
end) instead of a token-limit stop. The adjacent JSON report records the
evaluation result, generation stop reason, segment transitions, setup and
synchronized generation times, MLX peak memory, process RSS, input hashes, and
audio checks. The benchmark rejects unavailable backends and non-finite audio.
Model loading and audio-file writing are excluded from generation time. CoreML
allocations are not included in MLX's memory counter, and process RSS is a
separate metric, not an additive total.
