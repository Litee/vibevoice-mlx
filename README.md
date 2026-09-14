# VibeVoice MLX

MLX inference for [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) text-to-speech on Apple Silicon.

Zero-shot voice cloning TTS: synthesize speech from text, optionally cloning one or more reference voices. Pure MLX — no PyTorch dependency at inference time.

## What this fork adds

This fork builds on
[gafiatulin/vibevoice-mlx](https://github.com/gafiatulin/vibevoice-mlx), the MLX
port of Microsoft VibeVoice. Its changes focus on inference correctness,
long-form memory use, and reliable measurement.

| Area | Changes and practical benefit |
|------|-------------------------------|
| Generation correctness | Correct causal audio history, guidance-state handling, diffusion scheduling, and speech-token constraints bring inference closer to the reference implementation and address corrupted feedback and premature stopping. |
| Less work per generated frame | Selective control-token logits, block-grown KV caches, batched guidance forwards, and prepared diffusion conditioning reduce repeated projection, allocation, and copying work. |
| Lower startup and long-form memory | Layerwise runtime quantization avoids retaining the full unquantized backbone; chunked prompt processing bounds intermediate activations; long outputs without semantic feedback use bounded final VAE decoding. |
| Efficient Apple Silicon execution | Attention preserves the activation dtype, MLX semantic feedback stays on device, and corrected CoreML caching supports optional CPU/GPU or CPU/Neural Engine execution. |
| More reliable inputs and loading | Speaker labels are validated against supplied voices, reference audio and saved voices are checked, and model loading honors quantization metadata, shard indexes, and bundled tokenizer assets. |
| Observable generation and benchmarking | Generation reports why it stopped. Benchmarks distinguish fixed-duration throughput from natural completion, verify the effective configuration, and support serial paired comparisons. |

Selected historical paired benchmarks illustrate several improvements, using
INT8 7B unless otherwise noted. They used different code baselines and semantic
backends; the linked PRs record each configuration and its limitations.

| Improvement | Measured effect |
|-------------|-----------------|
| [Four-token logits](https://github.com/Litee/vibevoice-mlx/pull/20) | One five-minute ANE-semantic pair improved generation from 331.36 to 290.90 seconds and reduced peak MLX allocation from 14.54 to 12.76 GiB, with byte-identical audio. |
| [Batched guidance](https://github.com/Litee/vibevoice-mlx/pull/45) | One five-minute MLX-semantic pair improved generation from 361.42 to 321.99 seconds with byte-identical audio; three shorter ODE pairs also improved. |
| [Prepared diffusion conditioning](https://github.com/Litee/vibevoice-mlx/pull/52) | Three fixed-trace, MLX-semantic 30-second pairs reduced generation time by 13.86–14.44%. |
| [Layerwise INT8 loading](https://github.com/Litee/vibevoice-mlx/pull/57) | Fresh-process peak MLX allocation fell from 15.12 to 10.17 GiB during startup, with identical fingerprints of the model parameter tree. |
| [Bounded final VAE decode](https://github.com/Litee/vibevoice-mlx/pull/60) | A no-semantic 15-minute stress test reduced peak MLX allocation from 40.55 to 13.43 GiB. This path is not used by the default semantic mode. |
| [Chunked LM prefill](https://github.com/Litee/vibevoice-mlx/pull/67) | Peak MLX allocation fell by 155.4 MiB for a 7,575-token prompt, with 0.51% slower prefill. |
| [MLX 0.31.1 → 0.32.2](https://github.com/Litee/vibevoice-mlx/pull/83) | Five interleaved 7B INT8 MLX-semantic pairs, each generating 26.67 seconds, used 26.4% less generation time on the paired geometric mean; the 95% percentile-bootstrap interval was 22.3–30.6% less time. The changed audio passed human listening review. |
| Lightweight voice encoding | Four fresh-process 7B encode-only pairs with runtime quantization disabled reduced median time from 4.390 to 0.393 seconds, peak MLX allocation from 16.223 to 1.600 GiB, and RSS from 10.587 to 0.790 GiB, with byte-exact embeddings. |

These measurements compare individual changes on their original test workloads;
single-pair timing results are observations rather than stable speedup estimates,
and their percentages are not additive. Performance depends on the model,
precision, semantic backend, prompt length, and output duration. Some correctness
fixes change floating-point results or generation trajectories, so identical
seeds do not guarantee identical audio across versions.

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

Using `--save-voice` without `--text` runs in encode-only mode, loading only
the acoustic encoder and connector to create reusable voice embeddings.

The default 200-speech-token budget produces at most about 26.7 seconds of
audio. Raise `--max-speech-tokens` for longer input; the CLI warns when the
budget truncates generation.

## Performance

The synthesis CLI and `benchmarks/podcast.py` release the model's acoustic
reference encoder weights after preparing all voices, before semantic setup
and generation. The acoustic connector, audio decoder, and semantic feedback
remain available; library voice encoding keeps its existing caching behavior.
Five fresh-process 7B INT8 pairs reduced active MLX memory before semantic setup
and generation peak by 655.6 MiB in every pair, with byte-identical output.
Generation time was 1.4–8.3% lower; the geometric-mean reduction was 4.8%, with
an exploratory paired log-time 95% interval of 0.8–8.6%. With only five pairs,
the smallest possible two-sided sign-test p-value is 0.0625, so treat the timing
result as evidence from this workload rather than a general speedup guarantee.
Process-lifetime peak RSS did not improve consistently because earlier loading
and voice-encoding high-water marks can dominate it.

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

- **DPM-Solver++ 2M**: Second-order multistep solver with 10 steps by default
- **Streaming VAE decoder**: Causal conv caches for chunk-by-chunk decoding
- **Streaming semantic encoder**: 34-buffer causal CNN for real-time feedback
- **CoreML semantic encoder**: Explicit recurrent caches with CPU/GPU or opt-in CPU/Neural Engine execution
- **Selective quantization**: LLM backbone quantized (int4/int8), diffusion head stays full precision
- **Lightweight voice encoding**: Encode-only mode loads just the acoustic encoder and connector
- **Bounded long-form memory**: Chunked LM prefill and final VAE decode avoid retaining full-sequence intermediates
- **Selective logits**: Projects only the control-token logits used during speech generation
- **MLX-native RNG**: Seeded, on-device diffusion noise sampling
- **bf16→fp16 conversion**: Converts bfloat16 weights to float16 for MLX execution

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
- Apple Silicon Mac (M1/M2/M3/M4/M5)
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

Each benchmark voice reference may be raw audio or a saved `.safetensors` voice.

The podcast benchmark resolves tokenizer assets from the model bundle, with a
vocabulary-based fallback for incomplete or legacy bundles.

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

Paired timing ratios from `bench_paired.py` require matching stop reasons and
speech-token counts. Runs with mismatched or unknown workloads remain recorded
but are excluded from timing ratios.
