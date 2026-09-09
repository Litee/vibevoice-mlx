"""Two-speaker completion or fixed-duration benchmark (run backends separately)."""

import argparse
import hashlib
import json
import logging
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import soundfile as sf

from vibevoice_mlx.e2e_pipeline import (
    _try_coreml_semantic,
    _try_mlx_semantic,
    encode_voice_reference,
    tokenize_text,
)
from vibevoice_mlx.generate import (
    GenerationOptions,
    _validate_cfg_scale,
    _validate_diffusion_steps,
    generate,
)
from vibevoice_mlx.load_weights import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--ref-audio", nargs=2, required=True)
    parser.add_argument("--backend", choices=["mlx", "coreml", "ane"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["completion", "fixed-duration"],
        default="fixed-duration",
        help="Completion requires a natural stop; fixed-duration requires the token budget.",
    )
    parser.add_argument(
        "--generation-provenance",
        choices=["natural", "controlled-replay"],
        default="natural",
        help="Label externally controlled token replay; this does not enable replay.",
    )
    parser.add_argument("--tokens", type=int, default=2250)
    parser.add_argument("--solver", choices=["dpm", "sde"], default="dpm")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be a positive integer")
    try:
        _validate_diffusion_steps(args.diffusion_steps)
        _validate_cfg_scale(args.cfg_scale)
    except ValueError as error:
        parser.error(str(error))
    options = GenerationOptions(
        solver=args.solver,
        diffusion_steps=args.diffusion_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        max_speech_tokens=args.tokens,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_setup = time.perf_counter()
    model, config = load_model(args.model, quantize_bits=8)
    text = args.text_file.read_text()
    prompt = tokenize_text(text, args.model, config, ref_audio=args.ref_audio)
    voice_embeds = {}
    for speaker in prompt.speakers:
        embeds = encode_voice_reference(
            speaker.ref_audio_np, speaker.num_vae_tokens, model, config, args.model
        )
        for i, pos in enumerate(speaker.speech_embed_positions):
            voice_embeds[pos] = mx.array(embeds[i : i + 1]).astype(mx.float16)
    semantic = (
        _try_mlx_semantic(model, config, args.model)
        if args.backend == "mlx"
        else _try_coreml_semantic(model, config, use_ane=args.backend == "ane")
    )
    if semantic is None:
        raise RuntimeError(
            f"Requested {args.backend} backend did not load; refusing fallback"
        )
    semantic_fn, semantic_reset = semantic
    setup_seconds = time.perf_counter() - start_setup
    start_warmup = time.perf_counter()
    generate(
        model,
        prompt.input_ids,
        replace(options, max_speech_tokens=8),
        semantic_encoder_fn=semantic_fn,
        semantic_reset_fn=semantic_reset,
        voice_embeds=voice_embeds,
        estimated_total=8,
    )
    semantic_reset()
    mx.synchronize()
    warmup_seconds = time.perf_counter() - start_warmup
    mx.reset_peak_memory()
    # Warmup initializes the cached LM. Observe only measured selections;
    # this adds a Python call/list append, with no tensor copy or evaluation.
    fast_lm = model._fast_lm
    original_select = fast_lm.select_token
    selections: list[int] = []

    def select_token(*args: Any, **kwargs: Any) -> int:
        token = original_select(*args, **kwargs)
        selections.append(token)
        return token

    fast_lm.select_token = select_token
    start = time.perf_counter()
    try:
        audio, metrics = generate(
            model,
            prompt.input_ids,
            options,
            semantic_encoder_fn=semantic_fn,
            semantic_reset_fn=semantic_reset,
            voice_embeds=voice_embeds,
            estimated_total=args.tokens,
        )
        mx.synchronize()
    finally:
        fast_lm.select_token = original_select
    seconds = time.perf_counter() - start
    # The first selection precedes the loop; each consumed token selects the
    # next one. The final selection is unconsumed (stop, cap, or loop limit).
    controls = []
    end_start_diffusion_transitions = []
    speech_offset = 0
    last_end = None
    last_start = None
    for offset, token in enumerate(selections[:-1]):
        if token == config.speech_diffusion_id:
            if last_end is not None and last_start is not None:
                end_start_diffusion_transitions.append(
                    {
                        "end_token_offset": last_end,
                        "start_token_offset": last_start,
                        "diffusion_token_offset": offset,
                        "speech_offset": speech_offset,
                    }
                )
                last_end = last_start = None
            speech_offset += 1
        elif token in (config.speech_end_id, config.speech_start_id):
            is_end = token == config.speech_end_id
            controls.append(
                {
                    "token_offset": offset,
                    "speech_offset": speech_offset,
                    "control": "speech_end" if is_end else "speech_start",
                }
            )
            if is_end:
                last_end, last_start = offset, None
            else:
                last_start = offset
    if speech_offset != metrics.num_speech_tokens:
        raise RuntimeError("Consumed diffusion trace disagrees with generation metrics")
    duration = len(audio) / 24000
    nonfinite_samples = int(audio.size - np.count_nonzero(np.isfinite(audio)))
    natural_completion = args.generation_provenance == "natural" and (
        metrics.stop_reason in ("eos", "speech_end")
    )
    failure: str | None = None
    if args.mode == "completion":
        if args.generation_provenance == "controlled-replay":
            failure = "Controlled replay cannot establish natural completion"
        elif not natural_completion:
            failure = f"Incomplete generation: {metrics.stop_reason}"
        elif not len(audio):
            failure = "Empty audio"
    elif len(audio) != args.tokens * 3200:
        failure = (
            f"Generated {duration:.2f}s, expected {args.tokens * 3200 / 24000:.2f}s"
        )
    if nonfinite_samples:
        failure = "Non-finite audio"
    report = {
        "backend": args.backend,
        "mode": args.mode,
        "generation_provenance": args.generation_provenance,
        "evaluation": {
            "status": "complete" if natural_completion else "incomplete",
            "passed": failure is None,
            "failure": failure,
            "natural_completion": natural_completion,
            "natural_fidelity_eligible": args.mode == "completion" and failure is None,
        },
        "segment_trace": {
            "consumed_diffusion_tokens": speech_offset,
            "controls": controls,
            "end_start_diffusion_transitions": end_start_diffusion_transitions,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "--", "vibevoice_mlx"], text=True
        ),
        "model": args.model,
        "quantization_bits": 8,
        "solver": options.solver,
        "diffusion_steps": options.diffusion_steps,
        "seed": options.seed,
        "cfg_scale": options.cfg_scale,
        "max_speech_tokens": options.max_speech_tokens,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "reference_sha256": [
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.ref_audio
        ],
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "generation_seconds": seconds,
        "audio_seconds": duration,
        "audio_seconds_per_wall_second": duration / seconds,
        "wall_seconds_per_audio_second": seconds / duration if duration else None,
        "mlx_peak_memory_gib": mx.get_peak_memory() / 2**30,
        "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**30,
        "finite": nonfinite_samples == 0,
        "nonfinite_samples": nonfinite_samples,
        "peak_amplitude": (
            float(np.max(np.abs(audio)))
            if len(audio) and nonfinite_samples == 0
            else None
        ),
        "out_of_pcm_range_samples": int(np.count_nonzero(np.abs(audio) >= 1)),
        "metrics": metrics.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, 24000, subtype="PCM_16")
    report_json = json.dumps(report, indent=2, allow_nan=False)
    args.output.with_suffix(".json").write_text(report_json)
    print(report_json, flush=True)
    if failure is not None:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
