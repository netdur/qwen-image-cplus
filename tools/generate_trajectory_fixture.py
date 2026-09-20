#!/usr/bin/env python3
"""Generate the pinned 256px, 40-step latent-trajectory oracle.

This development-only tool follows the Qwen-Image-2.1 pipeline's exact
FlowMatch schedule and KV-cache transition.  The runtime fixture contains the
initial inputs, complete scheduler vectors, and latents after steps 1, 2, and
40.  All tensors are exported as little-endian F32, although the official
model and per-step scheduler result are BF16.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import __version__ as accelerate_version
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from diffusers import __version__ as diffusers_version
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
from transformers import __version__ as transformers_version

from calibrate_transformer_quantization import (
    CASES,
    DIFFUSERS_COMMIT,
    MODEL_SNAPSHOT,
    PROMPTS,
    encode_prompts,
    make_inputs,
    validate_model_root,
)


STEPS = 40
CASE_NAME = "short-256-early"
CHECKPOINTS = (1, 2, 40)


def write_tensor(directory: Path, name: str, value: np.ndarray, records: dict) -> None:
    value = np.asarray(value, dtype="<f4", order="C")
    raw = value.tobytes()
    filename = f"{name}.f32"
    (directory / filename).write_bytes(raw)
    records[name] = {
        "file": filename,
        "dtype": "F32",
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def error_summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().cpu().float()
    return {
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "rms": float(value.square().mean().sqrt()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/trajectory_256_fp32")
    )
    args = parser.parse_args()
    model_root = args.model.resolve()
    validate_model_root(model_root)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for trajectory fixture generation")

    case = next(case for case in CASES if case.name == CASE_NAME)
    device = torch.device("mps")
    prompt = encode_prompts(model_root, device)[case.prompt_index]

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_root / "scheduler", local_files_only=True
    )
    config = scheduler.config
    image_tokens = (case.pixels // 16) ** 2
    mu = calculate_shift(
        image_tokens,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    requested_sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    timesteps, _ = retrieve_timesteps(
        scheduler, device=device, sigmas=requested_sigmas, mu=mu
    )

    inputs = make_inputs(case, prompt, float(timesteps[0].cpu()), device)
    latents = inputs["hidden_states"]
    initial_latents = latents.detach().cpu().float()
    text_rows = inputs["encoder_hidden_states"].shape[1]
    side = case.pixels // 16
    image_pad_mask = torch.cat(
        [
            torch.zeros(text_rows, dtype=torch.bool, device=device),
            torch.ones(image_tokens, dtype=torch.bool, device=device),
        ]
    )

    started = time.perf_counter()
    model = QwenImage21Transformer2DModel.from_pretrained(
        model_root / "transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    print(f"loaded pinned transformer in {time.perf_counter() - started:.2f}s", flush=True)
    rotary = model.pos_embed([(1, side, side)], image_pad_mask, device=device)
    rope = torch.cat([rotary.real, rotary.imag], dim=-1).detach().cpu().float()

    cache = QwenImage21KVCache(len(model.transformer_blocks))
    checkpoints: dict[int, torch.Tensor] = {}
    summaries: dict[str, dict[str, float]] = {}
    step_seconds: list[float] = []
    scheduler.set_begin_index(0)
    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            step_started = time.perf_counter()
            noise = model(
                hidden_states=latents,
                encoder_hidden_states=inputs["encoder_hidden_states"],
                timestep=timestep.expand(1).to(latents.dtype) / 1000.0,
                img_shapes=inputs["img_shapes"],
                img_mask=inputs["img_mask"],
                kv_cache=cache,
                kv_cache_mode="extract" if index == 0 else "cached",
                return_dict=False,
            )[0]
            noise = noise[:, -image_tokens:]
            latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]
            torch.mps.synchronize()
            seconds = time.perf_counter() - step_started
            step_seconds.append(seconds)
            step = index + 1
            summaries[f"noise_step_{step:02d}"] = error_summary(noise)
            summaries[f"latent_step_{step:02d}"] = error_summary(latents)
            if step in CHECKPOINTS:
                checkpoints[step] = latents.detach().cpu().float()
            print(
                f"step {step:02d}/{STEPS}: timestep={float(timestep.cpu()):.7f} "
                f"seconds={seconds:.2f}",
                flush=True,
            )

    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    write_tensor(directory, "image_input", initial_latents.numpy(), records)
    write_tensor(
        directory,
        "text_input",
        inputs["encoder_hidden_states"].detach().cpu().float().numpy(),
        records,
    )
    write_tensor(directory, "rope", rope.numpy(), records)
    write_tensor(directory, "sigmas", scheduler.sigmas.detach().cpu().numpy(), records)
    write_tensor(directory, "timesteps", scheduler.timesteps.detach().cpu().numpy(), records)
    for step in CHECKPOINTS:
        write_tensor(directory, f"latent_step_{step:02d}", checkpoints[step].numpy(), records)

    metadata = {
        "schema_version": 1,
        "operation": "Qwen-Image-2.1 256px latent-only 40-step trajectory",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers_version,
            "transformers": transformers_version,
            "accelerate": accelerate_version,
            "device": "mps",
        },
        "case": case.name,
        "prompt": PROMPTS[case.prompt_index],
        "latent_seed": case.seed,
        "steps": STEPS,
        "checkpoint_steps": list(CHECKPOINTS),
        "shape": {
            "text_rows": text_rows,
            "target_rows": image_tokens,
            "joint_rows": text_rows + image_tokens,
            "latent_channels": 64,
        },
        "scheduler": {
            "class": "FlowMatchEulerDiscreteScheduler",
            "mu": mu,
            "config": dict(config),
            "requested_sigmas": requested_sigmas.tolist(),
            "euler_dtype": "FP32 update cast back to BF16 after every step",
        },
        "cache": {
            "step_1": "extract text-prefix K/V at every transformer block",
            "steps_2_to_40": "cached target-only transformer execution",
        },
        "step_seconds": step_seconds,
        "summaries": summaries,
        "reference_dtype": "BF16 model, model inputs, noise, and post-step latents; F32 exports",
        "tensors": records,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote trajectory fixture to {directory}", flush=True)

    del model, cache, prompt, latents, inputs
    gc.collect()
    torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
