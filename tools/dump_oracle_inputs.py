#!/usr/bin/env python3
"""Dump the pinned pipeline's transformer inputs and trajectory for one prompt.

Writes raw little-endian FP32 files the Linux engine can consume before its own
text encoder, scheduler, and RoPE exist, and against which they can later be
checked:

  prompt_embeds.f32   [text_rows, 4096]
  noise.f32           [image_rows, 64]   initial packed latents
  rope.f32            [text_rows + image_rows, 128]  64 cosines then 64 sines
  step_NN_latents.f32 [image_rows, 64]   transformer input at step NN
  step_NN_noise.f32   [image_rows, 64]   transformer output at step NN
  final_latents.f32   [image_rows, 64]
  metadata.json       shapes, timesteps as seen by the transformer, sigmas
  image.png           the pipeline's decoded image

Uses the same GGUF transformer and memory plan as benchmark_oracle_gguf.py.
Development tool only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_oracle_gguf as oracle


def save(directory: Path, name: str, tensor: torch.Tensor) -> list[int]:
    value = tensor.detach().float().cpu().contiguous().numpy()
    value.astype("<f4").tofile(directory / f"{name}.f32")
    return list(value.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory")
    parser.add_argument("--prompt", default=oracle.PROMPT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1301)
    arguments = parser.parse_args()
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)

    transformer = oracle.QwenImage21Transformer2DModel.from_single_file(
        str(arguments.gguf),
        quantization_config=oracle.GGUFQuantizationConfig(compute_dtype=torch.float16),
        config=str(arguments.model),
        subfolder="transformer",
        torch_dtype=torch.float16,
    )
    vae = oracle.AutoencoderKLQwenImage21.from_pretrained(str(arguments.model), subfolder="vae", torch_dtype=torch.float32)
    pipe = oracle.QwenImage21Pipeline.from_pretrained(
        str(arguments.model), transformer=transformer, vae=vae, torch_dtype=torch.bfloat16
    )
    oracle.stream_text_encoder(pipe.text_encoder)
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(prompt=arguments.prompt, device=torch.device("cuda"))
    shapes = {"prompt_embeds": save(out, "prompt_embeds", prompt_embeds[0])}
    pipe.text_encoder = None
    pipe.register_modules(text_encoder=None)
    pipe.enable_model_cpu_offload()

    captured: dict[str, torch.Tensor] = {}
    pipe.transformer.pos_embed.register_forward_hook(
        lambda module, args, output: captured.setdefault("rope", output.detach())
    )
    timesteps: list[float] = []

    def before(module, args, kwargs):
        step = len(timesteps)
        timesteps.append(float(kwargs["timestep"].float()[0]))
        shapes[f"step_{step:02d}_latents"] = save(out, f"step_{step:02d}_latents", kwargs["hidden_states"][0])

    def after(module, args, kwargs, output):
        step = len(timesteps) - 1
        noise = output[0] if isinstance(output, tuple) else output.sample
        shapes[f"step_{step:02d}_noise"] = save(out, f"step_{step:02d}_noise", noise[0])

    pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
    pipe.transformer.register_forward_hook(after, with_kwargs=True)

    latents = pipe(
        prompt_embeds=prompt_embeds.to(torch.float16),
        prompt_embeds_mask=prompt_embeds_mask,
        height=arguments.size,
        width=arguments.size,
        num_inference_steps=arguments.steps,
        true_cfg_scale=1.0,
        generator=torch.Generator("cpu").manual_seed(arguments.seed),
        output_type="latent",
    ).images
    shapes["final_latents"] = save(out, "final_latents", latents[0])
    shapes["noise"] = list(np.fromfile(out / "step_00_latents.f32", dtype="<f4").shape)
    (out / "noise.f32").write_bytes((out / "step_00_latents.f32").read_bytes())

    rope = captured["rope"]  # complex [rows, 64]
    table = torch.cat([rope.real, rope.imag], dim=-1)
    shapes["rope"] = save(out, "rope", table)

    metadata = {
        "prompt": arguments.prompt,
        "size": arguments.size,
        "steps": arguments.steps,
        "seed": arguments.seed,
        "text_rows": int(prompt_embeds.shape[1]),
        "image_rows": int(latents.shape[1]),
        "transformer_timesteps": timesteps,
        "scheduler_sigmas": [float(s) for s in pipe.scheduler.sigmas],
        "shapes": shapes,
        "transformer": arguments.gguf.name,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "shapes"}, indent=2))


if __name__ == "__main__":
    main()
