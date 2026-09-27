#!/usr/bin/env python3
"""Dump the pinned pipeline's intermediates for one image edit, as raw
little-endian files the Linux engine's stages are checked against:

  input_ids.i32            full template token IDs (system prefix included)
  image_grid_thw.i32       [images, 3] patch grid (t, h, w) per condition image
  pixel_values.f32         [patches, 1536] vision processor output
  vae_input.f32            [4, H, W] VAE condition input in [-1, 1]
                           (vae_input_N.f32 for image N > 0)
  vision_embeds.f32        [merged tokens, 4096] merger output
  deepstack_N.f32          [merged tokens, 4096] for N = 0, 1, 2
  position_ids.i32         [3, tokens] text-encoder M-RoPE positions
  prompt_embeds.f32        [rows, 4096] after the 14 system tokens
  image_pad_mask.u8        [rows] image-token positions in prompt_embeds
  condition_latents.f32    [condition rows, 64] packed, normalized
  noise.f32                [target rows, 64]
  rope.f32                 [joint rows, 128] transformer RoPE (cos | sin)
  step_NN_latents.f32 / step_NN_noise.f32
  final_latents.f32, image.png, metadata.json

Same GGUF transformer and memory plan as benchmark_oracle_edit.py.
Development tool only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import benchmark_oracle_gguf as oracle


def save(directory: Path, name: str, tensor: torch.Tensor, dtype: str = "<f4") -> list[int]:
    value = tensor.detach().cpu().contiguous()
    value = value.float().numpy() if dtype == "<f4" else value.numpy()
    value.astype(dtype).tofile(directory / f"{name}.{ {'<f4': 'f32', '<i4': 'i32', 'u1': 'u8'}[dtype] }")
    return list(value.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, nargs="+", help="one or more condition images")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1301)
    arguments = parser.parse_args()
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, list[int]] = {}

    transformer = oracle.QwenImage21Transformer2DModel.from_single_file(
        str(arguments.gguf),
        quantization_config=oracle.GGUFQuantizationConfig(compute_dtype=torch.float16),
        config=str(arguments.model), subfolder="transformer", torch_dtype=torch.float16,
    )
    vae = oracle.AutoencoderKLQwenImage21.from_pretrained(str(arguments.model), subfolder="vae", torch_dtype=torch.float32)
    pipe = oracle.QwenImage21Pipeline.from_pretrained(
        str(arguments.model), transformer=transformer, vae=vae, torch_dtype=torch.float16
    )
    text_encoder = pipe.text_encoder
    oracle.stream_text_encoder(text_encoder)
    text_encoder.model.visual.to("cuda")
    pipe.text_encoder = None
    pipe.register_modules(text_encoder=None)
    pipe.enable_model_cpu_offload()
    pipe.register_modules(text_encoder=text_encoder)

    # Processor output and text-encoder inputs.
    processor_call = pipe.processor.__call__

    def capture_processor(*args, **kwargs):
        result = processor_call(*args, **kwargs)
        shapes["input_ids"] = save(out, "input_ids", result.input_ids[0].to(torch.int32), "<i4")
        shapes["image_grid_thw"] = save(out, "image_grid_thw", result.image_grid_thw.to(torch.int32), "<i4")
        shapes["pixel_values"] = save(out, "pixel_values", result.pixel_values)
        return result

    pipe.processor.__call__ = capture_processor
    pipe.processor.__class__.__call__ = lambda self, *a, **k: capture_processor(*a, **k)

    visual = text_encoder.model.visual

    def after_visual(module, args, output):
        shapes["vision_embeds"] = save(out, "vision_embeds", output.pooler_output)
        for index, feature in enumerate(output.deepstack_features):
            shapes[f"deepstack_{index}"] = save(out, f"deepstack_{index}", feature)

    visual.register_forward_hook(after_visual)
    rope_index = text_encoder.model.get_rope_index

    def capture_rope_index(*args, **kwargs):
        position_ids, deltas = rope_index(*args, **kwargs)
        shapes["position_ids"] = save(out, "position_ids", position_ids[:, 0].to(torch.int32), "<i4")
        return position_ids, deltas

    text_encoder.model.get_rope_index = capture_rope_index

    encode = pipe.encode_prompt

    def capture_encode(*args, **kwargs):
        embeds, mask, image_pad_mask = encode(*args, **kwargs)
        shapes["prompt_embeds"] = save(out, "prompt_embeds", embeds[0])
        shapes["image_pad_mask"] = save(out, "image_pad_mask", image_pad_mask[0].to(torch.uint8), "u1")
        language_model = text_encoder.model.language_model
        text_encoder.model.visual.to("cpu")
        language_model.embed_tokens.to("cpu")
        language_model.norm.to("cpu")
        language_model.rotary_emb.to("cpu")
        torch.cuda.empty_cache()
        return embeds, mask, image_pad_mask

    pipe.encode_prompt = capture_encode
    vae_encode = pipe._encode_vae_image

    vae_calls = [0]

    def capture_vae_encode(image, generator):
        name = "vae_input" if vae_calls[0] == 0 else f"vae_input_{vae_calls[0]}"
        vae_calls[0] += 1
        shapes[name] = save(out, name, image[0, :, 0])
        result = vae_encode(image.to(torch.float32), generator).to(image.dtype)
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return result

    pipe._encode_vae_image = capture_vae_encode
    original_prepare = pipe.prepare_latents

    def capture_prepare(*args, **kwargs):
        latents, condition = original_prepare(*args, **kwargs)
        shapes["noise"] = save(out, "noise", latents[0])
        shapes["condition_latents"] = save(out, "condition_latents", condition[0])
        return latents, condition

    pipe.prepare_latents = capture_prepare
    captured: dict[str, torch.Tensor] = {}

    def capture_rope(module, args, output):
        # Saved at once, so a later out-of-memory failure still leaves it.
        if "rope" not in captured:
            captured["rope"] = output.detach()
            shapes["rope"] = save(out, "rope", torch.cat([output.real, output.imag], dim=-1))

    pipe.transformer.pos_embed.register_forward_hook(capture_rope)
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
        prompt=arguments.prompt, image=[Image.open(path) for path in arguments.image], height=arguments.size, width=arguments.size,
        num_inference_steps=arguments.steps, true_cfg_scale=1.0, output_resolution=arguments.size,
        generator=torch.Generator("cpu").manual_seed(arguments.seed), output_type="latent",
    ).images
    shapes["final_latents"] = save(out, "final_latents", latents[0])
    metadata = {
        "prompt": arguments.prompt, "image": [str(path) for path in arguments.image], "size": arguments.size, "steps": arguments.steps,
        "seed": arguments.seed, "transformer_timesteps": timesteps, "shapes": shapes,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
