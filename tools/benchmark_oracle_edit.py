#!/usr/bin/env python3
"""Time the pinned Diffusers pipeline end to end on an image edit with a GGUF
transformer, under the same 6 GB memory plan as benchmark_oracle_gguf.py.

The pipeline encodes the prompt itself (condition images need the image mask
its own encode returns), so the text encoder's decoder layers stream through
the GPU from hooks, its vision tower sits on the GPU, and model offload
manages only the transformer and the VAE. Development tool only.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import benchmark_oracle_gguf as oracle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, nargs="+", help="one or more condition images")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512, help="output side and condition-image area side")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--group-offload", action="store_true",
                        help="stream the transformer block by block (needed for 1024 edits on 6 GB)")
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    phases: dict[str, float] = {}
    started = oracle.synchronized()
    transformer = oracle.QwenImage21Transformer2DModel.from_single_file(
        str(arguments.gguf),
        quantization_config=oracle.GGUFQuantizationConfig(compute_dtype=torch.float16),
        config=str(arguments.model),
        subfolder="transformer",
        torch_dtype=torch.float16,
    )
    vae = oracle.AutoencoderKLQwenImage21.from_pretrained(str(arguments.model), subfolder="vae", torch_dtype=torch.float32)
    pipe = oracle.QwenImage21Pipeline.from_pretrained(
        str(arguments.model), transformer=transformer, vae=vae, torch_dtype=torch.float16
    )
    text_encoder = pipe.text_encoder
    oracle.stream_text_encoder(text_encoder)
    text_encoder.model.visual.to("cuda")
    # Offload hooks only for the transformer and the VAE (or, with
    # --group-offload, the VAE alone; the transformer then streams its blocks).
    pipe.text_encoder = None
    pipe.register_modules(text_encoder=None)
    if arguments.group_offload:
        transformer = pipe.transformer
        pipe.transformer = None
        pipe.register_modules(transformer=None)
        pipe.enable_model_cpu_offload()
        pipe.register_modules(transformer=transformer)
        transformer.enable_group_offload(
            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
            offload_type="block_level", num_blocks_per_group=1, use_stream=True,
        )
        # The end-of-call hook reset re-applies model offload, which Diffusers
        # refuses next to group offload; nothing needs resetting here.
        pipe.maybe_free_model_hooks = lambda: None
    else:
        pipe.enable_model_cpu_offload()
    pipe.register_modules(text_encoder=text_encoder)
    if arguments.vae_tiling:
        pipe.vae.enable_tiling()
    phases["load"] = oracle.synchronized() - started

    encode = pipe.encode_prompt

    def timed_encode(*args, **kwargs):
        begin = oracle.synchronized()
        result = encode(*args, **kwargs)
        phases["text_and_vision"] = oracle.synchronized() - begin
        # The transformer needs the room: release the encoder's resident parts.
        language_model = text_encoder.model.language_model
        text_encoder.model.visual.to("cpu")
        language_model.embed_tokens.to("cpu")
        language_model.norm.to("cpu")
        language_model.rotary_emb.to("cpu")
        torch.cuda.empty_cache()
        return result

    pipe.encode_prompt = timed_encode
    vae_encode = pipe._encode_vae_image

    def timed_vae_encode(image, generator):
        # The pipeline hands the condition image over in the embeddings' dtype
        # (FP16); the VAE runs in FP32 and its latents go back as FP16.
        begin = oracle.synchronized()
        result = vae_encode(image.to(torch.float32), generator).to(image.dtype)
        # Model offload expects the VAE after the transformer; the edit uses it
        # first, so move it out before the transformer loads (its hook brings
        # it back for decoding).
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        phases["vae_encode"] = oracle.synchronized() - begin
        return result

    pipe._encode_vae_image = timed_vae_encode
    decode = pipe.vae.decode

    def timed_decode(*args, **kwargs):
        begin = oracle.synchronized()
        result = decode(*args, **kwargs)
        phases["vae_decode"] = oracle.synchronized() - begin
        return result

    pipe.vae.decode = timed_decode
    steps: list[float] = []
    clock = {"last": None}

    def on_step_end(pipeline, index, timestep, callback_kwargs):
        now = oracle.synchronized()
        if clock["last"] is not None:
            steps.append(now - clock["last"])
        clock["last"] = now
        return callback_kwargs

    original_prepare = pipe.prepare_latents

    def timed_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        clock["last"] = oracle.synchronized()
        return result

    pipe.prepare_latents = timed_prepare

    images = [Image.open(path) for path in arguments.image]
    begin = oracle.synchronized()
    result = pipe(
        prompt=arguments.prompt,
        image=images,
        height=arguments.size,
        width=arguments.size,
        num_inference_steps=arguments.steps,
        true_cfg_scale=1.0,
        generator=torch.Generator("cpu").manual_seed(arguments.seed),
        output_resolution=arguments.size,
        callback_on_step_end=on_step_end,
    ).images[0]
    phases["pipeline_call"] = oracle.synchronized() - begin
    result.save(arguments.output)
    total = time.perf_counter() - PROCESS_START
    report = {
        "image": [str(path) for path in arguments.image],
        "prompt": arguments.prompt,
        "size": arguments.size,
        "steps": arguments.steps,
        "seed": arguments.seed,
        "phases_seconds": phases,
        "denoise_steps_seconds": steps,
        "end_to_end_seconds_from_process_start": total,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    print(json.dumps(report, indent=2))
    if arguments.json:
        arguments.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
