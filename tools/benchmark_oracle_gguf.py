#!/usr/bin/env python3
"""Time the pinned Diffusers pipeline end to end with a GGUF transformer.

This is the speed baseline for the Linux engine on small GPUs. It runs the
official QwenImage21Pipeline (Diffusers 80c7ed2) with a GGUF-quantized
transformer, streams the 16.5 GB BF16 text encoder layer by layer from host
memory, and moves the transformer and VAE to the GPU only while they run.
Every phase is timed with CUDA synchronization, and the total is measured from
interpreter start, so model loading is included.

Development tool only; the C+ product never imports it.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKLQwenImage21,
    GGUFQuantizationConfig,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.loaders import single_file_model  # noqa: E402

# The pinned Diffusers lists only the v1 QwenImageTransformer2DModel as
# single-file loadable. The GGUF already uses Diffusers tensor names for all
# 297 tensors, so the 2.1 class takes the same identity mapping the v1 entry
# uses; the rest of from_single_file (GGUF quantizer, loading) is unchanged.
single_file_model.SINGLE_FILE_LOADABLE_CLASSES.setdefault(
    "QwenImage21Transformer2DModel",
    {"checkpoint_mapping_fn": lambda checkpoint, **kwargs: checkpoint, "default_subfolder": "transformer"},
)

DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
PROMPT = 'a travel poster with the headline "CASABLANCA" and the tagline "MEET ME AT SUNSET"'
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def synchronized() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def stream_text_encoder(text_encoder) -> None:
    """Keep the language model's decoder layers and the LM head in host memory
    and move each one to the GPU only for its own forward pass. Embeddings,
    the final norm, and RoPE stay on the GPU; the unused vision tower stays on
    the CPU."""
    language_model = text_encoder.model.language_model
    cuda = torch.device("cuda")

    def load(module, _args):
        module.to(cuda, non_blocking=True)

    def unload(module, _args, _output):
        module.to("cpu")

    for module in [*language_model.layers, text_encoder.lm_head]:
        module.to("cpu")
        module.register_forward_pre_hook(load)
        module.register_forward_hook(unload)
    language_model.embed_tokens.to(cuda)
    language_model.norm.to(cuda)
    language_model.rotary_emb.to(cuda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Diffusers folder (configs, text encoder, VAE)")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--compute-dtype", choices=DTYPES, default="float16",
                        help="GGUF dequantization and activation dtype for the transformer")
    parser.add_argument("--text-dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--vae-dtype", choices=DTYPES, default="float32")
    parser.add_argument("--vae-tiling", action="store_true",
                        help="decode with the VAE's tiled decoder (needed at 1024x1024 on 6 GB)")
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    phases: dict[str, float] = {}
    started = synchronized()
    transformer = QwenImage21Transformer2DModel.from_single_file(
        str(arguments.gguf),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPES[arguments.compute_dtype]),
        config=str(arguments.model),
        subfolder="transformer",
        torch_dtype=DTYPES[arguments.compute_dtype],
    )
    vae = AutoencoderKLQwenImage21.from_pretrained(
        str(arguments.model), subfolder="vae", torch_dtype=DTYPES[arguments.vae_dtype]
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        str(arguments.model), transformer=transformer, vae=vae, torch_dtype=DTYPES[arguments.text_dtype]
    )
    # The text encoder cannot fit in 6 GB, so it streams module by module from
    # host memory, and the prompt is encoded before denoising. The transformer
    # and VAE then each move to the GPU only while they run.
    stream_text_encoder(pipe.text_encoder)
    if arguments.vae_tiling:
        pipe.vae.enable_tiling()
    phases["load"] = synchronized() - started

    torch.cuda.reset_peak_memory_stats()
    begin = synchronized()
    # The pipeline's __call__ runs under no_grad; this direct call must too, or
    # autograd keeps every streamed layer's GPU weights alive.
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, _image_pad_mask = pipe.encode_prompt(
            prompt=arguments.prompt, device=torch.device("cuda")
        )
    phases["text"] = synchronized() - begin
    # The pipeline creates latents in the embeddings' dtype, and the GGUF
    # transformer computes in its compute dtype, so the two must match.
    embed_peak = float(prompt_embeds.float().abs().max())
    print(f"prompt embeddings: {tuple(prompt_embeds.shape)} {prompt_embeds.dtype}, max |x| {embed_peak:.1f}", flush=True)
    prompt_embeds = prompt_embeds.to(DTYPES[arguments.compute_dtype])
    text_encoder = pipe.text_encoder
    pipe.text_encoder = None
    pipe.register_modules(text_encoder=None)
    del text_encoder
    begin = synchronized()
    pipe.enable_model_cpu_offload()
    phases["offload_setup"] = synchronized() - begin

    decode = pipe.vae.decode

    def timed_decode(*args, **kwargs):
        begin = synchronized()
        result = decode(*args, **kwargs)
        phases["vae"] = synchronized() - begin
        return result

    pipe.vae.decode = timed_decode
    steps: list[float] = []
    step_clock = {"last": None}

    def on_step_end(pipeline, index, timestep, callback_kwargs):
        now = synchronized()
        steps.append(now - step_clock["last"])
        step_clock["last"] = now
        return callback_kwargs

    original_prepare = pipe.prepare_latents

    def timed_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        step_clock["last"] = synchronized()
        return result

    pipe.prepare_latents = timed_prepare

    call_start = synchronized()
    image = pipe(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        height=arguments.size,
        width=arguments.size,
        num_inference_steps=arguments.steps,
        true_cfg_scale=1.0,
        generator=torch.Generator("cpu").manual_seed(arguments.seed),
        callback_on_step_end=on_step_end,
    ).images[0]
    phases["pipeline_call"] = synchronized() - call_start
    save_start = time.perf_counter()
    image.save(arguments.output)
    phases["png"] = time.perf_counter() - save_start
    total = time.perf_counter() - PROCESS_START

    result = {
        "diffusers_commit": DIFFUSERS_COMMIT,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gguf": arguments.gguf.name,
        "prompt": arguments.prompt,
        "size": arguments.size,
        "steps": arguments.steps,
        "seed": arguments.seed,
        "compute_dtype": arguments.compute_dtype,
        "text_dtype": arguments.text_dtype,
        "vae_dtype": arguments.vae_dtype,
        "vae_tiling": arguments.vae_tiling,
        "phases_seconds": phases,
        "denoise_steps_seconds": steps,
        "denoise_seconds": sum(steps),
        "end_to_end_seconds_from_process_start": total,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "output": str(arguments.output),
    }
    print(json.dumps(result, indent=2))
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
