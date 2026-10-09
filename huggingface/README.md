---
license: other
license_name: qwen-research
license_link: LICENSE
base_model:
  - Qwen/Qwen-Image-2.1
  - Qwen/Qwen-Image-2.1-Turbo
  - Viggle/Qwen-Image-2.1-viggle-turbo
pipeline_tag: text-to-image
tags:
  - qwen-image
  - apple-silicon
  - metal
  - cuda
  - nvidia
  - linux
  - int4
  - qipack
  - c-plus
---

# Qwen-Image-2.1 QIPACK for Apple Silicon and NVIDIA GPUs

**Built with Qwen.** These are ready-to-map QIPACK transformer files for
[`qwen-image-cplus`](https://github.com/netdur/qwen-image-cplus), a native C+
Qwen-Image-2.1 inference runtime: Metal on Apple Silicon, CUDA on NVIDIA GPUs
under Linux.

This repository contains converted transformer weights (FP16 for Apple
Silicon, 4-bit for NVIDIA), a Viggle v0.2.1 LoRA, and the unmodified processor,
text encoder, and VAE files needed by `qwen-image-cplus`. The base weights and shared support files come from the
pinned upstream Qwen snapshot; the eight-step pack is Qwen's official
Qwen-Image-2.1-Turbo; the four-step full fine-tune and six-step LoRA come from
Viggle. The runtime provides the scheduler.

## Files

| File | Source | Default generation policy | SHA-256 |
| --- | --- | --- | --- |
| `qwen-image-2.1-fp16-v4.qipack` | `Qwen/Qwen-Image-2.1` at `b3179ad355be050328e483a9dfdd9e60cd62adfa` | 40 steps, TaylorSeer | `9fe30bcae5678c6e21618d48d9f058fc150ed5b71b7534e5283bb7a05c163750` |
| `qwen-image-2.1-viggle-v0.1-4step-fp16-v4.qipack` | `Viggle/Qwen-Image-2.1-viggle-turbo` at `bafc91e4cc934f5fb1406b22496a0bed9b99c548` | 4 steps, no cache, unstretched schedule | `d05edecbf9d3e7b03b0e708ae41da5370d5fe245f1d3b5f6775aef18f18857d1` |
| `qwen-image-2.1-viggle-v0.2.1-lora-fp16-v4.qipack` | Unchanged Qwen base transformer; requires the Viggle LoRA below | 6 steps, no cache, Viggle v0.2.1 schedule | `be2e72ed75d30234a7f1934502d1d95b10eaedc935de2ce06b926b34e1b7b8b4` |
| `qwen-image-2.1-turbo-8step-fp16-v4.qipack` | `Qwen/Qwen-Image-2.1-Turbo` at `d65dbc9a7e8f6b5479e33dee6030eaab2a906509` | 8 steps, no cache, the checkpoint's own sigmas | `6d38ebf0f025dcd08429832d0b1619b94f9c105a129e07980c96e21013a048cc` |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors` | `Viggle/Qwen-Image-2.1-viggle-turbo` at `139e9492e6b81e85395877a549ec8f0afbb18f8f` | Rank-256 adapter beside the six-step QIPACK | `2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803` |
| `qwen-image-2.1-viggle-v0.1-4step-w4a4-h256-g64-clip-v6.qipack` | The four-step FP16 pack above, quantized to W4A4 | NVIDIA only: 4 steps, no cache | `6250f581430854d0dde0f11740b8b3644e7ce2ac1dc2f074b538d341f8e582e8` |
| `qwen-image-2.1-viggle-v0.1-4step-w4a16-g64-v6.qipack` | The four-step FP16 pack above, quantized to W4A16 | NVIDIA only: 4 steps, no cache | `8d837585b3204bc253bd9002d74d8cbf278e93efa50288850fb9f36790104063` |

All packs share `processor/vocab.json`, `processor/merges.txt`, four
`text_encoder/model-*.safetensors` shards, and
`vae/diffusion_pytorch_model.safetensors` in this repository's root directory.
Their sizes and checksums are recorded in `manifest.json`. The files remain in
those paths when downloaded, so selecting a root-level QIPACK in the GUI
also locates its support files. The six-step pack additionally requires its
LoRA file beside it. One 14.23 GB pack plus the 18.89 GB shared files requires
about 33.12 GB of local storage, or 34.48 GB with the 1.36 GB LoRA. “7B”
describes the transformer's parameter count, not the pipeline's size in bytes.

The eight-step file is the transformer of Qwen's official
Qwen-Image-2.1-Turbo. That checkpoint's text encoder matches the base one, and
its BF16 VAE is the base FP32 VAE rounded to BF16, so it uses the same shared
support files. Its metadata selects the eight sampling sigmas the checkpoint
ships, used unshifted at every resolution; it runs only eight steps.

The four-step file is Viggle v0.1's full transformer, not a LoRA. The six-step
file contains the original Qwen base transformer with metadata selecting the
separate Viggle v0.2.1 LoRA and its six-step schedule.

The four FP16 QIPACK files use QIPACK1 version 1 with policy
`transformer:all-matrix-f16-v4`. All 224 transformer-block matrices are stored
as FP16; vectors and the nine global tensors remain BF16. Each pack contains
297 tensors and 7,115,124,736 parameters. The format has fixed little-endian
metadata plus per-tensor and payload checksums. The base, four-step, and
eight-step packs passed exact round-trip verification against their source
tensors.

### 4-bit packs for NVIDIA GPUs

The two `-v6` packs are the four-step Viggle transformer quantized for the CUDA
engine. Each keeps the QIPACK1 container and replaces only the 224
transformer-block matrices with 4-bit codes and one FP16 scale per group of 64
inputs; vectors and the nine global tensors are copied unchanged. A pack is
about 4 GB, so with the 18.89 GB of shared files one NVIDIA setup needs about
23 GB of storage.

- `w4a4-h256-g64-clip` (policy `transformer:w4a4-h256-g64-v6`): signed 4-bit
  weights rotated by a 256-point Hadamard transform, with activations
  quantized to 4 bits at run time, and a per-group clipping range chosen to
  minimize reconstruction error. The faster of the two.
- `w4a16-g64` (policy `transformer:w4a16-g64-v6`): 4-bit weights with an FP16
  minimum per group, FP16 activations. Stays closer to the FP16 result.

Measured against the FP16 pack on the same inputs (512x512, 4 steps), both
render the reference poster's text exactly. W4A16 keeps the composition closer
(21.2 dB PSNR against 19.0 dB for W4A4), and W4A4 is 1.4–1.6x faster end to
end. These packs are not used by the Apple Silicon runtime.

## Requirements

Apple Silicon (FP16 packs):

- macOS 14 or newer on Apple Silicon
- [`qwen-image-cplus`](https://github.com/netdur/qwen-image-cplus); the six-step
  LoRA requires a build with Viggle v0.2.1 support, and the eight-step Turbo
  pack requires v0.2.5 or newer

NVIDIA (4-bit packs):

- x86_64 Linux with an NVIDIA GPU, Turing (RTX 20xx) or newer, 6 GB of VRAM or
  more, and the proprietary driver
- [`qwen-image-cplus`](https://github.com/netdur/qwen-image-cplus) with the
  CUDA engine (the Ubuntu snap, or a build from current `main`)

The runtime has been tested on an M1 Max with 32 GB unified memory. Lower-memory
machines have not yet been validated. Generation at 1024x1024 and the model's
native 2048x2048 resolution is supported; 2048x2048 is a high-memory capacity
mode on a 32 GB M1 Max.

## Download

Install the Hugging Face CLI and read the model license. Download the shared
files and the pack you want into one directory. For an NVIDIA GPU, for example:

```sh
hf download netdur/Qwen-Image-2.1-QIPACK --local-dir models \
  --include "processor/*" "text_encoder/*" "vae/*" \
  "qwen-image-2.1-viggle-v0.1-4step-w4a4-h256-g64-clip-v6.qipack"
```

For Apple Silicon, name an FP16 pack instead (and, for the six-step pack, its
LoRA file). Leaving out `--include` downloads everything, about 85 GB.

## Use

The command is the same on both platforms; the arguments after the prompt are
width, height, and seed. Base model, using the pack's 40-step TaylorSeer
defaults:

```sh
qwen-image-cplus generate \
  models/qwen-image-2.1-fp16-v4.qipack \
  models \
  output.png \
  "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET'" \
  1024 1024 1301
```

Four-step distilled model:

```sh
qwen-image-cplus generate \
  models/qwen-image-2.1-viggle-v0.1-4step-fp16-v4.qipack \
  models \
  output.png \
  "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET'" \
  1024 1024 1301
```

Official eight-step Qwen-Image-2.1-Turbo:

```sh
qwen-image-cplus generate \
  models/qwen-image-2.1-turbo-8step-fp16-v4.qipack \
  models \
  output.png \
  'a travel poster with the headline "CASABLANCA" and the tagline "MEET ME AT SUNSET"' \
  1024 1024 1301
```

Six-step Viggle v0.2.1 LoRA (keep its sidecar safetensors beside the pack):

```sh
qwen-image-cplus generate \
  models/qwen-image-2.1-viggle-v0.2.1-lora-fp16-v4.qipack \
  models \
  output.png \
  "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET'" \
  1024 1024 1301
```

Four-step model on an NVIDIA GPU (use the `w4a16` pack for the higher-fidelity
mode):

```sh
qwen-image-cplus generate \
  models/qwen-image-2.1-viggle-v0.1-4step-w4a4-h256-g64-clip-v6.qipack \
  models \
  output.png \
  "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET'" \
  512 512 1301
```

The pack metadata selects the correct step count, terminal-shift behavior, and
cache policy when those optional CLI arguments are omitted.

## Reference performance

On the development M1 Max, the adopted Viggle four-step path generated a
1024x1024 PNG end to end in roughly 38–39 seconds, and the official eight-step
Turbo pack in 74.4 seconds with both quoted poster strings rendered exactly.

On an RTX 2060 (6 GB, PCIe gen3 x8), the four-step packs took, end to end with
a warm file cache:

| Task | Size | W4A4 | W4A16 |
| --- | --- | ---: | ---: |
| Text to image | 512x512 | ~7.2 s | ~10.9 s |
| Text to image | 1024x1024 | ~17.4 s | ~30.8 s |
| Image edit (1 image) | 512x512 | ~8.7 s | ~13.6 s |
| Image edit (1 image) | 1024x1024 | ~24.8 s | ~41.8 s |

These are single-machine, single-prompt references rather than general
performance guarantees. See the
versioned benchmark records in the runtime repository for exact conditions and
accuracy gates.

## License and modifications

The model materials are distributed under the included
[Qwen Research License Agreement](LICENSE), which permits **non-commercial
research and evaluation only** unless a separate commercial license is
obtained from Qwen. Read the agreement before downloading or using the files.

The QIPACK files are modified redistributions: their tensor storage layout and
selected matrix dtypes were converted for the runtime, and the `-v6` packs
quantize the transformer-block matrices to 4 bits. The model
architecture and learned values were not retrained by this project. The
Turbo checkpoint was published by Qwen; the Viggle distilled model and LoRA
were created by Viggle. Both are attributed in [`NOTICE`](NOTICE).

The runtime source code has its own MIT license; that license does not replace
or relax the model license.
