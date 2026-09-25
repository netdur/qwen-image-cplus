---
license: other
license_name: qwen-research
license_link: LICENSE
base_model:
  - Qwen/Qwen-Image-2.1
  - Viggle/Qwen-Image-2.1-viggle-turbo
pipeline_tag: text-to-image
tags:
  - qwen-image
  - apple-silicon
  - metal
  - qipack
  - c-plus
---

# Qwen-Image-2.1 QIPACK for Apple Silicon

**Built with Qwen.** These are ready-to-map QIPACK transformer files for
[`qwen-image-cplus`](https://github.com/netdur/qwen-image-cplus), a native C+
and Metal Qwen-Image-2.1 inference runtime for Apple Silicon.

This repository contains the converted transformer weights and the unmodified
processor, text encoder, and VAE files needed by `qwen-image-cplus`. All files
come from the pinned upstream Qwen snapshot except the distilled transformer,
which comes from the pinned Viggle snapshot. The runtime provides the scheduler.

## Files

| File | Source | Default generation policy | SHA-256 |
| --- | --- | --- | --- |
| `qwen-image-2.1-fp16-v4.qipack` | `Qwen/Qwen-Image-2.1` at `b3179ad355be050328e483a9dfdd9e60cd62adfa` | 40 steps, TaylorSeer | `9fe30bcae5678c6e21618d48d9f058fc150ed5b71b7534e5283bb7a05c163750` |
| `qwen-image-2.1-viggle-v0.1-4step-fp16-v4.qipack` | `Viggle/Qwen-Image-2.1-viggle-turbo` at `bafc91e4cc934f5fb1406b22496a0bed9b99c548` | 4 steps, no cache, unstretched schedule | `d05edecbf9d3e7b03b0e708ae41da5370d5fe245f1d3b5f6775aef18f18857d1` |

Both packs share `processor/vocab.json`, `processor/merges.txt`, four
`text_encoder/model-*.safetensors` shards, and
`vae/diffusion_pytorch_model.safetensors` in this repository's root directory.
Their sizes and checksums are recorded in `manifest.json`. The files remain in
those paths when downloaded, so selecting either root-level QIPACK in the GUI
also locates its support files. One 14.23 GB pack plus the 18.89 GB shared
support files requires about 33.12 GB of local storage. “7B” describes the
transformer's parameter count, not the complete pipeline's size in bytes.

The distilled file is specifically the Viggle v0.1 four-step full transformer,
not its LoRA and not the newer v0.2.1 six-step release.

Both files use QIPACK1 version 1 with policy
`transformer:all-matrix-f16-v4`. All 224 transformer-block matrices are stored
as FP16; vectors and the nine global tensors remain BF16. Each pack contains
297 tensors and 7,115,124,736 parameters. The format has fixed little-endian
metadata plus per-tensor and payload checksums, and both packs passed exact
round-trip verification against their source tensors.

## Requirements

- macOS 14 or newer on Apple Silicon
- [`qwen-image-cplus`](https://github.com/netdur/qwen-image-cplus) v0.0.1 or newer

The runtime has been tested on an M1 Max with 32 GB unified memory. Lower-memory
machines have not yet been validated. Generation at 1024x1024 is supported;
the model's native 2048x2048 path is not yet supported by this runtime.

## Download

Install the Hugging Face CLI, accept the model license, and download this
repository into one directory:

```sh
hf download netdur/Qwen-Image-2.1-QIPACK --local-dir models
```

## Use

Base model, using the pack's 40-step TaylorSeer defaults:

```sh
qwen-image-cplus generate-1024 \
  models/qwen-image-2.1-fp16-v4.qipack \
  models \
  output.png \
  "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET'" \
  1301
```

Four-step distilled model:

```sh
qwen-image-cplus generate-1024 \
  models/qwen-image-2.1-viggle-v0.1-4step-fp16-v4.qipack \
  models \
  output.png \
  "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET'" \
  1301
```

The pack metadata selects the correct step count, terminal-shift behavior, and
cache policy when those optional CLI arguments are omitted.

## Reference performance

On the development M1 Max, the adopted Viggle four-step path generated a
1024x1024 PNG end to end in roughly 38–39 seconds. This is a single-machine,
single-prompt reference rather than a general performance guarantee. See the
versioned benchmark records in the runtime repository for exact conditions and
accuracy gates.

## License and modifications

The model materials are distributed under the included
[Qwen Research License Agreement](LICENSE), which permits **non-commercial
research and evaluation only** unless a separate commercial license is
obtained from Qwen. Read the agreement before downloading or using the files.

The QIPACK files are modified redistributions: their tensor storage layout and
selected matrix dtypes were converted for the Apple-Silicon runtime. The model
architecture and learned values were not retrained by this project. The
distilled source model was created by Viggle and is separately attributed in
[`NOTICE`](NOTICE).

The runtime source code has its own MIT license; that license does not replace
or relax the model license.
