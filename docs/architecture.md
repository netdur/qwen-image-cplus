# Package and API architecture

## Package and API layout

One product, built from C+ packages that sit side by side at the repository
root, in the layered shape C+'s own `facet` family uses: a portable core, one
backend per GPU API, and a runtime that picks the backend for the platform.

- `qwen_image/` is the portable core: the public API (`qwen_image/api`),
  generation control, QIPACK metadata (`pack_info`), positional file I/O, and
  the engine seam (`engine`) a backend fills. It names no platform and links
  nothing.
- `qwen_image_metal/` is the Metal backend (macOS): inference, model I/O,
  Metal kernels, scheduling, caching, VAE, and PNG output.
- `qwen_image_cuda/` is the CUDA backend (Linux today; the backend is CUDA,
  not Linux, so a Windows port adds platform files and a link table).
- `qwen_image_runtime/` installs the backend for the platform being built:
  `runtime_macos` installs Metal, `runtime_linux` installs CUDA, and any other
  target gets a neutral runtime under which the API reports that no backend
  exists.
- `cli/` is the `qwen-image-cplus` command-line client: the same source on
  every platform, over the public API only.
- `gui/` is the AppKit generation client (macOS), packaged as
  `Qwen Image.app` in the release archive.
- `ffi/` is a thin C-ABI adapter over the same API. C+ generates its
  `qwen_image.h`; there is no separately maintained handwritten header.
- `qwen_image_metal_dev/` and `qwen_image_cuda_dev/` hold each backend's
  development commands (probes, stage validation, benchmarks, pack tools);
  `qwen_image_quantize/` writes the CUDA backend's INT4 packs.

Applications depend on `qwen_image` and `qwen_image_runtime`, call
`runtime::install()` once, and then use `qwen_image/api`. The `qwen_image*`
packages resolve as sibling directories, so each package's `vendor/` holds
only third-party C+ packages (`scripts/link_vendor.sh` points them at a shared
C+ vendor folder).

This separation follows C+'s two library forms. An entry-less package is the
native, prebuilt library consumed by another C+ package. A `[library]` target
is a C-ABI product whose explicit entry exports bare symbols and generates a C
header. They cannot be the same package target, so `ffi/` adapts rather than
duplicates the engine. A Homebrew formula can still install all artifacts
from one repository and one formula.

The first public generation API is deliberately synchronous and file-oriented:
it accepts a packed transformer path, the model snapshot root, an output PNG
path, prompt, resolution, step count, seed, and cache policy. This preserves
the runtime's current phase-scoped memory behavior. It is not yet a resident
engine/session API; adding a reusable loaded-model handle is a later API
extension, not something callers should infer from the current surface.

Image-conditioned generation is exposed by the same engine as
`MultiImageGenerateRequest`/`generate_multi_image_to_png` for C+ and by
`QiMultiImageGenerateRequest`/`qi_generate_multi_image_to_png` for C. Those
original entry points produce square output (`pixels` 512 or 1024). The C+
`MultiImageSizedGenerateRequest`/`generate_multi_image_sized_to_png` entry point
and `generate-multi-image-sized` CLI command accept rectangular output within
the same 1 MP area limit; the C ABI remains square-only for now. All accept
1-10 borrowed image paths and take the same step counts as
text-to-image generation: 3, 4, 6, 8, 25, or 40. The pack's `shift_terminal`
metadata selects the schedule, so a distilled pack runs its own few-step
schedule. The CLI accepts `pack` in place of a step count to use the pack's
`steps` metadata. One and two steps are truncated prefixes of the pack's
schedule; they remain available to `test-multi-image-transformer` as K/V-cache
smokes, but the engine refuses to write a PNG from them because the latent is
still mostly noise.

Multiple references already share one vision/VAE weight load, and the
transformer packs their rows into one joint prefill. The encoders run images
serially, but parallelizing them can save only part of conditioning time; the
denoising steps depend on one another. For the 736x1280 Viggle v0.2.1 two-image
hat edit, a profiling-only two-step ablation found flash attention responsible
for much of the extra prefill work. The block-causal flash kernel now skips
entire key tiles that are invisible to all eight queries in a group. In two
adjacent baseline/pruned comparisons, prefill GPU time changed from
43.53 to 37.94 s and from 63.14 to 52.24 s, while the cached step was
essentially unchanged. The large drift between pairs prevents an end-to-end
speed claim. The six-step pruned output was byte-identical to the unpruned PNG
(SHA-256 `686939c6d20371e7742ae5943f91454a05c5188e067463e8b2b45ad6985cdefa`).
`QI_DISABLE_FLASH_TILE_PRUNING=1` restores the previous kernel behavior.
`benchmark-multi-image-sized PACKED MODEL_DIR - PROMPT SEED 1|2 WIDTH HEIGHT IMAGE...`
is a transformer-only diagnostic that accepts `QI_PROFILE_SKIP=attention`;
it does not write an image.

With the adopted Viggle 4-step pack (`pack` steps), two 512-area references
(a red circle and a yellow/green block layout) and seed 1301 produced a
correctly composed 512x512 PNG in **63.9 s end to end**: 15.2 s conditioning,
41.5 s trajectory (first step 21.2 s, cached-prefix steps about 6.75 s), 0.8 s
VAE, and a 17.5 GB peak footprint. Flash attention for every shape (see
[model behavior](model-behavior.md))
brings the same request to **34.5 s**.

The C ABI is versioned and self-describing. Callers set both `abi_version` and
`struct_size`, pass strings as pointer-length pairs, and receive a typed
`QiStatus`. String storage only has to remain alive for the synchronous call.

```c
#include "qwen_image.h"

QiGenerateRequest request = {0};
request.abi_version = qi_abi_version();
request.struct_size = qi_generate_request_size();
request.packed_path = (uint8_t *)packed;
request.packed_path_length = packed_length;
request.model_root = (uint8_t *)model_root;
request.model_root_length = model_root_length;
request.output_path = (uint8_t *)output_path;
request.output_path_length = output_path_length;
request.prompt = (uint8_t *)prompt;
request.prompt_length = prompt_length;
request.width = 1344;
request.height = 768;
request.steps = 40;
request.seed = 1301;
request.cache_mode = QiCacheMode_None;

QiStatus status = qi_generate_to_png(&request);
```

For image conditioning, paths and lengths are parallel borrowed arrays and
must remain alive until the synchronous call returns:

```c
uint8_t *images[] = {(uint8_t *)first_path, (uint8_t *)second_path};
size_t image_lengths[] = {first_path_length, second_path_length};

QiMultiImageGenerateRequest request = {0};
request.abi_version = qi_abi_version();
request.struct_size = qi_multi_image_generate_request_size();
request.packed_path = (uint8_t *)packed;
request.packed_path_length = packed_length;
request.model_root = (uint8_t *)model_root;
request.model_root_length = model_root_length;
request.output_path = (uint8_t *)output_path;
request.output_path_length = output_path_length;
request.prompt = (uint8_t *)prompt;
request.prompt_length = prompt_length;
request.image_paths = images;
request.image_path_lengths = image_lengths;
request.image_count = 2;
request.pixels = 512;
request.steps = 40;
request.seed = 1301;

QiStatus status = qi_generate_multi_image_to_png(&request);
```

`width` and `height` were appended without changing ABI version 1. The library
checks `struct_size` before reading them, so a binary built against the original
request layout remains valid and continues to use `pixels` as a square width
and height. New callers set both dimensions; setting only one is invalid.
