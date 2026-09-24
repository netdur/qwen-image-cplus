# qwen-image-cplus

Native Qwen-Image-2.1 inference work for Apple Silicon, written in C+ and
Metal Shading Language. The implementation target is an Apple M1 Max with
32 GB unified memory.

The runtime is intentionally model-specific. It does not embed Python,
PyTorch, Diffusers, C, C++, Objective-C source, or CMake. A separate Python
development tool may generate small oracle fixtures from the pinned official
Diffusers source; those fixtures are plain binary files consumed by C+ tests.

## Install

The supported binary distribution is macOS 14 or newer on Apple Silicon.
Homebrew downloads a prebuilt ARM64 CLI and C ABI; it does not install the C+
compiler or build this project on the user's machine.

Because this repository contains both the product and its formula rather than
using a separate `homebrew-*` repository, tap it with its explicit URL.
Homebrew 7 also requires explicit trust for formulae from repositories that do
not use the `homebrew-*` naming convention. Trust only this formula rather than
the whole tap:

```sh
brew tap netdur/qwen-image-cplus https://github.com/netdur/qwen-image-cplus.git
brew trust --formula netdur/qwen-image-cplus/qwen-image-cplus
brew install netdur/qwen-image-cplus/qwen-image-cplus
```

The installation contains:

```text
bin/qwen-image-cplus
include/qwen_image.h
lib/libqwen_image.a
lib/libqwen_image.dylib
```

### Model files

Model weights are intentionally not part of the Homebrew archive. The runtime
expects a QIPACK transformer and the corresponding Qwen-Image-2.1 snapshot at
paths supplied to the CLI or library request. Keeping models separate makes
application upgrades small and lets applications manage their own model
storage.

The QIPACK files are published at
[`netdur/Qwen-Image-2.1-QIPACK`](https://huggingface.co/netdur/Qwen-Image-2.1-QIPACK).
Their model card, Qwen license, required attribution, and artifact manifest
live in [`huggingface/`](huggingface/README.md) so the published metadata stays
versioned with the runtime.

## Pinned reference

- Model: `Qwen/Qwen-Image-2.1`
- Snapshot: `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- Diffusers commit: `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
- Machine-readable inventory: `manifests/qwen-image-2.1.json`
- Pinned official [transformer source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/transformers/transformer_qwenimage21.py),
  [VAE source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py),
  and [pipeline source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- Text model reference: Transformers 5.17.0
  [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

## Resolution and step-count policy

Qwen's own model documentation identifies **2048x2048** as the native and
recommended 1:1 output, with comparable 2K-pixel-budget dimensions for other
aspect ratios. The Diffusers implementation and Qwen's vLLM/SGLang examples
also accept 1024x1024, so 1024 is a supported and useful lower-resolution mode,
but it is not the model's native quality target. The runtime accepts independent
width and height values that are at least 256, divisible by 32, and no more
than 16,384 latent tokens in total (equivalently, at most 4,194,304 output
pixels). That includes 256x256, 512x512, 1024x1024, native 2048x2048, and
rectangular shapes within the same pixel budget. The 2048 path is functional,
but its measured 24.57 GB peak on a 32 GB M1 Max makes it a capacity mode, not
the practical performance default.

Rectangular generation carries the shape through latent allocation, the two
independent spatial RoPE coordinates, transformer scheduling, VAE decode, and
PNG output. Cache-DiT, TaylorSeer, and Bottleneck Sampling remain restricted
to the 256x256 and 1024x1024 shapes where their behavior was calibrated;
arbitrary dimensions currently use cache-off inference.

The first end-to-end rectangular acceptance run used the Viggle 4-step pack at
512x256 with seed 1301. It produced a correctly shaped PNG in **14.801 s**:
2.334 s text, 11.813 s transformer, 0.649 s VAE, and 4 ms PNG output. This is a
functional acceptance measurement on the M1 Max, not a claim that every aspect
ratio has the same throughput or quality calibration.

The native 2048 acceptance used the same distilled FP16 pack, four steps, and
seed 1301. It produced a finite 2048x2048 PNG in **309.664 s**: 2.219 s text,
295.868 s transformer phase, 11.459 s VAE, and 85 ms PNG output. Peak memory
footprint was 24,571,627,440 bytes with no swap. The first attempt found two
shape-dependent correctness bugs: attention projection had assumed exactly
4,096 image rows, and a single very tall `MPSMatrixMultiplication` produced
non-finite rows. Projection now uses the actual text-prefix length and matrix
work above 8,192 rows is encoded in verified 4,096-row slices. The benchmark
record is `benchmarks/m1-max-viggle-4step-2048.json`.

The authoritative Qwen and Diffusers sampling default is **40 Euler steps**.
The 25-step experiment in this repository came from the
[Comfy-Org workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json),
not an official Qwen recommendation. That template explicitly says the
official pipeline uses about 40-50 steps and that the template itself starts
at 25. Consequently, 40 remains this runtime's compatibility, oracle, and
default path for the base pack; 25 is only a measured opt-in speed/quality
tradeoff. It is not text-safe at 1024: on the poster prompt, 25 steps chose
to render CASABLANCA and misspelled it (see the Viggle comparison below).
Pack metadata can set another default. The adopted Viggle distillation pack
defaults to 4 steps with an unstretched schedule.

## Multi-image conditioning status

The official Qwen-Image-2.1 pipeline accepts **one to ten reference images**.
Native image-conditioned generation is under construction; it is not exposed
through the public generation API yet. Integration tests use a 512x512 output
budget so correctness work does not spend 1024 or 2048 generation time.

The first completed checkpoint reproduces the pinned Diffusers/Qwen3-VL input
contract: aspect-preserving area resize rounded to multiples of 32, one vision
token per merged 2x2 group of 16px patches, one VAE condition latent per 16x16
tile, the exact `<imageN>`/vision placeholder prompt layout, image-aware
three-axis MRoPE positions, image-embedding substitution, and DeepStack feature
injection after language layers 0, 1, and 2. For two square 512px references,
that is 256 vision tokens and 1,024 VAE latent tokens per image. Together they
produce 512 language-vision tokens and a 2,048-token VAE condition prefix; the
512px output itself is another 1,024 latent tokens.

The shape and MRoPE unit tests pass, the two-image tokenizer smoke produces the
expected 512 placeholders, and the existing text-only encoder oracle remains
unchanged at 0.0361069 output nRMSE. Remaining work is native image decode and
resize, the Qwen3-VL vision tower, the VAE encoder, transformer condition-prefix
assembly, and then CLI/C ABI exposure. Keeping those stages explicit avoids
calling prompt plumbing end-to-end image editing before pixels have passed
through both encoders.

## Package and API layout

The repository is split into three C+ packages, but remains one product:

- `qwen_image/` is the native engine package. It owns inference, model I/O,
  Metal kernels, scheduling, caching, VAE decode, and PNG output. Native C+
  clients import `qwen_image/api`.
- `cli/` is a client of that engine. Normal generation commands go through
  the public API; diagnostic and quantization commands can still reach the
  lower engine modules while those developer tools are being stabilized.
- `ffi/` is a thin C-ABI adapter over the same native API. C+ generates its
  `qwen_image.h`; there is no separately maintained handwritten header.

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

`width` and `height` were appended without changing ABI version 1. The library
checks `struct_size` before reading them, so a binary built against the original
request layout remains valid and continues to use `pixels` as a square width
and height. New callers set both dimensions; setting only one is invalid.

## Build and verify

`build.sh` is the distribution build. It builds the engine, CLI, and generated
C ABI, then consolidates C+'s dependency slices into libraries a C or
Objective-C application can link directly. It also compiles and runs the C
ABI smoke test. Release is the default; `BUILD_MODE=debug` selects debug. The
current source requires C+ 0.0.29 or newer because it uses `#bitcast`.

```sh
CPC=/path/to/cpc ./build.sh
```

The resulting install-shaped tree is:

```text
dist/bin/qwen-image-cplus
dist/include/qwen_image.h
dist/lib/libqwen_image.a
dist/lib/libqwen_image.dylib
```

For development, verify the three package boundaries independently:

```sh
cpc fmt --check qwen_image/src/api.cplus qwen_image/src/qwen_image.cplus cli/src/main.cplus ffi/src/ffi.cplus
(cd qwen_image && cpc check && cpc test)
(cd cli && cpc check && cpc build && cpc test)
(cd ffi && cpc check && cpc build && cpc test)
./cli/target/debug/qwen-image-cplus probe-stress
./cli/target/debug/qwen-image-cplus test-metal-primitives
./cli/target/debug/qwen-image-cplus test-metal-linear
./cli/target/debug/qwen-image-cplus benchmark-linear
./cli/target/debug/qwen-image-cplus test-metal-int8-linear
./cli/target/debug/qwen-image-cplus benchmark-int8-linear
./cli/target/debug/qwen-image-cplus test-attention-cache
./cli/target/debug/qwen-image-cplus benchmark-attention
./cli/target/debug/qwen-image-cplus benchmark-production-attention 256
./cli/target/debug/qwen-image-cplus benchmark-production-attention 1024
./cli/target/debug/qwen-image-cplus benchmark-transformer-trajectory-1024 transformer.qipack 1
./cli/target/debug/qwen-image-cplus benchmark-transformer-trajectory-1024 transformer.qipack 2
./cli/target/debug/qwen-image-cplus benchmark-transformer-trajectory-1024 transformer.qipack 13
./cli/target/debug/qwen-image-cplus benchmark-transformer-trajectory-2048 transformer.qipack
./cli/target/debug/qwen-image-cplus test-transformer-block /path/to/model/snapshot
./cli/target/debug/qwen-image-cplus quantize-block0 /path/to/model/snapshot block0.qipack
./cli/target/debug/qwen-image-cplus quantize-transformer /path/to/model/snapshot transformer.qipack
./cli/target/debug/qwen-image-cplus quantize-transformer-q4 /path/to/model/snapshot transformer-q4.qipack
./cli/target/debug/qwen-image-cplus verify-packed block0.qipack
./cli/target/debug/qwen-image-cplus verify-packed transformer.qipack
./cli/target/debug/qwen-image-cplus verify-packed-source block0.qipack /path/to/model/snapshot
./cli/target/debug/qwen-image-cplus verify-packed-source transformer.qipack /path/to/model/snapshot
./cli/target/debug/qwen-image-cplus test-transformer-block-int8 block0.qipack
./cli/target/debug/qwen-image-cplus test-transformer-mixed transformer.qipack
./cli/target/debug/qwen-image-cplus test-transformer-complete transformer.qipack
./cli/target/debug/qwen-image-cplus test-transformer-scale transformer.qipack 256
./cli/target/debug/qwen-image-cplus test-transformer-scale transformer.qipack 512
./cli/target/debug/qwen-image-cplus test-transformer-scale transformer.qipack 1024
./cli/target/debug/qwen-image-cplus test-transformer-trajectory transformer.qipack 1
./cli/target/debug/qwen-image-cplus test-transformer-trajectory transformer.qipack 2
./cli/target/debug/qwen-image-cplus test-transformer-trajectory transformer.qipack 40
./cli/target/debug/qwen-image-cplus test-transformer-trajectory-1024 transformer.qipack 40 /path/to/trajectory_1024
./cli/target/debug/qwen-image-cplus test-transformer-cache-dit transformer.qipack 0.12
./cli/target/debug/qwen-image-cplus test-transformer-cache-dit-1024 transformer.qipack 0.12 /path/to/trajectory_1024
./cli/target/debug/qwen-image-cplus test-transformer-taylorseer transformer.qipack 0.24
./cli/target/debug/qwen-image-cplus test-transformer-bottleneck transformer.qipack
./cli/target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot small
./cli/target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot 256
./cli/target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot 1024
./cli/target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot 1024-profile
./cli/target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot 2048
./cli/target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot trajectory-1024 /path/to/vae_oracle_1024
./cli/target/debug/qwen-image-cplus test-image-output reference.png
./cli/target/debug/qwen-image-cplus test-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./cli/target/debug/qwen-image-cplus test-pipeline-1024-oracle transformer.qipack /path/to/model/snapshot output.png /path/to/trajectory_1024 /path/to/vae_oracle_1024
./cli/target/debug/qwen-image-cplus test-pipeline-cache-dit-1024-oracle transformer.qipack /path/to/model/snapshot cache.png 0.12 /path/to/trajectory_1024 /path/to/vae_oracle_1024
./cli/target/debug/qwen-image-cplus test-pipeline-cache-dit-256 transformer.qipack /path/to/model/snapshot cache.png 0.12
./cli/target/debug/qwen-image-cplus test-native-inputs
./cli/target/debug/qwen-image-cplus test-native-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./cli/target/debug/qwen-image-cplus generate transformer.qipack /path/to/model/snapshot output.png "your prompt" 1344 768 1101 40
./cli/target/debug/qwen-image-cplus generate transformer.qipack /path/to/model/snapshot output.png "your prompt" 2048 2048 1301 4
./cli/target/debug/qwen-image-cplus generate-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101 25
./cli/target/debug/qwen-image-cplus generate-256-cache-dit transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.12 1101 25
./cli/target/debug/qwen-image-cplus generate-256-bottleneck transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101
./cli/target/debug/qwen-image-cplus generate-1024 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101 25 none
./cli/target/debug/qwen-image-cplus generate-1024 viggle.qipack /path/to/model/snapshot output.png "your prompt" 1301
./cli/target/debug/qwen-image-cplus generate-1024 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1301 40 cache-dit-0.16
./cli/target/debug/qwen-image-cplus pack-metadata transformer.qipack
./cli/target/debug/qwen-image-cplus pack-metadata transformer.qipack steps=40 cache=taylorseer
./cli/target/debug/qwen-image-cplus generate-1024-cache-dit transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.12 1101 25
./cli/target/debug/qwen-image-cplus generate-1024-taylorseer transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.24 1101 40
./cli/target/debug/qwen-image-cplus generate-1024-bottleneck transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101
./cli/target/debug/qwen-image-cplus benchmark-q4-eager-1024 transformer-q4.qipack /path/to/model/snapshot eager.png "your prompt" 1301 40
./cli/target/debug/qwen-image-cplus benchmark-q4-inference-1024 transformer-q4.qipack /path/to/model/snapshot destination.png "your prompt" 1301 40
./cli/target/debug/qwen-image-cplus benchmark-q4-direct-1024 transformer-q4.qipack /path/to/model/snapshot direct.png "your prompt" 1301 1
./cli/target/debug/qwen-image-cplus benchmark-process-reuse-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.24 2 1101
./cli/target/debug/qwen-image-cplus test-tokenizer /path/to/model/snapshot
./cli/target/debug/qwen-image-cplus test-multi-image-prompt /path/to/model/snapshot
./cli/target/debug/qwen-image-cplus test-text-encoder /path/to/model/snapshot
./cli/target/debug/qwen-image-cplus verify-model /path/to/model/snapshot
```

The storage-only Q4 builder implements the plan-6 H256 rotation rather than
plain scalar Q4. On the M1 Max, the pinned snapshot produced
`models/qwen-image-2.1-int4-rot-h256-v5.qipack` in 382.05 seconds: 297 tensors,
231 rotated-Q4 matrices, 3,561,009,408 bytes, and SHA-256
`11be8fc9939e9c4a16c0736045768a0a0b3edb81a536f1e8678a44d0466a0850`.
The writer verified the complete payload and independently regenerated every
rotated weight and scale from the BF16 source before the atomic rename. This
artifact was initially kept inference-disabled to separate the costly model
conversion from the decision between load-time expansion, a per-block FP16
ring, and a native packed Metal dot product. It is now accepted only by the
explicit speed-only Q4 benchmark paths described below; production generation
still requires the matching H256 transform.

The first real-weight speed-only integration compares two dequantization
placements using that exact 3.56 GB artifact and the complete native
1024px/40-step prompt-to-PNG process. `benchmark-q4-eager-1024` maps Q4,
expands all 231 matrices into a persistent FP16 buffer at startup, and then
uses the established FP16/MPS path. `benchmark-q4-inference-1024` keeps Q4
packed and expands each 32x64 destination tile into threadgroup FP16 inside
the SIMD-group GEMM. Both commands execute every transformer block, VAE
decode, and PNG write; neither uses Cache-DiT.

On the M1 Max, full sustained-load observations with the CASABLANCA prompt and
seed 1301 measured:

| Storage/execution path | Expand/startup wall | 40-step loop | Transformer wall | End to end | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original all-FP16 v4 | n/a | 395.946 s | 404.883 s | **418.590 s** | 5.74 GB |
| Eager persistent FP16 | 3.519 s | 520.060 s | 523.838 s | **539.549 s** | 15.73 GB |
| Destination tile at inference | none | 694.559 s | 696.318 s | **709.734 s** | 5.74 GB |

In those observations eager expansion completed 170.185 seconds before
destination expansion, while destination expansion saved about 9.98 GB of
peak footprint. These full-run wall times are not a controlled throughput A/B:
the runs began at different points in a long sustained-GPU sequence. A prior
destination-expansion run reported 934.777 seconds, but its raw ledger showed
one 266.599-second step between ordinary 16-18-second steps, consistent with a
laptop sleep/pause; it is retained as a discarded measurement rather than
mixed into the comparison.

The original all-FP16 v4 artifact was rerun as a sanity control with the same
prompt, seed, 1024x1024 resolution, 40 steps, cache-off path, VAE, and PNG
write. It completed 120.959 seconds sooner than the earlier eager-Q4
observation and 291.144 seconds sooner than destination-tile expansion. The
FP16 ledger itself contains one 26.829-second step between roughly 10-12-second
late steps. Because this control was not adjacent to the Q4 runs, the large
total difference must not be attributed to quantization arithmetic.

An adjacent one-step prompt-to-PNG A/B subsequently isolated the eager path:

| Adjacent one-step path | Expansion / first-touch startup | Transformer step | End to end | Peak footprint |
| --- | ---: | ---: | ---: | ---: |
| Original all-FP16 v4 | 7.353 s | 10.076 s | 31.676 s | 5.74 GB |
| Q4 expanded eagerly to FP16 | 2.966 s | 10.115 s | 27.207 s | 15.68 GB |

The post-expansion transformer step differs by only 39 ms (0.39%), directly
confirming that eager Q4 enters the same FP16 arithmetic path. Its expansion
GPU interval was 175.346 ms; the rest of its 2.881-second storage interval is
allocation and first touch. The apparent 121-second full-run penalty is
therefore a sustained-state artifact, not dequantization cost. It is consistent
with the existing all-FP16 sustained-load diagnostic, where identical cached
steps averaged 8.82 seconds in a 13-step run but 12.89 seconds when a 25-step
run immediately followed on the already-hot machine. A future long comparison
must be cooled, power-stable, adjacent, and order-reversed; the eager path's
extra 9.94 GB footprint remains a separate long-run memory-pressure variable.

A third path now tests genuinely direct packed-Q4 consumption. At startup it
rearranges the existing row-major `[N,K/2]` nibbles into coalesced
`[K/4,N]` words; this is a packed-to-packed layout conversion, not
dequantization. Each SIMD lane then loads one word containing four signed Q4
weights, forms one `half4` only at the dot instruction, accumulates in FP32,
and applies the per-output scale once at the destination. No global or
threadgroup FP16 weight tile exists. The installed M1 Metal compiler rejects
`dot(char4,char4)` and exposes no integer SIMD-group matrix type, so this
`half4` dot is the direct arithmetic available through the public language.

The isolated M=256, N=4096, K=4096 candidate measured 2.601 ms versus 2.440
ms for the custom FP16 SIMD-group MMA (0.938x). The more important real
1024 one-step prompt-to-PNG run measured:

| Direct-Q4 phase | Time |
| --- | ---: |
| Packed layout conversion | 112.721 ms GPU / 2.282 s storage interval |
| Full transformer step | **33.210 s wall** |
| Transformer phase | 35.977 s |
| One-step end to end | **49.181 s** internal / 49.61 s process wall |
| Peak footprint | 5.67 GB |

The adjacent one-step FP16 control was 31.676 s end to end and its transformer
step was 10.076 s, so this explicit direct kernel is 3.30x slower at the real
4,127-row prefill shape. The 49.181-second result must not be compared with a
40-step FP16 total. This is not
an unpack or layout-conversion loss: the one-time conversion is outside the
step and retains Q4 throughout. At thousands of rows the FP16 path reuses its
weights enough to become compute-bound and runs the matrix multiply on Apple's
SIMD-group/MPS matrix machinery; the direct candidate gives up that machinery
for lane-local vector dots. This closes only the measured half4 mapping. A
future packed-integer/SWAR candidate remains a distinct experiment, but it
cannot call an exposed M1 integer-dot or integer-matrix intrinsic because the
toolchain has none.

A subsequent controlled experiment corrected an important limitation of that
comparison. llama.cpp's single-token Q4 path does not need an accelerated
integer matrix instruction: reduced memory traffic can pay for ordinary fused
unpacking and floating-point arithmetic. The 33.210-second direct-half4 result
also changed tiling, occupancy, reuse, and arithmetic together, so it cannot
attribute the loss solely to leaving the matrix path.

The new `q4_mma_64x64` control holds the 64x64 tile, 512-thread geometry,
threadgroup storage, barriers, FP16 SIMD-group MMA, FP32 accumulation, and
direct output stores constant. Only the weight load changes from FP16 to two
packed signed nibbles expanded and scaled into the threadgroup half tile. Q4
won this controlled comparison at every measured shape: 26.592 versus 28.868
ms at `(4096,4096,4096)`, 5.051 versus 5.697 ms for cached MLP-up, and 5.205
versus 6.074 ms for cached MLP-down. This proves the expected bandwidth win is
real on the M1 Max.

It does not yet beat the complete production FP16 path. In an adjacent profiled
two-step 1024 run, Q4 step 2 took 13.255 seconds wall / 8.899 seconds GPU and
representative blocks took about 278 ms GPU. FP16 step 2 took 9.230 seconds
wall / 5.742 seconds GPU and representative blocks took about 183 ms. The
production FP16 path uses MPS for the large MLP projections, whereas packed Q4
must use the custom kernel; beating the custom FP16 control is therefore
necessary but insufficient. Exact 64-row Q4 workloads now use the fused
scaled-tile kernel, while non-divisible prompt tails retain the bounds-safe
kernel. Full measurements are in
`benchmarks/m1-max-q4-fused-mma.json`.

The scale placement in this experiment is specific to QIPACK v5. It stores one
FP16 scale for an entire output row, making a final row scale algebraically
valid; the accepted fused kernel instead applies that scale while filling the
weight tile so it can share the FP16 kernel's direct output path. A conventional
block-scaled Q4 format must apply each block's scale before its partial sum is
combined with other blocks.
Immediately after this FP16 control, macOS reported `AC Power` and an attached
charger but also a 58% battery that was still discharging. That conflicting
power state may contribute to the late-step drift and must accompany the
measurement; it does not make this cache-off run comparable to the earlier
roughly 165-second Cache-DiT runs.

This is intentionally a dequantization-placement speed test, not a Q4 quality
or inference-correctness claim. The stored matrices are H256-rotated. To keep
the comparison isolated, both modes execute the same rotated coefficients as
the matrix and omit inverse-H expansion in eager mode and activation H256 in
destination-expansion mode. The generated PNGs therefore only prove complete,
finite execution. A production Q4 path must include the matching transform;
these numbers answer only whether persistent expansion or destination-tile
expansion is faster under the full real-weight workload.

The production generation commands accept `25` or `40` as an optional step
argument. Omitting it preserves the canonical 40-step behavior, unless pack
metadata sets `steps`.

`generate-1024 PACK MODEL_DIR OUTPUT PROMPT [SEED] [STEPS] [CACHE]` reads
its defaults from pack metadata.

- **`STEPS`** accepts 3, 4, 8, 25, or 40. The default is the pack's `steps`,
  otherwise 40.
- **`CACHE`** accepts `none`, `taylorseer`, or `cache-dit-0.16`. The default
  is the pack's `cache`, otherwise `none`.
- **`shift_terminal=none`** in the pack selects the unstretched schedule
  that few-step distills need.
- **Refusals:** caching a `kind=distilled` pack, caching with fewer than 25
  steps, invalid metadata, a `kind=distilled` pack in any fixed-step command
  (`generate-256*`, `generate-1024-bottleneck`, the pipeline oracles), and a
  pack `steps` value the command cannot run. Metadata accepts only
  `steps=3|4|8|25|40`. `pack-metadata` refuses to write through an
  inconsistent header, writes in crash-safe stages, and can replace
  unreadable metadata. `verify-packed` now checks metadata as the loader
  does.
- **Environment switches:** generation refuses `QI_PROFILE_SKIP`. It also
  refuses `QI_EXPERIMENT_NO_SHIFT_TERMINAL` when the pack states
  `shift_terminal=0.02`, and `QI_CACHE_DIT_WARMUP` values outside 1-4. Every
  run prints a `trajectory switches:` and a `VAE switches:` line.

The local base pack carries `kind=base steps=40 shift_terminal=0.02
cache=taylorseer`. The Viggle pack carries `kind=distilled steps=4
shift_terminal=none`. Only its file name mentions Viggle. A
25-step run constructs a fresh 25-step FlowMatch schedule; it does not truncate
the first 25 points of the 40-step schedule. It reproduces a community ComfyUI
template choice rather than Qwen's official recommendation. Both 256 and 1024
paths have end-to-end measurements. The 1024/40 path now also has external
official transformer and VAE oracles; their numerical and visual findings are
recorded below.

Cache-DiT commands accept the calibrated threshold candidates `0.12`, `0.14`,
`0.16`, and `0.24`. At 1024/40, 0.12 remains the conservative quality setting,
0.16 is the measured speed-biased setting, and 0.24 is rejected. Intermediate
values outside that measured set are intentionally not accepted by the
production CLI. TaylorSeer retains its separately calibrated 0.12/0.24 input
surface.

The two `bottleneck` commands are research diagnostics, not production
recommendations. They use the fixed 4+13+8 stage experiment described below;
the FLUX-tuned policy failed Qwen's 256px visual gate and was not promoted to a
1024px run.

Inspect a shard, optionally filtering tensor names:

```sh
./cli/target/debug/qwen-image-cplus inspect /path/to/shard.safetensors proj_out
```

## Implemented foundation

- owned macOS read-only mappings and bounded Safetensors parsing;
- sharded index resolution and exact pinned-model inventory verification;
- C+ to Metal compilation, dispatch, GPU timing, ownership, no-copy mapping,
  and a 1000-dispatch stability test;
- fixed-rank tensor descriptors, checked row-major shapes, and a reusable
  aligned activation arena;
- CPU truth functions for checkpoint float formats, activations, norms,
  Qwen timestep embeddings, complex RoPE, block-causal masking, and the exact
  FlowMatch Euler schedule.
- Metal correctness kernels for BF16/FP16 decode, SiLU, tanh GELU, SwiGLU,
  LayerNorm, RMSNorm, zero-centered RMSNorm, timestep embedding, complex RoPE,
  and block-causal masks. The validation command reports error metrics and
  rejects non-finite output; measured tolerances are recorded in
  `manifests/metal-tolerances.json`.
- A correctness-first 16x16 tiled linear kernel for BF16 and FP16 `[out,in]`
  weights with FP32 accumulation. It is exact on partial-tile validation cases
  and has recorded full-4096-token timings for all three transformer matrix
  shapes in `benchmarks/m1-max-linear.json`.
- An exact model-width block-0 correctness path using the checkpoint's nine
  block tensors: affine-free LayerNorm and shared modulation, Q/K/V, learned
  per-head Q/K RMSNorm, three-axis complex RoPE, block-causal attention,
  attention output projection, tanh-gated residuals, and SwiGLU MLP. Fifteen
  named boundaries are checked against the committed FP32 oracle, and the M1
  Max result is recorded in `benchmarks/m1-max-block0.json`.
- Measured symmetric and affine INT4/INT8 candidates at K-group sizes 32, 64,
  and 128. No INT4 role met the provisional complete-block error budget; the
  current block-0 policy is affine INT8 group-64. The full measurements are in
  `benchmarks/m1-max-quantization.json` and the deliberately provisional policy
  is in `manifests/block0-quantization-policy.json`.
- A C+ block-0 packed writer/reader with fixed little-endian metadata,
  256-byte tensor alignment, a page-aligned data section, source identity,
  per-tensor and payload checksums, exact source round-trip validation, and
  atomic temp-file installation.
- An affine INT8/group-64 Metal linear kernel that consumes packed U8 weights,
  FP16 scales, and U8 zero points without materializing a dense weight matrix.
  It uses FP16 tiles with FP32 accumulation, matches its CPU oracle exactly,
  and is measured on all three model matrix shapes in
  `benchmarks/m1-max-int8-linear.json`.
- A complete packed block-0 Metal validation path. All seven matrix roles run
  through the INT8 kernel and finish at 0.837% normalized RMS error. Repeated
  M1 Max measurements and every intermediate error boundary are recorded in
  `benchmarks/m1-max-block0-int8.json`.
- Full-transformer calibration of the same Q8 policy on three real prompts,
  exact early/middle/late 40-step scheduler timesteps, 256/512/1024-pixel
  token layouts, all 224 block matrices, and sampled outputs from blocks 0,
  7, 15, 23, and 31. Uniform Q8 is rejected: four of six final-noise cases
  exceed the 1% gate and the worst reaches 1.921%. Results are recorded in
  `benchmarks/m1-max-transformer-quantization.json`; the decision is in
  `manifests/transformer-quantization-policy.json`.
- A mixed-precision search by block range and matrix role. The smallest tested
  policy under the 1% gate keeps blocks 0-23 fully BF16, quantizes Q/K/V and
  attention output in blocks 24-31, and additionally quantizes MLP projection
  and output in blocks 28-31. Its repeated worst-case error is 0.985%, and its
  block-matrix storage is 12.166 GiB instead of 13 GiB. The repeat is recorded
  in `benchmarks/m1-max-mixed-quantization.json`.
- A full-transformer QIPACK1 writer/reader for the complete 297-tensor
  inventory. Current policy v4 stores all 224 block matrices as the FP16
  operands consumed by MPS, orders Q, K, and V adjacently per block, and has
  no Q8 records; vectors and the nine global tensors remain BF16. The reader
  remains compatible with the mixed v3, MLP-FP16 v2, and original v1
  policies. The writer refuses a source other than the exact
  7,115,124,736-parameter, two-shard inventory; installs atomically only after
  structure, payload, per-tensor, and exact source round-trip checks; and
  produced a verified 14,230,327,296-byte artifact from the pinned snapshot.
  The loader validates layout and scope compatibility before use.
- Native execution of all 32 blocks directly from one read-only, page-aligned
  no-copy QIPACK1 Metal buffer. Kernel selection comes from each tensor record,
  not a second hard-coded policy. Current v4 artifacts discover zero Q8
  matrices; legacy v3 artifacts still discover zero in blocks 0-23, four in
  blocks 24-27, and six in blocks 28-31. A compact
  four-token FP32 oracle checks blocks 0, 23, 24, 27, 28, and 31 around both
  precision transitions. The final native mixed output measured 0.1300%
  nRMSE, with 581.5 ms summed GPU kernel time on the M1 Max; repeated results
  are recorded in `benchmarks/m1-max-transformer-mixed-native.json`.
- Native execution of the complete transformer boundary around those blocks:
  64-to-4096 image projection; zero-centered RMSNorm and two-layer tanh-GELU
  text projection; exact 256-channel sinusoidal timestep projection and
  two-layer timestep embedding; shared SiLU modulation; four-fold VLM image-
  slot expansion and condition/target latent substitution; final adaptive
  LayerNorm; and 4096-to-64 output projection. Linear dispatch is now row-
  variable for both BF16 and packed Q8 weights rather than fixed to the old
  four-token fixture.
- The complete fixture is deliberately small but structurally realistic: four
  VLM rows contain one condition-image slot, the target contributes two slots,
  expansion produces 15 joint rows (three text, four condition-image, eight
  target-image), and one interleaved text key is invalid. This catches treating
  padding as a prefix, confusing condition and target modulation, or omitting
  image-slot expansion. It checks the global boundaries and six block-depth
  checkpoints against a committed independent NumPy/FP32 oracle.
- On the M1 Max, the 15-row complete path passed at 0.3347% target-output
  nRMSE (1.2% gate), with 0.1300% at block 31 and 1,245.9 ms summed GPU kernel
  time on the first recorded run (1,242.1 ms on repeat). Results are in
  `benchmarks/m1-max-transformer-complete-native.json`.
  The full joint output is retained because the transformer API returns it;
  the target-only metric is decisive because the pinned pipeline slices the
  trailing target rows before its scheduler step.
- The packed Metal mapping uses a process-lifetime Objective-C block
  descriptor for its no-copy deallocator. This is required because Metal
  retains that callback until the buffer is released; a descriptor allocated
  on the helper's stack survives execution but crashes during buffer teardown.
  The dedicated mapping path now releases cleanly, including the short-lived
  block-0 validation command.
- Scalable block-causal attention now assigns one 128-thread group to each
  query/head pair. The group reduces each Q.K dot product once and all lanes
  update one output channel with online softmax. Compute remains necessarily
  quadratic in sequence length, but temporary storage is constant per group:
  no `[heads, queries, keys]` score or probability tensor is allocated. This
  also removes the old correctness kernel's repeated dot product per output
  channel, reducing its work from O(S^2 D^2) to O(S^2 D) per head.
- The attention path has both block-causal prefill and target-only cached
  execution. Each of the 32 layer caches stores the prefix K after learned
  RMSNorm and RoPE, plus the raw prefix V, matching the pinned transformer.
  Target queries use their global row offset for token metadata, combine the
  cached prefix with the current bidirectional target block, and continue to
  apply the text key-valid mask. Adjacent condition-image blocks remain
  distinct; equality is based on image ID, not merely on image/text type.
- Two independent gates cover these semantics. A committed seven-row FP32
  oracle includes two adjacent condition blocks, an invalid interleaved text
  key, and a two-row target block; prefill and cached decode both agree within
  4.48e-8 maximum absolute error. The complete 32-block mixed transformer also
  runs its eight target rows through a 7,340,032-byte two-arena prefix cache;
  every captured block and final target output are bit-identical to the
  uncached target slice. Measurements are in
  `benchmarks/m1-max-attention-cache.json`.
- A 4,096-token M1 Max gate completes with finite output in 2,546.05 ms. Its
  Q/K/V/output activations occupy 256 MiB and the online implementation avoids
  the 2 GiB FP32 score tensor that `[32,4096,4096]` would require. This is an
  attention memory/feasibility gate, not a production throughput claim; each
  dispatch still uses its own command buffer and CPU wait.
- Production-size full noise-prediction gates now run the image/text/timestep
  projections, all 32 mixed-precision blocks, final adaptive norm, and output
  projection at 256, 512, and 1024 pixels. Their pinned fixtures use real
  prompt-encoder embeddings, exact 40-step scheduler timesteps, and seeded
  latent noise from the earlier calibration cases. They store all inputs and
  final target noise but only 32 rows at blocks 0, 1, 7, 15, and 31, keeping
  three model-scale oracles to 15 MiB rather than committing full block states.
- All three scales complete with finite output. Target-output nRMSE versus the
  pinned official BF16 execution is 0.9114% at 256, 1.2452% at 512, and 0.8794%
  at 1024, under the declared 1.5% production-runtime gate. This gate is kept
  distinct from the 1% quantization-search gate: the native runtime retains
  FP32 activations while the official oracle rounds model activations to BF16,
  and sampled divergence is already 2.2933% at unquantized block 15 in the
  512-pixel case. The tolerance therefore covers the complete arithmetic path,
  not only the 40 Q8 matrices.
- The 1024-pixel case executes 4,096 target tokens plus a 31-token prefix in
  144,597 ms of summed GPU kernel time. Explicit non-weight buffers total
  1,665,752,320 bytes (1.551 GiB), including the extracted 32-layer prefix
  cache; the read-only 12.42 GiB QIPACK1 mapping remains separate. It completes
  on the 32 GB M1 Max without a quadratic attention allocation. Full results
  and limitations are recorded in
  `benchmarks/m1-max-transformer-scale-native.json`.
- The pinned FlowMatch Euler integration is now exercised as a complete
  40-step 256px latent trajectory. The native scheduler reproduces dynamic
  exponential shifting, terminal stretching to sigma 0.02, the 1000x model
  timestep convention, and Diffusers' FP32 Euler update followed by a cast
  back to the BF16 model dtype after every step. Unit tests cover 2-, 10-, and
  40-step schedules at several token lengths, while the model-scale command
  checks the complete official 40-step sigma and timestep fixtures before any
  transformer work begins.
- Step 1 runs the complete 278-row joint sequence and extracts every layer's
  post-RoPE text K and raw V. Steps 2-40 reuse the same 23,068,672-byte cache
  and recompute only 256 target rows. This is valid specifically because the
  pinned model has `causal_condition=true`: prefix tokens use the t=0
  modulation row and are independent of the sampled denoising timestep.
- The `1` and `2` trajectory commands are prefixes of the canonical 40-step
  schedule, not independently constructed short schedules. That makes each
  checkpoint comparable to one reference run and catches the cache transition
  at step 2. Against the official BF16 trajectory, latent nRMSE is 0.0896% at
  step 1, 0.1291% at step 2, and 1.0048% at step 40. Separate measured gates
  of 0.10%, 0.15%, and 1.10% preserve the observed accumulation curve rather
  than hiding early regressions behind one loose final limit.
- The full native run dispatches 1,600 selected Q8 matrices, stays finite, and
  originally totaled 147,234 ms of summed GPU kernel time. The production
  transformer linears now use a 32x32 output tile computed by each 16x16
  threadgroup: every thread owns four accumulators, so input and weight tiles
  are reused across twice as many rows and columns and four times as much
  arithmetic occurs per barrier. The K tile and inner-loop order remain 16,
  preserving the established accumulation order. The same 40-step gate now
  totals 97,428.8 ms, a 33.8% reduction, with identical step-1, step-2, and
  step-40 nRMSE. The four-row block fixture is slower because it underfills a
  32-row tile; that deliberate test-only tradeoff avoids runtime shape
  heuristics on the product path, where cached steps have exactly 256 target
  rows. Before/after measurements are in
  `benchmarks/m1-max-transformer-linear-32x32.json`. The trajectory is
  intentionally 256px so a
  40-step regression remains practical; the independent scale gate already
  covers one complete noise prediction at 512px and 1024px. Fixture provenance,
  timings, acceptance limits, and limitations are recorded in
  `benchmarks/m1-max-transformer-trajectory-native.json`.
- Transformer blocks now encode their dependent kernels into one compute
  encoder and one command buffer, with explicit buffer barriers only at true
  dependency boundaries. Independent Q/K/V, Q/K normalization, and MLP
  gate/projection dispatches do not receive barriers between one another. The
  unfused dispatch functions remain available for operator and boundary
  regression tests. In a direct two-step A/B, synchronous non-GPU overhead
  fell from 319.0 ms to 46.6 ms (85.4%); GPU-frequency variation obscures that
  saving in raw wall time, so both GPU and wall clocks are reported per step.
  The complete fixture trajectory measured 98,241.6 ms GPU and 99,577 ms wall.
  Results and caveats are recorded in
  `benchmarks/m1-max-command-submission-batching.json`.
- The real Qwen-Image-2.1 VAE still-image decoder now runs natively from its
  1.258 GiB FP32 Safetensors file through one read-only no-copy Metal mapping.
  It applies the exact 64-channel latent mean/std handoff, post-quant and input
  convolutions, the 1,152-channel residual/one-head-attention middle block,
  five residual up blocks, final RMSNorm/SiLU, output convolution, and clamp.
  Activations use HWC order so the implicit-GEMM convolution can read OIHW
  checkpoint weights without a persistent transpose or repack.
- The official class is named a causal 3D VAE, but its image specialization is
  materially different from a generic Conv3D decoder. Its checkpoint
  convolution weights are four-dimensional Conv2d tensors. On the first
  one-frame cached chunk, each nominal temporal-upsample main path records the
  `Rep` sentinel and skips `time_conv`; the `DupUp3D` residual shortcut still
  performs temporal/channel/spatial reshaping and then crops to the first
  frame. The native shortcut reconstructs that exact channel mapping directly,
  without materializing discarded temporal frames.
- Each residual up block must preserve its original input for that shortcut.
  Reusing it during the three residual layers initially caused the first
  observable divergence at up block 0 despite every preceding boundary being
  correct. A dedicated preserved-shortcut arena fixes the ownership hazard;
  the committed small oracle now checks all three residuals, the main
  upsample, shortcut, and combined result for every block. All 35 comparisons
  stay below 2.51e-6 nRMSE and the final output is 5.51e-7.
- The complete step-40 trajectory latent decodes from `[16,16,64]` to
  `[256,256,4]` at 8.09e-7 nRMSE against the pinned official FP32 MPS decoder,
  under a 3e-6 gate. The measured M1 Max run used 456,523,776 bytes of explicit
  scratch storage plus the no-copy weights and took 2,297.53 ms of summed GPU
  kernel time. Results, scope, and limitations are recorded in
  `benchmarks/m1-max-vae-decoder-native.json`.
- A dispatch-level VAE profile showed that convolution owned about 2.23 s of
  the 2.25 s GPU total, so command submission was not the first-order problem.
  The scalar 16x16 convolution is now an implicit 64x64 FP32 cooperative-matrix
  tile: sixteen SIMD groups share a gathered 64x32 input tile and transposed
  32x64 weight tile, while preserving FP32 operands and accumulation. It does
  not materialize the largest 680 MiB im2col matrix, adds no global scratch,
  and the production path no longer computes a no-SiLU final norm used only by
  the small fixture report. Three 256px repeats measured 795.151, 728.080, and
  728.438 ms GPU (728.438 ms median), a 3.15x speedup over the recorded
  2,297.53 ms baseline. Direct decoder wall time was 1.09, 1.01, and 0.99 s.
  All 35 small boundaries still pass; production output nRMSE is 7.19e-7 under
  the unchanged 3e-6 limit. The changed FP32 summation order moves only five
  of 262,144 final teapot RGBA bytes, each by one. In the full pipeline the VAE
  phase fell from 2,971 to 1,396 ms, saving 1.575 s; that run's end-to-end time
  was 35.528 s because text paging independently regressed from 12.674 to
  15.493 s. Exact measurements and the comparison caveat are in
  `benchmarks/m1-max-vae-decoder-native.json`.
- The production 256px VAE now encodes its complete dependency chain into one
  Metal command buffer and waits once; the small diagnostic path remains
  synchronous because it reads 35 intermediate boundaries on the CPU. On its
  own, removing 111 waits was deliberately a small win: median GPU time moved
  from 740.008 to 736.956 ms and warm process time from about 0.98 to 0.97 s.
  The larger accepted change decomposes nearest-neighbor 2x plus 3x3
  convolution into four output parities. A GPU packing kernel collapses the
  repeated samples into four 2x2 phase kernels, reducing the inner dimension
  from `9*C` to `4*C`; one 84,934,656-byte buffer is reused for all four
  upsample layers. Three runs measured 598.210, 600.717, and 597.453 ms GPU
  (598.210 ms median) and 0.85, 0.83, and 0.82 s process wall. That is a 19.2%
  GPU reduction from the synchronous 740.008 ms baseline, at 541,458,432
  bytes total decoder scratch. Production output nRMSE improved slightly from
  7.19e-7 to 6.65e-7; all small boundaries passed, and the native prompt's
  latent, decoded-FP32, and RGBA gates were unchanged. Its VAE phase measured
  581.037 ms GPU / 1,289 ms wall versus 1,398 ms wall in the preceding QKV
  run. Whole-process time is excluded because text paging varied independently.
  Pipelines and scratch already persist for the full decode; retaining them
  across images requires a future multi-request API rather than more work in
  the current single-image phase. Full measurements are in
  `benchmarks/m1-max-vae-decoder-native.json`.
- The first complete vertical slice now executes the committed prompt and
  initial-noise fixture through all 40 transformer/scheduler steps, hands the
  resulting `[16,16,64]` normalized latent to the VAE in caller-owned memory,
  converts the decoded tensor to RGBA, and writes a PNG in one process. The
  transformer function returns before VAE setup, deliberately releasing the
  12.42 GiB packed mapping and its workspaces before the 1.258 GiB VAE mapping
  is opened. The handoff is in memory but not claimed as a GPU zero-copy
  optimization: the shared Metal latent is copied into a 64 KiB host array and
  then into the VAE's shared input buffer.
- Image postprocessing exactly reproduces Diffusers' `(x * 0.5 + 0.5)` clamp,
  HWC conversion, multiplication by 255, and NumPy ties-to-even rounding. The
  independent official-output gate matches all 262,144 RGBA bytes. PNG output
  is implemented directly in C+ with RGBA scanlines, stored DEFLATE blocks,
  Adler-32, CRC-32, and atomic installation. An earlier AppKit encoder was
  rejected because its alpha/color handling changed 23,602 official RGB bytes
  by one; independently decoding the final C+ PNG preserves the input RGBA
  bytes exactly.
- On the final M1 Max run, the fixture-driven slice used 147,340 ms of summed
  transformer GPU time and 2,373.82 ms in the VAE. Native error versus the
  official pipeline was 1.0048% at the final normalized latent, 1.1579% in the
  decoded FP32 tensor, and 0.6752% after RGBA quantization. The pixel result has
  0.504 mean absolute byte error and a maximum error of 25 under measured gates
  of 1.3% FP32 and 0.8% RGBA. Results and limitations are recorded in
  `benchmarks/m1-max-pipeline-256-native.json`.
- Native text conditioning now begins at the downloaded tokenizer assets rather
  than a Python-produced token stream. The implementation performs NFC
  normalization, the pinned Unicode regex split, GPT-2 byte-to-Unicode mapping,
  all 151,387 ranked BPE merges, added-special-token recognition, the raw T2I
  chat template, and exact 14-token system-prefix removal. Six oracle cases
  cover empty, Unicode/multiline, whitespace-sensitive, special-token, and
  decomposed-NFC inputs; every output token ID matches Transformers 5.17.0.
  NFC and Unicode-category regex matching use macOS Foundation so the runtime
  does not carry an incomplete home-grown Unicode database; byte mapping and
  BPE execution remain C+ code and the fixture gate checks their combined
  behavior.
- Text execution now reports wall time separately for Safetensors mapping,
  `WILLNEED`, device/queue creation, no-copy storage buffers, Metal pipelines,
  scratch allocation, and each layer. On the measured 36-token run, mapping
  took 6 ms but whole-shard `WILLNEED` took 7,622 ms; execution then took
  another 7,865 ms despite only 761.357 ms of summed GPU work. The remaining
  stalls were concentrated before layer 0 and at layers 6, 19, and 32, which
  are the decoder-weight shard transitions. This falsifies the assumption that
  the current full-file advice cheaply overlaps setup: no-copy buffer creation
  was 1 ms and Metal compile/pipeline creation was 5 ms. Selective layer-range
  advice was therefore tested next. Advising the 13.892 GB of used embedding
  rows and decoder tensors reduced advice to 5,569 ms, but doing so in
  layer/tensor order destroyed sequential file access: execution rose to
  13,018 ms and total text time regressed by 3,048 ms to 18,646 ms. That
  implementation was rejected. Any retry must coalesce and sort ranges by file
  offset or use a decoder-only pack; the accepted runtime retains whole-shard
  advice.
- `QI_PROFILE_CACHED_BLOCK=1` samples blocks 0, 1, and 31 of trajectory step 2
  without changing the default log. The steady samples measured 24.86 ms GPU
  / 25 ms wall and 26.79 ms GPU / 27 ms wall; tensor metadata took 0–1 ms and
  MPS object setup rounded to 0 ms. Block 0 paid a one-time MPS cold start
  (24.92 ms GPU / 70 ms wall). Prebuilding descriptors can only recover part
  of the measured whole-loop wall-minus-GPU gap (about 0.5 s under Cache-DiT),
  not the multi-second saving initially projected. Raw timing and interpretation
  are recorded in `benchmarks/m1-max-phase-breakdown.json`.
- The complete text-only Qwen3-VL language path also runs natively from the
  original four no-copy BF16 Safetensors mappings: 36 decoder layers at width
  4096, 32 query heads, 8 KV heads, 128-wide RoPE, and 12,288-wide SwiGLU. It
  omits the unused vision tower, final RMSNorm, and LM head and returns the last
  decoder-layer state required by Qwen-Image. Explicit ties-to-even BF16
  activation boundaries were necessary: retaining FP32 between operations was
  rejected at 8.42% final nRMSE, while the accepted path measures 3.61% against
  the pinned Torch 2.9 pipeline fixture in 774.971 ms of summed GPU time.
  A separately regenerated Torch 2.8 MPS reference differs from that Torch 2.9
  fixture by 3.28% itself; consequently the local 4% gate is not treated as an
  end-to-end prompt-equivalence claim. Boundaries and limitations are recorded
  in `benchmarks/m1-max-text-conditioning-native.json`.
- The prompt path is now connected to the complete 256px image pipeline.
  `generate-256` accepts an arbitrary prompt and unsigned 64-bit seed, releases
  the 17.5 GB text mappings before opening the 12.42 GiB packed denoiser, and
  releases the denoiser before opening the VAE. Joint sequence buffers, masks,
  cache sizing, and three-axis RoPE are derived from the actual prompt length.
  Native noise reproduces PyTorch 2.9's CPU MT19937 plus scalar Box-Muller path
  exactly before BF16 rounding: the seed-1101 fixture has zero nRMSE, while
  generated RoPE measures `8.31e-8` nRMSE. A separate seed-42 blue-teapot
  smoke run completed with 18 text rows, proving that the handoff and KV-cache
  sizing are not accidentally fixed to the 22-row canonical prompt.
- The canonical end-to-end native-prompt gate measures 0.0896%, 0.1286%, and
  1.7561% latent nRMSE after denoising steps 1, 2, and 40; decoded FP32 error is
  1.6909%, and final RGBA error is 0.9573% with a 0.557 mean absolute byte
  error. The integrated text path has separate measured 2% latent/decoded
  budgets because of the already documented text-backend variation; the
  stricter 1.1%/1.3% fixture-fed gates remain unchanged. With the 32x32
  transformer tile, this gate's denoiser time is 98,843.8 ms instead of
  148,823 ms, while every recorded error metric remains identical. Full
  measurements and rationale are in
  `benchmarks/m1-max-native-prompt-pipeline-256.json`.
- Runtime transformer loading maps QIPACK1 once and always validates its exact
  model identity, tensor inventory, ordering, dimensions, quantization schemes,
  policy-specific file length, and byte ranges. It no longer rescans the
  complete 13.33 GB v3 or 14.23 GB v4 payload on every generation. Set
  `QI_VERIFY_PACKED_CHECKSUMS=1` when loading an artifact whose provenance is
  uncertain; that adds all per-tensor checksum checks. The explicit
  `verify-packed` audit remains exhaustive, checking both the whole payload
  and every tensor.
  Pack creation already performs an exact source round-trip and installs the
  completed artifact atomically, so normal inference trusts an artifact that
  was verified when it was built while retaining deliberate audit paths.
- The canonical prompt-to-PNG run now takes 131,262 ms, down from 175,625 ms
  (25.3%), with every numerical metric unchanged. Transformer pre-loop time
  fell from 59,060 ms to 13,881 ms (76.5%): 9,809 ms of parallel packed
  validation, 7 ms of Metal setup, no measurable storage/weight-view cost, and
  3,949 ms of buffers plus text-prefix projection were directly instrumented.
  The transformer phase fell from 158,794 ms to 113,161 ms while its denoising
  loop remained within normal run-to-run variation at 99,280 ms. Results,
  integrity boundaries, and caveats are recorded in
  `benchmarks/m1-max-packed-runtime-validation.json`.
- Production-shape profiling, rather than square microbenchmarks, now drives
  transformer kernel work. At the real cached shape (`M=256`), the former
  scalar BF16 kernels sustained roughly 1.5-2.0 TFLOP/s and affine Q8 only
  1.4-1.5 TFLOP/s; Q8 dequantization therefore saved storage but did not save
  compute. The accepted Apple-family-7 SIMD-group kernels convert BF16 or Q8
  operands to FP16 tiles while retaining FP32 accumulation. A bounds-safe
  64x32 path handles the 278-row prefix step, while an exact-shape 64x64
  direct-store path handles the 256-row cached steps. The latter measures
  4.0-4.3 TFLOP/s on the three dominant matrix shapes. A 128x32 experiment
  was rejected because register and threadgroup pressure reduced throughput to
  2.0-2.9 TFLOP/s despite greater weight reuse.
- Attention uses one 32-lane SIMD group per two queries of one head. Each lane
  owns four of the 128 head channels for both queries, `simd_sum` replaces the
  shared-memory dot-product tree, and a K/V load feeds both independent online
  softmax states. The two-query path is bit-identical to the prior one-query
  kernel in the production benchmark and reduces cached attention from 3.312
  to 3.003 ms (9.3%); projected cache-off attention falls from 4,233 to 3,846
  ms. A four-query version was rejected at 3.687 ms because register pressure
  and lower occupancy outweighed reuse. The full cache-off trajectory fell
  from 33,092.1 to 32,712.5 ms GPU and from 34,239 to 33,733 ms wall; step-40
  nRMSE is 1.01756% under the unchanged 1.1% gate. On Cache-DiT 0.24 the same
  change saves 220.2 ms GPU / 207 ms wall because only 13 steps run all 32
  blocks; its 27 cached decisions are unchanged. Measurements are in
  `benchmarks/m1-max-production-kernel-profile.json` and
  `benchmarks/m1-max-cache-dit-native.json`.
- Exact production-shape MPS probes established a hybrid rather than a blanket
  replacement. MPS on this M1 Max rejects BF16 matrix inputs, but accepts FP16
  inputs with FP32 outputs. Including the required on-GPU conversion, MPS
  completes the cached 4096->12288 and 12288->4096 MLP shapes in 4.24-4.56 ms,
  versus 6.03-6.51 ms for the accepted custom kernels. The custom kernel stays
  faster on an isolated BF16 4096x4096 attention projection (2.00 ms versus
  2.83 ms). That result still governs non-fused and Q8 projections; QIPACK v3
  later makes one adjacent FP16 4096-to-12288 QKV operation profitable.
  Isolated MPS output nRMSE is at most 0.02095%.
- Cached MLPs convert FP32 activations to FP16, bind v2/v3/v4 FP16 weights
  directly from the read-only QIPACK mapping, and invoke
  `MPSMatrixMultiplication` with FP32 output. Legacy Q8 packed weights are
  converted into reusable FP16 scratch storage.
  Conversion, MPS GEMM, SwiGLU, and residual work remain ordered in one command
  buffer per transformer block; there is no CPU inference fallback or host
  synchronization between operations. Legacy packs allocate one reusable
  96 MiB weight-conversion buffer, while v4 reduces that placeholder to two
  bytes because every block matrix is already FP16. A 6 MiB activation buffer
  feeds every MPS GEMM at 256.
  Both the 278-row first step and 256-row steady-state steps use MPS for all
  three MLP matrices; the first step also extracts prefix K/V in-buffer.
- Direct FP16 storage removes 88 repeated BF16 conversion dispatches per step
  without a second 9.7 GiB runtime weight copy or any increase in artifact
  size. The verified fixture-fed 40-step loop now takes 33,092 ms of summed
  GPU time and 34,239 ms wall time, down from the dynamic-conversion MPS result
  of 37,778 ms and 39,437 ms (12.4% GPU, 13.2% wall), and from the custom-Metal
  baseline of 46,283 ms and 47,989 ms (28.5% GPU, 28.7% wall). Step 2 is
  800.25 ms GPU and step 40 is 805.29 ms. Exact latent nRMSE remains unchanged
  at 0.09039%, 0.13008%, and 1.00775% for steps 1, 2, and 40. The loop is now
  near the low-30-second range but still above the 25-30-second target.
  Measurements are
  in `benchmarks/m1-max-transformer-trajectory-native.json`; custom-kernel
  baselines remain in `benchmarks/m1-max-production-kernel-profile.json`.
- The next production-shape profile found that the remaining normalization
  kernels were serial in the wrong dimension: one Metal thread reduced an
  entire 4,096-value LayerNorm row, and one thread reduced all 128 channels of
  each Q/K head. One 32-lane SIMD group now owns each row or head. Lanes visit
  every 32nd value, `simd_sum` combines their partial sums, and the same lanes
  write the normalized/modulated or RoPE-rotated result. The scalar kernels
  remain in the benchmark as numerical references. At 256 rows, LayerNorm
  fell from 2.241 to 0.149 ms (15.1x, 0.000161% nRMSE) and Q/K norm+RoPE from
  0.178 to 0.034 ms (5.19x, zero measured nRMSE). The isolated projection
  overestimated the benefit because production batches these dispatches with
  GEMM and MPS work; the authoritative cache-off trajectory nevertheless fell
  from 32,712.5 to **28,979.7 ms GPU** and from 33,733 to **29,908 ms loop
  wall**. Step-40 nRMSE is 1.03288%, below the unchanged 1.1% limit. This is
  the first exact cache-off loop measurement below the 30-second target.
  Cache-DiT 0.24 retains the same 27 cached decisions and falls from 11,942 to
  **10,639.5 ms GPU**, and from 12,530 to **11,181 ms loop wall**. The full
  native prompt regression also passes at 1.7982% final-latent, 1.7765%
  decoded-FP32, and 1.0031% RGBA nRMSE. Measurements and the difference
  between isolated and integrated projections are recorded in
  `benchmarks/m1-max-transformer-small-kernels.json`.
- Scheduler conditioning is now computed for the whole requested trajectory
  before the denoising loop. The two conditioning rows for each timestep are
  stacked into one table, so time projection, the two timestep linears and
  SiLUs, modulation, and final-scale projection require seven command buffers
  total instead of seven per step (7 versus 280 for 40 steps). The four
  linears deliberately retain the scalar FP32-operand kernel: automatically
  selecting the >=64-row FP16 SIMD path would change model arithmetic. Batching
  still cuts repeated reads of the roughly 192 MiB of conditioning weights
  from 40 row tiles to three. Per-step slices are copied into the existing
  small runtime buffers, keeping every downstream binding unchanged. All
  checkpoint nRMSE values and the 27 Cache-DiT decisions are unchanged.
  Cache-off summed GPU time fell from 28,979.7 to **28,794.4 ms**; its single
  wall comparison moved from 29,908 to 29,983 ms, which is run-to-run noise
  rather than a supported wall-speed claim. Cache-DiT 0.24 fell from 10,639.5
  to **10,499.6 ms GPU** and from 11,181 to **11,057 ms wall**; its cheap
  cached steps fell from about 27 to 24 ms GPU. The full native prompt and
  decoded-image gate remains unchanged. Exact timings and the reason for not
  using the faster FP16 kernel are in
  `benchmarks/m1-max-conditioning-precompute.json`.
- A blanket increase from 32 to 256 threads per elementwise threadgroup was
  also measured and rejected. It reduced isolated residual and SwiGLU time,
  but made the serial row-reduction and small conditioning dispatches slower;
  the two-step integrated GPU total regressed from 2,103.8 to 2,139.4 ms.
  Kernel-specific launch sizes remain a possible later refinement, but 256 is
  not a safe global default.
- The MPS MLP path now also handles the prompt-dependent first transformer
  step. Its reusable half-activation scratch is sized for all joint
  text+image rows (278 in the canonical case instead of 256), and the same
  command buffer copies the post-RoPE text-prefix K plus raw V into the
  per-layer cache before full joint attention. The later 256-row steps still
  consume that cache through the existing path. This removes the special
  custom-Metal MLP path without changing checkpoint values: step 1 fell from
  1,415.1 to **893.7 ms GPU**, while its nRMSE stayed at 0.0906469%.
  Cache-off fell from 28,794.4 to **28,440.4 ms GPU** and from 29,983 to
  **29,282 ms loop wall**. Cache-DiT 0.24 fell from 10,499.6 to **10,062.5 ms
  GPU** and from 11,057 to **10,543 ms wall**, with all 27 cache decisions
  unchanged. The native prompt, VAE, and RGBA gates also remain unchanged.
  Measurements are in `benchmarks/m1-max-first-step-mps.json`.
- Transformer submission now spans a full denoising step instead of stopping
  and waiting after each of its 32 blocks. The block executor can encode into
  a caller-owned command buffer; cache-off commits once after block 31.
  Cache-DiT must still complete block 0 so the CPU can inspect its residual,
  but each non-cached step then commits blocks 1-31 as one tail. The explicit
  block-profiling mode retains the old per-block wrapper. No kernels or
  arithmetic changed: the canonical step-1, step-2, and step-40 nRMSE values
  are identical. Cache-off loop wall time fell from 29,282 to **28,824 ms**
  while summed GPU time stayed effectively flat at 28,413.6 ms, confirming
  that this was a host synchronization optimization rather than a compute
  optimization. Cache-DiT 0.24 retained all 27 decisions and fell from 10,543
  to **10,452 ms wall**; its summed GPU time varied from 10,062.5 to 10,075.7
  ms. The native prompt pipeline also passed unchanged at 28,704 ms transformer
  loop wall, 1.7982% latent nRMSE, 1.77651% decoded-FP32 nRMSE, and 1.00307%
  RGBA nRMSE. Full measurements and rationale are in
  `benchmarks/m1-max-transformer-step-submission.json`.
- Dense Q, K, and V now execute as one MPS 4096-to-12288 multiplication. QIPACK
  v3 places each block's Q/K/V records adjacently and stores the 72 dense
  records in FP16; blocks 24-31 retain their calibrated Q8 records and custom
  kernels. The fused `[row,Q|K|V]` activation reuses the existing MLP-sized
  scratch, while Q/K normalization, attention V reads, and first-step prefix-V
  extraction consume explicit row strides and offsets. A custom fused Metal
  prototype was rejected: 64x32 and 64x64 tiles regressed step 2 because
  register pressure and occupancy outweighed activation reuse. The MPS path is
  both faster and exact at every printed checkpoint. Cache-off fell from
  28,413.6 to **25,626.5 ms GPU** and from 28,824 to **26,263 ms wall**.
  Cache-DiT 0.24 retained all 27 decisions and fell from 10,075.7 to **9,067.4
  ms GPU** and from 10,452 to **9,582 ms wall**. The native prompt pipeline
  passed unchanged at 25,696.8 ms GPU / 26,051 ms transformer-loop wall,
  1.7982% latent nRMSE, 1.77651% decoded-FP32 nRMSE, and 1.00307% RGBA nRMSE.
  The v3 artifact remains 13,334,843,392 bytes, passed exact source round-trip
  verification, and old v2 artifacts still pass. Measurements and rejected
  alternatives are in `benchmarks/m1-max-qkv-mps.json`.
- Cache-DiT is available as an explicit, off-by-default approximation. It
  follows the upstream
  [DBCache block flow](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_blocks/pattern_base.py)
  and [relative-L1 decision](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_contexts/cache_manager.py)
  rather than the earlier design sketch:
  block 0 always runs; its residual `h1 - h0` is compared with the residual
  from the last full step using relative L1; the first four steps are full;
  and at most three cached steps may run consecutively. A cached step adds the
  stored blocks-1-through-31 residual to the new block-0 output. The prefix KV
  cache remains layer-indexed and continues to serve block 0 on every step.
- At threshold 0.12, 25 of 40 steps are cached and the canonical loop takes
  13,762 ms GPU / 14,816 ms wall, a 2.41x / 2.31x speedup over cache-off.
  Threshold 0.24 caches 27 steps and takes 12,162 ms / 12,737 ms, a 2.72x /
  2.69x speedup. The algorithm intentionally changes output: versus the
  official fixture, threshold 0.12 measures 9.93% latent, 9.47% decoded-FP32,
  and 5.22% RGBA nRMSE; threshold 0.24 measures 10.08%, 9.49%, and 5.23%.
  Both canonical images remain coherent, and an arbitrary blue-teapot prompt
  completed with 27 cached steps, but these measurements are not an
  equivalence gate. Keep cache-off for reproducibility and choose Cache-DiT
  only when this quality tradeoff is acceptable. Detailed results and the
  reason for retaining both thresholds are in
  `benchmarks/m1-max-cache-dit-native.json`.
- The 12.74-second figure is the 40-step transformer/scheduler loop, not an
  end-to-end request. Startup work was subsequently reduced by forwarding the
  pipeline's existing token IDs into the text encoder, asynchronously paging
  its four mapped weight shards while Metal is initialized, and moving QIPACK
  payload hashing out of the default inference path. In a controlled A/B, the
  last committed implementation took **42.094 seconds** internally while the
  optimized implementation took **35.693 seconds** (15.2% faster); its repeat
  took **34.504 seconds internally / 34.52 seconds process wall**. The repeat
  comprised 12,674 ms for text, 18,837 ms for the complete transformer phase
  including a 12,503 ms denoising loop, 2,971 ms for VAE setup and decode, and
  21 ms for PNG output. All A/B and repeat PNGs are byte-identical. Readahead
  raised maximum resident set size from 13.35 GB to 17.60 GB in the controlled
  runs, a deliberate speed-for-memory tradeoff on the 32 GB test machine. A
  resident service would still be required to approach the loop time for
  repeated requests, but residency alone is constrained by the combined model
  working set as measured below.
- Plan-5 F3 supersedes that whole-shard readahead strategy. The encoder now
  keeps the Safetensors mappings metadata-only, reads the eleven tensors for
  one 385,892,864-byte decoder layer with positional `pread`, and binds their
  unchanged BF16 bytes from a reusable shared Metal buffer. Two 416 MiB slots
  alternate: while Metal executes layer N, one worker fills the other slot
  with layer N+1. `F_NOCACHE` marks this one-pass stream so text weights do not
  unnecessarily displace later denoiser and VAE pages. The vocabulary matrix
  is no longer wired in full either; only the prompt's requested 8 KiB rows
  are copied into a compact embedding buffer. This is deliberately streaming,
  not a persistent inference cache: every request still reads the weights it
  uses.
- **Aligned, parallel layer reads (2026-09-23).** Every layer span began at
  a page-unaligned shard offset (the Safetensors data bases are 49,800,
  18,528, 18,560 and 5,208 bytes into the shards), so no `F_NOCACHE` read
  took the direct path. Each span was also read by a single thread, at about
  3 GB/s (116-133 ms per layer). Spans are now widened to whole 16 KiB pages
  at 16 KiB-aligned slot offsets (385,908,736 bytes per layer, still inside
  the 416 MiB slot). Each span is read by eight page-aligned jobs, like the
  transformer pack copy. Layers now stream in 56-60 ms (about 6.6 GB/s). The
  1024 text phase fell from 3,978 to 2,275 ms in `generate-1024`, and
  `test-text-encoder` fell from about 3.1 s to 2,088 ms execution. The
  bytes and output are unchanged: nRMSE is still 0.0361069, and the Viggle
  quoted-text PNG SHA-256 is still `be8f75ea...2811`.
- The latest pre-F3 text timing was 16,242 ms, including 7,208 ms inside the
  four whole-shard `MADV_WILLNEED` calls. A single-slot streaming prototype
  took 4,226 ms, and the accepted two-slot path took **3,145 ms** with the
  exact same 3.61069% text-output nRMSE (5.16x faster than that recent
  whole-shard run and 1.34x faster than synchronous streaming). A separate
  `/usr/bin/time -l` sample took 2,903 ms internally / 3.06 seconds process
  wall and reported 831,946,752 bytes maximum RSS and 932,988,152 bytes peak
  footprint. In adjacent native 256 pipelines, text fell from 3,329 to
  **2,144 ms**. End to end moved only from 32,222 to **31,838 ms** because
  transformer buffer/prefix setup was 804 ms slower in the second run, so the
  text-phase delta is the supported speed claim and the 384 ms total delta is
  reported only as an observation. The final PNG remained byte-identical at
  SHA-256 `d9967772c1abc8869f23f6595bf6216b18ff63433d21ececc55dc01ce628bc29`.
  Exact phase, memory, oracle, and unit-test results are in
  `benchmarks/m1-max-text-layer-streaming.json`.
- A two-request same-process baseline disproved the assumption that process and
  tokenizer reuse alone would materially improve warm latency. The tokenizer
  initialized once in 138 ms, then request 1 took 35,675 ms and request 2 took
  35,688 ms. The output remained byte-identical and maximum RSS stayed at
  17.56 GB. Model resources deliberately remained phase-scoped: keeping
  the 17.5 GB text weights, 12.42 GiB denoiser, and VAE live together would
  exceed the safe working set. Cycling through them also displaces the useful
  pages, so a persistent process cannot remove the dominant weight-I/O cost
  without first reducing model storage. The reproducible command and exact
  phase timings are in `benchmarks/m1-max-process-reuse-native.json`.
- Affine Q8/group-64 text-weight calibration found no acceptable broad policy.
  Uniform Q8 added 5.72% text-output nRMSE under the custom-kernel FP32
  emulation; quantizing complete early layers reached 3.33% after only one
  layer while saving just 1.32% of linear storage. The best large isolated
  role was all 36 MLP `up_proj` matrices: it saved 1.608 GiB (12.43% of linear
  storage) but added 3.32-3.66% text nRMSE across three prompts. When its
  canonical embedding was passed into the native transformer, step-1 latent
  nRMSE rose from 0.08963% to 0.10334%, exceeding the 0.10% gate. Building a
  packed text runtime for that marginal policy is rejected. Exact calibration,
  limitations, and the downstream decision are in
  `benchmarks/m1-max-text-q8-calibration.json`.
- As an external full-precision baseline, the patched stable-diffusion.cpp
  build at commit `6dcb5bb` completed one cache-off 1024x1024, 40-step run in
  **1006.36 seconds process wall** (16:46.36) on AC power. Its internal
  `generate_image` timer was 1005.29 seconds: 63.24 seconds for text
  conditioning, 922.90 seconds for sampling, and 19.00 seconds for VAE decode.
  Progress timings averaged 23.07 seconds over all 40 iterations; iteration 1
  was 66.12 seconds because it included lazy transformer loading, Metal
  staging, and compilation, while iterations 2-40 averaged 21.97 seconds and
  ranged from 17.57 to 25.93 seconds under sustained load. The run used the
  original BF16/F32 checkpoint, ggml diffusion flash attention,
  `--offload-to-cpu`, CFG 1.0, Euler, no cache mode, and a warm filesystem
  cache. It reported 16.93 GB maximum RSS, 43.77 GB macOS peak memory
  footprint, and zero swaps. Its automatically selected Flux shift was 1.150
  rather than the pinned Diffusers/native oracle's 0.693548; that changes the
  trajectory but not the tensor shapes or amount of 40-step transformer work,
  so the result is retained strictly as a speed baseline. Exact command,
  binary identity, timings, and host conditions are in
  `benchmarks/m1-max-stable-diffusion-cpp-1024-40.json`.
- As an external Apple-Silicon baseline, the locally downloaded
  `mlx-community/Qwen-Image-2.1-MLX-4bit` snapshot `4db4e8c` took
  **36.48 seconds end to end on repeat** for the same blue-teapot prompt at
  256x256, seed 42, 40 steps, and guidance 1.0. The first valid run was 39.23
  seconds, so the measured fresh-process range is 36.48-39.23 seconds. The
  repeat produced a byte-identical PNG. The timed scope was a fresh process
  through model loading, tokenization/text encoding, denoising, VAE decode,
  and completed PNG output; its progress-timed generation loop was 32.0
  seconds. MLX reported 7.97 GB peak allocated memory, while macOS reported a
  10.72 GB peak memory footprint. The output was visually coherent. This was
  run with MLX 0.32.2 and the Qwen-Image-2.1 mflux reference branch at commit
  `dc5af52025a323e9b6dd44b702f6fc941498f978`; because the checkpoint is a
  generic MLX export and its model card supplies no inference command, a
  temporary loader adapter preserved its packed affine tensors, accepted its
  alternate Qwen3-VL key prefix, and avoided re-transposing already-MLX VAE
  kernels. All 761 transformer tensors, all 904 text-only encoder tensors,
  and all 226 still-image VAE tensors were mapped; only the unused vision,
  LM-head, and video-only `time_conv` tensors were skipped. The filesystem
  cache had been warmed by compatibility smoke tests, so this is cold model
  initialization in a new process, not a post-reboot disk-cold measurement.
  Exact measurements and the output checksum are in
  `benchmarks/m1-max-mlx-community-qwen-image-2.1-4bit.json`.
- With the startup changes above, native C+ is effectively tied with and
  slightly ahead of that MLX baseline on this case: 34.52 seconds process wall
  versus MLX's 36.48-second repeat, a 1.96-second (5.4%) difference. This is a
  single prompt and warm-filesystem-cache comparison, not a broad throughput
  claim; the implementations also use different quantization and caching
  paths.
- QIPACK stores persistent packed weights and integrity/model metadata; it is
  not a serialized inference cache. Each generation must rebuild the
  prompt-dependent per-layer prefix K/V cache (about 18-23 MiB in the measured
  256px cases) and roughly 16 MiB of Cache-DiT first-block and middle-residual
  state. A resident process could retain validated mappings, compiled
  pipelines, and reusable workspaces, but it must refresh both caches for every
  new prompt and denoising trajectory. The measured same-process baseline
  showed no warm-request gain while mappings remain phase-scoped, and retaining
  all three weight sets concurrently is outside the 32 GB memory budget. New
  processes still benefit from macOS's filesystem page cache and always repeat
  structural QIPACK validation and runtime setup, but payload checksum scans
  are explicit verification work rather than a cost paid by every generation.
- The Plan 4 I/O/attention review was tested at 256 rather than accepted from
  projections. Its straightforward cooperative-matrix flash prototype reused
  K/V across eight queries and stayed within 6.28e-7 nRMSE, but regressed
  cached attention from 3.236 to 10.097 ms (3.12x slower); the two QK passes,
  threadgroup staging, barriers, and occupancy cost more than the reuse saves.
  The prototype was removed. A different item was accepted: Cache-DiT now
  reduces its relative-L1 decision on Metal and reads back two scalars instead
  of scanning 1,048,576 FP32 values on the CPU. All 27 ratios agree with the
  former FP64 host calculation within 1.77e-8 and all decisions are unchanged.
  Two clean 40-step loops measured 9,104 and 9,479 ms wall versus an immediate
  9,570 ms pre-change run; GPU-frequency variation prevents attributing the
  whole spread, but cached-step wall readings consistently fell from roughly
  26-36 to 22-27 ms. This also removes a resolution-squared CPU cost before
  returning to 1024. Exact measurements and deferred findings are in
  `benchmarks/m1-max-plan4-review-256.json`.
- Native prompt generation now has real 1024x1024 product paths, not a 256px
  decode followed by upscaling. `generate-1024` runs the reproducible cache-off
  trajectory when the pack has no `cache` key or the final argument is
  `none`. (The local base pack now carries `cache=taylorseer`, so pass
  `none` to reproduce these cache-off records with it.)
  `generate-1024-cache-dit` selects the explicit approximation.
  Both derive a centered 64x64 latent grid, 4,096 target rows, prompt-sized
  metadata/RoPE and prefix caches, decode to `[1024,1024,4]`, and write a
  correctly dimensioned RGBA PNG. A one-step diagnostic completed the full
  handoff in 74.313 s and produced a valid 1024px PNG.
- The first complete 40-step 1024 run used Cache-DiT 0.24 on a 31-row poster
  prompt. It cached 27 steps and produced a coherent, substantially legible
  1024px result in **687.920 s end to end**: 13.316 s text, 662.176 s for the
  transformer phase (651.573 s GPU / 654.686 s wall in the denoising loop),
  12.095 s VAE, and 0.308 s PNG output. This is a functional and perceptual
  smoke result, not an equivalence claim. It predates the official 1024
  trajectory and image oracle described below and is superseded for quality
  decisions.
- The official 1024x1024, 40-step oracle gate now replays the pinned
  Diffusers transformer with the same 31-row `CASABLANCA` poster prompt,
  seed 1301, scheduler, and per-layer prefix K/V cache. Large tensors remain
  outside Git. Native input construction is exact: seeded-noise nRMSE is zero
  and dynamic-RoPE nRMSE is 2.46e-7. Native all-FP16-v4 latents pass the early
  checkpoints at **0.09597%** (step 1) and **0.14312%** (step 2), but diverge
  to **13.211%** at step 40 against the official BF16 trajectory (13.1609%
  after the 2026-09-23 timestep-rounding fix). This is a
  newly exposed long-horizon precision limitation; the old 256-only gate did
  not justify calling the 1024 path numerically equivalent.

  The pixel-space result is substantially better than the latent number alone
  suggests. The native 1024 VAE independently matches Diffusers at
  **6.47e-7 output nRMSE** when fed the same official latent. Decoding the
  native final latent gives 22.9042% FP32-output nRMSE, 6.42 mean U8 absolute
  error, 20.55 dB PSNR, and 80.06% of pixels within eight U8 levels on every
  channel. Visual inspection finds only small high-contrast letter-edge and
  texture shifts; `CASABLANCA / MEET ME AT / SUNSET` remains exact and the
  composition matches. Cache-off is therefore the reproducible native policy,
  but it is visually close rather than numerically equivalent to BF16.

  A confirmed-AC fixture-to-PNG run measured 387.707 s for the transformer
  loop, 394.789 s for transformer setup plus loop, 10.277 s for VAE setup and
  decode, and **405.434 s total**. It intentionally bypasses text encoding by
  consuming the oracle embedding. A preceding full run took 479.823 s as
  cached-step time drifted from about 10.5 to 14.6 seconds, demonstrating why
  power and thermal state must accompany long-run timings.
- The same oracle qualifies ordinary Cache-DiT at 1024/40. Threshold 0.12
  caches **25/40** steps, measures 17.4586% final-latent and 30.2985%
  decoded-output nRMSE, and preserves all poster text and overall structure.
  Its observed loop was 137.934 s and fixture-to-PNG total was 157.122 s;
  cached steps cost about 0.28 s while full steps cost about 8.58 s in that
  run. The intermediate sweep found that 0.14 caches 26 steps at 21.5439%
  latent and 35.7685% decoded nRMSE, while **0.16 caches 27 steps** at 22.0154%
  latent and 36.5093% decoded nRMSE. Both preserve all requested text; 0.16 is
  retained as the speed-biased option because it removes two full passes.
  Threshold 0.24 also caches 27 steps but rises to 26.8949% latent and 42.4777%
  decoded-output nRMSE and visibly corrupts `CASABLANCA`, so it remains
  rejected. Ordinary **0.12 remains the conservative quality setting** and
  **0.16 is the speed-biased setting**; both are explicit and off by default.
  Sequential thermal differences make absolute fixture wall times unsuitable
  for comparing thresholds. Full provenance and measurements are in
  `benchmarks/m1-max-1024-40-oracle.json`.
- Fresh-process native-prompt 1024x1024, 40-step inference is now measured from
  tokenization through PNG completion with the same `CASABLANCA` prompt and
  seed 1301. Cache-off took **451.12 seconds process wall**: 3.493 seconds for
  text, 434.489 seconds for the transformer phase (425.513-second loop),
  12.797 seconds for VAE, and 0.305 seconds for PNG output. Cache-DiT 0.12 ran
  second on the already-hot machine, made the expected 25 cached decisions,
  and took **204.60 seconds process wall**: 3.102 seconds for text, 190.092
  seconds for the transformer phase (182.412-second loop), 11.061 seconds for
  VAE, and 0.309 seconds for PNG output. That observed sequential-run result is
  2.20x faster than cache-off. Both ran on AC from a warm filesystem cache,
  used fresh processes, completed without swapping, and excluded the one-time
  offline QIPACK build. Cache-off is 2.23x faster than the measured
  stable-diffusion.cpp cache-off process baseline (451.12 versus 1006.36
  seconds); opt-in Cache-DiT is 4.92x faster, although that comparison also
  includes its intentional trajectory approximation. Exact timings, memory,
  commands, ordering, and output checksums are in
  `benchmarks/m1-max-native-prompt-pipeline-1024.json`.
- The speed-biased Cache-DiT 0.16 setting was subsequently measured through
  the same fresh-process native-prompt path. It made 27 cached decisions and
  completed in **167.15 seconds process wall**: 3.235 seconds for text,
  153.082 seconds for the transformer phase (145.842-second loop), 10.493
  seconds for VAE, and 0.307 seconds for PNG. This is an observed 37.45-second
  reduction, or 1.224x speedup, against the 204.60-second 0.12 run and 6.02x
  faster than the measured stable-diffusion.cpp process baseline. The runs
  occurred at different points in a sustained sequence, so not all 37.45
  seconds can be attributed to the two extra cached steps; their removal is
  the repeatable algorithmic difference. The full-prompt output retained all
  requested text exactly. A later powered confirmation, with the battery
  charging from 25% to 26%, completed in **145.18 seconds process wall**:
  3.364 seconds text, 131.237 seconds transformer including a 122.601-second
  loop, 10.230 seconds VAE, and 0.304 seconds PNG output. It made the same 27
  cached decisions and produced the exact same 4,195,716-byte PNG checksum as
  the 167.15-second run. This confirms **about 145 seconds as the best observed
  1024/40 FP16 + Cache-DiT 0.16 end-to-end result**, while the 21.97-second
  spread between identical outputs remains an operating-condition effect, not
  an algorithmic speedup.
- A transformer-only sustained-load diagnostic closes the question of whether
  later 1024 steps accumulate software work. The benchmark command now accepts
  4, 8, 13, 25, and 40 steps in addition to its original canonical-prefix
  1/2-step modes. On AC at 100%, a 13-step run held cached steps between 8.677
  and 9.072 seconds and its last six averaged 0.218 seconds faster than its
  first six. An immediately following, already-hot 25-step run started at a
  13.337-second average for cached steps 2-7, then improved to 12.621 seconds
  for steps 20-25. Cached steps use the same 4,096-row shape, fixed buffers,
  and operation count throughout. The large absolute spread is therefore
  sustained GPU frequency/thermal state, not a denoising-loop leak or
  step-dependent algorithmic cost. Short A/B tests remain the right kernel
  acceptance tool; end-to-end numbers must record power state and cannot be
  extrapolated from one hot run. Exact series are in
  `benchmarks/m1-max-sustained-transformer-1024.json`.
- The 1024 VAE initially reserved every activation arena for the largest
  `[1024,1024,288]` boundary. Shape-specific maxima reduce its explicit scratch
  from 7,389,315,072 to **5,577,375,744 bytes**, saving exactly 1.6875 GiB.
  The block-4 normalization temporary must retain the larger 288-channel
  capacity; a more aggressive first attempt was rejected by the 256 oracle.
  The corrected layout passes the 256 oracle at 6.65e-7 nRMSE and a direct
  finite 1024 smoke in 9,390.94 ms GPU. Full timing, the output checksum,
  prior single-prediction correctness evidence, and limitations are recorded in
  `benchmarks/m1-max-native-prompt-pipeline-1024.json`.
- Plan-5 F5 now has the missing dispatch-level 1024 profile. The normal
  `1024` case retains one command buffer; `1024-profile` intentionally commits
  and waits per dispatch so each GPU interval is observable. In the final
  9,373.18 ms diagnostic run, FP32 convolution consumed **8,957.36 ms
  (95.56%)**, the 4,096-token middle attention consumed 272.31 ms (2.91%),
  and every norm, add, clamp, and other operation combined consumed 143.51 ms
  (1.53%). The five up blocks accounted for 8,687.17 ms (92.68%). This closes
  the plan's attention question: even deleting middle attention entirely
  cannot materially change decode latency, and production already batches
  submission. A post-profile production run remained at 8,997.58 ms GPU, and
  the exact 256 oracle still passed at 6.64769e-7 nRMSE under its 3e-6 limit.
  Future VAE compute work must improve the existing FP32 cooperative-matrix
  convolution or demonstrate an equally accurate FP32 framework path; FP16
  operands remain outside the numerical gate. Raw per-residual and per-block
  measurements are in `benchmarks/m1-max-vae-1024-profile.json`.
- That FP32 framework path now exists and is the default. Every decoder
  convolution runs through MPSGraph FP32 NHWC convolution in
  `qwen_image/src/mps_graph_conv.cplus`, including 1x1 layers and the nearest-2x
  upsample followed by 3x3. Weights are bound in place from the Safetensors
  mapping as flat `MPSNDArray`s and reshaped to OIHW inside the graph.
  - **Why the old kernel was slow.** From layer FLOPs, the custom kernel
    reached only about 1.7 TFLOP/s, 16% of peak. Its gather does per-element
    integer division, walks OIHW taps so HWC reads don't coalesce, and
    performs five threadgroup loads for every four MMAs.
  - **Why MPSGraph is faster, not less precise.** An FP32 probe ran MPS
    convolution at 10.4-11.5 TFLOP/s at every VAE shape, above the FP32 FMA
    peak. Its error against FP64 was lower than CPU FP32's, which indicates
    a fast algorithm computed in FP32, not reduced precision.
  - **Accuracy.** The small fixture's 35 boundaries all pass, and every
    oracle improved: 256 from 6.65e-7 to **2.43e-7**, and the official 1024
    oracle from 6.47e-7 to **2.25e-7**.
  - **Speed and memory.** A 1024 decode fell from 9.28 s to **2.40 s**
    process wall, at a cost of 2.15 GB more peak footprint (7.76 GB).
  - **End to end.** The CASABLANCA 1024/40 Cache-DiT 0.16 run fell from
    145.18 s to **135.55 s**: 3.256 s text, 129.125 s transformer, 2.843 s
    VAE, and 0.301 s PNG. All text stayed exact. A fresh AC-powered run after
    the current fixed-cost work measured **126.69 s process wall** / 126.266 s
    internal: 2.323 s text, 121.576 s transformer including a 118.736 s loop,
    2.223 s VAE, and 0.120 s PNG. It again made 27 cached decisions and kept
    all text exact. The PNG is not byte-identical to the old baseline because
    the updated arithmetic paths produce small letter-edge and texture changes;
    this current-worktree number is therefore a visual product check rather
    than a bit-exact regression result.

  `QI_DISABLE_VAE_MPSGRAPH_CONV=1` restores the custom kernels.
  MPSGraph commit-and-continue fragments GPU timestamps, so wall time is
  authoritative. Details are in `benchmarks/m1-max-vae-mpsgraph-conv.json`.
- Transformer startup no longer maps the pack. Setup at 1024 took about
  6.6 s, almost all of it the first GPU dispatch that touches the weights:
  Metal faults and wires the whole 14.2 GB no-copy mapping at about 2 GB/s,
  even with a warm page cache. A release build does not help.
  - **What changed.** Eight threads now `pread` the pack through an
    `F_NOCACHE` descriptor into an ordinary shared buffer.
  - **Result.** The copy takes about 1.9 s (about 7.5 GB/s), and setup falls
    to **2.3-2.6 s**. Output PNGs are byte-identical.
  - **Under swap pressure** (about 5 GB of swap in use), the copy took 5.0 s.
  - **Footprint.** The weights now count toward the process footprint
    (about 16 GB peak) instead of as wired file pages.

  A pipeline A/B was confounded by sustained heat: full steps drifted from
  8.7 to 11.7 s across back-to-back runs. The loop comparison is therefore
  not supported, while the startup saving is. `QI_DISABLE_PACK_PREAD=1`
  restores the mapping. Details are in `benchmarks/m1-max-pack-pread.json`.
- Metal FlashAttention was probed as a replacement for MPSGraph SDPA at the
  cached 1024 shape and rejected.
  - **Compiler problem.** Its generated kernels use private
    `air.simdgroup_async_copy` intrinsics that the macOS 26 Metal compiler
    rejects.
  - **Result with a synchronous replacement.** The kernel was accurate (1.2e-6
    against FP64 with FP32 intermediates). The best of 13 block and register
    configurations took **58.3 ms** per 32-head block, or 4.75 TFLOP/s.
  - **Comparison.** MPSGraph takes about 42 ms integrated and 49.5 ms
    isolated, before MFA's layout conversions are even counted.

  Its published 83% M1 Max utilization depends on the removed intrinsics.
  Details are in `benchmarks/m1-max-mfa-attention-probe.json`.
- Plan-5 F8 first-order TaylorSeer is implemented as a separate, explicit
  Cache-DiT mode. Full steps update the blocks-1-through-31 residual and its
  per-step finite difference; cached steps evaluate `Y + elapsed * dY` in one
  Metal kernel. Ordinary Cache-DiT is unchanged: it retains its original
  residual kernels and three-consecutive-step cap. Taylor allocates one extra
  residual-sized buffer only when selected (64 MiB at 1024) and raises its own
  cap to four.
- The 256 oracle explains why this is retained experimentally but not promoted.
  At the same 27 cached steps, Taylor reduced step-40 latent nRMSE from
  **0.100293 to 0.0681592**, proving the predictor is better than repeating the
  last residual. Its cap-four setting cached **29/40** steps and reduced the
  measured trajectory wall time from 7,577 to 6,474 ms, while nRMSE rose only
  to 0.106648. Cap five was rejected at 0.145265. The qualified 1024 A/B then
  measured 128.592 s end to end for Taylor versus 175.339 s for ordinary
  Cache-DiT, but sequential-run thermal variation makes the full timing gap
  non-causal; the durable gain is two avoided full passes. More importantly,
  the controlled poster image changed the correctly rendered `CASABLANCA` to
  `CASABLANCCA`. Taylor therefore stays off by default and does not replace
  ordinary Cache-DiT. The implementation, all calibration points, checksums,
  timing caveat, and visual decision are recorded in
  `benchmarks/m1-max-taylorseer.json`.
- A 1024 follow-up is also parked. It explained the error and tried to
  place Taylor's full steps better. Threshold 0.24 made no decisions: the
  four-step cap alone produced one full step in five. Three opt-in switches
  each left the text wrong:
  - `QI_TAYLOR_SIGMA_SPACING` extrapolates over sigma distance instead of
    step count. Result: 125.6 s, still `CASABLANCCA`.
  - Adding `QI_TAYLOR_FULL_FINAL_STEP` never predicts the last evaluation.
    Result: still `CASABLANCCA`.
  - `QI_TAYLOR_EARLY_CAP3` caps the first half at three predicted steps,
    giving 12 full steps. Result: 134.4 s; the extra C shrank to a stray
    stroke.

  A current fixed-cost-worktree rerun of the original Taylor policy measured
  **112.64 s process wall** with 11 full and 29 predicted passes, versus
  126.69 s for the adjacent Cache-DiT 0.16 run. It still rendered
  `CASABLANCCA`, while Cache-DiT rendered `CASABLANCA` exactly, so the faster
  result does not change the rejection for this prompt.

  The letter count is fixed during early layout. Ordinary Cache-DiT 0.16
  already renders exact text with 13 full steps, so at most about one full
  step (about 7%) remains to win.

  **Superseded judgment:** CASABLANCA was not quoted in that prompt. On the
  quoted-text prompt, default TaylorSeer (0.24, 11 full steps) renders both
  strings exactly in 115.1 s. It is the fastest base-model mode and is
  available as the base pack's metadata default.
- `QI_PROFILE_SKIP=attention|qkv|attnout|mlp|gate|down` is a profiling-only
  switch. It omits one kind of work from every block, so the change in
  step-2 wall time of `benchmark-transformer-trajectory-1024 ... 2` gives
  that part's cost inside the whole step. Output is meaningless while it is
  set, so every generation command refuses it; only the transformer-only
  benchmarks accept it. Every trajectory prints a `trajectory switches:`
  line with the state of each `QI_*` switch that can change its output, and
  the VAE prints a `VAE switches:` line. On the 8.8-8.9 s cached 1024 step:

  | Part | Time per step | Share of step | % of peak |
  | --- | ---: | ---: | ---: |
  | MLP GEMMs | 4.95 s | 55% | 76% |
  | Fused QKV | 1.55 s | 17% | 80% |
  | MPSGraph attention | 1.35 s | 15% | 62% |
  | Attention output | 0.47 s | 5% | 88% |
  | Everything else | 0.60 s | 7% | — |

  Within the MLP, the 12288-deep down projection is the slowest, at 70% of
  peak. Splitting every wide or deep MPS GEMM into 4096-wide slices was
  rejected: 8.72 s versus 8.81 s, within run-to-run noise. MPS runs the
  slices no more efficiently than the whole GEMM. Attention therefore holds
  the clearest kernel headroom, about 5 s per 13-full-step image. Details
  are in `benchmarks/m1-max-1024-step-ablation-profile.json`.
- Plan-5 F9 implements the Bottleneck Sampling mechanics as an explicit
  experiment: a 256→128→256 or 1024→512→1024 resolution path, Lanczos-3
  spatial latent resizing, fresh-noise reinjection, and independent 4+13+8
  rationally shifted schedules with strengths 1.0/0.8/0.6 and shifts 9/6/9.
  This model does **not** use FLUX-style 16-channel 2×2 latent packing: the
  pinned official Qwen-Image 2.1 pipeline exposes a native 64-channel VAE
  latent and only spatially flattens it, so the implementation resizes those
  64 channels directly.
- The transfer failed the cheap 256 visual gate. The literal paper schedule
  produced 0.699725 diagnostic nRMSE against the ordinary 40-step final latent
  and a severely blurred, structurally wrong image. Preserving Qwen's sigma
  0.02 final model evaluation in each stage did not rescue it: nRMSE became
  0.844049 and the output remained a flat, pale sleeping animal without the
  requested red fox, forest, mossy-stone detail, or lighting. That corrected
  run took **36.780 s end to end**, including 32.260 s for the three-stage
  transformer phase; its present research implementation also pays transformer
  setup three times. Because both bounded variants failed visibly, no costly
  1024 run was made. This closes the sub-native 1024→512→1024 proposal, but the
  later discovery that Qwen's native target is 2048 means it does not settle a
  2048→1024→2048 schedule: its middle stage would still be an officially
  supported resolution rather than the extreme 128px bottleneck used by this
  cheap gate. That native-resolution variant should be reconsidered only after
  a 2048 one-step and memory feasibility gate. The machinery remains explicit
  and off by default, while all measurements, checksums, and the revisit
  condition are recorded in `benchmarks/m1-max-bottleneck-sampling.json`.
- Short 1024 transformer runs now make optimization practical without decoding
  an image: `benchmark-transformer-trajectory-1024 ... 1|2` uses the committed
  31-row text fixture, seed 1301, and the first one or two steps of the normal
  40-step schedule. The pre-change one-step measurements were 35.584-39.606 s
  GPU; the two-step run measured 35.584 s for cache extraction and 37.935 s for
  the cached-prefix step. Prompt K/V caching therefore does not remove the
  dominant 4,096-token work.
- The first 1024-specific attention change is accepted. One SIMD group now owns
  four queries at 1024, reusing each K/V vector twice as broadly as the existing
  two-query kernel while retaining separate online-softmax state. It is
  bit-exact to the scalar attention reference at the measured production shape.
  Reversing benchmark order still measured 755/748 ms for four-query
  prefill/cached attention versus 872/885 ms for two-query, a 15-18% speedup.
  Integrated one-step transformer repeats measured 32.216 and 32.553 s GPU.
  The policy is resolution-specific: at 256 the same kernel regressed
  prefill/cached attention from 3.050/3.407 to 3.713/3.769 ms, so 256 and 512
  retain two-query attention. A separate eight-query shared-threadgroup
  candidate was also exact but 5% slower at 1024 because its barriers outweighed
  reduced reads, and was removed. Measurements and caveats are in
  `benchmarks/m1-max-transformer-trajectory-1024-short.json`.
- The scalar attention line is now superseded at 1024 by a fixed-shape Metal
  Flash Attention kernel. It specializes the MIT-licensed llama.cpp
  8-query/64-key/four-SIMD-group structure to 32 heads of width 128: Q and
  padded contiguous K/V matrix operands are FP16, while scores, online-softmax
  state, and output accumulation stay FP32. The isolated production shape fell
  from 745.523 to 75.223 ms for prefill and from 776.494 to 74.583 ms for the
  cached query, **9.91x and 10.41x faster**. Relative to the scalar FP32
  reference, nRMSE is 0.000269969 and 0.000434663. Those are conversion error,
  not a change to the Qwen block-causal mask; the corrected benchmark avoids
  power-of-two fixture denominators that had accidentally hidden FP16 loss.
- The production 1024 path prepares padded FP16 K/V in reusable scratch,
  caches each block's text prefix directly in FP16, and restores it ahead of
  target K/V on later steps. It does not allocate another full K/V pair. The
  integrated one-step transformer fell from 32.216-32.553 s GPU to **11.426 s
  GPU / 11.512 s step wall**. A separate two-step run measured 11.169 s for
  joint cache extraction and 11.701 s for the cached-prefix step, 22.887 s GPU
  / 23.023 s loop wall total. Prefix K/V storage fell from about 31 MiB to
  **15.5 MiB**. Process wall still includes roughly 6.5-7.0 s of startup buffer
  work, which is intentionally reported separately from transformer execution.
- MPSGraph's fused scaled-dot-product attention now supersedes the custom flash
  kernel at 1024. The graph accepts the runtime's row-major FP32 Q and padded
  FP16 K/V, performs its FP16 cast and row/head transposes itself, preserves the
  exact block-causal prefill mask, and returns row-major FP32 output. Those
  conversions are included in the isolated result: prefill fell from 74.232 to
  **53.644 ms** and cached attention from 72.979 to **49.512 ms**, with
  0.000306/0.000490 nRMSE against scalar FP32. Two reversed adjacent full-step
  pairs reduced loop wall from 11.545-11.654 s to **10.573-10.820 s**, a
  6.3-9.3% saving. A two-step run fell from 22.769 to **21.361 s**; its cached
  step fell from 11.294 to **10.520 s**. MPSGraph internally uses
  `commitAndContinue`, so a single command-buffer GPU timestamp covers only a
  fragment; logs mark this as `gpu_timing_fragmented=true` and wall time is the
  authoritative integrated measure. `QI_DISABLE_MPSGRAPH_ATTENTION=1` selects
  the custom flash control. At 256 the isolated gain was only 1.8-3.8%, so the
  lower-overhead custom kernel remains selected. Full evidence is in
  `benchmarks/m1-max-mpsgraph-attention-1024.json`.
- Plan-5 F2 replaces the mixed v3 policy with the all-block-matrix FP16 v4
  policy. All 224 block matrices now bind directly from the 14,230,327,296-byte
  pack, Q8 dispatches fall from 40 to zero per full step, and v4 avoids the
  reusable 96 MiB legacy weight-conversion buffer. Two reversed adjacent 1024
  two-step A/B pairs reduced loop wall from 21.262-21.283 s with v3 to
  **18.985-19.002 s with v4**, a 10.6-10.8% saving. The FP16 prefill step was
  9.967-9.976 s and the cached step 9.003-9.014 s; the cached improvement was
  about 13.9%. MPS handles fused QKV and all three MLP matrices. It also handles
  the attention-output projection for the exact 4,096-row cached shape, but a
  measured M1 Max `MPSMatrixMultiplication` failure produced partial non-finite
  output at the 4,096-plus-prompt prefill shape, so prefill keeps the finite
  custom FP16-weight/FP32-input projection. Smaller resolutions retain that
  custom projection for their better numerical behavior.

  **Prefill row split (2026-09-23).** The joint prefill layout is
  `[text rows][4,096 image rows]`. The prefill now runs the MPS projection
  on exactly the trusted 4,096-row shape: the left operand is offset by
  `text_rows * 8192` bytes and the output by `text_rows * 16384` bytes. The
  custom kernel handles only the 31-32 text rows. The split is finite. The
  1024 two-step oracle is unchanged at 0.000959688 / 0.0014312 nRMSE, and the
  Viggle quoted-text PNG is byte-identical. The earlier failure was
  therefore tied to MPS on the ragged 4,127/4,128-row shape, not to
  overflowing text-row activations. Across two ABBA pairs, step 1 fell from
  9,537-9,552 ms to 8,897-8,901 ms (-0.65 s per image). Cached steps are
  unchanged.

  The official 256 1-, 2-, and 40-step trajectory gates pass at 0.08727%,
  0.12607%, and **1.07623%** nRMSE, respectively, under their unchanged limits
  (the 40-step value is now 0.834196%; see "Timestep rounding" below);
  the 40-step loop measured 20.934 s wall. Cache-DiT 0.24 retained 27 cached
  steps and measured 7.494 s wall with 10.0293% approximation nRMSE. The full
  native prompt-to-PNG regression also passes with zero Q8 dispatches at
  1.90087% final-latent and 1.03176% RGBA nRMSE. A 1024 25-step retry completed
  in 307.868 s end to end, but step time drifted from about 8.64 s to 13.67 s
  while macOS reported `AC Power` and a discharging battery, so it is recorded
  as a thermally/power-confounded observation rather than a speed comparison.
  The adjacent two-step A/B is the F2 throughput result. Details are in
  `benchmarks/m1-max-all-fp16-v4-1024.json`.
- Plan-5 F4 now writes LayerNorm/modulation and SwiGLU results directly to the
  reusable FP16 MPS input buffer. It removes three full FP32-to-FP16 conversion
  dispatches per block and no longer allocates the two FP32 normalization
  intermediates or the FP32 SwiGLU intermediate for v4. At the 4,127-row 1024
  prefill shape this removes **338,083,834 bytes (322.42 MiB)** of scratch.
  `QI_DISABLE_DIRECT_FP16_ACTIVATIONS=1` restores the old path for A/B testing;
  legacy v1-v3 packs select it automatically.

  The gain is real but much smaller than Plan 5 projected. An adjacent 256
  40-step control fell from 20,234.3 to **20,027.7 ms GPU** and from 20,579 to
  **20,356 ms wall** (about 1.0-1.1%). Step-1, step-2, and step-40 nRMSE are
  bit-identical at 0.08727%, 0.12607%, and 1.07623%. Cache-DiT retained all 27
  decisions and measured 7.194 s GPU / 7.404 s wall. Three adjacent 1024
  two-step pairs all favored the direct path, but only by **22-198 ms** over
  roughly 18.2-18.9 s (0.1-1.1%); cached-step savings were the consistent part
  at 45-151 ms. The full native 256 output is byte-for-byte identical to the
  F2 output (`d9967772...` SHA-256). The large elementwise transfers overlap or
  consume less of the integrated step than the plan's bandwidth estimate
  assumed. Exact controls and power-state caveats are in
  `benchmarks/m1-max-direct-fp16-activations.json`.
- Smaller scalar variations were closed before adopting flash: FP16 K/V alone
  saved only 3-3.5% and introduced the same numerical loss; eight queries in
  one scalar SIMD group regressed 33-38% from register pressure; and a
  model-specific fast mask saved only 1.7% in an adjacent full-step control.
  They are not retained because flash captures the useful FP16 bandwidth change
  and removes the dominant repeated K/V pass.
- The same flash kernel is now accepted at 256 after a separate production
  gate. Isolated prefill/cached attention fell from 3.140/3.228 ms to
  **0.441/0.378 ms**, with 0.000266/0.000339 nRMSE against scalar FP32. The
  official 1-, 2-, and 40-step trajectory gates all pass; step-40 latent nRMSE
  is 1.04073% under the 1.1% limit. The reproducible 40-step loop fell from
  25.627 s GPU / 26.263 s wall to **21.727 s GPU / 22.083 s wall**. Cache-DiT
  0.24 retains all 27 decisions and falls from 9.067/9.582 s to **7.700/7.919
  s**. Its measured final latent nRMSE improves slightly from 10.079% to
  10.024%; it remains an explicit approximation rather than an equivalence
  path. FP16 prefix storage halves the canonical cache from 22 to 11 MiB.
- Current blue-teapot prompt-to-PNG measurements at 256 are **44.675 s** for
  cache-off and **27.681 s** for Cache-DiT 0.24. The cache-off run comprised
  13.851 s text, 29.495 s transformer phase including a thermally variable
  23.478 s loop, 1.306 s VAE, and 21 ms PNG output. Cache-DiT comprised 12.368
  s text, 14.056 s transformer phase including a 7.876 s loop, 1.236 s VAE,
  and 19 ms PNG output. Exact isolated, oracle, and end-to-end measurements are
  in `benchmarks/m1-max-flash-attention-256.json`.
- An explicit 25-step FlowMatch mode reduces the same 256 blue-teapot run to
  **35.109 s end to end** without Cache-DiT: 13.330 s text, 20.536 s
  transformer phase including a 14.180 s loop, 1.222 s VAE, and 20 ms PNG.
  That is 21.4% lower end-to-end latency and 39.6% lower denoising-loop wall
  time than the adjacent 40-step run. Manual side-by-side inspection found the
  25-step image extremely close in composition and detail to the 40-step
  image, with only minor highlight and texture differences. There is no
  official 25-step numerical oracle. The value came from Comfy-Org's community
  workflow template, whose own note distinguishes its 25-step starting point
  from the official 40-50-step pipeline. Therefore 40 remains the default and
  the 25-step result is an explicit speed/quality choice.
- Combining 25 steps with Cache-DiT 0.24 reaches **25.225 s end to end**: 12.604
  s text, 11.366 s transformer phase including a 5.463 s loop, 1.232 s VAE,
  and 21 ms PNG; 16 of 25 steps were cached. This is only 8.9% below the
  40-step Cache-DiT run because text, startup, and VAE costs do not scale with
  the denoising count. The image remained coherent and prompt-correct, but its
  body, handle, lid, and highlights diverged more visibly because step
  reduction and residual caching are compounded approximations. It therefore
  remains opt-in. Full measurements and the acceptance rationale are in
  `benchmarks/m1-max-25-step-256.json`.

- The first complete 1024, 25-step, cache-off prompt-to-PNG measurement is
  **298.408 s end to end (4m 58.4s)** for the blue-teapot prompt at seed 42.
  It comprised 12.982 s text conditioning, 275.028 s transformer phase
  including a 268.459 s denoising loop, 10.065 s VAE decode, and 308 ms PNG
  output. The loop averaged **10.738 s per step** and produced a verified
  1024x1024, 4,195,716-byte PNG. MPSGraph fragments command-buffer timestamps,
  so these are wall measurements. During the run macOS reported `AC Power`
  together with `discharging`; battery charge moved from 95% to 87%, and the
  conflicting state is retained rather than normalized away. Full details are
  in `benchmarks/m1-max-25-step-1024.json`.

- Viggle's 4-step DMD distillation of Qwen-Image-2.1
  (`Viggle/Qwen-Image-2.1-viggle-turbo`, revision `bafc91e`, non-commercial
  Qwen Research License, self-described v0.1 preview) was tested and parked.
  Its full fine-tune has the exact pinned 297-tensor BF16 inventory, so the
  unchanged `quantize-transformer` packed it after its single file was split
  into the base two-shard layout; the source round-trip passed. The pack
  header still names the base snapshot because the loader requires it, so
  that pack is experiment-only. At the time, running it needed two opt-in
  switches: `generate-1024 ... 1301 4` and
  `QI_EXPERIMENT_NO_SHIFT_TERMINAL=1`, which reproduces the checkpoint's
  `shift_terminal: null` schedule. Both are now pack metadata. The
  environment switch applies only to packs that do not state
  `shift_terminal`, and it is refused when a pack states `0.02`.
- With the `CASABLANCA` prompt at seed 1301, on AC with the battery charging:

  | 1024 path | Full / cached steps | Loop | End to end | `CASABLANCA` |
  | --- | ---: | ---: | ---: | --- |
  | Viggle, 4 steps | 4 / 0 | 35.856 s | **57.692 s** | missing; `MEET MEAT` |
  | Base, 25 steps | 25 / 0 | 226.128 s | 248.044 s | garbled, invented footer |
  | Base, 25 + Cache-DiT 0.16 | 10 / 15 | 93.564 s | 116.100 s | `CASABLANCHA` |
  | Base, 40 + Cache-DiT 0.16 (earlier) | 13 / 27 | 122.601 s | 145.18 s | exact |

  Per-step cost is identical across checkpoints (about 9 s). The distill is
  therefore 2.5x faster than the fastest exact-text path.

  **Superseded judgment:** this prompt does not quote CASABLANCA, so its
  absence was not a failure. On the quoted-text prompt, Viggle's HF Space and
  this runtime both render every string exactly. The pack is now adopted and
  identified by pack metadata, not by environment switches. See "Current
  records" below. At four steps, the VAE, transformer setup,
  and text take 38% of the total. They become the next targets if a later
  distill passes. These are single-prompt, single-seed observations. Details
  and checksums are in `benchmarks/m1-max-viggle-turbo-4step-1024.json`.

## Remaining optimization phases

**Current records (2026-09-23).** These are 1024x1024 end-to-end runs using
the quoted-text test prompt `a travel poster with the headline "CASABLANCA"
and the tagline "MEET ME AT SUNSET"` at seed 1301:

| Pack / mode | Full steps | End to end | Text |
| --- | ---: | ---: | --- |
| **Viggle distill, 4 steps (adopted default model)** | 4 | **38.2 s** (48.7 s before the fixed-cost pass) | exact |
| **Base, TaylorSeer** | 11 | **115.1 s** | exact |
| Base, Cache-DiT 0.16 | 13 | 135.8 s | exact |
| Base, Cache-DiT 0.24 | 13 | 169.8 s (thermally confounded) | exact |

The base-pack runs had 5-7 GB of swap in use, which raised weight loading
from about 2.4 s to about 5 s. With free memory they would be about 2.5 s
faster. The Viggle record streams its block weights (see the fixed-cost pass
below), so it no longer pays that penalty.

A fresh-process AC-powered confirmation of the adopted Viggle path measured
**39.32 s process wall** / 39.314 s internal: 2.288 s text, 34.672 s
transformer including a 34.370-second four-step loop, 2.203 s VAE, and 0.126 s
PNG. Its output is byte-identical to the 38.18-second record, keeps both quoted
strings exact, uses no swap, and peaks at 7.04 GB process footprint. The
repeatable observed range is therefore about 38-39 seconds.

- **VAE:** fell from 10.2 to 2.8 s after moving convolution to MPSGraph.
- **Startup:** fell by about 4 s after replacing the pack mapping with a
  parallel `pread` copy.
- **Fixed-cost pass (2026-09-23).** Same
  Viggle run, same machine, 5.8 GB swap in use:

  | Phase | Before | After | Change |
  | --- | ---: | ---: | --- |
  | Text | 3,978 ms | 2,275-2,310 ms | page-aligned, 8-way parallel layer reads |
  | Step 1 (prefill) | 9,537-9,552 ms | 8,897-8,901 ms | attention-output row split (ABBA, benchmark) |
  | VAE | 2,752-3,038 ms | 2,133 ms | MPS FP32 attention; weights copied by parallel `pread` |
  | PNG | 299-304 ms | 121-126 ms | table CRC-32, deferred Adler-32, bulk chunk copy |
  | Pack setup | 2,018-4,743 ms | 169-205 ms | block-weight ring (below) |
  | Cached steps | 8,704-8,742 ms | 8,169-8,183 ms | steel attention (below) |
  | End to end | 48.46 s | **38.18-38.21 s** | two quiet-GPU runs |
  | Process peak footprint | 15.73 GB | **7.04 GB** | ring, VAE upsample scratch |

  The baseline's pack load was 4.74 s under swap, against a 2.0-2.3 s
  swap-free copy, so about 2.5 s of the end-to-end gain is swap
  variance. Full numbers, SHA-256s and gates are in
  `benchmarks/m1-max-viggle-4step-1024-fixed-cost.json`. Text and PNG changes are byte-identical,
  and so is the prefill split, whose PNG SHA-256 stayed
  `be8f75ea...2811`. The VAE attention change moves 45 of 4,194,304 RGBA
  bytes by one level (PSNR 96.6 dB); that image's SHA-256 is
  `337c16d9...a8a0`. Steel attention changes the image slightly (PSNR
  38.5 dB), and both quoted strings stay exact. The current reference PNG
  SHA-256 is `b4c54100...0970`.
  - **Review items not applied.**
    - S3 graph precompilation: MPSGraph's first-use compile cannot be
      separated from GPU time in this profile, so it is unmeasured.
    - S6 early layer-0 read: layer 0 now reads in about 66 ms, so there
      is little left to hide.
    - S7 mask-free prefill: subsumed by the steel prefill. Its 34 MB mask
      is still built (about 54 ms at startup) for the fallback path.
    - S8 scalar final LayerNorm and per-step lookups: small, and the
      LayerNorm change would alter reduction order. Per-block commits
      under the ring now hide the lookups.
    - S10 pack/text overlap: superseded by S1 and the ring.
    - S11 resident mode: only helps batched prompts.
    - M3: conflicts with the upsample scratch reuse.
    - M4 activation aliasing: the transformer is no longer the peak phase.
    - M5 transformer placeholders: never resident, so there is nothing to
      free.
    - M6 Q8-on-disk: superseded by the ring.
    - B11 saturating FP16 stores: a clamp would turn a loud non-finite
      abort into silently wrong values. No prompt has overflowed.
    - B12 payload-bound metadata checksum: a format change. Instead,
      every trajectory now prints a `trajectory pack:` line with the
      pack's metadata.
  - **Viggle schedule check.** Viggle's HF Space runs `steps=4` with
    `sigmas=None`, `shift_terminal=None` and `true_cfg_scale=1.0`. That is
    the default `linspace(1, 1/4)` with the resolution mu shift and no
    terminal stretch, giving model timesteps 1000 / 857.19 / 666.76 /
    400.10. This matches `qwen_image/src/scheduler.cplus` and its unit test, and
    confirms that CFG stays off. Viggle has since published a 5-step
    rank-256 LoRA (v0.2) whose card recommends explicit nodes
    `[1.0, 0.875, 0.75, 0.5, 0.25]`. Running it would need a `sigmas=`
    metadata key and a pack-time LoRA merge. It would be a quality
    candidate, not a speed one.
  - **Block-weight ring.** The all-FP16 pack stores the 272 MB of global
    tensors first, then each block's nine tensors contiguously
    (436,224,000 bytes per page-widened block). The production batched
    loop now copies only that global prefix and streams each block into
    one of three ring slots. It uses the same 8-way `F_NOCACHE` reads, two
    blocks ahead of the GPU. Every block is its own command buffer, and a
    slot is refilled only after the command buffer that last read it has
    completed. The ring reads 14 GB per step at about 7 GB/s behind an
    8.6 s GPU step. In two adjacent ABAB pairs of
    `benchmark-transformer-trajectory-1024 viggle 4`, the loop measured
    34,770/34,800 ms against 34,670/34,688 ms resident (+0.1 s), and pack
    setup fell from 2,257/2,275 ms to 175/180 ms. The 1024 two-step oracle
    is identical to every printed digit (0.000959688 / 0.0014312). Its
    peak footprint fell from 15,685,742,208 to 3,119,712,944 bytes. The
    Viggle PNG is byte-identical. Because the pack is no longer the
    largest allocation, the 5-7 GB swap penalty on pack loading no longer
    applies. The ring covers the cache-off batched 1024px path on all-FP16
    packs, which is the adopted Viggle path. At 256px its cached step
    measured about 1.9 s with the ring versus 0.58 s resident, so smaller
    resolutions use the resident copy by default. `QI_ENABLE_WEIGHT_RING=1`
    opts into streaming there to save memory. Cache-DiT/TaylorSeer, Q4,
    profiling and per-block validation runs keep the resident copy.
    `QI_DISABLE_WEIGHT_RING=1` restores it everywhere and overrides the
    opt-in. The trajectory startup line reports the ring choice, slot size
    and resident bytes. Per-step lines include ring read and slot wait times;
    `gpu_ms` includes completed ring block command buffers.
  - **Timestep rounding (precision fix).** The pinned pipeline casts the
    scheduler timestep to the BF16 latent dtype and then divides by 1000
    in BF16, so the model sees `bf16(bf16(t) / 1000)`. The runtime rounded
    only once, `bf16(t / 1000)`. That is one BF16 ulp off on 15 of the 40
    base-pack steps at 256 and 12 of 40 at 1024. The four Viggle timesteps
    are unaffected, so its PNG is unchanged. With the pipeline's rounding,
    the official 256 40-step oracle falls from 1.07623% to **0.834196%**
    final-latent nRMSE (limit 1.1%); steps 1 and 2 are unchanged.
  - **Steel attention for cached steps.** MLX's steel attention kernel
    (`ml-explore/mlx` v0.32.2, MIT; flattened into
    `qwen_image/kernels/mlx_steel_attention.metal`) uses only `simdgroup_matrix`
    operations, so it compiles here, unlike Metal FlashAttention. At the
    cached 1024 shape (4,096 queries x 4,127-4,128 keys, 32 heads, D 128,
    FP16), MLX's own SDPA measured 36.4-40.8 ms per block against about
    49.5 ms for MPSGraph. The runtime now runs the FP16/D128 instantiation
    (BQ 32, BK 16, four simdgroups) on every cached step. It reads the
    FP16 K/V the MPSGraph path already prepares. It writes FP16 output
    straight into the attention-output GEMM's half input, which also
    removes the FP32 round trip and conversion dispatch. The masked
    prefill step keeps MPSGraph. The 1024 two-step oracle's step 2 measures
    0.00143106 (MPSGraph 0.0014312; limit 0.0015). The Viggle quoted-text
    image still renders both strings exactly (PSNR 38.5 dB against the
    MPSGraph image). In two ABAB pairs of the 4-step benchmark, the loop
    fell from 35,566/35,857 ms to 34,045/34,597 ms (about -1.4 s per
    image; cached steps about -0.45 s each). `QI_DISABLE_STEEL_ATTENTION=1`
    restores MPSGraph SDPA.

    The masked prefill splits into two unmasked problems. The 4,096 image
    queries see every key, so they use the cached kernel. The 31-32 text
    queries use a causal specialization over the text keys only, which is
    exactly the prefill mask. The text rows' attention output is then FP16,
    so it goes through its own small MPS multiplication instead of the
    custom FP32-input kernel. The 1024 two-step oracle improves to
    0.000954478 / 0.00142946 (previously 0.000959688 / 0.00143106).
    **It is slower, so it is opt-in (`QI_STEEL_PREFILL=1`).** On a quiet
    GPU (0% utilization between runs, 30 s pauses), the 4-step loop measured
    41,097-44,013 ms with it and 33,483-40,629 ms without. Every later
    cached step slowed as well, to 10.2-11 s against 8.18 s, although the
    cached code path is identical. The mechanism is not understood. The
    graph-based prefill stays the default.
  - **MPS weight layout (rejected).** Every block GEMM binds weights as
    `[out, in]` with `transposeRight = YES`. A throwaway spike timed the
    production M = 4096 shapes with synthetic FP16 data against weights
    stored `[in, out]` (`transposeRight = NO`), ten multiplications per
    command buffer, in two ABBA passes. QKV/gate (N 12,288, K 4,096)
    changed by -2.4% to +0.2%, and attention-out (4,096 x 4,096) by -0.6%
    to -0.4%. The down projection (K 12,288) became 1.43-1.91x slower.
    MPS's kernel selection does not favour the other layout, so rewriting
    the pack is not worth it, and the spike was removed. The down
    projection remains the least efficient shape (about 5.8 against
    8.6 TFLOP/s).
  - **Sustained load.** A 1024/40 base-pack oracle run with the ring
    rose from 8.6 s to 12-13 s per step over 40 steps. This matches the
    earlier sustained-load record (hot 25-step cached average 12.9 s) and
    does not reach the 4-step path.
  - **VAE attention.** The middle-block attention was one threadgroup per
    query, with a 7-level barrier reduction for every key: 253.6 ms. It is
    now scores = QK^T / sqrt(C) (FP32 MPS GEMM over strided column slices
    of the qkv buffer), an in-place row softmax, then P V (FP32 MPS GEMM):
    10.2 ms. The official 1024 VAE oracle improved from 2.25238e-07 to
    2.19341e-07. The 256 oracle moved from 2.42725e-07 to 2.43162e-07
    (limit 3e-06), and the small-fixture middle block measures 1.92e-07
    (limit 1e-06). `QI_DISABLE_VAE_MPS_ATTENTION=1` restores the old
    kernel. The score matrix adds 64 MiB of scratch at 1024.
  - **VAE weights.** A no-copy mapping makes the first dispatch fault and
    wire 1.26 GiB inside the decode. With warm pages the copy saves only
    about 25 ms. After the transformer phase has evicted them, it saves
    about 0.37 s (221 + 1,781 ms versus 1 + 2,373 ms).
    `QI_DISABLE_VAE_PREAD=1` restores the mapping.
  - **VAE memory.** The fused `resize -> conv2d -> bias` upsample graphs
    made MPSGraph materialise every nearest-2x resize output (75.5 + 302
    + 604 + 1,208 MB). A small `vae_nearest_upsample_2x` kernel now writes
    that tensor into `buffer_c`, which is dead scratch at every upsampler,
    and a plain conv graph runs at the doubled shape. The upsample
    boundaries and the 1024 oracle are unchanged to every printed digit.
    Timing is unchanged. The unused 81 MiB parity-packed upsample buffer is
    no longer allocated on the MPSGraph path. VAE-phase peak footprint
    (`test-vae-decoder ... trajectory-1024`) was 7,757,677,312 bytes at
    HEAD and 9,176,844,016 bytes with the weight copy. It is now
    **6,985,303,768 bytes**, 0.77 GB below HEAD. The review's `buffer_c`
    narrowing (M3) conflicts with this reuse and was not applied.
  - **VAE timing line.** `VAE timing:` splits the phase. Setup is under
    0.1 s. The rest is one commit/wait of about 1.8 s. The per-dispatch
    profile cannot separate MPSGraph's first-use compile from GPU time, so
    the review's graph precompilation item remains unmeasured.

A full 1024 transformer step costs about 8.8 s on either pack (about 8.2 s
for cached steps with steel attention, below). MPS GEMMs run it at 76-88% of
peak (see the 1024 step ablation profile). Attention ran at 62% with
MPSGraph. Metal FlashAttention was slower on this compiler, but MLX's steel
attention kernel is about 25% faster and now runs the cached steps. Further
large gains still come from fewer full steps.

**Test prompt correction.** Earlier today, text was judged with the old
prompt: `A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET',
bold geometric lettering.` That prompt quotes only `MEET ME AT SUNSET`, so
leaving CASABLANCA out was not an error; only misspellings count.

- **Re-judged on the new prompt:** Viggle and TaylorSeer both pass.
- **Still invalid:** 25 steps with Cache-DiT misspelled CASABLANCA
  (`CASABLANCHA`) after choosing to render it.
- **Viggle step counts:** 3 steps (39.5 s) added a garbled extra text line.
  8 steps (80.2 s) was clean but slower than 4.
- **Caching the distill:** Cache-DiT cannot cache the 4-step distill. Its
  consecutive steps differ by 0.28-0.32 relative L1, far above any usable
  threshold. That measurement used `QI_CACHE_DIT_WARMUP=N`, which lowers
  the four full warmup steps. It is read once per trajectory, printed in the
  `trajectory switches:` line, and must be 1-4. Any other value is refused.

The native-quality product target is 2048x2048; 1024x1024 remains the practical
performance mode. Native 2048 feasibility is complete: the 16,384-token
one-step transformer smoke, standalone VAE, and four-step end-to-end generation
all produced finite output. The full run took 309.664 s and peaked at 24.57 GB
without swap, so future 2048 work must continue to record memory as well as
time. An optimization that only helps the 256x256 benchmark is no longer
sufficient evidence:

1. **1024 precision follow-up.** The official 40-step transformer and VAE
   oracles are complete. Cache-off is visually close but reaches 13.211%
   final-latent nRMSE against official BF16; determine whether a bounded
   higher-precision accumulation or operand path can reduce that long-horizon
   drift without losing the all-FP16 v4 throughput. Do not loosen a numerical
   gate merely because this poster remains readable. Cache-DiT calibration is
   also complete: use 0.12 for the conservative quality tradeoff, use 0.16 for
   the measured speed-biased tradeoff, and reject 0.24 at 1024/40.
2. **1024 transformer tail.** MPSGraph SDPA, the all-FP16 v4 pack, and direct
   FP16 normalization/SwiGLU output have completed Plan-5 F1, F2, and F4.
   First-order TaylorSeer F8 is also complete: its predictor passed the 256
   numerical calibration, but the faster cap-four policy failed the controlled
   1024 exact-text visual A/B, so it remains experimental. The
   Bottleneck Sampling F9's sub-native spike is complete: both the literal FLUX
   policy and a Qwen terminal-sigma adaptation failed the 256 visual gate. A
   1024→512→1024 sweep remains closed, but 2048→1024→2048 is a materially
   different native-resolution proposal that can now be reconsidered because
   2048 memory feasibility is established. The
   direct FP16 K/V preparation is now also closed. A K-only prototype that
   preserved FP32 QKV projection output and rounded normalized/rotated K at its
   final store was numerically identical to the established two-step oracle,
   but changed the 1024 two-step transformer loop by only 14 ms (18.390 s
   versus 18.404 s, 0.08%). A broader prototype made the fused MPS QKV
   multiplication write FP16 directly, then consumed half Q/K/V without the
   intermediate FP32 traffic. Across two adjacent pairs it averaged 18.442 s
   versus 18.222 s for the current FP32-output path: a 0.220 s / 1.21%
   regression. Its fragmented GPU counter was slightly lower, but end-to-end
   loop wall time is the acceptance metric. Both prototypes were removed;
   see `benchmarks/m1-max-direct-fp16-qkv-1024.json`. Scalar attention
   variants, further Q8, and additional broad activation conversions remain
   closed.

   The 2026-09-23 ablation profile, split-GEMM, and Metal FlashAttention
   results close the obvious exact-arithmetic kernel levers. Any further
   large gain needs fewer full transformer evaluations while exact text
   survives. The candidates are a future distilled checkpoint and MeanCache.
3. **1024 working memory.** Shape-specific VAE arena sizing already removed
   1.6875 GiB, and F5 measured a 5,603,394,208-byte peak footprint with
   5,577,375,744 bytes of explicit scratch. Further reduction requires a real
   liveness/aliasing schedule rather than smaller fixed capacities: block 4
   simultaneously needs two wide and three narrow residual buffers, while the
   block-3 upsample owns the remaining wide shortcut. Measure peak footprint
   after every aliasing change so transformer, text, and VAE phases remain safe
   on the 32 GB M1 Max. The default MPSGraph convolution path raises the
   1024 VAE peak to 7,758,496,536 bytes, about 2.15 GB of internal graph
   scratch. When MPSGraph is active, the 81 MiB (84,934,656-byte)
   parity-packed upsample weight buffer is unused and could be skipped.
4. **Shared startup and small-kernel work.** Text layer streaming has completed
   Plan-5 F3: the measured text working set is below 1 GB and standalone text
   time fell from 16,242 to 3,145 ms without changing output.
   - **Transformer setup, diagnosed 2026-09-23.** The former 6.5-7.5 s was
     almost entirely the first dispatch wiring the no-copy pack. The parallel
     `pread` copy reduces it to 2.3-2.6 s.
   - **Text reads now run near SSD speed.** Aligned, parallel layer reads
     cut the text phase to about 2.3 s. Overlapping the pack copy with it
     would now compete for the same saturated SSD, so it is no longer a
     candidate. Whole-shard text
   readahead is superseded; tensor-order `madvise`, broad text Q8, four-query
   attention at 256, and blanket 256-thread elementwise groups stay rejected
   unless new evidence changes their tradeoffs.
Completed transformer submission batching, fused QKV, and the remaining VAE
work are no longer remaining phases. Native-prompt 1024/40 cache-off and
Cache-DiT 0.12 production timings are also complete for warm-filesystem,
fresh-process conditions; a reboot-cold run is an environmental repeat rather
than an implementation phase. Other closed branches are not remaining phases: selective text readahead
regressed wall time; the calibrated text-Q8 policies failed the downstream latent gate;
process reuse without simultaneous model residency provided no warm-request
gain; four-query attention remains rejected at 256 (but is now selected at
1024), and blanket 256-thread elementwise groups regressed throughput.

## Quantization decision log

The quantization policy is measurement-driven and deliberately conservative:

1. Uniform Q4 was rejected at block 0. Its best complete-block candidate was
   already 13.68% nRMSE, so propagating it through 32 blocks had no plausible
   path to the 1% full-transformer budget.
2. Affine Q8/group-64 passed the original isolated block-0 fixture at 0.837%.
   That justified implementing the packed format and Metal kernel, but not
   generalizing the policy: one four-token block does not expose accumulated
   error or prompt/timestep/resolution sensitivity.
3. Uniform Q8 across all 32 blocks was then tested on six full-transformer
   cases. It failed four, reaching 1.921% final-noise nRMSE. Error at sampled
   internal outputs reached 4.000% around block 15, showing that local block
   success does not compose uniformly.
4. Coarse mixed policies did not solve this. Quantizing either 16-block half,
   all attention matrices, all MLP matrices, or any single matrix role across
   every block exceeded the gate. Even every contiguous eight-block group
   failed; blocks 24-31 were best at 1.130% while blocks 0-7 reached 1.986%.
   The strong depth dependence is why the policy is expressed per block rather
   than only per tensor role.
5. Four-block scans found only blocks 28-31 safe when all seven roles were Q8
   (0.906%). A role refinement then found that attention can extend through
   blocks 24-31. Adding both MLP projection and MLP output in blocks 28-31
   produced the smallest tested passing layout at 0.985%; substituting the MLP
   gate crossed the threshold. This is the current measured frontier, not a
   final release claim.

The calibration uses real prompt embeddings and exact scheduler timesteps,
but deterministic Gaussian latent states rather than a replayed denoising
trajectory. It also emulates packed FP16 values with PyTorch MPS matmul, which
is not bit-exact Metal product rounding. Because the selected frontier has only
a narrow margin, it remains provisional until the mixed packed runtime passes
native Metal comparisons and replayed-trajectory calibration. The simpler
blocks-28-31-only policy remains the fallback; it measured 0.906% at the cost
of about 61 MiB more block-matrix storage.

The full writer intentionally keeps the nine non-block tensors in BF16: they
were not part of the 224-matrix calibration, so quantizing them would extend
the policy beyond its evidence. MLP gates and dense Q/K/V are not quantized;
v3 stores their already-selected FP16 execution operands instead of converting
BF16 on every denoising step. QIPACK1 still accepts v1, v2, and the original
block-0 scope, preserving existing artifacts and fixtures.

The original four-token fixture remains useful as a cheap block-chain
regression. The newer 15-token fixture proves the whole transformer boundary
and the real interleaved token semantics, but it is still not a throughput
claim. The packed mapping is exposed to Metal without copying 12.42 GiB of
weights; only activations, cache arenas, and 128-element Q/K norm vectors are
copied.

Tokenization, text encoding, prompt-sized transformer metadata/RoPE, seeded
noise, denoising, VAE decode, postprocessing, and PNG output now form one
native `generate-256` command. The pinned fox case verifies the complete path;
arbitrary prompts use the same path but naturally have no numeric oracle unless
a matching reference fixture is generated. Production use still needs larger
output-size validation and further kernel optimization. Text weights now use
the exact two-slot layer streamer rather than the rejected affine-Q8 candidates.
The current VAE path intentionally implements the pinned one-frame first-chunk
semantics; temporal continuation and tiled decode remain outside its verified
scope.
