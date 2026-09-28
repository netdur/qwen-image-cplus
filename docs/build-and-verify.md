# Build and verify

Run the commands below from the repository root unless a command changes
directories explicitly.

## Distribution build

`build.sh` is the distribution build. It builds the engine, CLI, generated C
ABI, and GUI, then packages the app bundle and consolidates C+'s dependency
slices into libraries a C or Objective-C application can link directly. It
also compiles and runs the C ABI smoke test. Release is the default;
`BUILD_MODE=debug` selects debug. The current source requires C+ 0.0.29 or
newer because it uses `#bitcast`.

```sh
CPC=/path/to/cpc ./build.sh
```

The resulting install-shaped tree is:

```text
dist/bin/qwen-image-cplus
dist/Qwen Image.app/
dist/include/qwen_image.h
dist/lib/libqwen_image.a
dist/lib/libqwen_image.dylib
```

### Windows

On Windows, `build.sh` (from Git Bash) hands over to
`scripts/build-windows.ps1`, which can also be run directly from PowerShell.
It builds the CUDA engine, the CLI, the GUI and the C library, checks that
every binary finds every DLL it imports, and runs the C ABI smoke test. It
needs Visual Studio 2022 with the C++ x64 tools, LLVM 19 or newer (`clang`,
`llvm-ar`, `llvm-readobj`), and three libraries named by environment variable:

- `CUDA_HOME`: a CUDA 12 toolkit.
- `CUDNN_HOME`: cuDNN 8 for CUDA 12.
- `JPEG_HOME`: libjpeg-turbo, built static against the static C runtime,
  because cpc links C+ programs with `/MT`.

`scripts/install-windows-deps.ps1 FOLDER` installs all three from pinned,
hash-checked downloads (CUDA from NVIDIA's per-component archives, so no
installer, administrator or driver change) and prints the three variables.
The release workflow uses the same script.

libjpeg-turbo is required rather than `stb_image` alone, as on Linux: it
decodes JPEGs to Pillow's exact pixels, and `stb_image` differs by up to three
levels on ordinary 4:2:0 photos (see `qwen_image_cuda/native/image.c`).

The C+ compiler comes from the commit `scripts/install-cpc-source.sh` pins,
with each package's `vendor\` pointed at that checkout by
`scripts/link_vendor.sh` (directory junctions on Windows).

```powershell
scripts\install-windows-deps.ps1 C:\deps   # once; prints CUDA_HOME, CUDNN_HOME, JPEG_HOME
$env:CPC = "C:\path\to\cpc.exe"
$env:CUDA_HOME = "C:\deps\cuda"
$env:CUDNN_HOME = "C:\deps\cudnn"
$env:JPEG_HOME = "C:\deps\libjpeg-turbo"
scripts\build-windows.ps1
```

```text
dist\bin\qwen-image-cplus.exe
dist\bin\qwen-image-gui.exe
dist\bin\qwen_image.dll
dist\bin\cudart64_12.dll, cublas*.dll, cudnn*.dll
dist\include\qwen_image.h
dist\lib\qwen_image.lib
dist\lib\qwen_image_static.lib
```

For development, verify the package boundaries independently:

```sh
cpc fmt --check qwen_image/src/api.cplus qwen_image/src/engine.cplus cli/src/main.cplus ffi/src/ffi.cplus
(cd qwen_image && cpc check && cpc test)
(cd qwen_image_metal && cpc check && cpc test)
(cd cli && cpc check && cpc build && cpc test)
(cd ffi && cpc check && cpc build && cpc test)
(cd gui && cpc check && cpc build)
(cd qwen_image_metal_dev && cpc build)
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev probe-stress
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-metal-primitives
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-metal-linear
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-linear
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-metal-int8-linear
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-int8-linear
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-attention-cache
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-attention
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-production-attention 256
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-production-attention 1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-transformer-trajectory-1024 transformer.qipack 1
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-transformer-trajectory-1024 transformer.qipack 2
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-transformer-trajectory-1024 transformer.qipack 13
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-transformer-trajectory-2048 transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-block /path/to/model/snapshot
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev quantize-block0 /path/to/model/snapshot block0.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev quantize-transformer /path/to/model/snapshot transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev quantize-transformer-q4 /path/to/model/snapshot transformer-q4.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev verify-packed block0.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev verify-packed transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev verify-packed-source block0.qipack /path/to/model/snapshot
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev verify-packed-source transformer.qipack /path/to/model/snapshot
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-block-int8 block0.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-mixed transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-complete transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-scale transformer.qipack 256
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-scale transformer.qipack 512
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-scale transformer.qipack 1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-trajectory transformer.qipack 1
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-trajectory transformer.qipack 2
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-trajectory transformer.qipack 40
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-trajectory-1024 transformer.qipack 40 /path/to/trajectory_1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-cache-dit transformer.qipack 0.12
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-cache-dit-1024 transformer.qipack 0.12 /path/to/trajectory_1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-taylorseer transformer.qipack 0.24
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-transformer-bottleneck transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-decoder /path/to/model/snapshot small
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-decoder /path/to/model/snapshot 256
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-decoder /path/to/model/snapshot 1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-decoder /path/to/model/snapshot 1024-profile
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-decoder /path/to/model/snapshot 2048
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-decoder /path/to/model/snapshot trajectory-1024 /path/to/vae_oracle_1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-image-output reference.png
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-image-input /tmp/reference-input.png
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-image-orientation tests/fixtures/image_orientation
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vae-encoder /path/to/model/snapshot /path/to/reference.png 512
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vision-encoder /path/to/model/snapshot /path/to/reference.png 512
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-vision-encoder-oracle /path/to/model/snapshot /path/to/vision_fixture
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-pipeline-1024-oracle transformer.qipack /path/to/model/snapshot output.png /path/to/trajectory_1024 /path/to/vae_oracle_1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-pipeline-cache-dit-1024-oracle transformer.qipack /path/to/model/snapshot cache.png 0.12 /path/to/trajectory_1024 /path/to/vae_oracle_1024
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-pipeline-cache-dit-256 transformer.qipack /path/to/model/snapshot cache.png 0.12
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-native-inputs
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-native-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101 25
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-256-cache-dit transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.12 1101 25
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-256-bottleneck transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-1024 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101 25 none
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-1024 viggle.qipack /path/to/model/snapshot output.png "your prompt" 1301
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-1024 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1301 40 cache-dit-0.16
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev pack-metadata transformer.qipack
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev pack-metadata transformer.qipack steps=40 cache=taylorseer
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-1024-cache-dit transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.12 1101 25
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-1024-taylorseer transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.24 1101 40
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev generate-1024-bottleneck transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-q4-eager-1024 transformer-q4.qipack /path/to/model/snapshot eager.png "your prompt" 1301 40
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-q4-inference-1024 transformer-q4.qipack /path/to/model/snapshot destination.png "your prompt" 1301 40
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-q4-direct-1024 transformer-q4.qipack /path/to/model/snapshot direct.png "your prompt" 1301 1
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev benchmark-process-reuse-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.24 2 1101
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-tokenizer /path/to/model/snapshot
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-multi-image-prompt /path/to/model/snapshot
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-multi-image-conditioning /path/to/model/snapshot first.png second.png "Combine both references"
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-multi-image-transformer transformer.qipack /path/to/model/snapshot first.png second.png "Combine both references" 2
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev test-text-encoder /path/to/model/snapshot
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev verify-model /path/to/model/snapshot
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
recommendations. They use the fixed 4+13+8 stage experiment described in
[the 1024 validation record](1024-validation.md);
the FLUX-tuned policy failed Qwen's 256px visual gate and was not promoted to a
1024px run.

Inspect a shard, optionally filtering tensor names:

```sh
./qwen_image_metal_dev/target/debug/qwen_image_metal_dev inspect /path/to/shard.safetensors proj_out
```
