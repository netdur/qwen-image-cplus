# qwen-image-cplus

Native Qwen-Image-2.1 inference work for Apple Silicon, written in C+ and
Metal Shading Language. The implementation target is an Apple M1 Max with
32 GB unified memory.

The runtime is intentionally model-specific. It does not embed Python,
PyTorch, Diffusers, C, C++, Objective-C source, or CMake. A separate Python
development tool may generate small oracle fixtures from the pinned official
Diffusers source; those fixtures are plain binary files consumed by C+ tests.

## Pinned reference

- Model: `Qwen/Qwen-Image-2.1`
- Snapshot: `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- Diffusers commit: `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
- Machine-readable inventory: `manifests/qwen-image-2.1.json`

## Build and verify

```sh
cpc fmt --check src
cpc check
cpc build
cpc test
./target/debug/qwen-image-cplus probe-stress
./target/debug/qwen-image-cplus test-metal-primitives
./target/debug/qwen-image-cplus test-metal-linear
./target/debug/qwen-image-cplus benchmark-linear
./target/debug/qwen-image-cplus test-metal-int8-linear
./target/debug/qwen-image-cplus benchmark-int8-linear
./target/debug/qwen-image-cplus test-transformer-block /path/to/model/snapshot
./target/debug/qwen-image-cplus quantize-block0 /path/to/model/snapshot block0.qipack
./target/debug/qwen-image-cplus quantize-transformer /path/to/model/snapshot transformer.qipack
./target/debug/qwen-image-cplus verify-packed block0.qipack
./target/debug/qwen-image-cplus verify-packed transformer.qipack
./target/debug/qwen-image-cplus verify-packed-source block0.qipack /path/to/model/snapshot
./target/debug/qwen-image-cplus verify-packed-source transformer.qipack /path/to/model/snapshot
./target/debug/qwen-image-cplus test-transformer-block-int8 block0.qipack
./target/debug/qwen-image-cplus verify-model /path/to/model/snapshot
```

Inspect a shard, optionally filtering tensor names:

```sh
./target/debug/qwen-image-cplus inspect /path/to/shard.safetensors proj_out
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
  atomic temp-file installation. The format is documented in
  `docs/packed-format-v1.md`.
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
- A full-transformer QIPACK1 writer/reader for that mixed policy. It generates
  the complete 297-tensor inventory from nine global tensor definitions and a
  nine-role block schema, preserving BF16 everywhere except the measured 40
  Q8 matrices. The writer refuses a source other than the exact 7,115,124,736-
  parameter, two-shard inventory; installs atomically only after structure,
  payload, per-tensor, and exact source round-trip checks; and produced a
  verified 13,334,843,392-byte artifact from the pinned snapshot. The layout
  and scope compatibility rules are documented in `docs/packed-format-v1.md`.

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
the policy beyond its evidence. Likewise, MLP gates remain BF16 because the
role search showed that exchanging projection/output for the gate crossed the
1% error threshold. QIPACK1 still accepts the original block-0 scope so the
existing isolated Metal fixture remains reproducible.

Image generation is not implemented yet. The next transformer step is native
execution directly from the full mixed artifact, followed by native Metal
boundary comparisons and denoising-state replay. The native prompt encoder and
causal 3D VAE are also tracked in `plan.md`.
