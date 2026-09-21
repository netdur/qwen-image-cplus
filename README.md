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
- Pinned official [transformer source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/transformers/transformer_qwenimage21.py),
  [VAE source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py),
  and [pipeline source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- Text model reference: Transformers 5.17.0
  [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

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
./target/debug/qwen-image-cplus test-attention-cache
./target/debug/qwen-image-cplus benchmark-attention
./target/debug/qwen-image-cplus test-transformer-block /path/to/model/snapshot
./target/debug/qwen-image-cplus quantize-block0 /path/to/model/snapshot block0.qipack
./target/debug/qwen-image-cplus quantize-transformer /path/to/model/snapshot transformer.qipack
./target/debug/qwen-image-cplus verify-packed block0.qipack
./target/debug/qwen-image-cplus verify-packed transformer.qipack
./target/debug/qwen-image-cplus verify-packed-source block0.qipack /path/to/model/snapshot
./target/debug/qwen-image-cplus verify-packed-source transformer.qipack /path/to/model/snapshot
./target/debug/qwen-image-cplus test-transformer-block-int8 block0.qipack
./target/debug/qwen-image-cplus test-transformer-mixed transformer.qipack
./target/debug/qwen-image-cplus test-transformer-complete transformer.qipack
./target/debug/qwen-image-cplus test-transformer-scale transformer.qipack 256
./target/debug/qwen-image-cplus test-transformer-scale transformer.qipack 512
./target/debug/qwen-image-cplus test-transformer-scale transformer.qipack 1024
./target/debug/qwen-image-cplus test-transformer-trajectory transformer.qipack 1
./target/debug/qwen-image-cplus test-transformer-trajectory transformer.qipack 2
./target/debug/qwen-image-cplus test-transformer-trajectory transformer.qipack 40
./target/debug/qwen-image-cplus test-transformer-cache-dit transformer.qipack 0.12
./target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot small
./target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot 256
./target/debug/qwen-image-cplus test-image-output reference.png
./target/debug/qwen-image-cplus test-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./target/debug/qwen-image-cplus test-pipeline-cache-dit-256 transformer.qipack /path/to/model/snapshot cache.png 0.12
./target/debug/qwen-image-cplus test-native-inputs
./target/debug/qwen-image-cplus test-native-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./target/debug/qwen-image-cplus generate-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101
./target/debug/qwen-image-cplus generate-256-cache-dit transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.12 1101
./target/debug/qwen-image-cplus benchmark-process-reuse-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 0.24 2 1101
./target/debug/qwen-image-cplus test-tokenizer /path/to/model/snapshot
./target/debug/qwen-image-cplus test-text-encoder /path/to/model/snapshot
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
  nine-role block schema. QIPACK policy v2 preserves BF16 for vectors, global
  matrices, and attention; stores the 88 non-Q8 MLP matrices as the FP16
  operands consumed by Metal/MPS; and retains the measured 40 Q8 matrices.
  The reader remains compatible with the original all-BF16/Q8 v1 policy. The
  writer refuses a source other than the exact 7,115,124,736-
  parameter, two-shard inventory; installs atomically only after structure,
  payload, per-tensor, and exact source round-trip checks; and produced a
  verified 13,334,843,392-byte artifact from the pinned snapshot. The layout
  and scope compatibility rules are documented in `docs/packed-format-v1.md`.
- Native execution of all 32 blocks directly from one read-only, page-aligned
  no-copy QIPACK1 Metal buffer. Kernel selection comes from each tensor record,
  not a second hard-coded policy: blocks 0-23 discover zero Q8 matrices,
  blocks 24-27 discover four, and blocks 28-31 discover six. A compact
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
  and byte ranges. It no longer rescans the complete 13.33 GB payload on every
  generation. Set `QI_VERIFY_PACKED_CHECKSUMS=1` when loading an artifact whose
  provenance is uncertain; that adds all per-tensor checksum checks. The
  explicit `verify-packed` audit remains exhaustive, checking both the whole
  payload and every tensor, and passes the full 13,334,843,392-byte artifact.
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
  faster on BF16 4096x4096 attention projections (2.00 ms versus 2.83 ms), so
  those remain custom Metal. Isolated MPS output nRMSE is at most 0.02095%.
- Cached MLPs convert FP32 activations and only Q8 packed weights to FP16
  scratch storage, bind v2 FP16 weights directly from the read-only QIPACK
  mapping, and invoke `MPSMatrixMultiplication` with FP32 output.
  Conversion, MPS GEMM, SwiGLU, and residual work remain ordered in one command
  buffer per transformer block; there is no CPU inference fallback or host
  synchronization between operations. One reusable 96 MiB weight buffer still
  handles Q8 matrices, while a 6 MiB activation buffer feeds every MPS GEMM.
  The 278-row first step deliberately retains the bounds-safe custom path; its
  kernels distinguish BF16 and FP16 records and produce the same FP16 operands
  as before. Steps 2-40 use MPS for all three MLP matrices.
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

## Remaining optimization phases

The original timing target is no longer a stopping condition. The remaining
work is ordered by architectural leverage and evidence, not by whether a
particular total has already been reached:

1. **Fused QKV with FP16 attention storage.** Define a versioned QIPACK policy
   that stores dense attention matrices in the same FP16 operand form already
   consumed by the accepted custom kernels. Fuse Q, K, and V into a wider
   projection and add row-stride/offset support to Q/K norm, attention, and V
   consumers. Keep the existing Q8 policy for its calibrated late-block
   matrices. This requires regenerating and validating the packed artifact.
2. **Remaining VAE work.** The dominant convolution is already a verified
   FP32 SIMD-group kernel. Still open are fewer command-buffer waits, persistent
   pipeline/scratch reuse where the phase lifetime permits it, and the
   parity-decomposed nearest-upsample convolution experiment. The 3e-6 decoder
   and RGBA byte gates remain stricter than the transformer gates.
3. **Text-encoder GPU efficiency.** Replace the low-row scalar linear path and
   hundreds of synchronous dispatches without changing its BF16 store
   boundaries. Whole-shard readahead stays accepted; tensor-order selective
   advice and broad Q8 text policies stay rejected unless a materially new
   layout or calibration method is introduced.
4. **Kernel-specific elementwise fusion.** Fuse attention residual with the
   following LayerNorm and evaluate a cooperative final LayerNorm. A global
   256-thread elementwise launch was measured and rejected, so residual,
   SwiGLU, final norm, and small conditioning kernels must be tuned separately.
5. **End-to-end remeasurement.** After the structural phases, repeat both
   cache-off and Cache-DiT native prompts with warm-filesystem and cold-page
   conditions reported separately. Transformer loop time, phase wall time,
   process wall time, memory footprint, and numerical/perceptual gates remain
   separate measurements.

Completed transformer submission batching is also no longer a remaining
phase. Other closed branches are not remaining phases: selective text readahead
regressed wall time; the calibrated text-Q8 policies failed the downstream latent gate;
process reuse without simultaneous model residency provided no warm-request
gain; four-query attention and blanket 256-thread elementwise groups regressed
throughput.

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
the policy beyond its evidence. MLP gates are not quantized; v2 merely stores
their already-selected FP16 execution operands instead of converting BF16 on
every denoising step. QIPACK1 still accepts both the legacy full-transformer
policy and the original block-0 scope, preserving existing fixtures.

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
output sizes, a text-weight storage strategy that passes the downstream gates,
and further kernel optimization; the measured affine-Q8 candidates do not.
The current VAE path intentionally implements the pinned one-frame first-chunk
semantics; temporal continuation and tiled decode remain outside its verified
scope.
