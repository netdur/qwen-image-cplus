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
./target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot small
./target/debug/qwen-image-cplus test-vae-decoder /path/to/model/snapshot 256
./target/debug/qwen-image-cplus test-image-output reference.png
./target/debug/qwen-image-cplus test-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./target/debug/qwen-image-cplus test-native-inputs
./target/debug/qwen-image-cplus test-native-pipeline-256 transformer.qipack /path/to/model/snapshot output.png
./target/debug/qwen-image-cplus generate-256 transformer.qipack /path/to/model/snapshot output.png "your prompt" 1101
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
  nine-role block schema, preserving BF16 everywhere except the measured 40
  Q8 matrices. The writer refuses a source other than the exact 7,115,124,736-
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
- The real Qwen-Image-2.1 VAE still-image decoder now runs natively from its
  1.258 GiB FP32 Safetensors file through one read-only no-copy Metal mapping.
  It applies the exact 64-channel latent mean/std handoff, post-quant and input
  convolutions, the 1,152-channel residual/one-head-attention middle block,
  five residual up blocks, final RMSNorm/SiLU, output convolution, and clamp.
  Activations use HWC order so the shared 16x16 tiled OIHW convolution can read
  checkpoint weights without transposing or repacking them.
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
output sizes, text-encoder quantization, and further kernel/command-submission
optimization.
The current VAE path intentionally implements the pinned one-frame first-chunk
semantics; temporal continuation and tiled decode remain outside its verified
scope.
