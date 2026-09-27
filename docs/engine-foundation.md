# Engine foundation and correctness gates

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
