# 1024 validation and optimization

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
  `qwen_image_metal/src/mps_graph_conv.cplus`, including 1x1 layers and the nearest-2x
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
  (the 40-step value is now 0.834196%; see
  [Timestep rounding](performance.md#remaining-optimization-phases));
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
  identified by pack metadata, not by environment switches. See the
  [current records](performance.md#remaining-optimization-phases). At four steps, the VAE, transformer setup,
  and text take 38% of the total. They become the next targets if a later
  distill passes. These are single-prompt, single-seed observations. Details
  and checksums are in `benchmarks/m1-max-viggle-turbo-4step-1024.json`.
