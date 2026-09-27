# Metal optimization history

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
- With the same sd.cpp binary and 1024x1024, 40-step, cache-off command, replacing
  only the diffusion checkpoint with Unsloth's 14.23 GB F16 GGUF took **984.34
  seconds process wall** (16:24.34) on AC power. Its internal timer was 983.24
  seconds: 59.08 seconds for text conditioning, 904.76 seconds for sampling,
  and 19.26 seconds for VAE decode. That is 22.02 seconds (2.2%) less process
  time than the BF16/F32 safetensors run above, and 18.14 seconds (2.0%) less
  sampling time. This is one run under different background conditions, not a
  demonstrated format speedup. The GGUF contained only the diffusion
  transformer; the same separate Qwen text-encoder shards and VAE safetensors
  were still loaded. The run reported 17.45 GB maximum RSS, 43.76 GB peak
  footprint, and zero swaps. Exact paths, hashes, command, and timings are in
  `benchmarks/m1-max-stable-diffusion-cpp-gguf-f16-1024-40.json`.
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
