# Performance and optimization record

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
    400.10. This matches `qwen_image_metal/src/scheduler.cplus` and its unit test, and
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
    `qwen_image_metal/kernels/mlx_steel_attention.metal`) uses only `simdgroup_matrix`
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

## Viggle v0.2.1 LoRA (six-step alternative)

The newer Viggle v0.2.1 adapter is supported **alongside**, not in place of,
the original four-step full-fine-tune and the 40-step base pack. The selected
`qwen-image-2.1-viggle-v0.2.1-lora-fp16-v4.qipack` contains the unchanged base
FP16 transformer and declares `schedule=viggle-v0.2.1-6`,
`lora=viggle-v0.2.1-r256`, `steps=6`, and `shift_terminal=none`. It requires
`Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors` in the **same
folder**. The GUI checks for this sidecar and switches its Steps picker to 6
when this pack is chosen. The processor, text-encoder shards, and VAE must also
remain in their usual subfolders beside the pack.

The adapter's 227 A/B pairs are validated for BF16 dtype and exact rank-256
shapes. They are copied once to Metal and converted there to FP16; each
affected transformer/timestep projection applies `base(x) + B(A(x))` at
inference rather than expanding or merging the adapter into base weights.
The six raw sigma nodes are `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]`, shifted
for the target token count, followed by zero. For 832×1248, the shifted nodes
match a pinned Diffusers calculation. No CFG or Cache-DiT was used in the
comparison below.

The Eiffel edit used the same pre-resized 832×1248 reference,
prompt `change weather to storm`, and seed 42 across packs. On this M1 Max,
the new native six-step result took **129.453 s** from input preparation
through PNG completion (7.264 s conditioning, 119.644 s transformer, 2.366 s
VAE, 0.125 s PNG). Its tower silhouette, platforms, and lattice remained
coherent, unlike the older four-step full-fine-tune's deformation. The old
four-step run took 78.415 s but failed this quality case; the 40-step base
run took 745.104 s. Thus v0.2.1 is **5.76× faster than base** here, but
**1.65× slower than the malformed four-step run**. Viggle's hosted demo showed
lightning/rain and reported 4.36 s on a remote GPU; its latency is not
comparable to this laptop, and the native output is not pixel-identical.
An additional 1024×1024 six-step text-to-image smoke exercised the streamed
weights and graph/steel attention path and produced a coherent red balloon
in **75.974 s** end to end; this is not a matched base-model benchmark.

**Conditioned-prefill MPS safety (2026-09-25).** A 736×1280 portrait edit
with one reference produced 7,379 joint prefill rows. MPS's one-shot FP16
QKV multiply returned a non-finite row even though direct-FP16 and
FP32-staged paths supplied byte-identical, finite FP16 input. Tiling only QKV
did not suffice: the MLP gate multiply also became non-finite. The runtime now
uses at most 4,096 rows per base-model MPS multiplication for conditioned
prefills without graph attention. It retains direct FP16 activations and does
not add a command-buffer split; text-only and graph-attention paths keep their
previous matrix selection. The corrected portrait completed in **111.04 s**
end to end and produced a PNG byte-identical to the 118.31 s FP32-staged
fallback. A 512×512 text-only regression and the 832×1248 Eiffel edit were
also byte-identical to their pre-change PNGs. These are correctness checks,
not a controlled speed comparison.

**Two-reference cost and masked-tile pruning (2026-09-25, historical six-step path).**
Matched one/two/one-reference portrait edits at 736×1280 used Viggle v0.2.1,
seed 1301, and the then-required FP32-staged activation fallback. The joint
prefill grew from 7,379 to 11,025 rows. Input conditioning took
6.24 / 10.98 / 6.06 s, the first transformer step took
28.19 / 49.82 / 34.39 s, and end-to-end time was
116.69 / 159.93 / 134.07 s. The one-reference rerun itself varied by
17.38 s, so these runs do not establish a precise per-reference cost.

On the same two-reference setup, adjacent two-step comparisons of masked
flash-attention tile pruning reduced first-step GPU time from
43.53 to 37.94 s and, in a slower period, from 63.14 to 52.24 s. The second
step was essentially unchanged. A full six-step pruned PNG was byte-identical
to the unpruned output. Those full runs were not adjacent, so their wall times
are not a controlled end-to-end speed comparison. The direct-FP16 activation
path was fixed later and is now the default.

**Six-step edit speed pass (2026-09-26).** Four changes target the
conditioned edit shapes, which sit below the 4,096-row graph-attention gate
(736×1280 is 3,680 target rows, 832×1248 is 4,056):

- **LoRA merged once at load.** The 224 per-block A/B pairs are folded into
  the resident FP16 pack as `half(W + B·A)` (0.63 s GPU), so denoising runs
  only the base GEMMs. The GPU forms each merged tensor in scratch and the
  host copies it into the pack. Writing the merged weights straight into the
  14 GB pack buffer from the GPU made later block GEMMs return NaN tiles
  intermittently at 832×1248: 9 of 11 two-step runs, even when the kernel
  wrote back unchanged values. With host copies it passed 4 of 4, and every
  later run passed too. The three global pairs (timestep embedder,
  modulation) still run unmerged once per trajectory. Ring-streamed shapes
  (4,096+ target rows) keep the per-step path. `QI_DISABLE_LORA_MERGE=1`
  restores it everywhere.
- **Steel attention on cached steps at any target size.** The MLX kernel is
  specialized per (target rows, total rows), so it no longer requires exactly
  4,096 rows. The prefill keeps flash attention. `QI_DISABLE_STEEL_ATTENTION=1`
  restores flash.
- **MPS attention-output projection on flash shapes.** It replaces the custom
  simdgroup GEMM (about 4 TFLOP/s against about 9 for MPS).
  `QI_DISABLE_MPS_ATTENTION_OUTPUT_FLASH=1` restores the custom kernel.
- **Setup overlap.** The adapter is read with the pack's 8-way `F_NOCACHE`
  `pread` (0.27 s for 1.36 GB) instead of faulting in its mapping (about
  1.3 s saved). For resident-pack shapes, the 14.2 GB pack read starts on a
  background thread when image conditioning begins. `QI_DISABLE_PACK_PRELOAD=1`
  restores the sequential read. On this machine, with 5.7 GB of swap in use,
  the preload competes with the encoders' reads. It moved about 3 s out of
  transformer setup but slowed conditioning by 0.5-3 s. The preload checks
  cancellation between 8 MiB read chunks; a failed read releases its buffer
  before the normal loading path retries.

A failed batched step now reports the Metal command-buffer error instead of
only a later non-finite latent.

Adjacent full runs on the M1 Max. GPU load from other apps varied, so only
pairs run back to back are comparable:

| Case | Before | After | PSNR vs before |
| --- | ---: | ---: | ---: |
| Eiffel edit 832×1248 | 163.7 s (cached steps 20.2-25.1 s) | **102.3 s** (11.4-12.1 s) | 58.0 dB |
| Portrait edit 736×1280 | 171.6 s (22.1-24.3 s) | **130.5 s** (14.8-16.2 s) | 58.2 dB |
| 512×512 text | 31.2 s | 30.2 s | 63.5 dB |
| 1024 Viggle 4-step text (graph path) | 55.9 s | 57.7 s | byte-identical |

In an interleaved two-step benchmark of the portrait edit, the first cached
step fell from 16.2-17.9 s to 11.2-12.0 s. MPS attention-output saved about
1.5 s of the 7,379-row prefill. The two-reference portrait now runs on the
default direct-FP16 path: 137.3 s end to end, against 153-168 s with
`QI_DISABLE_DIRECT_FP16_ACTIVATIONS=1` earlier. Its image matches that run
at 48.9 dB, and the hat and pose are unchanged. `cpc test` passes (361).

**Caches on image-conditioned runs (experimental, 2026-09-26).** Conditioned
trajectories stay uncached by default. `QI_CONDITIONED_CACHE` selects
`cache-dit-0.12|0.14|0.16|0.24` or `taylorseer-0.12|0.24` for testing. The
existing Cache-DiT and TaylorSeer code already offsets by the prefix rows,
and the conditioned layout puts the target block last, so it applies
unchanged. Base pack, 512×512 edit ("make her wear a hat", seed 1301,
512×512 input), 40 steps, PSNR against the uncached image:

| Mode | End to end | Cached steps | PSNR |
| --- | ---: | ---: | ---: |
| none | 87.4 s | 0 | — |
| Cache-DiT 0.12 | 37.1 s | 26 | 43.9 dB |
| Cache-DiT 0.14 | 34.9 s | 27 | 40.1 dB |
| Cache-DiT 0.16 / 0.24 | 36.5 / 39.4 s | 27 | 39.6 dB (identical images) |
| TaylorSeer 0.12 | 41.0 s | 26 | 47.3 dB |
| TaylorSeer 0.24 | 36.0 s | 29 | 37.9 dB |

All six keep the hat, face, tattoo and pose visually unchanged. Cache-DiT
0.16 and 0.24 cache the same 27 steps because the continuous-cache cap binds
first. This is one prompt at one size, not a calibration.
