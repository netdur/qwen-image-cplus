# Model behavior and validation

## Pinned reference

- Model: `Qwen/Qwen-Image-2.1`
- Snapshot: `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- Diffusers commit: `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
- Machine-readable inventory: `manifests/qwen-image-2.1.json`
- Pinned official [transformer source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/transformers/transformer_qwenimage21.py),
  [VAE source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py),
  and [pipeline source](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- Vision model reference: model-pinned Transformers 4.57.1
  [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

## Resolution and step-count policy

Qwen's own model documentation identifies **2048x2048** as the native and
recommended 1:1 output, with comparable 2K-pixel-budget dimensions for other
aspect ratios. The Diffusers implementation and Qwen's vLLM/SGLang examples
also accept 1024x1024, so 1024 is a supported and useful lower-resolution mode,
but it is not the model's native quality target. The runtime accepts independent
width and height values that are at least 256, divisible by 32, and no more
than 16,384 latent tokens in total (equivalently, at most 4,194,304 output
pixels). That includes 256x256, 512x512, 1024x1024, native 2048x2048, and
rectangular shapes within the same pixel budget. The rule lives in one place,
`pipeline::output_dimensions_supported`, which the public API also calls, and
its token cap comes from the trajectory's `max_target_tokens`. The 2048 path is functional,
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

Below that threshold a single MPS operation is measured clean, not assumed.
With the Viggle pack, the CASABLANCA poster prompt, and seed 1301, 1536x1024
(about 6,170 prefill rows in one operation) completed in **65.4 s** end to
end with a 10.0 GB peak footprint, and 1984x1024 (about 7,970 rows, the tallest
unsliced shape) in **87.8 s** with 12.5 GB. Every step was finite and both
images rendered both quoted strings exactly. Lowering the threshold to 4,096
would also slice the 1024 prefill GEMMs (about 4,130 rows) and change MPS
kernel selection on the production path, so the threshold stays at 8,192.

Every output shape now uses the flash attention kernel with an FP16 K/V
prefix cache. Previously only the 256-token and >=4096-token shapes did, and
everything else (512x512, rectangles below 4096 tokens, and every conditioned
prefix) fell back to the per-key FP32 kernel with an FP32 prefix cache. The
flash kernel already bounds query tails and padded keys and applies the same
image-block mask, so the gate only reflected which shapes had been calibrated.
`QI_DISABLE_FLASH_ATTENTION=1` restores the old selection. A/B runs with the
Viggle pack and seed 1301 measured:

| Shape | Per-key FP32 | Flash FP16 K/V | Image PSNR (A vs B) |
| --- | ---: | ---: | ---: |
| Two-image conditioned 512 (4 steps) | 62.4 s, step 1 24.4 s, cache 2.16 GB | **34.5 s**, step 1 7.3 s, cache 1.08 GB | 48.6 dB |
| Text-only 512x512 (4 steps) | 18.9 s | **14.2 s** | 38.3 dB |
| 1344x768 poster (4 steps) | 173.0 s | **61.8 s** | 38.5 dB |

Both 1344x768 images render CASABLANCA and MEET ME AT SUNSET exactly. The
1024x1024 path already used flash and is unchanged: the Viggle poster PNG is
byte-identical to the previous build (SHA-256 `5f2501d2…`).

The authoritative Qwen and Diffusers sampling default is **40 Euler steps**.
The 25-step experiment in this repository came from the
[Comfy-Org workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json),
not an official Qwen recommendation. That template explicitly says the
official pipeline uses about 40-50 steps and that the template itself starts
at 25. Consequently, 40 remains this runtime's compatibility, oracle, and
default path for the base pack; 25 is only a measured opt-in speed/quality
tradeoff. It is not text-safe at 1024: on the poster prompt, 25 steps chose
to render CASABLANCA and misspelled it (see the
[Viggle comparison](1024-validation.md)).
Pack metadata can set another default. The adopted Viggle distillation pack
defaults to 4 steps with an unstretched schedule.

## Multi-image conditioning status

The official Qwen-Image-2.1 pipeline accepts **one to ten reference images**.
Native image-conditioned generation is available through the CLI and the
public C+/C generation APIs (see Package and API layout). The original entry
points produce square 512x512 or 1024x1024 output. The native sized entry
point also accepts rectangular dimensions from 256 through 2048 in multiples
of 32, within a 1024x1024 pixel budget. References are aspect-preserving
resized to approximately the output area. Integration tests use the 512
budget so correctness work does not spend 1024 or 2048 generation time.
A 768x512, three-step rectangular image-conditioned smoke completed through
vision encoding, the conditioned transformer, sized VAE decode, and PNG output
in 26.895 seconds on the M1 Max. This is a functional check, not a quality or
like-for-like speed comparison. The 256x256 lower bound also completed through
the same path in 8.710 seconds.

The first completed checkpoint reproduces the pinned Diffusers/Qwen3-VL input
contract: aspect-preserving area resize rounded to multiples of 32, one vision
token per merged 2x2 group of 16px patches, one VAE condition latent per 16x16
tile, the exact `<imageN>`/vision placeholder prompt layout, image-aware
three-axis MRoPE positions, image-embedding substitution, and DeepStack feature
injection after language layers 0, 1, and 2. For two square 512px references,
that is 256 vision tokens and 1,024 VAE latent tokens per image. Together they
produce 512 language-vision tokens and a 2,048-token VAE condition prefix; the
512px output itself is another 1,024 latent tokens.

The shape and MRoPE unit tests pass, and the existing text-only encoder oracle
remains unchanged at 0.0361069 output nRMSE. Text-only prompts still embed a
literal `<|image_pad|>` as an ordinary token, as the reference encoder does
without pixel values. The first multimodal encoder briefly rejected such a
prompt with "image placeholder has no vision feature"; a 512x512 Viggle run of
`a flat app icon labelled <|image_pad|>` now completes again. `test-multi-image-prompt` compares
every token of a three-image template against IDs from the pinned HF tokenizer
(`tests/fixtures/tokenizer/multi_image_three.i32`). That exact comparison caught
a label-spacing bug the earlier count-only smoke missed: the reference
tokenizes `" <image2>"` as one pre-token run (`" <"`, `image`, `2`, `>`), while
the native builder had encoded a separate `" "` and `"<image2>"`, adding one
token and shifting every later position for each image after the first. Native ImageIO/Core Graphics input now
decodes and resizes without Python or a GUI framework. It preserves straight
RGBA FP32 for the VAE while separately compositing RGB over white for Qwen3-VL,
matching the official split. Its asymmetric 64x32 smoke verifies alpha and
channel handling, top-down row orientation, and the official 512-area result
of 736x352 (253 vision tokens and 1,012 VAE latent tokens). Input also honours
the EXIF/TIFF orientation tag the way Diffusers' `load_image`
(`ImageOps.exif_transpose`) does. Without it a phone photo stored sideways
was encoded rotated, with width and height swapped. The orientation is applied
as a Core Graphics transform in the same single resample.
`test-image-orientation tests/fixtures/image_orientation` decodes one picture
stored under all eight orientations
(`tools/generate_orientation_fixtures.py`) and requires every decode to match
orientation 1. Each one does, with a mean absolute error of at most 2.2e-7.

The native VAE condition encoder is also structurally complete. It implements
the five down blocks, four 2x spatial reductions, middle attention, posterior
mode selection, and per-channel latent normalization. A 32x32 smoke produces
four finite nonzero latent tokens; the requested 512-area smoke encodes the
736x352 reference into 1,012 finite nonzero 64-channel tokens. It is now also
numerically validated (see "Image-conditioned validation against the pinned
Python pipeline" below): 0.0055 normalized-latent nRMSE against an FP32 CPU
reference on a 1024x1024 image.

The native Qwen3-VL vision tower is complete and independently measured. It
implements the pinned BF16 patch projection, exact 48x48 learned-position
interpolation, axial vision RoPE, 27 bidirectional transformer blocks, 2x2
final patch merger, and DeepStack mergers after blocks 8, 16, and 24. A
dependency-light CPU PyTorch oracle reproduces the pinned Transformers 4.57.1
equations without importing Transformers. On its deterministic four-patch
32x32 fixture, final output nRMSE is **0.0282456**; the three DeepStack output
nRMSE values are **0.00854384**, **0.013003**, and **0.0242081**. All recorded
boundaries remain below the enforced 0.06 ceiling. The real requested 512-area
case resizes the asymmetric input to 736x352, processes 1,012 patches into 253
merged tokens, and completes the tower in **2.653 s** wall time on the M1 Max.
This phase streams one contiguous 1.153 GB vision-weight region and keeps every
operation on Metal after native input preparation.

Vision attention runs as matrix work. For each of the 16 heads, QK^T (scaled by
1/sqrt(72)) is an MPS FP32 multiplication over a strided view of the
`[patches, heads, 72]` buffers. A row softmax and the PV product follow, written
straight into that head's output columns, with the output rounded to BF16 as
before. The original kernel walked the keys one at a time with four
threadgroup barriers per key: on a 1024x1024 reference (4,096 patches) it
spent **27.5 of the tower's 31.2 s** in attention, about 76 GFLOP/s. The
matrix path takes **1.41 s** (tower 5.9 s), and 155 ms instead of 1,504 ms at
512. `QI_VALIDATE_VISION_ATTENTION=1` also runs the old kernel on layer 0 and
reports the difference: nRMSE 5.8e-5 at 1,024 patches and 1.0e-4 at 4,096.
`QI_DISABLE_VISION_MPS_ATTENTION=1` restores the old kernel.

Multi-image conditioning runs both reference encoders as one batch each
(`vision_encoder::encode_batch_to_host`, `vae::encode_batch_from_host`). The
weight read, Metal library compile, pipelines, and the VAE's MPSGraph
convolution cache are set up once, and only activations are per image. Vision
output and DeepStack features are written directly at each image's offset in
the layer-major arrays, so there is no repack. Before this, every reference
reloaded both encoders. On two references the generated PNG is byte-identical
(SHA-256 `b1b7a08f…`). With six 512-area references, alternating runs against
the previous build measured the VAE phase at **2.57 → 1.09 s** and the vision
phase at 17.1 → 15.1-16.7 s (about 12% background GPU load; vision time is
dominated by per-image GPU work, not setup).

The transformer condition-prefix assembly now works across the same one-to-ten
image contract. Qwen-Image 2.1 does not simply prepend all condition latents:
each Qwen3-VL image placeholder expands fourfold, the corresponding VAE rows
replace those positions, and the target block is appended last. Text is causal,
each image block is internally bidirectional, and shape lengths keep adjacent
images as distinct blocks. Text and condition-image rows use the zero-timestep
modulation and form the reusable K/V prefix; only the target rows use the
sampled timestep and Euler update. Unit tests pin this ordering, block identity,
target mask, and centred three-axis RoPE coordinates.

The first full two-image conditioning assembly now passes at the 512-area test
budget. Two copies of the asymmetric reference each resize to 736x352 and
produce, in official image order, 506 merged vision rows, a 540-row multimodal
prompt after the 14-row system prefix is removed, and 2,024 normalized VAE
condition rows. The measured wall time is **14.645 s**: 5.634 s for both vision
towers, 8.102 s for the joint Qwen3-VL language pass, and 0.880 s for both VAE
encodes (plus native input/tokenizer overhead). The test checks the image-pad
mask count and finite nonzero prompt/latent outputs while exercising DeepStack
layer-major repacking and condition-latent order. It also establishes why the
denoiser cannot retain its old 512-row text-only ceiling: this valid two-image
prompt has 540 rows before any VAE condition latents are prepended.

The next integration checkpoint connects that assembly to the real FP16
transformer. With the same asymmetric reference supplied twice, a current run
produced 506 vision rows, a 542-row prompt, 2,024 condition-latent rows, and an
expanded 2,060-row cacheable prefix. A one-step 512x512 target run completed
with finite output in **38.435 s end to end**: **14.122 s** for native
two-image conditioning and **24.313 s** for transformer loading, setup, and the
first denoising step. The first transformer step itself took **21.298 s wall**.
The two-step variant also passes, proving the enlarged prefix cache is consumed
correctly: its first step took **19.107 s**, while the target-only cached-prefix
second step took **6.529 s**; total end-to-end wall time was **43.265 s**. These
are correctness smokes, not optimized benchmarks.

The complete native two-image path now also runs all 40 official Euler steps,
decodes the final latent with the native VAE, and writes a 512x512 PNG. The
first full run took **323.631 s end to end** on the M1 Max: **14.519 s** for
input preparation and both image/text encoders, **308.302 s** for transformer
setup and the trajectory, **0.799 s** for VAE decode, and **0.008 s** for PNG
output. The first denoising step took **20.287 s**; cached-prefix steps then
ranged from about **6.46 to 8.22 s**. With the same red-circle reference used
twice, the output visibly contained two differentiated red circular forms, so
this checks semantic conditioning rather than only finite tensors. This is a
correctness baseline, not an optimized benchmark. The low-level layout accepts
the official one-to-ten image range, and the native orchestration/CLI now do as
well. One-image and three-image one-step boundary smokes complete in **21.153
s** and **57.904 s** end to end respectively. Their prefix caches are **1.087
GB** and **3.234 GB**, showing the approximately linear memory cost per 512-area
reference. Those caches were FP32. With flash attention on every shape the prefix cache
is FP16 and exactly half the size (two images: 2.157 → 1.078 GB), so one and
three images now need about 0.54 and 1.62 GB, and ten about **5.4 GB** instead
of the earlier 10.7 GB estimate. Ten images are accepted by the model contract
and implementation but have not been run on this 32 GB M1 Max. The cache plus
the **14.23 GB** resident transformer and working memory should now fit, but
the practical image-count ceiling still depends on unified memory. The public
C+/C exposure is described in [the API architecture](architecture.md).

### 1024x1024 text and one-image timings

Measured on 2026-09-24 on the M1 Max, seed 1301, with the CASABLANCA poster
prompt for text. The image runs use one 1024x1024 reference (the Viggle poster)
and the prompt `Change the headline to "MARRAKECH" and keep everything else
the same.` The conditioned trajectory has no Cache-DiT, so the base image run
uses the official 40 uncached steps.

| Pack and mode | Conditioning | Trajectory | End to end | Peak footprint |
| --- | ---: | ---: | ---: | ---: |
| Viggle 4-step, text | 2.2 s text | 34.6 s | **39.0 s** | 7.0 GB |
| Viggle 4-step, 1 image | 47.5 s (vision 30.3, text 16.4, VAE 0.8) | 60.4 s | 110.4 s | 7.2 GB |
| Viggle 4-step, 1 image, MPS vision attention | 23.1 s (vision 6.2, text 16.1, VAE 0.8) | 61.1 s | 86.6 s | 7.2 GB |
| Viggle 4-step, 1 image, + MPS text linears and attention | 10.6 s (vision 5.5, text 4.3, VAE 0.8) | 59.7 s | **72.5 s** | 7.2 GB |
| + MPS vision linears, FP16-operand text linears | **6.4 s** (vision 2.6, text 2.9, VAE 0.8) | see note | about 68 s | 7.2 GB |
| Base 40 steps Cache-DiT 0.16, text | 2.2 s text | 119.4 s | **123.8 s** | 16.0 GB |
| Base 40 steps uncached, text | 2.2 s text | 435.2 s | **440.6 s** | 7.0 GB |
| Base 40 steps uncached, 1 image | 59.7 s (vision 36.5, text 22.3, VAE 0.8) | 683.8 s | **746.4 s** | 7.2 GB |

One 1024 reference adds 4,096 condition-latent rows and 1,024 vision tokens
(1,055-row prompt). The prefix K/V cache is 2.16 GB, and cached steps rise from
about 8.4 s to 11.4 s (Viggle) or about 16 s (base). The vision tower's 4,096
patches cost about 30 s, almost all of it in a per-key attention kernel. The
MPS attention path above reduces that to about 6 s; the base image run was
measured before it and would drop by about the same 25 s. The text encoder
had the same problem at 1,069 rows. Its scalar 16x16 BF16 linear kernel ran at
about 1.1 TFLOP/s (about 13.5 of 16 s), and its causal attention walked keys
one at a time (2.8 s). Prompts of 256 rows or more now run each linear as
three steps in one command buffer: BF16 weights expanded exactly to FP32
scratch, an MPS FP32 multiplication, and BF16 rounding of the output.
Attention runs per head as MPS QK^T, a causal row softmax, and PV, with query
head h reading KV head h/4. The encoder drops from **16.1 s to 4.3 s**
(attention 2.77 s to 0.23 s). Forced onto the 22-row oracle
(`QI_FORCE_TEXT_MPS_LINEAR=1`), the new path lands closer to PyTorch: output
nRMSE 0.0324 against 0.0361. Shorter prompts keep the original kernels, where
the weight reads dominate; the 1024 Viggle text-only PNG is byte-identical.
`QI_DISABLE_TEXT_MPS_LINEAR` and `QI_DISABLE_TEXT_MPS_ATTENTION` restore the
old kernels.

Two further encoder changes bring conditioning from 10.6 s to **6.4 s**:

- **Vision linears** at 256+ patch rows use the same three-step MPS linear,
  with a BF16 bias-and-round epilogue matching the scalar kernel's. On a 1024
  reference they drop from 2.93 s to 0.48 s (tower 5.5 to 2.4 s). The vision
  features are bit-identical in accuracy: 0.0934 nRMSE against FP32 before and
  after. `QI_DISABLE_VISION_MPS_LINEAR` restores the scalar kernel.
- **Text-encoder linears** take FP16 operands with FP32 accumulation. BF16
  weights and BF16-rounded activations convert to FP16 exactly within its
  range, and non-finite output would fail the encoder's finite check. Encoder
  GPU time drops from 6.4 s to 2.5 s. Against the reference language model
  (same vision in), image rows improve from 0.089 to 0.074 and text rows move
  from 0.031 to 0.036; the forced 22-row oracle moves from 0.0324 to 0.0379
  (the original kernel's figure was 0.0361). `QI_DISABLE_TEXT_FP16_LINEAR`
  keeps the FP32 multiplication.

The end-to-end run for this row had background GPU load (a running GUI build
and a Chrome renderer held the GPU at 13-16%), which slowed the unchanged
denoising steps (step 1 32.5 s against 26.6 s). The total shown is 72.5 s minus
the measured 4.2 s conditioning saving, not a clean measurement.

With the encoders fixed, denoising is compute-bound near the M1 Max limit.
Step 1 processes about 9.2k rows: about 131 TFLOP of matrix multiplication
(7.1 B parameters) plus about 45 TFLOP of attention, which at 26.3 s is about
6.7 TFLOP/s against roughly 10.4 TFLOP/s peak. Each cached step (about 78
TFLOP) runs at the same rate. The flash kernel is slower than the MPSGraph and
steel attention here (step 1 30.3 s, cached steps 15.1 s). Further image-mode
savings therefore need less work, not faster kernels: reusing the text and
reference prefix across seeds, or fewer reference tokens. With both encoders fixed, one 1024 reference adds about 33 s to
the distilled run (39.0 to 72.5 s), almost all of it in denoising: step 1
covers about 9.2k rows (26 s against 9 s), and each cached step attends to
4,096 extra keys (about 11.4 s against 8.4 s). The Viggle image run performed
the edit (MARRAKECH, scene and tagline kept). The base run reproduced the
reference but kept CASABLANCA, over-saturated; that quality difference is not
yet explained.

### Image-conditioned validation against the pinned Python pipeline

On 2026-09-24 the base pack ignored an edit instruction that the distilled pack
followed ("Change the headline to "MARRAKECH""; one 1024x1024 reference). To
tell a native bug from model behaviour, each stage was compared against the
pinned `QwenImage21Pipeline` (diffusers `80c7ed2`, transformers 5.17, torch
2.9) with Viggle's distilled transformer (`bafc91e`, BF16), the same
reference, prompt and seed. `QI_DUMP_DIR=dir` makes the native run write the
tensors used here (vision and DeepStack features, prompt embeddings and
image mask, condition latents, initial and final latents, and per-stage VAE
encoder activations).

| Stage | Native vs reference | Reference's own BF16 error |
| --- | ---: | ---: |
| Vision pixel values (resize, normalize, patch order) | 6.7e-8 | — |
| Vision tower output, vs FP32 | 0.093 | 0.059 |
| Language model, same vision features in, image / text rows | 0.089 / 0.031 | — |
| VAE condition encode, vs FP32 on CPU | 0.0055 | — |
| Transformer + scheduler, 4 steps, identical inputs (final latent) | 0.038 (image PSNR 30.3 dB) | — |

The whole native conditioned path therefore matches the reference within BF16
precision. The reference produces the same image, saturation included, so
the base pack's result is model behaviour, not a native bug.

Two reference-side findings came out of this:

- PyTorch's MPS backend returns zeros from `QwenImage21AvgDown3D` (the
  encoder's view/permute/mean shortcut) once the input reaches 512x512
  (96->192 channels), while CPU is correct. The official pipeline run on a
  Mac therefore encodes condition images wrongly: 0.59 latent nRMSE against
  CPU FP32. The VAE encoder comparison above uses CPU for that reason.
- The pipeline draws its initial noise with `torch.randn(..., dtype=bfloat16)`.
  Native reproduces PyTorch's FP32 `randn` and rounds to BF16, which matched
  the earlier trajectory fixtures but gives different noise for the same seed
  than the current pipeline. The comparison above passes native's noise to
  the pipeline through `latents=`.
