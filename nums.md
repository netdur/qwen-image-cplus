# Numbers

RTX 2060 (6 GB), Linux, seed 1301, prompt
`a travel poster with the headline "CASABLANCA" and the tagline "MEET ME AT SUNSET"`.

## 512×512

| Runtime | Model | Quant | Steps | Prompt | ~Speed (end to end) |
| --- | --- | --- | ---: | --- | ---: |
| Diffusers oracle (80c7ed2, Torch 2.14) | Viggle Qwen-Image-2.1 turbo v0.1 | GGUF Q4_K_M (Abiray), FP16 compute; text encoder BF16, VAE FP32 | 4 | `a travel poster with the headline "CASABLANCA" and the tagline "MEET ME AT SUNSET"` | ~39–45 s |
| C+ / CUDA engine (linux/) | Viggle Qwen-Image-2.1 turbo v0.1 | W4A4 H256 g64 clip v6 pack; text encoder BF16 streamed, VAE FP32 | 4 | same | ~8.8 s warm cache (~19 s with shards read from NVMe, before the later optimizations) |
| C+ / CUDA engine (linux/) | Viggle Qwen-Image-2.1 turbo v0.1 | W4A16 g64 v6 pack; text encoder BF16 streamed, VAE FP32 | 4 | same | ~23 s (shards partly from NVMe) |

Oracle phases (run 1 / run 2): load 4.1 / 4.7 s, text 16.3 / 17.6 s,
denoise 7.7 / 12.1 s, VAE 3.2–3.7 s. Both strings exact; PNGs byte-identical.
Script: `tools/benchmark_oracle_gguf.py`.

C+ engine phases (W4A4 / W4A16), shards read from NVMe: text 11.5 / 11.3 s,
transformer 5.4 / 9.8 s (load about 2.4 s, then 0.80 / 1.54 s per step), VAE +
PNG 2.0 s. W4A4 with every file in the page cache: text 4.3 s, transformer
4.5 s, VAE + PNG 1.4 s. Both strings exact.

After pipelined text-encoder reads, one activation quantization per shared
input, FP16 tensor-core attention, and the 256-deep W4A4 GEMM (W4A4, warm
cache): text 3.3 s, transformer 3.8 s (0.65 s per step), VAE + PNG 1.4 s,
8.75-8.86 s end to end. Command: `linux/dev/target/release/qwen_image_dev generate PACK models
OUT.png PROMPT 512 512 1301 4`.

## 1024×1024

| Runtime | Model | Quant | Steps | ~Speed (end to end) |
| --- | --- | --- | ---: | ---: |
| Diffusers oracle (80c7ed2, Torch 2.14) | Viggle turbo v0.1 | GGUF Q4_K_M, FP16 compute; VAE tiling (untiled FP32 decode runs out of memory) | 4 | ~74.5 s (78.5 s process wall) |
| C+ / CUDA engine (linux/) | Viggle turbo v0.1 | W4A4 H256 g64 clip v6, blocks streamed from pinned memory with overlapped uploads | 4 | ~23.5 s |

Oracle phases: load 4.8 s, text 19.9 s, denoise 32.2 s (about 8 s per step),
VAE 9.5 s. Engine phases: text 3.3 s, transformer 16.1 s (3.3-3.4 s per step),
VAE + PNG 3.7 s (untiled FP32). Both render both strings exactly.

## Image edit, 512×512

One condition image: a 1600×1598 JPEG of a cat (resized to 512×512 for both
the vision tower and the VAE), prompt `make the cat wear a red wizard hat`,
seed 1301, output 512×512.

| Runtime | Model | Quant | Steps | Prompt | ~Speed (end to end) |
| --- | --- | --- | ---: | --- | ---: |
| Diffusers oracle (80c7ed2, Torch 2.14) | Viggle turbo v0.1 | GGUF Q4_K_M, FP16 compute; text encoder and vision tower FP16 (pipeline dtype), VAE FP32 | 4 | `make the cat wear a red wizard hat` | ~76.3 s (80.2 s process wall) |
| C+ / CUDA engine (linux/) | Viggle turbo v0.1 | W4A4 H256 g64 clip v6; text encoder BF16 streamed, vision tower BF16 weights / FP32 math, VAE FP32 | 4 | same | ~12.2 s (12.3 s process wall) |

Oracle phases: load 21.1 s, text + vision 33.6 s, VAE encode 1.8 s, denoise
~9.0 s, VAE decode 3.0 s. Script: `tools/benchmark_oracle_edit.py`.

Engine phases (warm cache): decode + resize 0.1 s, vision tower 0.9-1.1 s
(including its 1.2 GB weight load), VAE encode 0.36 s, text encoder (292
tokens, 256 of them image) 4.0 s, transformer 5.5 s (load, 2070-row
conditioned prefix, then 0.71 s per step), VAE + PNG 1.1 s. Command:
`linux/dev/target/release/qwen_image_dev edit PACK models OUT.png PROMPT 512
1301 4 IMAGE`.

Both images show the cat in the same pose wearing a red, gold-starred wizard
hat; only the star pattern differs. Stage checks against the oracle's
intermediates (`tools/dump_oracle_edit.py`): pixel patches equal (6e-8; JPEGs
decode through libjpeg-turbo like Pillow, and the resize is Pillow's Lanczos
bit for bit), VAE condition latents within 0.65% (the oracle rounds the image
to FP16 first), vision tower within 2e-5 of an FP32 PyTorch run of the same
weights (the FP16 oracle is 1.3% away from both), M-RoPE positions equal, and
prompt embeddings within 0.7% (FP32) of the FP16 oracle. The native image
scores 17.6 dB PSNR against the oracle image, the same as native denoising
from the oracle's own encoder outputs (17.0 dB): the difference is W4A4 versus
Q4_K_M, not the encoders.

## After the text-encoder and attention changes (2026-09-27)

Text encoder: layer uploads overlap the previous layer's compute (two device
slots, a copy stream), attention runs as cuBLAS GEMMs, and the layer
matrices run on FP16 tensor cores with FP32 accumulation (BF16 activation
boundaries kept). It now takes ~3.0 s for 32 or 1060 tokens, its PCIe floor
(16 GB of BF16 weights at 6.45 GB/s). Its prompt rows are 2.8% from the BF16
oracle's (2.6% with FP32 GEMMs; 6.5% for pure FP32). The largest linear input
is 6410, well inside FP16. The vision tower uses the same FP16 GEMMs (0.9% /
2.3% from FP32 at 512 / 1024; the FP16 oracle is 1.3% away at 512). The
transformer's FP16 attention softmax keeps each score row in registers (one
read, one write instead of three reads and two writes).

W4A4 H256 g64 clip v6, 4 steps, seed 1301, warm cache, end to end:

| Task | Size | GGUF Q4_K_M oracle | Engine W4A4 | Engine W4A16 |
| --- | --- | ---: | ---: | ---: |
| Text to image | 512×512 | ~39–45 s | ~8.3 s | ~11.7 s |
| Text to image | 1024×1024 | ~74.5 s | ~22.6 s | ~35.6 s |
| Image edit, 1 image | 512×512 | ~76.3 s | ~11.1 s | ~18.3 s |
| Image edit, 1 image | 1024×1024 | ~101.5 s (group offload) | ~33.4 s | ~50.9 s |
| Image edit, 2 images (cat 512 area + dog 576×480) | 512×512 | ~74.9 s (group offload) | ~14.2 s | |

1024 edit, oracle: model offload runs out of memory in the transformer; with
block-level group offload (`tools/benchmark_oracle_edit.py --group-offload
--vae-tiling`): load 26.8 s, text + vision 27.1 s, VAE encode 3.4 s, denoise
29.8 s (11.4 s, then ~6.1 s per step), VAE decode 7.3 s; 105.2 s process wall.
Engine: vision 2.1 s, VAE encode 0.9 s, text (1060 tokens) 3.0 s, transformer
23.6 s (4.5 s per step over 8214 keys), VAE + PNG 3.7 s.

Two-image edit, oracle: model offload runs out of memory in the transformer;
with `--group-offload --vae-tiling`: load 24.8 s, text + vision 27.5 s, VAE
encode 1.3 s, denoise 10.4 s, VAE decode 1.9 s; 78.7 s process wall.
Engine: vision 1.2 s, VAE encode 0.6 s, text (579 tokens) 3.0 s, transformer
8.1 s (1.05 s per step), VAE + PNG 1.2 s.

Two-image edit, checked against the oracle's intermediates: token IDs, M-RoPE
positions and image-pad mask equal; joint transformer RoPE within 2.4e-7;
vision rows 0.6%, condition latents 0.5%.

### W4A4 against W4A16

Same oracle inputs (512, 4 steps) denoised with the FP16, W4A16 and W4A4
packs, then decoded; distance to the FP16 result:

| Case | W4A16 latents / PSNR | W4A4 latents / PSNR | Denoise FP16 / W4A16 / W4A4 |
| --- | --- | --- | --- |
| Text to image (poster) | 14.3% / 21.2 dB | 21.3% / 19.0 dB | 30.3 / 5.7 / 2.6 s |
| Edit (cat) | 20.5% / 20.6 dB | 28.7% / 18.6 dB | 30.6 / 6.1 / 3.0 s |

All render both poster strings exactly. W4A16 keeps the FP16 composition
closer (the W4A4 poster changes the smoke and figure); W4A4 is 1.4–1.6x faster
end to end.

## Flash attention (2026-09-27)

The transformer's attention is one fused kernel (`linux/qwen_image/cuda/flash.cu`):
mma.m16n8k8 FP16 tensor cores with FP32 accumulation, online softmax in
registers, 64-key K/V tiles in shared memory, 8 warps x 16 query rows per
block; no score matrix. Per block at the 1024-edit step shape (4096 queries x
8214 keys x 32 heads): 32.9 ms against 74 ms for cuBLAS + the stored FP16
softmax; 15.2 ms against 34 ms at the 1024 text-to-image shape. It is closer to
FP32 attention than the old path (3.8e-4 against 1.15e-3), and the FP16-weight
transformer fixture moves from 1.7e-3 to 5.0e-4 of the NumPy reference. The
score buffers are gone (about 540 MB at the 1024 edit). A 1024-edit step
drops from 4.5 s to 3.26 s.

Warm cache, 4 steps, seed 1301, end to end:

| Task | Size | GGUF Q4_K_M oracle | Engine W4A4 | Engine W4A16 |
| --- | --- | ---: | ---: | ---: |
| Text to image | 512×512 | ~39–45 s | ~8.2 s | ~11.6 s |
| Text to image | 1024×1024 | ~74.5 s | ~21.3 s | ~34.5 s |
| Image edit, 1 image | 512×512 | ~76.3 s | ~11.0 s | ~15.4 s |
| Image edit, 1 image | 1024×1024 | ~101.5 s | ~29.4 s | ~46.1 s |
| Image edit, 2 images | 512×512 | ~74.9 s | ~12.7 s | ~17.7 s |

## Pack preload and FP16 VAE convolutions (2026-09-27)

Pack preload: the INT4 pack's data section (4 GB) is read into ordinary host
memory by 8 threads while the vision tower, VAE encoder and text encoder run;
`transformer::load` then uploads it (pageable uploads run at full PCIe speed
here, 0.70 s for 4.2 GB) or registers it for streaming (0.19 s). Page-locking
4 GB up front costs 1.1 s and slowed the text encoder beside it by 1 s, so the
buffer stays unpinned until load. Outputs are byte-identical.

VAE: stride-1 convolutions run on FP16 tensor cores. cuDNN is only faster in
FP16 NHWC (implicit precomputed GEMM: 99 ms against 200 ms for FP32 Winograd
on a 288-channel 3x3 convolution at 1024x1024); activations stay FP32 NCHW and
each horizontal strip of rows (with a halo) converts through a tiled transpose.
1024 decode: 4.1 s to 2.4–2.6 s (process). Decoded image vs FP32: 64.3 dB,
max 3 levels. The encoder's condition latents move 0.7% from FP32 and land
closer to the oracle (0.25% against 0.65%; the oracle rounds its input to FP16).

Warm cache, 4 steps, seed 1301, end to end:

| Task | Size | GGUF Q4_K_M oracle | Engine W4A4 | Engine W4A16 |
| --- | --- | ---: | ---: | ---: |
| Text to image | 512×512 | ~39–45 s | ~7.2 s | ~10.9 s |
| Text to image | 1024×1024 | ~74.5 s | ~17.4 s | ~30.8 s |
| Image edit, 1 image | 512×512 | ~76.3 s | ~8.7 s | ~13.6 s |
| Image edit, 1 image | 1024×1024 | ~101.5 s | ~24.8 s | ~41.8 s |
| Image edit, 2 images | 512×512 | ~74.9 s | ~10.0 s | ~15.5 s |

### Run-to-run variation on this machine

Repeated runs occasionally differ slightly (text-encoder rows ~2e-6 relative,
then amplified by the 4-step denoiser). The cause is the host memory, not the
engine: a plain C program that reads the text-encoder shards with 8 threads
(no CUDA) sees about one flipped bit per ~40 GB read (for example 1 byte / 1
bit wrong in a 436 MB block, correct on re-read); the page cache and the files
match the disk. The text encoder reads 16 GB per run, so roughly one run in
three sees a flip. The GPU is clean (17 GB of uploads verified, 1500 SGEMM
repeats bit-identical). This RAM is not ECC; memtest86+ would confirm.

## Measured and not adopted (2026-09-27)

- W4A4 GEMM: this RTX 2060 sustains ~115 INT4 TOPS in a register-only
  mma.m8n8k32 loop; the GEMM runs at ~35 TOPS (4096 rows). Neither an exact
  full-rate int-to-float conversion in the per-group epilogue nor a swizzled
  launch order for L2 reuse changed it beyond run-to-run noise (clocks vary
  with temperature on this laptop GPU); outputs were bit-identical. Further
  gains need a new mainloop.
- INT8 text encoder (weights symmetric per output row, simulated on the
  device before the FP16 GEMMs): prompt rows 7.6% from the BF16 oracle,
  against 2.8% now. Both poster strings still render; the scene changes a
  little; the edit is nearly the same. It would halve the 16 GB upload (about
  1.4 s per run) at the cost of an extra 8 GB weight file and that quality
  gap, so it is left as an option.
