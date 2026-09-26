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
