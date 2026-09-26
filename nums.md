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
