// W4A4 GEMM for the w4a4-h256-g64-v6 pack format on sm_75 INT4 tensor cores.
//
// Weights: QIPACK scheme 6, signed codes -7..7 as two's-complement nibbles
// [N, K/2] (even k in the low nibble), FP16 scale per (row, 64 inputs), stored
// already rotated by the H256 butterfly along K.
// Activations: each FP16 row is rotated with the same butterfly, then
// quantized per 64 inputs to signed codes -7..7 with an FP16 scale
// (quantize_activations below; same rule as tools/evaluate_int4_quantization.py).
//
// y[m,n] = sum_g sa[m,g] * sw[n,g] * sum_{k in g} qa[m,k] * qw[n,k]
// The inner sums run on mma.sync.m8n8k32.s4 (two per 64-wide group) in int32;
// each group is then scaled into FP32 accumulators.
//
//   nvcc -O3 -arch=sm_75 -o bench_w4a4 bench_w4a4.cu

#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

constexpr int GROUP = 64;

// MARK: - Activation rotation and quantization

// One block per row. The row is rotated in shared memory with the radix-4
// H256 butterfly (strides 1, 4, 16, 64 inside every 256-chunk), then each
// 64-group gets scale = fp16(max|x| / 7) and codes rint(x / scale) in -7..7.
__global__ void quantize_activations(const half *__restrict__ x, uint8_t *__restrict__ codes,
                                     half *__restrict__ scales, int k) {
    extern __shared__ float row[];
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < k; i += blockDim.x) row[i] = __half2float(x[(size_t)m * k + i]);
    __syncthreads();
    for (int stride = 1; stride < 256; stride *= 4) {
        const int span = stride * 4;
        for (int b = threadIdx.x; b < k / 4; b += blockDim.x) {
            const int chunk_base = (b / stride) * span;
            const int offset = b % stride;
            const int i0 = chunk_base + offset;
            float a = row[i0], bb = row[i0 + stride], c = row[i0 + 2 * stride], d = row[i0 + 3 * stride];
            row[i0] = (a + bb + c - d) * 0.5f;
            row[i0 + stride] = (a + bb - c + d) * 0.5f;
            row[i0 + 2 * stride] = (a - bb + c + d) * 0.5f;
            row[i0 + 3 * stride] = (-a + bb + c + d) * 0.5f;
        }
        __syncthreads();
    }
    // One warp per 64-group: two values per lane.
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (int g = warp; g < k / GROUP; g += warps) {
        float v0 = row[g * GROUP + 2 * lane], v1 = row[g * GROUP + 2 * lane + 1];
        float peak = fmaxf(fabsf(v0), fabsf(v1));
        for (int s = 16; s > 0; s /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, s));
        half scale_half = __float2half(fmaxf(peak / 7.0f, 5.9604645e-08f));
        float scale = __half2float(scale_half);
        int c0 = (int)fminf(fmaxf(rintf(v0 / scale), -7.0f), 7.0f);
        int c1 = (int)fminf(fmaxf(rintf(v1 / scale), -7.0f), 7.0f);
        codes[(size_t)m * (k / 2) + g * (GROUP / 2) + lane] = (uint8_t)((c0 & 0xF) | ((c1 & 0xF) << 4));
        if (lane == 0) scales[(size_t)m * (k / GROUP) + g] = scale_half;
    }
}

// MARK: - GEMM

constexpr int BM = 128, BN = 128;
constexpr int THREADS = 256;           // 8 warps: 4 (M) x 2 (N), each 32 x 64
constexpr int ROW_BYTES = GROUP / 2;   // 32 bytes of codes per row per group
constexpr int SMEM_STRIDE = 48;        // bytes; spreads 8 fragment rows over all banks

__device__ __forceinline__ void mma_s4(int &c0, int &c1, unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}

__global__ void __launch_bounds__(THREADS)
w4a4_gemm(const uint8_t *__restrict__ a_codes, const half *__restrict__ a_scales,
          const uint8_t *__restrict__ w_codes, const half *__restrict__ w_scales,
          float *__restrict__ y, int m, int n, int k) {
    __shared__ __align__(16) uint8_t a_tile[BM * SMEM_STRIDE];
    __shared__ __align__(16) uint8_t b_tile[BN * SMEM_STRIDE];
    __shared__ float a_scale_tile[BM];
    __shared__ float b_scale_tile[BN];

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int warp_m = warp / 2, warp_n = warp % 2;
    const int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    const int groups = k / GROUP;

    float accumulators[4][8][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) accumulators[i][j][0] = accumulators[i][j][1] = 0.0f;

    for (int g = 0; g < groups; ++g) {
        // 128 rows x 32 bytes for each operand: one 16-byte chunk per thread each.
        {
            int row = tid / 2, part = tid % 2;
            *reinterpret_cast<uint4 *>(&a_tile[row * SMEM_STRIDE + part * 16]) =
                *reinterpret_cast<const uint4 *>(&a_codes[(size_t)(block_m + row) * (k / 2) + g * ROW_BYTES + part * 16]);
            *reinterpret_cast<uint4 *>(&b_tile[row * SMEM_STRIDE + part * 16]) =
                *reinterpret_cast<const uint4 *>(&w_codes[(size_t)(block_n + row) * (k / 2) + g * ROW_BYTES + part * 16]);
        }
        if (tid < BM) a_scale_tile[tid] = __half2float(a_scales[(size_t)(block_m + tid) * groups + g]);
        else b_scale_tile[tid - BM] = __half2float(w_scales[(size_t)(block_n + tid - BM) * groups + g]);
        __syncthreads();

        int partial[4][8][2];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j) partial[i][j][0] = partial[i][j][1] = 0;
        #pragma unroll
        for (int half_k = 0; half_k < 2; ++half_k) {
            // Fragment: lane holds 8 consecutive k (one 32-bit word) of row lane/4.
            const int byte = half_k * 16 + (lane % 4) * 4;
            unsigned a_fragment[4], b_fragment[8];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                a_fragment[i] = *reinterpret_cast<const unsigned *>(&a_tile[(warp_m * 32 + i * 8 + lane / 4) * SMEM_STRIDE + byte]);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                b_fragment[j] = *reinterpret_cast<const unsigned *>(&b_tile[(warp_n * 64 + j * 8 + lane / 4) * SMEM_STRIDE + byte]);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) mma_s4(partial[i][j][0], partial[i][j][1], a_fragment[i], b_fragment[j]);
        }
        // Accumulator lane layout: row lane/4, columns 2*(lane%4) and +1.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float sa = a_scale_tile[warp_m * 32 + i * 8 + lane / 4];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int col = warp_n * 64 + j * 8 + 2 * (lane % 4);
                accumulators[i][j][0] += (float)partial[i][j][0] * (sa * b_scale_tile[col]);
                accumulators[i][j][1] += (float)partial[i][j][1] * (sa * b_scale_tile[col + 1]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) {
            int row = block_m + warp_m * 32 + i * 8 + lane / 4;
            int col = block_n + warp_n * 64 + j * 8 + 2 * (lane % 4);
            *reinterpret_cast<float2 *>(&y[(size_t)row * n + col]) = make_float2(accumulators[i][j][0], accumulators[i][j][1]);
        }
}


// MARK: - GEMM v2: 256-deep stages, register prefetch

constexpr int STAGE_K = 256;                 // four 64-groups per stage
constexpr int STAGE_BYTES = STAGE_K / 2;     // 128 code bytes per row per stage
constexpr int V2_STRIDE = STAGE_BYTES + 16;  // 144 bytes: fragment rows cover all banks

__global__ void __launch_bounds__(THREADS)
w4a4_gemm_v2(const uint8_t *__restrict__ a_codes, const half *__restrict__ a_scales,
             const uint8_t *__restrict__ w_codes, const half *__restrict__ w_scales,
             float *__restrict__ y, int m, int n, int k) {
    __shared__ __align__(16) uint8_t a_tile[BM * V2_STRIDE];
    __shared__ __align__(16) uint8_t b_tile[BN * V2_STRIDE];
    __shared__ float a_scale_tile[4][BM];
    __shared__ float b_scale_tile[4][BN];

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int warp_m = warp / 2, warp_n = warp % 2;
    const int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    const int groups = k / GROUP;
    const int stages = k / STAGE_K;

    // Each thread moves 4 x 16 bytes of A and of B per stage: rows tid / 8 and
    // tid / 8 + 32, + 64, + 96; 16-byte column (tid % 8).
    const int load_row = tid / 8, load_column = (tid % 8) * 16;
    uint4 a_next[4], b_next[4];
    float a_scale_next[2], b_scale_next[2];
    auto fetch = [&](int stage) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = load_row + r * 32;
            a_next[r] = *reinterpret_cast<const uint4 *>(&a_codes[(size_t)(block_m + row) * (k / 2) + stage * STAGE_BYTES + load_column]);
            b_next[r] = *reinterpret_cast<const uint4 *>(&w_codes[(size_t)(block_n + row) * (k / 2) + stage * STAGE_BYTES + load_column]);
        }
        // 4 groups x 128 rows of scales for each operand: two per thread.
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int index = tid + s * THREADS;  // 0..511
            const int group = index / BM, row = index % BM;
            a_scale_next[s] = __half2float(a_scales[(size_t)(block_m + row) * groups + stage * 4 + group]);
            b_scale_next[s] = __half2float(w_scales[(size_t)(block_n + row) * groups + stage * 4 + group]);
        }
    };
    auto store = [&]() {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = load_row + r * 32;
            *reinterpret_cast<uint4 *>(&a_tile[row * V2_STRIDE + load_column]) = a_next[r];
            *reinterpret_cast<uint4 *>(&b_tile[row * V2_STRIDE + load_column]) = b_next[r];
        }
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int index = tid + s * THREADS;
            a_scale_tile[index / BM][index % BM] = a_scale_next[s];
            b_scale_tile[index / BM][index % BM] = b_scale_next[s];
        }
    };

    float accumulators[4][8][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) accumulators[i][j][0] = accumulators[i][j][1] = 0.0f;

    fetch(0);
    for (int stage = 0; stage < stages; ++stage) {
        store();
        __syncthreads();
        if (stage + 1 < stages) fetch(stage + 1);
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            int partial[4][8][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) partial[i][j][0] = partial[i][j][1] = 0;
            #pragma unroll
            for (int half_k = 0; half_k < 2; ++half_k) {
                const int byte = group * 32 + half_k * 16 + (lane % 4) * 4;
                unsigned a_fragment[4], b_fragment[8];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    a_fragment[i] = *reinterpret_cast<const unsigned *>(&a_tile[(warp_m * 32 + i * 8 + lane / 4) * V2_STRIDE + byte]);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    b_fragment[j] = *reinterpret_cast<const unsigned *>(&b_tile[(warp_n * 64 + j * 8 + lane / 4) * V2_STRIDE + byte]);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) mma_s4(partial[i][j][0], partial[i][j][1], a_fragment[i], b_fragment[j]);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float sa = a_scale_tile[group][warp_m * 32 + i * 8 + lane / 4];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = warp_n * 64 + j * 8 + 2 * (lane % 4);
                    accumulators[i][j][0] += (float)partial[i][j][0] * (sa * b_scale_tile[group][col]);
                    accumulators[i][j][1] += (float)partial[i][j][1] * (sa * b_scale_tile[group][col + 1]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = block_m + warp_m * 32 + i * 8 + lane / 4;
            const int col = block_n + warp_n * 64 + j * 8 + 2 * (lane % 4);
            *reinterpret_cast<float2 *>(&y[(size_t)row * n + col]) = make_float2(accumulators[i][j][0], accumulators[i][j][1]);
        }
}

// MARK: - GEMM v3: v2 with one A row-fragment of partial sums live at a time


__global__ void __launch_bounds__(THREADS, 2)
w4a4_gemm_v3(const uint8_t *__restrict__ a_codes, const half *__restrict__ a_scales,
             const uint8_t *__restrict__ w_codes, const half *__restrict__ w_scales,
             float *__restrict__ y, int m, int n, int k) {
    __shared__ __align__(16) uint8_t a_tile[BM * V2_STRIDE];
    __shared__ __align__(16) uint8_t b_tile[BN * V2_STRIDE];
    __shared__ float a_scale_tile[4][BM];
    __shared__ float b_scale_tile[4][BN];

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int warp_m = warp / 2, warp_n = warp % 2;
    const int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    const int groups = k / GROUP;
    const int stages = k / STAGE_K;

    // Each thread moves 4 x 16 bytes of A and of B per stage: rows tid / 8 and
    // tid / 8 + 32, + 64, + 96; 16-byte column (tid % 8).
    const int load_row = tid / 8, load_column = (tid % 8) * 16;
    uint4 a_next[4], b_next[4];
    float a_scale_next[2], b_scale_next[2];
    auto fetch = [&](int stage) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = load_row + r * 32;
            a_next[r] = *reinterpret_cast<const uint4 *>(&a_codes[(size_t)(block_m + row) * (k / 2) + stage * STAGE_BYTES + load_column]);
            b_next[r] = *reinterpret_cast<const uint4 *>(&w_codes[(size_t)(block_n + row) * (k / 2) + stage * STAGE_BYTES + load_column]);
        }
        // 4 groups x 128 rows of scales for each operand: two per thread.
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int index = tid + s * THREADS;  // 0..511
            const int group = index / BM, row = index % BM;
            a_scale_next[s] = __half2float(a_scales[(size_t)(block_m + row) * groups + stage * 4 + group]);
            b_scale_next[s] = __half2float(w_scales[(size_t)(block_n + row) * groups + stage * 4 + group]);
        }
    };
    auto store = [&]() {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = load_row + r * 32;
            *reinterpret_cast<uint4 *>(&a_tile[row * V2_STRIDE + load_column]) = a_next[r];
            *reinterpret_cast<uint4 *>(&b_tile[row * V2_STRIDE + load_column]) = b_next[r];
        }
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int index = tid + s * THREADS;
            a_scale_tile[index / BM][index % BM] = a_scale_next[s];
            b_scale_tile[index / BM][index % BM] = b_scale_next[s];
        }
    };

    float accumulators[4][8][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) accumulators[i][j][0] = accumulators[i][j][1] = 0.0f;

    fetch(0);
    for (int stage = 0; stage < stages; ++stage) {
        store();
        __syncthreads();
        if (stage + 1 < stages) fetch(stage + 1);
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            // B fragments for both 32-deep halves of this group.
            unsigned b_fragment[2][8];
            #pragma unroll
            for (int half_k = 0; half_k < 2; ++half_k)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    b_fragment[half_k][j] = *reinterpret_cast<const unsigned *>(
                        &b_tile[(warp_n * 64 + j * 8 + lane / 4) * V2_STRIDE + group * 32 + half_k * 16 + (lane % 4) * 4]);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int partial[8][2];
                #pragma unroll
                for (int j = 0; j < 8; ++j) partial[j][0] = partial[j][1] = 0;
                #pragma unroll
                for (int half_k = 0; half_k < 2; ++half_k) {
                    const unsigned a_fragment = *reinterpret_cast<const unsigned *>(
                        &a_tile[(warp_m * 32 + i * 8 + lane / 4) * V2_STRIDE + group * 32 + half_k * 16 + (lane % 4) * 4]);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) mma_s4(partial[j][0], partial[j][1], a_fragment, b_fragment[half_k][j]);
                }
                const float sa = a_scale_tile[group][warp_m * 32 + i * 8 + lane / 4];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = warp_n * 64 + j * 8 + 2 * (lane % 4);
                    accumulators[i][j][0] += (float)partial[j][0] * (sa * b_scale_tile[group][col]);
                    accumulators[i][j][1] += (float)partial[j][1] * (sa * b_scale_tile[group][col + 1]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = block_m + warp_m * 32 + i * 8 + lane / 4;
            const int col = block_n + warp_n * 64 + j * 8 + 2 * (lane % 4);
            *reinterpret_cast<float2 *>(&y[(size_t)row * n + col]) = make_float2(accumulators[i][j][0], accumulators[i][j][1]);
        }
}


// MARK: - Reference and harness

__device__ int nibble(const uint8_t *codes, size_t row, int k, int col) {
    uint8_t byte = codes[row * (k / 2) + col / 2];
    int v = (col % 2 == 0) ? (byte & 0xF) : (byte >> 4);
    return v >= 8 ? v - 16 : v;
}

// Straightforward per-output evaluation of the same formula, for the first rows.
__global__ void reference_gemm(const uint8_t *a_codes, const half *a_scales, const uint8_t *w_codes,
                               const half *w_scales, float *y, int rows, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (col >= n || row >= rows) return;
    double total = 0.0;
    for (int g = 0; g < k / GROUP; ++g) {
        int sum = 0;
        for (int c = g * GROUP; c < (g + 1) * GROUP; ++c) sum += nibble(a_codes, row, k, c) * nibble(w_codes, col, k, c);
        total += (double)sum * __half2float(a_scales[(size_t)row * (k / GROUP) + g]) * __half2float(w_scales[(size_t)col * (k / GROUP) + g]);
    }
    y[(size_t)row * n + col] = (float)total;
}

static uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

struct Shape { const char *name; int n; int k; };

int main(int argc, char **argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int version = argc > 2 ? std::atoi(argv[2]) : 1;
    const bool use_v2 = version == 2;
    if (m % BM != 0) { std::fprintf(stderr, "M must be a multiple of %d\n", BM); return 1; }
    const Shape shapes[] = {
        {"qkv (3x fused)", 12288, 4096},
        {"attn out", 4096, 4096},
        {"gate+proj (fused)", 24576, 4096},
        {"mlp down", 4096, 12288},
    };
    double step_gemm = 0.0, step_quantize = 0.0;
    for (const Shape &s : shapes) {
        const int n = s.n, k = s.k, groups = k / GROUP;
        std::vector<uint8_t> host_w((size_t)n * k / 2);
        std::vector<half> host_ws((size_t)n * groups), host_x((size_t)m * k);
        for (size_t i = 0; i < host_w.size(); ++i) {
            uint32_t h = hash((uint32_t)i);
            int lo = (int)(h % 15) - 7, hi = (int)((h >> 8) % 15) - 7;
            host_w[i] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
        }
        for (size_t i = 0; i < host_ws.size(); ++i) host_ws[i] = __float2half(0.001f + (hash((uint32_t)i * 3u) % 1000) * 1e-6f);
        for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = __float2half(((int)(hash((uint32_t)i * 11u) % 2001) - 1000) * 1e-3f);

        uint8_t *w, *a; half *ws, *as, *x; float *y, *reference;
        CHECK_CUDA(cudaMalloc(&w, host_w.size()));
        CHECK_CUDA(cudaMalloc(&ws, host_ws.size() * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&x, host_x.size() * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&a, (size_t)m * k / 2));
        CHECK_CUDA(cudaMalloc(&as, (size_t)m * groups * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&y, (size_t)m * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&reference, (size_t)m * n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(w, host_w.data(), host_w.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws, host_ws.data(), host_ws.size() * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(x, host_x.data(), host_x.size() * sizeof(half), cudaMemcpyHostToDevice));

        auto run_quantize = [&]() { quantize_activations<<<m, 256, k * sizeof(float)>>>(x, a, as, k); };
        dim3 grid(n / BN, m / BM);
        auto run_gemm = [&]() {
            if (version == 3) w4a4_gemm_v3<<<grid, THREADS>>>(a, as, w, ws, y, m, n, k);
            else if (use_v2) w4a4_gemm_v2<<<grid, THREADS>>>(a, as, w, ws, y, m, n, k);
            else w4a4_gemm<<<grid, THREADS>>>(a, as, w, ws, y, m, n, k);
        };
        run_quantize();
        run_gemm();
        const int check_rows = 128;
        reference_gemm<<<dim3((n + 127) / 128, check_rows), 128>>>(a, as, w, ws, reference, check_rows, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> ours((size_t)check_rows * n), expected((size_t)check_rows * n);
        CHECK_CUDA(cudaMemcpy(ours.data(), y, ours.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double error = 0.0, energy = 0.0;
        for (size_t i = 0; i < ours.size(); ++i) {
            double d = (double)ours[i] - expected[i];
            error += d * d;
            energy += (double)expected[i] * expected[i];
        }

        auto time = [&](auto run) {
            for (int i = 0; i < 3; ++i) run();
            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < 20; ++i) run();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            return ms / 20.0f;
        };
        float gemm_ms = time(run_gemm), quantize_ms = time(run_quantize);
        double ops = 2.0 * m * n * k;
        std::printf("M=%d %-18s N=%5d K=%5d  w4a4 gemm %7.3f ms %6.1f TOPS | act rotate+quant %6.3f ms | nRMSE vs reference %.2e\n",
                    m, s.name, n, k, gemm_ms, ops / (gemm_ms * 1e-3) / 1e12, quantize_ms, std::sqrt(error / energy));
        step_gemm += gemm_ms * 32;
        step_quantize += quantize_ms * 32;
        cudaFree(w); cudaFree(ws); cudaFree(x); cudaFree(a); cudaFree(as); cudaFree(y); cudaFree(reference);
    }
    std::printf("block matrices per step: w4a4 gemm %.1f ms + activation quantization %.1f ms "
                "(qkv and gate+proj inputs are each quantized once)\n", step_gemm, step_quantize);
    return 0;
}
