// INT4 block-matrix GEMMs for the v6 packs (see src/int4_pack.cplus for the
// storage formats). Both compute y[M,N] (FP32) = x[M,K] . W[N,K]^T with M and
// N multiples of 128 and K a multiple of 256; callers pad rows to 128.
// Benchmarks and derivations: cuda/bench/bench_w4a4.cu and bench_w4a16.cu.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

constexpr int GROUP = 64;
constexpr int BM = 128, BN = 128;
constexpr int THREADS = 256;

// MARK: - W4A4 (scheme 6)

// One block per row: rotate the FP32 row by the H256 butterfly (strides 1, 4,
// 16, 64 in every 256-chunk), then quantize each 64-group to signed codes
// -7..7 with scale fp16(max|x| / 7). Same rule as the NumPy evaluator.
__global__ void quantize_activations_kernel(const float *__restrict__ x, uint8_t *__restrict__ codes,
                                            half *__restrict__ scales, int k) {
    extern __shared__ float row[];
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < k; i += blockDim.x) row[i] = x[(size_t)m * k + i];
    __syncthreads();
    for (int stride = 1; stride < 256; stride *= 4) {
        const int span = stride * 4;
        for (int b = threadIdx.x; b < k / 4; b += blockDim.x) {
            const int i0 = (b / stride) * span + b % stride;
            const float a = row[i0], bb = row[i0 + stride], c = row[i0 + 2 * stride], d = row[i0 + 3 * stride];
            row[i0] = (a + bb + c - d) * 0.5f;
            row[i0 + stride] = (a + bb - c + d) * 0.5f;
            row[i0 + 2 * stride] = (a - bb + c + d) * 0.5f;
            row[i0 + 3 * stride] = (-a + bb + c + d) * 0.5f;
        }
        __syncthreads();
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (int g = warp; g < k / GROUP; g += warps) {
        const float v0 = row[g * GROUP + 2 * lane], v1 = row[g * GROUP + 2 * lane + 1];
        float peak = fmaxf(fabsf(v0), fabsf(v1));
        for (int s = 16; s > 0; s /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, s));
        const half scale_half = __float2half(fmaxf(peak / 7.0f, 5.9604645e-08f));
        const float scale = __half2float(scale_half);
        const int c0 = (int)fminf(fmaxf(rintf(v0 / scale), -7.0f), 7.0f);
        const int c1 = (int)fminf(fmaxf(rintf(v1 / scale), -7.0f), 7.0f);
        codes[(size_t)m * (k / 2) + g * (GROUP / 2) + lane] = (uint8_t)((c0 & 0xF) | ((c1 & 0xF) << 4));
        if (lane == 0) scales[(size_t)m * (k / GROUP) + g] = scale_half;
    }
}

extern "C" int qi_w4a4_quantize_activations(const float *x, void *codes, void *scales, int m, int k) {
    quantize_activations_kernel<<<m, 256, (size_t)k * sizeof(float)>>>(x, (uint8_t *)codes, (half *)scales, k);
    return (int)cudaGetLastError();
}


__device__ __forceinline__ void mma_s4(int &c0, int &c1, unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}

// 8 warps as 4 (M) x 2 (N), each 32 x 64. Each stage covers 256 inputs (four
// 64-groups) behind one pair of barriers, and every thread prefetches the next
// stage into registers while the current one computes. Per group: two
// m8n8k32 steps in int32, then FP32 += int32 * (activation scale * weight
// scale). 255 registers, one block per SM: forcing two blocks spills and
// measured 47% slower (cuda/bench/bench_w4a4.cu, versions 1-3).
constexpr int STAGE_K = 256;                 // four 64-groups per stage
constexpr int STAGE_BYTES = STAGE_K / 2;     // 128 code bytes per row per stage
constexpr int V2_STRIDE = STAGE_BYTES + 16;  // 144 bytes: fragment rows cover all banks

__global__ void __launch_bounds__(THREADS)
w4a4_gemm_kernel(const uint8_t *__restrict__ a_codes, const half *__restrict__ a_scales,
             const uint8_t *__restrict__ w_codes, const half *__restrict__ w_scales,
             float *__restrict__ y, int n, int k) {
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


extern "C" int qi_w4a4_gemm(const void *a_codes, const void *a_scales, const void *w_codes, const void *w_scales,
                            float *y, int m, int n, int k) {
    if (m % BM != 0 || n % BN != 0 || k % 256 != 0) return (int)cudaErrorInvalidValue;
    w4a4_gemm_kernel<<<dim3(n / BN, m / BM), THREADS>>>((const uint8_t *)a_codes, (const half *)a_scales,
                                                        (const uint8_t *)w_codes, (const half *)w_scales, y, n, k);
    return (int)cudaGetLastError();
}

// MARK: - W4A16 (scheme 5)

constexpr int BK = 64;
constexpr int LDS = BK + 8;

// 8 warps as 2 (M) x 4 (N), each 64 x 32. One 64-wide k tile is one group,
// so each weight row needs one scale/minimum pair per tile.
__global__ void __launch_bounds__(THREADS)
w4a16_gemm_kernel(const half *__restrict__ x, const uint8_t *__restrict__ codes,
                  const half *__restrict__ scales, const half *__restrict__ minimums,
                  float *__restrict__ y, int n, int k) {
    __shared__ __align__(16) half a_tile[BM * LDS];
    __shared__ __align__(16) half b_tile[BN * LDS];
    const int tid = threadIdx.x, warp = tid / 32;
    const int warp_m = warp / 4, warp_n = warp % 4;
    const int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    const int groups = k / GROUP;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(accumulators[i][j], 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int chunk = tid; chunk < BM * BK / 8; chunk += THREADS) {
            const int row = chunk / (BK / 8), col = (chunk % (BK / 8)) * 8;
            *reinterpret_cast<uint4 *>(&a_tile[row * LDS + col]) =
                *reinterpret_cast<const uint4 *>(&x[(size_t)(block_m + row) * k + k0 + col]);
        }
        {
            const int row = tid / 2, half_index = tid % 2, weight_row = block_n + row;
            const uint4 packed = *reinterpret_cast<const uint4 *>(&codes[(size_t)weight_row * (k / 2) + k0 / 2 + half_index * 16]);
            const int group = k0 / GROUP;
            const float scale = __half2float(scales[(size_t)weight_row * groups + group]);
            const float minimum = __half2float(minimums[(size_t)weight_row * groups + group]);
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&packed);
            __align__(16) half expanded[32];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                expanded[2 * i] = __float2half(fmaf((float)(bytes[i] & 0x0F), scale, minimum));
                expanded[2 * i + 1] = __float2half(fmaf((float)(bytes[i] >> 4), scale, minimum));
            }
            uint4 *destination = reinterpret_cast<uint4 *>(&b_tile[row * LDS + half_index * 32]);
            const uint4 *source = reinterpret_cast<const uint4 *>(expanded);
            #pragma unroll
            for (int i = 0; i < 4; ++i) destination[i] = source[i];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragments[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragments[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(a_fragments[i], &a_tile[(warp_m * 64 + i * 16) * LDS + kk], LDS);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b_fragments[j], &b_tile[(warp_n * 32 + j * 16) * LDS + kk], LDS);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(accumulators[i][j], a_fragments[i], b_fragments[j], accumulators[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
            const int row = block_m + warp_m * 64 + i * 16, col = block_n + warp_n * 32 + j * 16;
            wmma::store_matrix_sync(&y[(size_t)row * n + col], accumulators[i][j], n, wmma::mem_row_major);
        }
}

extern "C" int qi_w4a16_gemm(const void *x_half, const void *codes, const void *scales, const void *minimums,
                             float *y, int m, int n, int k) {
    if (m % BM != 0 || n % BN != 0 || k % GROUP != 0) return (int)cudaErrorInvalidValue;
    w4a16_gemm_kernel<<<dim3(n / BN, m / BM), THREADS>>>((const half *)x_half, (const uint8_t *)codes,
                                                         (const half *)scales, (const half *)minimums, y, n, k);
    return (int)cudaGetLastError();
}
