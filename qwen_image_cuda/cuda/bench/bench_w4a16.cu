// W4A16 GEMM for the w4a16-g64-v6 pack format on sm_75.
//
// y[M,N] (FP32) = x[M,K] (FP16) . W[N,K]^T where W is stored as QIPACK
// scheme 5: codes [N, K/2] (even k in the low nibble, unsigned 0..15), and
// FP16 scale and minimum per (row, 64 inputs): w = code * scale + minimum.
//
// One 64-wide k tile is exactly one quantization group, so each weight row
// needs a single scale/minimum pair per tile. Weights are expanded to FP16 in
// shared memory and multiplied with WMMA (FP16 operands, FP32 accumulation).
// M must be a multiple of 128 (the engine keeps image rows at that size).
//
// Checks against cuBLAS on the same dequantized weights, then times both.
//
//   nvcc -O3 -arch=sm_75 -o bench_w4a16 bench_w4a16.cu -lcublas

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nvcuda;

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s:%d cublas status %d\n", __FILE__, __LINE__, (int)s); std::exit(1); } } while (0)

constexpr int GROUP = 64;
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int PAD = 8;
constexpr int LDS = BK + PAD;  // shared row stride in halves
constexpr int THREADS = 256;   // 8 warps: 2 (M) x 4 (N), each 64 x 32

__global__ void __launch_bounds__(THREADS)
w4a16_gemm(const half *__restrict__ x, const uint8_t *__restrict__ codes,
           const half *__restrict__ scales, const half *__restrict__ minimums,
           float *__restrict__ y, int m, int n, int k) {
    __shared__ __align__(16) half a_tile[BM * LDS];
    __shared__ __align__(16) half b_tile[BN * LDS];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / 4;  // 0..1
    const int warp_n = warp % 4;  // 0..3
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int groups = k / GROUP;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(accumulators[i][j], 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // A: 128 x 64 halves, 1024 16-byte chunks, 4 per thread.
        for (int chunk = tid; chunk < BM * BK / 8; chunk += THREADS) {
            int row = chunk / (BK / 8);
            int col = (chunk % (BK / 8)) * 8;
            *reinterpret_cast<uint4 *>(&a_tile[row * LDS + col]) =
                *reinterpret_cast<const uint4 *>(&x[(size_t)(block_m + row) * k + k0 + col]);
        }
        // B: 128 weight rows x 64 codes = 32 bytes per row; each thread expands
        // 16 bytes (32 codes) of one row with that row's group scale/minimum.
        {
            int row = tid / 2;
            int half_index = tid % 2;
            int weight_row = block_n + row;
            uint4 packed = *reinterpret_cast<const uint4 *>(
                &codes[(size_t)weight_row * (k / 2) + k0 / 2 + half_index * 16]);
            int group = k0 / GROUP;
            float scale = __half2float(scales[(size_t)weight_row * groups + group]);
            float minimum = __half2float(minimums[(size_t)weight_row * groups + group]);
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&packed);
            half expanded[32];
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
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(accumulators[i][j], a_fragments[i], b_fragments[j], accumulators[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
            int row = block_m + warp_m * 64 + i * 16;
            int col = block_n + warp_n * 32 + j * 16;
            wmma::store_matrix_sync(&y[(size_t)row * n + col], accumulators[i][j], n, wmma::mem_row_major);
        }
}

__global__ void dequantize(const uint8_t *codes, const half *scales, const half *minimums,
                           half *weights, int n, int k) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * k) return;
    size_t row = i / k, col = i % k;
    uint8_t byte = codes[(row * k + col) / 2];
    int code = (col % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    size_t group = row * (k / GROUP) + col / GROUP;
    weights[i] = __float2half(fmaf((float)code, __half2float(scales[group]), __half2float(minimums[group])));
}

__global__ void to_float(const half *source, float *destination, size_t count) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) destination[i] = __half2float(source[i]);
}

static uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

struct Shape { const char *name; int n; int k; };

int main(int argc, char **argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1024;
    if (m % BM != 0) { std::fprintf(stderr, "M must be a multiple of %d\n", BM); return 1; }
    const Shape shapes[] = {
        {"qkv (3x fused)", 12288, 4096},
        {"attn out", 4096, 4096},
        {"gate+proj (fused)", 24576, 4096},
        {"mlp down", 4096, 12288},
    };
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    double step_ours = 0.0, step_cublas = 0.0;
    for (const Shape &s : shapes) {
        const int n = s.n, k = s.k, groups = k / GROUP;
        std::vector<uint8_t> host_codes((size_t)n * k / 2);
        std::vector<half> host_scales((size_t)n * groups), host_minimums((size_t)n * groups);
        std::vector<half> host_x((size_t)m * k);
        for (size_t i = 0; i < host_codes.size(); ++i) host_codes[i] = (uint8_t)hash((uint32_t)i);
        for (size_t i = 0; i < host_scales.size(); ++i) {
            host_scales[i] = __float2half(0.001f + (hash((uint32_t)i * 3u) % 1000) * 1e-6f);
            host_minimums[i] = __float2half(-0.008f - (hash((uint32_t)i * 7u) % 1000) * 1e-6f);
        }
        for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = __float2half(((int)(hash((uint32_t)i * 11u) % 2001) - 1000) * 1e-3f);

        uint8_t *codes; half *scales, *minimums, *x, *weights, *reference_half; float *y, *reference;
        CHECK_CUDA(cudaMalloc(&codes, host_codes.size()));
        CHECK_CUDA(cudaMalloc(&scales, host_scales.size() * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&minimums, host_minimums.size() * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&x, host_x.size() * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&weights, (size_t)n * k * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&y, (size_t)m * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&reference, (size_t)m * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&reference_half, (size_t)m * n * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(codes, host_codes.data(), host_codes.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(scales, host_scales.data(), host_scales.size() * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(minimums, host_minimums.data(), host_minimums.size() * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(x, host_x.data(), host_x.size() * sizeof(half), cudaMemcpyHostToDevice));
        size_t weight_count = (size_t)n * k;
        dequantize<<<(unsigned)((weight_count + 255) / 256), 256>>>(codes, scales, minimums, weights, n, k);

        dim3 grid(n / BN, m / BM);
        auto run_ours = [&]() { w4a16_gemm<<<grid, THREADS>>>(x, codes, scales, minimums, y, m, n, k); };
        const float one = 1.0f, zero = 0.0f;
        auto run_cublas = [&]() {
            CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, weights, CUDA_R_16F, k,
                                      x, CUDA_R_16F, k, &zero, reference, CUDA_R_32F, n,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        run_ours();
        run_cublas();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> ours((size_t)m * n), expected((size_t)m * n);
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
        float ours_ms = time(run_ours), cublas_ms = time(run_cublas);
        double flops = 2.0 * m * n * k;
        std::printf("M=%d %-18s N=%5d K=%5d  w4a16 %7.3f ms %6.2f TFLOP/s | cublas fp16 %7.3f ms %6.2f TFLOP/s | nRMSE %.2e\n",
                    m, s.name, n, k, ours_ms, flops / (ours_ms * 1e-3) / 1e12, cublas_ms,
                    flops / (cublas_ms * 1e-3) / 1e12, std::sqrt(error / energy));
        step_ours += ours_ms * 32;
        step_cublas += cublas_ms * 32;
        cudaFree(codes); cudaFree(scales); cudaFree(minimums); cudaFree(x); cudaFree(weights);
        cudaFree(y); cudaFree(reference); cudaFree(reference_half);
    }
    std::printf("block matrices per step: w4a16 %.1f ms, cublas fp16 (resident, not possible in 6 GB) %.1f ms\n",
                step_ours, step_cublas);
    cublasDestroy(handle);
    return 0;
}
