// cuBLAS FP16 GEMM ceiling at the transformer's real shapes.
//
// y[M,N] = x[M,K] . W[N,K]^T with row-major x, W, y (QIPACK stores weights
// [out, in]). Reports ms and TFLOP/s per shape for FP32 and FP16 accumulation,
// and the implied block-matrix time per transformer step (32 blocks).
//
//   nvcc -O3 -arch=sm_75 -o bench_gemm_cublas bench_gemm_cublas.cu -lcublas

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s:%d cublas status %d\n", __FILE__, __LINE__, (int)s); std::exit(1); } } while (0)

struct Shape { const char *name; int n; int k; int per_block; };

// Per block: Q, K, V, attention out (4096x4096), gate and proj (12288x4096),
// and the MLP down projection (4096x12288).
static const Shape SHAPES[] = {
    {"qkv (3x fused)", 12288, 4096, 1},
    {"attn out", 4096, 4096, 1},
    {"gate+proj (fused)", 24576, 4096, 1},
    {"mlp down", 4096, 12288, 1},
};

__global__ void fill(half *data, size_t count, unsigned seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned x = (unsigned)i * 2654435761u + seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        data[i] = __float2half(((x & 0xFFFF) / 65535.0f - 0.5f) * 0.1f);
    }
}

static float time_gemm(cublasHandle_t handle, cublasComputeType_t compute, int m, int n, int k,
                       const half *x, const half *w, half *y, int iterations) {
    const float one_f = 1.0f, zero_f = 0.0f;
    const half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
    const void *alpha = compute == CUBLAS_COMPUTE_16F ? (const void *)&one_h : (const void *)&one_f;
    const void *beta = compute == CUBLAS_COMPUTE_16F ? (const void *)&zero_h : (const void *)&zero_f;
    // Column-major view: y^T[N,M] = W[N,K] (as col-major K x N, transposed) . x^T[K,M].
    auto run = [&]() {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha,
                                  w, CUDA_R_16F, k, x, CUDA_R_16F, k, beta,
                                  y, CUDA_R_16F, n, compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    for (int i = 0; i < 3; ++i) run();
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) run();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

int main(int argc, char **argv) {
    std::vector<int> rows = {1056, 4128};
    if (argc > 1) { rows.clear(); for (int i = 1; i < argc; ++i) rows.push_back(std::atoi(argv[i])); }
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    size_t max_weights = 0, max_x = 0, max_y = 0;
    int max_m = 0;
    for (int m : rows) max_m = m > max_m ? m : max_m;
    for (const Shape &s : SHAPES) {
        size_t wsz = (size_t)s.n * s.k, xsz = (size_t)max_m * s.k, ysz = (size_t)max_m * s.n;
        if (wsz > max_weights) max_weights = wsz;
        if (xsz > max_x) max_x = xsz;
        if (ysz > max_y) max_y = ysz;
    }
    half *w, *x, *y;
    CHECK_CUDA(cudaMalloc(&w, max_weights * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&x, max_x * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&y, max_y * sizeof(half)));
    fill<<<(unsigned)((max_weights + 255) / 256), 256>>>(w, max_weights, 1u);
    fill<<<(unsigned)((max_x + 255) / 256), 256>>>(x, max_x, 2u);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s, %d SMs\n", prop.name, prop.multiProcessorCount);
    const cublasComputeType_t computes[] = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_16F};
    const char *compute_names[] = {"fp32-acc", "fp16-acc"};
    for (int m : rows) {
        for (int c = 0; c < 2; ++c) {
            double step_ms = 0.0;
            for (const Shape &s : SHAPES) {
                float ms = time_gemm(handle, computes[c], m, s.n, s.k, x, w, y, 20);
                double tflops = 2.0 * m * s.n * s.k / (ms * 1e-3) / 1e12;
                std::printf("M=%5d %-8s %-18s N=%5d K=%5d  %8.3f ms  %6.2f TFLOP/s\n",
                            m, compute_names[c], s.name, s.n, s.k, ms, tflops);
                step_ms += ms * s.per_block * 32;
            }
            std::printf("M=%5d %-8s block matrices per step (32 blocks): %.1f ms\n\n", m, compute_names[c], step_ms);
        }
    }
    cudaFree(w); cudaFree(x); cudaFree(y);
    cublasDestroy(handle);
    return 0;
}
