// cuBLAS-backed pieces: FP32 linear layers for the global (BF16-stored)
// matrices, and attention as two strided batched GEMMs around a masked
// softmax. One process-wide handle.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>

static cublasHandle_t handle = nullptr;

static int ensure_handle() {
    if (handle != nullptr) return 0;
    return cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorInitializationError;
}

static int blas_status(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? (int)cudaGetLastError() : (int)cudaErrorUnknown;
}

// y[M,N] = x[M,K] . W[N,K]^T (+ y if accumulate), all FP32 row-major.
extern "C" int qi_linear_f32(const float *x, const float *weights, float *y, int m, int n, int k, int accumulate) {
    if (int status = ensure_handle()) return status;
    const float one = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    return blas_status(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, weights, k, x, k, &beta, y, n));
}

// Scaled scores, masked, softmaxed in place. scores[h][q][c] for q < queries,
// c < keys. Query q is a prefix (text) row when q + query_offset < causal_rows:
// it sees keys c <= q + query_offset only. Every other query sees all keys.
__global__ void masked_softmax_kernel(float *scores, int queries, int keys, int query_offset, int causal_rows,
                                      float scale) {
    __shared__ float scratch[32];
    const int q = blockIdx.x, h = blockIdx.y;
    float *row = scores + ((size_t)h * queries + q) * keys;
    const int global_q = q + query_offset;
    const int visible = global_q < causal_rows ? global_q + 1 : keys;
    float peak = -INFINITY;
    for (int c = threadIdx.x; c < visible; c += blockDim.x) peak = fmaxf(peak, row[c] * scale);
    for (int o = 16; o > 0; o /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = peak;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    peak = scratch[0];
    __syncthreads();
    float sum = 0.0f;
    for (int c = threadIdx.x; c < keys; c += blockDim.x) {
        const float e = c < visible ? expf(row[c] * scale - peak) : 0.0f;
        row[c] = e;
        sum += e;
    }
    for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    const float inverse = 1.0f / scratch[0];
    for (int c = threadIdx.x; c < keys; c += blockDim.x) row[c] *= inverse;
}

// out[q, h, :] = softmax(Q[q,h] . K[:,h]^T / sqrt(128), mask) . V[:, h]
// q: [queries, heads*128], k and v: [keys, heads*128], all FP32 row-major.
// scores: scratch of heads * queries * keys floats.
extern "C" int qi_attention(const float *q, const float *k, const float *v, float *out, float *scores,
                            int queries, int keys, int heads, int query_offset, int causal_rows) {
    if (int status = ensure_handle()) return status;
    const int width = heads * 128;
    const float one = 1.0f, zero = 0.0f;
    // Column-major: S^T[keys, queries] = K_h^T . Q_h per head.
    cublasStatus_t status = cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, keys, queries, 128, &one,
        k, width, 128, q, width, 128, &zero, scores, keys, (long long)queries * keys, heads);
    if (status != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    masked_softmax_kernel<<<dim3(queries, heads), 256>>>(scores, queries, keys, query_offset, causal_rows,
                                                         1.0f / sqrtf(128.0f));
    if (cudaError_t error = cudaGetLastError()) return (int)error;
    // Column-major: O_h^T[128, queries] = V_h^T[128, keys] . P^T[keys, queries].
    status = cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, queries, keys, &one,
        v, width, 128, scores, keys, (long long)queries * keys, &zero, out, width, 128, heads);
    return blas_status(status);
}

__global__ void to_half_kernel(const float *source, __half *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half(source[i]);
}

static unsigned half_blocks(unsigned long long count) { return (unsigned)((count + 255) / 256); }

// qi_attention with both batched GEMMs on FP16 tensor cores (FP32
// accumulation). q/k/v are converted to FP16; scores and the softmax stay
// FP32; probabilities are converted to FP16 for the value product.
// halves: scratch of (queries + 2 * keys) * heads * 128 + heads * queries * keys halves.
extern "C" int qi_attention_f16(const float *q, const float *k, const float *v, float *out, float *scores, void *halves,
                                int queries, int keys, int heads, int query_offset, int causal_rows) {
    if (int status = ensure_handle()) return status;
    const int width = heads * 128;
    __half *q_half = (__half *)halves;
    __half *k_half = q_half + (size_t)queries * width;
    __half *v_half = k_half + (size_t)keys * width;
    __half *p_half = v_half + (size_t)keys * width;
    to_half_kernel<<<half_blocks((unsigned long long)queries * width), 256>>>(q, q_half, (unsigned long long)queries * width);
    to_half_kernel<<<half_blocks((unsigned long long)keys * width), 256>>>(k, k_half, (unsigned long long)keys * width);
    to_half_kernel<<<half_blocks((unsigned long long)keys * width), 256>>>(v, v_half, (unsigned long long)keys * width);
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, keys, queries, 128, &one,
        k_half, CUDA_R_16F, width, 128, q_half, CUDA_R_16F, width, 128, &zero,
        scores, CUDA_R_32F, keys, (long long)queries * keys, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    masked_softmax_kernel<<<dim3(queries, heads), 256>>>(scores, queries, keys, query_offset, causal_rows,
                                                         1.0f / sqrtf(128.0f));
    const unsigned long long probabilities = (unsigned long long)heads * queries * keys;
    to_half_kernel<<<half_blocks(probabilities), 256>>>(scores, p_half, probabilities);
    if (cudaError_t error = cudaGetLastError()) return (int)error;
    status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, queries, keys, &one,
        v_half, CUDA_R_16F, width, 128, p_half, CUDA_R_16F, keys, (long long)queries * keys, &zero,
        out, CUDA_R_32F, width, 128, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return blas_status(status);
}
