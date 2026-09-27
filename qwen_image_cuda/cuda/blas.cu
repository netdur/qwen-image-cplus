// cuBLAS-backed pieces: FP32 linear layers for the global (BF16-stored)
// matrices, and attention as two strided batched GEMMs around a masked
// softmax. One process-wide handle.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// y[M, n] (row stride ldy) = x[M,K] . W[n,K]^T, FP32; for computing an output
// in column slices when the full weight does not fit in scratch.
extern "C" int qi_linear_f32_strided(const float *x, const float *weights, float *y, int m, int n, int k, int ldy) {
    if (int status = ensure_handle()) return status;
    const float one = 1.0f, zero = 0.0f;
    return blas_status(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, weights, k, x, k, &zero, y, ldy));
}

// Masked softmax over FP16 scores in place (FP32 arithmetic); see
// masked_softmax_kernel for the mask.
__global__ void masked_softmax_half_kernel(__half *scores, int queries, int keys, int query_offset, int causal_rows,
                                           float scale, const int *visible_keys) {
    __shared__ float scratch[32];
    const int q = blockIdx.x, h = blockIdx.y;
    __half *row = scores + ((size_t)h * queries + q) * keys;
    const int global_q = q + query_offset;
    // With `visible_keys`, row global_q sees keys [0, visible_keys[global_q]) (the
    // block-causal prefill: text causal, each image block through its own end).
    const int visible = visible_keys != nullptr ? visible_keys[global_q]
        : (global_q < causal_rows ? global_q + 1 : keys);
    float peak = -INFINITY;
    for (int c = threadIdx.x; c < visible; c += blockDim.x) peak = fmaxf(peak, __half2float(row[c]) * scale);
    for (int o = 16; o > 0; o /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = peak;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o /= 2) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, o));
        if (threadIdx.x == 0) scratch[0] = value;
    }
    __syncthreads();
    peak = scratch[0];
    __syncthreads();
    float sum = 0.0f;
    for (int c = threadIdx.x; c < keys; c += blockDim.x)
        sum += c < visible ? expf(__half2float(row[c]) * scale - peak) : 0.0f;
    for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o /= 2) value += __shfl_xor_sync(0xffffffffu, value, o);
        if (threadIdx.x == 0) scratch[0] = value;
    }
    __syncthreads();
    const float inverse = 1.0f / scratch[0];
    for (int c = threadIdx.x; c < keys; c += blockDim.x)
        row[c] = __float2half(c < visible ? expf(__half2float(row[c]) * scale - peak) * inverse : 0.0f);
}

// masked_softmax_half_kernel with the row held in registers: each of the 256
// threads keeps PAIRS half2 values, so the row is read once and written once
// (the loop version reads it three times). Needs an even key count and
// keys <= 512 * PAIRS.
template <int PAIRS>
__global__ void masked_softmax_half_registers_kernel(__half *scores, int queries, int keys, int query_offset,
                                                     int causal_rows, float scale, const int *visible_keys) {
    __shared__ float scratch[32];
    const int q = blockIdx.x, h = blockIdx.y;
    __half2 *row = (__half2 *)(scores + ((size_t)h * queries + q) * keys);
    const int global_q = q + query_offset;
    const int visible = visible_keys != nullptr ? visible_keys[global_q]
        : (global_q < causal_rows ? global_q + 1 : keys);
    const int pairs = keys / 2;
    float values[2 * PAIRS];
    float peak = -INFINITY;
#pragma unroll
    for (int i = 0; i < PAIRS; ++i) {
        const int pair = threadIdx.x + i * 256;
        float first = -INFINITY, second = -INFINITY;
        if (pair < pairs) {
            const float2 loaded = __half22float2(row[pair]);
            if (2 * pair < visible) first = loaded.x * scale;
            if (2 * pair + 1 < visible) second = loaded.y * scale;
        }
        values[2 * i] = first;
        values[2 * i + 1] = second;
        peak = fmaxf(peak, fmaxf(first, second));
    }
    for (int o = 16; o > 0; o /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = peak;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < 8 ? scratch[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o /= 2) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, o));
        if (threadIdx.x == 0) scratch[0] = value;
    }
    __syncthreads();
    peak = scratch[0];
    __syncthreads();
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 2 * PAIRS; ++i) {
        values[i] = expf(values[i] - peak);
        sum += values[i];
    }
    for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o /= 2) value += __shfl_xor_sync(0xffffffffu, value, o);
        if (threadIdx.x == 0) scratch[0] = value;
    }
    __syncthreads();
    const float inverse = 1.0f / scratch[0];
#pragma unroll
    for (int i = 0; i < PAIRS; ++i) {
        const int pair = threadIdx.x + i * 256;
        if (pair < pairs) row[pair] = __floats2half2_rn(values[2 * i] * inverse, values[2 * i + 1] * inverse);
    }
}

// Launches the register kernel sized to the row, or the loop kernel for odd or
// very long rows.
static void masked_softmax_half(__half *scores, int queries, int keys, int heads, int query_offset, int causal_rows,
                                float scale, const int *visible_keys) {
    const dim3 grid(queries, heads);
    const int pairs_per_thread = (keys / 2 + 255) / 256;
    if (keys % 2 != 0 || pairs_per_thread > 32) {
        masked_softmax_half_kernel<<<grid, 256>>>(scores, queries, keys, query_offset, causal_rows, scale, visible_keys);
    } else if (pairs_per_thread <= 4) {
        masked_softmax_half_registers_kernel<4><<<grid, 256>>>(scores, queries, keys, query_offset, causal_rows, scale, visible_keys);
    } else if (pairs_per_thread <= 8) {
        masked_softmax_half_registers_kernel<8><<<grid, 256>>>(scores, queries, keys, query_offset, causal_rows, scale, visible_keys);
    } else if (pairs_per_thread <= 16) {
        masked_softmax_half_registers_kernel<16><<<grid, 256>>>(scores, queries, keys, query_offset, causal_rows, scale, visible_keys);
    } else {
        masked_softmax_half_registers_kernel<32><<<grid, 256>>>(scores, queries, keys, query_offset, causal_rows, scale, visible_keys);
    }
}

// FP32 -> FP16 for K and V once per block, ahead of qi_attention_half.
extern "C" int qi_to_half(const float *source, void *out, unsigned long long count) {
    to_half_kernel<<<half_blocks(count), 256>>>(source, (__half *)out, count);
    return (int)cudaGetLastError();
}

// Attention with FP16 K and V already converted, FP16 scores written by the
// first GEMM (FP32 accumulation) and softmaxed in place.
// scratch: queries * heads * 128 halves for Q, then heads * queries * keys
// halves of scores.
extern "C" int qi_attention_half(const float *q, const void *k_half, const void *v_half, float *out, void *scratch,
                                 int queries, int keys, int heads, int query_offset, int causal_rows,
                                 const int *visible_keys) {
    if (int status = ensure_handle()) return status;
    const int width = heads * 128;
    __half *q_half = (__half *)scratch;
    __half *scores = q_half + (size_t)queries * width;
    to_half_kernel<<<half_blocks((unsigned long long)queries * width), 256>>>(q, q_half, (unsigned long long)queries * width);
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, keys, queries, 128, &one,
        k_half, CUDA_R_16F, width, 128, q_half, CUDA_R_16F, width, 128, &zero,
        scores, CUDA_R_16F, keys, (long long)queries * keys, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    masked_softmax_half(scores, queries, keys, heads, query_offset, causal_rows, 1.0f / sqrtf(128.0f), visible_keys);
    if (cudaError_t error = cudaGetLastError()) return (int)error;
    status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, queries, keys, &one,
        v_half, CUDA_R_16F, width, 128, scores, CUDA_R_16F, keys, (long long)queries * keys, &zero,
        out, CUDA_R_32F, width, 128, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return blas_status(status);
}

// Causal grouped-query attention for the text encoder with cuBLAS: q [tokens,
// q_heads * 128], k and v [tokens, kv_heads * 128]; query head h reads kv head
// h / (q_heads / kv_heads). Queries run in chunks sized to `score_floats`, and
// a chunk only multiplies the keys it can see.
extern "C" int qi_text_attention_gemm(const float *q, const float *k, const float *v, float *out, float *scores,
                                      unsigned long long score_floats, int tokens, int q_heads, int kv_heads) {
    if (int status = ensure_handle()) return status;
    const int group = q_heads / kv_heads;
    const int q_width = q_heads * 128, kv_width = kv_heads * 128;
    long long chunk = (long long)(score_floats / ((unsigned long long)group * tokens));
    if (chunk < 1) return (int)cudaErrorInvalidValue;
    if (chunk > tokens) chunk = tokens;
    const float one = 1.0f, zero = 0.0f;
    for (int start = 0; start < tokens; start += (int)chunk) {
        const int queries = tokens - start < chunk ? tokens - start : (int)chunk;
        const int keys = start + queries;
        for (int g = 0; g < kv_heads; ++g) {
            // Column-major S^T[keys, queries] per query head of the group.
            cublasStatus_t status = cublasSgemmStridedBatched(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, keys, queries, 128, &one,
                k + (size_t)g * 128, kv_width, 0,
                q + (size_t)start * q_width + (size_t)g * group * 128, q_width, 128,
                &zero, scores, keys, (long long)queries * keys, group);
            if (status != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
            masked_softmax_kernel<<<dim3(queries, group), 256>>>(scores, queries, keys, start, tokens, 1.0f / sqrtf(128.0f));
            if (cudaError_t error = cudaGetLastError()) return (int)error;
            status = cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, queries, keys, &one,
                v + (size_t)g * 128, kv_width, 0,
                scores, keys, (long long)queries * keys,
                &zero, out + (size_t)start * q_width + (size_t)g * group * 128, q_width, 128, group);
            if (status != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
        }
    }
    return (int)cudaGetLastError();
}

__global__ void bf16_to_half_kernel(const unsigned short *source, __half *out, unsigned long long count) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half(__uint_as_float((unsigned int)source[i] << 16));
}

__global__ void absmax_kernel(const float *x, unsigned long long count, unsigned int *out) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) atomicMax(out, __float_as_uint(fabsf(x[i])));
}

// y[M,N] = x[M,K] . W[N,K]^T with W stored BF16, on FP16 tensor cores with
// FP32 accumulation: W is converted into w_half (N*K halves) and x into x_half
// (M*K halves). FP16 holds every activation this is used for (checked with
// QI_TRACE_ABSMAX=1, which prints each input's peak).
extern "C" int qi_linear_bf16_half(const float *x, const void *w_bf16, float *y, void *w_half, void *x_half, int m, int n,
                                   int k) {
    if (int status = ensure_handle()) return status;
    const unsigned long long weights = (unsigned long long)n * k, inputs = (unsigned long long)m * k;
    if (getenv("QI_TRACE_ABSMAX") != nullptr) {
        unsigned int *peak;
        cudaMalloc(&peak, 4);
        cudaMemset(peak, 0, 4);
        absmax_kernel<<<half_blocks(inputs), 256>>>(x, inputs, peak);
        unsigned int bits = 0;
        cudaMemcpy(&bits, peak, 4, cudaMemcpyDeviceToHost);
        cudaFree(peak);
        float value;
        memcpy(&value, &bits, 4);
        fprintf(stderr, "absmax m=%d n=%d k=%d: %g\n", m, n, k, value);
    }
    bf16_to_half_kernel<<<half_blocks(weights), 256>>>((const unsigned short *)w_bf16, (__half *)w_half, weights);
    to_half_kernel<<<half_blocks(inputs), 256>>>(x, (__half *)x_half, inputs);
    const float one = 1.0f, zero = 0.0f;
    return blas_status(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w_half, CUDA_R_16F, k, x_half, CUDA_R_16F,
                                    k, &zero, y, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
