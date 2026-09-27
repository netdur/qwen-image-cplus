// Qwen3-VL vision tower kernels. Activations are FP32 [tokens, width]
// row-major; one image is one attention sequence.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

static cublasHandle_t vision_blas = nullptr;

static unsigned blocks_for(unsigned long long count, unsigned threads) {
    return (unsigned)((count + threads - 1) / threads);
}

// LayerNorm with weight and bias (eps 1e-6), one block per row.
__global__ void layer_norm_affine_kernel(const float *x, const float *weight, const float *bias, float *out, int width) {
    const float *row = x + (size_t)blockIdx.x * width;
    float *target = out + (size_t)blockIdx.x * width;
    __shared__ float partial[32];
    __shared__ float mean_shared, inverse_shared;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += row[i];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (int)(blockDim.x / 32); ++i) total += partial[i];
        mean_shared = total / width;
    }
    __syncthreads();
    const float mean = mean_shared;
    float squares = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float centered = row[i] - mean;
        squares += centered * centered;
    }
    for (int offset = 16; offset > 0; offset /= 2) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    __syncthreads();
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (int)(blockDim.x / 32); ++i) total += partial[i];
        inverse_shared = rsqrtf(total / width + 1e-6f);
    }
    __syncthreads();
    const float inverse = inverse_shared;
    for (int i = threadIdx.x; i < width; i += blockDim.x) target[i] = (row[i] - mean) * inverse * weight[i] + bias[i];
}

extern "C" int qi_layer_norm_affine(const float *x, const float *weight, const float *bias, float *out, int rows, int width) {
    layer_norm_affine_kernel<<<rows, 256>>>(x, weight, bias, out, width);
    return (int)cudaGetLastError();
}

__global__ void add_bias_rows_kernel(float *y, const float *bias, int width, unsigned long long count) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) y[i] += bias[i % width];
}

// y[r, c] += bias[c].
extern "C" int qi_add_bias_rows(float *y, const float *bias, int rows, int width) {
    const unsigned long long count = (unsigned long long)rows * width;
    add_bias_rows_kernel<<<blocks_for(count, 256), 256>>>(y, bias, width, count);
    return (int)cudaGetLastError();
}

__global__ void gelu_erf_kernel(float *x, unsigned long long count) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

// Exact (erf) GELU in place, nn.GELU().
extern "C" int qi_gelu_erf(float *x, unsigned long long count) {
    gelu_erf_kernel<<<blocks_for(count, 256), 256>>>(x, count);
    return (int)cudaGetLastError();
}

// Rotate-half rotary on q and k inside the packed qkv rows [tokens, 3 * heads
// * head_dim]; table [tokens, 2 * head_dim] holds cos then sin.
__global__ void vision_rope_kernel(float *qkv, const float *table, int tokens, int heads, int head_dim) {
    const int half = head_dim / 2;
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)tokens * 2 * heads * half;
    if (i >= count) return;
    const int pair = (int)(i % half);
    const int head = (int)((i / half) % (2 * heads));  // q heads then k heads
    const int token = (int)(i / ((unsigned long long)half * 2 * heads));
    float *vector = qkv + (size_t)token * 3 * heads * head_dim + (size_t)head * head_dim;
    const float *cosine = table + (size_t)token * 2 * head_dim;
    const float *sine = cosine + head_dim;
    const float first = vector[pair], second = vector[pair + half];
    vector[pair] = first * cosine[pair] - second * sine[pair];
    vector[pair + half] = second * cosine[pair + half] + first * sine[pair + half];
}

extern "C" int qi_vision_rope(float *qkv, const float *table, int tokens, int heads, int head_dim) {
    const unsigned long long count = (unsigned long long)tokens * heads * head_dim;
    vision_rope_kernel<<<blocks_for(count, 256), 256>>>(qkv, table, tokens, heads, head_dim);
    return (int)cudaGetLastError();
}

// Softmax over each row of scores [rows, columns] after scaling, in place.
__global__ void softmax_scaled_kernel(float *scores, int columns, float scale) {
    float *row = scores + (size_t)blockIdx.x * columns;
    __shared__ float partial[32];
    __shared__ float shared_value;
    float peak = -INFINITY;
    for (int i = threadIdx.x; i < columns; i += blockDim.x) peak = fmaxf(peak, row[i] * scale);
    for (int offset = 16; offset > 0; offset /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, offset));
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = peak;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = -INFINITY;
        for (int i = 0; i < (int)(blockDim.x / 32); ++i) total = fmaxf(total, partial[i]);
        shared_value = total;
    }
    __syncthreads();
    peak = shared_value;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < columns; i += blockDim.x) {
        const float e = expf(row[i] * scale - peak);
        row[i] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    __syncthreads();
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (int)(blockDim.x / 32); ++i) total += partial[i];
        shared_value = 1.0f / total;
    }
    __syncthreads();
    const float inverse = shared_value;
    for (int i = threadIdx.x; i < columns; i += blockDim.x) row[i] *= inverse;
}

// Full (non-causal) multi-head attention over the packed qkv rows [tokens, 3 *
// heads * head_dim]; out [tokens, heads * head_dim]; scores: tokens^2 floats.
// One head at a time keeps the score buffer at tokens^2.
extern "C" int qi_vision_attention(const float *qkv, float *out, float *scores, int tokens, int heads, int head_dim) {
    if (vision_blas == nullptr && cublasCreate(&vision_blas) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorInitializationError;
    const int width = heads * head_dim, ld = 3 * width;
    const float one = 1.0f, zero = 0.0f;
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (int head = 0; head < heads; ++head) {
        const float *q = qkv + (size_t)head * head_dim;
        const float *k = q + width;
        const float *v = q + 2 * width;
        // Column-major S^T[keys, queries] = K_h . Q_h^T, i.e. row-major S[query][key].
        if (cublasSgemm(vision_blas, CUBLAS_OP_T, CUBLAS_OP_N, tokens, tokens, head_dim, &one, k, ld, q, ld, &zero, scores,
                        tokens) != CUBLAS_STATUS_SUCCESS) {
            return (int)cudaErrorUnknown;
        }
        softmax_scaled_kernel<<<tokens, 256>>>(scores, tokens, scale);
        if (cudaError_t error = cudaGetLastError()) return (int)error;
        // Column-major O_h^T[head_dim, queries] = V_h^T[head_dim, keys] . P^T[keys, queries].
        if (cublasSgemm(vision_blas, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, tokens, tokens, &one, v, ld, scores, tokens, &zero,
                        out + (size_t)head * head_dim, width) != CUBLAS_STATUS_SUCCESS) {
            return (int)cudaErrorUnknown;
        }
    }
    return (int)cudaGetLastError();
}

// hidden[p, c] += sum_t weights[p, t] * table[indices[p, t], c] for 4 taps
// (the bilinear position-embedding resample); table is BF16.
__global__ void gather4_add_kernel(float *hidden, const unsigned short *table, const int *indices, const float *weights,
                                   int width, unsigned long long count) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int column = (int)(i % width);
    const size_t patch = i / width;
    float sum = 0.0f;
    for (int t = 0; t < 4; ++t) {
        const unsigned int bits = (unsigned int)table[(size_t)indices[patch * 4 + t] * width + column] << 16;
        sum += __uint_as_float(bits) * weights[patch * 4 + t];
    }
    hidden[i] += sum;
}

extern "C" int qi_gather4_add(float *hidden, const void *table_bf16, const int *indices, const float *weights, int rows,
                              int width) {
    const unsigned long long count = (unsigned long long)rows * width;
    gather4_add_kernel<<<blocks_for(count, 256), 256>>>(hidden, (const unsigned short *)table_bf16, indices, weights, width,
                                                        count);
    return (int)cudaGetLastError();
}
