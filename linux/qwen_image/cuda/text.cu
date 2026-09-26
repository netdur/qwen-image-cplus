// Text encoder (Qwen3-VL language model) kernels. Activations are FP32
// [tokens, width] row-major. Prompts are short, so attention is a direct
// per-(query, head) kernel rather than a GEMM.

#include <cuda_runtime.h>
#include <cmath>

static unsigned blocks_for(unsigned long long count, unsigned threads) {
    return (unsigned)((count + threads - 1) / threads);
}

// Per-head RMSNorm (weight [128], eps 1e-6) then rotate-half RoPE: pair i
// rotates elements i and i + 64 by angle position * theta^(-2i/128). With
// text-only input every M-RoPE axis carries the same position, so the
// interleaved [24, 20, 20] sections reduce to this 1D form. Frequencies and
// angles are computed in FP32 as the reference does. One warp per (row, head).
__global__ void head_norm_rope_half_kernel(float *x, const float *weight, int rows, int heads, float theta,
                                           int position_offset) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_global >= rows * heads) return;
    const int row = warp_global / heads;
    float *head = x + (size_t)warp_global * 128;
    float values[4];
    float squares = 0.0f;
    for (int i = 0; i < 4; ++i) {
        values[i] = head[lane + 32 * i];
        squares += values[i] * values[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    const float inverse = rsqrtf(squares / 128.0f + 1e-6f);
    for (int i = 0; i < 4; ++i) values[i] = values[i] * inverse * weight[lane + 32 * i];
    // Lane owns elements lane, lane+32 (first half) and lane+64, lane+96 (second half).
    const float position = (float)(row + position_offset);
    for (int pair = 0; pair < 2; ++pair) {
        const int index = lane + 32 * pair;
        const float inverse_frequency = 1.0f / powf(theta, (float)(2 * index) / 128.0f);
        const float angle = position * inverse_frequency;
        const float cosine = cosf(angle), sine = sinf(angle);
        const float first = values[pair], second = values[pair + 2];
        head[index] = first * cosine - second * sine;
        head[index + 64] = second * cosine + first * sine;
    }
}

extern "C" int qi_text_head_norm_rope(float *x, const float *weight, int rows, int heads, float theta,
                                      int position_offset) {
    const unsigned long long threads = (unsigned long long)rows * heads * 32;
    head_norm_rope_half_kernel<<<blocks_for(threads, 256), 256>>>(x, weight, rows, heads, theta, position_offset);
    return (int)cudaGetLastError();
}

// Causal grouped-query attention: q [tokens, q_heads*128], k and v
// [tokens, kv_heads*128]; query head h reads kv head h / (q_heads / kv_heads).
// One block (128 threads) per (query, head): scores in shared memory.
__global__ void causal_gqa_kernel(const float *q, const float *k, const float *v, float *out, int tokens,
                                  int q_heads, int kv_heads) {
    extern __shared__ float scores[];
    const int query = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int kv_head = head / (q_heads / kv_heads);
    const float *q_row = q + ((size_t)query * q_heads + head) * 128;
    __shared__ float q_shared[128];
    q_shared[d] = q_row[d];
    __syncthreads();
    const float scale = 1.0f / sqrtf(128.0f);
    for (int key = d; key <= query; key += blockDim.x) {
        const float *k_row = k + ((size_t)key * kv_heads + kv_head) * 128;
        float dot = 0.0f;
        for (int i = 0; i < 128; ++i) dot += q_shared[i] * k_row[i];
        scores[key] = dot * scale;
    }
    __syncthreads();
    __shared__ float peak_shared, sum_shared;
    if (d == 0) {
        float peak = -INFINITY;
        for (int key = 0; key <= query; ++key) peak = fmaxf(peak, scores[key]);
        float sum = 0.0f;
        for (int key = 0; key <= query; ++key) {
            scores[key] = expf(scores[key] - peak);
            sum += scores[key];
        }
        peak_shared = peak;
        sum_shared = sum;
    }
    __syncthreads();
    float accumulator = 0.0f;
    for (int key = 0; key <= query; ++key) accumulator += scores[key] * v[((size_t)key * kv_heads + kv_head) * 128 + d];
    out[((size_t)query * q_heads + head) * 128 + d] = accumulator / sum_shared;
}

extern "C" int qi_text_attention(const float *q, const float *k, const float *v, float *out, int tokens, int q_heads,
                                 int kv_heads) {
    causal_gqa_kernel<<<dim3(tokens, q_heads), 128, (size_t)tokens * sizeof(float)>>>(q, k, v, out, tokens, q_heads, kv_heads);
    return (int)cudaGetLastError();
}

// Rounds FP32 values to the nearest BF16 (ties to even), in place, to put the
// reference implementation's BF16 activation boundaries into an FP32 path.
__global__ void round_bf16_kernel(float *x, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int bits = __float_as_uint(x[i]);
    bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    x[i] = __uint_as_float(bits);
}

extern "C" int qi_round_bf16(float *x, unsigned long long count) {
    round_bf16_kernel<<<blocks_for(count, 256), 256>>>(x, count);
    return (int)cudaGetLastError();
}

extern "C" int qi_cuda_malloc_host(void **pointer, unsigned long long bytes) {
    return (int)cudaMallocHost(pointer, (size_t)bytes);
}

extern "C" int qi_cuda_free_host(void *pointer) {
    return (int)cudaFreeHost(pointer);
}

// head_norm_rope_half_kernel with Qwen3-VL's interleaved M-RoPE: frequency
// pair i takes its position from axis H when i % 3 == 1 and W when i % 3 == 2
// (both only for i < 60, the [24, 20, 20] sections), else from T. positions:
// [rows, 3] (t, h, w).
__global__ void head_norm_mrope_kernel(float *x, const float *weight, int rows, int heads, float theta,
                                       const int *positions) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_global >= rows * heads) return;
    const int row = warp_global / heads;
    float *head = x + (size_t)warp_global * 128;
    float values[4];
    float squares = 0.0f;
    for (int i = 0; i < 4; ++i) {
        values[i] = head[lane + 32 * i];
        squares += values[i] * values[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    const float inverse = rsqrtf(squares / 128.0f + 1e-6f);
    for (int i = 0; i < 4; ++i) values[i] = values[i] * inverse * weight[lane + 32 * i];
    for (int pair = 0; pair < 2; ++pair) {
        const int index = lane + 32 * pair;
        const int axis = index < 60 && index % 3 == 1 ? 1 : (index < 60 && index % 3 == 2 ? 2 : 0);
        const float position = (float)positions[row * 3 + axis];
        const float inverse_frequency = 1.0f / powf(theta, (float)(2 * index) / 128.0f);
        const float angle = position * inverse_frequency;
        const float cosine = cosf(angle), sine = sinf(angle);
        const float first = values[pair], second = values[pair + 2];
        head[index] = first * cosine - second * sine;
        head[index + 64] = second * cosine + first * sine;
    }
}

extern "C" int qi_text_head_norm_mrope(float *x, const float *weight, int rows, int heads, float theta,
                                       const int *positions) {
    const unsigned long long threads = (unsigned long long)rows * heads * 32;
    head_norm_mrope_kernel<<<blocks_for(threads, 256), 256>>>(x, weight, rows, heads, theta, positions);
    return (int)cudaGetLastError();
}

__global__ void scatter_rows_kernel(float *target, const float *source, const int *indices, int width, int accumulate,
                                    unsigned long long count) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned long long row = i / width;
    const int column = (int)(i % width);
    float *cell = target + (size_t)indices[row] * width + column;
    *cell = accumulate ? *cell + source[i] : source[i];
}

// target[indices[r], :] = source[r, :] (or += with `accumulate`) for r < rows.
extern "C" int qi_scatter_rows(float *target, const float *source, const int *indices, int rows, int width, int accumulate) {
    const unsigned long long count = (unsigned long long)rows * width;
    scatter_rows_kernel<<<blocks_for(count, 256), 256>>>(target, source, indices, width, accumulate, count);
    return (int)cudaGetLastError();
}
