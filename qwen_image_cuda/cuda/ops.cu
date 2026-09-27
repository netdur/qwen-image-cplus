// Element-wise, normalization, and positional kernels for the transformer.
// Activations are row-major FP32 unless a name says otherwise. Every launcher
// returns the cudaError_t of its launch (0 is success).
//
// The equations follow tools/generate_transformer_model_fixture.py, the NumPy
// transcription of the pinned Diffusers QwenImage21Transformer2DModel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr float EPSILON = 1e-6f;

static unsigned blocks_for(unsigned long long count, unsigned threads) {
    return (unsigned)((count + threads - 1) / threads);
}

__device__ float block_sum(float value, float *scratch) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_xor_sync(0xffffffffu, value, offset);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    const int warps = blockDim.x / 32;
    value = lane < warps ? scratch[lane] : 0.0f;
    if (warp == 0)
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_xor_sync(0xffffffffu, value, offset);
    if (threadIdx.x == 0) scratch[0] = value;
    __syncthreads();
    value = scratch[0];
    __syncthreads();
    return value;
}

// out[r] = layer_norm(x[r]) * (1 + scale[r < split ? 1 : 0]) where scale holds
// two rows of `width` (row 0: target timestep, row 1: prefix, t = 0).
// Affine-free LayerNorm, two-pass variance, as in the reference.
__global__ void layer_norm_scale_kernel(const float *x, const float *scale, int scale_stride,
                                        float *out, int width, int split) {
    __shared__ float scratch[32];
    const int row = blockIdx.x;
    const float *source = x + (size_t)row * width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += source[i];
    const float mean = block_sum(sum, scratch) / width;
    float squares = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float d = source[i] - mean;
        squares += d * d;
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / width + EPSILON);
    const float *row_scale = scale + (row < split ? scale_stride : 0);
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        out[(size_t)row * width + i] = (source[i] - mean) * inverse * (1.0f + row_scale[i]);
}

extern "C" int qi_layer_norm_scale(const float *x, const float *scale, int scale_stride, float *out,
                                   int rows, int width, int split) {
    layer_norm_scale_kernel<<<rows, 256>>>(x, scale, scale_stride, out, width, split);
    return (int)cudaGetLastError();
}

// RMSNorm over each full row with weight (+ offset, 1 for txt_in.text_norm).
__global__ void rms_norm_kernel(const float *x, const float *weight, float offset, float *out, int width) {
    __shared__ float scratch[32];
    const int row = blockIdx.x;
    const float *source = x + (size_t)row * width;
    float squares = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) squares += source[i] * source[i];
    const float inverse = rsqrtf(block_sum(squares, scratch) / width + EPSILON);
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        out[(size_t)row * width + i] = source[i] * inverse * (weight[i] + offset);
}

extern "C" int qi_rms_norm(const float *x, const float *weight, float offset, float *out, int rows, int width) {
    rms_norm_kernel<<<rows, 256>>>(x, weight, offset, out, width);
    return (int)cudaGetLastError();
}

// Per-head RMSNorm (weight [128]) followed by complex RoPE on interleaved
// pairs, in place on x [rows, heads, 128]. rope [rope_rows, 128] holds 64
// cosines then 64 sines; row r of x uses rope row (rope_offset + r).
// One warp per (row, head): each lane owns two pairs.
__global__ void head_norm_rope_kernel(float *x, const float *weight, const float *rope, int rope_offset,
                                      int rows, int heads) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_global >= rows * heads) return;
    const int row = warp_global / heads;
    float *head = x + (size_t)warp_global * 128;
    float values[4];
    float squares = 0.0f;
    for (int i = 0; i < 4; ++i) {
        values[i] = head[lane * 4 + i];
        squares += values[i] * values[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    const float inverse = rsqrtf(squares / 128.0f + EPSILON);
    for (int i = 0; i < 4; ++i) values[i] = values[i] * inverse * weight[lane * 4 + i];
    const float *table = rope + (size_t)(rope_offset + row) * 128;
    for (int pair = 0; pair < 2; ++pair) {
        const int index = lane * 2 + pair;  // pair index 0..63
        const float cosine = table[index], sine = table[64 + index];
        const float real = values[pair * 2], imaginary = values[pair * 2 + 1];
        head[lane * 4 + pair * 2] = real * cosine - imaginary * sine;
        head[lane * 4 + pair * 2 + 1] = real * sine + imaginary * cosine;
    }
}

extern "C" int qi_head_norm_rope(float *x, const float *weight, const float *rope, int rope_offset,
                                 int rows, int heads) {
    const unsigned long long threads = (unsigned long long)rows * heads * 32;
    head_norm_rope_kernel<<<blocks_for(threads, 256), 256>>>(x, weight, rope, rope_offset, rows, heads);
    return (int)cudaGetLastError();
}

// hidden[r] += tanh(gate[r < split ? 1 : 0]) * update[r]
__global__ void gated_residual_kernel(float *hidden, const float *update, const float *gate, int gate_stride,
                                      int width, int split, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int row = (int)(i / width), col = (int)(i % width);
    const float g = gate[(row < split ? gate_stride : 0) + col];
    hidden[i] += tanhf(g) * update[i];
}

extern "C" int qi_gated_residual(float *hidden, const float *update, const float *gate, int gate_stride,
                                 int rows, int width, int split) {
    const unsigned long long count = (unsigned long long)rows * width;
    gated_residual_kernel<<<blocks_for(count, 256), 256>>>(hidden, update, gate, gate_stride, width, split, count);
    return (int)cudaGetLastError();
}

// out = silu(gate) * up
__global__ void swiglu_kernel(const float *gate, const float *up, float *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = gate[i];
    out[i] = g / (1.0f + expf(-g)) * up[i];
}

extern "C" int qi_swiglu(const float *gate, const float *up, float *out, unsigned long long count) {
    swiglu_kernel<<<blocks_for(count, 256), 256>>>(gate, up, out, count);
    return (int)cudaGetLastError();
}

__global__ void silu_kernel(const float *x, float *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = x[i] / (1.0f + expf(-x[i]));
}

extern "C" int qi_silu(const float *x, float *out, unsigned long long count) {
    silu_kernel<<<blocks_for(count, 256), 256>>>(x, out, count);
    return (int)cudaGetLastError();
}

__global__ void gelu_tanh_kernel(const float *x, float *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = x[i];
    out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

extern "C" int qi_gelu_tanh(const float *x, float *out, unsigned long long count) {
    gelu_tanh_kernel<<<blocks_for(count, 256), 256>>>(x, out, count);
    return (int)cudaGetLastError();
}

// Sinusoidal projection of (timestep / 1000) for each entry of `timesteps`:
// out[r] = [cos(t * f), sin(t * f)], f_i = exp(-ln(10000) * i / 128), i < 128.
__global__ void timestep_projection_kernel(const float *timesteps, float *out, int count) {
    const int row = blockIdx.x, i = threadIdx.x;
    if (row >= count || i >= 128) return;
    const float frequency = expf(-logf(10000.0f) * (float)i / 128.0f);
    const float argument = (timesteps[row] / 1000.0f) * 1000.0f * frequency;
    out[row * 256 + i] = cosf(argument);
    out[row * 256 + 128 + i] = sinf(argument);
}

extern "C" int qi_timestep_projection(const float *timesteps, float *out, int count) {
    timestep_projection_kernel<<<count, 128>>>(timesteps, out, count);
    return (int)cudaGetLastError();
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16 *source, float *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = __bfloat162float(source[i]);
}

extern "C" int qi_bf16_to_f32(const void *source, float *out, unsigned long long count) {
    bf16_to_f32_kernel<<<blocks_for(count, 256), 256>>>((const __nv_bfloat16 *)source, out, count);
    return (int)cudaGetLastError();
}

__global__ void f16_to_f32_kernel(const half *source, float *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(source[i]);
}

extern "C" int qi_f16_to_f32(const void *source, float *out, unsigned long long count) {
    f16_to_f32_kernel<<<blocks_for(count, 256), 256>>>((const half *)source, out, count);
    return (int)cudaGetLastError();
}

__global__ void f32_to_f16_kernel(const float *source, half *out, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half(source[i]);
}

extern "C" int qi_f32_to_f16(const float *source, void *out, unsigned long long count) {
    f32_to_f16_kernel<<<blocks_for(count, 256), 256>>>(source, (half *)out, count);
    return (int)cudaGetLastError();
}

extern "C" int qi_copy(void *destination, const void *source, unsigned long long bytes) {
    return (int)cudaMemcpy(destination, source, (size_t)bytes, cudaMemcpyDeviceToDevice);
}

extern "C" int qi_zero(void *destination, unsigned long long bytes) {
    return (int)cudaMemset(destination, 0, (size_t)bytes);
}

// Copies `rows` rows of `width` floats between buffers with different strides.
extern "C" int qi_copy_rows(float *destination, int destination_stride, const float *source, int source_stride,
                            int rows, int width) {
    return (int)cudaMemcpy2D(destination, (size_t)destination_stride * sizeof(float), source,
                             (size_t)source_stride * sizeof(float), (size_t)width * sizeof(float), rows,
                             cudaMemcpyDeviceToDevice);
}

// Flow-matching Euler update: latents += (sigma_next - sigma) * velocity.
__global__ void euler_kernel(float *latents, const float *velocity, float delta, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) latents[i] += delta * velocity[i];
}

extern "C" int qi_euler(float *latents, const float *velocity, float delta, unsigned long long count) {
    euler_kernel<<<blocks_for(count, 256), 256>>>(latents, velocity, delta, count);
    return (int)cudaGetLastError();
}
