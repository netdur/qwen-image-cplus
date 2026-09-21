#include <metal_stdlib>
using namespace metal;

constant uint VAE_REDUCTION_THREADS = 128;
constant uint VAE_ATTENTION_CHANNELS = 1152;
constant uint VAE_ATTENTION_LANES = 9;

struct ConvParams {
    uint input_height;
    uint input_width;
    uint input_channels;
    uint output_height;
    uint output_width;
    uint output_channels;
    uint kernel_size;
    uint padding;
    uint upsample;
};

struct NormParams {
    uint pixels;
    uint channels;
    uint silu;
};

struct DupParams {
    uint input_height;
    uint input_width;
    uint input_channels;
    uint output_channels;
    uint factor_t;
    uint repeats;
};

struct ElementParams {
    uint elements;
    uint channels;
};

// Treat NHWC convolution as an implicit [pixels, K] x [K, output_channels]
// matrix product. The input tile is gathered directly from the image, so the
// decoder does not need to materialize an im2col buffer (680 MiB at its largest
// 256x256 layer). Sixteen SIMD groups reuse each input tile across a 64x64
// output tile with FP32 cooperative matrices and FP32 accumulation.
kernel void vae_conv2d_f32_simdgroup_64x64(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ConvParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[64][32];
    threadgroup float weight_tile[32][64];
    threadgroup float output_tile[64][64];

    const uint pixel_base = group_position.y * 64;
    const uint output_base = group_position.x * 64;
    const uint simd_row = simd_index / 8;
    const uint simd_column = simd_index % 8;
    const uint output_pixels = params.output_height * params.output_width;
    const uint kernel_area = params.kernel_size * params.kernel_size;
    const uint inner_count = params.input_channels * kernel_area;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint inner_base = 0; inner_base < inner_count; inner_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_pixel = linear / 32;
            const uint local_inner = linear % 32;
            const uint pixel = pixel_base + local_pixel;
            const uint input_inner = inner_base + local_inner;
            float input_value = 0.0f;
            if (pixel < output_pixels && input_inner < inner_count) {
                const uint output_y = pixel / params.output_width;
                const uint output_x = pixel % params.output_width;
                const uint input_channel = input_inner / kernel_area;
                const uint kernel_position = input_inner % kernel_area;
                const int kernel_y = int(kernel_position / params.kernel_size);
                const int kernel_x = int(kernel_position % params.kernel_size);
                const int sample_y = int(output_y) + kernel_y - int(params.padding);
                const int sample_x = int(output_x) + kernel_x - int(params.padding);
                if (params.upsample != 0) {
                    const int expanded_height = int(params.input_height * 2);
                    const int expanded_width = int(params.input_width * 2);
                    if (sample_y >= 0 && sample_y < expanded_height
                        && sample_x >= 0 && sample_x < expanded_width) {
                        const uint source_y = uint(sample_y) / 2;
                        const uint source_x = uint(sample_x) / 2;
                        input_value = input[(source_y * params.input_width + source_x)
                                            * params.input_channels + input_channel];
                    }
                } else if (sample_y >= 0 && sample_y < int(params.input_height)
                           && sample_x >= 0 && sample_x < int(params.input_width)) {
                    input_value = input[(uint(sample_y) * params.input_width + uint(sample_x))
                                        * params.input_channels + input_channel];
                }
            }
            input_tile[local_pixel][local_inner] = input_value;
        }
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_channel = linear / 32;
            const uint local_inner = linear % 32;
            const uint input_inner = inner_base + local_inner;
            const uint output_channel = output_base + local_channel;
            weight_tile[local_inner][local_channel] =
                output_channel < params.output_channels && input_inner < inner_count
                    ? weights[output_channel * inner_count + input_inner]
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_float8x8 left_0;
            simdgroup_float8x8 left_1;
            simdgroup_float8x8 left_2;
            simdgroup_float8x8 left_3;
            simdgroup_float8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[16 + simd_row * 8][inner], 32);
            simdgroup_load(left_2, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(left_3, &input_tile[48 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 64);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, left_2, right, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, left_3, right, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 64);
    simdgroup_store(accumulator_1, &output_tile[16 + simd_row * 8][simd_column * 8], 64);
    simdgroup_store(accumulator_2, &output_tile[32 + simd_row * 8][simd_column * 8], 64);
    simdgroup_store(accumulator_3, &output_tile[48 + simd_row * 8][simd_column * 8], 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 4096; linear += 512) {
        const uint local_pixel = linear / 64;
        const uint local_channel = linear % 64;
        const uint pixel = pixel_base + local_pixel;
        const uint output_channel = output_base + local_channel;
        if (pixel < output_pixels && output_channel < params.output_channels) {
            output[pixel * params.output_channels + output_channel] =
                output_tile[local_pixel][local_channel] + bias[output_channel];
        }
    }
}

// Nearest-neighbor 2x followed by a 3x3 convolution has only four distinct
// spatial phases. Collapse the repeated samples into a 2x2 kernel for each
// output parity once per layer, reducing the production convolution's inner
// dimension from 9*C to 4*C. Layout is [parity, output, input, corner].
kernel void vae_pack_nearest_upsample_3x3(
    device const float *weights [[buffer(0)]],
    device float *packed [[buffer(1)]],
    constant ConvParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint packed_inner = params.input_channels * 4;
    const uint count = 4 * params.output_channels * packed_inner;
    if (index >= count) {
        return;
    }
    const uint parity = index / (params.output_channels * packed_inner);
    const uint remainder = index % (params.output_channels * packed_inner);
    const uint output_channel = remainder / packed_inner;
    const uint input_inner = remainder % packed_inner;
    const uint input_channel = input_inner / 4;
    const uint corner = input_inner % 4;
    const uint parity_y = parity / 2;
    const uint parity_x = parity % 2;
    const uint corner_y = corner / 2;
    const uint corner_x = corner % 2;
    float combined = 0.0f;
    for (uint kernel_y = 0; kernel_y < 3; ++kernel_y) {
        const uint selected_y = parity_y == 0
            ? (kernel_y == 0 ? 0 : 1)
            : (kernel_y == 2 ? 1 : 0);
        if (selected_y != corner_y) {
            continue;
        }
        for (uint kernel_x = 0; kernel_x < 3; ++kernel_x) {
            const uint selected_x = parity_x == 0
                ? (kernel_x == 0 ? 0 : 1)
                : (kernel_x == 2 ? 1 : 0);
            if (selected_x == corner_x) {
                const uint source = output_channel * params.input_channels * 9
                    + input_channel * 9 + kernel_y * 3 + kernel_x;
                combined += weights[source];
            }
        }
    }
    packed[index] = combined;
}

kernel void vae_nearest_upsample_conv2d_f32_simdgroup_64x64(
    device const float *input [[buffer(0)]],
    device const float *packed_weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ConvParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[64][32];
    threadgroup float weight_tile[32][64];
    threadgroup float output_tile[64][64];

    const uint source_pixel_base = group_position.y * 64;
    const uint output_base = group_position.x * 64;
    const uint parity = group_position.z;
    const uint parity_y = parity / 2;
    const uint parity_x = parity % 2;
    const uint simd_row = simd_index / 8;
    const uint simd_column = simd_index % 8;
    const uint source_pixels = params.input_height * params.input_width;
    const uint packed_inner = params.input_channels * 4;
    device const float *phase_weights = packed_weights
        + parity * params.output_channels * packed_inner;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint inner_base = 0; inner_base < packed_inner; inner_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_pixel = linear / 32;
            const uint local_inner = linear % 32;
            const uint source_pixel = source_pixel_base + local_pixel;
            const uint packed_index = inner_base + local_inner;
            float input_value = 0.0f;
            if (source_pixel < source_pixels && packed_index < packed_inner) {
                const int source_y = int(source_pixel / params.input_width)
                    + int(packed_index % 4 / 2) + (parity_y == 0 ? -1 : 0);
                const int source_x = int(source_pixel % params.input_width)
                    + int(packed_index % 2) + (parity_x == 0 ? -1 : 0);
                if (source_y >= 0 && source_y < int(params.input_height)
                    && source_x >= 0 && source_x < int(params.input_width)) {
                    const uint input_channel = packed_index / 4;
                    input_value = input[(uint(source_y) * params.input_width + uint(source_x))
                                        * params.input_channels + input_channel];
                }
            }
            input_tile[local_pixel][local_inner] = input_value;
        }
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_channel = linear / 32;
            const uint local_inner = linear % 32;
            const uint packed_index = inner_base + local_inner;
            const uint output_channel = output_base + local_channel;
            weight_tile[local_inner][local_channel] =
                output_channel < params.output_channels && packed_index < packed_inner
                    ? phase_weights[output_channel * packed_inner + packed_index]
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_float8x8 left_0;
            simdgroup_float8x8 left_1;
            simdgroup_float8x8 left_2;
            simdgroup_float8x8 left_3;
            simdgroup_float8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[16 + simd_row * 8][inner], 32);
            simdgroup_load(left_2, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(left_3, &input_tile[48 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 64);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, left_2, right, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, left_3, right, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 64);
    simdgroup_store(accumulator_1, &output_tile[16 + simd_row * 8][simd_column * 8], 64);
    simdgroup_store(accumulator_2, &output_tile[32 + simd_row * 8][simd_column * 8], 64);
    simdgroup_store(accumulator_3, &output_tile[48 + simd_row * 8][simd_column * 8], 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 4096; linear += 512) {
        const uint local_pixel = linear / 64;
        const uint local_channel = linear % 64;
        const uint source_pixel = source_pixel_base + local_pixel;
        const uint output_channel = output_base + local_channel;
        if (source_pixel < source_pixels && output_channel < params.output_channels) {
            const uint source_y = source_pixel / params.input_width;
            const uint source_x = source_pixel % params.input_width;
            const uint output_y = source_y * 2 + parity_y;
            const uint output_x = source_x * 2 + parity_x;
            const uint output_pixel = output_y * params.output_width + output_x;
            output[output_pixel * params.output_channels + output_channel] =
                output_tile[local_pixel][local_channel] + bias[output_channel];
        }
    }
}

kernel void vae_rms_norm(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NormParams &params [[buffer(3)]],
    uint pixel [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[VAE_REDUCTION_THREADS];
    if (pixel >= params.pixels) {
        return;
    }
    const uint start = pixel * params.channels;
    float square_sum = 0.0f;
    for (uint channel = lane; channel < params.channels; channel += VAE_REDUCTION_THREADS) {
        square_sum = fma(input[start + channel], input[start + channel], square_sum);
    }
    reduction[lane] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = VAE_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = sqrt(reduction[0]);
    const float multiplier = sqrt(float(params.channels)) / max(norm, 1.0e-12f);
    for (uint channel = lane; channel < params.channels; channel += VAE_REDUCTION_THREADS) {
        float value = input[start + channel] * multiplier * gamma[channel];
        if (params.silu != 0) {
            value = value / (1.0f + exp(-value));
        }
        output[start + channel] = value;
    }
}

kernel void vae_add(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ElementParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.elements) {
        output[index] = left[index] + right[index];
    }
}

kernel void vae_denormalize(
    device const float *input [[buffer(0)]],
    device const float *mean [[buffer(1)]],
    device const float *standard_deviation [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ElementParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.elements) {
        const uint channel = index % params.channels;
        output[index] = input[index] * standard_deviation[channel] + mean[channel];
    }
}

kernel void vae_dup_up_shortcut(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DupParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint output_height = params.input_height * 2;
    const uint output_width = params.input_width * 2;
    const uint elements = output_height * output_width * params.output_channels;
    if (index >= elements) {
        return;
    }
    const uint output_channel = index % params.output_channels;
    const uint pixel = index / params.output_channels;
    const uint output_y = pixel / output_width;
    const uint output_x = pixel % output_width;
    // The official first-chunk path selects temporal sub-index factor_t-1
    // after repeat_interleave -> view -> permute. Reconstruct the originating
    // repeated-channel index without materializing the duplicated tensor.
    const uint repeated_channel =
        (((output_channel * params.factor_t + (params.factor_t - 1)) * 2 + (output_y % 2)) * 2
         + (output_x % 2));
    const uint input_channel = repeated_channel / params.repeats;
    output[index] = input[((output_y / 2) * params.input_width + (output_x / 2))
                          * params.input_channels + input_channel];
}

kernel void vae_single_head_attention(
    device const float *qkv [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ElementParams &params [[buffer(2)]],
    uint query_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[VAE_REDUCTION_THREADS];
    float accumulated[VAE_ATTENTION_LANES];
    for (uint slot = 0; slot < VAE_ATTENTION_LANES; ++slot) {
        accumulated[slot] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const uint channels = params.channels;
    const uint q_start = query_index * channels * 3;
    for (uint key_index = 0; key_index < params.elements; ++key_index) {
        const uint k_start = key_index * channels * 3 + channels;
        float partial = 0.0f;
        for (uint channel = lane; channel < channels; channel += VAE_REDUCTION_THREADS) {
            partial = fma(qkv[q_start + channel], qkv[k_start + channel], partial);
        }
        reduction[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = VAE_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float score = reduction[0] * rsqrt(float(channels));
        const float next_max = max(running_max, score);
        const float old_scale = running_max == -INFINITY ? 0.0f : exp(running_max - next_max);
        const float weight = exp(score - next_max);
        running_sum = running_sum * old_scale + weight;
        const uint v_start = key_index * channels * 3 + channels * 2;
        for (uint slot = 0; slot < VAE_ATTENTION_LANES; ++slot) {
            const uint channel = lane + slot * VAE_REDUCTION_THREADS;
            if (channel < channels) {
                accumulated[slot] = accumulated[slot] * old_scale + weight * qkv[v_start + channel];
            }
        }
        running_max = next_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint slot = 0; slot < VAE_ATTENTION_LANES; ++slot) {
        const uint channel = lane + slot * VAE_REDUCTION_THREADS;
        if (channel < channels) {
            output[query_index * channels + channel] = accumulated[slot] / running_sum;
        }
    }
}

kernel void vae_clamp(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ElementParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.elements) {
        output[index] = clamp(input[index], -1.0f, 1.0f);
    }
}
