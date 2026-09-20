#include <metal_stdlib>
using namespace metal;

constant uint VAE_TILE = 16;
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

kernel void vae_conv2d_f32_16x16(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ConvParams &params [[buffer(4)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[VAE_TILE][VAE_TILE];
    threadgroup float weight_tile[VAE_TILE][VAE_TILE];

    const uint pixel = group_position.y * VAE_TILE + thread_position.y;
    const uint output_channel = group_position.x * VAE_TILE + thread_position.x;
    const uint output_pixels = params.output_height * params.output_width;
    const uint kernel_area = params.kernel_size * params.kernel_size;
    const uint inner_count = params.input_channels * kernel_area;
    float sum = output_channel < params.output_channels ? bias[output_channel] : 0.0f;

    for (uint inner_base = 0; inner_base < inner_count; inner_base += VAE_TILE) {
        const uint input_inner = inner_base + thread_position.x;
        float input_value = 0.0f;
        if (pixel < output_pixels && input_inner < inner_count) {
            const uint output_y = pixel / params.output_width;
            const uint output_x = pixel % params.output_width;
            const uint input_channel = input_inner / kernel_area;
            const uint kernel_position = input_inner % kernel_area;
            const int kernel_y = int(kernel_position / params.kernel_size);
            const int kernel_x = int(kernel_position % params.kernel_size);
            int sample_y = int(output_y) + kernel_y - int(params.padding);
            int sample_x = int(output_x) + kernel_x - int(params.padding);
            if (params.upsample != 0) {
                const int expanded_height = int(params.input_height * 2);
                const int expanded_width = int(params.input_width * 2);
                if (sample_y >= 0 && sample_y < expanded_height && sample_x >= 0 && sample_x < expanded_width) {
                    const uint source_y = uint(sample_y) / 2;
                    const uint source_x = uint(sample_x) / 2;
                    input_value = input[(source_y * params.input_width + source_x) * params.input_channels + input_channel];
                }
            } else if (sample_y >= 0 && sample_y < int(params.input_height)
                       && sample_x >= 0 && sample_x < int(params.input_width)) {
                input_value = input[(uint(sample_y) * params.input_width + uint(sample_x))
                                    * params.input_channels + input_channel];
            }
        }
        input_tile[thread_position.y][thread_position.x] = input_value;

        const uint weight_inner = inner_base + thread_position.y;
        weight_tile[thread_position.y][thread_position.x] =
            output_channel < params.output_channels && weight_inner < inner_count
                ? weights[output_channel * inner_count + weight_inner]
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < VAE_TILE; ++inner) {
            sum = fma(input_tile[thread_position.y][inner], weight_tile[inner][thread_position.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (pixel < output_pixels && output_channel < params.output_channels) {
        output[pixel * params.output_channels + output_channel] = sum;
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
