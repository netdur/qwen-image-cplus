#include <metal_stdlib>
using namespace metal;

struct DecodeParams {
    uint count;
    uint mode;
};

struct ElementwiseParams {
    uint count;
    uint mode;
};

struct NormParams {
    uint rows;
    uint width;
    float epsilon;
    uint mode;
};

struct TimestepParams {
    uint batch;
    uint dimension;
    float max_period;
    float time_factor;
};

struct RopeParams {
    uint complex_count;
};

struct MaskParams {
    uint sequence_length;
};

kernel void qi_decode_16(
    device const ushort *input [[buffer(0)]],
    device float *output [[buffer(2)]],
    constant DecodeParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.count) {
        return;
    }
    if (params.mode == 0) {
        output[index] = as_type<float>(uint(input[index]) << 16);
    } else {
        output[index] = float(as_type<half>(input[index]));
    }
}

kernel void qi_elementwise(
    device const float *input [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ElementwiseParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.count) {
        return;
    }
    const float value = input[index];
    if (params.mode == 0) {
        output[index] = value / (1.0f + exp(-value));
    } else if (params.mode == 1) {
        const float inner = 0.7978845608028654f * (value + 0.044715f * value * value * value);
        output[index] = 0.5f * value * (1.0f + tanh(inner));
    } else {
        output[index] = (value / (1.0f + exp(-value))) * projected[index];
    }
}

// Correctness kernel: one GPU thread owns one row and accumulates in FP32.
// The optimized reduction kernel comes after this path is fixture-verified.
kernel void qi_norm_rows(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NormParams &params [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= params.rows || params.width == 0) {
        return;
    }
    const uint start = row * params.width;
    float sum = 0.0f;
    if (params.mode == 0) {
        for (uint column = 0; column < params.width; ++column) {
            sum += input[start + column];
        }
        const float mean = sum / float(params.width);
        float variance = 0.0f;
        for (uint column = 0; column < params.width; ++column) {
            const float centered = input[start + column] - mean;
            variance += centered * centered;
        }
        const float inverse_std = rsqrt(variance / float(params.width) + params.epsilon);
        for (uint column = 0; column < params.width; ++column) {
            output[start + column] = (input[start + column] - mean) * inverse_std;
        }
        return;
    }

    for (uint column = 0; column < params.width; ++column) {
        const float value = input[start + column];
        sum += value * value;
    }
    const float inverse_rms = rsqrt(sum / float(params.width) + params.epsilon);
    for (uint column = 0; column < params.width; ++column) {
        float value = input[start + column] * inverse_rms;
        if (params.mode == 2) {
            value *= weight[column] + 1.0f;
        }
        output[start + column] = value;
    }
}

kernel void qi_timestep_embedding(
    device const float *timesteps [[buffer(0)]],
    device float *output [[buffer(2)]],
    constant TimestepParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint half_dimension = params.dimension / 2;
    const uint total = params.batch * half_dimension;
    if (index >= total || half_dimension == 0) {
        return;
    }
    const uint batch_index = index / half_dimension;
    const uint frequency_index = index % half_dimension;
    const float exponent = -log(params.max_period) * float(frequency_index) / float(half_dimension);
    const float phase = params.time_factor * timesteps[batch_index] * exp(exponent);
    const uint output_start = batch_index * params.dimension;
    output[output_start + frequency_index] = cos(phase);
    output[output_start + half_dimension + frequency_index] = sin(phase);
}

kernel void qi_complex_rope(
    device const float *input [[buffer(0)]],
    device const float *cosine [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *sine [[buffer(3)]],
    constant RopeParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.complex_count) {
        return;
    }
    const float real = input[index * 2];
    const float imaginary = input[index * 2 + 1];
    output[index * 2] = real * cosine[index] - imaginary * sine[index];
    output[index * 2 + 1] = real * sine[index] + imaginary * cosine[index];
}

kernel void qi_block_causal_mask(
    device const int *image_ids [[buffer(0)]],
    device const uchar *key_valid [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant MaskParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.sequence_length * params.sequence_length;
    if (index >= count) {
        return;
    }
    const uint query = index / params.sequence_length;
    const uint key = index % params.sequence_length;
    const int query_image = image_ids[query];
    const int key_image = image_ids[key];
    const bool same_image = query_image >= 0 && query_image == key_image;
    output[index] = uchar(key_valid[key] != 0 && (query >= key || same_image));
}
