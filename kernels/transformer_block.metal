#include <metal_stdlib>
using namespace metal;

constant uint QI_TILE = 16;

struct LinearParams {
    uint rows;
    uint output_columns;
    uint input_columns;
    uint weight_mode;
};

struct Int8LinearParams {
    uint rows;
    uint output_columns;
    uint input_columns;
    uint group_size;
};

struct BlockParams {
    uint rows;
    uint width;
    uint heads;
    uint head_dimension;
    float epsilon;
    uint slot;
};

inline float qi_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void qi_block_linear_16x16(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];

    const uint row = group_position.y * QI_TILE + thread_position.y;
    const uint output_column = group_position.x * QI_TILE + thread_position.x;
    float sum = 0.0f;
    for (uint input_base = 0; input_base < params.input_columns; input_base += QI_TILE) {
        const uint input_column = input_base + thread_position.x;
        input_tile[thread_position.y][thread_position.x] =
            row < params.rows && input_column < params.input_columns
                ? input[row * params.input_columns + input_column]
                : 0.0f;
        const uint weight_input_column = input_base + thread_position.y;
        weight_tile[thread_position.y][thread_position.x] =
            output_column < params.output_columns && weight_input_column < params.input_columns
                ? qi_bf16(weights[output_column * params.input_columns + weight_input_column])
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < QI_TILE; ++inner) {
            sum = fma(input_tile[thread_position.y][inner], weight_tile[inner][thread_position.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.rows && output_column < params.output_columns) {
        output[row * params.output_columns + output_column] = sum;
    }
}

kernel void qi_block_int8_affine_linear_16x16(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uchar *zeros [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Int8LinearParams &params [[buffer(5)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[16][16];
    threadgroup half weight_tile[16][16];

    const uint row = group_position.y * QI_TILE + thread_position.y;
    const uint output_column = group_position.x * QI_TILE + thread_position.x;
    const uint groups_per_row = params.input_columns / params.group_size;
    float sum = 0.0f;
    for (uint input_base = 0; input_base < params.input_columns; input_base += QI_TILE) {
        const uint input_column = input_base + thread_position.x;
        input_tile[thread_position.y][thread_position.x] =
            row < params.rows && input_column < params.input_columns
                ? half(input[row * params.input_columns + input_column])
                : half(0.0h);
        const uint weight_input_column = input_base + thread_position.y;
        if (output_column < params.output_columns && weight_input_column < params.input_columns) {
            const uint weight_index = output_column * params.input_columns + weight_input_column;
            const uint group_index =
                output_column * groups_per_row + weight_input_column / params.group_size;
            weight_tile[thread_position.y][thread_position.x] =
                half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
        } else {
            weight_tile[thread_position.y][thread_position.x] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < QI_TILE; ++inner) {
            sum += float(
                input_tile[thread_position.y][inner] * weight_tile[inner][thread_position.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.rows && output_column < params.output_columns) {
        output[row * params.output_columns + output_column] = sum;
    }
}

// modulation is [real/zero, scale1/gate1/scale2/gate2, width].
kernel void qi_block_layernorm_modulate(
    device const float *input [[buffer(0)]],
    device const float *modulation [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uchar *target_mask [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    const uint start = row * params.width;
    float mean = 0.0f;
    for (uint column = 0; column < params.width; ++column) {
        mean += input[start + column];
    }
    mean /= float(params.width);
    float variance = 0.0f;
    for (uint column = 0; column < params.width; ++column) {
        const float centered = input[start + column] - mean;
        variance += centered * centered;
    }
    const float inverse_std = rsqrt(variance / float(params.width) + params.epsilon);
    const uint modulation_row = target_mask[row] != 0 ? 0 : 1;
    const uint modulation_start = (modulation_row * 4 + params.slot) * params.width;
    for (uint column = 0; column < params.width; ++column) {
        output[start + column] =
            (input[start + column] - mean) * inverse_std * (1.0f + modulation[modulation_start + column]);
    }
}

// One thread owns one [token, head]. The learned RMSNorm scale is shared by heads.
kernel void qi_block_qk_norm_rope(
    device const float *input [[buffer(0)]],
    device const ushort *norm_weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *rope [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.heads;
    if (index >= count) {
        return;
    }
    const uint row = index / params.heads;
    const uint head = index % params.heads;
    const uint start = (row * params.heads + head) * params.head_dimension;
    float mean_square = 0.0f;
    for (uint column = 0; column < params.head_dimension; ++column) {
        const float value = input[start + column];
        mean_square += value * value;
    }
    const float inverse_rms = rsqrt(mean_square / float(params.head_dimension) + params.epsilon);
    const uint complex_count = params.head_dimension / 2;
    const uint rope_start = row * complex_count * 2;
    for (uint pair = 0; pair < complex_count; ++pair) {
        const uint real_column = pair * 2;
        const uint imaginary_column = real_column + 1;
        const float real = input[start + real_column] * inverse_rms * qi_bf16(norm_weight[real_column]);
        const float imaginary =
            input[start + imaginary_column] * inverse_rms * qi_bf16(norm_weight[imaginary_column]);
        const float cosine = rope[rope_start + pair];
        const float sine = rope[rope_start + complex_count + pair];
        output[start + real_column] = real * cosine - imaginary * sine;
        output[start + imaginary_column] = real * sine + imaginary * cosine;
    }
}

// Correctness path: segmented/block-causal scaled dot-product attention without
// materializing the score matrix. One thread owns one [query, head].
kernel void qi_block_attention(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *value [[buffer(3)]],
    device const int *image_ids [[buffer(5)]],
    device const uchar *key_valid [[buffer(6)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.heads;
    if (index >= count) {
        return;
    }
    const uint query_row = index / params.heads;
    const uint head = index % params.heads;
    const uint query_start = (query_row * params.heads + head) * params.head_dimension;
    const float scale = rsqrt(float(params.head_dimension));

    float maximum = -INFINITY;
    for (uint key_row = 0; key_row < params.rows; ++key_row) {
        const bool same_image = image_ids[query_row] >= 0 && image_ids[query_row] == image_ids[key_row];
        if (key_valid[key_row] == 0 || !(query_row >= key_row || same_image)) {
            continue;
        }
        const uint key_start = (key_row * params.heads + head) * params.head_dimension;
        float score = 0.0f;
        for (uint column = 0; column < params.head_dimension; ++column) {
            score = fma(query[query_start + column], key[key_start + column], score);
        }
        maximum = max(maximum, score * scale);
    }

    float denominator = 0.0f;
    for (uint key_row = 0; key_row < params.rows; ++key_row) {
        const bool same_image = image_ids[query_row] >= 0 && image_ids[query_row] == image_ids[key_row];
        if (key_valid[key_row] == 0 || !(query_row >= key_row || same_image)) {
            continue;
        }
        const uint key_start = (key_row * params.heads + head) * params.head_dimension;
        float score = 0.0f;
        for (uint column = 0; column < params.head_dimension; ++column) {
            score = fma(query[query_start + column], key[key_start + column], score);
        }
        denominator += exp(score * scale - maximum);
    }

    for (uint column = 0; column < params.head_dimension; ++column) {
        float sum = 0.0f;
        for (uint key_row = 0; key_row < params.rows; ++key_row) {
            const bool same_image = image_ids[query_row] >= 0 && image_ids[query_row] == image_ids[key_row];
            if (key_valid[key_row] == 0 || !(query_row >= key_row || same_image)) {
                continue;
            }
            const uint key_start = (key_row * params.heads + head) * params.head_dimension;
            float score = 0.0f;
            for (uint inner = 0; inner < params.head_dimension; ++inner) {
                score = fma(query[query_start + inner], key[key_start + inner], score);
            }
            const float probability = exp(score * scale - maximum) / denominator;
            const uint value_index = (key_row * params.heads + head) * params.head_dimension + column;
            sum = fma(probability, value[value_index], sum);
        }
        output[query_start + column] = sum;
    }
}

kernel void qi_block_residual(
    device const float *input [[buffer(0)]],
    device const float *modulation [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *branch [[buffer(3)]],
    device const uchar *target_mask [[buffer(5)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) {
        return;
    }
    const uint row = index / params.width;
    const uint column = index % params.width;
    const uint modulation_row = target_mask[row] != 0 ? 0 : 1;
    const uint gate_start = (modulation_row * 4 + params.slot) * params.width;
    output[index] = input[index] + tanh(modulation[gate_start + column]) * branch[index];
}

kernel void qi_block_swiglu(
    device const float *gate [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width * 3;
    if (index >= count) {
        return;
    }
    const float value = gate[index];
    output[index] = (value / (1.0f + exp(-value))) * projected[index];
}
