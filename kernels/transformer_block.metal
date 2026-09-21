#include <metal_stdlib>
using namespace metal;

constant uint QI_K_TILE = 16;
constant uint QI_OUTPUT_TILE = 32;

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

struct AttentionParams {
    uint query_rows;
    uint key_rows;
    uint heads;
    uint head_dimension;
    uint query_position_offset;
    uint block_causal;
    uint cached_prefix_rows;
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

kernel void qi_time_projection(
    device const float *timestep [[buffer(0)]],
    device float *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) {
        return;
    }
    const uint row = index / params.width;
    const uint column = index % params.width;
    const uint half_width = params.width / 2;
    const uint frequency_column = column % half_width;
    const float frequency = exp(-log(10000.0f) * float(frequency_column) / float(half_width));
    const float argument = timestep[row] * 1000.0f * frequency;
    output[index] = column < half_width ? cos(argument) : sin(argument);
}

kernel void qi_silu(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        const float value = input[index];
        output[index] = value / (1.0f + exp(-value));
    }
}

kernel void qi_gelu_tanh(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        const float value = input[index];
        output[index] = 0.5f * value *
            (1.0f + tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));
    }
}

kernel void qi_zero_center_rms_norm(
    device const float *input [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    const uint start = row * params.width;
    float mean_square = 0.0f;
    for (uint column = 0; column < params.width; ++column) {
        mean_square = fma(input[start + column], input[start + column], mean_square);
    }
    const float inverse_rms = rsqrt(mean_square / float(params.width) + params.epsilon);
    for (uint column = 0; column < params.width; ++column) {
        output[start + column] =
            input[start + column] * inverse_rms * (1.0f + qi_bf16(weight[column]));
    }
}

// source_index >= 0 selects an image row. Negative values encode text row as -1-row.
kernel void qi_joint_assemble(
    device const float *text [[buffer(0)]],
    device const float *image [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const int *source_index [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) {
        return;
    }
    const uint row = index / params.width;
    const uint column = index % params.width;
    const int source = source_index[row];
    output[index] = source >= 0
        ? image[uint(source) * params.width + column]
        : text[uint(-source - 1) * params.width + column];
}

kernel void qi_final_layernorm_modulate(
    device const float *input [[buffer(0)]],
    device const float *scale [[buffer(1)]],
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
        variance = fma(centered, centered, variance);
    }
    const float inverse_std = rsqrt(variance / float(params.width) + params.epsilon);
    const uint scale_start = (target_mask[row] != 0 ? 0 : 1) * params.width;
    for (uint column = 0; column < params.width; ++column) {
        output[start + column] =
            (input[start + column] - mean) * inverse_std * (1.0f + scale[scale_start + column]);
    }
}

kernel void qi_copy_prefix_kv(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device float *cache_key [[buffer(2)]],
    device float *cache_value [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        // K reaches this point after learned RMSNorm and RoPE; V is raw.
        cache_key[index] = key[index];
        cache_value[index] = value[index];
    }
}

// One 16x16 threadgroup computes a 32x32 output tile. Each thread owns four
// accumulators, which quadruples useful arithmetic per barrier and reuses each
// input/weight value across twice as many output positions as the original
// 16x16 kernel. K stays tiled by 16 so the accumulation order—and therefore
// the established numerical envelope—does not change.
kernel void qi_block_linear_32x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[32][16];
    threadgroup float weight_tile[16][32];

    const uint row0 = group_position.y * QI_OUTPUT_TILE + thread_position.y;
    const uint row1 = row0 + 16;
    const uint output_column0 = group_position.x * QI_OUTPUT_TILE + thread_position.x;
    const uint output_column1 = output_column0 + 16;
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    for (uint input_base = 0; input_base < params.input_columns; input_base += QI_K_TILE) {
        const uint input_column = input_base + thread_position.x;
        input_tile[thread_position.y][thread_position.x] =
            row0 < params.rows && input_column < params.input_columns
                ? input[row0 * params.input_columns + input_column]
                : 0.0f;
        input_tile[thread_position.y + 16][thread_position.x] =
            row1 < params.rows && input_column < params.input_columns
                ? input[row1 * params.input_columns + input_column]
                : 0.0f;
        const uint weight_input_column = input_base + thread_position.y;
        weight_tile[thread_position.y][thread_position.x] =
            output_column0 < params.output_columns && weight_input_column < params.input_columns
                ? qi_bf16(weights[output_column0 * params.input_columns + weight_input_column])
                : 0.0f;
        weight_tile[thread_position.y][thread_position.x + 16] =
            output_column1 < params.output_columns && weight_input_column < params.input_columns
                ? qi_bf16(weights[output_column1 * params.input_columns + weight_input_column])
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < QI_K_TILE; ++inner) {
            sum00 = fma(input_tile[thread_position.y][inner], weight_tile[inner][thread_position.x], sum00);
            sum01 = fma(input_tile[thread_position.y][inner], weight_tile[inner][thread_position.x + 16], sum01);
            sum10 = fma(input_tile[thread_position.y + 16][inner], weight_tile[inner][thread_position.x], sum10);
            sum11 = fma(input_tile[thread_position.y + 16][inner], weight_tile[inner][thread_position.x + 16], sum11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < params.rows && output_column0 < params.output_columns) {
        output[row0 * params.output_columns + output_column0] = sum00;
    }
    if (row0 < params.rows && output_column1 < params.output_columns) {
        output[row0 * params.output_columns + output_column1] = sum01;
    }
    if (row1 < params.rows && output_column0 < params.output_columns) {
        output[row1 * params.output_columns + output_column0] = sum10;
    }
    if (row1 < params.rows && output_column1 < params.output_columns) {
        output[row1 * params.output_columns + output_column1] = sum11;
    }
}

kernel void qi_block_int8_affine_linear_32x32(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uchar *zeros [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Int8LinearParams &params [[buffer(5)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[32][16];
    threadgroup half weight_tile[16][32];

    const uint row0 = group_position.y * QI_OUTPUT_TILE + thread_position.y;
    const uint row1 = row0 + 16;
    const uint output_column0 = group_position.x * QI_OUTPUT_TILE + thread_position.x;
    const uint output_column1 = output_column0 + 16;
    const uint groups_per_row = params.input_columns / params.group_size;
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    for (uint input_base = 0; input_base < params.input_columns; input_base += QI_K_TILE) {
        const uint input_column = input_base + thread_position.x;
        input_tile[thread_position.y][thread_position.x] =
            row0 < params.rows && input_column < params.input_columns
                ? half(input[row0 * params.input_columns + input_column])
                : half(0.0h);
        input_tile[thread_position.y + 16][thread_position.x] =
            row1 < params.rows && input_column < params.input_columns
                ? half(input[row1 * params.input_columns + input_column])
                : half(0.0h);
        const uint weight_input_column = input_base + thread_position.y;
        if (output_column0 < params.output_columns && weight_input_column < params.input_columns) {
            const uint weight_index = output_column0 * params.input_columns + weight_input_column;
            const uint group_index =
                output_column0 * groups_per_row + weight_input_column / params.group_size;
            weight_tile[thread_position.y][thread_position.x] =
                half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
        } else {
            weight_tile[thread_position.y][thread_position.x] = half(0.0h);
        }
        if (output_column1 < params.output_columns && weight_input_column < params.input_columns) {
            const uint weight_index = output_column1 * params.input_columns + weight_input_column;
            const uint group_index =
                output_column1 * groups_per_row + weight_input_column / params.group_size;
            weight_tile[thread_position.y][thread_position.x + 16] =
                half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
        } else {
            weight_tile[thread_position.y][thread_position.x + 16] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < QI_K_TILE; ++inner) {
            sum00 += float(input_tile[thread_position.y][inner] * weight_tile[inner][thread_position.x]);
            sum01 += float(input_tile[thread_position.y][inner] * weight_tile[inner][thread_position.x + 16]);
            sum10 += float(input_tile[thread_position.y + 16][inner] * weight_tile[inner][thread_position.x]);
            sum11 += float(input_tile[thread_position.y + 16][inner] * weight_tile[inner][thread_position.x + 16]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < params.rows && output_column0 < params.output_columns) {
        output[row0 * params.output_columns + output_column0] = sum00;
    }
    if (row0 < params.rows && output_column1 < params.output_columns) {
        output[row0 * params.output_columns + output_column1] = sum01;
    }
    if (row1 < params.rows && output_column0 < params.output_columns) {
        output[row1 * params.output_columns + output_column0] = sum10;
    }
    if (row1 < params.rows && output_column1 < params.output_columns) {
        output[row1 * params.output_columns + output_column1] = sum11;
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

// One 128-thread group owns one [query, head]. Each Q.K score is reduced once,
// then every lane updates one output channel with online softmax. Temporary
// storage is constant per group and no [heads,S,S] score matrix is formed.
kernel void qi_block_attention(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *value [[buffer(3)]],
    device const int *image_ids [[buffer(5)]],
    device const uchar *key_valid [[buffer(6)]],
    device const float *cache_key [[buffer(7)]],
    device const float *cache_value [[buffer(8)]],
    constant AttentionParams &params [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[128];
    threadgroup float state[4]; // maximum, denominator, old scale, new weight

    const uint query_row = group / params.heads;
    const uint head = group % params.heads;
    if (query_row >= params.query_rows || lane >= params.head_dimension) {
        return;
    }
    const uint global_query_row = query_row + params.query_position_offset;
    const uint query_start = (query_row * params.heads + head) * params.head_dimension;
    const float scale = rsqrt(float(params.head_dimension));
    if (lane == 0) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0f;
    for (uint key_row = 0; key_row < params.key_rows; ++key_row) {
        const bool same_image = image_ids[global_query_row] >= 0 &&
            image_ids[global_query_row] == image_ids[key_row];
        const bool allowed = key_valid[key_row] != 0 &&
            (params.block_causal == 0 || global_query_row >= key_row || same_image);
        if (!allowed) {
            continue;
        }
        const bool from_cache = key_row < params.cached_prefix_rows;
        const uint local_key_row = from_cache ? key_row : key_row - params.cached_prefix_rows;
        const uint key_start = (local_key_row * params.heads + head) * params.head_dimension;
        const float key_element = from_cache ? cache_key[key_start + lane] : key[key_start + lane];
        reduction[lane] = query[query_start + lane] * key_element;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = params.head_dimension / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float score = reduction[0] * scale;
            const float next_maximum = max(state[0], score);
            const float old_scale = isinf(state[0]) ? 0.0f : exp(state[0] - next_maximum);
            const float new_weight = exp(score - next_maximum);
            state[0] = next_maximum;
            state[1] = state[1] * old_scale + new_weight;
            state[2] = old_scale;
            state[3] = new_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint value_index = key_start + lane;
        const float value_element = from_cache ? cache_value[value_index] : value[value_index];
        accumulator = fma(accumulator, state[2], state[3] * value_element);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[query_start + lane] = accumulator / state[1];
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
