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

inline float qi_dense_weight(ushort bits, uint weight_mode) {
    return weight_mode == 1 ? float(as_type<half>(bits)) : qi_bf16(bits);
}

kernel void qi_mps_input_to_half(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant LinearParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.input_columns;
    if (index < count) {
        output[index] = half(input[index]);
    }
}

kernel void qi_mps_bf16_weight_to_half(
    device const ushort *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant LinearParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.output_columns * params.input_columns;
    if (index < count) {
        output[index] = half(qi_bf16(input[index]));
    }
}

kernel void qi_mps_q8_weight_to_half(
    device const uchar *weights [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const uchar *zeros [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant Int8LinearParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.output_columns * params.input_columns;
    if (index >= count) {
        return;
    }
    const uint output_column = index / params.input_columns;
    const uint input_column = index % params.input_columns;
    const uint groups_per_row = params.input_columns / params.group_size;
    const uint group_index =
        output_column * groups_per_row + input_column / params.group_size;
    output[index] =
        half(int(weights[index]) - int(zeros[group_index])) * scales[group_index];
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
                ? qi_dense_weight(weights[output_column0 * params.input_columns + weight_input_column], params.weight_mode)
                : 0.0f;
        weight_tile[thread_position.y][thread_position.x + 16] =
            output_column1 < params.output_columns && weight_input_column < params.input_columns
                ? qi_dense_weight(weights[output_column1 * params.input_columns + weight_input_column], params.weight_mode)
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

// Prototype cooperative-matrix path. Sixteen SIMD groups share coalesced
// 32x32 input and transposed-weight tiles, and each SIMD group owns one 8x8
// quadrant of the 32x32 result. The final shared-memory store permits masked
// writes for the 278-row prefill without an out-of-bounds matrix store.
kernel void qi_block_linear_simdgroup_32x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[32][32];
    threadgroup float weight_tile[32][32];
    threadgroup float output_tile[32][32];

    const uint row_base = group_position.y * 32;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    simdgroup_float8x8 accumulator(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? input[input_row * params.input_columns + input_column]
                    : 0.0f;

            const uint output_column = output_base + local_row;
            weight_tile[local_input][local_row] =
                output_column < params.output_columns && input_column < params.input_columns
                    ? qi_dense_weight(weights[output_column * params.input_columns + input_column], params.weight_mode)
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_float8x8 left;
            simdgroup_float8x8 right;
            simdgroup_load(
                left, &input_tile[simd_row * 8][inner], 32
            );
            simdgroup_load(
                right, &weight_tile[inner][simd_column * 8], 32
            );
            simdgroup_multiply_accumulate(accumulator, left, right, accumulator);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator, &output_tile[simd_row * 8][simd_column * 8], 32
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 1024; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
    }
}

kernel void qi_block_linear_simdgroup_half_32x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[32][32];
    threadgroup half weight_tile[32][32];
    threadgroup float output_tile[32][32];

    const uint row_base = group_position.y * 32;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    simdgroup_float8x8 accumulator(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? half(input[input_row * params.input_columns + input_column])
                    : half(0.0h);

            const uint output_column = output_base + local_row;
            weight_tile[local_input][local_row] =
                output_column < params.output_columns && input_column < params.input_columns
                    ? half(qi_dense_weight(weights[output_column * params.input_columns + input_column], params.weight_mode))
                    : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left;
            simdgroup_half8x8 right;
            simdgroup_load(left, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator, left, right, accumulator);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator, &output_tile[simd_row * 8][simd_column * 8], 32
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 1024; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
    }
}

kernel void qi_block_linear_simdgroup_half_64x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][32];
    threadgroup float output_tile[64][32];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? half(input[input_row * params.input_columns + input_column])
                    : half(0.0h);
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            const uint output_column = output_base + local_output;
            const uint input_column = input_base + local_input;
            weight_tile[local_input][local_output] =
                output_column < params.output_columns && input_column < params.input_columns
                    ? half(qi_dense_weight(weights[output_column * params.input_columns + input_column], params.weight_mode))
                    : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 32
    );
    simdgroup_store(
        accumulator_1, &output_tile[32 + simd_row * 8][simd_column * 8], 32
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 2048; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
    }
}

kernel void qi_block_int8_simdgroup_half_64x32(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uchar *zeros [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Int8LinearParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][32];
    threadgroup float output_tile[64][32];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    const uint groups_per_row = params.input_columns / params.group_size;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? half(input[input_row * params.input_columns + input_column])
                    : half(0.0h);
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            const uint output_column = output_base + local_output;
            const uint input_column = input_base + local_input;
            if (output_column < params.output_columns && input_column < params.input_columns) {
                const uint weight_index = output_column * params.input_columns + input_column;
                const uint group_index =
                    output_column * groups_per_row + input_column / params.group_size;
                weight_tile[local_input][local_output] =
                    half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
            } else {
                weight_tile[local_input][local_output] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 32
    );
    simdgroup_store(
        accumulator_1, &output_tile[32 + simd_row * 8][simd_column * 8], 32
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 2048; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
    }
}

kernel void qi_block_linear_simdgroup_half_64x32_direct(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][32];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 256) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            input_tile[local_row][local_input] =
                half(input[(row_base + local_row) * params.input_columns + input_base + local_input]);
        }
        for (uint linear = thread_index; linear < 1024; linear += 256) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            weight_tile[local_input][local_output] = half(qi_dense_weight(
                weights[(output_base + local_output) * params.input_columns
                    + input_base + local_input], params.weight_mode
            ));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 left_2;
            simdgroup_half8x8 left_3;
            simdgroup_half8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[16 + simd_row * 8][inner], 32);
            simdgroup_load(left_2, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(left_3, &input_tile[48 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, left_2, right, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, left_3, right, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint output_column = output_base + simd_column * 8;
    simdgroup_store(
        accumulator_0, output + (row_base + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
    simdgroup_store(
        accumulator_1, output + (row_base + 16 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
    simdgroup_store(
        accumulator_2, output + (row_base + 32 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
    simdgroup_store(
        accumulator_3, output + (row_base + 48 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
}

kernel void qi_block_int8_simdgroup_half_64x32_direct(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uchar *zeros [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Int8LinearParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][32];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    const uint groups_per_row = params.input_columns / params.group_size;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 256) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            input_tile[local_row][local_input] =
                half(input[(row_base + local_row) * params.input_columns + input_base + local_input]);
        }
        for (uint linear = thread_index; linear < 1024; linear += 256) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            const uint output_column = output_base + local_output;
            const uint input_column = input_base + local_input;
            const uint weight_index = output_column * params.input_columns + input_column;
            const uint group_index =
                output_column * groups_per_row + input_column / params.group_size;
            weight_tile[local_input][local_output] =
                half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 left_2;
            simdgroup_half8x8 left_3;
            simdgroup_half8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[16 + simd_row * 8][inner], 32);
            simdgroup_load(left_2, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(left_3, &input_tile[48 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, left_2, right, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, left_3, right, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint output_column = output_base + simd_column * 8;
    simdgroup_store(
        accumulator_0, output + (row_base + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
    simdgroup_store(
        accumulator_1, output + (row_base + 16 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
    simdgroup_store(
        accumulator_2, output + (row_base + 32 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
    simdgroup_store(
        accumulator_3, output + (row_base + 48 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns
    );
}

kernel void qi_block_linear_simdgroup_half_64x64_direct(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][64];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 64;
    const uint simd_row = simd_index / 8;
    const uint simd_column = simd_index % 8;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            input_tile[local_row][local_input] =
                half(input[(row_base + local_row) * params.input_columns + input_base + local_input]);
        }
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            weight_tile[local_input][local_output] = half(qi_dense_weight(
                weights[(output_base + local_output) * params.input_columns
                    + input_base + local_input], params.weight_mode
            ));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 left_2;
            simdgroup_half8x8 left_3;
            simdgroup_half8x8 right;
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

    const uint output_column = output_base + simd_column * 8;
    simdgroup_store(accumulator_0,
        output + (row_base + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
    simdgroup_store(accumulator_1,
        output + (row_base + 16 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
    simdgroup_store(accumulator_2,
        output + (row_base + 32 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
    simdgroup_store(accumulator_3,
        output + (row_base + 48 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
}

kernel void qi_block_int8_simdgroup_half_64x64_direct(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uchar *zeros [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Int8LinearParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][64];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 64;
    const uint simd_row = simd_index / 8;
    const uint simd_column = simd_index % 8;
    const uint groups_per_row = params.input_columns / params.group_size;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            input_tile[local_row][local_input] =
                half(input[(row_base + local_row) * params.input_columns + input_base + local_input]);
        }
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            const uint output_column = output_base + local_output;
            const uint input_column = input_base + local_input;
            const uint weight_index = output_column * params.input_columns + input_column;
            const uint group_index =
                output_column * groups_per_row + input_column / params.group_size;
            weight_tile[local_input][local_output] =
                half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 left_2;
            simdgroup_half8x8 left_3;
            simdgroup_half8x8 right;
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

    const uint output_column = output_base + simd_column * 8;
    simdgroup_store(accumulator_0,
        output + (row_base + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
    simdgroup_store(accumulator_1,
        output + (row_base + 16 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
    simdgroup_store(accumulator_2,
        output + (row_base + 32 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
    simdgroup_store(accumulator_3,
        output + (row_base + 48 + simd_row * 8) * params.output_columns + output_column,
        params.output_columns);
}

kernel void qi_block_linear_simdgroup_half_128x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[128][32];
    threadgroup half weight_tile[32][32];
    threadgroup float output_tile[128][32];

    const uint row_base = group_position.y * 128;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 4096; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? half(input[input_row * params.input_columns + input_column])
                    : half(0.0h);
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            const uint output_column = output_base + local_output;
            const uint input_column = input_base + local_input;
            weight_tile[local_input][local_output] =
                output_column < params.output_columns && input_column < params.input_columns
                    ? half(qi_dense_weight(weights[output_column * params.input_columns + input_column], params.weight_mode))
                    : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 left_2;
            simdgroup_half8x8 left_3;
            simdgroup_half8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(left_2, &input_tile[64 + simd_row * 8][inner], 32);
            simdgroup_load(left_3, &input_tile[96 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, left_2, right, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, left_3, right, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 32);
    simdgroup_store(accumulator_1, &output_tile[32 + simd_row * 8][simd_column * 8], 32);
    simdgroup_store(accumulator_2, &output_tile[64 + simd_row * 8][simd_column * 8], 32);
    simdgroup_store(accumulator_3, &output_tile[96 + simd_row * 8][simd_column * 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 4096; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
    }
}

kernel void qi_block_int8_simdgroup_half_128x32(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const uchar *zeros [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Int8LinearParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[128][32];
    threadgroup half weight_tile[32][32];
    threadgroup float output_tile[128][32];

    const uint row_base = group_position.y * 128;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    const uint groups_per_row = params.input_columns / params.group_size;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 4096; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? half(input[input_row * params.input_columns + input_column])
                    : half(0.0h);
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            const uint output_column = output_base + local_output;
            const uint input_column = input_base + local_input;
            if (output_column < params.output_columns && input_column < params.input_columns) {
                const uint weight_index = output_column * params.input_columns + input_column;
                const uint group_index =
                    output_column * groups_per_row + input_column / params.group_size;
                weight_tile[local_input][local_output] =
                    half(int(weights[weight_index]) - int(zeros[group_index])) * scales[group_index];
            } else {
                weight_tile[local_input][local_output] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_half8x8 left_0;
            simdgroup_half8x8 left_1;
            simdgroup_half8x8 left_2;
            simdgroup_half8x8 left_3;
            simdgroup_half8x8 right;
            simdgroup_load(left_0, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(left_1, &input_tile[32 + simd_row * 8][inner], 32);
            simdgroup_load(left_2, &input_tile[64 + simd_row * 8][inner], 32);
            simdgroup_load(left_3, &input_tile[96 + simd_row * 8][inner], 32);
            simdgroup_load(right, &weight_tile[inner][simd_column * 8], 32);
            simdgroup_multiply_accumulate(accumulator_0, left_0, right, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, left_1, right, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, left_2, right, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, left_3, right, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 32);
    simdgroup_store(accumulator_1, &output_tile[32 + simd_row * 8][simd_column * 8], 32);
    simdgroup_store(accumulator_2, &output_tile[64 + simd_row * 8][simd_column * 8], 32);
    simdgroup_store(accumulator_3, &output_tile[96 + simd_row * 8][simd_column * 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 4096; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
    }
}

kernel void qi_block_linear_simdgroup_bfloat_32x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup bfloat input_tile[32][32];
    threadgroup float output_tile[32][32];
    device const bfloat *bfloat_weights =
        reinterpret_cast<device const bfloat *>(weights);

    const uint row_base = group_position.y * 32;
    const uint output_base = group_position.x * 32;
    const uint simd_row = simd_index / 4;
    const uint simd_column = simd_index % 4;
    simdgroup_float8x8 accumulator(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            const uint input_row = row_base + local_row;
            const uint input_column = input_base + local_input;
            input_tile[local_row][local_input] =
                input_row < params.rows && input_column < params.input_columns
                    ? bfloat(input[input_row * params.input_columns + input_column])
                    : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; inner += 8) {
            simdgroup_bfloat8x8 left;
            simdgroup_bfloat8x8 right;
            simdgroup_load(left, &input_tile[simd_row * 8][inner], 32);
            simdgroup_load(
                right,
                bfloat_weights
                    + (output_base + simd_column * 8) * params.input_columns
                    + input_base + inner,
                params.input_columns,
                ulong2(0),
                true
            );
            simdgroup_multiply_accumulate(accumulator, left, right, accumulator);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(
        accumulator, &output_tile[simd_row * 8][simd_column * 8], 32
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 1024; linear += 512) {
        const uint local_row = linear / 32;
        const uint local_column = linear % 32;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_column;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_column];
        }
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

// One 32-lane SIMD group owns one [query, head]. Each lane holds four head
// channels, so the Q.K reduction uses a hardware SIMD reduction and the
// online-softmax state stays in registers. This removes all threadgroup
// barriers and shared memory from the production 128-channel attention path.
kernel void qi_block_attention_simdgroup(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *value [[buffer(3)]],
    device const int *image_ids [[buffer(5)]],
    device const uchar *key_valid [[buffer(6)]],
    device const float *cache_key [[buffer(7)]],
    device const float *cache_value [[buffer(8)]],
    constant AttentionParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint query_row = group / params.heads;
    const uint head = group % params.heads;
    if (query_row >= params.query_rows) {
        return;
    }
    const uint global_query_row = query_row + params.query_position_offset;
    const uint query_start = (query_row * params.heads + head) * params.head_dimension;
    const uint channel_0 = lane;
    const uint channel_1 = lane + 32;
    const uint channel_2 = lane + 64;
    const uint channel_3 = lane + 96;
    const float query_0 = query[query_start + channel_0];
    const float query_1 = query[query_start + channel_1];
    const float query_2 = query[query_start + channel_2];
    const float query_3 = query[query_start + channel_3];
    const float scale = rsqrt(float(params.head_dimension));
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator_0 = 0.0f;
    float accumulator_1 = 0.0f;
    float accumulator_2 = 0.0f;
    float accumulator_3 = 0.0f;

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
        device const float *key_source = from_cache ? cache_key : key;
        const float partial =
            query_0 * key_source[key_start + channel_0] +
            query_1 * key_source[key_start + channel_1] +
            query_2 * key_source[key_start + channel_2] +
            query_3 * key_source[key_start + channel_3];
        const float score = simd_sum(partial) * scale;
        const float next_maximum = max(maximum, score);
        const float old_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        const float new_weight = exp(score - next_maximum);
        maximum = next_maximum;
        denominator = denominator * old_scale + new_weight;
        device const float *value_source = from_cache ? cache_value : value;
        accumulator_0 = fma(accumulator_0, old_scale,
            new_weight * value_source[key_start + channel_0]);
        accumulator_1 = fma(accumulator_1, old_scale,
            new_weight * value_source[key_start + channel_1]);
        accumulator_2 = fma(accumulator_2, old_scale,
            new_weight * value_source[key_start + channel_2]);
        accumulator_3 = fma(accumulator_3, old_scale,
            new_weight * value_source[key_start + channel_3]);
    }
    output[query_start + channel_0] = accumulator_0 / denominator;
    output[query_start + channel_1] = accumulator_1 / denominator;
    output[query_start + channel_2] = accumulator_2 / denominator;
    output[query_start + channel_3] = accumulator_3 / denominator;
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
