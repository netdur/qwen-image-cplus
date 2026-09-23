#include <metal_stdlib>
using namespace metal;

struct Q4DotParams {
    uint vector_count;
    uint dimension;
};

struct Q4GemmParams {
    uint rows;
    uint output_columns;
    uint input_columns;
    uint reserved;
};

inline short qi_q4(ushort packed, uint shift) {
    const ushort nibble = (packed >> shift) & 0xFu;
    return short((nibble ^ 0x8u) - 0x8u);
}

// Streaming reference: one FP16 weight per coordinate. One SIMD group owns
// one independent dot product, matching a moving-weight/GEMV workload.
kernel void qi_fp16_streaming_dot(
    device const half *query [[buffer(0)]],
    device const half *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4DotParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint vector_index [[threadgroup_position_in_grid]]) {
    if (vector_index >= params.vector_count) return;
    device const half *row = weights + vector_index * params.dimension;
    float partial = 0.0f;
    for (uint index = lane; index < params.dimension; index += 32) {
        partial += float(query[index] * row[index]);
    }
    const float sum = simd_sum(partial);
    if (lane == 0) output[vector_index] = sum;
}

// Four signed Q4 weights are loaded in one ushort. This variant expresses the
// four products as one half4 dot operation, then accumulates the partial dot
// in FP32. The per-row scale is applied once after the complete reduction.
kernel void qi_q4_streaming_dot_half4(
    device const half *query [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4DotParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint vector_index [[threadgroup_position_in_grid]]) {
    if (vector_index >= params.vector_count) return;
    const uint words_per_row = params.dimension / 4;
    device const ushort *row = weights + vector_index * words_per_row;
    float partial = 0.0f;
    for (uint word_index = lane; word_index < words_per_row; word_index += 32) {
        const ushort packed = row[word_index];
        const half4 q = half4(
            half(qi_q4(packed, 0)),
            half(qi_q4(packed, 4)),
            half(qi_q4(packed, 8)),
            half(qi_q4(packed, 12))
        );
        const uint input_index = word_index * 4;
        const half4 x = half4(
            query[input_index], query[input_index + 1],
            query[input_index + 2], query[input_index + 3]
        );
        partial += float(dot(x, q));
    }
    const float sum = simd_sum(partial);
    if (lane == 0) output[vector_index] = sum * float(scales[vector_index]);
}

// Fallback control: identical packed storage, but four explicit half products.
// It still performs no persistent or global-memory dequantization.
kernel void qi_q4_streaming_dot_scalar(
    device const half *query [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4DotParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint vector_index [[threadgroup_position_in_grid]]) {
    if (vector_index >= params.vector_count) return;
    const uint words_per_row = params.dimension / 4;
    device const ushort *row = weights + vector_index * words_per_row;
    float partial = 0.0f;
    for (uint word_index = lane; word_index < words_per_row; word_index += 32) {
        const ushort packed = row[word_index];
        const uint input_index = word_index * 4;
        partial += float(query[input_index] * half(qi_q4(packed, 0)));
        partial += float(query[input_index + 1] * half(qi_q4(packed, 4)));
        partial += float(query[input_index + 2] * half(qi_q4(packed, 8)));
        partial += float(query[input_index + 3] * half(qi_q4(packed, 12)));
    }
    const float sum = simd_sum(partial);
    if (lane == 0) output[vector_index] = sum * float(scales[vector_index]);
}

// Production-shape FP16 control. A 64x64 output tile reuses each 32-wide
// weight tile across 64 activation rows and feeds Apple's half SIMD-group MMA.
kernel void qi_fp16_gemm_64x64(
    device const half *input [[buffer(0)]],
    device const half *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
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
                input[(row_base + local_row) * params.input_columns + input_base + local_input];
        }
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_output = linear / 32;
            const uint local_input = linear % 32;
            weight_tile[local_input][local_output] =
                weights[(output_base + local_output) * params.input_columns
                    + input_base + local_input];
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

// Destination-expansion control. Packed nibbles cross unified memory; each
// 32x64 weight tile is expanded once into threadgroup half, reused by 64 rows,
// and immediately consumed by the same half SIMD-group MMA as the FP16 path.
// The row scale is applied once to each final output, never to every weight.
kernel void qi_q4_gemm_64x64(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][64];
    threadgroup float output_tile[64][64];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 64;
    const uint simd_row = simd_index / 8;
    const uint simd_column = simd_index % 8;
    const uint packed_columns = params.input_columns / 2;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            input_tile[local_row][local_input] =
                input[(row_base + local_row) * params.input_columns + input_base + local_input];
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 16;
            const uint local_pair = linear % 16;
            const uint output_column_index = output_base + local_output;
            const uint local_input = local_pair * 2;
            const uint packed_index = output_column_index * packed_columns
                + input_base / 2 + local_pair;
            const ushort packed = ushort(weights[packed_index]);
            weight_tile[local_input][local_output] = half(qi_q4(packed, 0));
            weight_tile[local_input + 1][local_output] = half(qi_q4(packed, 4));
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

    simdgroup_store(
        accumulator_0, &output_tile[simd_row * 8][simd_column * 8], 64);
    simdgroup_store(
        accumulator_1, &output_tile[16 + simd_row * 8][simd_column * 8], 64);
    simdgroup_store(
        accumulator_2, &output_tile[32 + simd_row * 8][simd_column * 8], 64);
    simdgroup_store(
        accumulator_3, &output_tile[48 + simd_row * 8][simd_column * 8], 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = thread_index; linear < 4096; linear += 512) {
        const uint local_row = linear / 64;
        const uint local_output = linear % 64;
        output[(row_base + local_row) * params.output_columns + output_base + local_output] =
            output_tile[local_row][local_output] * float(scales[output_base + local_output]);
    }
}

// Cost-isolation variant: identical Q4 tile expansion and half MMA, but writes
// the unscaled integer-code dot directly. Production can recover this path by
// fusing each output-column scale into the following norm, activation, RoPE,
// or residual operation instead of staging a complete output tile here.
kernel void qi_q4_gemm_64x64_unscaled_direct(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup half input_tile[64][32];
    threadgroup half weight_tile[32][64];

    const uint row_base = group_position.y * 64;
    const uint output_base = group_position.x * 64;
    const uint simd_row = simd_index / 8;
    const uint simd_column = simd_index % 8;
    const uint packed_columns = params.input_columns / 2;
    simdgroup_float8x8 accumulator_0(0.0f);
    simdgroup_float8x8 accumulator_1(0.0f);
    simdgroup_float8x8 accumulator_2(0.0f);
    simdgroup_float8x8 accumulator_3(0.0f);

    for (uint input_base = 0; input_base < params.input_columns; input_base += 32) {
        for (uint linear = thread_index; linear < 2048; linear += 512) {
            const uint local_row = linear / 32;
            const uint local_input = linear % 32;
            input_tile[local_row][local_input] =
                input[(row_base + local_row) * params.input_columns + input_base + local_input];
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 16;
            const uint local_pair = linear % 16;
            const uint output_column_index = output_base + local_output;
            const uint local_input = local_pair * 2;
            const uint packed_index = output_column_index * packed_columns
                + input_base / 2 + local_pair;
            const ushort packed = ushort(weights[packed_index]);
            weight_tile[local_input][local_output] = half(qi_q4(packed, 0));
            weight_tile[local_input + 1][local_output] = half(qi_q4(packed, 4));
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

// Standalone upper-bound cost for destination scaling. Real transformer
// integration can fuse this multiply into the next consumer, but sequencing
// it after the direct Q4 GEMM measures complete dequantization semantics.
kernel void qi_q4_scale_output(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
    if (position.x >= params.output_columns || position.y >= params.rows) return;
    const uint index = position.y * params.output_columns + position.x;
    output[index] *= float(scales[position.x]);
}

// Primary direct-Q4 experiment. Weights stay packed until four signed nibbles
// enter one half4 dot expression; there is no FP16 weight tile and no matrix
// MMA. The offline layout is [K/4, N], so the 32 SIMD lanes read adjacent
// packed words. Each lane owns one output column and reuses its word across
// four activation rows with independent accumulators.
kernel void qi_q4_direct_half4_gemm_4x32_interleaved(
    device const half *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    const uint output_column = group_position.x * 32 + lane;
    const uint row_base = group_position.y * 4;
    const uint words = params.input_columns / 4;
    float4 accumulator = float4(0.0f);
    for (uint word_index = 0; word_index < words; ++word_index) {
        const ushort packed = weights[word_index * params.output_columns + output_column];
        const half4 q = half4(
            half(qi_q4(packed, 0)),
            half(qi_q4(packed, 4)),
            half(qi_q4(packed, 8)),
            half(qi_q4(packed, 12))
        );
        const uint input_column = word_index * 4;
        for (uint local_row = 0; local_row < 4; ++local_row) {
            device const half *x = input
                + (row_base + local_row) * params.input_columns + input_column;
            accumulator[local_row] += float(dot(half4(x[0], x[1], x[2], x[3]), q));
        }
    }
    const float scale = float(scales[output_column]);
    for (uint local_row = 0; local_row < 4; ++local_row) {
        output[(row_base + local_row) * params.output_columns + output_column] =
            accumulator[local_row] * scale;
    }
}

kernel void qi_q4_direct_half4_gemm_8x32_interleaved(
    device const half *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    const uint output_column = group_position.x * 32 + lane;
    const uint row_base = group_position.y * 8;
    const uint words = params.input_columns / 4;
    float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint word_index = 0; word_index < words; ++word_index) {
        const ushort packed = weights[word_index * params.output_columns + output_column];
        const half4 q = half4(
            half(qi_q4(packed, 0)),
            half(qi_q4(packed, 4)),
            half(qi_q4(packed, 8)),
            half(qi_q4(packed, 12))
        );
        const uint input_column = word_index * 4;
        for (uint local_row = 0; local_row < 8; ++local_row) {
            device const half *x = input
                + (row_base + local_row) * params.input_columns + input_column;
            accumulator[local_row] += float(dot(half4(x[0], x[1], x[2], x[3]), q));
        }
    }
    const float scale = float(scales[output_column]);
    for (uint local_row = 0; local_row < 8; ++local_row) {
        output[(row_base + local_row) * params.output_columns + output_column] =
            accumulator[local_row] * scale;
    }
}

kernel void qi_q4_direct_half4_gemm_16x32_interleaved(
    device const half *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Q4GemmParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    const uint output_column = group_position.x * 32 + lane;
    const uint row_base = group_position.y * 16;
    const uint words = params.input_columns / 4;
    float accumulator[16] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (uint word_index = 0; word_index < words; ++word_index) {
        const ushort packed = weights[word_index * params.output_columns + output_column];
        const half4 q = half4(
            half(qi_q4(packed, 0)),
            half(qi_q4(packed, 4)),
            half(qi_q4(packed, 8)),
            half(qi_q4(packed, 12))
        );
        const uint input_column = word_index * 4;
        for (uint local_row = 0; local_row < 16; ++local_row) {
            device const half *x = input
                + (row_base + local_row) * params.input_columns + input_column;
            accumulator[local_row] += float(dot(half4(x[0], x[1], x[2], x[3]), q));
        }
    }
    const float scale = float(scales[output_column]);
    for (uint local_row = 0; local_row < 16; ++local_row) {
        output[(row_base + local_row) * params.output_columns + output_column] =
            accumulator[local_row] * scale;
    }
}
