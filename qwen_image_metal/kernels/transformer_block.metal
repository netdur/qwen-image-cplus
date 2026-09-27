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
    uint value_stride;
    uint value_offset;
};

struct FlashAttentionParams {
    uint query_rows;
    uint key_rows;
    uint padded_key_rows;
    uint query_position_offset;
    uint block_causal;
    uint prune_masked_tiles;
};

struct FlashPrepareParams {
    uint rows;
    uint prefix_rows;
    uint padded_rows;
    uint cache_mode;
    uint value_stride;
    uint value_offset;
};

struct BlockParams {
    uint rows;
    uint width;
    uint heads;
    uint head_dimension;
    float epsilon;
    uint slot;
    uint input_stride;
    uint input_offset;
};

inline float qi_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

struct LoraAddParams {
    uint rows;
    uint columns;
    uint output_stride;
    uint output_offset;
};

kernel void qi_lora_bf16_to_half(
    device ushort *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        values[index] = as_type<ushort>(half(qi_bf16(values[index])));
    }
}

kernel void qi_lora_add(
    device const float *delta [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant LoraAddParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.rows * params.columns) {
        const uint row = index / params.columns;
        const uint column = index % params.columns;
        output[row * params.output_stride + params.output_offset + column] += delta[index];
    }
}

// Forms one merged LoRA weight, W' = half(float(W) + B x A), rounding once,
// into a separate buffer; the host copies it into the resident pack.
kernel void qi_lora_merge(
    device const half *weight [[buffer(0)]],
    device const float *delta [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    device half *merged [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        merged[index] = half(float(weight[index]) + delta[index]);
    }
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

// Builds the contiguous padded FP16 K/V tensors consumed by Flash Attention.
// On the joint pass it also saves the text prefix; on target-only passes it
// restores that prefix before appending the current image K/V.
kernel void qi_flash_prepare_kv(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device half *prepared_key [[buffer(2)]],
    device half *prepared_value [[buffer(3)]],
    constant FlashPrepareParams &params [[buffer(4)]],
    device half *cache_key [[buffer(5)]],
    device half *cache_value [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint width = 4096;
    const uint count = params.padded_rows * width;
    if (index >= count) {
        return;
    }
    const uint row = index / width;
    const uint column = index % width;
    if (params.cache_mode == 2 && row < params.prefix_rows) {
        prepared_key[index] = cache_key[index];
        prepared_value[index] = cache_value[index];
        return;
    }
    const uint source_row = params.cache_mode == 2
        ? row - params.prefix_rows : row;
    if (source_row >= params.rows) {
        prepared_key[index] = 0.0h;
        prepared_value[index] = 0.0h;
        return;
    }
    const uint source_index = source_row * width + column;
    const half key_element = half(key[source_index]);
    const half value_element = half(value[
        source_row * params.value_stride + params.value_offset + column]);
    prepared_key[index] = key_element;
    prepared_value[index] = value_element;
    if (params.cache_mode == 1 && row < params.prefix_rows) {
        cache_key[index] = key_element;
        cache_value[index] = value_element;
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

inline short qi_signed_q4(uchar packed, uint shift) {
    const ushort nibble = (ushort(packed) >> shift) & 0xFu;
    return short((nibble ^ 0x8u) - 0x8u);
}

// Startup-only eager expansion. QIPACK stores two signed nibbles per byte and
// one FP16 scale per output row. The destination follows the established v4
// FP16 record offsets, preserving contiguous Q/K/V matrices for MPS.
kernel void qi_q4_weight_to_half(
    device const uchar *weights [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant Int8LinearParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.output_columns * params.input_columns;
    if (index >= count) { return; }
    const uchar packed = weights[index / 2];
    const short code = qi_signed_q4(packed, (index & 1u) * 4u);
    const uint output_column = index / params.input_columns;
    output[index] = half(code) * scales[output_column];
}

// Startup-only packed-layout conversion for the direct Q4 experiment. The
// QIPACK source is row-major [N,K/2]. The direct dot keeps all four weights in
// one ushort and stores [K/4,N], so adjacent SIMD lanes read adjacent output
// columns without ever materializing an FP16 weight.
kernel void qi_q4_repack_direct(
    device const uchar *source [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant Int8LinearParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint words_per_row = params.input_columns / 4;
    const uint count = words_per_row * params.output_columns;
    if (index >= count) { return; }
    const uint word = index / params.output_columns;
    const uint output_column = index % params.output_columns;
    const uint source_byte = output_column * (params.input_columns / 2) + word * 2;
    destination[index] = ushort(source[source_byte])
        | (ushort(source[source_byte + 1]) << 8);
}

// Genuinely direct packed-Q4 dot. Packed nibbles stay packed until each lane's
// half4 dot; there is no threadgroup or global FP16 weight tile. One SIMD group
// owns 32 output columns and reuses each packed word across up to 16 rows.
kernel void qi_block_q4_direct_half4_16x32(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Int8LinearParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_group = 16;
    const uint output_column = group_position.x * 32 + lane;
    const uint row_base = group_position.y * rows_per_group;
    const uint words = params.input_columns / 4;
    float accumulator[rows_per_group];
    for (uint local_row = 0; local_row < rows_per_group; ++local_row) {
        accumulator[local_row] = 0.0f;
    }
    if (output_column < params.output_columns) {
        for (uint word = 0; word < words; ++word) {
            const ushort packed = weights[word * params.output_columns + output_column];
            const half4 q = half4(
                half(qi_signed_q4(uchar(packed), 0)),
                half(qi_signed_q4(uchar(packed), 4)),
                half(qi_signed_q4(uchar(packed >> 8), 0)),
                half(qi_signed_q4(uchar(packed >> 8), 4))
            );
            const uint input_column = word * 4;
            for (uint local_row = 0; local_row < rows_per_group; ++local_row) {
                const uint row = row_base + local_row;
                if (row < params.rows) {
                    device const float *x = input + row * params.input_columns + input_column;
                    accumulator[local_row] += float(dot(
                        half4(half(x[0]), half(x[1]), half(x[2]), half(x[3])), q));
                }
            }
        }
        const float scale = float(scales[output_column]);
        for (uint local_row = 0; local_row < rows_per_group; ++local_row) {
            const uint row = row_base + local_row;
            if (row < params.rows) {
                output[row * params.output_columns + output_column] =
                    accumulator[local_row] * scale;
            }
        }
    }
}

// Inference-time destination expansion. Packed weights remain in the mapped
// QIPACK file; each 32x64 tile is expanded into threadgroup FP16 exactly where
// the native SIMD-group MMA consumes it. Bounds cover the 4127-row prompt pass.
kernel void qi_block_q4_simdgroup_half_64x64(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Int8LinearParams &params [[buffer(4)]],
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
            const uint input_row = row_base + local_row;
            input_tile[local_row][local_input] = input_row < params.rows
                ? half(input[input_row * params.input_columns + input_base + local_input])
                : half(0.0h);
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 16;
            const uint local_pair = linear % 16;
            const uint output_column = output_base + local_output;
            const uint local_input = local_pair * 2;
            if (output_column < params.output_columns) {
                const uint packed_index = output_column * packed_columns
                    + input_base / 2 + local_pair;
                const uchar packed = weights[packed_index];
                weight_tile[local_input][local_output] = half(qi_signed_q4(packed, 0));
                weight_tile[local_input + 1][local_output] = half(qi_signed_q4(packed, 4));
            } else {
                weight_tile[local_input][local_output] = half(0.0h);
                weight_tile[local_input + 1][local_output] = half(0.0h);
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
        const uint local_row = linear / 64;
        const uint local_output = linear % 64;
        const uint output_row = row_base + local_row;
        const uint output_column = output_base + local_output;
        if (output_row < params.rows && output_column < params.output_columns) {
            output[output_row * params.output_columns + output_column] =
                output_tile[local_row][local_output] * float(scales[output_column]);
        }
    }
}

// Exact-tile Q4 counterpart of qi_block_linear_simdgroup_half_64x64_direct.
// Everything after the weight load is deliberately identical. The packed
// path expands and scales one 32x64 weight tile directly into threadgroup
// half, then uses the same FP16 SIMD-group MMA and direct output stores.
// Dispatch only when both output dimensions are complete 64-element tiles.
kernel void qi_block_q4_simdgroup_half_64x64_direct(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Int8LinearParams &params [[buffer(4)]],
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
            input_tile[local_row][local_input] = half(
                input[(row_base + local_row) * params.input_columns + input_base + local_input]
            );
        }
        for (uint linear = thread_index; linear < 1024; linear += 512) {
            const uint local_output = linear / 16;
            const uint local_pair = linear % 16;
            const uint output_column = output_base + local_output;
            const uint local_input = local_pair * 2;
            const uchar packed = weights[
                output_column * packed_columns + input_base / 2 + local_pair
            ];
            const half scale = scales[output_column];
            weight_tile[local_input][local_output] =
                half(qi_signed_q4(packed, 0)) * scale;
            weight_tile[local_input + 1][local_output] =
                half(qi_signed_q4(packed, 4)) * scale;
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

// Cache-DiT compares the first-block residual h_1 - h_0 and reuses the
// residual produced by the remaining blocks. `slot` is a row offset used by
// the first joint text+image pass; subsequent target-only passes use zero.
kernel void qi_cache_first_residual(
    device const float *before [[buffer(0)]],
    device const float *after [[buffer(1)]],
    device float *first_output [[buffer(2)]],
    device float *residual [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        const uint source_index = params.slot * params.width + index;
        const float value = after[source_index];
        first_output[index] = value;
        residual[index] = value - before[source_index];
    }
}

kernel void qi_cache_store_residual(
    device const float *final_output [[buffer(0)]],
    device const float *first_output [[buffer(1)]],
    device float *residual [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        const uint source_index = params.slot * params.width + index;
        residual[index] = final_output[source_index] - first_output[index];
    }
}

kernel void qi_cache_apply_residual(
    device const float *first_output [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        output[index] = first_output[index] + residual[index];
    }
}

// First-order TaylorSeer models the residual produced by blocks 1..31 as a
// value and a per-denoising-step derivative. A full step updates both in
// place; a cached step evaluates the model at `epsilon` steps after that full
// step. Keeping the ordinary Cache-DiT kernels separate makes the approximation
// an explicit opt-in rather than silently changing the established path.
kernel void qi_taylor_store_residual(
    device const float *final_output [[buffer(0)]],
    device const float *first_output [[buffer(1)]],
    device float *residual [[buffer(2)]],
    device float *derivative [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        const uint source_index = params.slot * params.width + index;
        const float next_residual = final_output[source_index] - first_output[index];
        derivative[index] = (next_residual - residual[index]) * params.epsilon;
        residual[index] = next_residual;
    }
}

kernel void qi_taylor_apply_residual(
    device const float *first_output [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device const float *derivative [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) {
        output[index] = first_output[index]
            + residual[index]
            + params.epsilon * derivative[index];
    }
}

// Two-pass relative-L1 reduction for Cache-DiT. The product trajectory must
// synchronize after block 0 to make its cache decision, but only these two
// totals need to cross to the CPU rather than the entire residual tensor.
kernel void qi_cache_relative_l1_partials(
    device const float *previous [[buffer(0)]],
    device const float *current [[buffer(1)]],
    device float2 *partials [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float2 sums[256];
    const uint count = params.rows * params.width;
    const uint group_count = params.heads;
    float2 local = float2(0.0f);
    for (uint index = group * 256 + lane; index < count;
         index += group_count * 256) {
        const float prior = previous[index];
        local.x += fabs(current[index] - prior);
        local.y += fabs(prior);
    }
    sums[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        partials[group] = sums[0];
    }
}

kernel void qi_cache_relative_l1_finish(
    device const float2 *partials [[buffer(0)]],
    device float2 *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 sums[256];
    sums[lane] = lane < params.heads ? partials[lane] : float2(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[0] = sums[0];
    }
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

// Conditioned generation projects the fixed condition images once and the
// changing target image on every denoising step. Non-negative source indices
// address their logical concatenation; params.slot is the condition row count.
kernel void qi_conditioned_joint_assemble(
    device const float *text [[buffer(0)]],
    device const float *condition [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const int *source_index [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    device const float *target [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) {
        return;
    }
    const uint row = index / params.width;
    const uint column = index % params.width;
    const int source = source_index[row];
    if (source < 0) {
        output[index] = text[uint(-source - 1) * params.width + column];
    } else if (uint(source) < params.slot) {
        output[index] = condition[uint(source) * params.width + column];
    } else {
        output[index] = target[(uint(source) - params.slot) * params.width + column];
    }
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
        const uint row = index / params.width;
        const uint column = index % params.width;
        cache_key[index] = key[index];
        cache_value[index] =
            value[row * params.input_stride + params.input_offset + column];
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
kernel void qi_block_layernorm_modulate_scalar(
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

// One SIMD group owns one row. Each lane visits every 32nd column, then the
// SIMD reduction combines the 32 partial sums without threadgroup memory.
kernel void qi_block_layernorm_modulate(
    device const float *input [[buffer(0)]],
    device const float *modulation [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uchar *target_mask [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    const uint start = row * params.width;
    float partial_mean = 0.0f;
    for (uint column = lane; column < params.width; column += 32) {
        partial_mean += input[start + column];
    }
    const float mean = simd_sum(partial_mean) / float(params.width);
    float partial_variance = 0.0f;
    for (uint column = lane; column < params.width; column += 32) {
        const float centered = input[start + column] - mean;
        partial_variance = fma(centered, centered, partial_variance);
    }
    const float inverse_std =
        rsqrt(simd_sum(partial_variance) / float(params.width) + params.epsilon);
    const uint modulation_row = target_mask[row] != 0 ? 0 : 1;
    const uint modulation_start = (modulation_row * 4 + params.slot) * params.width;
    for (uint column = lane; column < params.width; column += 32) {
        output[start + column] = (input[start + column] - mean) * inverse_std *
            (1.0f + modulation[modulation_start + column]);
    }
}

// MPS consumes FP16 activations. This variant performs the same FP32
// normalization and modulation arithmetic, then stores the final value as
// half directly instead of writing an FP32 tensor for a second conversion
// pass to read back.
kernel void qi_block_layernorm_modulate_half(
    device const float *input [[buffer(0)]],
    device const float *modulation [[buffer(1)]],
    device half *output [[buffer(2)]],
    device const uchar *target_mask [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    const uint start = row * params.width;
    float partial_mean = 0.0f;
    for (uint column = lane; column < params.width; column += 32) {
        partial_mean += input[start + column];
    }
    const float mean = simd_sum(partial_mean) / float(params.width);
    float partial_variance = 0.0f;
    for (uint column = lane; column < params.width; column += 32) {
        const float centered = input[start + column] - mean;
        partial_variance = fma(centered, centered, partial_variance);
    }
    const float inverse_std =
        rsqrt(simd_sum(partial_variance) / float(params.width) + params.epsilon);
    const uint modulation_row = target_mask[row] != 0 ? 0 : 1;
    const uint modulation_start = (modulation_row * 4 + params.slot) * params.width;
    for (uint column = lane; column < params.width; column += 32) {
        output[start + column] = half((input[start + column] - mean) * inverse_std *
            (1.0f + modulation[modulation_start + column]));
    }
}

// One thread owns one [token, head]. The learned RMSNorm scale is shared by heads.
kernel void qi_block_qk_norm_rope_scalar(
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
    const uint start = row * params.input_stride + params.input_offset
        + head * params.head_dimension;
    const uint output_start = (row * params.heads + head) * params.head_dimension;
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
        output[output_start + real_column] = real * cosine - imaginary * sine;
        output[output_start + imaginary_column] = real * sine + imaginary * cosine;
    }
}

// One SIMD group owns one [token, head]. With a 128-wide head each lane reads
// four channels and writes two complex RoPE pairs.
kernel void qi_block_qk_norm_rope(
    device const float *input [[buffer(0)]],
    device const ushort *norm_weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *rope [[buffer(3)]],
    constant BlockParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint index [[threadgroup_position_in_grid]]) {
    const uint count = params.rows * params.heads;
    if (index >= count) {
        return;
    }
    const uint row = index / params.heads;
    const uint head = index % params.heads;
    const uint start = row * params.input_stride + params.input_offset
        + head * params.head_dimension;
    const uint output_start = index * params.head_dimension;
    float partial_square = 0.0f;
    for (uint column = lane; column < params.head_dimension; column += 32) {
        const float value = input[start + column];
        partial_square = fma(value, value, partial_square);
    }
    const float inverse_rms =
        rsqrt(simd_sum(partial_square) / float(params.head_dimension) + params.epsilon);
    const uint complex_count = params.head_dimension / 2;
    const uint rope_start = row * complex_count * 2;
    for (uint pair = lane; pair < complex_count; pair += 32) {
        const uint real_column = pair * 2;
        const uint imaginary_column = real_column + 1;
        const float real = input[start + real_column] * inverse_rms *
            qi_bf16(norm_weight[real_column]);
        const float imaginary = input[start + imaginary_column] * inverse_rms *
            qi_bf16(norm_weight[imaginary_column]);
        const float cosine = rope[rope_start + pair];
        const float sine = rope[rope_start + complex_count + pair];
        output[output_start + real_column] = real * cosine - imaginary * sine;
        output[output_start + imaginary_column] = real * sine + imaginary * cosine;
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
        const uint value_index = from_cache
            ? key_start + lane
            : local_key_row * params.value_stride + params.value_offset
                + head * params.head_dimension + lane;
        const float value_element = from_cache ? cache_value[value_index] : value[value_index];
        accumulator = fma(accumulator, state[2], state[3] * value_element);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[query_start + lane] = accumulator / state[1];
}

// One 32-lane SIMD group owns two queries of one head. Each lane holds four
// head channels for every query. The K/V vector is loaded once per key and
// reused across both online softmaxes, reducing the dominant cache traffic
// without allocating a score tensor or changing the per-query reduction order.
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
    const uint query_base = (group / params.heads) * 2;
    const uint head = group % params.heads;
    if (query_base >= params.query_rows) {
        return;
    }
    const uint channel_0 = lane;
    const uint channel_1 = lane + 32;
    const uint channel_2 = lane + 64;
    const uint channel_3 = lane + 96;
    float4 query_channels[2];
    float4 accumulators[2];
    float maximums[2];
    float denominators[2];
    for (uint query_slot = 0; query_slot < 2; ++query_slot) {
        const uint query_row = query_base + query_slot;
        if (query_row < params.query_rows) {
            const uint query_start =
                (query_row * params.heads + head) * params.head_dimension;
            query_channels[query_slot] = float4(
                query[query_start + channel_0],
                query[query_start + channel_1],
                query[query_start + channel_2],
                query[query_start + channel_3]);
        } else {
            query_channels[query_slot] = float4(0.0f);
        }
        accumulators[query_slot] = float4(0.0f);
        maximums[query_slot] = -INFINITY;
        denominators[query_slot] = 0.0f;
    }
    const float scale = rsqrt(float(params.head_dimension));

    for (uint key_row = 0; key_row < params.key_rows; ++key_row) {
        const bool from_cache = key_row < params.cached_prefix_rows;
        const uint local_key_row = from_cache ? key_row : key_row - params.cached_prefix_rows;
        const uint key_start = (local_key_row * params.heads + head) * params.head_dimension;
        const uint value_start = from_cache ? key_start
            : local_key_row * params.value_stride + params.value_offset
                + head * params.head_dimension;
        device const float *key_source = from_cache ? cache_key : key;
        device const float *value_source = from_cache ? cache_value : value;
        const float4 key_channels = float4(
            key_source[key_start + channel_0],
            key_source[key_start + channel_1],
            key_source[key_start + channel_2],
            key_source[key_start + channel_3]);
        const float4 value_channels = float4(
            value_source[value_start + channel_0],
            value_source[value_start + channel_1],
            value_source[value_start + channel_2],
            value_source[value_start + channel_3]);
        for (uint query_slot = 0; query_slot < 2; ++query_slot) {
            const uint query_row = query_base + query_slot;
            if (query_row >= params.query_rows) {
                continue;
            }
            const uint global_query_row = query_row + params.query_position_offset;
            const bool same_image = image_ids[global_query_row] >= 0 &&
                image_ids[global_query_row] == image_ids[key_row];
            const bool allowed = key_valid[key_row] != 0 &&
                (params.block_causal == 0 || global_query_row >= key_row || same_image);
            if (!allowed) {
                continue;
            }
            const float score = simd_sum(dot(query_channels[query_slot], key_channels)) * scale;
            const float next_maximum = max(maximums[query_slot], score);
            const float old_scale = isinf(maximums[query_slot])
                ? 0.0f : exp(maximums[query_slot] - next_maximum);
            const float new_weight = exp(score - next_maximum);
            maximums[query_slot] = next_maximum;
            denominators[query_slot] =
                denominators[query_slot] * old_scale + new_weight;
            accumulators[query_slot] = fma(
                accumulators[query_slot], old_scale, new_weight * value_channels);
        }
    }
    for (uint query_slot = 0; query_slot < 2; ++query_slot) {
        const uint query_row = query_base + query_slot;
        if (query_row < params.query_rows) {
            const uint query_start =
                (query_row * params.heads + head) * params.head_dimension;
            const float4 result = accumulators[query_slot] / denominators[query_slot];
            output[query_start + channel_0] = result.x;
            output[query_start + channel_1] = result.y;
            output[query_start + channel_2] = result.z;
            output[query_start + channel_3] = result.w;
        }
    }
}

// One SIMD group owns four queries of one head. This doubles K/V reuse over
// the production two-query kernel without threadgroup staging or barriers.
// The long 4096-key path is benchmarked separately because its bandwidth
// savings may outweigh the additional register pressure that loses at 256px.
kernel void qi_block_attention_simdgroup4(
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
    const uint query_group_base = (group / params.heads) * 4;
    const uint head = group % params.heads;
    if (query_group_base >= params.query_rows) {
        return;
    }
    const uint channel_0 = lane;
    const uint channel_1 = lane + 32;
    const uint channel_2 = lane + 64;
    const uint channel_3 = lane + 96;
    float4 query_channels[4];
    float4 accumulators[4];
    float maximums[4];
    float denominators[4];
    for (uint query_slot = 0; query_slot < 4; ++query_slot) {
        const uint query_row = query_group_base + query_slot;
        if (query_row < params.query_rows) {
            const uint query_start =
                (query_row * params.heads + head) * params.head_dimension;
            query_channels[query_slot] = float4(
                query[query_start + channel_0],
                query[query_start + channel_1],
                query[query_start + channel_2],
                query[query_start + channel_3]);
        } else {
            query_channels[query_slot] = float4(0.0f);
        }
        accumulators[query_slot] = float4(0.0f);
        maximums[query_slot] = -INFINITY;
        denominators[query_slot] = 0.0f;
    }
    const float scale = rsqrt(float(params.head_dimension));

    for (uint key_row = 0; key_row < params.key_rows; ++key_row) {
        const bool from_cache = key_row < params.cached_prefix_rows;
        const uint local_key_row = from_cache ? key_row : key_row - params.cached_prefix_rows;
        const uint key_start = (local_key_row * params.heads + head) * params.head_dimension;
        const uint value_start = from_cache ? key_start
            : local_key_row * params.value_stride + params.value_offset
                + head * params.head_dimension;
        device const float *key_source = from_cache ? cache_key : key;
        device const float *value_source = from_cache ? cache_value : value;
        const float4 key_channels = float4(
            key_source[key_start + channel_0],
            key_source[key_start + channel_1],
            key_source[key_start + channel_2],
            key_source[key_start + channel_3]);
        const float4 value_channels = float4(
            value_source[value_start + channel_0],
            value_source[value_start + channel_1],
            value_source[value_start + channel_2],
            value_source[value_start + channel_3]);
        for (uint query_slot = 0; query_slot < 4; ++query_slot) {
            const uint query_row = query_group_base + query_slot;
            if (query_row >= params.query_rows) {
                continue;
            }
            const uint global_query_row = query_row + params.query_position_offset;
            const bool same_image = image_ids[global_query_row] >= 0 &&
                image_ids[global_query_row] == image_ids[key_row];
            const bool allowed = key_valid[key_row] != 0 &&
                (params.block_causal == 0 || global_query_row >= key_row || same_image);
            if (!allowed) {
                continue;
            }
            const float score = simd_sum(dot(query_channels[query_slot], key_channels)) * scale;
            const float next_maximum = max(maximums[query_slot], score);
            const float old_scale = isinf(maximums[query_slot])
                ? 0.0f : exp(maximums[query_slot] - next_maximum);
            const float new_weight = exp(score - next_maximum);
            maximums[query_slot] = next_maximum;
            denominators[query_slot] = denominators[query_slot] * old_scale + new_weight;
            accumulators[query_slot] = fma(
                accumulators[query_slot], old_scale, new_weight * value_channels);
        }
    }
    for (uint query_slot = 0; query_slot < 4; ++query_slot) {
        const uint query_row = query_group_base + query_slot;
        if (query_row < params.query_rows) {
            const uint query_start =
                (query_row * params.heads + head) * params.head_dimension;
            const float4 result =
                accumulators[query_slot] / denominators[query_slot];
            output[query_start + channel_0] = result.x;
            output[query_start + channel_1] = result.y;
            output[query_start + channel_2] = result.z;
            output[query_start + channel_3] = result.w;
        }
    }
}

// Fixed-shape Flash Attention for Qwen-Image's 32 heads with dimension 128.
// The 8-query/64-key/four-SIMD-group structure follows llama.cpp's MIT-licensed
// Metal Flash Attention kernel, specialized here to contiguous FP16 K/V and
// this model's block-causal image mask. Scores, softmax state, and output
// accumulation remain FP32; no quadratic score tensor is materialized.
kernel void qi_block_attention_flash128(
    device const float *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const half *value [[buffer(3)]],
    device const int *image_ids [[buffer(5)]],
    device const uchar *key_valid [[buffer(6)]],
    constant FlashAttentionParams &params [[buffer(4)]],
    threadgroup half *shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint heads = 32;
    constexpr uint head_dimension = 128;
    constexpr uint width = heads * head_dimension;
    constexpr uint queries_per_group = 8;
    constexpr uint keys_per_tile = 64;
    constexpr uint score_stride = 128;

    const uint query_group_base = (group / heads) * queries_per_group;
    const uint head = group % heads;
    if (query_group_base >= params.query_rows) {
        return;
    }

    threadgroup half *query_shared = shared;
    threadgroup float *output_shared =
        reinterpret_cast<threadgroup float *>(shared + queries_per_group * head_dimension);
    threadgroup float *score_shared = reinterpret_cast<threadgroup float *>(
        shared + 3 * queries_per_group * head_dimension);
    threadgroup half4 *query_shared4 =
        reinterpret_cast<threadgroup half4 *>(query_shared);
    threadgroup float4 *output_shared4 =
        reinterpret_cast<threadgroup float4 *>(output_shared);

    // Each SIMD group stages two of the eight query rows.
    for (uint query_slot = 0; query_slot < 2; ++query_slot) {
        const uint local_query = query_slot * 4 + simd_group;
        const uint query_row = query_group_base + local_query;
        if (query_row < params.query_rows) {
            const uint query_start = (query_row * heads + head) * head_dimension;
            device const float4 *query4 =
                reinterpret_cast<device const float4 *>(query + query_start);
            query_shared4[local_query * 32 + lane] = half4(query4[lane]);
        } else {
            query_shared4[local_query * 32 + lane] = half4(0.0h);
        }
        output_shared4[local_query * 32 + lane] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximums[2] = { -INFINITY, -INFINITY };
    float denominators[2] = { 0.0f, 0.0f };
    const float scale = rsqrt(float(head_dimension));
    const uint last_query_row = min(
        query_group_base + queries_per_group - 1,
        params.query_rows - 1) + params.query_position_offset;

    for (uint key_base = 0; key_base < params.padded_key_rows;
            key_base += keys_per_tile) {
        // Image IDs occupy contiguous blocks in the joint layout. Once a key
        // tile lies beyond this query group's last row, only that last row's
        // image block can still be visible to any query in the group.
        if (params.prune_masked_tiles != 0 && params.block_causal != 0
            && key_base > last_query_row
            && (key_base >= params.key_rows
                || image_ids[last_query_row] < 0
                || image_ids[key_base] != image_ids[last_query_row])) {
            continue;
        }
        // Q.K^T: each SIMD group produces two 8-key stripes for all 8 queries.
        device const half *key_tile =
            key + (key_base * heads + head) * head_dimension
            + simd_group * 8 * width;
        threadgroup float *score_tile = score_shared + simd_group * 8;
        for (uint key_stripe = 0; key_stripe < 2; ++key_stripe) {
            simdgroup_float8x8 scores(0.0f);
#pragma unroll(8)
            for (uint channels = 0; channels < head_dimension; channels += 16) {
                simdgroup_half8x8 query_matrix_0;
                simdgroup_half8x8 query_matrix_1;
                simdgroup_half8x8 key_matrix_0;
                simdgroup_half8x8 key_matrix_1;
                simdgroup_load(
                    query_matrix_0, query_shared + channels, head_dimension);
                simdgroup_load(
                    query_matrix_1, query_shared + channels + 8, head_dimension);
                simdgroup_load(
                    key_matrix_0, key_tile + channels, width, 0, true);
                simdgroup_load(
                    key_matrix_1, key_tile + channels + 8, width, 0, true);
                simdgroup_multiply_accumulate(
                    scores, query_matrix_0, key_matrix_0, scores);
                simdgroup_multiply_accumulate(
                    scores, query_matrix_1, key_matrix_1, scores);
            }
            simdgroup_store(scores, score_tile, score_stride);
            key_tile += 32 * width;
            score_tile += 32;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online FP32 softmax. Each SIMD group owns two query rows.
        for (uint query_slot = 0; query_slot < 2; ++query_slot) {
            const uint local_query = query_slot * 4 + simd_group;
            const uint query_row = query_group_base + local_query;
            const uint global_query_row = query_row + params.query_position_offset;
            const uint key_row_0 = key_base + lane * 2;
            const uint key_row_1 = key_row_0 + 1;
            float2 scores = reinterpret_cast<threadgroup float2 *>(
                score_shared + local_query * score_stride)[lane] * scale;
            const bool valid_query = query_row < params.query_rows;
            const int query_image = valid_query ? image_ids[global_query_row] : -1;
            const bool allowed_0 = valid_query && key_row_0 < params.key_rows
                && key_valid[key_row_0] != 0
                && (params.block_causal == 0 || global_query_row >= key_row_0
                    || (query_image >= 0 && query_image == image_ids[key_row_0]));
            const bool allowed_1 = valid_query && key_row_1 < params.key_rows
                && key_valid[key_row_1] != 0
                && (params.block_causal == 0 || global_query_row >= key_row_1
                    || (query_image >= 0 && query_image == image_ids[key_row_1]));
            // A partial final query group still participates in matrix MMAs.
            // Give its unused rows a finite softmax so NaNs never enter shared
            // matrices, even though those rows are not written to output.
            scores[0] = !valid_query ? 0.0f : (allowed_0 ? scores[0] : -INFINITY);
            scores[1] = !valid_query ? 0.0f : (allowed_1 ? scores[1] : -INFINITY);

            const float previous_maximum = maximums[query_slot];
            maximums[query_slot] = simd_max(max(
                previous_maximum, max(scores[0], scores[1])));
            const float previous_scale = isinf(previous_maximum)
                ? 0.0f : exp(previous_maximum - maximums[query_slot]);
            const float2 weights = exp(scores - maximums[query_slot]);
            denominators[query_slot] = denominators[query_slot] * previous_scale
                + simd_sum(weights[0] + weights[1]);
            reinterpret_cast<threadgroup float2 *>(
                score_shared + local_query * score_stride)[lane] = weights;
            output_shared4[local_query * 32 + lane] *= previous_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P.V: four SIMD groups jointly cover all 128 output channels.
        simdgroup_float8x8 accumulators[4];
        threadgroup float *output_tile = output_shared + 8 * simd_group;
#pragma unroll(4)
        for (uint output_matrix = 0; output_matrix < 4; ++output_matrix) {
            simdgroup_load(
                accumulators[output_matrix], output_tile, head_dimension);
            output_tile += 32;
        }
        device const half *value_tile =
            value + (key_base * heads + head) * head_dimension + 8 * simd_group;
#pragma unroll(4)
        for (uint key_pair = 0; key_pair < 4; ++key_pair) {
            simdgroup_float8x8 weights_0;
            simdgroup_float8x8 weights_1;
            simdgroup_load(
                weights_0, score_shared + key_pair * 16, score_stride);
            simdgroup_load(
                weights_1, score_shared + key_pair * 16 + 8, score_stride);
#pragma unroll(2)
            for (uint output_pair = 0; output_pair < 2; ++output_pair) {
                simdgroup_half8x8 value_matrix_0;
                simdgroup_half8x8 value_matrix_1;
                simdgroup_half8x8 value_matrix_2;
                simdgroup_half8x8 value_matrix_3;
                const uint channel_offset = output_pair * 64;
                simdgroup_load(
                    value_matrix_0, value_tile + channel_offset, width);
                simdgroup_load(
                    value_matrix_1, value_tile + channel_offset + 32, width);
                simdgroup_load(
                    value_matrix_2, value_tile + channel_offset + 8 * width, width);
                simdgroup_load(
                    value_matrix_3,
                    value_tile + channel_offset + 8 * width + 32, width);
                simdgroup_multiply_accumulate(
                    accumulators[output_pair * 2], weights_0,
                    value_matrix_0, accumulators[output_pair * 2]);
                simdgroup_multiply_accumulate(
                    accumulators[output_pair * 2 + 1], weights_0,
                    value_matrix_1, accumulators[output_pair * 2 + 1]);
                simdgroup_multiply_accumulate(
                    accumulators[output_pair * 2], weights_1,
                    value_matrix_2, accumulators[output_pair * 2]);
                simdgroup_multiply_accumulate(
                    accumulators[output_pair * 2 + 1], weights_1,
                    value_matrix_3, accumulators[output_pair * 2 + 1]);
            }
            value_tile += 16 * width;
        }
        output_tile = output_shared + 8 * simd_group;
#pragma unroll(4)
        for (uint output_matrix = 0; output_matrix < 4; ++output_matrix) {
            simdgroup_store(
                accumulators[output_matrix], output_tile, head_dimension);
            output_tile += 32;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint query_slot = 0; query_slot < 2; ++query_slot) {
        const uint local_query = query_slot * 4 + simd_group;
        const uint query_row = query_group_base + local_query;
        if (query_row < params.query_rows) {
            const uint output_start = (query_row * heads + head) * head_dimension;
            device float4 *destination =
                reinterpret_cast<device float4 *>(output + output_start);
            destination[lane] = output_shared4[local_query * 32 + lane]
                / denominators[query_slot];
        }
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

// The following MPS multiplication consumes FP16. Store the FP32 SwiGLU
// result as half here so no full-size FP32 intermediate or conversion pass is
// required.
kernel void qi_block_swiglu_half(
    device const float *gate [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant BlockParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width * 3;
    if (index >= count) {
        return;
    }
    const float value = gate[index];
    output[index] = half((value / (1.0f + exp(-value))) * projected[index]);
}
