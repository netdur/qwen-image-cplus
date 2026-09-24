#include <metal_stdlib>
using namespace metal;

constant uint QT_TILE = 16;

struct LinearParams {
    uint rows;
    uint output_columns;
    uint input_columns;
    uint weight_mode;
};

struct TextParams {
    uint rows;
    uint width;
    uint heads;
    uint kv_heads;
    uint head_dimension;
    uint intermediate_width;
    float epsilon;
    float rope_theta;
};

inline float qt_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline float qt_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    const uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits = (bits + rounding_bias) & 0xFFFF0000u;
    return as_type<float>(bits);
}

kernel void qt_embedding(
    device const uint *token_ids [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TextParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) return;
    const uint row = index / params.width;
    const uint column = index % params.width;
    output[index] = qt_bf16(weights[token_ids[row] * params.width + column]);
}

kernel void qt_rms_norm(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TextParams &params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) return;
    const uint start = row * params.width;
    float sum = 0.0f;
    for (uint column = 0; column < params.width; ++column) {
        sum = fma(input[start + column], input[start + column], sum);
    }
    const float inverse_rms = rsqrt(sum / float(params.width) + params.epsilon);
    for (uint column = 0; column < params.width; ++column) {
        output[start + column] = qt_round_bf16(
            input[start + column] * inverse_rms * qt_bf16(weights[column]));
    }
}

kernel void qt_linear_16x16(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    const uint row = group_position.y * QT_TILE + thread_position.y;
    const uint output_column = group_position.x * QT_TILE + thread_position.x;
    float sum = 0.0f;
    for (uint input_base = 0; input_base < params.input_columns; input_base += QT_TILE) {
        const uint input_column = input_base + thread_position.x;
        input_tile[thread_position.y][thread_position.x] =
            row < params.rows && input_column < params.input_columns
                ? input[row * params.input_columns + input_column] : 0.0f;
        const uint weight_input_column = input_base + thread_position.y;
        weight_tile[thread_position.y][thread_position.x] =
            output_column < params.output_columns && weight_input_column < params.input_columns
                ? qt_bf16(weights[output_column * params.input_columns + weight_input_column]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < QT_TILE; ++inner) {
            sum = fma(input_tile[thread_position.y][inner], weight_tile[inner][thread_position.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.rows && output_column < params.output_columns) {
        output[row * params.output_columns + output_column] = qt_round_bf16(sum);
    }
}

// Qwen3-VL composes its 64 rotary frequencies from temporal, height, and width
// positions in an interleaved 24/20/20 split. Text rows carry the same value on
// all three axes, while image placeholders carry their merged 2D patch grid.
// Q and K use different head counts but share the learned 128-element RMSNorm
// scale.
kernel void qt_qk_norm_rope(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TextParams &params [[buffer(3)]],
    device const uint *position_ids [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.heads;
    if (index >= count) return;
    const uint row = index / params.heads;
    const uint head = index % params.heads;
    const uint start = (row * params.heads + head) * params.head_dimension;
    float sum = 0.0f;
    for (uint column = 0; column < params.head_dimension; ++column) {
        sum = fma(input[start + column], input[start + column], sum);
    }
    const float inverse_rms = rsqrt(sum / float(params.head_dimension) + params.epsilon);
    const uint half_dimension = params.head_dimension / 2;
    for (uint column = 0; column < params.head_dimension; ++column) {
        const uint frequency_column = column % half_dimension;
        const float inverse_frequency = exp(
            -log(params.rope_theta) * (2.0f * float(frequency_column)) /
            float(params.head_dimension));
        uint axis = 0;
        if (frequency_column < 60) {
            const uint interleave = frequency_column % 3;
            axis = interleave == 1 ? 1 : (interleave == 2 ? 2 : 0);
        }
        const float angle = float(position_ids[row * 3 + axis]) * inverse_frequency;
        const uint paired_column = column < half_dimension
            ? column + half_dimension : column - half_dimension;
        const float value = qt_round_bf16(
            input[start + column] * inverse_rms * qt_bf16(weights[column]));
        const float paired = qt_round_bf16(
            input[start + paired_column] * inverse_rms * qt_bf16(weights[paired_column]));
        const float rotated = column < half_dimension ? -paired : paired;
        const float cosine = qt_round_bf16(cos(angle));
        const float sine = qt_round_bf16(sin(angle));
        output[start + column] = qt_round_bf16(value * cosine + rotated * sine);
    }
}

// One 128-lane group owns one [query, query-head]. Qwen3-VL uses 32 query
// heads and 8 KV heads, so four adjacent query heads share one KV head.
kernel void qt_causal_gqa(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *value [[buffer(3)]],
    constant TextParams &params [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[128];
    threadgroup float state[4];
    const uint query_row = group / params.heads;
    const uint query_head = group % params.heads;
    if (query_row >= params.rows || lane >= params.head_dimension) return;
    const uint kv_head = query_head / (params.heads / params.kv_heads);
    const uint query_start = (query_row * params.heads + query_head) * params.head_dimension;
    const float scale = rsqrt(float(params.head_dimension));
    if (lane == 0) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float accumulator = 0.0f;
    for (uint key_row = 0; key_row <= query_row; ++key_row) {
        const uint key_start = (key_row * params.kv_heads + kv_head) * params.head_dimension;
        reduction[lane] = query[query_start + lane] * key[key_start + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = params.head_dimension / 2; stride > 0; stride /= 2) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
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
        accumulator = fma(accumulator, state[2], state[3] * value[key_start + lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[query_start + lane] = qt_round_bf16(accumulator / state[1]);
}

kernel void qt_add(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TextParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) output[index] = qt_round_bf16(left[index] + right[index]);
}

// DeepStack adds one vision-tower feature to every image-placeholder row
// after each of the first three language-model layers. Text rows carry -1.
kernel void qt_add_visual(
    device const float *input [[buffer(0)]],
    device const float *visual [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TextParams &params [[buffer(3)]],
    device const int *visual_row [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) return;
    const uint row = index / params.width;
    const uint column = index % params.width;
    const int source_row = visual_row[row];
    output[index] = source_row < 0
        ? input[index]
        : qt_round_bf16(input[index] + visual[uint(source_row) * params.width + column]);
}

kernel void qt_swiglu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant TextParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.intermediate_width;
    if (index < count) {
        const float value = gate[index];
        output[index] = qt_round_bf16((value / (1.0f + exp(-value))) * up[index]);
    }
}

// Expands one BF16 weight matrix to FP32 for the MPS linear path. The values
// are exact: BF16 is the upper half of an FP32 word.
kernel void qt_bf16_to_f32(
    device const ushort *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = qt_bf16(input[index]);
}

// Rounds an MPS linear output to BF16, as qt_linear_16x16 does per element.
kernel void qt_round_bf16_inplace(
    device float *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) values[index] = qt_round_bf16(values[index]);
}

// Causal row softmax over [rows, rows] FP32 scores already scaled by
// 1/sqrt(head_dimension): row i keeps keys 0..i and zeroes the rest.
kernel void qt_causal_row_softmax(
    device float *scores [[buffer(0)]],
    constant uint &rows [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[256];
    device float *values = scores + ulong(row) * rows;
    const uint valid = row + 1;
    float local_max = -INFINITY;
    for (uint index = lane; index < valid; index += 256) {
        local_max = max(local_max, values[index]);
    }
    reduction[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = max(reduction[lane], reduction[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = reduction[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint index = lane; index < valid; index += 256) {
        const float weight = exp(values[index] - row_max);
        values[index] = weight;
        local_sum += weight;
    }
    reduction[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_sum = 1.0f / reduction[0];
    for (uint index = lane; index < rows; index += 256) {
        values[index] = index < valid ? values[index] * inverse_sum : 0.0f;
    }
}
