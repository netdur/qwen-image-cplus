#include <metal_stdlib>
using namespace metal;

constant uint QV_TILE = 16;

struct VisionParams {
    uint rows;
    uint width;
    uint output_width;
    uint input_width;
    uint heads;
    uint head_dimension;
    float epsilon;
    float rope_theta;
};

inline float qv_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline float qv_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    const uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits = (bits + rounding_bias) & 0xFFFF0000u;
    return as_type<float>(bits);
}

kernel void qv_linear_bf16_bias_16x16(
    device const float *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant VisionParams &params [[buffer(4)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    const uint row = group_position.y * QV_TILE + thread_position.y;
    const uint output_column = group_position.x * QV_TILE + thread_position.x;
    float sum = 0.0f;
    for (uint input_base = 0; input_base < params.input_width; input_base += QV_TILE) {
        const uint input_column = input_base + thread_position.x;
        input_tile[thread_position.y][thread_position.x] =
            row < params.rows && input_column < params.input_width
                ? input[row * params.input_width + input_column] : 0.0f;
        const uint weight_input_column = input_base + thread_position.y;
        weight_tile[thread_position.y][thread_position.x] =
            output_column < params.output_width && weight_input_column < params.input_width
                ? qv_bf16(weights[output_column * params.input_width + weight_input_column]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < QV_TILE; ++inner) {
            sum = fma(input_tile[thread_position.y][inner], weight_tile[inner][thread_position.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.rows && output_column < params.output_width) {
        output[row * params.output_width + output_column] =
            qv_round_bf16(sum + qv_bf16(bias[output_column]));
    }
}

kernel void qv_layer_norm(
    device const float *input [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant VisionParams &params [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) return;
    const uint start = row * params.width;
    float mean = 0.0f;
    for (uint column = 0; column < params.width; ++column) mean += input[start + column];
    mean /= float(params.width);
    float squared = 0.0f;
    for (uint column = 0; column < params.width; ++column) {
        const float centered = input[start + column] - mean;
        squared = fma(centered, centered, squared);
    }
    const float inverse_std = rsqrt(squared / float(params.width) + params.epsilon);
    for (uint column = 0; column < params.width; ++column) {
        output[start + column] = qv_round_bf16(
            (input[start + column] - mean) * inverse_std * qv_bf16(weight[column])
                + qv_bf16(bias[column]));
    }
}

kernel void qv_add(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant VisionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index < count) output[index] = qv_round_bf16(left[index] + right[index]);
}

kernel void qv_gelu_tanh(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant VisionParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.width;
    if (index >= count) return;
    const float value = input[index];
    const float inner = 0.7978845608028654f
        * (value + 0.044715f * value * value * value);
    output[index] = qv_round_bf16(0.5f * value * (1.0f + tanh(inner)));
}

// QKV arrives as [row, 3, head, 72]. The pinned vision RoPE concatenates
// 18 height and 18 width frequencies, then repeats that 36-value half.
kernel void qv_split_qkv_rope(
    device const float *qkv [[buffer(0)]],
    device float *query [[buffer(1)]],
    device float *key [[buffer(2)]],
    device float *value [[buffer(3)]],
    device const uint2 *positions [[buffer(4)]],
    constant VisionParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = params.rows * params.heads;
    if (index >= count) return;
    const uint row = index / params.heads;
    const uint head = index % params.heads;
    const uint head_start = (row * params.heads + head) * params.head_dimension;
    const uint qkv_row = row * params.width * 3;
    const uint half_dimension = params.head_dimension / 2;
    const uint axis_frequencies = half_dimension / 2;
    const uint2 position = positions[row];
    for (uint column = 0; column < params.head_dimension; ++column) {
        const uint half_column = column % half_dimension;
        const bool height_axis = half_column < axis_frequencies;
        const uint frequency_column = height_axis ? half_column : half_column - axis_frequencies;
        const float inverse_frequency = exp(
            -log(params.rope_theta) * (2.0f * float(frequency_column)) / float(half_dimension));
        const float angle = float(height_axis ? position.x : position.y) * inverse_frequency;
        const uint paired = column < half_dimension
            ? column + half_dimension : column - half_dimension;
        const uint q_start = qkv_row + head * params.head_dimension;
        const uint k_start = qkv_row + params.width + head * params.head_dimension;
        const uint v_start = qkv_row + params.width * 2 + head * params.head_dimension;
        const float q = qkv[q_start + column];
        const float q_pair = qkv[q_start + paired];
        const float k = qkv[k_start + column];
        const float k_pair = qkv[k_start + paired];
        const float rotated_q = column < half_dimension ? -q_pair : q_pair;
        const float rotated_k = column < half_dimension ? -k_pair : k_pair;
        query[head_start + column] = qv_round_bf16(q * cos(angle) + rotated_q * sin(angle));
        key[head_start + column] = qv_round_bf16(k * cos(angle) + rotated_k * sin(angle));
        value[head_start + column] = qkv[v_start + column];
    }
}

// One 128-lane group owns one [query row, head]. Each image is dispatched
// separately, so every patch attends bidirectionally to every patch here.
kernel void qv_attention(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant VisionParams &params [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[128];
    threadgroup float state[4];
    const uint query_row = group / params.heads;
    const uint head = group % params.heads;
    const bool active = lane < params.head_dimension;
    const uint query_start = (query_row * params.heads + head) * params.head_dimension;
    if (lane == 0) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float accumulator = 0.0f;
    for (uint key_row = 0; key_row < params.rows; ++key_row) {
        const uint key_start = (key_row * params.heads + head) * params.head_dimension;
        reduction[lane] = active ? query[query_start + lane] * key[key_start + lane] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64; stride > 0; stride /= 2) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float score = reduction[0] * rsqrt(float(params.head_dimension));
            const float next_maximum = max(state[0], score);
            const float old_scale = isinf(state[0]) ? 0.0f : exp(state[0] - next_maximum);
            const float new_weight = exp(score - next_maximum);
            state[0] = next_maximum;
            state[1] = state[1] * old_scale + new_weight;
            state[2] = old_scale;
            state[3] = new_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            accumulator = fma(accumulator, state[2], state[3] * value[key_start + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) output[query_start + lane] = qv_round_bf16(accumulator / state[1]);
}

// Row softmax over [rows, rows] FP32 attention scores already scaled by
// 1/sqrt(head_dimension). One threadgroup per row; the row length is
// params.rows.
kernel void qv_row_softmax(
    device float *scores [[buffer(0)]],
    constant VisionParams &params [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[256];
    device float *values = scores + ulong(row) * params.rows;
    float local_max = -INFINITY;
    for (uint index = lane; index < params.rows; index += 256) {
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
    for (uint index = lane; index < params.rows; index += 256) {
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
    for (uint index = lane; index < params.rows; index += 256) {
        values[index] *= inverse_sum;
    }
}

// Rounds the attention output to BF16, as qv_attention does per element.
kernel void qv_round_bf16_inplace(
    device float *values [[buffer(0)]],
    constant VisionParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.rows * params.width) values[index] = qv_round_bf16(values[index]);
}
