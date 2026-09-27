#include <metal_stdlib>
using namespace metal;

constant uint QI_TILE = 16;

struct LinearParams {
    uint rows;
    uint output_columns;
    uint input_columns;
    uint weight_mode;
};

inline float qi_decode_weight(ushort bits, uint mode) {
    if (mode == 0) {
        return as_type<float>(uint(bits) << 16);
    }
    return float(as_type<half>(bits));
}

// input:   [rows, input_columns]
// weights: [output_columns, input_columns] (Safetensors [out, in])
// output:  [rows, output_columns]
kernel void qi_linear_16x16(
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
                ? qi_decode_weight(
                      weights[output_column * params.input_columns + weight_input_column],
                      params.weight_mode)
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
