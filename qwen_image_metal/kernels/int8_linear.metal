#include <metal_stdlib>
using namespace metal;

constant uint QI_INT8_TILE = 16;

struct Int8LinearParams {
    uint rows;
    uint output_columns;
    uint input_columns;
    uint group_size;
};

// input:   [rows, input_columns] F32
// weights: [output_columns, input_columns] affine U8
// scales:  [output_columns, input_columns / group_size] F16
// zeros:   [output_columns, input_columns / group_size] U8
// output:  [rows, output_columns] F32
kernel void qi_int8_affine_linear_16x16(
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

    const uint row = group_position.y * QI_INT8_TILE + thread_position.y;
    const uint output_column = group_position.x * QI_INT8_TILE + thread_position.x;
    const uint groups_per_row = params.input_columns / params.group_size;
    float sum = 0.0f;

    for (uint input_base = 0; input_base < params.input_columns; input_base += QI_INT8_TILE) {
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
        for (uint inner = 0; inner < QI_INT8_TILE; ++inner) {
            sum += float(
                input_tile[thread_position.y][inner] * weight_tile[inner][thread_position.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.rows && output_column < params.output_columns) {
        output[row * params.output_columns + output_column] = sum;
    }
}
