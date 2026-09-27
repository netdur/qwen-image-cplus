#include <metal_stdlib>

using namespace metal;

kernel void qi_vector_add(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    output[index] = lhs[index] + rhs[index];
}
