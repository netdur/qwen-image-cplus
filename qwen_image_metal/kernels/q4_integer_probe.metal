#include <metal_stdlib>
using namespace metal;

// Compiler capability probe for the direct packed-Q4 spike. This file is not
// included by the runtime; each candidate is compiled independently while we
// establish which integer dot/matrix primitives exist on the M1 Metal stack.
kernel void qi_probe_char4_dot(
    device const char4 *left [[buffer(0)]],
    device const char4 *right [[buffer(1)]],
    device int *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    output[index] = dot(left[index], right[index]);
}
