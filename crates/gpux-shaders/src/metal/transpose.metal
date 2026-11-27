#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Transpose and data layout kernels
// ============================================================================

// 2D transpose: input [M, N] -> output [N, M]
kernel void transpose_2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [M, N]
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims[0];
    uint N = dims[1];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    output[col * M + row] = input[row * N + col];
}

// Copy kernel
kernel void copy_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id];
}

// Accumulate: output += input
kernel void accumulate_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] += input[id];
}

// Broadcast add: output[i,j] = a[i,j] + bias[j]
// a: [rows, cols], bias: [cols]
kernel void broadcast_add_bias_f32(
    device const float* a [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [rows, cols]
    uint2 gid [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= rows || col >= cols) return;

    output[row * cols + col] = a[row * cols + col] + bias[col];
}

// Concat along last dimension
// a: [rows, cols_a], b: [rows, cols_b] -> output: [rows, cols_a + cols_b]
kernel void concat_cols_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [rows, cols_a, cols_b]
    uint2 gid [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols_a = dims[1];
    uint cols_b = dims[2];
    uint total_cols = cols_a + cols_b;

    uint row = gid.y;
    uint col = gid.x;

    if (row >= rows || col >= total_cols) return;

    if (col < cols_a) {
        output[row * total_cols + col] = a[row * cols_a + col];
    } else {
        output[row * total_cols + col] = b[row * cols_b + (col - cols_a)];
    }
}

// Slice along last dimension
// input: [rows, total_cols] -> output: [rows, end - start]
kernel void slice_cols_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, total_cols, start, end]
    uint2 gid [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint total_cols = dims[1];
    uint start = dims[2];
    uint end = dims[3];
    uint out_cols = end - start;

    uint row = gid.y;
    uint col = gid.x;

    if (row >= rows || col >= out_cols) return;

    output[row * out_cols + col] = input[row * total_cols + start + col];
}
