#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Reduction operations (sum, max, mean along axes)
// ============================================================================

// Sum reduction along last axis
// input: [rows, cols], output: [rows]
kernel void sum_rows_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += input[id * cols + j];
    }
    output[id] = sum;
}

// Sum all elements
kernel void sum_all_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* n [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    uint count = n[0];
    float sum = 0.0f;
    for (uint i = id; i < count; i += threads) {
        sum += input[i];
    }

    // Use threadgroup reduction
    threadgroup float shared[256];
    uint lid = id % 256;
    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        // Atomic add to output
        // For simplicity, we use a single threadgroup approach
        output[0] = shared[0];
    }
}

// Max reduction along last axis (for softmax stability)
kernel void max_rows_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    float max_val = input[id * cols];
    for (uint j = 1; j < cols; j++) {
        max_val = max(max_val, input[id * cols + j]);
    }
    output[id] = max_val;
}

// Mean along last axis
kernel void mean_rows_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += input[id * cols + j];
    }
    output[id] = sum / float(cols);
}

// Variance along last axis (for layernorm)
kernel void var_rows_f32(
    device const float* input [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    float m = mean[id];
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = input[id * cols + j] - m;
        var += diff * diff;
    }
    output[id] = var / float(cols);
}

// Sum along columns (reduce axis 0)
// input: [rows, cols], output: [cols]
kernel void sum_cols_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= cols) return;

    float sum = 0.0f;
    for (uint i = 0; i < rows; i++) {
        sum += input[i * cols + id];
    }
    output[id] = sum;
}

// Argmax along last axis
kernel void argmax_rows_f32(
    device const float* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    float max_val = input[id * cols];
    uint max_idx = 0;
    for (uint j = 1; j < cols; j++) {
        float val = input[id * cols + j];
        if (val > max_val) {
            max_val = val;
            max_idx = j;
        }
    }
    output[id] = max_idx;
}
