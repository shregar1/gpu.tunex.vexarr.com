#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Softmax and cross-entropy loss kernels
// ============================================================================

// Online softmax: numerically stable, single-pass
// input: [rows, cols], output: [rows, cols]
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    uint offset = id * cols;

    // Find max for numerical stability
    float max_val = input[offset];
    for (uint j = 1; j < cols; j++) {
        max_val = max(max_val, input[offset + j]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float e = exp(input[offset + j] - max_val);
        output[offset + j] = e;
        sum += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < cols; j++) {
        output[offset + j] *= inv_sum;
    }
}

// Softmax backward
// grad_output: [rows, cols], softmax_output: [rows, cols], grad_input: [rows, cols]
kernel void softmax_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* softmax_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    uint offset = id * cols;

    // dot = sum(grad_output * softmax_output)
    float dot = 0.0f;
    for (uint j = 0; j < cols; j++) {
        dot += grad_output[offset + j] * softmax_output[offset + j];
    }

    // grad_input = softmax_output * (grad_output - dot)
    for (uint j = 0; j < cols; j++) {
        grad_input[offset + j] = softmax_output[offset + j] * (grad_output[offset + j] - dot);
    }
}

// Cross-entropy loss from logits (fused softmax + nll_loss)
// logits: [batch, vocab], targets: [batch] (uint indices)
// output: [batch] (per-sample losses)
kernel void cross_entropy_forward_f32(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* losses [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [batch, vocab]
    uint id [[thread_position_in_grid]]
) {
    uint batch = dims[0];
    uint vocab = dims[1];

    if (id >= batch) return;

    uint offset = id * vocab;
    uint target = targets[id];

    // Log-sum-exp for numerical stability
    float max_val = logits[offset];
    for (uint j = 1; j < vocab; j++) {
        max_val = max(max_val, logits[offset + j]);
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < vocab; j++) {
        sum_exp += exp(logits[offset + j] - max_val);
    }

    float log_sum_exp = max_val + log(sum_exp);
    losses[id] = log_sum_exp - logits[offset + target];
}

// Cross-entropy backward (produces gradients w.r.t. logits)
// softmax_output: [batch, vocab], targets: [batch], grad_logits: [batch, vocab]
kernel void cross_entropy_backward_f32(
    device const float* softmax_output [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* grad_logits [[buffer(2)]],
    device const uint* dims [[buffer(3)]],   // [batch, vocab]
    device const float* grad_scale [[buffer(4)]],  // scalar: typically 1/batch
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = dims[0];
    uint vocab = dims[1];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= batch || col >= vocab) return;

    float scale = grad_scale[0];
    float grad = softmax_output[row * vocab + col];
    if (col == targets[row]) {
        grad -= 1.0f;
    }
    grad_logits[row * vocab + col] = grad * scale;
}
