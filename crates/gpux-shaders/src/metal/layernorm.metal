#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Layer Normalization kernels
// ============================================================================

// LayerNorm forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
// input: [rows, cols], gamma: [cols], beta: [cols]
// output: [rows, cols], mean_out: [rows], rstd_out: [rows]
kernel void layernorm_forward_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* mean_out [[buffer(4)]],
    device float* rstd_out [[buffer(5)]],
    device const uint* dims [[buffer(6)]],  // [rows, cols]
    device const float* eps_buf [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];
    float eps = eps_buf[0];

    if (id >= rows) return;

    uint offset = id * cols;

    // Compute mean
    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) {
        mean += input[offset + j];
    }
    mean /= float(cols);

    // Compute variance
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = input[offset + j] - mean;
        var += diff * diff;
    }
    var /= float(cols);

    float rstd = rsqrt(var + eps);

    mean_out[id] = mean;
    rstd_out[id] = rstd;

    // Normalize and apply affine transform
    for (uint j = 0; j < cols; j++) {
        float normalized = (input[offset + j] - mean) * rstd;
        output[offset + j] = gamma[j] * normalized + beta[j];
    }
}

// LayerNorm backward
// Computes grad_input, grad_gamma, grad_beta
kernel void layernorm_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* mean_saved [[buffer(3)]],
    device const float* rstd_saved [[buffer(4)]],
    device float* grad_input [[buffer(5)]],
    device const uint* dims [[buffer(6)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= rows) return;

    uint offset = id * cols;
    float mean = mean_saved[id];
    float rstd = rstd_saved[id];
    float inv_cols = 1.0f / float(cols);

    // Compute partial sums needed for grad_input
    float sum_grad_gamma_xhat = 0.0f;
    float sum_grad = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float xhat = (input[offset + j] - mean) * rstd;
        sum_grad_gamma_xhat += grad_output[offset + j] * gamma[j] * xhat;
        sum_grad += grad_output[offset + j] * gamma[j];
    }

    // grad_input
    for (uint j = 0; j < cols; j++) {
        float xhat = (input[offset + j] - mean) * rstd;
        float grad_xhat = grad_output[offset + j] * gamma[j];
        grad_input[offset + j] = rstd * (grad_xhat - inv_cols * (sum_grad + xhat * sum_grad_gamma_xhat));
    }
}

// Accumulate gamma gradients across rows
kernel void layernorm_grad_gamma_beta_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* mean_saved [[buffer(2)]],
    device const float* rstd_saved [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],
    device float* grad_beta [[buffer(5)]],
    device const uint* dims [[buffer(6)]],  // [rows, cols]
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];

    if (id >= cols) return;

    float dg = 0.0f;
    float db = 0.0f;
    for (uint i = 0; i < rows; i++) {
        float xhat = (input[i * cols + id] - mean_saved[i]) * rstd_saved[i];
        dg += grad_output[i * cols + id] * xhat;
        db += grad_output[i * cols + id];
    }
    grad_gamma[id] = dg;
    grad_beta[id] = db;
}

// RMSNorm forward: y = gamma * x / sqrt(mean(x^2) + eps)
kernel void rmsnorm_forward_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* rstd_out [[buffer(3)]],
    device const uint* dims [[buffer(4)]],  // [rows, cols]
    device const float* eps_buf [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];
    float eps = eps_buf[0];

    if (id >= rows) return;

    uint offset = id * cols;

    float sum_sq = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum_sq += input[offset + j] * input[offset + j];
    }
    float rstd = rsqrt(sum_sq / float(cols) + eps);
    rstd_out[id] = rstd;

    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * input[offset + j] * rstd;
    }
}
