#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Element-wise operations
// ============================================================================

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + b[id];
}

kernel void add_scalar_f32(
    device const float* a [[buffer(0)]],
    device const float* scalar [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + scalar[0];
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] - b[id];
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * b[id];
}

kernel void mul_scalar_f32(
    device const float* a [[buffer(0)]],
    device const float* scalar [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * scalar[0];
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] / b[id];
}

kernel void neg_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = -a[id];
}

kernel void exp_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = exp(a[id]);
}

kernel void log_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = log(a[id]);
}

kernel void sqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = sqrt(a[id]);
}

kernel void rsqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = rsqrt(a[id]);
}

kernel void tanh_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = tanh(a[id]);
}

kernel void relu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = max(a[id], 0.0f);
}

kernel void relu_backward_f32(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = input[id] > 0.0f ? grad[id] : 0.0f;
}

kernel void gelu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    out[id] = x * cdf;
}

kernel void gelu_backward_f32(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    float cube = 0.044715f * x * x * x;
    float inner = 0.7978845608f * (x + cube);
    float tanh_inner = tanh(inner);
    float dtanh = 1.0f - tanh_inner * tanh_inner;
    float dcdf = 0.5f * dtanh * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    float cdf = 0.5f * (1.0f + tanh_inner);
    out[id] = grad[id] * (cdf + x * dcdf);
}

kernel void silu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    float sigmoid = 1.0f / (1.0f + exp(-x));
    out[id] = x * sigmoid;
}

kernel void fill_f32(
    device float* out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = value[0];
}

kernel void pow_f32(
    device const float* a [[buffer(0)]],
    device const float* power [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = pow(a[id], power[0]);
}

kernel void clamp_f32(
    device const float* a [[buffer(0)]],
    device const float* bounds [[buffer(1)]],  // [min, max]
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = clamp(a[id], bounds[0], bounds[1]);
}
