#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Optimizer kernels (AdamW, SGD)
// ============================================================================

// AdamW update step
// params, grads, m (first moment), v (second moment): all [N]
// config: [lr, beta1, beta2, eps, weight_decay, beta1_t, beta2_t, N]
kernel void adamw_step_f32(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    device const float* config [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint N = as_type<uint>(config[7]);
    if (id >= N) return;

    float lr = config[0];
    float beta1 = config[1];
    float beta2 = config[2];
    float eps = config[3];
    float weight_decay = config[4];
    float beta1_t = config[5];
    float beta2_t = config[6];

    float g = grads[id];

    // Update biased first moment estimate
    float m_new = beta1 * m[id] + (1.0f - beta1) * g;
    m[id] = m_new;

    // Update biased second raw moment estimate
    float v_new = beta2 * v[id] + (1.0f - beta2) * g * g;
    v[id] = v_new;

    // Bias correction
    float m_hat = m_new / (1.0f - beta1_t);
    float v_hat = v_new / (1.0f - beta2_t);

    // Weight decay (decoupled)
    float p = params[id];
    p = p - lr * weight_decay * p;

    // Parameter update
    p = p - lr * m_hat / (sqrt(v_hat) + eps);
    params[id] = p;
}

// SGD with momentum
// params, grads, velocity: all [N]
// config: [lr, momentum, weight_decay]
kernel void sgd_momentum_step_f32(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* velocity [[buffer(2)]],
    device const float* config [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float lr = config[0];
    float momentum = config[1];
    float weight_decay = config[2];

    float g = grads[id];

    // Weight decay
    if (weight_decay != 0.0f) {
        g = g + weight_decay * params[id];
    }

    // Momentum
    float v = momentum * velocity[id] + g;
    velocity[id] = v;

    // Update
    params[id] = params[id] - lr * v;
}

// Gradient clipping by norm (per-element after computing global norm)
kernel void clip_grad_norm_f32(
    device float* grads [[buffer(0)]],
    device const float* clip_coeff [[buffer(1)]],  // min(max_norm / total_norm, 1.0)
    uint id [[thread_position_in_grid]]
) {
    grads[id] *= clip_coeff[0];
}

// Compute squared norm of a buffer (partial reduction)
kernel void squared_norm_f32(
    device const float* data [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* n [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint count = n[0];
    float sum = 0.0f;

    for (uint i = id; i < count; i += tg_size * 256) {
        float val = data[i];
        sum += val * val;
    }

    threadgroup float shared[256];
    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[0] = shared[0];
    }
}
