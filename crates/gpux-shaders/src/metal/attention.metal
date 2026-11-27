#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Causal self-attention kernels for transformer models
// ============================================================================

// Apply causal mask: set upper-triangular elements to -inf
// scores: [batch*heads, seq_len, seq_len]
kernel void causal_mask_f32(
    device float* scores [[buffer(0)]],
    device const uint* dims [[buffer(1)]],  // [batch_heads, seq_len]
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_heads = dims[0];
    uint seq_len = dims[1];

    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch_idx >= batch_heads || row >= seq_len || col >= seq_len) return;

    if (col > row) {
        scores[batch_idx * seq_len * seq_len + row * seq_len + col] = -INFINITY;
    }
}

// Scale attention scores: scores = scores / sqrt(d_k)
kernel void attention_scale_f32(
    device float* scores [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    scores[id] *= scale[0];
}

// Fused attention score: Q @ K^T with scaling and causal mask
// Q: [B*H, T, D], K: [B*H, T, D]
// output: [B*H, T, T]
kernel void attention_scores_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [BH, T, D]
    device const float* scale [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint BH = dims[0];
    uint T = dims[1];
    uint D = dims[2];

    uint bh = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (bh >= BH || row >= T || col >= T) return;

    // Causal mask
    if (col > row) {
        scores[bh * T * T + row * T + col] = -INFINITY;
        return;
    }

    float s = scale[0];
    float sum = 0.0f;
    uint q_offset = bh * T * D + row * D;
    uint k_offset = bh * T * D + col * D;

    for (uint d = 0; d < D; d++) {
        sum += Q[q_offset + d] * K[k_offset + d];
    }

    scores[bh * T * T + row * T + col] = sum * s;
}

// Attention output: attn_weights @ V
// weights: [B*H, T, T], V: [B*H, T, D]
// output: [B*H, T, D]
kernel void attention_weighted_values_f32(
    device const float* weights [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [BH, T, D]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BH = dims[0];
    uint T = dims[1];
    uint D = dims[2];

    uint bh = gid.z;
    uint row = gid.y;
    uint d = gid.x;

    if (bh >= BH || row >= T || d >= D) return;

    float sum = 0.0f;
    uint w_offset = bh * T * T + row * T;
    uint v_offset = bh * T * D;

    for (uint t = 0; t < T; t++) {
        sum += weights[w_offset + t] * V[v_offset + t * D + d];
    }

    output[bh * T * D + row * D + d] = sum;
}

// Reshape and transpose for multi-head attention
// input: [B, T, H*D] -> output: [B*H, T, D]
kernel void reshape_for_attention_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [B, T, H, D]
    uint3 gid [[thread_position_in_grid]]
) {
    uint B = dims[0];
    uint T = dims[1];
    uint H = dims[2];
    uint D = dims[3];

    uint b = gid.z / H;
    uint h = gid.z % H;
    uint t = gid.y;
    uint d = gid.x;

    if (b >= B || t >= T || d >= D) return;

    uint in_idx = b * T * H * D + t * H * D + h * D + d;
    uint out_idx = (b * H + h) * T * D + t * D + d;

    output[out_idx] = input[in_idx];
}

// Inverse: [B*H, T, D] -> [B, T, H*D]
kernel void reshape_from_attention_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],  // [B, T, H, D]
    uint3 gid [[thread_position_in_grid]]
) {
    uint B = dims[0];
    uint T = dims[1];
    uint H = dims[2];
    uint D = dims[3];

    uint b = gid.z / H;
    uint h = gid.z % H;
    uint t = gid.y;
    uint d = gid.x;

    if (b >= B || t >= T || d >= D) return;

    uint in_idx = (b * H + h) * T * D + t * D + d;
    uint out_idx = b * T * H * D + t * H * D + h * D + d;

    output[out_idx] = input[in_idx];
}

// GQA attention scores: Q @ K^T with scaling and causal mask
// Q: [B*H_q, T, D], K: [B*H_kv, T, D] where H_q = H_kv * n_rep
// output: [B*H_q, T, T]
kernel void gqa_attention_scores_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [B*H_q, T, D, n_rep]
    device const float* scale [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint BHq = dims[0];
    uint T = dims[1];
    uint D = dims[2];
    uint n_rep = dims[3];

    uint bhq = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (bhq >= BHq || row >= T || col >= T) return;

    if (col > row) {
        scores[bhq * T * T + row * T + col] = -INFINITY;
        return;
    }

    uint bhkv = bhq / n_rep;
    float s = scale[0];
    float sum = 0.0f;
    uint q_offset = bhq * T * D + row * D;
    uint k_offset = bhkv * T * D + col * D;

    for (uint d = 0; d < D; d++) {
        sum += Q[q_offset + d] * K[k_offset + d];
    }

    scores[bhq * T * T + row * T + col] = sum * s;
}

// GQA attention weighted values: attn_weights @ V
// weights: [B*H_q, T, T], V: [B*H_kv, T, D] where H_q = H_kv * n_rep
// output: [B*H_q, T, D]
kernel void gqa_attention_weighted_values_f32(
    device const float* weights [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [B*H_q, T, D, n_rep]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BHq = dims[0];
    uint T = dims[1];
    uint D = dims[2];
    uint n_rep = dims[3];

    uint bhq = gid.z;
    uint row = gid.y;
    uint d = gid.x;

    if (bhq >= BHq || row >= T || d >= D) return;

    uint bhkv = bhq / n_rep;
    float sum = 0.0f;
    uint w_offset = bhq * T * T + row * T;
    uint v_offset = bhkv * T * D;

    for (uint t = 0; t < T; t++) {
        sum += weights[w_offset + t] * V[v_offset + t * D + d];
    }

    output[bhq * T * D + row * D + d] = sum;
}

// Rope positional encoding
kernel void rope_f32(
    device float* x [[buffer(0)]],
    device const uint* dims [[buffer(1)]],  // [B*H, T, D]
    device const float* inv_freq [[buffer(2)]],  // [D/2]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BH = dims[0];
    uint T = dims[1];
    uint D = dims[2];

    uint bh = gid.z;
    uint t = gid.y;
    uint d = gid.x;

    if (bh >= BH || t >= T || d >= D / 2) return;

    float freq = float(t) * inv_freq[d];
    float cos_val = cos(freq);
    float sin_val = sin(freq);

    uint idx = bh * T * D + t * D;
    float x0 = x[idx + d];
    float x1 = x[idx + d + D / 2];

    x[idx + d] = x0 * cos_val - x1 * sin_val;
    x[idx + d + D / 2] = x0 * sin_val + x1 * cos_val;
}
