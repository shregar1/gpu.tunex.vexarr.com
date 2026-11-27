#include <metal_stdlib>
using namespace metal;

// ============================================================================
// KV Cache kernels for autoregressive generation
// ============================================================================

// Append new K/V vectors to cache at position offset
// new_kv: [B*H_kv, new_tokens, D], cache: [B*H_kv, max_seq, D]
kernel void kv_cache_append_f32(
    device const float* new_kv [[buffer(0)]],
    device float*       cache  [[buffer(1)]],
    device const uint*  dims   [[buffer(2)]],  // [B_H_kv, max_seq, D, cache_pos, new_tokens]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BHkv     = dims[0];
    uint max_seq  = dims[1];
    uint D        = dims[2];
    uint cache_pos = dims[3];
    uint new_tok  = dims[4];

    uint bh = gid.z;
    uint t  = gid.y;
    uint d  = gid.x;

    if (bh >= BHkv || t >= new_tok || d >= D) return;

    uint src_idx = bh * new_tok * D + t * D + d;
    uint dst_idx = bh * max_seq * D + (cache_pos + t) * D + d;

    cache[dst_idx] = new_kv[src_idx];
}

// Fused K+V cache append: writes both K and V in one dispatch
// new_k/new_v: [B*H_kv, new_tokens, D], k_cache/v_cache: [B*H_kv, max_seq, D]
kernel void kv_cache_append_fused_f32(
    device const float* new_k   [[buffer(0)]],
    device const float* new_v   [[buffer(1)]],
    device float*       k_cache [[buffer(2)]],
    device float*       v_cache [[buffer(3)]],
    device const uint*  dims    [[buffer(4)]],  // [B_H_kv, max_seq, D, cache_pos, new_tokens]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BHkv     = dims[0];
    uint max_seq  = dims[1];
    uint D        = dims[2];
    uint cache_pos = dims[3];
    uint new_tok  = dims[4];

    uint bh = gid.z;
    uint t  = gid.y;
    uint d  = gid.x;

    if (bh >= BHkv || t >= new_tok || d >= D) return;

    uint src_idx = bh * new_tok * D + t * D + d;
    uint dst_idx = bh * max_seq * D + (cache_pos + t) * D + d;

    k_cache[dst_idx] = new_k[src_idx];
    v_cache[dst_idx] = new_v[src_idx];
}

// RoPE with absolute position offset for KV cache decode
kernel void rope_with_offset_f32(
    device float* x [[buffer(0)]],
    device const uint* dims [[buffer(1)]],       // [B*H, T, D, offset]
    device const float* inv_freq [[buffer(2)]],  // [D/2]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BH     = dims[0];
    uint T      = dims[1];
    uint D      = dims[2];
    uint offset = dims[3];

    uint bh = gid.z;
    uint t  = gid.y;
    uint d  = gid.x;

    if (bh >= BH || t >= T || d >= D / 2) return;

    float freq = float(offset + t) * inv_freq[d];
    float cos_val = cos(freq);
    float sin_val = sin(freq);

    uint idx = bh * T * D + t * D;
    float x0 = x[idx + d];
    float x1 = x[idx + d + D / 2];

    x[idx + d]         = x0 * cos_val - x1 * sin_val;
    x[idx + d + D / 2] = x0 * sin_val + x1 * cos_val;
}

// Fused RoPE + KV cache append.
//   Inputs:
//     qk_in [n_qk_heads, T, D]        — Q+K concatenated (n_qk = n_q + n_kv), pre-RoPE
//     v_in  [n_kv_heads, T, D]        — V, no RoPE
//   Outputs:
//     q_out   [n_q_heads, T, D]                — RoPE-applied Q
//     k_cache [n_kv_heads, max_seq, D]         — RoPE-applied K written at cache_pos
//     v_cache [n_kv_heads, max_seq, D]         — V written at cache_pos
//   Dims layout: [n_q_heads, n_kv_heads, T, D, cache_pos, max_seq]
//
// Replaces two dispatches (rope_with_offset_fused + kv_cache_append_fused) plus
// eliminates the temporary roped-K buffer write/read.
// Each thread handles one RoPE pair (d_half, d_half+D/2) for one head/token.
kernel void rope_kv_append_fused_f32(
    device const float* qk_in    [[buffer(0)]],
    device const float* v_in     [[buffer(1)]],
    device float*       q_out    [[buffer(2)]],
    device float*       k_cache  [[buffer(3)]],
    device float*       v_cache  [[buffer(4)]],
    device const float* inv_freq [[buffer(5)]],
    device const uint*  dims     [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint n_q_heads  = dims[0];
    uint n_kv_heads = dims[1];
    uint T          = dims[2];
    uint D          = dims[3];
    uint cache_pos  = dims[4];
    uint max_seq    = dims[5];

    uint d_half = gid.x;     // 0..D/2
    uint t      = gid.y;     // 0..T
    uint h      = gid.z;     // 0..(n_q + 2*n_kv)

    uint n_qk_heads = n_q_heads + n_kv_heads;
    uint n_total    = n_q_heads + 2u * n_kv_heads;

    if (d_half >= D / 2 || t >= T || h >= n_total) return;

    if (h < n_qk_heads) {
        // Q or K head: apply RoPE to (qk_in[h, t, d_half], qk_in[h, t, d_half+D/2]).
        float freq = float(cache_pos + t) * inv_freq[d_half];
        float c = cos(freq);
        float s = sin(freq);

        uint qk_base = h * T * D + t * D;
        float x0 = qk_in[qk_base + d_half];
        float x1 = qk_in[qk_base + d_half + D / 2];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;

        if (h < n_q_heads) {
            uint q_base = h * T * D + t * D;
            q_out[q_base + d_half]         = r0;
            q_out[q_base + d_half + D / 2] = r1;
        } else {
            uint kv_h = h - n_q_heads;
            uint k_base = kv_h * max_seq * D + (cache_pos + t) * D;
            k_cache[k_base + d_half]         = r0;
            k_cache[k_base + d_half + D / 2] = r1;
        }
    } else {
        // V head: no RoPE, just copy (d_half and d_half+D/2 in one thread for symmetry).
        uint kv_h = h - n_qk_heads;
        uint v_in_base    = kv_h * T * D + t * D;
        uint v_cache_base = kv_h * max_seq * D + (cache_pos + t) * D;
        v_cache[v_cache_base + d_half]         = v_in[v_in_base + d_half];
        v_cache[v_cache_base + d_half + D / 2] = v_in[v_in_base + d_half + D / 2];
    }
}

// ============================================================================
// Fused decode attention: scores + softmax + weighted values in one dispatch.
// Uses online softmax — no intermediate buffers needed.
// Multiple SIMD groups per query head partition T_kv for parallelism.
// After local processing, merge online softmax states via threadgroup memory.
// ============================================================================

constant uint ATTN_SG_PER_HEAD = 8;   // SIMD groups per query head
constant uint ATTN_TG_THREADS = ATTN_SG_PER_HEAD * 32;  // 256 threads per TG

kernel void fused_attention_decode_f32(
    device const float* Q       [[buffer(0)]],  // [B*H_q, 1, D]
    device const float* K_cache [[buffer(1)]],  // [B*H_kv, max_seq, D]
    device const float* V_cache [[buffer(2)]],  // [B*H_kv, max_seq, D]
    device float*       output  [[buffer(3)]],  // [B*H_q, 1, D]
    device const uint*  dims    [[buffer(4)]],  // [B*H_q, T_kv, D, n_rep, max_seq]
    device const float* scale   [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint BHq     = dims[0];
    uint T_kv    = dims[1];
    uint D       = dims[2];
    uint n_rep   = dims[3];
    uint max_seq = dims[4];
    float s      = scale[0];

    uint bhq = tgid;
    if (bhq >= BHq) return;

    uint bhkv = bhq / n_rep;
    uint q_base = bhq * D;
    uint kv_stride = max_seq * D;

    // Shared memory for merging online softmax states across SIMD groups
    // Each SG stores: m (1 float), l (1 float), o (D floats) — distributed across lanes
    threadgroup float sg_m[ATTN_SG_PER_HEAD];
    threadgroup float sg_l[ATTN_SG_PER_HEAD];
    // Each lane stores its o elements; we use a flat array [SG][32] (lane-indexed)
    threadgroup float sg_o0[ATTN_SG_PER_HEAD * 32];
    threadgroup float sg_o1[ATTN_SG_PER_HEAD * 32];

    // Each SIMD group processes a partition of T_kv
    uint chunk = (T_kv + ATTN_SG_PER_HEAD - 1) / ATTN_SG_PER_HEAD;
    uint j_start = sgid * chunk;
    uint j_end = min(j_start + chunk, T_kv);

    float o0 = 0.0f, o1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (uint j = j_start; j < j_end; j++) {
        uint k_base = bhkv * kv_stride + j * D;
        float partial = 0.0f;
        for (uint di = lane; di < D; di += 32) {
            partial += Q[q_base + di] * K_cache[k_base + di];
        }
        float score = simd_sum(partial) * s;

        float m_new = max(m, score);
        float correction = fast::exp(m - m_new);
        float exp_score = fast::exp(score - m_new);
        l = l * correction + exp_score;
        m = m_new;

        uint v_base = bhkv * kv_stride + j * D;
        if (lane < D) {
            o0 = o0 * correction + exp_score * V_cache[v_base + lane];
        }
        if (lane + 32 < D) {
            o1 = o1 * correction + exp_score * V_cache[v_base + lane + 32];
        }
    }

    // Store local state to shared memory
    if (lane == 0) {
        sg_m[sgid] = m;
        sg_l[sgid] = l;
    }
    sg_o0[sgid * 32 + lane] = o0;
    sg_o1[sgid * 32 + lane] = o1;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD group 0 merges all states
    if (sgid == 0) {
        float final_m = sg_m[0];
        float final_l = sg_l[0];
        float fo0 = sg_o0[lane];
        float fo1 = sg_o1[lane];

        for (uint p = 1; p < ATTN_SG_PER_HEAD; p++) {
            float pm = sg_m[p];
            float pl = sg_l[p];

            float m_new = max(final_m, pm);
            float c_old = fast::exp(final_m - m_new);
            float c_new = fast::exp(pm - m_new);

            fo0 = fo0 * c_old + sg_o0[p * 32 + lane] * c_new;
            fo1 = fo1 * c_old + sg_o1[p * 32 + lane] * c_new;
            final_l = final_l * c_old + pl * c_new;
            final_m = m_new;
        }

        float inv_l = 1.0f / final_l;
        uint out_base = bhq * D;
        if (lane < D)      output[out_base + lane]      = fo0 * inv_l;
        if (lane + 32 < D) output[out_base + lane + 32] = fo1 * inv_l;
    }
}

// Cached GQA attention scores: Q against K_cache
// Q: [B*H_q, T_q, D], K_cache: [B*H_kv, max_seq, D]
// scores: [B*H_q, T_q, T_kv]
kernel void gqa_attention_scores_cached_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device float*       scores  [[buffer(2)]],
    device const uint*  dims    [[buffer(3)]],  // [B*H_q, T_q, T_kv, D, n_rep, max_seq]
    device const float* scale   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint BHq     = dims[0];
    uint T_q     = dims[1];
    uint T_kv    = dims[2];
    uint D       = dims[3];
    uint n_rep   = dims[4];
    uint max_seq = dims[5];

    uint bhq = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (bhq >= BHq || row >= T_q || col >= T_kv) return;

    // Causal mask: query absolute position = (T_kv - T_q) + row
    uint q_abs_pos = (T_kv - T_q) + row;
    if (col > q_abs_pos) {
        scores[bhq * T_q * T_kv + row * T_kv + col] = -INFINITY;
        return;
    }

    uint bhkv = bhq / n_rep;
    float s = scale[0];
    float sum = 0.0f;
    uint q_offset = bhq * T_q * D + row * D;
    uint k_offset = bhkv * max_seq * D + col * D;

    for (uint d = 0; d < D; d++) {
        sum += Q[q_offset + d] * K_cache[k_offset + d];
    }

    scores[bhq * T_q * T_kv + row * T_kv + col] = sum * s;
}

// Cached GQA weighted values: weights @ V_cache
// weights: [B*H_q, T_q, T_kv], V_cache: [B*H_kv, max_seq, D]
// output: [B*H_q, T_q, D]
kernel void gqa_attention_weighted_values_cached_f32(
    device const float* weights [[buffer(0)]],
    device const float* V_cache [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    device const uint*  dims    [[buffer(3)]],  // [B*H_q, T_q, T_kv, D, n_rep, max_seq]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BHq     = dims[0];
    uint T_q     = dims[1];
    uint T_kv    = dims[2];
    uint D       = dims[3];
    uint n_rep   = dims[4];
    uint max_seq = dims[5];

    uint bhq = gid.z;
    uint row = gid.y;
    uint d   = gid.x;

    if (bhq >= BHq || row >= T_q || d >= D) return;

    uint bhkv = bhq / n_rep;
    float sum = 0.0f;
    uint w_offset = bhq * T_q * T_kv + row * T_kv;
    uint v_base   = bhkv * max_seq * D;

    for (uint t = 0; t < T_kv; t++) {
        sum += weights[w_offset + t] * V_cache[v_base + t * D + d];
    }

    output[bhq * T_q * D + row * D + d] = sum;
}
