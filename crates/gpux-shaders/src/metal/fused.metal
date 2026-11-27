#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused kernels for inference performance
// ============================================================================

// Fused RoPE: reads from input, applies rotation, writes to output.
// Eliminates the separate copy_f32 + rope_f32 two-dispatch pattern.
// input: [B*H, T, D], output: [B*H, T, D]
kernel void rope_fused_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],       // [B*H, T, D]
    device const float* inv_freq [[buffer(3)]],  // [D/2]
    uint3 gid [[thread_position_in_grid]]
) {
    uint BH = dims[0];
    uint T  = dims[1];
    uint D  = dims[2];

    uint bh = gid.z;
    uint t  = gid.y;
    uint d  = gid.x;

    if (bh >= BH || t >= T || d >= D / 2) return;

    float freq = float(t) * inv_freq[d];
    float cos_val = cos(freq);
    float sin_val = sin(freq);

    uint idx = bh * T * D + t * D;
    float x0 = input[idx + d];
    float x1 = input[idx + d + D / 2];

    output[idx + d]         = x0 * cos_val - x1 * sin_val;
    output[idx + d + D / 2] = x0 * sin_val + x1 * cos_val;
}

// Fused RoPE with offset: reads from input, applies rotation with position offset.
// input: [B*H, T, D], output: [B*H, T, D]
kernel void rope_with_offset_fused_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]],       // [B*H, T, D, offset]
    device const float* inv_freq [[buffer(3)]],  // [D/2]
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
    float x0 = input[idx + d];
    float x1 = input[idx + d + D / 2];

    output[idx + d]         = x0 * cos_val - x1 * sin_val;
    output[idx + d + D / 2] = x0 * sin_val + x1 * cos_val;
}

// Fused SiLU + elementwise multiply: out[i] = silu(gate[i]) * up[i]
kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float g = gate[id];
    float sigmoid = 1.0f / (1.0f + exp(-g));
    out[id] = (g * sigmoid) * up[id];
}

// ============================================================================
// Fused gate+up+SiLU matvec: out[i] = silu(dot(x, W_gate[i])) * dot(x, W_up[i])
// One SIMD group computes one output row, reading x once for both projections.
// 4 SIMD groups per threadgroup = 4 rows per threadgroup.
// Eliminates 3 dispatches (gate matvec + up matvec + silu_mul) → 1 dispatch.
// ============================================================================

constant uint FUSED_SG_PER_TG = 4;

kernel void fused_gate_up_silu_f16w_f32(
    device const float* x [[buffer(0)]],
    device const half*  W_gate [[buffer(1)]],
    device const half*  W_up [[buffer(2)]],
    device float*       out [[buffer(3)]],
    device const uint*  dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];

    uint row = tgid * FUSED_SG_PER_TG + sgid;
    if (row >= N) return;

    device const half4* wg = (device const half4*)(W_gate + row * K);
    device const half4* wu = (device const half4*)(W_up + row * K);
    device const float4* x4 = (device const float4*)x;

    uint K4 = K / 4;
    float sum_gate = 0.0f, sum_up = 0.0f;

    for (uint k = lane; k < K4; k += 32) {
        float4 xv = x4[k];
        half4 gv = wg[k];
        half4 uv = wu[k];
        sum_gate += xv.x * float(gv.x) + xv.y * float(gv.y) + xv.z * float(gv.z) + xv.w * float(gv.w);
        sum_up   += xv.x * float(uv.x) + xv.y * float(uv.y) + xv.z * float(uv.z) + xv.w * float(uv.w);
    }

    sum_gate = simd_sum(sum_gate);
    sum_up   = simd_sum(sum_up);

    if (lane == 0) {
        float sigmoid = 1.0f / (1.0f + exp(-sum_gate));
        out[row] = (sum_gate * sigmoid) * sum_up;
    }
}

kernel void fused_gate_up_silu_f32(
    device const float* x [[buffer(0)]],
    device const float* W_gate [[buffer(1)]],
    device const float* W_up [[buffer(2)]],
    device float*       out [[buffer(3)]],
    device const uint*  dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];

    uint row = tgid * FUSED_SG_PER_TG + sgid;
    if (row >= N) return;

    device const float* wg = W_gate + row * K;
    device const float* wu = W_up + row * K;

    float sum_gate = 0.0f, sum_up = 0.0f;

    for (uint k = lane; k < K; k += 32) {
        float xk = x[k];
        sum_gate += xk * float(wg[k]);
        sum_up   += xk * float(wu[k]);
    }

    sum_gate = simd_sum(sum_gate);
    sum_up   = simd_sum(sum_up);

    if (lane == 0) {
        float sigmoid = 1.0f / (1.0f + exp(-sum_gate));
        out[row] = (sum_gate * sigmoid) * sum_up;
    }
}

// ============================================================================
// SIMD group matvec: y = x @ W^T (specialized for M=1 decode)
// 32 threads (1 SIMD group) collaborate on each output row via simd_sum.
// Each SIMD group computes 2 rows, sharing the x vector read.
// 4 SIMD groups per threadgroup = 8 rows per threadgroup.
// ============================================================================

constant uint SIMD_W = 32;
constant uint SG_PER_TG = 4;
constant uint ROWS_PER_SG = 4;
constant uint ROWS_PER_TG = SG_PER_TG * ROWS_PER_SG;  // 16

kernel void matvec_a_bt_f32(
    device const float* x [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];

    uint base = tgid * ROWS_PER_TG + sgid * ROWS_PER_SG;
    device const float4* x4 = (device const float4*)x;
    device const float4* w0 = (device const float4*)(W + min(base,     N - 1) * K);
    device const float4* w1 = (device const float4*)(W + min(base + 1, N - 1) * K);
    device const float4* w2 = (device const float4*)(W + min(base + 2, N - 1) * K);
    device const float4* w3 = (device const float4*)(W + min(base + 3, N - 1) * K);

    uint K4 = K / 4;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint k = lane; k < K4; k += SIMD_W) {
        float4 xv = x4[k];
        float4 v0 = w0[k], v1 = w1[k], v2 = w2[k], v3 = w3[k];
        s0 += xv.x*v0.x + xv.y*v0.y + xv.z*v0.z + xv.w*v0.w;
        s1 += xv.x*v1.x + xv.y*v1.y + xv.z*v1.z + xv.w*v1.w;
        s2 += xv.x*v2.x + xv.y*v2.y + xv.z*v2.z + xv.w*v2.w;
        s3 += xv.x*v3.x + xv.y*v3.y + xv.z*v3.z + xv.w*v3.w;
    }

    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);

    if (lane == 0) {
        if (base     < N) y[base]     = s0;
        if (base + 1 < N) y[base + 1] = s1;
        if (base + 2 < N) y[base + 2] = s2;
        if (base + 3 < N) y[base + 3] = s3;
    }
}

kernel void matvec_a_bt_bias_f32(
    device const float* x [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const uint* dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];

    uint base = tgid * ROWS_PER_TG + sgid * ROWS_PER_SG;
    device const float4* x4 = (device const float4*)x;
    device const float4* w0 = (device const float4*)(W + min(base,     N - 1) * K);
    device const float4* w1 = (device const float4*)(W + min(base + 1, N - 1) * K);
    device const float4* w2 = (device const float4*)(W + min(base + 2, N - 1) * K);
    device const float4* w3 = (device const float4*)(W + min(base + 3, N - 1) * K);

    uint K4 = K / 4;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint k = lane; k < K4; k += SIMD_W) {
        float4 xv = x4[k];
        float4 v0 = w0[k], v1 = w1[k], v2 = w2[k], v3 = w3[k];
        s0 += xv.x*v0.x + xv.y*v0.y + xv.z*v0.z + xv.w*v0.w;
        s1 += xv.x*v1.x + xv.y*v1.y + xv.z*v1.z + xv.w*v1.w;
        s2 += xv.x*v2.x + xv.y*v2.y + xv.z*v2.z + xv.w*v2.w;
        s3 += xv.x*v3.x + xv.y*v3.y + xv.z*v3.z + xv.w*v3.w;
    }

    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);

    if (lane == 0) {
        if (base     < N) y[base]     = s0 + bias[base];
        if (base + 1 < N) y[base + 1] = s1 + bias[base + 1];
        if (base + 2 < N) y[base + 2] = s2 + bias[base + 2];
        if (base + 3 < N) y[base + 3] = s3 + bias[base + 3];
    }
}

kernel void matvec_a_bt_bias_f16w_f32(
    device const float* x [[buffer(0)]],
    device const half*  W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const uint* dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];

    uint base = tgid * ROWS_PER_TG + sgid * ROWS_PER_SG;
    device const float4* x4 = (device const float4*)x;
    device const half4* w0 = (device const half4*)(W + min(base,     N - 1) * K);
    device const half4* w1 = (device const half4*)(W + min(base + 1, N - 1) * K);
    device const half4* w2 = (device const half4*)(W + min(base + 2, N - 1) * K);
    device const half4* w3 = (device const half4*)(W + min(base + 3, N - 1) * K);

    uint K4 = K / 4;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint k = lane; k < K4; k += SIMD_W) {
        float4 xv = x4[k];
        half4 v0 = w0[k], v1 = w1[k], v2 = w2[k], v3 = w3[k];
        s0 += xv.x*float(v0.x) + xv.y*float(v0.y) + xv.z*float(v0.z) + xv.w*float(v0.w);
        s1 += xv.x*float(v1.x) + xv.y*float(v1.y) + xv.z*float(v1.z) + xv.w*float(v1.w);
        s2 += xv.x*float(v2.x) + xv.y*float(v2.y) + xv.z*float(v2.z) + xv.w*float(v2.w);
        s3 += xv.x*float(v3.x) + xv.y*float(v3.y) + xv.z*float(v3.z) + xv.w*float(v3.w);
    }

    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);

    if (lane == 0) {
        if (base     < N) y[base]     = s0 + bias[base];
        if (base + 1 < N) y[base + 1] = s1 + bias[base + 1];
        if (base + 2 < N) y[base + 2] = s2 + bias[base + 2];
        if (base + 3 < N) y[base + 3] = s3 + bias[base + 3];
    }
}

// Fused residual add + RMSNorm:
//   residual_out[row] = a[row] + b[row]
//   normed_out[row] = rmsnorm(residual_out[row]) * gamma
// One thread per row. Eliminates separate add + rmsnorm dispatches.
kernel void rmsnorm_residual_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device float* residual_out [[buffer(3)]],
    device float* normed_out [[buffer(4)]],
    device const uint* dims [[buffer(5)]],    // [rows, cols]
    device const float* eps_buf [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    uint rows = dims[0];
    uint cols = dims[1];
    float eps = eps_buf[0];

    if (id >= rows) return;

    uint offset = id * cols;

    float sum_sq = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float val = a[offset + j] + b[offset + j];
        residual_out[offset + j] = val;
        sum_sq += val * val;
    }

    float rstd = rsqrt(sum_sq / float(cols) + eps);
    for (uint j = 0; j < cols; j++) {
        normed_out[offset + j] = gamma[j] * residual_out[offset + j] * rstd;
    }
}

// ============================================================================
// Mixed-precision kernels: f16 weights, f32 activations
// ============================================================================

// Scalar matvec with f16 weights (baseline for comparison)
kernel void matvec_scalar_f16w_f32(
    device const float* x [[buffer(0)]],
    device const half*  W [[buffer(1)]],
    device float*       y [[buffer(2)]],
    device const uint*  dims [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint N = dims[0];
    uint K = dims[1];
    if (id >= N) return;
    device const half* w_row = W + id * K;
    float sum = 0.0f;
    for (uint k = 0; k < K; k += 4) {
        sum += x[k]   * float(w_row[k])
             + x[k+1] * float(w_row[k+1])
             + x[k+2] * float(w_row[k+2])
             + x[k+3] * float(w_row[k+3]);
    }
    y[id] = sum;
}

// SIMD matvec with f16 weights: y = x @ W^T, W is f16, x/y are f32
// Uses float4/half4 vectorized loads, 4 rows per SIMD group.
kernel void matvec_a_bt_f16w_f32(
    device const float* x [[buffer(0)]],
    device const half*  W [[buffer(1)]],
    device float*       y [[buffer(2)]],
    device const uint*  dims [[buffer(3)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];

    uint base = tgid * ROWS_PER_TG + sgid * ROWS_PER_SG;
    device const float4* x4 = (device const float4*)x;
    device const half4* w0 = (device const half4*)(W + min(base,     N - 1) * K);
    device const half4* w1 = (device const half4*)(W + min(base + 1, N - 1) * K);
    device const half4* w2 = (device const half4*)(W + min(base + 2, N - 1) * K);
    device const half4* w3 = (device const half4*)(W + min(base + 3, N - 1) * K);

    uint K4 = K / 4;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint k = lane; k < K4; k += SIMD_W) {
        float4 xv = x4[k];
        half4 v0 = w0[k], v1 = w1[k], v2 = w2[k], v3 = w3[k];
        s0 += xv.x*float(v0.x) + xv.y*float(v0.y) + xv.z*float(v0.z) + xv.w*float(v0.w);
        s1 += xv.x*float(v1.x) + xv.y*float(v1.y) + xv.z*float(v1.z) + xv.w*float(v1.w);
        s2 += xv.x*float(v2.x) + xv.y*float(v2.y) + xv.z*float(v2.z) + xv.w*float(v2.w);
        s3 += xv.x*float(v3.x) + xv.y*float(v3.y) + xv.z*float(v3.z) + xv.w*float(v3.w);
    }

    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);

    if (lane == 0) {
        if (base     < N) y[base]     = s0;
        if (base + 1 < N) y[base + 1] = s1;
        if (base + 2 < N) y[base + 2] = s2;
        if (base + 3 < N) y[base + 3] = s3;
    }
}

// Tiled matmul with f16 weights: C = A @ B^T, B is f16, A/C are f32
constant uint TILE_MIX = 32;

kernel void matmul_a_bt_f16w_f32(
    device const float* A [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    device const uint*  dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    threadgroup float As[TILE_MIX][TILE_MIX];
    threadgroup float Bs[TILE_MIX][TILE_MIX];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_MIX - 1) / TILE_MIX;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_MIX + lid.x;
        uint b_col = t * TILE_MIX + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[lid.y][lid.x] = (b_col < K && col < N) ? float(B[col * K + b_col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_MIX; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Embedding lookup with f16 table: reads f16, writes f32
kernel void embedding_forward_f16w_f32(
    device const uint* indices [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [total_tokens, dim]
    uint2 gid [[thread_position_in_grid]]
) {
    uint total_tokens = dims[0];
    uint dim = dims[1];

    uint token = gid.y;
    uint d = gid.x;

    if (token >= total_tokens || d >= dim) return;

    uint idx = indices[token];
    output[token * dim + d] = float(weight[idx * dim + d]);
}

// RMSNorm with f16 gamma: f32 input/output, f16 gamma weights
// SIMD-cooperative RMSNorm. Same layout as rmsnorm_residual_f16w_f32:
// 4 SIMD groups (128 threads) per row.
kernel void rmsnorm_forward_f16w_f32(
    device const float* input    [[buffer(0)]],
    device const half*  gamma    [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    device float*       rstd_out [[buffer(3)]],
    device const uint*  dims     [[buffer(4)]],
    device const float* eps_buf  [[buffer(5)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint rows = dims[0];
    uint cols = dims[1];
    float eps = eps_buf[0];

    if (tgid >= rows) return;

    uint offset = tgid * cols;
    uint tid = sgid * 32 + lane;          // 0..127

    float sum_sq = 0.0f;
    for (uint j = tid; j < cols; j += 128) {
        float v = input[offset + j];
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);

    threadgroup float sg_sums[4];
    if (lane == 0) sg_sums[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float tg_rstd;
    if (sgid == 0) {
        float total = (lane < 4) ? sg_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            float rstd = rsqrt(total / float(cols) + eps);
            tg_rstd = rstd;
            rstd_out[tgid] = rstd;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rstd = tg_rstd;

    for (uint j = tid; j < cols; j += 128) {
        output[offset + j] = float(gamma[j]) * input[offset + j] * rstd;
    }
}

// Fused residual add + RMSNorm with f16 gamma
// SIMD-cooperative add+RMSNorm.
// Layout: 4 SIMD groups (128 threads) per row.
// Each lane processes cols/(4*32) elements of the row, contributes to a SIMD-local
// sum-of-squares, then a tiny threadgroup-memory cross-SG reduction yields rstd.
// Phase 2: each lane writes its share of normed_out.
//
// For decode (rows=1) this gives ~128× more parallelism than the one-thread-per-row
// scalar version that was 35% of GPU time per the profiler.
// (rmsnorm_residual / rmsnorm_forward use 4 SG/TG = 128 threads, inlined)

kernel void rmsnorm_residual_f16w_f32(
    device const float* a            [[buffer(0)]],
    device const float* b            [[buffer(1)]],
    device const half*  gamma        [[buffer(2)]],
    device float*       residual_out [[buffer(3)]],
    device float*       normed_out   [[buffer(4)]],
    device const uint*  dims         [[buffer(5)]],
    device const float* eps_buf      [[buffer(6)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint rows = dims[0];
    uint cols = dims[1];
    float eps = eps_buf[0];

    if (tgid >= rows) return;

    uint offset = tgid * cols;
    uint tid = sgid * 32 + lane;          // 0..127

    // Pass 1: each thread strides over the row, computes (a+b) and val^2.
    // We also write residual_out here so pass 2 doesn't need to recompute (a+b).
    float sum_sq = 0.0f;
    for (uint j = tid; j < cols; j += 128) {
        float val = a[offset + j] + b[offset + j];
        residual_out[offset + j] = val;
        sum_sq += val * val;
    }

    // Reduce within SIMD group, then across SGs via threadgroup memory.
    sum_sq = simd_sum(sum_sq);

    threadgroup float sg_sums[4];
    if (lane == 0) sg_sums[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG reduction in SG 0; broadcast rstd via threadgroup memory.
    threadgroup float tg_rstd;
    if (sgid == 0) {
        float total = (lane < 4) ? sg_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) tg_rstd = rsqrt(total / float(cols) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rstd = tg_rstd;

    // Pass 2: each thread writes its share of normed_out.
    for (uint j = tid; j < cols; j += 128) {
        normed_out[offset + j] = float(gamma[j]) * residual_out[offset + j] * rstd;
    }
}
