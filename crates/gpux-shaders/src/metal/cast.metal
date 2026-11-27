#include <metal_stdlib>
using namespace metal;

kernel void cast_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = half(input[id]);
}

kernel void cast_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = float(input[id]);
}

// f16 elementwise ops with f32 accumulation

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = half(float(a[id]) + float(b[id]));
}

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = half(float(a[id]) * float(b[id]));
}

kernel void mul_scalar_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const float* scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = half(float(input[id]) * scalar[0]);
}

kernel void gelu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = float(input[id]);
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[id] = half(x * cdf);
}

// f16 matmul: reads f16, accumulates in f32, writes f16
// A: [M, K], B: [K, N], C: [M, N]
constant uint TILE_F16 = 32;

kernel void matmul_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    threadgroup float As[TILE_F16][TILE_F16];
    threadgroup float Bs[TILE_F16][TILE_F16];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_F16 - 1) / TILE_F16;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_F16 + lid.x;
        uint b_row = t * TILE_F16 + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;
        Bs[lid.y][lid.x] = (b_row < K && col < N) ? float(B[b_row * N + col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_F16; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = half(sum);
}

// f16 matmul: C = A @ B^T
kernel void matmul_a_bt_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    threadgroup float As[TILE_F16][TILE_F16];
    threadgroup float Bs[TILE_F16][TILE_F16];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_F16 - 1) / TILE_F16;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_F16 + lid.x;
        uint b_col = t * TILE_F16 + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;
        Bs[lid.y][lid.x] = (b_col < K && col < N) ? float(B[col * K + b_col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_F16; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = half(sum);
}

// f16 matmul: C = A^T @ B
kernel void matmul_at_b_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    threadgroup float As[TILE_F16][TILE_F16];
    threadgroup float Bs[TILE_F16][TILE_F16];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_F16 - 1) / TILE_F16;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_k = t * TILE_F16 + lid.x;
        uint b_k = t * TILE_F16 + lid.y;

        As[lid.y][lid.x] = (row < M && a_k < K) ? float(A[a_k * M + row]) : 0.0f;
        Bs[lid.y][lid.x] = (b_k < K && col < N) ? float(B[b_k * N + col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_F16; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = half(sum);
}

// LayerNorm for f16 (accumulates in f32)
kernel void layernorm_forward_f16(
    device const half* input [[buffer(0)]],
    device const half* gamma [[buffer(1)]],
    device const half* beta [[buffer(2)]],
    device half* output [[buffer(3)]],
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

    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) {
        mean += float(input[offset + j]);
    }
    mean /= float(cols);
    mean_out[id] = mean;

    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = float(input[offset + j]) - mean;
        var += diff * diff;
    }
    var /= float(cols);
    float rstd = rsqrt(var + eps);
    rstd_out[id] = rstd;

    for (uint j = 0; j < cols; j++) {
        float normed = (float(input[offset + j]) - mean) * rstd;
        output[offset + j] = half(normed * float(gamma[j]) + float(beta[j]));
    }
}
