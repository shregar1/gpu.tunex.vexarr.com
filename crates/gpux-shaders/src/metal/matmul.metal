#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Matrix multiplication kernels
// Tiled implementation for Apple Silicon GPU cache hierarchy
// All threads participate in barriers regardless of bounds to avoid UB.
// ============================================================================

constant uint TILE_SIZE = 32;

// Standard matmul: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + lid.x;
        uint b_row = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[lid.y][lid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matmul with transpose: C = A^T @ B
// A: [K, M] (transposed to [M, K]), B: [K, N], C: [M, N]
kernel void matmul_at_b_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_k = t * TILE_SIZE + lid.x;
        uint b_k = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && a_k < K) ? A[a_k * M + row] : 0.0f;
        Bs[lid.y][lid.x] = (b_k < K && col < N) ? B[b_k * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matmul: C = A @ B^T
// A: [M, K], B: [N, K] (transposed to [K, N]), C: [M, N]
kernel void matmul_a_bt_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + lid.x;
        uint b_col = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[lid.y][lid.x] = (b_col < K && col < N) ? B[col * K + b_col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batched matmul: C[b] = A[b] @ B[b]
// A: [B, M, K], B: [B, K, N], C: [B, M, N]
kernel void batched_matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [batch, M, N, K]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    uint batch = dims[0];
    uint M = dims[1];
    uint N = dims[2];
    uint K = dims[3];

    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch) return;

    uint a_offset = b * M * K;
    uint b_offset = b * K * N;
    uint c_offset = b * M * N;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + lid.x;
        uint b_row = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? A[a_offset + row * K + a_col] : 0.0f;
        Bs[lid.y][lid.x] = (b_row < K && col < N) ? B[b_offset + b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[c_offset + row * N + col] = sum;
    }
}

// Batched matmul: C[b] = A[b] @ B[b]^T
// A: [B, M, K], B: [B, N, K], C: [B, M, N]
kernel void batched_matmul_a_bt_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [batch, M, N, K]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    uint batch = dims[0];
    uint M = dims[1];
    uint N = dims[2];
    uint K = dims[3];

    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch) return;

    uint a_offset = b * M * K;
    uint b_offset = b * N * K;
    uint c_offset = b * M * N;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + lid.x;
        uint b_col = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && a_col < K) ? A[a_offset + row * K + a_col] : 0.0f;
        Bs[lid.y][lid.x] = (b_col < K && col < N) ? B[b_offset + col * K + b_col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[c_offset + row * N + col] = sum;
    }
}

// Batched matmul: C[b] = A[b]^T @ B[b]
// A: [B, K, M], B: [B, K, N], C: [B, M, N]
kernel void batched_matmul_at_b_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* dims [[buffer(3)]],  // [batch, M, N, K]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    uint batch = dims[0];
    uint M = dims[1];
    uint N = dims[2];
    uint K = dims[3];

    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch) return;

    uint a_offset = b * K * M;
    uint b_offset = b * K * N;
    uint c_offset = b * M * N;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_k = t * TILE_SIZE + lid.x;
        uint b_k = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && a_k < K) ? A[a_offset + a_k * M + row] : 0.0f;
        Bs[lid.y][lid.x] = (b_k < K && col < N) ? B[b_offset + b_k * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[c_offset + row * N + col] = sum;
    }
}

// Outer product accumulate: C += a * b^T
kernel void outer_product_acc_f32(
    device const float* a [[buffer(0)]],    // [M]
    device const float* b [[buffer(1)]],    // [N]
    device float* C [[buffer(2)]],          // [M, N]
    device const uint* dims [[buffer(3)]],  // [M, N]
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims[0];
    uint N = dims[1];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    C[row * N + col] += a[row] * b[col];
}
