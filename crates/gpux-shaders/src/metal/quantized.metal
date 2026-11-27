#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Quantized matmul kernels (weights quantized, activations F32)
// ============================================================================

struct block_q8_0 {
    half scale;
    char qs[32];
};

struct block_q4_0 {
    half scale;
    uchar qs[16];
};

// Q4_K_M super-block: 256 elements, 144 bytes total. Format matches llama.cpp's
// `block_q4_K`. 8 sub-blocks of 32 elements each; each sub-block has its own
// 6-bit scale and 6-bit min, packed into 12 bytes:
//   bytes 0..3:  low 6 bits = scales[0..4]; top 2 bits = high 2 bits of scales[4..8]
//   bytes 4..7:  low 6 bits = mins[0..4];   top 2 bits = high 2 bits of mins[4..8]
//   bytes 8..11: low nibble = low 4 bits of scales[4..8]; high nibble = low 4 bits of mins[4..8]
// Dequantization: value[i] = d * sc * q - dmin * m  (q is unsigned 4-bit)
struct block_q4_k {
    half     d;          // super-block scale for sub-block scales
    half     dmin;       // super-block scale for sub-block mins
    uchar    scales[12]; // packed 6-bit scales + 6-bit mins for 8 sub-blocks
    uchar    qs[128];    // 256 4-bit quants packed
};

// Unpack the 6-bit scale (sc) and 6-bit min (m) for sub-block index `j` (0..8).
// Mirrors llama.cpp's get_scale_min_k4.
static inline void q4k_unpack_scale_min(uint j, device const uchar* q,
                                        thread uchar& sc, thread uchar& m) {
    if (j < 4) {
        sc = q[j]     & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
    }
}

// Same, but reading from a thread-local copy (used in the SIMD matvec where
// we keep the whole block in registers).
static inline void q4k_unpack_scale_min_thread(uint j, thread const uchar* q,
                                               thread uchar& sc, thread uchar& m) {
    if (j < 4) {
        sc = q[j]     & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
    }
}

// Same SIMD-matvec layout as fused.metal: 4 SG/TG, 4 rows/SG = 16 rows/TG.
constant uint Q_SIMD_W = 32;
constant uint Q_SG_PER_TG = 8;        // 8 SGs/TG → 256 threads/TG (sweet spot vs 4 or 16)
constant uint Q_ROWS_PER_SG = 4;
constant uint Q_ROWS_PER_TG = Q_SG_PER_TG * Q_ROWS_PER_SG;  // 32

// Tiled Q4_0 GEMM using simdgroup_matrix (Apple Silicon hardware matrix units).
//
// Layout:
//   1 SIMD group per threadgroup → produces one 8 × N_TILE output sub-tile
//   N_TILE = 32 (4 simdgroup_float8x8 outputs side-by-side)
//   K_TILE = 32 (one Q4_0 block per K iteration)
//
// Per K iteration:
//   1. The 32 lanes cooperatively load A[8 × 32] into threadgroup memory (each
//      lane reads 8 floats).
//   2. The 32 lanes cooperatively dequantize 4 Q4_0 blocks (one per N sub-tile)
//      into threadgroup memory as half precision.
//   3. simdgroup_load 1 A tile (8×8) and 4 B tiles (8×8 each), do 4
//      multiply_accumulate ops to update 4 output 8×8 accumulators.
//   (Loop K_TILE/8 = 4 times per K iteration — each Q4_0 block has 32 elements,
//    handled as 4 sub-tiles of 8 elements.)
//
// Total per TG: produces 8 × 32 outputs. For an M × N output:
//   grid = (ceil(N/32), ceil(M/8), 1)
kernel void matmul_a_bt_q4_0_sgm_f32(
    device const float*       A    [[buffer(0)]],
    device const block_q4_0*  B    [[buffer(1)]],
    device float*             C    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [M, N, K]
    threadgroup char*         shmem [[threadgroup(0)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgid  [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    constexpr uint M_TILE = 8;
    constexpr uint N_TILE = 32;   // 4 × 8 columns
    constexpr uint K_TILE = 32;   // 1 Q4_0 block

    uint m_base = tgid.y * M_TILE;
    uint n_base = tgid.x * N_TILE;
    if (m_base >= M || n_base >= N) return;

    // Threadgroup memory: A tile (8×32 floats = 1 KB) + B tile (32×32 halves = 2 KB)
    threadgroup float* sA = (threadgroup float*)(shmem);                // [M_TILE][K_TILE] row-major
    threadgroup half*  sB = (threadgroup half*)(shmem + M_TILE * K_TILE * sizeof(float)); // [N_TILE][K_TILE]

    // 4 output 8×8 accumulators (covering N_TILE = 32 columns)
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    uint blocks_per_row = K / 32;

    for (uint kb = 0; kb < blocks_per_row; kb++) {
        // ===== Phase 1: Load A[m_base..m_base+8, kb*32..(kb+1)*32] into sA =====
        // 32 lanes × 8 elements each = 256 elements, but we need 8×32 = 256. Match!
        // Lane `tiisg` loads sA[tiisg / 32][tiisg % 32] = A[m_base + row][kb*32 + col]
        // Wait, 32 lanes for 8 rows × 32 cols = 256 elems. Each lane handles 8 elems.
        // Layout: lane l handles sA[l/4][l%4 * 8 + i] for i in 0..8 — too scattered.
        // Simpler: each lane loads 1 column-pair across all 8 rows? Or...
        // Easiest: lanes 0..31 each load 8 elems of A: lane l → sA[l%8][l/8 * 8 + i] for i in 0..8
        // Cleaner: for col_quart in 0..4: each lane loads sA[lane%8][col_quart*8 + lane/8]
        // This is messy — let me use a flat striped approach:
        //   total = 8 rows × 32 cols = 256 elements; 32 lanes; each loads 8 (=256/32).
        for (uint i = 0; i < 8; i++) {
            uint flat = tiisg * 8 + i;            // 0..256
            uint row = flat / K_TILE;             // 0..8
            uint col = flat % K_TILE;             // 0..32
            uint a_row = m_base + row;
            uint a_col = kb * K_TILE + col;
            sA[row * K_TILE + col] = (a_row < M) ? A[a_row * K + a_col] : 0.0f;
        }

        // ===== Phase 2: Dequantize B blocks into sB =====
        // For each of N_TILE=32 output columns, we need K_TILE=32 weight values.
        // That's 32 Q4_0 blocks × 32 elements = 1024 values total. 32 lanes × 32 = match.
        // Each lane handles 1 column (= 1 Q4_0 block of 32 elements).
        // Lane `tiisg` is responsible for column `tiisg` of sB.
        {
            uint n_col = n_base + tiisg;
            if (n_col < N) {
                block_q4_0 blk = B[n_col * blocks_per_row + kb];
                float scale = float(blk.scale);
                for (uint j = 0; j < 16; j++) {
                    uchar q = blk.qs[j];
                    sB[(j * 2 + 0) * N_TILE + tiisg] = (half)(scale * float(int(q & 0xF) - 8));
                    sB[(j * 2 + 1) * N_TILE + tiisg] = (half)(scale * float(int(q >> 4) - 8));
                }
            } else {
                for (uint j = 0; j < 32; j++) {
                    sB[j * N_TILE + tiisg] = (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase 3: 4 multiply_accumulates per K sub-tile of 8 =====
        // For each k_sub in 0..K_TILE/8 = 4:
        //   load A tile (8×8) at sA[*, k_sub*8..k_sub*8+8]
        //   load 4 B tiles (8×8 each) at sB[k_sub*8..k_sub*8+8, n_sub*8..n_sub*8+8]
        //   c[n_sub] += A_tile @ B_tile
        for (uint k_sub = 0; k_sub < 4; k_sub++) {
            simdgroup_float8x8 a_tile;
            simdgroup_load(a_tile, sA + k_sub * 8, K_TILE);

            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, sB + (k_sub * 8) * N_TILE +  0, N_TILE);
            simdgroup_load(b1, sB + (k_sub * 8) * N_TILE +  8, N_TILE);
            simdgroup_load(b2, sB + (k_sub * 8) * N_TILE + 16, N_TILE);
            simdgroup_load(b3, sB + (k_sub * 8) * N_TILE + 24, N_TILE);

            simdgroup_multiply_accumulate(c0, a_tile, b0, c0);
            simdgroup_multiply_accumulate(c1, a_tile, b1, c1);
            simdgroup_multiply_accumulate(c2, a_tile, b2, c2);
            simdgroup_multiply_accumulate(c3, a_tile, b3, c3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== Store outputs =====
    // c0..c3 cover (m_base..m_base+8) × (n_base..n_base+32).
    // Bounds checking for partial tiles is handled by writing through a guard;
    // simplest path: if the tile fits fully, do simdgroup_store directly.
    if (m_base + 8 <= M && n_base + 32 <= N) {
        simdgroup_store(c0, C + m_base * N + n_base +  0, N);
        simdgroup_store(c1, C + m_base * N + n_base +  8, N);
        simdgroup_store(c2, C + m_base * N + n_base + 16, N);
        simdgroup_store(c3, C + m_base * N + n_base + 24, N);
    } else {
        // Edge tile: write to TG mem, then copy with bounds check.
        threadgroup float scratch[M_TILE * N_TILE];
        simdgroup_store(c0, scratch +  0, N_TILE);
        simdgroup_store(c1, scratch +  8, N_TILE);
        simdgroup_store(c2, scratch + 16, N_TILE);
        simdgroup_store(c3, scratch + 24, N_TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++) {
            uint flat = tiisg * 8 + i;
            uint row = flat / N_TILE;
            uint col = flat % N_TILE;
            uint c_row = m_base + row;
            uint c_col = n_base + col;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = scratch[row * N_TILE + col];
            }
        }
    }
}

// Tiled Q4_0 GEMM v2: 4 SIMD groups cooperate per TG, producing a 32×32 output tile.
//
// Compared to the v1 (single-SG, 8×32 output): each TG produces 4× more outputs,
// dequantized weights in TG memory are reused across all 4 SGs (so each weight is
// loaded once but feeds 32 M tokens of dot products instead of 8). For decode-shaped
// workloads (small M) this is overkill; for long prefill it dramatically improves
// arithmetic intensity per memory load.
//
// Layout per TG (128 threads = 4 SGs):
//   sgid in 0..4 → owns M rows [sgid*8 .. sgid*8 + 8) of the 32-row M tile
//   All 4 SGs share the same N tile (32 columns)
//   Each SG holds 4 output 8×8 accumulators → 256 outputs per SG → 1024 per TG ✓
//
// Per K iteration (K_TILE = 32 = 1 Q4_0 block):
//   Phase 1: 128 threads cooperatively load A[32×32] → sA (8 floats per thread)
//   Phase 2: 128 threads cooperatively dequantize 32 Q4_0 blocks → sB[32×32] f16
//   Phase 3: For each of 4 K sub-tiles of 8: each SG loads 1 ma + 4 mb tiles,
//            does 4 multiply_accumulates against its accumulators
constant uint MM2_M_TILE = 32;
constant uint MM2_N_TILE = 32;
constant uint MM2_K_TILE = 32;
constant uint MM2_THREADS = 128;

kernel void matmul_a_bt_q4_0_sgm2_f32(
    device const float*       A    [[buffer(0)]],
    device const block_q4_0*  B    [[buffer(1)]],
    device float*             C    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [M, N, K]
    threadgroup char*         shmem [[threadgroup(0)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgid  [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort tiitg [[thread_index_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint m_base = tgid.y * MM2_M_TILE;
    uint n_base = tgid.x * MM2_N_TILE;
    if (m_base >= M || n_base >= N) return;

    threadgroup float* sA = (threadgroup float*)(shmem);                                    // [32][32] f32
    threadgroup half*  sB = (threadgroup half*)(shmem + MM2_M_TILE * MM2_K_TILE * sizeof(float)); // [32][32] f16

    // Each SG owns 4 output 8×8 tiles spanning N (covering all 32 cols at this M offset).
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    uint blocks_per_row = K / 32;

    for (uint kb = 0; kb < blocks_per_row; kb++) {
        // ===== Phase 1: cooperative A load (128 threads × 8 elements = 1024 = 32×32) =====
        for (uint i = 0; i < 8; i++) {
            uint flat = tiitg * 8 + i;
            uint row = flat / MM2_K_TILE;     // 0..32
            uint col = flat % MM2_K_TILE;     // 0..32
            uint a_row = m_base + row;
            uint a_col = kb * MM2_K_TILE + col;
            sA[row * MM2_K_TILE + col] = (a_row < M) ? A[a_row * K + a_col] : 0.0f;
        }

        // ===== Phase 2: cooperative dequant of 32 Q4_0 blocks (one per N column) =====
        // 128 threads / 32 cols = 4 threads per column. Lane handles 1 column;
        // 4 SGs each independently fill the same column → wasted work. Better:
        // tiitg / 4 = column index (0..31), tiitg % 4 = which 8-element quarter.
        {
            uint n_col = tiitg / 4;             // 0..31
            uint qchunk = tiitg % 4;            // 0..3 (which 8-elem quarter of the block)
            uint global_n = n_base + n_col;
            if (global_n < N) {
                block_q4_0 blk = B[global_n * blocks_per_row + kb];
                float scale = float(blk.scale);
                // Each thread writes 8 elements (4 packed bytes) of the column.
                for (uint jj = 0; jj < 4; jj++) {
                    uint j = qchunk * 4 + jj;
                    uchar q = blk.qs[j];
                    sB[(j * 2 + 0) * MM2_N_TILE + n_col] = (half)(scale * float(int(q & 0xF) - 8));
                    sB[(j * 2 + 1) * MM2_N_TILE + n_col] = (half)(scale * float(int(q >> 4) - 8));
                }
            } else {
                for (uint jj = 0; jj < 4; jj++) {
                    uint j = qchunk * 4 + jj;
                    sB[(j * 2 + 0) * MM2_N_TILE + n_col] = (half)0.0f;
                    sB[(j * 2 + 1) * MM2_N_TILE + n_col] = (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase 3: 4 K sub-tiles × 4 N sub-tiles = 16 multiply_accumulates per SG =====
        // Each SG owns M rows [sgid*8 .. sgid*8 + 8).
        uint m_off_in_tile = sgid * 8;
        for (uint k_sub = 0; k_sub < 4; k_sub++) {
            simdgroup_float8x8 a_tile;
            simdgroup_load(a_tile, sA + m_off_in_tile * MM2_K_TILE + k_sub * 8, MM2_K_TILE);

            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, sB + (k_sub * 8) * MM2_N_TILE +  0, MM2_N_TILE);
            simdgroup_load(b1, sB + (k_sub * 8) * MM2_N_TILE +  8, MM2_N_TILE);
            simdgroup_load(b2, sB + (k_sub * 8) * MM2_N_TILE + 16, MM2_N_TILE);
            simdgroup_load(b3, sB + (k_sub * 8) * MM2_N_TILE + 24, MM2_N_TILE);

            simdgroup_multiply_accumulate(c0, a_tile, b0, c0);
            simdgroup_multiply_accumulate(c1, a_tile, b1, c1);
            simdgroup_multiply_accumulate(c2, a_tile, b2, c2);
            simdgroup_multiply_accumulate(c3, a_tile, b3, c3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== Store outputs =====
    uint m_out = m_base + sgid * 8;
    if (m_out + 8 <= M && n_base + 32 <= N) {
        simdgroup_store(c0, C + m_out * N + n_base +  0, N);
        simdgroup_store(c1, C + m_out * N + n_base +  8, N);
        simdgroup_store(c2, C + m_out * N + n_base + 16, N);
        simdgroup_store(c3, C + m_out * N + n_base + 24, N);
    } else {
        // Edge tile: write to per-SG scratch, then copy with bounds.
        threadgroup float scratch[4 * 8 * MM2_N_TILE];  // 4 SGs × 8 rows × 32 cols
        threadgroup float* sg_scratch = scratch + sgid * 8 * MM2_N_TILE;
        simdgroup_store(c0, sg_scratch +  0, MM2_N_TILE);
        simdgroup_store(c1, sg_scratch +  8, MM2_N_TILE);
        simdgroup_store(c2, sg_scratch + 16, MM2_N_TILE);
        simdgroup_store(c3, sg_scratch + 24, MM2_N_TILE);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++) {
            uint flat = tiisg * 8 + i;        // 0..256 per SG
            uint row = flat / MM2_N_TILE;     // 0..8
            uint col = flat % MM2_N_TILE;     // 0..32
            uint c_row = m_out + row;
            uint c_col = n_base + col;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = sg_scratch[row * MM2_N_TILE + col];
            }
        }
    }
}

// Tiled Q4_0 GEMM v4 — wider N tile (M_TILE=32, N_TILE=64) for better arithmetic
// intensity per TG. Each SG owns 8 M rows but covers ALL 64 N cols → 8 mc tiles.
//
// Per TG (4 SGs, 128 threads): produces 32 × 64 = 2048 outputs (2× v2's 1024).
//   sgid ∈ {0..3}: M direction; each SG owns rows [sgid*8 .. sgid*8 + 8)
//   All SGs cover all 64 N cols → each SG holds 8 mc tiles (1 ma × 8 mb)
//
// Per K iteration (K_TILE = 32):
//   Phase 1: 128 threads load A[32×32] cooperatively (8 floats/thread)
//   Phase 2: 128 threads dequant 64 Q4_0 blocks (one per N col) → sB[32×64]
//            (2 threads per col, each producing 16 elements)
//   Phase 3: 4 K sub-tiles × 8 MACs/sub-tile = 32 MACs per SG
//
// Trade: 17 simdgroup_matrix per SG (1 ma + 8 mb + 8 mc) — at the limit but
// empirically fits on M-series. ½ the TG count of v2 for the same problem.
constant uint MM4_M_TILE = 32;
constant uint MM4_N_TILE = 64;
constant uint MM4_K_TILE = 32;

kernel void matmul_a_bt_q4_0_sgm4_f32(
    device const float*       A    [[buffer(0)]],
    device const block_q4_0*  B    [[buffer(1)]],
    device float*             C    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [M, N, K]
    threadgroup char*         shmem [[threadgroup(0)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgid  [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort tiitg [[thread_index_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint m_base = tgid.y * MM4_M_TILE;
    uint n_base = tgid.x * MM4_N_TILE;
    if (m_base >= M || n_base >= N) return;

    threadgroup float* sA = (threadgroup float*)(shmem);                                    // [32][32] f32
    threadgroup half*  sB = (threadgroup half*)(shmem + MM4_M_TILE * MM4_K_TILE * sizeof(float)); // [32][64] f16

    uint m_off_in_tile = sgid * 8;

    // 8 mc tiles per SG covering N=64
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c4 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c5 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c6 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c7 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    uint blocks_per_row = K / 32;

    for (uint kb = 0; kb < blocks_per_row; kb++) {
        // ===== Phase 1: cooperative A load (128 threads × 8 elements = 1024 = 32×32) =====
        for (uint i = 0; i < 8; i++) {
            uint flat = tiitg * 8 + i;
            uint row = flat / MM4_K_TILE;
            uint col = flat % MM4_K_TILE;
            uint a_row = m_base + row;
            uint a_col = kb * MM4_K_TILE + col;
            sA[row * MM4_K_TILE + col] = (a_row < M) ? A[a_row * K + a_col] : 0.0f;
        }

        // ===== Phase 2: cooperative dequant of 64 Q4_0 blocks (one per N col) =====
        // 128 threads / 64 cols = 2 threads per col, each produces 16 elements.
        {
            uint n_col = tiitg / 2;          // 0..63
            uint half_id = tiitg & 1;        // 0 = lower 8 bytes (qs[0..8]), 1 = upper (qs[8..16])
            uint global_n = n_base + n_col;
            if (global_n < N) {
                block_q4_0 blk = B[global_n * blocks_per_row + kb];
                float scale = float(blk.scale);
                uint j_start = half_id * 8;
                for (uint jj = 0; jj < 8; jj++) {
                    uint j = j_start + jj;
                    uchar q = blk.qs[j];
                    sB[(j * 2 + 0) * MM4_N_TILE + n_col] = (half)(scale * float(int(q & 0xF) - 8));
                    sB[(j * 2 + 1) * MM4_N_TILE + n_col] = (half)(scale * float(int(q >> 4) - 8));
                }
            } else {
                uint j_start = half_id * 8;
                for (uint jj = 0; jj < 8; jj++) {
                    uint j = j_start + jj;
                    sB[(j * 2 + 0) * MM4_N_TILE + n_col] = (half)0.0f;
                    sB[(j * 2 + 1) * MM4_N_TILE + n_col] = (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase 3: 4 K sub-tiles × 8 MACs per SG =====
        for (uint k_sub = 0; k_sub < 4; k_sub++) {
            simdgroup_float8x8 a_tile;
            simdgroup_load(a_tile, sA + m_off_in_tile * MM4_K_TILE + k_sub * 8, MM4_K_TILE);

            simdgroup_half8x8 b0, b1, b2, b3, b4, b5, b6, b7;
            simdgroup_load(b0, sB + (k_sub * 8) * MM4_N_TILE +  0, MM4_N_TILE);
            simdgroup_load(b1, sB + (k_sub * 8) * MM4_N_TILE +  8, MM4_N_TILE);
            simdgroup_load(b2, sB + (k_sub * 8) * MM4_N_TILE + 16, MM4_N_TILE);
            simdgroup_load(b3, sB + (k_sub * 8) * MM4_N_TILE + 24, MM4_N_TILE);
            simdgroup_load(b4, sB + (k_sub * 8) * MM4_N_TILE + 32, MM4_N_TILE);
            simdgroup_load(b5, sB + (k_sub * 8) * MM4_N_TILE + 40, MM4_N_TILE);
            simdgroup_load(b6, sB + (k_sub * 8) * MM4_N_TILE + 48, MM4_N_TILE);
            simdgroup_load(b7, sB + (k_sub * 8) * MM4_N_TILE + 56, MM4_N_TILE);

            simdgroup_multiply_accumulate(c0, a_tile, b0, c0);
            simdgroup_multiply_accumulate(c1, a_tile, b1, c1);
            simdgroup_multiply_accumulate(c2, a_tile, b2, c2);
            simdgroup_multiply_accumulate(c3, a_tile, b3, c3);
            simdgroup_multiply_accumulate(c4, a_tile, b4, c4);
            simdgroup_multiply_accumulate(c5, a_tile, b5, c5);
            simdgroup_multiply_accumulate(c6, a_tile, b6, c6);
            simdgroup_multiply_accumulate(c7, a_tile, b7, c7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== Store outputs =====
    uint m_out = m_base + m_off_in_tile;
    if (m_out + 8 <= M && n_base + 64 <= N) {
        simdgroup_store(c0, C + m_out * N + n_base +  0, N);
        simdgroup_store(c1, C + m_out * N + n_base +  8, N);
        simdgroup_store(c2, C + m_out * N + n_base + 16, N);
        simdgroup_store(c3, C + m_out * N + n_base + 24, N);
        simdgroup_store(c4, C + m_out * N + n_base + 32, N);
        simdgroup_store(c5, C + m_out * N + n_base + 40, N);
        simdgroup_store(c6, C + m_out * N + n_base + 48, N);
        simdgroup_store(c7, C + m_out * N + n_base + 56, N);
    } else {
        // Edge tile: per-SG scratch, then bounds-checked copy.
        threadgroup float scratch[4 * 8 * MM4_N_TILE];
        threadgroup float* sg_scratch = scratch + sgid * 8 * MM4_N_TILE;
        simdgroup_store(c0, sg_scratch +  0, MM4_N_TILE);
        simdgroup_store(c1, sg_scratch +  8, MM4_N_TILE);
        simdgroup_store(c2, sg_scratch + 16, MM4_N_TILE);
        simdgroup_store(c3, sg_scratch + 24, MM4_N_TILE);
        simdgroup_store(c4, sg_scratch + 32, MM4_N_TILE);
        simdgroup_store(c5, sg_scratch + 40, MM4_N_TILE);
        simdgroup_store(c6, sg_scratch + 48, MM4_N_TILE);
        simdgroup_store(c7, sg_scratch + 56, MM4_N_TILE);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; i++) {
            uint flat = tiisg * 16 + i;
            uint row = flat / MM4_N_TILE;
            uint col = flat % MM4_N_TILE;
            uint c_row = m_out + row;
            uint c_col = n_base + col;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = sg_scratch[row * MM4_N_TILE + col];
            }
        }
    }
}

// Tiled Q4_0 GEMM v3 — matches llama.cpp's proven layout.
//
// Per TG (4 SGs in 2×2 arrangement, 128 threads):
//   sg_y = sgid / 2 ∈ {0,1}: owns M rows [sg_y*32 .. sg_y*32 + 32)
//   sg_x = sgid % 2 ∈ {0,1}: owns N cols [sg_x*16 .. sg_x*16 + 16)
//   Per SG: 4 ma × 2 mb = 8 mc tiles in registers (covers 32M × 16N output)
//   Per TG: 64M × 32N = 2048 outputs
//
// Per K iteration (K_TILE = 32 = 1 Q4_0 block):
//   Phase 1: 128 threads load A[64×32] cooperatively → sA (16 floats per thread)
//   Phase 2: 128 threads dequantize B[32 cols × 32 elems] → sB[32×32] f16
//            (4 threads per column, each producing 8 elements)
//   Phase 3: 4 K sub-tiles of 8; per sub-tile each SG does 4 ma loads + 2 mb loads
//            + 8 multiply_accumulates → 32 MACs per SG per K iter
//
// vs v2 (4 SGs all-M, 32×32 output): 2× more mc tiles per SG → 2× more compute
// per memory load cycle, ½ the TGs needed for the same total work.
constant uint MM3_M_TILE = 64;
constant uint MM3_N_TILE = 32;
constant uint MM3_K_TILE = 32;

kernel void matmul_a_bt_q4_0_sgm3_f32(
    device const float*       A    [[buffer(0)]],
    device const block_q4_0*  B    [[buffer(1)]],
    device float*             C    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [M, N, K]
    threadgroup char*         shmem [[threadgroup(0)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgid  [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort tiitg [[thread_index_in_threadgroup]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint m_base = tgid.y * MM3_M_TILE;
    uint n_base = tgid.x * MM3_N_TILE;
    if (m_base >= M || n_base >= N) return;

    threadgroup float* sA = (threadgroup float*)(shmem);                                    // [64][32] f32
    threadgroup half*  sB = (threadgroup half*)(shmem + MM3_M_TILE * MM3_K_TILE * sizeof(float)); // [32][32] f16

    // 2×2 SG grid: sg_y = M direction, sg_x = N direction
    ushort sg_y = sgid >> 1;
    ushort sg_x = sgid & 1;
    uint m_off = sg_y * 32u;     // each sg_y owns 32 M rows
    uint n_off = sg_x * 16u;     // each sg_x owns 16 N cols

    // 8 output 8×8 accumulators per SG (4 ma × 2 mb)
    simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c20 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c21 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c30 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 c31 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    uint blocks_per_row = K / 32;

    for (uint kb = 0; kb < blocks_per_row; kb++) {
        // ===== Phase 1: cooperative A load (128 threads × 16 elements = 2048 = 64×32) =====
        for (uint i = 0; i < 16; i++) {
            uint flat = tiitg * 16 + i;
            uint row = flat / MM3_K_TILE;     // 0..64
            uint col = flat % MM3_K_TILE;     // 0..32
            uint a_row = m_base + row;
            uint a_col = kb * MM3_K_TILE + col;
            sA[row * MM3_K_TILE + col] = (a_row < M) ? A[a_row * K + a_col] : 0.0f;
        }

        // ===== Phase 2: cooperative dequant of 32 Q4_0 blocks =====
        // Same scheme as v2: tiitg / 4 = column (0..31), tiitg % 4 = quarter (0..3)
        {
            uint n_col = tiitg / 4;
            uint qchunk = tiitg % 4;
            uint global_n = n_base + n_col;
            if (global_n < N) {
                block_q4_0 blk = B[global_n * blocks_per_row + kb];
                float scale = float(blk.scale);
                for (uint jj = 0; jj < 4; jj++) {
                    uint j = qchunk * 4 + jj;
                    uchar q = blk.qs[j];
                    sB[(j * 2 + 0) * MM3_N_TILE + n_col] = (half)(scale * float(int(q & 0xF) - 8));
                    sB[(j * 2 + 1) * MM3_N_TILE + n_col] = (half)(scale * float(int(q >> 4) - 8));
                }
            } else {
                for (uint jj = 0; jj < 4; jj++) {
                    uint j = qchunk * 4 + jj;
                    sB[(j * 2 + 0) * MM3_N_TILE + n_col] = (half)0.0f;
                    sB[(j * 2 + 1) * MM3_N_TILE + n_col] = (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase 3: 4 K sub-tiles × 8 MACs per SG =====
        for (uint k_sub = 0; k_sub < 4; k_sub++) {
            // Load 4 ma tiles (one per M sub-row of 8) — each SG handles its own M region
            simdgroup_float8x8 a0, a1, a2, a3;
            simdgroup_load(a0, sA + (m_off +  0) * MM3_K_TILE + k_sub * 8, MM3_K_TILE);
            simdgroup_load(a1, sA + (m_off +  8) * MM3_K_TILE + k_sub * 8, MM3_K_TILE);
            simdgroup_load(a2, sA + (m_off + 16) * MM3_K_TILE + k_sub * 8, MM3_K_TILE);
            simdgroup_load(a3, sA + (m_off + 24) * MM3_K_TILE + k_sub * 8, MM3_K_TILE);

            // Load 2 mb tiles (one per N sub-col of 8) — each SG handles its own N region
            simdgroup_half8x8 b0, b1;
            simdgroup_load(b0, sB + (k_sub * 8) * MM3_N_TILE + n_off +  0, MM3_N_TILE);
            simdgroup_load(b1, sB + (k_sub * 8) * MM3_N_TILE + n_off +  8, MM3_N_TILE);

            // 8 multiply_accumulates: mc[i][j] += a[i] @ b[j]
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
            simdgroup_multiply_accumulate(c20, a2, b0, c20);
            simdgroup_multiply_accumulate(c21, a2, b1, c21);
            simdgroup_multiply_accumulate(c30, a3, b0, c30);
            simdgroup_multiply_accumulate(c31, a3, b1, c31);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== Store outputs =====
    // Each SG writes its 8 mc tiles to C[m_base + m_off + i*8 .. + 8, n_base + n_off + j*8 .. + 8]
    uint base_m = m_base + m_off;
    uint base_n = n_base + n_off;
    if (base_m + 32 <= M && base_n + 16 <= N) {
        simdgroup_store(c00, C + (base_m +  0) * N + base_n + 0, N);
        simdgroup_store(c01, C + (base_m +  0) * N + base_n + 8, N);
        simdgroup_store(c10, C + (base_m +  8) * N + base_n + 0, N);
        simdgroup_store(c11, C + (base_m +  8) * N + base_n + 8, N);
        simdgroup_store(c20, C + (base_m + 16) * N + base_n + 0, N);
        simdgroup_store(c21, C + (base_m + 16) * N + base_n + 8, N);
        simdgroup_store(c30, C + (base_m + 24) * N + base_n + 0, N);
        simdgroup_store(c31, C + (base_m + 24) * N + base_n + 8, N);
    } else {
        // Edge tile: store via TG memory then bounds-checked copy.
        threadgroup float scratch[4 * 32 * 16];   // 4 SGs × 32 M rows × 16 N cols = 2048
        threadgroup float* sg_scratch = scratch + sgid * 32 * 16;
        simdgroup_store(c00, sg_scratch +  0 * 16 + 0, 16);
        simdgroup_store(c01, sg_scratch +  0 * 16 + 8, 16);
        simdgroup_store(c10, sg_scratch +  8 * 16 + 0, 16);
        simdgroup_store(c11, sg_scratch +  8 * 16 + 8, 16);
        simdgroup_store(c20, sg_scratch + 16 * 16 + 0, 16);
        simdgroup_store(c21, sg_scratch + 16 * 16 + 8, 16);
        simdgroup_store(c30, sg_scratch + 24 * 16 + 0, 16);
        simdgroup_store(c31, sg_scratch + 24 * 16 + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; i++) {
            uint flat = tiisg * 16 + i;       // 0..512 per SG
            uint row = flat / 16;             // 0..32
            uint col = flat % 16;             // 0..16
            uint c_row = base_m + row;
            uint c_col = base_n + col;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = sg_scratch[row * 16 + col];
            }
        }
    }
}

// M-batched fused gate+up+silu+mul with Q4_0 weights. Mirrors matmul_a_bt_q4_0_m_f32
// but reads BOTH gate and up weight blocks per K iter (sharing the input read across
// gate, up, AND M tokens). Output: silu(gate[m,n]) * up[m,n] for all (m, n).
//
// Speeds up speculative-decode verify (M=2..8) by replacing 3 unfused dispatches
// (gate matvec + up matvec + silu_mul) with one fused kernel.
kernel void fused_gate_up_silu_q4_0_m_f32(
    device const float*       A       [[buffer(0)]],   // [M_total, K]
    device const block_q4_0*  W_gate  [[buffer(1)]],   // [N, K/32]
    device const block_q4_0*  W_up    [[buffer(2)]],   // [N, K/32]
    device float*             out     [[buffer(3)]],   // [M_total, N]
    device const uint*        dims    [[buffer(4)]],   // [M_total, N, K, m_off]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint M_total = dims[0];
    uint N       = dims[1];
    uint K       = dims[2];
    uint m_off   = dims[3];
    uint blocks_per_row = K / 32;

    uint M = min(8u, M_total - m_off);

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    // Per-(row, M) accumulators for gate and up. 4 rows × M_MAX × 2 = 64 floats max.
    float g0[8] = {0}, g1[8] = {0};
    float g2[8] = {0}, g3[8] = {0};
    float u0[8] = {0}, u1[8] = {0};
    float u2[8] = {0}, u3[8] = {0};

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        // Load 4 gate weight blocks + 4 up weight blocks for this K block.
        block_q4_0 g_b0 = W_gate[r0 * blocks_per_row + b];
        block_q4_0 g_b1 = W_gate[r1 * blocks_per_row + b];
        block_q4_0 g_b2 = W_gate[r2 * blocks_per_row + b];
        block_q4_0 g_b3 = W_gate[r3 * blocks_per_row + b];

        block_q4_0 u_b0 = W_up[r0 * blocks_per_row + b];
        block_q4_0 u_b1 = W_up[r1 * blocks_per_row + b];
        block_q4_0 u_b2 = W_up[r2 * blocks_per_row + b];
        block_q4_0 u_b3 = W_up[r3 * blocks_per_row + b];

        float sg0 = float(g_b0.scale), sg1 = float(g_b1.scale);
        float sg2 = float(g_b2.scale), sg3 = float(g_b3.scale);
        float su0 = float(u_b0.scale), su1 = float(u_b1.scale);
        float su2 = float(u_b2.scale), su3 = float(u_b3.scale);

        // Inner product per M token; the 2 weight reads (gate, up) are shared
        // across all M tokens for this K block.
        for (uint m = 0; m < M; m++) {
            uint a_off = (m_off + m) * K + b * 32;
            float ag0 = 0, ag1 = 0, ag2 = 0, ag3 = 0;
            float au0 = 0, au1 = 0, au2 = 0, au3 = 0;

            for (uint j = 0; j < 16; j++) {
                float xv0 = A[a_off + j * 2];
                float xv1 = A[a_off + j * 2 + 1];

                uchar pg0 = g_b0.qs[j], pg1 = g_b1.qs[j];
                uchar pg2 = g_b2.qs[j], pg3 = g_b3.qs[j];
                uchar pu0 = u_b0.qs[j], pu1 = u_b1.qs[j];
                uchar pu2 = u_b2.qs[j], pu3 = u_b3.qs[j];

                ag0 += xv0 * float(int(pg0 & 0xF) - 8) + xv1 * float(int(pg0 >> 4) - 8);
                ag1 += xv0 * float(int(pg1 & 0xF) - 8) + xv1 * float(int(pg1 >> 4) - 8);
                ag2 += xv0 * float(int(pg2 & 0xF) - 8) + xv1 * float(int(pg2 >> 4) - 8);
                ag3 += xv0 * float(int(pg3 & 0xF) - 8) + xv1 * float(int(pg3 >> 4) - 8);

                au0 += xv0 * float(int(pu0 & 0xF) - 8) + xv1 * float(int(pu0 >> 4) - 8);
                au1 += xv0 * float(int(pu1 & 0xF) - 8) + xv1 * float(int(pu1 >> 4) - 8);
                au2 += xv0 * float(int(pu2 & 0xF) - 8) + xv1 * float(int(pu2 >> 4) - 8);
                au3 += xv0 * float(int(pu3 & 0xF) - 8) + xv1 * float(int(pu3 >> 4) - 8);
            }

            g0[m] += ag0 * sg0; g1[m] += ag1 * sg1;
            g2[m] += ag2 * sg2; g3[m] += ag3 * sg3;
            u0[m] += au0 * su0; u1[m] += au1 * su1;
            u2[m] += au2 * su2; u3[m] += au3 * su3;
        }
    }

    // SIMD-reduce and apply silu(gate) * up per (m, row).
    for (uint m = 0; m < M; m++) {
        float vg0 = simd_sum(g0[m]); float vg1 = simd_sum(g1[m]);
        float vg2 = simd_sum(g2[m]); float vg3 = simd_sum(g3[m]);
        float vu0 = simd_sum(u0[m]); float vu1 = simd_sum(u1[m]);
        float vu2 = simd_sum(u2[m]); float vu3 = simd_sum(u3[m]);

        if (lane == 0) {
            uint c_row = m_off + m;
            uint c_base = c_row * N;
            if (base     < N) { float s = 1.0f / (1.0f + exp(-vg0)); out[c_base + base    ] = (vg0 * s) * vu0; }
            if (base + 1 < N) { float s = 1.0f / (1.0f + exp(-vg1)); out[c_base + base + 1] = (vg1 * s) * vu1; }
            if (base + 2 < N) { float s = 1.0f / (1.0f + exp(-vg2)); out[c_base + base + 2] = (vg2 * s) * vu2; }
            if (base + 3 < N) { float s = 1.0f / (1.0f + exp(-vg3)); out[c_base + base + 3] = (vg3 * s) * vu3; }
        }
    }
}

// M-batched Q4_0 matmul (prefill path).
// C[m, n] = sum_k A[m, k] * dequant(W[n, k])  for m in [0, M), n in [0, N)
//
// Same SIMD layout as matvec_a_bt_q4_0 (4 SG/TG × 4 rows/SG = 16 rows/TG, plus
// the SG_PER_TG=8 variant from quantized_q4_0 → 32 rows/TG with 256 threads),
// but each lane accumulates M dot products per assigned row — one per input token.
// The weight reads are shared across all M tokens, giving M-fold weight bandwidth
// reuse vs running matvec_a_bt_q4_0 M times.
//
// M is bounded at compile-time (M_MAX) for register-allocation safety. For larger
// inputs we dispatch multiple TGs along the M axis (gid.y).
//
// Layout note: dims = [M_total, N, K, m_off]; the kernel processes tokens
// [m_off, m_off + min(M_MAX, M_total - m_off)).
// (M_MAX = 8 inlined)

kernel void matmul_a_bt_q4_0_m_f32(
    device const float*       A    [[buffer(0)]],
    device const block_q4_0*  W    [[buffer(1)]],
    device float*             C    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [M_total, N, K, m_off]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint M_total = dims[0];
    uint N       = dims[1];
    uint K       = dims[2];
    uint m_off   = dims[3];
    uint blocks_per_row = K / 32;

    // Number of M tokens this TG processes (clamped to M_MAX).
    uint M = min(8u, M_total - m_off);

    // Output rows for this SG.
    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    // Accumulators: 4 rows × M_MAX tokens.
    float acc0[8] = {0};
    float acc1[8] = {0};
    float acc2[8] = {0};
    float acc3[8] = {0};

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        // Load 4 weight blocks once, reuse across all M tokens.
        block_q4_0 bk0 = W[r0 * blocks_per_row + b];
        block_q4_0 bk1 = W[r1 * blocks_per_row + b];
        block_q4_0 bk2 = W[r2 * blocks_per_row + b];
        block_q4_0 bk3 = W[r3 * blocks_per_row + b];

        float sc0 = float(bk0.scale), sc1 = float(bk1.scale);
        float sc2 = float(bk2.scale), sc3 = float(bk3.scale);

        // Per-token inner product over the 32 elements of this block.
        for (uint m = 0; m < M; m++) {
            uint a_off = (m_off + m) * K + b * 32;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            float xsum = 0;

            for (uint j = 0; j < 16; j++) {
                float xv0 = A[a_off + j * 2];
                float xv1 = A[a_off + j * 2 + 1];
                xsum += xv0 + xv1;

                uchar p0 = bk0.qs[j];
                uchar p1 = bk1.qs[j];
                uchar p2 = bk2.qs[j];
                uchar p3 = bk3.qs[j];

                a0 += xv0 * float(int(p0 & 0xF) - 8) + xv1 * float(int(p0 >> 4) - 8);
                a1 += xv0 * float(int(p1 & 0xF) - 8) + xv1 * float(int(p1 >> 4) - 8);
                a2 += xv0 * float(int(p2 & 0xF) - 8) + xv1 * float(int(p2 >> 4) - 8);
                a3 += xv0 * float(int(p3 & 0xF) - 8) + xv1 * float(int(p3 >> 4) - 8);
            }

            acc0[m] += a0 * sc0;
            acc1[m] += a1 * sc1;
            acc2[m] += a2 * sc2;
            acc3[m] += a3 * sc3;
            (void)xsum;
        }
    }

    // SIMD-reduce and write each (m, row) result.
    for (uint m = 0; m < M; m++) {
        float v0 = simd_sum(acc0[m]);
        float v1 = simd_sum(acc1[m]);
        float v2 = simd_sum(acc2[m]);
        float v3 = simd_sum(acc3[m]);
        if (lane == 0) {
            uint c_row = m_off + m;
            uint c_base = c_row * N;
            if (base     < N) C[c_base + base    ] = v0;
            if (base + 1 < N) C[c_base + base + 1] = v1;
            if (base + 2 < N) C[c_base + base + 2] = v2;
            if (base + 3 < N) C[c_base + base + 3] = v3;
        }
    }
}

// SIMD matvec with Q4_0 weights: y = x @ W^T (M=1 decode path).
// Each SIMD group computes Q_ROWS_PER_SG rows; each lane strides over Q4_0 blocks.
// The Metal compiler vectorizes the scalar inner loop well; explicit float4/uchar4
// loads + the "divider trick" (pre-divide x by 16/256/4096) didn't measurably help
// — left as the simple form so the compiler can do its thing.
// SIMD matvec with Q4_K weights: y = x @ W^T (M=1 decode path).
// 4 SG/TG, 4 rows/SG, lane stride over Q4_K super-blocks (256 elem each).
//
// Memory: 144 bytes/256 elem = 4.5 bits/elem (vs Q4_0's 4.5 bits effective with
// per-32 scales). Better quality at the same bandwidth.
//
// Per super-block dequant + dot: extract 8 sub-block scales/mins, then for each
// sub-block dot with the matching 32 input floats. We unroll by sub-block to
// keep register pressure manageable.
kernel void matvec_a_bt_q4_k_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q4_k*  W    [[buffer(1)]],
    device float*             y    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 256;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    // Each lane handles `blocks_per_row / 32` super-blocks (typically 0–1 for
    // K=896, more for K=4864). Striding by SIMD width.
    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 256;

        // Process all 4 rows for this super-block index.
        // Extract per-row d, dmin, sub-scales/mins; then accumulate over 8 sub-blocks.
        for (uint r_idx = 0; r_idx < 4; r_idx++) {
            uint row = (r_idx == 0) ? r0 : (r_idx == 1) ? r1 : (r_idx == 2) ? r2 : r3;
            device const block_q4_k& blk = W[row * blocks_per_row + b];

            float d    = float(blk.d);
            float dmin = float(blk.dmin);

            // Copy scales bytes to thread storage to allow easy re-indexing.
            uchar scales_buf[12];
            for (uint i = 0; i < 12; i++) scales_buf[i] = blk.scales[i];

            float row_acc = 0.0f;

            // 8 sub-blocks of 32 elements each. Pairs share 32 bytes of qs:
            //   pair 0 (sub 0,1) → qs[ 0..32]: low nibbles = sub 0, high = sub 1
            //   pair 1 (sub 2,3) → qs[32..64]: ...
            //   ...
            for (uint pair = 0; pair < 4; pair++) {
                uchar sc_lo, m_lo, sc_hi, m_hi;
                q4k_unpack_scale_min_thread(pair * 2 + 0, scales_buf, sc_lo, m_lo);
                q4k_unpack_scale_min_thread(pair * 2 + 1, scales_buf, sc_hi, m_hi);
                float d_lo = d * float(sc_lo);
                float d_hi = d * float(sc_hi);
                float m_lo_f = dmin * float(m_lo);
                float m_hi_f = dmin * float(m_hi);

                uint qs_off = pair * 32;
                uint x_lo_off = x_off + pair * 64;
                uint x_hi_off = x_lo_off + 32;

                float dot_lo = 0.0f;
                float dot_hi = 0.0f;
                float xsum_lo = 0.0f;
                float xsum_hi = 0.0f;

                for (uint l = 0; l < 32; l++) {
                    uchar qb = blk.qs[qs_off + l];
                    float xl = x[x_lo_off + l];
                    float xh = x[x_hi_off + l];
                    dot_lo += xl * float(qb & 0xF);
                    dot_hi += xh * float(qb >> 4);
                    xsum_lo += xl;
                    xsum_hi += xh;
                }

                // Deferred subtraction: value = d_sc * q - m_f, so total contribution
                // = d_sc * Σ(x*q) - m_f * Σx
                row_acc += d_lo * dot_lo - m_lo_f * xsum_lo;
                row_acc += d_hi * dot_hi - m_hi_f * xsum_hi;
            }

            if (r_idx == 0) s0 += row_acc;
            else if (r_idx == 1) s1 += row_acc;
            else if (r_idx == 2) s2 += row_acc;
            else                 s3 += row_acc;
        }
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

// Q4_K matvec v2 — lane-per-sub-block layout.
//
// v1's problem: each lane handled one whole 256-element super-block, so for
// K=2048 only 8 of 32 SIMD lanes had work and the kernel ran at ¼ utilization.
//
// v2: each lane handles one 32-element sub-block. With 8 sub-blocks per super-
// block, 32 lanes process 4 super-blocks per inner iteration. All lanes are
// always busy regardless of K size. For K=2048 (8 super-blocks per row) we
// just need 2 inner iterations; for K=4096+ even fewer iterations per lane.
//
// Lanes that fall on the same super-block redundantly read its `d`/`dmin`/
// scales bytes — that's 8× redundancy on those bytes. In practice L1 absorbs
// the duplicate reads; the unique-bytes-per-super-block (~144 B) sets the
// real bandwidth floor.
kernel void matvec_a_bt_q4_k_v2_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q4_k*  W    [[buffer(1)]],
    device float*             y    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint super_blocks_per_row = K / 256;
    uint sub_blocks_per_row   = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint sub_idx = lane; sub_idx < sub_blocks_per_row; sub_idx += 32) {
        uint super_idx    = sub_idx >> 3;        // / 8
        uint sub_in_super = sub_idx & 7;         // % 8
        uint pair         = sub_in_super >> 1;   // / 2
        bool is_low       = (sub_in_super & 1) == 0;
        uint qs_off       = pair * 32;
        uint x_off        = sub_idx * 32;

        for (uint r_idx = 0; r_idx < 4; r_idx++) {
            uint row = (r_idx == 0) ? r0 : (r_idx == 1) ? r1 : (r_idx == 2) ? r2 : r3;
            device const block_q4_k& blk = W[row * super_blocks_per_row + super_idx];

            float d    = float(blk.d);
            float dmin = float(blk.dmin);

            uchar scales_buf[12];
            for (uint i = 0; i < 12; i++) scales_buf[i] = blk.scales[i];
            uchar sc, m;
            q4k_unpack_scale_min_thread(sub_in_super, scales_buf, sc, m);
            float d_sc = d    * float(sc);
            float m_f  = dmin * float(m);

            float dot = 0.0f, xsum = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uchar qb = blk.qs[qs_off + l];
                float xv = x[x_off + l];
                float q  = is_low ? float(qb & 0xF) : float(qb >> 4);
                dot  += xv * q;
                xsum += xv;
            }

            float acc = d_sc * dot - m_f * xsum;
            if      (r_idx == 0) s0 += acc;
            else if (r_idx == 1) s1 += acc;
            else if (r_idx == 2) s2 += acc;
            else                 s3 += acc;
        }
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

// Q4_K v2 + bias. Same lane-per-sub-block layout as v2; bias added at write.
kernel void matvec_a_bt_bias_q4_k_v2_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q4_k*  W    [[buffer(1)]],
    device const float*       bias [[buffer(2)]],
    device float*             y    [[buffer(3)]],
    device const uint*        dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint super_blocks_per_row = K / 256;
    uint sub_blocks_per_row   = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    for (uint sub_idx = lane; sub_idx < sub_blocks_per_row; sub_idx += 32) {
        uint super_idx    = sub_idx >> 3;
        uint sub_in_super = sub_idx & 7;
        uint pair         = sub_in_super >> 1;
        bool is_low       = (sub_in_super & 1) == 0;
        uint qs_off       = pair * 32;
        uint x_off        = sub_idx * 32;

        for (uint r_idx = 0; r_idx < 4; r_idx++) {
            uint row = (r_idx == 0) ? r0 : (r_idx == 1) ? r1 : (r_idx == 2) ? r2 : r3;
            device const block_q4_k& blk = W[row * super_blocks_per_row + super_idx];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            uchar scales_buf[12];
            for (uint i = 0; i < 12; i++) scales_buf[i] = blk.scales[i];
            uchar sc, m;
            q4k_unpack_scale_min_thread(sub_in_super, scales_buf, sc, m);
            float d_sc = d    * float(sc);
            float m_f  = dmin * float(m);
            float dot = 0, xsum = 0;
            for (uint l = 0; l < 32; l++) {
                uchar qb = blk.qs[qs_off + l];
                float xv = x[x_off + l];
                float q  = is_low ? float(qb & 0xF) : float(qb >> 4);
                dot  += xv * q;
                xsum += xv;
            }
            float acc = d_sc * dot - m_f * xsum;
            if      (r_idx == 0) s0 += acc;
            else if (r_idx == 1) s1 += acc;
            else if (r_idx == 2) s2 += acc;
            else                 s3 += acc;
        }
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

// Same but with bias added (single dispatch for QKV with bias / O proj with bias).
kernel void matvec_a_bt_bias_q4_k_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q4_k*  W    [[buffer(1)]],
    device const float*       bias [[buffer(2)]],
    device float*             y    [[buffer(3)]],
    device const uint*        dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 256;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 256;
        for (uint r_idx = 0; r_idx < 4; r_idx++) {
            uint row = (r_idx == 0) ? r0 : (r_idx == 1) ? r1 : (r_idx == 2) ? r2 : r3;
            device const block_q4_k& blk = W[row * blocks_per_row + b];
            float d    = float(blk.d);
            float dmin = float(blk.dmin);
            uchar scales_buf[12];
            for (uint i = 0; i < 12; i++) scales_buf[i] = blk.scales[i];
            float row_acc = 0.0f;
            for (uint pair = 0; pair < 4; pair++) {
                uchar sc_lo, m_lo, sc_hi, m_hi;
                q4k_unpack_scale_min_thread(pair * 2 + 0, scales_buf, sc_lo, m_lo);
                q4k_unpack_scale_min_thread(pair * 2 + 1, scales_buf, sc_hi, m_hi);
                float d_lo = d * float(sc_lo);
                float d_hi = d * float(sc_hi);
                float m_lo_f = dmin * float(m_lo);
                float m_hi_f = dmin * float(m_hi);
                uint qs_off = pair * 32;
                uint x_lo_off = x_off + pair * 64;
                uint x_hi_off = x_lo_off + 32;
                float dot_lo = 0, dot_hi = 0, xsum_lo = 0, xsum_hi = 0;
                for (uint l = 0; l < 32; l++) {
                    uchar qb = blk.qs[qs_off + l];
                    float xl = x[x_lo_off + l];
                    float xh = x[x_hi_off + l];
                    dot_lo  += xl * float(qb & 0xF);
                    dot_hi  += xh * float(qb >> 4);
                    xsum_lo += xl;
                    xsum_hi += xh;
                }
                row_acc += d_lo * dot_lo - m_lo_f * xsum_lo;
                row_acc += d_hi * dot_hi - m_hi_f * xsum_hi;
            }
            if (r_idx == 0) s0 += row_acc;
            else if (r_idx == 1) s1 += row_acc;
            else if (r_idx == 2) s2 += row_acc;
            else                 s3 += row_acc;
        }
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

// Quantize F32 → Q4_K_M, one thread per super-block (256 elements).
//
// Strategy (simpler than llama.cpp's iterative search; produces decent quality):
//   For each super-block:
//     1. Per sub-block (8 of them): find v_min, v_max
//     2. d = max(v_max[i] - v_min[i]) / (15 * 63)  (so largest sub-block scale = 63)
//     3. dmin = max(-v_min[i] for v_min[i] < 0) / 63 (or 0 if all v_min ≥ 0)
//     4. For each sub-block:
//          sc[i] = clamp(round((v_max[i] - v_min[i]) / 15 / d), 0, 63)
//          m[i]  = clamp(round(-v_min[i] / dmin), 0, 63)
//        Then quantize each value: q = clamp(round((v - (-dmin*m)) / (d*sc)), 0, 15)
//     5. Pack scales/mins into 12 bytes, pack quants into 128 bytes.
kernel void quantize_q4_k_f32(
    device const float* input  [[buffer(0)]],
    device block_q4_k*  output [[buffer(1)]],
    device const uint*  dims   [[buffer(2)]],  // [total_super_blocks]
    uint id [[thread_position_in_grid]]
) {
    uint total = dims[0];
    if (id >= total) return;
    uint base = id * 256;

    // Per-sub-block min/max.
    float v_min[8], v_max[8];
    for (uint sb = 0; sb < 8; sb++) {
        float lo = INFINITY, hi = -INFINITY;
        for (uint l = 0; l < 32; l++) {
            float v = input[base + sb * 32 + l];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        v_min[sb] = lo;
        v_max[sb] = hi;
    }

    // Pick super-block scales d (for sub-block scales) and dmin (for mins).
    float max_range  = 0.0f;
    float max_minabs = 0.0f;
    for (uint sb = 0; sb < 8; sb++) {
        float r = v_max[sb] - v_min[sb];
        if (r > max_range) max_range = r;
        // We use the most-negative value (v_min < 0 ⇒ -v_min > 0) for dmin.
        float ma = -v_min[sb];
        if (ma > max_minabs) max_minabs = ma;
    }
    float d    = max_range  / (15.0f * 63.0f);
    float dmin = max_minabs / 63.0f;
    if (d == 0.0f)    d = 1.0f / 4096.0f;
    if (dmin == 0.0f) dmin = 1.0f / 4096.0f;

    output[id].d    = (half)d;
    output[id].dmin = (half)dmin;

    // Per-sub-block sc, m. Pack into 12 bytes per llama.cpp's get_scale_min_k4 layout.
    uchar sc_arr[8], m_arr[8];
    for (uint sb = 0; sb < 8; sb++) {
        float sub_range = v_max[sb] - v_min[sb];
        int sc = (int)round(sub_range / 15.0f / d);
        int m  = (int)round(-v_min[sb] / dmin);
        sc_arr[sb] = (uchar)clamp(sc, 0, 63);
        m_arr[sb]  = (uchar)clamp(m,  0, 63);
    }

    // Pack: bytes 0..3 = scales[0..4] (low 6 bits), bytes 4..7 = mins[0..4],
    //       bytes 8..11 low nibble = scales[4..8] low 4 bits, high nibble = mins[4..8] low 4 bits,
    //       bytes 0..3 top 2 bits  = scales[4..8] top 2 bits,
    //       bytes 4..7 top 2 bits  = mins[4..8]   top 2 bits.
    for (uint i = 0; i < 4; i++) {
        uchar sc_low_lo = sc_arr[i];
        uchar m_low_lo  = m_arr[i];
        uchar sc_hi     = sc_arr[i + 4];
        uchar m_hi      = m_arr[i + 4];
        output[id].scales[i]     = (sc_low_lo & 0x3F) | ((sc_hi & 0x30) << 2);
        output[id].scales[i + 4] = (m_low_lo  & 0x3F) | ((m_hi  & 0x30) << 2);
        output[id].scales[i + 8] = (sc_hi & 0x0F) | ((m_hi & 0x0F) << 4);
    }

    // Quantize each value. Layout matches dequant: pair `p` ∈ {0..3}, lo half (sub p*2)
    // → qs[p*32+l] low nibble, hi half (sub p*2+1) → qs[p*32+l] high nibble.
    for (uint pair = 0; pair < 4; pair++) {
        uint sub_lo = pair * 2;
        uint sub_hi = pair * 2 + 1;
        float d_lo = d * float(sc_arr[sub_lo]);
        float d_hi = d * float(sc_arr[sub_hi]);
        float m_lo = dmin * float(m_arr[sub_lo]);
        float m_hi = dmin * float(m_arr[sub_hi]);
        if (d_lo == 0.0f) d_lo = 1e-6f;
        if (d_hi == 0.0f) d_hi = 1e-6f;
        for (uint l = 0; l < 32; l++) {
            float vl = input[base + sub_lo * 32 + l];
            float vh = input[base + sub_hi * 32 + l];
            int q_lo = (int)round((vl + m_lo) / d_lo);
            int q_hi = (int)round((vh + m_hi) / d_hi);
            q_lo = clamp(q_lo, 0, 15);
            q_hi = clamp(q_hi, 0, 15);
            output[id].qs[pair * 32 + l] = (uchar)((q_hi << 4) | q_lo);
        }
    }
}

kernel void matvec_a_bt_q4_0_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q4_0*  W    [[buffer(1)]],
    device float*             y    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 32;

        block_q4_0 bk0 = W[r0 * blocks_per_row + b];
        block_q4_0 bk1 = W[r1 * blocks_per_row + b];
        block_q4_0 bk2 = W[r2 * blocks_per_row + b];
        block_q4_0 bk3 = W[r3 * blocks_per_row + b];

        float sc0 = float(bk0.scale), sc1 = float(bk1.scale);
        float sc2 = float(bk2.scale), sc3 = float(bk3.scale);

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (uint j = 0; j < 16; j++) {
            float xv0 = x[x_off + j * 2];
            float xv1 = x[x_off + j * 2 + 1];

            uchar p0 = bk0.qs[j];
            uchar p1 = bk1.qs[j];
            uchar p2 = bk2.qs[j];
            uchar p3 = bk3.qs[j];

            acc0 += xv0 * float(int(p0 & 0xF) - 8) + xv1 * float(int(p0 >> 4) - 8);
            acc1 += xv0 * float(int(p1 & 0xF) - 8) + xv1 * float(int(p1 >> 4) - 8);
            acc2 += xv0 * float(int(p2 & 0xF) - 8) + xv1 * float(int(p2 >> 4) - 8);
            acc3 += xv0 * float(int(p3 & 0xF) - 8) + xv1 * float(int(p3 >> 4) - 8);
        }

        s0 += acc0 * sc0;
        s1 += acc1 * sc1;
        s2 += acc2 * sc2;
        s3 += acc3 * sc3;
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

// SIMD matvec with Q4_0 weights + bias: y = x @ W^T + bias (M=1 decode path).
kernel void matvec_a_bt_bias_q4_0_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q4_0*  W    [[buffer(1)]],
    device const float*       bias [[buffer(2)]],
    device float*             y    [[buffer(3)]],
    device const uint*        dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 32;

        block_q4_0 bk0 = W[r0 * blocks_per_row + b];
        block_q4_0 bk1 = W[r1 * blocks_per_row + b];
        block_q4_0 bk2 = W[r2 * blocks_per_row + b];
        block_q4_0 bk3 = W[r3 * blocks_per_row + b];

        float sc0 = float(bk0.scale), sc1 = float(bk1.scale);
        float sc2 = float(bk2.scale), sc3 = float(bk3.scale);

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (uint j = 0; j < 16; j++) {
            float xv0 = x[x_off + j * 2];
            float xv1 = x[x_off + j * 2 + 1];

            uchar p0 = bk0.qs[j];
            uchar p1 = bk1.qs[j];
            uchar p2 = bk2.qs[j];
            uchar p3 = bk3.qs[j];

            acc0 += xv0 * float(int(p0 & 0xF) - 8) + xv1 * float(int(p0 >> 4) - 8);
            acc1 += xv0 * float(int(p1 & 0xF) - 8) + xv1 * float(int(p1 >> 4) - 8);
            acc2 += xv0 * float(int(p2 & 0xF) - 8) + xv1 * float(int(p2 >> 4) - 8);
            acc3 += xv0 * float(int(p3 & 0xF) - 8) + xv1 * float(int(p3 >> 4) - 8);
        }

        s0 += acc0 * sc0;
        s1 += acc1 * sc1;
        s2 += acc2 * sc2;
        s3 += acc3 * sc3;
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

// SIMD matvec with Q8_0 weights: y = x @ W^T.
// 34 bytes per 32 elements (vs 64 for f16). ~1.9× bandwidth reduction.
kernel void matvec_a_bt_q8_0_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q8_0*  W    [[buffer(1)]],
    device float*             y    [[buffer(2)]],
    device const uint*        dims [[buffer(3)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 32;

        block_q8_0 bk0 = W[r0 * blocks_per_row + b];
        block_q8_0 bk1 = W[r1 * blocks_per_row + b];
        block_q8_0 bk2 = W[r2 * blocks_per_row + b];
        block_q8_0 bk3 = W[r3 * blocks_per_row + b];

        float sc0 = float(bk0.scale), sc1 = float(bk1.scale);
        float sc2 = float(bk2.scale), sc3 = float(bk3.scale);

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (uint j = 0; j < 32; j++) {
            float xv = x[x_off + j];
            acc0 += xv * float(bk0.qs[j]);
            acc1 += xv * float(bk1.qs[j]);
            acc2 += xv * float(bk2.qs[j]);
            acc3 += xv * float(bk3.qs[j]);
        }

        s0 += acc0 * sc0;
        s1 += acc1 * sc1;
        s2 += acc2 * sc2;
        s3 += acc3 * sc3;
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

// SIMD matvec with Q8_0 weights + bias: y = x @ W^T + bias.
kernel void matvec_a_bt_bias_q8_0_f32(
    device const float*       x    [[buffer(0)]],
    device const block_q8_0*  W    [[buffer(1)]],
    device const float*       bias [[buffer(2)]],
    device float*             y    [[buffer(3)]],
    device const uint*        dims [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 32;

        block_q8_0 bk0 = W[r0 * blocks_per_row + b];
        block_q8_0 bk1 = W[r1 * blocks_per_row + b];
        block_q8_0 bk2 = W[r2 * blocks_per_row + b];
        block_q8_0 bk3 = W[r3 * blocks_per_row + b];

        float sc0 = float(bk0.scale), sc1 = float(bk1.scale);
        float sc2 = float(bk2.scale), sc3 = float(bk3.scale);

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (uint j = 0; j < 32; j++) {
            float xv = x[x_off + j];
            acc0 += xv * float(bk0.qs[j]);
            acc1 += xv * float(bk1.qs[j]);
            acc2 += xv * float(bk2.qs[j]);
            acc3 += xv * float(bk3.qs[j]);
        }

        s0 += acc0 * sc0;
        s1 += acc1 * sc1;
        s2 += acc2 * sc2;
        s3 += acc3 * sc3;
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

// C = A @ B^T where A is [M, K] F32, B is [N, K] Q8_0
kernel void matmul_q8_0_f32(
    device const float*      A   [[buffer(0)]],
    device const block_q8_0* B_q [[buffer(1)]],
    device float*            C   [[buffer(2)]],
    device const uint*       dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims[0], N = dims[1], K = dims[2];
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocks_per_row = K / 32;
    float sum = 0.0f;

    for (uint b = 0; b < blocks_per_row; b++) {
        block_q8_0 block = B_q[col * blocks_per_row + b];
        float scale = float(block.scale);
        uint a_offset = row * K + b * 32;

        for (uint j = 0; j < 32; j++) {
            sum += A[a_offset + j] * (scale * float(block.qs[j]));
        }
    }

    C[row * N + col] = sum;
}

// C = A @ B^T where A is [M, K] F32, B is [N, K] Q4_0
kernel void matmul_q4_0_f32(
    device const float*      A   [[buffer(0)]],
    device const block_q4_0* B_q [[buffer(1)]],
    device float*            C   [[buffer(2)]],
    device const uint*       dims [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims[0], N = dims[1], K = dims[2];
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocks_per_row = K / 32;
    float sum = 0.0f;

    for (uint b = 0; b < blocks_per_row; b++) {
        block_q4_0 block = B_q[col * blocks_per_row + b];
        float scale = float(block.scale);
        uint a_offset = row * K + b * 32;

        for (uint j = 0; j < 16; j++) {
            uchar packed = block.qs[j];
            float v0 = float(int(packed & 0xF) - 8);
            float v1 = float(int(packed >> 4) - 8);
            sum += A[a_offset + j * 2]     * (scale * v0);
            sum += A[a_offset + j * 2 + 1] * (scale * v1);
        }
    }

    C[row * N + col] = sum;
}

// Quantize F32 -> Q8_0 on GPU (one thread per block of 32)
kernel void quantize_q8_0_f32(
    device const float*  input  [[buffer(0)]],
    device block_q8_0*   output [[buffer(1)]],
    device const uint*   dims   [[buffer(2)]],  // [total_blocks]
    device const uint*   K_val  [[buffer(3)]],  // [K] (inner dimension)
    uint id [[thread_position_in_grid]]
) {
    uint total_blocks = dims[0];
    uint K = K_val[0];
    if (id >= total_blocks) return;

    uint blocks_per_row = K / 32;
    uint row = id / blocks_per_row;
    uint b   = id % blocks_per_row;
    uint offset = row * K + b * 32;

    float amax = 0.0f;
    for (uint j = 0; j < 32; j++) {
        amax = max(amax, abs(input[offset + j]));
    }

    float scale = amax / 127.0f;
    float inv_scale = (scale != 0.0f) ? (127.0f / amax) : 0.0f;

    output[id].scale = half(scale);
    for (uint j = 0; j < 32; j++) {
        int q = int(round(input[offset + j] * inv_scale));
        q = clamp(q, -128, 127);
        output[id].qs[j] = char(q);
    }
}

// Quantize F32 -> Q4_0 on GPU (one thread per block of 32)
kernel void quantize_q4_0_f32(
    device const float*  input  [[buffer(0)]],
    device block_q4_0*   output [[buffer(1)]],
    device const uint*   dims   [[buffer(2)]],  // [total_blocks]
    device const uint*   K_val  [[buffer(3)]],  // [K]
    uint id [[thread_position_in_grid]]
) {
    uint total_blocks = dims[0];
    uint K = K_val[0];
    if (id >= total_blocks) return;

    uint blocks_per_row = K / 32;
    uint row = id / blocks_per_row;
    uint b   = id % blocks_per_row;
    uint offset = row * K + b * 32;

    float amax = 0.0f;
    for (uint j = 0; j < 32; j++) {
        amax = max(amax, abs(input[offset + j]));
    }

    float scale = amax / 7.0f;
    float inv_scale = (scale != 0.0f) ? (7.0f / amax) : 0.0f;

    output[id].scale = half(scale);
    for (uint j = 0; j < 16; j++) {
        float v0 = input[offset + j * 2];
        float v1 = input[offset + j * 2 + 1];
        int q0 = clamp(int(round(v0 * inv_scale)) + 8, 0, 15);
        int q1 = clamp(int(round(v1 * inv_scale)) + 8, 0, 15);
        output[id].qs[j] = uchar(q0 | (q1 << 4));
    }
}

// Fused gate+up+silu+mul: out[n] = silu(gate_w[n] @ x) * (up_w[n] @ x), Q4_0 weights.
// 4 SIMD groups per TG, 4 rows per SG = 16 rows per TG (matches Q4_0 matvec layout).
// Higher compute density than 1-row-per-SG since Q4_0 is compute-bound (32 dequants per 18-byte block).
kernel void fused_gate_up_silu_q4_0_f32(
    device const float*       x       [[buffer(0)]],
    device const block_q4_0*  W_gate  [[buffer(1)]],
    device const block_q4_0*  W_up    [[buffer(2)]],
    device float*             out     [[buffer(3)]],
    device const uint*        dims    [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 32;

        block_q4_0 g_b0 = W_gate[r0 * blocks_per_row + b];
        block_q4_0 g_b1 = W_gate[r1 * blocks_per_row + b];
        block_q4_0 g_b2 = W_gate[r2 * blocks_per_row + b];
        block_q4_0 g_b3 = W_gate[r3 * blocks_per_row + b];

        block_q4_0 u_b0 = W_up[r0 * blocks_per_row + b];
        block_q4_0 u_b1 = W_up[r1 * blocks_per_row + b];
        block_q4_0 u_b2 = W_up[r2 * blocks_per_row + b];
        block_q4_0 u_b3 = W_up[r3 * blocks_per_row + b];

        float sg0 = float(g_b0.scale), sg1 = float(g_b1.scale);
        float sg2 = float(g_b2.scale), sg3 = float(g_b3.scale);
        float su0 = float(u_b0.scale), su1 = float(u_b1.scale);
        float su2 = float(u_b2.scale), su3 = float(u_b3.scale);

        float ag0 = 0, ag1 = 0, ag2 = 0, ag3 = 0;
        float au0 = 0, au1 = 0, au2 = 0, au3 = 0;

        for (uint j = 0; j < 16; j++) {
            float xv0 = x[x_off + j * 2];
            float xv1 = x[x_off + j * 2 + 1];

            uchar pg0 = g_b0.qs[j], pg1 = g_b1.qs[j], pg2 = g_b2.qs[j], pg3 = g_b3.qs[j];
            uchar pu0 = u_b0.qs[j], pu1 = u_b1.qs[j], pu2 = u_b2.qs[j], pu3 = u_b3.qs[j];

            ag0 += xv0 * float(int(pg0 & 0xF) - 8) + xv1 * float(int(pg0 >> 4) - 8);
            ag1 += xv0 * float(int(pg1 & 0xF) - 8) + xv1 * float(int(pg1 >> 4) - 8);
            ag2 += xv0 * float(int(pg2 & 0xF) - 8) + xv1 * float(int(pg2 >> 4) - 8);
            ag3 += xv0 * float(int(pg3 & 0xF) - 8) + xv1 * float(int(pg3 >> 4) - 8);

            au0 += xv0 * float(int(pu0 & 0xF) - 8) + xv1 * float(int(pu0 >> 4) - 8);
            au1 += xv0 * float(int(pu1 & 0xF) - 8) + xv1 * float(int(pu1 >> 4) - 8);
            au2 += xv0 * float(int(pu2 & 0xF) - 8) + xv1 * float(int(pu2 >> 4) - 8);
            au3 += xv0 * float(int(pu3 & 0xF) - 8) + xv1 * float(int(pu3 >> 4) - 8);
        }

        g0 += ag0 * sg0; g1 += ag1 * sg1; g2 += ag2 * sg2; g3 += ag3 * sg3;
        u0 += au0 * su0; u1 += au1 * su1; u2 += au2 * su2; u3 += au3 * su3;
    }

    g0 = simd_sum(g0); g1 = simd_sum(g1); g2 = simd_sum(g2); g3 = simd_sum(g3);
    u0 = simd_sum(u0); u1 = simd_sum(u1); u2 = simd_sum(u2); u3 = simd_sum(u3);

    if (lane == 0) {
        if (base     < N) { float s = 1.0f / (1.0f + exp(-g0)); out[base]     = (g0 * s) * u0; }
        if (base + 1 < N) { float s = 1.0f / (1.0f + exp(-g1)); out[base + 1] = (g1 * s) * u1; }
        if (base + 2 < N) { float s = 1.0f / (1.0f + exp(-g2)); out[base + 2] = (g2 * s) * u2; }
        if (base + 3 < N) { float s = 1.0f / (1.0f + exp(-g3)); out[base + 3] = (g3 * s) * u3; }
    }
}

// Fused gate+up+silu+mul with Q8_0 weights. 4 SG/TG, 4 rows/SG = 16 rows/TG.
kernel void fused_gate_up_silu_q8_0_f32(
    device const float*       x       [[buffer(0)]],
    device const block_q8_0*  W_gate  [[buffer(1)]],
    device const block_q8_0*  W_up    [[buffer(2)]],
    device float*             out     [[buffer(3)]],
    device const uint*        dims    [[buffer(4)]],  // [N, K]
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint N = dims[0];
    uint K = dims[1];
    uint blocks_per_row = K / 32;

    uint base = tgid * Q_ROWS_PER_TG + sgid * Q_ROWS_PER_SG;
    uint r0 = min(base,     N - 1);
    uint r1 = min(base + 1, N - 1);
    uint r2 = min(base + 2, N - 1);
    uint r3 = min(base + 3, N - 1);

    float g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    float u0 = 0, u1 = 0, u2 = 0, u3 = 0;

    for (uint b = lane; b < blocks_per_row; b += Q_SIMD_W) {
        uint x_off = b * 32;

        block_q8_0 g_b0 = W_gate[r0 * blocks_per_row + b];
        block_q8_0 g_b1 = W_gate[r1 * blocks_per_row + b];
        block_q8_0 g_b2 = W_gate[r2 * blocks_per_row + b];
        block_q8_0 g_b3 = W_gate[r3 * blocks_per_row + b];

        block_q8_0 u_b0 = W_up[r0 * blocks_per_row + b];
        block_q8_0 u_b1 = W_up[r1 * blocks_per_row + b];
        block_q8_0 u_b2 = W_up[r2 * blocks_per_row + b];
        block_q8_0 u_b3 = W_up[r3 * blocks_per_row + b];

        float sg0 = float(g_b0.scale), sg1 = float(g_b1.scale);
        float sg2 = float(g_b2.scale), sg3 = float(g_b3.scale);
        float su0 = float(u_b0.scale), su1 = float(u_b1.scale);
        float su2 = float(u_b2.scale), su3 = float(u_b3.scale);

        float ag0 = 0, ag1 = 0, ag2 = 0, ag3 = 0;
        float au0 = 0, au1 = 0, au2 = 0, au3 = 0;

        for (uint j = 0; j < 32; j++) {
            float xv = x[x_off + j];
            ag0 += xv * float(g_b0.qs[j]);
            ag1 += xv * float(g_b1.qs[j]);
            ag2 += xv * float(g_b2.qs[j]);
            ag3 += xv * float(g_b3.qs[j]);
            au0 += xv * float(u_b0.qs[j]);
            au1 += xv * float(u_b1.qs[j]);
            au2 += xv * float(u_b2.qs[j]);
            au3 += xv * float(u_b3.qs[j]);
        }

        g0 += ag0 * sg0; g1 += ag1 * sg1; g2 += ag2 * sg2; g3 += ag3 * sg3;
        u0 += au0 * su0; u1 += au1 * su1; u2 += au2 * su2; u3 += au3 * su3;
    }

    g0 = simd_sum(g0); g1 = simd_sum(g1); g2 = simd_sum(g2); g3 = simd_sum(g3);
    u0 = simd_sum(u0); u1 = simd_sum(u1); u2 = simd_sum(u2); u3 = simd_sum(u3);

    if (lane == 0) {
        if (base     < N) { float s = 1.0f / (1.0f + exp(-g0)); out[base]     = (g0 * s) * u0; }
        if (base + 1 < N) { float s = 1.0f / (1.0f + exp(-g1)); out[base + 1] = (g1 * s) * u1; }
        if (base + 2 < N) { float s = 1.0f / (1.0f + exp(-g2)); out[base + 2] = (g2 * s) * u2; }
        if (base + 3 < N) { float s = 1.0f / (1.0f + exp(-g3)); out[base + 3] = (g3 * s) * u3; }
    }
}

// Embedding lookup with Q4_0 weights: output[token, d] = dequant(weight[indices[token], d])
kernel void embedding_forward_q4_0_f32(
    device const uint*        indices [[buffer(0)]],
    device const block_q4_0*  weight  [[buffer(1)]],
    device float*             output  [[buffer(2)]],
    device const uint*        dims    [[buffer(3)]],  // [total_tokens, dim]
    uint2 gid [[thread_position_in_grid]]
) {
    uint total_tokens = dims[0];
    uint dim = dims[1];
    uint token = gid.y;
    uint d = gid.x;
    if (token >= total_tokens || d >= dim) return;

    uint idx = indices[token];
    uint blocks_per_row = dim / 32;
    uint b = d / 32;
    uint j = d % 32;

    block_q4_0 blk = weight[idx * blocks_per_row + b];
    float scale = float(blk.scale);
    uchar packed = blk.qs[j / 2];
    int val = (j & 1) == 0 ? (int(packed & 0xF) - 8) : (int(packed >> 4) - 8);
    output[token * dim + d] = scale * float(val);
}

// Embedding lookup with Q8_0 weights
kernel void embedding_forward_q8_0_f32(
    device const uint*        indices [[buffer(0)]],
    device const block_q8_0*  weight  [[buffer(1)]],
    device float*             output  [[buffer(2)]],
    device const uint*        dims    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint total_tokens = dims[0];
    uint dim = dims[1];
    uint token = gid.y;
    uint d = gid.x;
    if (token >= total_tokens || d >= dim) return;

    uint idx = indices[token];
    uint blocks_per_row = dim / 32;
    uint b = d / 32;
    uint j = d % 32;

    block_q8_0 blk = weight[idx * blocks_per_row + b];
    output[token * dim + d] = float(blk.scale) * float(blk.qs[j]);
}

// Dequantize Q8_0 -> F32
kernel void dequantize_q8_0_f32(
    device const block_q8_0* input  [[buffer(0)]],
    device float*            output [[buffer(1)]],
    device const uint*       dims   [[buffer(2)]],  // [total_blocks]
    uint2 gid [[thread_position_in_grid]]
) {
    uint block_id = gid.y;
    uint j = gid.x;
    uint total_blocks = dims[0];

    if (block_id >= total_blocks || j >= 32) return;

    float scale = float(input[block_id].scale);
    output[block_id * 32 + j] = scale * float(input[block_id].qs[j]);
}

// Dequantize Q4_0 -> F32
kernel void dequantize_q4_0_f32(
    device const block_q4_0* input  [[buffer(0)]],
    device float*            output [[buffer(1)]],
    device const uint*       dims   [[buffer(2)]],  // [total_blocks]
    uint2 gid [[thread_position_in_grid]]
) {
    uint block_id = gid.y;
    uint j = gid.x;
    uint total_blocks = dims[0];

    if (block_id >= total_blocks || j >= 32) return;

    float scale = float(input[block_id].scale);
    uint byte_idx = j / 2;
    uchar packed = input[block_id].qs[byte_idx];
    int val;
    if (j % 2 == 0) {
        val = int(packed & 0xF) - 8;
    } else {
        val = int(packed >> 4) - 8;
    }
    output[block_id * 32 + j] = scale * float(val);
}
