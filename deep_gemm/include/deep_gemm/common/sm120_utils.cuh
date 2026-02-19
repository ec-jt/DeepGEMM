#pragma once

#include <cstdint>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm::sm120 {

// SM120 FP8 MMA: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// Per thread:
//   A: 4 x uint32_t = 16 bytes = 16 e4m3 values (covers M=16, K=32 across warp)
//   B: 2 x uint32_t = 8 bytes = 8 e4m3 values (covers N=8, K=32 across warp)
//   D: 4 x float (covers M=16, N=8 across warp)
//
// Thread layout in warp (32 threads):
//   A matrix (16 rows x 32 cols): each group of 4 threads covers 2 rows, 32 cols
//     Thread T covers rows [T/4*2, T/4*2+1] for K columns based on T%4
//   B matrix (8 cols x 32 rows): each group of 4 threads covers 2 cols, 32 rows
//   D matrix (16 rows x 8 cols): thread T owns:
//     d[0] = D[T/4*2][T%4*2], d[1] = D[T/4*2][T%4*2+1]
//     d[2] = D[T/4*2+1][T%4*2], d[3] = D[T/4*2+1][T%4*2+1]

__device__ __forceinline__ void fp8_mma_m16n8k32(
    float d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const float c[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// Load A matrix fragment from shared memory for mma.sync m16n8k32
// A is row-major: [M=16, K=32] in e4m3 format
// Each thread loads 16 bytes (4 x uint32_t)
// Thread mapping: thread T in warp loads:
//   a[0]: row (T%4)*2,     cols [(T/4)*4 .. (T/4)*4+3]  (first K=16 half)
//   a[1]: row (T%4)*2+1,   cols [(T/4)*4 .. (T/4)*4+3]  (first K=16 half)
//   a[2]: row (T%4)*2,     cols [(T/4)*4 .. (T/4)*4+3]  (second K=16 half)
//   a[3]: row (T%4)*2+1,   cols [(T/4)*4 .. (T/4)*4+3]  (second K=16 half)
//
// For swizzled shared memory with stride `stride` (in bytes):
// A[row][col] is at smem_base + row * stride + col
__device__ __forceinline__ void load_a_fragment(
    uint32_t a[4],
    const __nv_fp8_e4m3* smem_a,
    const uint32_t row_offset,  // starting row in the M dimension
    const uint32_t k_offset,    // starting column in the K dimension
    const uint32_t stride       // row stride in elements (not bytes)
) {
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t row_in_tile_0 = (lane_idx % 4) * 2;
    const uint32_t row_in_tile_1 = row_in_tile_0 + 1;
    const uint32_t col_in_k = (lane_idx / 4) * 4;

    // First K=16 half
    const uint32_t* ptr0 = reinterpret_cast<const uint32_t*>(
        smem_a + (row_offset + row_in_tile_0) * stride + k_offset + col_in_k);
    const uint32_t* ptr1 = reinterpret_cast<const uint32_t*>(
        smem_a + (row_offset + row_in_tile_1) * stride + k_offset + col_in_k);
    a[0] = *ptr0;
    a[1] = *ptr1;

    // Second K=16 half
    const uint32_t* ptr2 = reinterpret_cast<const uint32_t*>(
        smem_a + (row_offset + row_in_tile_0) * stride + k_offset + 16 + col_in_k);
    const uint32_t* ptr3 = reinterpret_cast<const uint32_t*>(
        smem_a + (row_offset + row_in_tile_1) * stride + k_offset + 16 + col_in_k);
    a[2] = *ptr2;
    a[3] = *ptr3;
}

// Load B matrix fragment from shared memory for mma.sync m16n8k32
// B is column-major: [K=32, N=8] in e4m3 format
// Each thread loads 8 bytes (2 x uint32_t)
__device__ __forceinline__ void load_b_fragment(
    uint32_t b[2],
    const __nv_fp8_e4m3* smem_b,
    const uint32_t n_offset,    // starting column in the N dimension
    const uint32_t k_offset,    // starting row in the K dimension
    const uint32_t stride       // row stride in elements
) {
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t col_in_tile = (lane_idx % 4) * 2;
    const uint32_t row_in_k = (lane_idx / 4) * 4;

    // First K=16 half
    const uint32_t* ptr0 = reinterpret_cast<const uint32_t*>(
        smem_b + (k_offset + row_in_k) * stride + n_offset + col_in_tile);
    b[0] = *ptr0;

    // Second K=16 half
    const uint32_t* ptr1 = reinterpret_cast<const uint32_t*>(
        smem_b + (k_offset + 16 + row_in_k) * stride + n_offset + col_in_tile);
    b[1] = *ptr1;
}

// SM120 FP8 MMA wrapper that tiles over larger M and N dimensions
// Computes C[M_tile, N_tile] += A[M_tile, K=32] * B[K=32, N_tile]
// using multiple m16n8k32 MMA instructions
template <uint32_t M_TILE, uint32_t N_TILE>
struct FP8MMA_SM120 {
    static constexpr uint32_t M = 16;  // MMA M
    static constexpr uint32_t N = 8;   // MMA N
    static constexpr uint32_t K = 32;  // MMA K
    
    static constexpr uint32_t M_TILES = M_TILE / M;
    static constexpr uint32_t N_TILES = N_TILE / N;
    static constexpr uint32_t kNumAccum = M_TILES * N_TILES * 4;  // 4 floats per MMA per thread
    
    // Execute tiled MMA: A[M_TILE, K=32] * B[K=32, N_TILE] -> accum[M_TILE, N_TILE]
    __device__ __forceinline__ static void mma(
        float* accum,
        const __nv_fp8_e4m3* smem_a,  // [M_TILE, K] row-major
        const __nv_fp8_e4m3* smem_b,  // [K, N_TILE] row-major (will be treated as col-major for B)
        const uint32_t stride_a,       // row stride for A in elements
        const uint32_t stride_b,       // row stride for B in elements
        const uint32_t k_offset,       // K offset for this iteration
        bool first_k                   // whether to zero-init accumulators
    ) {
        uint32_t a_frag[4];
        uint32_t b_frag[2];
        
        #pragma unroll
        for (uint32_t m = 0; m < M_TILES; ++m) {
            load_a_fragment(a_frag, smem_a, m * M, k_offset, stride_a);
            
            #pragma unroll
            for (uint32_t n = 0; n < N_TILES; ++n) {
                const uint32_t acc_idx = (m * N_TILES + n) * 4;
                load_b_fragment(b_frag, smem_b, n * N, k_offset, stride_b);
                
                float c[4];
                if (first_k) {
                    c[0] = c[1] = c[2] = c[3] = 0.0f;
                } else {
                    c[0] = accum[acc_idx + 0];
                    c[1] = accum[acc_idx + 1];
                    c[2] = accum[acc_idx + 2];
                    c[3] = accum[acc_idx + 3];
                }
                
                float d[4];
                fp8_mma_m16n8k32(d, a_frag, b_frag, c);
                
                accum[acc_idx + 0] = d[0];
                accum[acc_idx + 1] = d[1];
                accum[acc_idx + 2] = d[2];
                accum[acc_idx + 3] = d[3];
            }
        }
    }
};

} // namespace deep_gemm::sm120
