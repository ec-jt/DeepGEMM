#pragma once

#include <cutlass/arch/barrier.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm89.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>

// SM120 paged MQA logits kernel
// Uses SM89 MMA atom (mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32)
// which is supported on SM120 (same instruction as SM89, per-warp level).
// Reuses SM90 infrastructure: TMA, barriers, scheduler, shared memory layout.

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm120_fp8_paged_mqa_logits(const uint32_t batch_size,
                                const uint64_t logits_stride, const uint64_t block_table_stride,
                                const uint32_t* context_lens, float* logits,
                                const uint32_t* block_table, const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // SM120 uses SM89 MMA atom: m16n8k32 e4m3*e4m3->f32, per-warp
    // The SM90 WGMMA type is used only for kNumAccum calculation
    using WGMMA = typename FP8MMASelector<kNextN * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups, "Invalid `SPLIT_KV`");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory — identical to SM90
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers  = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    const auto& kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);

    const auto& smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });

    // Initialize barriers — identical to SM90
    if (warp_idx >= kNumMathThreads / 32 and cute::elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++ i) {
                full_q_barriers[i]->init(1);
                empty_q_barriers[i]->init(kNumMathThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(128);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    auto scheduler = PagedMQALogitsScheduler<kNextN, kIsContextLens2D, BLOCK_KV, kNumMathWarpGroups>(batch_size, blockIdx.x, context_lens, schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    const auto& get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1};
    };
    const auto& get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1};
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // ===== TMA loading warps — identical to SM90 =====
        // No register reconfig needed on SM120
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                tma_copy<kHeadDim, kNextN * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma_copy<kNextN * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv ?
                    __ldg(block_table + q_idx * block_table_stride + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)) : 0);
            }
            const auto& kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            if (cute::elect_one_sync()) {
                tma_copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                     smem_kv[kv_stage_idx], 0, 0, 1, kv_block_idx);
                tma_copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                         smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else {
        // ===== Math warps — SM120 per-warp MMA using SM89 atom =====
        // No register reconfig needed on SM120

        // SM89 MMA atom: m16n8k32, per-warp
        // Each warp handles 16 rows of BLOCK_KV=64 (4 warps × 16 = 64)
        using MMA_Op = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
        auto tiled_mma = cute::make_tiled_mma(cute::MMA_Atom<MMA_Op>{});

        // N dimension for the MMA
        static constexpr uint32_t N_DIM = kNextN * kNumHeads;
        static constexpr uint32_t N_TILES = N_DIM / 8;
        DG_STATIC_ASSERT(N_DIM % 8 == 0, "N_DIM must be multiple of 8");

        // Accumulators and weights — same layout as SM90
        float accum[WGMMA::kNumAccum], weights[kNextN][kNumHeads / 4];
        const auto& sub_warp_offset = (warp_idx % 4) * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        // Lane index within the warp for CuTe MMA
        const uint32_t lane_in_warp = lane_idx;

        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            if (q_idx != next_q_idx) {
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights — same as SM90
                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            auto kv_offset = q_idx * kNextN * logits_stride + ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // ===== SM120 MMA computation using CuTe SM89 atom =====
            DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % 32 == 0, "Invalid head dim");

            // Zero accumulators
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                accum[i] = 0.0f;

            // For each N tile (groups of 8 heads)
            #pragma unroll
            for (uint32_t n = 0; n < N_TILES; ++ n) {
                // Create CuTe tensors for this warp's KV slice and Q slice
                // KV: [16, kHeadDim] starting at row sub_warp_offset
                auto sKV = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<cute::float_e4m3_t*>(smem_kv[kv_stage_idx] + sub_warp_offset * kHeadDim)),
                    cute::make_layout(cute::make_shape(cute::Int<16>{}, cute::_32{}),
                                      cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));

                // Q: [8, kHeadDim] for this N tile (8 heads)
                auto sQ = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<cute::float_e4m3_t*>(smem_q[q_stage_idx] + n * 8 * kHeadDim)),
                    cute::make_layout(cute::make_shape(cute::_8{}, cute::_32{}),
                                      cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));

                auto thr_mma = tiled_mma.get_thread_slice(lane_in_warp);

                // For each K chunk (kHeadDim / 32 iterations)
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / 32; ++ k) {
                    // Create K-sliced views
                    auto sKV_k = cute::make_tensor(
                        cute::make_smem_ptr(reinterpret_cast<cute::float_e4m3_t*>(smem_kv[kv_stage_idx] + sub_warp_offset * kHeadDim + k * 32)),
                        cute::make_layout(cute::make_shape(cute::Int<16>{}, cute::Int<32>{}),
                                          cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));

                    auto sQ_k = cute::make_tensor(
                        cute::make_smem_ptr(reinterpret_cast<cute::float_e4m3_t*>(smem_q[q_stage_idx] + n * 8 * kHeadDim + k * 32)),
                        cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<32>{}),
                                          cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));

                    // Partition for this thread
                    auto tAsA = thr_mma.partition_A(sKV_k);
                    auto tBsB = thr_mma.partition_B(sQ_k);

                    // Create register fragments
                    auto tArA = thr_mma.partition_fragment_A(sKV_k);
                    auto tBrB = thr_mma.partition_fragment_B(sQ_k);

                    // Copy from shared to registers
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(tArA); ++i)
                        tArA(i) = tAsA(i);
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(tBrB); ++i)
                        tBrB(i) = tBsB(i);

                    // Execute MMA: accumulate into local accum array
                    // The MMA produces 4 floats per thread: d[0..3]
                    // d[0] = C[lane/4][lane%4*2], d[1] = C[lane/4][lane%4*2+1]
                    // d[2] = C[lane/4+8][lane%4*2], d[3] = C[lane/4+8][lane%4*2+1]
                    float d[4];
                    float c[4] = {
                        accum[n * 4 + 0],
                        accum[n * 4 + 1],
                        accum[n * 4 + 2],
                        accum[n * 4 + 3]
                    };

                    // Use raw MMA instruction with the CuTe-loaded fragments
                    uint32_t* ar = reinterpret_cast<uint32_t*>(&tArA);
                    uint32_t* br = reinterpret_cast<uint32_t*>(&tBrB);

                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                        : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
                          "r"(br[0]), "r"(br[1]),
                          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
                    );

                    accum[n * 4 + 0] = d[0];
                    accum[n * 4 + 1] = d[1];
                    accum[n * 4 + 2] = d[2];
                    accum[n * 4 + 3] = d[3];
                }
            }

            // Read per-KV scales
            float scale_kv_0 = ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset);
            float scale_kv_1 = ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset);

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store — same as SM90
            // The MMA output layout matches SM90's WGMMA output:
            // accum[n*4+0] = C[v_0_offset][n*8 + lane%4*2]
            // accum[n*4+1] = C[v_0_offset][n*8 + lane%4*2+1]
            // accum[n*4+2] = C[v_1_offset][n*8 + lane%4*2]
            // accum[n*4+3] = C[v_1_offset][n*8 + lane%4*2+1]
            //
            // The SM90 reduction code uses accum indexed as:
            // accum[i * kNumAccumPerReduce + j] where i=0..kNextN-1, j=0..kNumHeads/2-1
            // kNumAccumPerReduce = kNumHeads / 2
            // accum[j] maps to head (j/2)*8 + j%2 + (lane%4)*2 for the first kNextN block
            //
            // For SM120 with N_TILES tiles of 8:
            // accum[n*4 + r] where n=0..N_TILES-1, r=0..3
            // This maps to: head_col = n*8 + (lane%4)*2 + (r%2), row = v_0_offset (r<2) or v_1_offset (r>=2)
            //
            // The SM90 WGMMA accumulator has kNumAccum = N_DIM/2 values.
            // Our SM120 accumulator has N_TILES*4 = N_DIM/2 values. Same count!
            //
            // SM90 accum layout: accum[j] where j = n_tile * 4 + local_idx
            // This matches our layout: accum[n*4 + r]
            //
            // The SM90 reduction code:
            //   shifted_accum = accum + i * kNumAccumPerReduce  (i = kNextN index)
            //   transform(j) = fmaxf(shifted_accum[j], 0) * weights[i][(j/4)*2 + j%2]
            //   sum[k] += transform(j*4+k)
            //
            // For this to work, we need accum to be laid out as:
            //   accum[i * (kNumHeads/2) + j] where j indexes within heads
            //
            // Our layout: accum[n*4 + r] where n = i*(kNumHeads/8) + head_tile, r = 0..3
            // Flattened: accum[i*(kNumHeads/8)*4 + head_tile*4 + r] = accum[i*(kNumHeads/2) + head_tile*4 + r]
            // This matches! kNumAccumPerReduce = kNumHeads/2, j = head_tile*4 + r

            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
            DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == kNextN, "Invalid accumulation");
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                auto shifted_accum = accum + i * kNumAccumPerReduce;
                const auto& transform = [&](const uint32_t& j) {
                    return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                };

                // Intra-thread reduction
                float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                #pragma unroll
                for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k ++)
                        sum[k] += transform(j * 4 + k);
                }
                float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                // Inter-thread reduction
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const auto& offset = static_cast<int>(1u << j);
                    v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                    v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                }

                // Store into the global memory
                logits[kv_offset + i * logits_stride + v_0_offset] = v_0;
                logits[kv_offset + i * logits_stride + v_1_offset] = v_1;
            }
        }
    }
}

} // namespace deep_gemm
