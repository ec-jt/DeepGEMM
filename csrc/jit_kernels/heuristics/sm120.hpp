#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.hpp>

#include "common.hpp"

namespace deep_gemm {

struct SM120ArchSpec {
    // SM120 (RTX 5090) has 100 KB shared memory per SM
    static constexpr int smem_capacity = 102400;

    static std::vector<int> get_block_m_candidates(const KernelType& kernel_type, const cute::UMMA::Major& major_a, const int& m) {
        // SM120 uses mma.sync 16x8x32 tiles — smaller than SM90's wgmma 64x tiles
        // Use block sizes that are multiples of 16 (MMA M dimension)
        std::vector<int> candidates{16, 32, 64, 128};
        return candidates;
    }

    static std::vector<int> get_block_n_candidates(const KernelType& kernel_type, const at::ScalarType& cd_dtype) {
        // SM120 mma.sync has N=8, so block_n should be multiples of 8
        std::vector<int> candidates;
        for (int i = 8; i <= 128; i += 8)
            candidates.push_back(i);
        return candidates;
    }

    static int get_ab_load_block_m(const MulticastConfig& multicast_config, const int& block_m) {
        return block_m;
    }

    static int get_ab_load_block_n(const MulticastConfig& multicast_config, const int& block_n) {
        return block_n;
    }

    static int get_cd_store_block_m(const int& block_m, const bool& single_warpgroup_sync = false) {
        // SM120 uses per-warp mma.sync, not warp-group level
        return block_m;
    }

    static int get_cd_store_block_n(const int& block_n) {
        return block_n;
    }

    static bool enable_cd_swizzle(const at::ScalarType& cd_dtype) {
        return cd_dtype != torch::kFloat;
    }

    static bool is_block_size_legal(const KernelType& kernel_type,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                    const int& m, const int& n, const int& k,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // SM120 has limited shared memory (100 KB), so constrain block sizes
        // Ensure block_m * block_k + block_n * block_k fits in shared memory
        // with some room for barriers and other metadata
        const int ab_size = (block_m * block_k + block_n * block_k) * 1; // FP8 = 1 byte
        if (ab_size > smem_capacity / 2)
            return false;

        // Block sizes must be multiples of MMA tile dimensions
        if (block_m % 16 != 0 || block_n % 8 != 0)
            return false;

        return true;
    }

    static bool is_num_stages_legal(const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                    const int& num_stages,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // SM120 has less shared memory, so fewer stages
        return num_stages <= 3;
    }

    static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type, const int& num_groups,
                                                        const int& m, const int& n, const int& block_m, const int& block_n,
                                                        const int& num_sms) {
        // SM120 does not support multicast (consumer GPU)
        return {false, false};
    }

    static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                          const int& block_m, const int& block_n) {
        // SM120 uses per-warp mma.sync, not warp-group level
        // Use 128 threads for specialized + 128 for math (4 warps)
        return ThreadConfig::sm90(128, 128);
    }

    static int get_smem_cd_size(const KernelType& kernel_type,
                                const int& block_m, const int& block_n,
                                const int& swizzle_cd_mode, const at::ScalarType& cd_dtype) {
        return align(block_m * block_n * static_cast<int>(c10::elementSize(cd_dtype)), 1024);
    }

    static std::pair<int, int> get_sf_smem_size_per_stage(const KernelType& kernel_type,
                                                          const int& block_m, const int& block_n, const int& block_k,
                                                          const MmaKind& mma_kind, const at::ScalarType& cd_dtype) {
        if (mma_kind == MmaKind::BF16)
            return {0, 0};

        int smem_sfa_per_stage = align(block_m * static_cast<int>(sizeof(float)), 128);
        int smem_sfb_per_stage = 0;
        if (kernel_type == KernelType::Kernel1D1D)
            smem_sfb_per_stage = align(block_n * 4, 128);
        return {smem_sfa_per_stage, smem_sfb_per_stage};
    }

    static int get_extra_sfb_smem_size(const int& m, const int& n, const int& k,
                                       const int& block_m, const int& block_n, const int& block_k) {
        const auto& use_uniform_sfb = block_k % block_n == 0 ? 1 : 2;
        return align<int>(ceil_div(k, block_k) * static_cast<int>(sizeof(float)) * use_uniform_sfb, 8);
    }

    static int get_barrier_smem_size(const int& num_stages) {
        return num_stages * 8 * 2;
    }

    static int get_tmem_ptr_smem_size() {
        return 0;
    }

    static int get_tensormap_smem_size(const GemmType& gemm_type) {
        return gemm_type == GemmType::KGroupedContiguous ? 4 * static_cast<int>(sizeof(CUtensorMap)) : 0;
    }
};

} // namespace deep_gemm
