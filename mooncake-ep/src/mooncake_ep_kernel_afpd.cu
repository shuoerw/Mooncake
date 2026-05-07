// AFPD kernels — single-node NVLink-IPC POC.
//
// Asymmetric attn<->expert routing. Caller pre-shifts every recv buffer by
// dst_arena_offset_bytes so PREFILL/DECODE arenas get disjoint memory on the
// expert side. M1 POC: IBGDA paths removed; M2 will reintroduce them by
// importing the static __device__ helpers from mooncake_ep_kernel.cu.

// clang-format off
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda/atomic>

#include <mooncake_ep_buffer_afpd.h>
#include <mooncake_ep_configs.cuh>
#include <mooncake_ep_exception.cuh>
#include <mooncake_ep_launch.cuh>
#include <mooncake_ibgda/mlx5gda.h>
#include <mooncake_ep_utils.cuh>

namespace mooncake {
namespace afpd {

// =====================================================================
// dispatch_afpd_send_kernel  — attn rank -> expert rank, single arena
// =====================================================================
template <bool kUseFP8, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
dispatch_afpd_send_kernel(
    void* mxa_buffer,
    int* /*rdma_send_signal_buffer*/,        // unused on M1 (NVLink-only path)
    int* rdma_recv_signal_buffer,             // already arena-shifted
    void* rdma_send_data_buffer,
    void* rdma_recv_data_buffer,              // already arena-shifted
    void* /*raddrs*/, void* /*rkeys*/, void* /*qp_devctxs*/,
    const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
    const void* x, const int64_t* topk_idx,
    int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
    int* next_clean_buffer,
    int num_tokens, int num_max_dispatch_tokens_per_rank,
    int num_topk, int num_experts, int num_local_experts_per_expert_rank,
    int /*my_rank*/, int expert_rank_base, int /*num_world_ranks*/)
{
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    constexpr int kNumPerChannels = 128;
    constexpr float kFP8Margin = 1e-4, kFP8Amax = 448, kFP8AmaxInv = 1.0f / 448.0f;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_send_data_buffer) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                auto int4_value = __ldg(x_int4 + i);
                if (kUseFP8) {
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = __bfloat162float(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = half_warp_reduce_max(amax);
                    scale = kFP8Amax / amax;
                    scale_inv = amax * kFP8AmaxInv;
                    if (lane_id == 0 or lane_id == 16)
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;
                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;
                } else {
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);

                // AFPD asymmetric routing: kernel computes dst_rank per token.
                // Single-expert (M1) collapses naturally: num_local_experts_per_expert_rank
                // == num_experts -> dst_rank = expert_rank_base, dst_expert_local_idx = dst_expert_idx.
                const auto dst_rank = expert_rank_base + (dst_expert_idx / num_local_experts_per_expert_rank);
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts_per_expert_rank;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_data_buffer) +
                                     dst_expert_local_idx * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;

                EP_DEVICE_ASSERT(nvlink_available[dst_rank] != 0);
                {
                    size_t offset = (char *)dst_ptr - (char *)(mxa_buffer);
                    void* peer_dst_ptr = (char *)ipc_peer_ptrs[dst_rank] + offset;
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(peer_dst_ptr);
                    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32) next_clean_buffer[i] = 0;
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }
        int expert_count[kNumWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * kNumWarpGroups;
        const auto expert_end_idx = min(expert_begin_idx + kNumWarpGroups, num_experts);
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        // Same asymmetric mapping for the per-expert count flush.
        const auto dst_rank = expert_rank_base + (responsible_expert_idx / num_local_experts_per_expert_rank);
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts_per_expert_rank;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * kNumWarpGroups];

        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);

        EP_DEVICE_ASSERT(nvlink_available[dst_rank] != 0);
        {
            int* signal_ptr = rdma_recv_signal_buffer + dst_expert_local_idx;
            size_t offset = (char *)signal_ptr - (char *)(mxa_buffer);
            int* peer_signal_ptr = (int *)((char *)ipc_peer_ptrs[dst_rank] + offset);
            st_na_release(peer_signal_ptr, -num_tokens_sent - 1);
        }
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
    }
    __syncwarp();
}

// =====================================================================
// dispatch_afpd_recv_kernel  — expert rank, single arena, single-shot
// =====================================================================
template <bool kUseFP8, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
dispatch_afpd_recv_kernel(
    void* packed_recv_x, float* packed_recv_x_scales,
    int* packed_recv_src_info, int64_t* packed_recv_layout_range,
    int* packed_recv_count, int32_t* active_ranks,
    int* rdma_recv_signal_buffer, void* rdma_recv_data_buffer,
    int num_max_dispatch_tokens_per_rank, int num_local_experts,
    int64_t timeout_ticks)
{
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    constexpr int kNumPerChannels = 128;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));

    if (responsible_expert_idx >= num_local_experts) return;

    const auto local_expert_idx = responsible_expert_idx;
    const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) +
            local_expert_idx * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
    const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
            local_expert_idx * num_max_dispatch_tokens_per_rank * hidden_int4;
    const auto recv_x_scales = packed_recv_x_scales + local_expert_idx * num_max_dispatch_tokens_per_rank * num_scales;
    const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_max_dispatch_tokens_per_rank;
    const auto recv_range = packed_recv_layout_range + local_expert_idx;

    __shared__ int shared_num_recv_tokens[kNumWarpGroups], shared_recv_token_begin_idx[kNumWarpGroups];
    int num_recv_tokens = 0, recv_token_begin_idx = 0;

    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
    if (sub_warp_id == 1 and lane_id == 0) {
        unsigned long long start_time = clock64();
        while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_signal_buffer + local_expert_idx)) == 0) {
            unsigned long long end_time = clock64();
            if (timeout_ticks != -1 && end_time - start_time > timeout_ticks) active_ranks[0] = 0;
            if (!active_ranks[0]) { num_recv_tokens = -1; break; }
        }
        num_recv_tokens = -num_recv_tokens - 1;
        recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
        shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
        shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
        *recv_range = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
        rdma_recv_signal_buffer[local_expert_idx] = 0;
    }
    asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(kNumWarpsPerGroup * 32));
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    EP_DEVICE_ASSERT(num_scales <= 64);
    for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
        const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
        if (lane_id == 0) recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
        __syncwarp();
        const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
        const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
        UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

        if (kUseFP8) {
            const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
            const auto dst_scales = reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
            const auto scale_stride = num_max_dispatch_tokens_per_rank;
            auto scale_0 = lane_id < num_scales ? ld_nc_global(src_scales + lane_id) : 0;
            auto scale_1 = (lane_id + 32) < num_scales ? ld_nc_global(src_scales + lane_id + 32) : 0;
            lane_id < num_scales ? dst_scales[lane_id * scale_stride] = scale_0 : 0.0f;
            (lane_id + 32) < num_scales ? dst_scales[(lane_id + 32) * scale_stride] = scale_1 : 0.0f;
        }
    }
}

// =====================================================================
// combine_afpd_send_kernel — expert rank -> attn rank, single arena
// =====================================================================
template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
combine_afpd_send_kernel(
    void* mxa_buffer,
    int* /*rdma_send_signal_buffer*/,
    int* rdma_recv_signal_buffer,
    void* rdma_send_data_buffer,
    void* rdma_recv_data_buffer,
    void* /*raddrs*/, void* /*rkeys*/, void* /*qp_devctxs*/,
    const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
    const void* x, const int* src_info, const int64_t* layout_range,
    int* next_clean_buffer, int* atomic_clean_flag,
    int num_max_dispatch_tokens_per_rank,
    int num_local_experts, int dst_attn_rank,
    int global_expert_base)
{
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_local_experts; i += 32) next_clean_buffer[i] = 0;
        __syncwarp();
        if (lane_id == 0) atomic_add_release_global(atomic_clean_flag, num_local_experts);
    }

    if (responsible_expert_idx < num_local_experts) {
        const auto local_expert_idx = responsible_expert_idx;
        // AFPD: layout_range entry is per-local-expert (single src per arena).
        const auto layout = __ldg(layout_range + local_expert_idx);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_data_buffer) +
                local_expert_idx * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);
            auto src_idx = __ldg(local_src_info + token_idx);
            // AFPD: dst layout is [local_expert][src_idx][bytes] on attn-side mirror
            // (one global expert dim — attn keeps a recv slot per-(expert,src_idx)
            // which maps back to its original token via src_idx).
            const auto global_expert_idx = global_expert_base + local_expert_idx;
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_data_buffer) +
                                 (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;

            EP_DEVICE_ASSERT(nvlink_available[dst_attn_rank] != 0);
            {
                size_t off = (char *)dst_ptr - (char *)(mxa_buffer);
                void* peer_dst_ptr = (char *)ipc_peer_ptrs[dst_attn_rank] + off;
                const auto dst_int4_ptr = reinterpret_cast<int4*>(peer_dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            }
        }

        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            int* signal_ptr = rdma_recv_signal_buffer + global_expert_base + local_expert_idx;
            size_t off = (char *)signal_ptr - (char *)(mxa_buffer);
            int* peer_signal_ptr = (int *)((char *)ipc_peer_ptrs[dst_attn_rank] + off);
            st_na_release(peer_signal_ptr, 1);
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
    }
}

// =====================================================================
// combine_afpd_recv_kernel — attn rank, blocks on flags then reduces
// =====================================================================
template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
combine_afpd_recv_kernel(
    void* combined_x, int32_t* active_ranks,
    int* rdma_recv_signal_buffer, void* rdma_recv_data_buffer,
    const int64_t* topk_idx, const float* topk_weights,
    int num_combined_tokens, int num_topk,
    int num_max_dispatch_tokens_per_rank, int num_experts,
    int64_t timeout_ticks)
{
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);

    // Wait for all experts to have signalled.
    if (responsible_expert_idx < num_experts) {
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0) {
            unsigned long long start_time = clock64();
            while (ld_acquire_sys_global(rdma_recv_signal_buffer + responsible_expert_idx) == 0) {
                unsigned long long end_time = clock64();
                if (timeout_ticks != -1 && end_time - start_time > timeout_ticks) active_ranks[0] = 0;
                if (!active_ranks[0]) break;
            }
            // Reset signal slot for next round.
            rdma_recv_signal_buffer[responsible_expert_idx] = 0;
        }
    }
    cooperative_groups::this_grid().sync();

    EP_DEVICE_ASSERT(num_topk <= 32 and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_recv_data_buffer) +
                                       (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += __bfloat162float(x_bf16[j]) * reg_topk_weights[i];
            }

            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = __float2bfloat16(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
}

// =====================================================================
// Host launchers
// =====================================================================
void dispatch_afpd_send(void* mxa_buffer,
                        int* rdma_send_signal_buffer,
                        void* rdma_send_data_buffer,
                        void* raddrs, void* rkeys, void* qp_devctxs,
                        const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
                        const void* x, const int64_t* topk_idx,
                        int num_tokens, int hidden,
                        int num_max_dispatch_tokens_per_rank,
                        int num_topk, int num_experts,
                        int my_attn_rank, int expert_rank_base,
                        int num_attn_ranks, int num_expert_ranks,
                        int64_t dst_arena_offset_bytes,
                        bool use_fp8,
                        void* workspace, cudaStream_t stream,
                        int64_t timeout_ticks, int phases)
{
    const int num_world_ranks = num_attn_ranks + 1;
    constexpr int kNumWarpsPerGroup = 4;
    constexpr int kNumWarpGroups = 8;
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);

    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    auto next_clean_buffer = atomic_finish_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 3 <= NUM_WORKSPACE_BYTES);

    auto rdma_recv_signal_buffer = reinterpret_cast<int*>(
        reinterpret_cast<char*>(mxa_buffer) + dst_arena_offset_bytes);
    auto rdma_recv_data_buffer = reinterpret_cast<void*>(
        reinterpret_cast<char*>(rdma_recv_signal_buffer) + num_experts * sizeof(int));

const int num_local_experts_per_expert_rank = num_experts / num_expert_ranks;
    EP_HOST_ASSERT(num_experts % num_expert_ranks == 0);

#define DISPATCH_AFPD_SEND_LAUNCH_CASE(hidden) {                                                  \
    auto kernel_func = use_fp8                                                                     \
        ? dispatch_afpd_send_kernel<true,  kNumWarpGroups, kNumWarpsPerGroup, hidden>             \
        : dispatch_afpd_send_kernel<false, kNumWarpGroups, kNumWarpsPerGroup, hidden>;            \
    LAUNCH_KERNEL(&cfg, kernel_func,                                                               \
                  mxa_buffer,                                                                      \
                  rdma_send_signal_buffer, rdma_recv_signal_buffer,                                \
                  rdma_send_data_buffer, rdma_recv_data_buffer,                                    \
                  raddrs, rkeys, qp_devctxs,                                                       \
                  nvlink_available, ipc_peer_ptrs,                                                 \
                  x, topk_idx,                                                                     \
                  atomic_counter_per_expert, atomic_finish_counter_per_expert,                     \
                  next_clean_buffer,                                                               \
                  num_tokens, num_max_dispatch_tokens_per_rank,                                    \
                  num_topk, num_experts, num_local_experts_per_expert_rank,                       \
                  my_attn_rank, expert_rank_base, num_world_ranks);                                 \
} break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_AFPD_SEND_LAUNCH_CASE);
#undef DISPATCH_AFPD_SEND_LAUNCH_CASE
    (void)timeout_ticks;
    (void)phases;
}

void dispatch_afpd_recv(void* packed_recv_x, float* packed_recv_x_scales,
                        int* packed_recv_src_info, int64_t* packed_recv_layout_range,
                        int* packed_recv_count, int32_t* active_ranks,
                        void* /*mxa_buffer*/,
                        int* rdma_recv_signal_buffer, void* rdma_recv_data_buffer,
                        int hidden, int num_max_dispatch_tokens_per_rank,
                        int /*num_topk*/, int num_local_experts,
                        int /*src_attn_rank*/, int /*num_attn_ranks*/,
                        bool use_fp8, void* /*workspace*/, cudaStream_t stream,
                        int64_t timeout_ticks)
{
    constexpr int kNumWarpsPerGroup = 4;
    constexpr int kNumWarpGroups = 8;
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_local_experts, kNumWarpGroups);

#define DISPATCH_AFPD_RECV_LAUNCH_CASE(hidden) {                                                   \
    auto kernel_func = use_fp8                                                                      \
        ? dispatch_afpd_recv_kernel<true,  kNumWarpGroups, kNumWarpsPerGroup, hidden>              \
        : dispatch_afpd_recv_kernel<false, kNumWarpGroups, kNumWarpsPerGroup, hidden>;             \
    LAUNCH_KERNEL(&cfg, kernel_func,                                                                \
                  packed_recv_x, packed_recv_x_scales,                                              \
                  packed_recv_src_info, packed_recv_layout_range,                                   \
                  packed_recv_count, active_ranks,                                                  \
                  rdma_recv_signal_buffer, rdma_recv_data_buffer,                                   \
                  num_max_dispatch_tokens_per_rank, num_local_experts,                              \
                  timeout_ticks);                                                                   \
} break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_AFPD_RECV_LAUNCH_CASE);
#undef DISPATCH_AFPD_RECV_LAUNCH_CASE
}

void combine_afpd_send(void* mxa_buffer,
                       int* rdma_send_signal_buffer,
                       void* rdma_send_data_buffer,
                       void* raddrs, void* rkeys, void* qp_devctxs,
                       const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
                       const void* x, const int* src_info,
                       const int64_t* layout_range,
                       int num_max_dispatch_tokens_per_rank,
                       int hidden, int /*num_topk*/, int num_local_experts,
                       int my_expert_rank_in_role, int dst_attn_rank,
                       int64_t dst_arena_offset_bytes,
                       void* workspace, cudaStream_t stream, int /*phases*/)
{
    constexpr int kNumWarpsPerGroup = 4;
    constexpr int kNumWarpGroups = 8;
    constexpr int kNumMaxTopk = 11;
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_local_experts, kNumWarpGroups);

    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    auto next_clean_buffer = reinterpret_cast<int*>(atomic_clean_flag + 1);
    EP_HOST_ASSERT((1 + num_local_experts) * sizeof(int) <= NUM_WORKSPACE_BYTES);

    auto rdma_recv_signal_buffer = reinterpret_cast<int*>(
        reinterpret_cast<char*>(mxa_buffer) + dst_arena_offset_bytes);
    auto rdma_recv_data_buffer = reinterpret_cast<void*>(
        reinterpret_cast<char*>(rdma_recv_signal_buffer) + num_local_experts * sizeof(int));

const int global_expert_base = my_expert_rank_in_role * num_local_experts;

#define COMBINE_AFPD_SEND_LAUNCH_CASE(hidden) {                                                    \
    auto kernel_func = combine_afpd_send_kernel<kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>; \
    LAUNCH_KERNEL(&cfg, kernel_func,                                                                \
                  mxa_buffer,                                                                       \
                  rdma_send_signal_buffer, rdma_recv_signal_buffer,                                 \
                  rdma_send_data_buffer, rdma_recv_data_buffer,                                     \
                  raddrs, rkeys, qp_devctxs,                                                        \
                  nvlink_available, ipc_peer_ptrs,                                                  \
                  x, src_info, layout_range,                                                        \
                  next_clean_buffer, atomic_clean_flag,                                             \
                  num_max_dispatch_tokens_per_rank, num_local_experts, dst_attn_rank,               \
                  global_expert_base);              \
} break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_AFPD_SEND_LAUNCH_CASE);
#undef COMBINE_AFPD_SEND_LAUNCH_CASE
}

void combine_afpd_recv(void* combined_x, int32_t* active_ranks,
                       void* /*mxa_buffer*/,
                       int* rdma_recv_signal_buffer, void* rdma_recv_data_buffer,
                       const int64_t* topk_idx, const float* topk_weights,
                       int num_combined_tokens, int hidden, int num_topk,
                       int num_max_dispatch_tokens_per_rank, int num_experts,
                       int /*my_attn_rank*/, int /*num_attn_ranks*/,
                       void* /*workspace*/, cudaStream_t stream,
                       int64_t timeout_ticks)
{
    constexpr int kNumWarpsPerGroup = 4;
    constexpr int kNumWarpGroups = 8;
    constexpr int kNumMaxTopk = 11;
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_AFPD_RECV_LAUNCH_CASE(hidden) {                                                    \
    auto kernel_func = combine_afpd_recv_kernel<kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>; \
    LAUNCH_KERNEL(&cfg, kernel_func,                                                                \
                  combined_x, active_ranks,                                                         \
                  rdma_recv_signal_buffer, rdma_recv_data_buffer,                                   \
                  topk_idx, topk_weights,                                                           \
                  num_combined_tokens, num_topk,                                                    \
                  num_max_dispatch_tokens_per_rank, num_experts,                                    \
                  timeout_ticks);                                                                   \
} break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_AFPD_RECV_LAUNCH_CASE);
#undef COMBINE_AFPD_RECV_LAUNCH_CASE
}

}  // namespace afpd
}  // namespace mooncake
