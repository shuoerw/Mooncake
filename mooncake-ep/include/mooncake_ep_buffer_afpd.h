// AFPD (Attention-FFN Prefill-Decode) extensions to MooncakeEpBuffer.
//
// These are role-aware variants of the symmetric dispatch/combine. Goal:
//   - attn ranks send only (dispatch) and receive only (combine)
//   - expert rank receives only (dispatch) and sends only (combine)
//   - the expert side allocates two physically-separate recv arenas
//     (PREFILL and DECODE) so the two attn sources never share a kernel
//     write target and thus never serialise on the SM scheduler.
//
// M1 POC topology (hard-coded constants below) — generalised in M2.
//
// PASS A (this commit): only declarations + thin host launchers that abort.
// Subsequent passes implement the kernels.
//
// The new entry points live in namespace mooncake::afpd to avoid clashing
// with mooncake::dispatch / mooncake::combine.

#ifndef MOONCAKE_EP_BUFFER_AFPD_H
#define MOONCAKE_EP_BUFFER_AFPD_H

#include <cstdint>
#include <cuda_runtime.h>

namespace mooncake {
namespace afpd {

// Arena IDs: which recv buffer on the expert side a dispatch is targeting.
enum AfpdArena : int {
    AFPD_ARENA_PREFILL = 0,
    AFPD_ARENA_DECODE = 1,
    AFPD_NUM_ARENAS = 2,
};

// dispatch_afpd_send — attn-side, sends tokens to the (single) expert rank.
// Mirrors mooncake::dispatch's send phase, with:
//   - dst_rank hardcoded to the expert rank index (sender knows from world layout)
//   - dst_ptr base += dst_arena_offset_bytes (selects PREFILL or DECODE arena)
//   - leading dim of recv buffer collapsed from num_ranks to 1
//
// Note: signature deliberately matches the existing dispatch as closely as
// possible so we can copy-and-modify the kernel in PASS B.
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
                        int64_t timeout_ticks, int phases);

// dispatch_afpd_recv — expert-side, persistent receiver for one arena.
// Reads from one arena's rdma_recv buffers, packs into packed_recv_x.
void dispatch_afpd_recv(void* packed_recv_x, float* packed_recv_x_scales,
                        int* packed_recv_src_info, int64_t* packed_recv_layout_range,
                        int* packed_recv_count, int32_t* active_ranks,
                        void* mxa_buffer,
                        int* rdma_recv_signal_buffer,
                        void* rdma_recv_data_buffer,
                        int hidden, int num_max_dispatch_tokens_per_rank,
                        int num_topk, int num_local_experts,
                        int src_attn_rank, int num_attn_ranks,
                        bool use_fp8,
                        void* workspace, cudaStream_t stream,
                        int64_t timeout_ticks);

// combine_afpd_send — expert-side, sends weighted expert outputs back to one
// attn rank. Mirrors mooncake::combine's send phase but the dst_rank is the
// attn rank corresponding to a single arena.
void combine_afpd_send(void* mxa_buffer,
                       int* rdma_send_signal_buffer,
                       void* rdma_send_data_buffer,
                       void* raddrs, void* rkeys, void* qp_devctxs,
                       const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
                       const void* x, const int* src_info,
                       const int64_t* layout_range,
                       int num_max_dispatch_tokens_per_rank,
                       int hidden, int num_topk, int num_local_experts,
                       int my_expert_rank_in_role, int dst_attn_rank,
                       int64_t dst_arena_offset_bytes,
                       void* workspace, cudaStream_t stream, int phases);

// combine_afpd_recv — attn-side, blocks on per-expert flags then reduces.
void combine_afpd_recv(void* combined_x, int32_t* active_ranks,
                       void* mxa_buffer,
                       int* rdma_recv_signal_buffer,
                       void* rdma_recv_data_buffer,
                       const int64_t* topk_idx, const float* topk_weights,
                       int num_combined_tokens, int hidden, int num_topk,
                       int num_max_dispatch_tokens_per_rank,
                       int num_experts,
                       int my_attn_rank, int num_attn_ranks,
                       void* workspace, cudaStream_t stream,
                       int64_t timeout_ticks);

}  // namespace afpd
}  // namespace mooncake

#endif  // MOONCAKE_EP_BUFFER_AFPD_H
