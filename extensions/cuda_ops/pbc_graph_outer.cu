#include "pbc_graph_outer.h"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int TILE = 32;              // Block tile size (must match warp size)
constexpr int BUFFER_SIZE = 256;      // Legacy constant (unused)
constexpr int STAGING_CAP = 4096;     // Shared-memory staging capacity per block (edges)

template<typename T>
__device__ __forceinline__ T get_epsilon() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<T>(1e-8);
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<T>(1e-12);
    } else {
        return static_cast<T>(1e-8);
    }
}

// =============================================================================
// New kernel: emit edges in-place with one global atomic reservation per block
// Two-phase: (1) count edges per block; (2) reserve global range once and write
// =============================================================================
template<typename scalar_t>
__global__ void radius_graph_pbc_emit_kernel(
    const scalar_t* __restrict__ pos,
    const int64_t* __restrict__ img_offset,
    const int64_t* __restrict__ natoms,
    const scalar_t* __restrict__ cell,
    const int32_t* __restrict__ rep,
    scalar_t radius_sq,
    int batch_size,
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_dst,
    int32_t* __restrict__ edge_rx,
    int32_t* __restrict__ edge_ry,
    int32_t* __restrict__ edge_rz,
    unsigned long long* __restrict__ global_edge_count,
    int64_t max_edges
) {
    // Shared memory for tiling positions
    extern __shared__ char shared_mem[];
    scalar_t* pos_i = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* pos_j = pos_i + TILE * 3;

    // Small shared state for phase synchronization and offsets
    __shared__ unsigned int warp_sums[32];
    __shared__ unsigned int block_count_sh;
    __shared__ unsigned int s_cursor;          // block-local cursor within reserved range
    __shared__ unsigned int s_limit;           // how many entries this block is allowed to write
    __shared__ unsigned long long s_base_out;  // global base offset for this block

    // Block identification
    int b = blockIdx.z;
    if (b >= batch_size) return;

    int n = natoms[b];
    int64_t offset = img_offset[b];

    int i_local = blockIdx.x * TILE + threadIdx.x;
    int j_local = blockIdx.y * TILE + threadIdx.y;
    bool active = (i_local < n) && (j_local < n);

    // Keep all threads; inactive threads will skip computations but still hit barriers

    int64_t i_global = offset + i_local;
    int64_t j_global = offset + j_local;

    // Cooperative load of tile corners
    if (threadIdx.y == 0 && i_local < n) {
        pos_i[threadIdx.x * 3 + 0] = pos[i_global * 3 + 0];
        pos_i[threadIdx.x * 3 + 1] = pos[i_global * 3 + 1];
        pos_i[threadIdx.x * 3 + 2] = pos[i_global * 3 + 2];
    }
    if (threadIdx.x == 0 && j_local < n) {
        pos_j[threadIdx.y * 3 + 0] = pos[j_global * 3 + 0];
        pos_j[threadIdx.y * 3 + 1] = pos[j_global * 3 + 1];
        pos_j[threadIdx.y * 3 + 2] = pos[j_global * 3 + 2];
    }

    // Load cell matrix rows for this image
    scalar_t a1[3], a2[3], a3[3];
    int cell_base = b * 9;
    a1[0] = cell[cell_base + 0]; a1[1] = cell[cell_base + 1]; a1[2] = cell[cell_base + 2];
    a2[0] = cell[cell_base + 3]; a2[1] = cell[cell_base + 4]; a2[2] = cell[cell_base + 5];
    a3[0] = cell[cell_base + 6]; a3[1] = cell[cell_base + 7]; a3[2] = cell[cell_base + 8];

    // Replication factors
    int rep_base = b * 3;
    int32_t rep_x = rep[rep_base + 0];
    int32_t rep_y = rep[rep_base + 1];
    int32_t rep_z = rep[rep_base + 2];
    int num_rep_x = 2 * rep_x + 1;
    int num_rep_y = 2 * rep_y + 1;
    int num_rep_z = 2 * rep_z + 1;

    __syncthreads();

    scalar_t eps = get_epsilon<scalar_t>();

    scalar_t pi0 = 0, pi1 = 0, pi2 = 0;
    scalar_t pj0 = 0, pj1 = 0, pj2 = 0;
    if (active) {
        pi0 = pos_i[threadIdx.x * 3 + 0];
        pi1 = pos_i[threadIdx.x * 3 + 1];
        pi2 = pos_i[threadIdx.x * 3 + 2];
        pj0 = pos_j[threadIdx.y * 3 + 0];
        pj1 = pos_j[threadIdx.y * 3 + 1];
        pj2 = pos_j[threadIdx.y * 3 + 2];
    }

    // Phase 1: count how many edges this block will emit
    unsigned int thread_edge_count = 0;

    for (int32_t rx = -rep_x; rx <= rep_x; rx++) {
        for (int32_t ry = -rep_y; ry <= rep_y; ry++) {
            for (int32_t rz = -rep_z; rz <= rep_z; rz++) {
                scalar_t offset_vec_x = rx * a1[0] + ry * a2[0] + rz * a3[0];
                scalar_t offset_vec_y = rx * a1[1] + ry * a2[1] + rz * a3[1];
                scalar_t offset_vec_z = rx * a1[2] + ry * a2[2] + rz * a3[2];

                bool is_valid = false;
                if (active) {
                    scalar_t dx = pi0 - (pj0 + offset_vec_x);
                    scalar_t dy = pi1 - (pj1 + offset_vec_y);
                    scalar_t dz = pi2 - (pj2 + offset_vec_z);
                    scalar_t d2 = dx * dx + dy * dy + dz * dz;
                    bool is_self_loop = (i_local == j_local) && (rx == 0) && (ry == 0) && (rz == 0);
                    is_valid = (d2 <= radius_sq) && (d2 > eps) && !is_self_loop;
                }
                thread_edge_count += static_cast<unsigned int>(is_valid);
            }
        }
    }

    // Warp-level reduction of thread_edge_count
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = linear_tid & 31;
    int warp_id = linear_tid >> 5;
    const unsigned int ACTIVE_MASK_WARP = __activemask();
    unsigned int val = __reduce_add_sync(ACTIVE_MASK_WARP, thread_edge_count);
    int leader_lane = __ffs(ACTIVE_MASK_WARP) - 1;
    if (lane == leader_lane) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned int block_count = 0;
        int warps_per_block = (blockDim.x * blockDim.y + 31) / 32;
        for (int w = 0; w < warps_per_block; w++) block_count += warp_sums[w];
        block_count_sh = block_count;
        s_cursor = 0;
    }
    __syncthreads();

    // Proceed even if block_count_sh==0; s_limit will be set to 0 and we will skip writes

    // Single global reservation per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned long long base = atomicAdd(global_edge_count, static_cast<unsigned long long>(block_count_sh));
        s_base_out = base;
        unsigned long long remaining = (max_edges > static_cast<int64_t>(base)) ? static_cast<unsigned long long>(max_edges - static_cast<int64_t>(base)) : 0ull;
        unsigned long long lim = (remaining < static_cast<unsigned long long>(block_count_sh)) ? remaining : static_cast<unsigned long long>(block_count_sh);
        s_limit = static_cast<unsigned int>(lim);
    }
    __syncthreads();

    // If s_limit==0, all subsequent writes are skipped; still continue to keep threads converged

    // Phase 2: write edges into the reserved contiguous block using
    // warp-aggregated shared cursor for coalesced writes
    for (int32_t rx = -rep_x; rx <= rep_x; rx++) {
        int32_t rx_off = rx;
        for (int32_t ry = -rep_y; ry <= rep_y; ry++) {
            int32_t ry_off = ry;
            for (int32_t rz = -rep_z; rz <= rep_z; rz++) {
                int32_t rz_off = rz;

                scalar_t offset_vec_x = rx_off * a1[0] + ry_off * a2[0] + rz_off * a3[0];
                scalar_t offset_vec_y = rx_off * a1[1] + ry_off * a2[1] + rz_off * a3[1];
                scalar_t offset_vec_z = rx_off * a1[2] + ry_off * a2[2] + rz_off * a3[2];

                bool is_valid = false;
                if (active) {
                    scalar_t dx = pi0 - (pj0 + offset_vec_x);
                    scalar_t dy = pi1 - (pj1 + offset_vec_y);
                    scalar_t dz = pi2 - (pj2 + offset_vec_z);
                    scalar_t d2 = dx * dx + dy * dy + dz * dz;
                    bool is_self_loop = (i_local == j_local) && (rx_off == 0) && (ry_off == 0) && (rz_off == 0);
                    is_valid = (d2 <= radius_sq) && (d2 > eps) && !is_self_loop;
                }

                unsigned int active_mask = __activemask();
                unsigned int ballot = __ballot_sync(active_mask, is_valid);
                int warp_cnt = __popc(ballot);
                if (warp_cnt == 0) continue;

                // Lane-local offset among valid threads
                unsigned int lanemask_lt = (1u << lane) - 1u;
                int lane_off = __popc(ballot & lanemask_lt);

                // Reserve a contiguous segment in the block-local cursor
                unsigned int local_base = 0u;
                int leader_lane2 = __ffs(active_mask) - 1;
                if (lane == leader_lane2) {
                    local_base = atomicAdd(&s_cursor, static_cast<unsigned int>(warp_cnt));
                }
                local_base = __shfl_sync(active_mask, local_base, leader_lane2);

                // Bounds check against block limit to avoid overflow writes
                unsigned int out_idx_in_block = local_base + static_cast<unsigned int>(lane_off);
                if (is_valid && out_idx_in_block < s_limit) {
                    unsigned long long gidx = s_base_out + static_cast<unsigned long long>(out_idx_in_block);
                    edge_src[gidx] = j_global;  // index2
                    edge_dst[gidx] = i_global;  // index1
                    edge_rx[gidx]  = rx_off;
                    edge_ry[gidx]  = ry_off;
                    edge_rz[gidx]  = rz_off;
                }
            }
        }
    }
}

template<typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
radius_graph_pbc_outer_cuda_template(
    torch::Tensor pos,
    torch::Tensor natoms,
    torch::Tensor img_offset,
    torch::Tensor cell,
    torch::Tensor rep,
    double radius
) {
    // ========================================
    // Step 1: Setup and estimates (no mask allocation)
    // ========================================
    
    auto device = pos.device();
    int batch_size = natoms.size(0);
    int64_t sum_natoms = pos.size(0);
    int64_t max_n = natoms.max().item<int64_t>();
    scalar_t radius_sq = static_cast<scalar_t>(radius * radius);
    
    auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_scalar = torch::TensorOptions().dtype(pos.dtype()).device(device);
    
    auto natoms_cpu = natoms.cpu();
    auto rep_cpu = rep.cpu();
    
    auto natoms_acc = natoms_cpu.accessor<int64_t, 1>();
    auto rep_acc = rep_cpu.accessor<int32_t, 2>();
    
    // ========================================
    // Step 2: Allocate outputs and launch emit kernel (with dynamic growth)
    // ========================================

    // Initial capacity guess
    dim3 grid(
        (max_n + TILE - 1) / TILE,
        (max_n + TILE - 1) / TILE,
        batch_size
    );
    dim3 block(TILE, TILE, 1);
    size_t shared_mem = 2 * TILE * 3 * sizeof(scalar_t);  // pos_i and pos_j

    auto start_time = std::chrono::high_resolution_clock::now();

    // Preallocate once using a tight upper bound; never reallocate
    int64_t capacity = 0;
    for (int b = 0; b < batch_size; b++) {
        int64_t n_b = natoms_acc[b];
        int32_t rx_b = rep_acc[b][0];
        int32_t ry_b = rep_acc[b][1];
        int32_t rz_b = rep_acc[b][2];
        int64_t prod = static_cast<int64_t>(2 * rx_b + 1) * static_cast<int64_t>(2 * ry_b + 1) * static_cast<int64_t>(2 * rz_b + 1);
        capacity += n_b * n_b * prod / 16 - n_b; // exclude self loops for (0,0,0)
    }

    torch::Tensor edge_src = torch::empty({capacity}, options_int64);
    torch::Tensor edge_dst = torch::empty({capacity}, options_int64);
    torch::Tensor edge_rx = torch::empty({capacity}, options_int32);
    torch::Tensor edge_ry = torch::empty({capacity}, options_int32);
    torch::Tensor edge_rz = torch::empty({capacity}, options_int32);
    torch::Tensor global_edge_count = torch::zeros({1}, options_int64);

    radius_graph_pbc_emit_kernel<scalar_t><<<grid, block, shared_mem>>>(
        pos.data_ptr<scalar_t>(),
        img_offset.data_ptr<int64_t>(),
        natoms.data_ptr<int64_t>(),
        cell.data_ptr<scalar_t>(),
        rep.data_ptr<int32_t>(),
        radius_sq,
        batch_size,
        edge_src.data_ptr<int64_t>(),
        edge_dst.data_ptr<int64_t>(),
        edge_rx.data_ptr<int32_t>(),
        edge_ry.data_ptr<int32_t>(),
        edge_rz.data_ptr<int32_t>(),
        reinterpret_cast<unsigned long long*>(global_edge_count.data_ptr<int64_t>()),
        capacity
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "radius_graph_pbc_emit_kernel launch failed: ", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess,
                "radius_graph_pbc_emit_kernel execution failed: ", cudaGetErrorString(err));
    
    // ========================================
    // Step 4: Post-process results
    // ========================================

    auto before_post_process_time = std::chrono::high_resolution_clock::now();
    
    int64_t actual_edges = global_edge_count.item<int64_t>();
    
    if (actual_edges == 0) {
        edge_src = torch::empty({0}, options_int64);
        edge_dst = torch::empty({0}, options_int64);
        torch::Tensor edge_unit_cell = torch::empty({0, 3}, options_scalar);
        torch::Tensor num_neighbors_image = torch::zeros({batch_size}, options_int64);
        torch::Tensor edge_index = torch::stack({edge_src, edge_dst}, 0);
        return std::make_tuple(edge_index, edge_unit_cell, num_neighbors_image);
    }
    
    // Trim to actual size
    edge_src = edge_src.slice(0, 0, actual_edges);
    edge_dst = edge_dst.slice(0, 0, actual_edges);
    edge_rx = edge_rx.slice(0, 0, actual_edges);
    edge_ry = edge_ry.slice(0, 0, actual_edges);
    edge_rz = edge_rz.slice(0, 0, actual_edges);
    
    // Combine unit cell offsets
    torch::Tensor edge_unit_cell = torch::stack({
        edge_rx.to(pos.dtype()),
        edge_ry.to(pos.dtype()),
        edge_rz.to(pos.dtype())
    }, 1);
    
    // Count edges per image on GPU without sorting
    torch::Tensor boundaries = img_offset.slice(0, 1, batch_size + 1);
    torch::Tensor bins = torch::bucketize(edge_dst, boundaries, /*right=*/false).to(torch::kLong);
    torch::Tensor num_neighbors_image = torch::zeros({batch_size}, bins.options());
    num_neighbors_image.scatter_add_(0, bins, torch::ones_like(bins));
    
    // Assemble edge_index
    torch::Tensor edge_index = torch::stack({edge_src, edge_dst}, 0);

    auto after_post_process_time = std::chrono::high_resolution_clock::now();
    auto post_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_post_process_time - before_post_process_time).count();
    // std::cout << "[DEBUG] Post process time: " << post_process_time << " ms" << std::endl;
    
    return std::make_tuple(edge_index, edge_unit_cell, num_neighbors_image);
}

// =============================================================================
// CUDA Implementation Entry Point
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
radius_graph_pbc_outer_cuda_impl(
    torch::Tensor pos,
    torch::Tensor natoms,
    torch::Tensor img_offset,
    torch::Tensor cell,
    torch::Tensor rep,
    double radius
) {
    // Dispatch based on dtype
    if (pos.dtype() == torch::kFloat32) {
        return radius_graph_pbc_outer_cuda_template<float>(
            pos, natoms, img_offset, cell, rep, radius
        );
    } else if (pos.dtype() == torch::kFloat64) {
        return radius_graph_pbc_outer_cuda_template<double>(
            pos, natoms, img_offset, cell, rep, radius
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float32 and float64 are supported");
    }
}

