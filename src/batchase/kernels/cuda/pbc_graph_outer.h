#pragma once

#include <torch/extension.h>
#include <vector>

// ============================================================================
// C++ Interface Layer
// ============================================================================

/**
 * @brief Outer-product/tiling-based PBC radius graph construction (CUDA)
 * 
 * @param pos           [sumN, 3] atom positions, float32/64, CUDA
 * @param natoms        [B] number of atoms per image, int64, CUDA
 * @param img_offset    [B+1] cumulative atom offsets, int64, CUDA
 * @param cell          [B, 3, 3] cell matrices (row vectors), float32/64, CUDA
 * @param rep           [B, 3] PBC replication numbers, int32, CUDA
 * @param radius        cutoff radius (scalar)
 * 
 * @return tuple of (edge_index, unit_cell, num_neighbors_image)
 *         - edge_index: [2, E] int64, [src, dst] = [index2, index1]
 *         - unit_cell: [E, 3] same dtype as pos, PBC offsets (rx, ry, rz)
 *         - num_neighbors_image: [B] int64, edge count per image
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
radius_graph_pbc_outer_cuda(
    torch::Tensor pos,
    torch::Tensor natoms,
    torch::Tensor img_offset,
    torch::Tensor cell,
    torch::Tensor rep,
    double radius
);

// ============================================================================
// CUDA Implementation Layer (declared in .cu)
// ============================================================================

/**
 * @brief CUDA implementation (guaranteed contiguous and on CUDA)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
radius_graph_pbc_outer_cuda_impl(
    torch::Tensor pos,
    torch::Tensor natoms,
    torch::Tensor img_offset,
    torch::Tensor cell,
    torch::Tensor rep,
    double radius
);

/**
 * @brief Template dispatcher for float32/float64
 */
template<typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
radius_graph_pbc_outer_cuda_template(
    torch::Tensor pos,
    torch::Tensor natoms,
    torch::Tensor img_offset,
    torch::Tensor cell,
    torch::Tensor rep,
    double radius
);

