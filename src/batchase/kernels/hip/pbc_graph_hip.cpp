#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <type_traits>

// Template function to get appropriate epsilon for different floating point types
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

// Templated HIP kernel for computing pairwise distances with PBC offsets
template<typename T>
__global__ void pbc_distance_kernel_optimized(
    const T* pos1,
    const T* pos2,
    const T* pbc_offsets,
    const int64_t* num_atoms_per_image_sqr,
    const int64_t* batch_offsets,
    T* distances_squared,
    bool* valid_mask,
    int num_pairs,
    T radius_squared
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        int batch_idx = 0;
        while (batch_idx < num_pairs && idx >= batch_offsets[batch_idx + 1]) {
            batch_idx++;
        }

        T offset_x = pbc_offsets[batch_idx * 3];
        T offset_y = pbc_offsets[batch_idx * 3 + 1];
        T offset_z = pbc_offsets[batch_idx * 3 + 2];

        T dx = pos2[idx * 3] - pos1[idx * 3] + offset_x;
        T dy = pos2[idx * 3 + 1] - pos1[idx * 3 + 1] + offset_y;
        T dz = pos2[idx * 3 + 2] - pos1[idx * 3 + 2] + offset_z;

        T dist_sq = dx * dx + dy * dy + dz * dz;
        distances_squared[idx] = dist_sq;

        valid_mask[idx] = (dist_sq <= radius_squared) && (dist_sq > get_epsilon<T>());
    }
}

// Helper to launch HIP kernel
template<typename T>
inline void launch_pbc_distance_kernel_optimized(
    const T* pos1,
    const T* pos2,
    const T* pbc_offsets,
    const int64_t* num_atoms_per_image_sqr,
    const int64_t* batch_offsets,
    T* distances_squared,
    bool* valid_mask,
    int num_pairs,
    T radius_squared,
    int blocks,
    int threads_per_block
) {
    hipLaunchKernelGGL(
        pbc_distance_kernel_optimized<T>,
        dim3(blocks),
        dim3(threads_per_block),
        0,
        at::cuda::getCurrentCUDAStream(),
        pos1,
        pos2,
        pbc_offsets,
        num_atoms_per_image_sqr,
        batch_offsets,
        distances_squared,
        valid_mask,
        num_pairs,
        radius_squared
    );
}

// HIP function to compute distances for all unit cell offsets
std::vector<torch::Tensor> pbc_distance_hip(
    torch::Tensor pos1,
    torch::Tensor pos2,
    torch::Tensor data_cell,
    torch::Tensor num_atoms_per_image_sqr,
    int batch_size,
    std::vector<int> max_rep,
    float radius,
    torch::Device device
) {
    pos1 = pos1.to(device).contiguous();
    pos2 = pos2.to(device).contiguous();
    data_cell = data_cell.to(device).contiguous();
    num_atoms_per_image_sqr = num_atoms_per_image_sqr.to(device);

    TORCH_CHECK(pos1.dtype() == pos2.dtype(), "pos1 and pos2 must have the same dtype");
    TORCH_CHECK(pos1.dtype() == data_cell.dtype(), "pos1 and data_cell must have the same dtype");

    bool is_float64 = pos1.dtype() == torch::kFloat64;
    int num_pairs = pos1.size(0);

    std::vector<torch::Tensor> all_index1, all_unit_cell, all_distances_sq;

    torch::Tensor base_indices = torch::arange(num_pairs, torch::dtype(torch::kLong).device(device));

    int threads_per_block = 512;
    int blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

    torch::Tensor distances_squared = torch::zeros({num_pairs}, torch::dtype(pos1.dtype()).device(device));
    torch::Tensor valid_mask = torch::zeros({num_pairs}, torch::dtype(torch::kBool).device(device));
    torch::Tensor unit_cell_offset = torch::zeros({3}, torch::dtype(pos1.dtype()).device(device));
    torch::Tensor unit_cell_offset_batch = torch::zeros({batch_size, 3, 1}, torch::dtype(pos1.dtype()).device(device));

    torch::Tensor batch_offsets = torch::zeros({batch_size + 1}, torch::dtype(torch::kLong).device(device));
    torch::Tensor cumsum = torch::cumsum(num_atoms_per_image_sqr, 0);
    batch_offsets.slice(0, 1, batch_size + 1) = cumsum;

    for (int i = -max_rep[0]; i <= max_rep[0]; i++) {
        for (int j = -max_rep[1]; j <= max_rep[1]; j++) {
            for (int k = -max_rep[2]; k <= max_rep[2]; k++) {
                unit_cell_offset[0] = static_cast<float>(i);
                unit_cell_offset[1] = static_cast<float>(j);
                unit_cell_offset[2] = static_cast<float>(k);

                unit_cell_offset_batch.select(2, 0) = unit_cell_offset.unsqueeze(0).expand({batch_size, -1});
                torch::Tensor pbc_offsets = torch::bmm(data_cell, unit_cell_offset_batch).squeeze(-1);

                if (is_float64) {
                    double radius_squared = static_cast<double>(radius) * static_cast<double>(radius);
                    launch_pbc_distance_kernel_optimized<double>(
                        pos1.data_ptr<double>(),
                        pos2.data_ptr<double>(),
                        pbc_offsets.data_ptr<double>(),
                        num_atoms_per_image_sqr.data_ptr<int64_t>(),
                        batch_offsets.data_ptr<int64_t>(),
                        distances_squared.data_ptr<double>(),
                        valid_mask.data_ptr<bool>(),
                        num_pairs,
                        radius_squared,
                        blocks,
                        threads_per_block
                    );
                } else {
                    float radius_squared = radius * radius;
                    launch_pbc_distance_kernel_optimized<float>(
                        pos1.data_ptr<float>(),
                        pos2.data_ptr<float>(),
                        pbc_offsets.data_ptr<float>(),
                        num_atoms_per_image_sqr.data_ptr<int64_t>(),
                        batch_offsets.data_ptr<int64_t>(),
                        distances_squared.data_ptr<float>(),
                        valid_mask.data_ptr<bool>(),
                        num_pairs,
                        radius_squared,
                        blocks,
                        threads_per_block
                    );
                }

                torch::Tensor valid_indices = torch::nonzero(valid_mask).squeeze(-1);
                if (valid_indices.numel() > 0) {
                    torch::Tensor valid_base_indices = base_indices.index_select(0, valid_indices);
                    torch::Tensor valid_distances = distances_squared.index_select(0, valid_indices);
                    torch::Tensor valid_unit_cell = unit_cell_offset.unsqueeze(0).repeat({valid_indices.size(0), 1});

                    all_index1.push_back(valid_base_indices);
                    all_unit_cell.push_back(valid_unit_cell);
                    all_distances_sq.push_back(valid_distances);
                }
            }
        }
    }

    hipDeviceSynchronize();

    torch::Tensor final_indices, final_unit_cell, final_distances;
    if (all_index1.size() > 0) {
        final_indices = torch::cat(all_index1);
        final_unit_cell = torch::cat(all_unit_cell);
        final_distances = torch::cat(all_distances_sq);
    } else {
        final_indices = torch::empty({0}, torch::dtype(torch::kLong).device(device));
        final_unit_cell = torch::empty({0, 3}, torch::dtype(pos1.dtype()).device(device));
        final_distances = torch::empty({0}, torch::dtype(pos1.dtype()).device(device));
    }

    return {final_indices, final_unit_cell, final_distances};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbc_distance_hip", &pbc_distance_hip, "PBC distance computation with HIP");
}
