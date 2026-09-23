#include "pbc_graph_outer.h"
#include <torch/extension.h>

// ============================================================================
// Type and Device Checking
// ============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// C++ Wrapper: Type Checking and Preprocessing
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
radius_graph_pbc_outer_cuda(
    torch::Tensor pos,          // [sumN, 3]
    torch::Tensor natoms,       // [B]
    torch::Tensor img_offset,   // [B+1]
    torch::Tensor cell,         // [B, 3, 3]
    torch::Tensor rep,          // [B, 3]
    double radius
) {
    // ========================================
    // Input Validation
    // ========================================
    
    // Device checks
    CHECK_CUDA(pos);
    CHECK_CUDA(natoms);
    CHECK_CUDA(img_offset);
    CHECK_CUDA(cell);
    CHECK_CUDA(rep);
    
    // Contiguous checks
    CHECK_CONTIGUOUS(pos);
    CHECK_CONTIGUOUS(cell);
    
    // Shape checks
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, 
                "pos must be [sumN, 3]");
    TORCH_CHECK(natoms.dim() == 1, 
                "natoms must be [B]");
    TORCH_CHECK(img_offset.dim() == 1 && img_offset.size(0) == natoms.size(0) + 1,
                "img_offset must be [B+1]");
    TORCH_CHECK(cell.dim() == 3 && cell.size(1) == 3 && cell.size(2) == 3,
                "cell must be [B, 3, 3]");
    TORCH_CHECK(rep.dim() == 2 && rep.size(0) == natoms.size(0) && rep.size(1) == 3,
                "rep must be [B, 3]");
    
    // Dtype checks
    TORCH_CHECK(pos.dtype() == cell.dtype(),
                "pos and cell must have the same dtype");
    TORCH_CHECK(natoms.dtype() == torch::kInt64,
                "natoms must be int64");
    TORCH_CHECK(img_offset.dtype() == torch::kInt64,
                "img_offset must be int64");
    TORCH_CHECK(rep.dtype() == torch::kInt32,
                "rep must be int32");
    
    // Value checks
    TORCH_CHECK(radius > 0, "radius must be positive");
    
    int64_t batch_size = natoms.size(0);
    int64_t sum_natoms = pos.size(0);
    
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(sum_natoms > 0, "sum_natoms must be positive");
    
    // ========================================
    // Ensure all tensors are on the same device
    // ========================================
    
    auto device = pos.device();
    natoms = natoms.to(device).contiguous();
    img_offset = img_offset.to(device).contiguous();
    rep = rep.to(device).contiguous();
    pos = pos.contiguous();
    cell = cell.contiguous();
    
    // ========================================
    // Call CUDA implementation
    // ========================================
    
    return radius_graph_pbc_outer_cuda_impl(
        pos, natoms, img_offset, cell, rep, radius
    );
}

// ============================================================================
// PyBind11 Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radius_graph_pbc_outer_cuda", 
          &radius_graph_pbc_outer_cuda,
          "Outer-product/tiling-based PBC radius graph (CUDA)",
          py::arg("pos"),
          py::arg("natoms"),
          py::arg("img_offset"),
          py::arg("cell"),
          py::arg("rep"),
          py::arg("radius"));
}

