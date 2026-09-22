#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

// HIP kernel for vector addition
__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* result,
    int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// HIP launcher function
torch::Tensor vector_add_hip(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Input tensors must be on HIP/ROCm device (torch CUDA backend)");

    a = a.contiguous();
    b = b.contiguous();

    auto result = torch::zeros_like(a);
    auto numel = a.numel();

    const int threads_per_block = 256;
    const int blocks = (numel + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        vector_add_kernel,
        dim3(blocks),
        dim3(threads_per_block),
        0,
        at::cuda::getCurrentCUDAStream(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        numel
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP error: ", hipGetErrorString(err));

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_hip, "Vector addition (HIP implementation)");
}
