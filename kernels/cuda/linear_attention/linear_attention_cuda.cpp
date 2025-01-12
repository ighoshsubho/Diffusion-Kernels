// linear_attention_cuda.cpp
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor linear_attention_cuda_forward(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_attention_cuda_forward, "Linear Attention forward (CUDA)");
}