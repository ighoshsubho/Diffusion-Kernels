#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include "linear_attention_cuda_kernel.cuh"

torch::Tensor linear_attention_cuda_forward(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values) {
    
    // Get tensor dimensions
    const int batch_size = queries.size(0);
    const int num_heads = queries.size(1);
    const int seq_len = queries.size(2);
    const int head_dim = queries.size(3);
    
    // Create output tensor
    auto output = torch::zeros_like(values);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel with appropriate scalar type
    AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "linear_attention_forward", ([&] {
        launch_linear_attention_forward<scalar_t>(
            queries.data_ptr<scalar_t>(),
            keys.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len, head_dim,
            stream
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_attention_cuda_forward, "Linear Attention forward (CUDA)");
}