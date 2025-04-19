#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Naive 3D convolution kernel
void launchNaiveConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream = 0
);

// Coalesced 3D convolution kernel
void launchCoalescedConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream = 0
);

// Tiled 3D convolution kernel
void launchTiledConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream = 0
);

// cuDNN 3D convolution
void launchCudnnConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream = 0
);

#endif // KERNELS_H
