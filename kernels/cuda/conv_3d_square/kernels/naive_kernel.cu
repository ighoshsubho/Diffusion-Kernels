#include <cuda_runtime.h>
#include "../include/kernels.h"

__global__ void naiveConv3DKernel(
    const float* A,
    const float* B,
    float* C,
    size_t size,
    size_t K,
    int radius
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < size && j < size && k < size){
        float sum = 0.0f;

        for (int x = 0; x < K; x++){
            for (int y = 0; y < K; y++){
                for (int z=0; z < K; z++){
                    int iPos = i + (x - radius);
                    int jPos = j + (y - radius);
                    int kPos = k + (z - radius);

                    if (
                        iPos >= 0 && iPos < size && 
                        jPos >= 0 && jPos < size &&
                        kPos >= 0 && kPos < size
                    ){
                        sum += A[(iPos * size + jPos) * size + kPos] * B[(x * K + y) * K + z];
                    }
                }
            }
        }

        C[(i * size + j) * size + k] = sum; 
    }
}

void launchNaiveConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream
) {
    int radius = (K - 1) / 2;
    
    // Define grid and block dimensions for naive kernel
    dim3 blockDim(8, 8, 8);
    dim3 gridDim(
        (size + blockDim.x - 1) / blockDim.x,
        (size + blockDim.y - 1) / blockDim.y,
        (size + blockDim.z - 1) / blockDim.z
    );
    
    naiveConv3DKernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, size, K, radius);
}
