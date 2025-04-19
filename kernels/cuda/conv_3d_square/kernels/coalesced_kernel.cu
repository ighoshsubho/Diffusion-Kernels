#include <cuda_runtime.h>
#include "../include/kernels.h"

__global__ void coalescedConv3DKernel(
    const float* A,
    const float* B,
    float* C,
    size_t size,
    size_t K,
    int radius
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size * size * size){
        int k = idx % size;
        int j = (idx / size) % size;
        int i = idx / (size * size);

        float sum = 0.0f;

        for (int x = 0; x < K; x++){
            for (int y = 0; y < K; y++){
                int z = 0;
                if (K%4 == 0){
                    for (; z <= K - 4; z+=4){
                        float4 kernelBWeights = *reinterpret_cast<const float4*>(&B[(x * K + y) * K + z]);

                        for (int zOffset = 0; zOffset < 4; zOffset++) {
                            int iPos = i + (x - radius);
                            int jPos = j + (y - radius);
                            int kPos = k + (z + zOffset - radius);

                            if (iPos >= 0 && iPos < size && 
                                jPos >= 0 && jPos < size &&
                                kPos >= 0 && kPos < size) {
                                
                                float weight;
                                if (zOffset == 0) weight = kernelBWeights.x;
                                else if (zOffset == 1) weight = kernelBWeights.y;
                                else if (zOffset == 2) weight = kernelBWeights.z;
                                else weight = kernelBWeights.w;
                                
                                sum += A[(iPos * size + jPos) * size + kPos] * weight;
                            }
                        }
                    }
                }

                for (; z < K; z++){
                    int iPos = i + (x - radius);
                    int jPos = j + (y - radius);
                    int kPos = k + (z - radius);

                    if (iPos >= 0 && iPos < size &&
                        jPos >= 0 && jPos < size &&
                        kPos >= 0 && kPos < size) {
                        sum += A[(iPos * size + jPos) * size + kPos] * B[(x * K + y) * K + z];
                    }
                }
            }
        } 
        
        C[idx] = sum;
    }
}

void launchCoalescedConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream
) {
    int radius = (K - 1) / 2;
    
    // Use 1D grid and block for coalesced memory access
    dim3 blockDim(256);
    dim3 gridDim((size * size * size + blockDim.x - 1) / blockDim.x);
    
    coalescedConv3DKernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, size, K, radius);
}
