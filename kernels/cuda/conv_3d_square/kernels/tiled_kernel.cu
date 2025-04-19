#include <cuda_runtime.h>
#include "../include/kernels.h"

// Tiled 3D convolution kernel with dynamically sized shared memory
template <int TILE_SIZE, int MAX_RADIUS>
__global__ void tiledConv3DKernel(
    const float* A,
    const float* B,
    float* C,
    size_t size,
    size_t K,
    int radius
) {
    // Declare shared memory for input tiles using extern shared memory
    extern __shared__ float tileA[];
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = (threadIdx.x / TILE_SIZE) % TILE_SIZE;
    int tz = threadIdx.x / (TILE_SIZE * TILE_SIZE);
    
    int base_i = blockIdx.x * TILE_SIZE;
    int base_j = blockIdx.y * TILE_SIZE;
    int base_k = blockIdx.z * TILE_SIZE;
    
    int i = base_i + tx;
    int j = base_j + ty;
    int k = base_k + tz;
    
    const int tile_width = TILE_SIZE + 2*MAX_RADIUS;
    const int tile_area = tile_width * tile_width;
    const int tile_volume = tile_area * tile_width;
    
    // Collaborative loading of input data into shared memory (including halo regions)
    // Each thread loads multiple elements to fill the shared memory tile
    for (int load_idx = threadIdx.x; load_idx < tile_volume; load_idx += blockDim.x) {
        int local_k = load_idx % tile_width;
        int local_j = (load_idx / tile_width) % tile_width;
        int local_i = load_idx / tile_area;
        
        int global_i = base_i + local_i - radius;
        int global_j = base_j + local_j - radius;
        int global_k = base_k + local_k - radius;
        
        if (global_i >= 0 && global_i < size &&
            global_j >= 0 && global_j < size &&
            global_k >= 0 && global_k < size) {
            tileA[local_i * tile_area + local_j * tile_width + local_k] = 
                A[(global_i * size + global_j) * size + global_k];
        } else {
            tileA[local_i * tile_area + local_j * tile_width + local_k] = 0.0f;
        }
    }
    
    // Ensuring all threads finish loading before computation
    __syncthreads();
    
    // Perform convolution using shared memory
    if (i < size && j < size && k < size) {
        float sum = 0.0f;
        
        for (int x = 0; x < K; x++) {
            for (int y = 0; y < K; y++) {
                // Vector loading of kernel weights when possible
                int z = 0;
                if (K % 4 == 0) {
                    for (; z <= K - 4; z += 4) {
                        float4 kernelBWeights = *reinterpret_cast<const float4*>(&B[(x * K + y) * K + z]);
                        
                        for (int zOffset = 0; zOffset < 4; zOffset++) {
                            // Access shared memory instead of global memory
                            // Convert from global indices to local indices in shared memory
                            int local_i = tx + x;
                            int local_j = ty + y;
                            int local_k = tz + z + zOffset;
                            
                            float weight;
                            if (zOffset == 0) weight = kernelBWeights.x;
                            else if (zOffset == 1) weight = kernelBWeights.y;
                            else if (zOffset == 2) weight = kernelBWeights.z;
                            else weight = kernelBWeights.w;
                            
                            sum += tileA[local_i * tile_area + local_j * tile_width + local_k] * weight;
                        }
                    }
                }
                
                // Handle remaining elements
                for (; z < K; z++) {
                    int local_i = tx + x;
                    int local_j = ty + y;
                    int local_k = tz + z;
                    
                    sum += tileA[local_i * tile_area + local_j * tile_width + local_k] * 
                          B[(x * K + y) * K + z];
                }
            }
        }
        
        // Write output
        C[(i * size + j) * size + k] = sum;
    }
}

void launchTiledConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream
) {
    const int TILE_SIZE = 8;  // Tunable parameter based on your GPU architecture
    const int MAX_RADIUS = 6; // Maximum supported radius (supports kernels up to 13x13x13)
    
    // Calculate radius from kernel size
    int radius = (K - 1) / 2;
    
    // Calculate grid dimensions based on tile size
    dim3 blockDim(TILE_SIZE * TILE_SIZE * TILE_SIZE);  // Total threads per block
    dim3 gridDim(
        (size + TILE_SIZE - 1) / TILE_SIZE,
        (size + TILE_SIZE - 1) / TILE_SIZE,
        (size + TILE_SIZE - 1) / TILE_SIZE
    );
    
    // Calculate shared memory size
    int tile_width = TILE_SIZE + 2*MAX_RADIUS;
    int sharedMemSize = tile_width * tile_width * tile_width * sizeof(float);
    
    // Launch kernel with dynamic shared memory allocation
    tiledConv3DKernel<8, MAX_RADIUS><<<gridDim, blockDim, sharedMemSize, stream>>>(A, B, C, size, K, radius);
}
