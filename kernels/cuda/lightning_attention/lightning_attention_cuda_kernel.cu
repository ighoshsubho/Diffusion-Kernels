#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCUDA Error: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0); } }

struct AttentionParams {
    int batch_size;      // Number of sequences to process in parallel
    int seq_length;      // Length of input sequence
    int num_heads;       // Number of attention heads
    int head_dim;        // Dimension of each attention head
    float decay_rate;    // Decay rate λ for attention weights
    int block_size;      // Size of processing blocks (B in the paper)
};

// Kernel to compute the diagonal decay matrix Λ
__global__ void computeLambdaKernel(float* lambda, int block_size, float decay_rate) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < block_size) {
        // Λ = diag{λ, λ^2, ..., λ^B}
        // This would get used in the inter-block attention computation for every subset of seq length
        // To reduce the effect of further away tokens, we decay the weights by λ and its powers
        lambda[idx * block_size + idx] = pow(decay_rate, idx + 1);
    }
}

// Kernel to compute the exponential decay mask matrix M
__global__ void computeMaskKernel(float* mask, int block_size, float decay_rate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < block_size && j < block_size) {
        // M_{ij} = λ^(i-j) if i >= j, else 0
        // This would get used in the intra-block attention computation for Masked Attention for diagonal elements
        mask[i * block_size + j] = (i >= j) ? pow(decay_rate, i - j) : 0.0f;
    }
}

// Kernel for computing intra-block attention
__global__ void intraBlockAttentionKernel(
    const float* Q,
    const float* K,
    const float* V,
    const float* mask,
    float* output,
    int block_size,
    int head_dim,
    float decay_rate,
    int seq_length
) {
    extern __shared__ float shared_mem[];
    
    float* shared_Q = shared_mem;
    float* shared_K = shared_Q + block_size * head_dim;
    float* shared_V = shared_K + block_size * head_dim;
    float* QK = shared_V + block_size * head_dim;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (tid < block_size) {
        for (int d = 0; d < head_dim; d++) {
            shared_Q[tid * head_dim + d] = 0.0f;
            shared_K[tid * head_dim + d] = 0.0f;
            shared_V[tid * head_dim + d] = 0.0f;
        }
        for (int j = 0; j < block_size; j++) {
            QK[tid * block_size + j] = 0.0f;
        }
    }
    __syncthreads();
    
    if (tid < block_size) {
        const int base_idx = bid * block_size * head_dim + tid * head_dim;
        for (int d = 0; d < head_dim; d++) {
            if (base_idx + d < seq_length * head_dim) {
                shared_Q[tid * head_dim + d] = Q[base_idx + d];
                shared_K[tid * head_dim + d] = K[base_idx + d];
                shared_V[tid * head_dim + d] = V[base_idx + d];
            }
        }
    }
    __syncthreads();
    
    // Compute QK^T with bounds checking
    if (tid < block_size) {
        for (int j = 0; j < block_size; j++) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += shared_Q[tid * head_dim + d] * shared_K[j * head_dim + d];
            }
            QK[tid * block_size + j] = sum;
        }
    }
    __syncthreads();
    
    // Compute final output with bounds checking
    if (tid < block_size) {
        const int out_base_idx = bid * block_size * head_dim + tid * head_dim;
        for (int d = 0; d < head_dim; d++) {
            if (out_base_idx + d < seq_length * head_dim) {
                float sum = 0.0f;
                for (int j = 0; j < block_size; j++) {
                    sum += QK[tid * block_size + j] * mask[tid * block_size + j] * shared_V[j * head_dim + d];
                }
                output[out_base_idx + d] = sum;
            }
        }
    }
}

// New kernel for computing inter-block attention
__global__ void interBlockAttentionKernel(
    const float* Q,
    const float* KV,
    const float* lambda,
    float* output,
    int block_size,
    int head_dim,
    int seq_length
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (tid < block_size) {
        const int global_idx = bid * block_size + tid;
        
        if (global_idx < seq_length) {
            // Apply Λ to Q first
            for (int d = 0; d < head_dim; d++) {
                float q_scaled = Q[global_idx * head_dim + d] * lambda[tid * block_size + tid];
                float sum = 0.0f;
                
                // Multiply with KV
                for (int k = 0; k < head_dim; k++) {
                    sum += q_scaled * KV[d * head_dim + k];
                }
                
                output[global_idx * head_dim + d] = sum;
            }
        }
    }
}

__global__ void addOutputsKernel(
    float* output,
    const float* intra_output,
    const float* inter_output,
    int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = intra_output[idx] + inter_output[idx];
    }
}

// New kernel for updating KV cache
__global__ void updateKVCacheKernel(
    float* KV,
    const float* K,
    const float* V,
    const float* lambda,
    int block_size,
    int head_dim,
    float decay_rate
) {
    const int d1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int d2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (d1 < head_dim && d2 < head_dim) {
        // First term: λ^B * KV
        float decay_factor = pow(decay_rate, block_size);
        float result = decay_factor * KV[d1 * head_dim + d2];
        
        // Second term: (λ^B Λ^-1 Ki)^T Vi
        for (int b = 0; b < block_size; b++) {
            // λ^B Λ^-1 = λ^B / λ^(b+1)
            float scale = decay_factor / pow(decay_rate, b + 1);
            float k_scaled = scale * K[b * head_dim + d1];
            result += k_scaled * V[b * head_dim + d2];
        }
        
        KV[d1 * head_dim + d2] = result;
    }
}

class LightningAttention {
private:
    AttentionParams params;
    float *d_Q, *d_K, *d_V, *d_mask, *d_lambda, *d_output, *d_KV;
    float *d_intra_output, *d_inter_output;
    float gpu_alloc_time;
    float h2d_transfer_time;
    float kernel_time;
    float d2h_transfer_time;

    void initializeMatrices() {
        // Initialize mask M
        dim3 block_mask(16, 16);
        dim3 grid_mask(
            (params.block_size + block_mask.x - 1) / block_mask.x,
            (params.block_size + block_mask.y - 1) / block_mask.y
        );
        computeMaskKernel<<<grid_mask, block_mask>>>(
            d_mask, params.block_size, params.decay_rate);

        // Initialize diagonal decay matrix Λ
        dim3 block_lambda(256);
        dim3 grid_lambda((params.block_size + block_lambda.x - 1) / block_lambda.x);
        computeLambdaKernel<<<grid_lambda, block_lambda>>>(
            d_lambda, params.block_size, params.decay_rate);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

public:
    LightningAttention(const AttentionParams& p) : params(p) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        const size_t qkv_size = params.batch_size * params.seq_length * 
                               params.head_dim * sizeof(float);
        const size_t mask_size = params.block_size * params.block_size * sizeof(float);
        const size_t lambda_size = params.block_size * params.block_size * sizeof(float);
        
        CHECK_CUDA(cudaMalloc(&d_Q, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_K, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_V, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_mask, mask_size));
        CHECK_CUDA(cudaMalloc(&d_lambda, lambda_size));
        CHECK_CUDA(cudaMalloc(&d_output, qkv_size));
        CHECK_CUDA(cudaMemset(d_output, 0, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_intra_output, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_inter_output, qkv_size));
        CHECK_CUDA(cudaMalloc(&d_KV, params.head_dim * params.head_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_KV, 0, params.head_dim * params.head_dim * sizeof(float)));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&gpu_alloc_time, start, stop));

        initializeMatrices();

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void forward(const float* Q, const float* K, const float* V, float* output) {
        cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
        
        CHECK_CUDA(cudaEventCreate(&start_h2d));
        CHECK_CUDA(cudaEventCreate(&stop_h2d));
        CHECK_CUDA(cudaEventCreate(&start_kernel));
        CHECK_CUDA(cudaEventCreate(&stop_kernel));
        CHECK_CUDA(cudaEventCreate(&start_d2h));
        CHECK_CUDA(cudaEventCreate(&stop_d2h));

        const size_t qkv_size = params.batch_size * params.seq_length * 
                               params.head_dim * sizeof(float);

        CHECK_CUDA(cudaEventRecord(start_h2d));
        CHECK_CUDA(cudaMemcpy(d_Q, Q, qkv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, K, qkv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, V, qkv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop_h2d));

        const int num_blocks = params.seq_length / params.block_size;
        
        CHECK_CUDA(cudaEventRecord(start_kernel));

        const size_t shared_mem_size = (3 * params.block_size * params.head_dim + 
                                  params.block_size * params.block_size) * sizeof(float);
        
        // Process each block in T (n/B) steps
        for (int i = 0; i < num_blocks; i++) {
            CHECK_CUDA(cudaMemset(d_intra_output + i * params.block_size * params.head_dim, 0,
                             params.block_size * params.head_dim * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_inter_output + i * params.block_size * params.head_dim, 0,
                             params.block_size * params.head_dim * sizeof(float)));
        

            // 1. Compute intra-block attention
            dim3 block_intra(params.block_size);
            dim3 grid_intra(1);
            
            intraBlockAttentionKernel<<<grid_intra, block_intra, shared_mem_size>>>(
                d_Q + i * params.block_size * params.head_dim,
                d_K + i * params.block_size * params.head_dim,
                d_V + i * params.block_size * params.head_dim,
                d_mask,
                d_intra_output + i * params.block_size * params.head_dim,
                params.block_size,
                params.head_dim,
                params.decay_rate,
                params.seq_length
            );
            CHECK_CUDA(cudaGetLastError());

            // 2. Update KV cache
            dim3 block_kv(32, 32);
            dim3 grid_kv((params.head_dim + block_kv.x - 1) / block_kv.x,
                        (params.head_dim + block_kv.y - 1) / block_kv.y);
            
            updateKVCacheKernel<<<grid_kv, block_kv>>>(
                d_KV,
                d_K + i * params.block_size * params.head_dim,
                d_V + i * params.block_size * params.head_dim,
                d_lambda,
                params.block_size,
                params.head_dim,
                params.decay_rate
            );

            // 3. Compute inter-block attention
            dim3 block_inter(params.block_size);
            dim3 grid_inter(1);
            
            interBlockAttentionKernel<<<grid_inter, block_inter>>>(
                d_Q + i * params.block_size * params.head_dim,
                d_KV,
                d_lambda,
                d_inter_output + i * params.block_size * params.head_dim,
                params.block_size,
                params.head_dim,
                params.seq_length
            );

            // 4. Combine results: O_i = O_intra + O_inter
            const int block_size = 256;
            const int num_elements = params.block_size * params.head_dim;
            const int grid_size = (num_elements + block_size - 1) / block_size;
            
            addOutputsKernel<<<grid_size, block_size>>>(
                d_output + i * params.block_size * params.head_dim,
                d_intra_output + i * params.block_size * params.head_dim,
                d_inter_output + i * params.block_size * params.head_dim,
                num_elements
            );
        }
        
        CHECK_CUDA(cudaEventRecord(stop_kernel));

        // Transfer results back to CPU
        CHECK_CUDA(cudaEventRecord(start_d2h));
        CHECK_CUDA(cudaMemcpy(output, d_output, qkv_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop_d2h));

        CHECK_CUDA(cudaEventSynchronize(stop_d2h));
        
        CHECK_CUDA(cudaEventElapsedTime(&h2d_transfer_time, start_h2d, stop_h2d));
        CHECK_CUDA(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
        CHECK_CUDA(cudaEventElapsedTime(&d2h_transfer_time, start_d2h, stop_d2h));

        // Cleanup timing events
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
    }

    float getGPUAllocationTime() const { return gpu_alloc_time; }
    float getH2DTransferTime() const { return h2d_transfer_time; }
    float getKernelTime() const { return kernel_time; }
    float getD2HTransferTime() const { return d2h_transfer_time; }
    
    ~LightningAttention() {
        if (d_Q) cudaFree(d_Q);
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_mask) cudaFree(d_mask);
        if (d_lambda) cudaFree(d_lambda);
        if (d_output) cudaFree(d_output);
        if (d_intra_output) cudaFree(d_intra_output);
        if (d_inter_output) cudaFree(d_inter_output);
        if (d_KV) cudaFree(d_KV);
    }
};

int main() {
    AttentionParams params {
        .batch_size = 1,
        .seq_length = 1024,
        .num_heads = 8,
        .head_dim = 64,
        .decay_rate = 0.9f,
        .block_size = 256
    };
    
    const size_t qkv_size = params.batch_size * params.seq_length * params.head_dim;
    float *h_Q = new float[qkv_size];
    float *h_K = new float[qkv_size];
    float *h_V = new float[qkv_size];
    float *h_output = new float[qkv_size];
    
    for (int i = 0; i < qkv_size; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    try {
        LightningAttention attention(params);
        attention.forward(h_Q, h_K, h_V, h_output);

        // Print timing information
        std::cout << "Performance Metrics:\n";
        std::cout << ">> GPU allocation time: " << attention.getGPUAllocationTime() << " ms\n";
        std::cout << ">> Host to device transfer time: " << attention.getH2DTransferTime() << " ms\n";
        std::cout << ">> Kernel execution time: " << attention.getKernelTime() << " ms\n";
        std::cout << ">> Device to host transfer time: " << attention.getD2HTransferTime() << " ms\n\n";
        
        std::cout << "Sample Outputs:\n";
        std::cout << "First few Query values:\n";
        for (int i = 0; i < 5; i++) {
            std::cout << h_Q[i] << " ";
        }
        std::cout << "\n\nFirst few Key values:\n";
        for (int i = 0; i < 5; i++) {
            std::cout << h_K[i] << " ";
        }
        std::cout << "\n\nFirst few Value values:\n";
        for (int i = 0; i < 5; i++) {
            std::cout << h_V[i] << " ";
        }
        std::cout << "\n\nFirst few Output values:\n";
        for (int i = 0; i < 5; i++) {
            std::cout << h_output[i] << " ";
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;
    
    return 0;
}