#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Configuration parameters for the kernel
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

// Tiling parameters - tuned for modern GPUs
constexpr int BM = 128;  // Block tile size M dimension
constexpr int BN = 128;  // Block tile size N dimension
constexpr int BK = 16;   // Block tile size K dimension
constexpr int WM = 64;   // Warp tile size M dimension
constexpr int WN = 64;   // Warp tile size N dimension
constexpr int TM = 8;    // Thread tile size M dimension
constexpr int TN = 8;    // Thread tile size N dimension

// Shared memory bank configuration
constexpr int BANKS = 32;
constexpr int BANK_STRIDE = (BK * BM + BANKS - 1) / BANKS;

// ELU activation function for feature mapping
template<typename scalar_t>
__device__ __forceinline__ scalar_t elu_plus_one(scalar_t x) {
    return x > 0 ? (x + 1) : (exp(x));
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Main linear attention kernel with warptiling and ELU feature mapping
template<typename scalar_t>
__global__ void linear_attention_forward_kernel(
    const scalar_t* __restrict__ queries,    // [B, H, N, D]
    const scalar_t* __restrict__ keys,       // [B, H, N, D]
    const scalar_t* __restrict__ values,     // [B, H, N, D]
    scalar_t* __restrict__ output,           // [B, H, N, D]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {
    
    // Shared memory declarations with extra space for values
    extern __shared__ char shared_mem[];
    scalar_t* shared_queries = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* shared_keys = shared_queries + BM * BK;
    scalar_t* shared_values = shared_keys + BK * BN;
    
    // Calculate thread positions
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    
    // Calculate starting positions
    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;
    
    // Register arrays for thread-local computation
    scalar_t thread_kv[TM * TN] = {0};     // Store K*V intermediate results
    scalar_t reg_queries[TM];              // Cache for queries
    scalar_t reg_keys[TN];                 // Cache for keys
    scalar_t reg_values[TN];               // Cache for values
    scalar_t normalizer[TM] = {0};         // Per-row normalizers
    
    // Main loop over sequence length
    for (int k = 0; k < seq_len; k += BK) {
        // Load and transform queries with ELU
        #pragma unroll
        for (int i = threadIdx.x; i < BM * BK; i += BLOCK_SIZE) {
            const int m = block_m + i / BK;
            const int d = k + i % BK;
            if (m < seq_len && d < head_dim) {
                scalar_t q = queries[
                    batch_idx * (num_heads * seq_len * head_dim) +
                    head_idx * (seq_len * head_dim) +
                    m * head_dim + d];
                shared_queries[i] = elu_plus_one(q);
            }
        }
        
        // Load and transform keys with ELU, and load values
        #pragma unroll
        for (int i = threadIdx.x; i < BK * BN; i += BLOCK_SIZE) {
            const int d = k + i / BN;
            const int n = block_n + i % BN;
            if (d < head_dim && n < seq_len) {
                const int base_idx = 
                    batch_idx * (num_heads * seq_len * head_dim) +
                    head_idx * (seq_len * head_dim) +
                    n * head_dim + d;
                shared_keys[i] = elu_plus_one(keys[base_idx]);
                shared_values[i] = values[base_idx];
            }
        }
        __syncthreads();
        
        // Compute warp-level tiles
        #pragma unroll
        for (int wm = 0; wm < WM; wm += TM) {
            #pragma unroll
            for (int wn = 0; wn < WN; wn += TN) {
                // Load queries into registers
                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    const int m = block_m + warp_id * WM + wm + tm;
                    if (m < seq_len) {
                        reg_queries[tm] = shared_queries[(wm + tm) * BK + lane_id];
                    }
                }
                
                // Load keys and values into registers
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    const int n = block_n + (wn + tn);
                    if (n < seq_len) {
                        reg_keys[tn] = shared_keys[lane_id * BN + (wn + tn)];
                        reg_values[tn] = shared_values[lane_id * BN + (wn + tn)];
                    }
                }
                
                // Compute K*V and accumulate normalizers
                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < TN; tn++) {
                        const scalar_t qk = reg_queries[tm] * reg_keys[tn];
                        thread_kv[tm * TN + tn] += qk * reg_values[tn];
                        normalizer[tm] += qk;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write back
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        const int m = block_m + warp_id * WM + tm;
        if (m < seq_len) {
            // Reduce normalizer across warp
            scalar_t row_norm = warp_reduce_sum(normalizer[tm]);
            
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                const int n = block_n + tn;
                if (n < seq_len) {
                    // Write normalized result to output
                    const int out_idx = 
                        batch_idx * (num_heads * seq_len * head_dim) +
                        head_idx * (seq_len * head_dim) +
                        m * head_dim + n;
                    output[out_idx] = thread_kv[tm * TN + tn] / (row_norm + 1e-6f);
                }
            }
        }
    }
}

// Launch helper function
template<typename scalar_t>
void launch_linear_attention_forward(
    const scalar_t* queries,
    const scalar_t* keys,
    const scalar_t* values,
    scalar_t* output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    cudaStream_t stream) {
    
    dim3 grid(
        (seq_len + BM - 1) / BM,
        (seq_len + BN - 1) / BN,
        batch_size * num_heads
    );
    
    const int shared_mem_size = 
        (BM * BK + BK * BN + BN * BM) * sizeof(scalar_t);
    
    linear_attention_forward_kernel<scalar_t>
        <<<grid, BLOCK_SIZE, shared_mem_size, stream>>>(
        queries, keys, values, output,
        batch_size, num_heads, seq_len, head_dim
    );
}