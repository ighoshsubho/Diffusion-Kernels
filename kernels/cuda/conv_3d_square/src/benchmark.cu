#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "../include/kernels.h"
#include "../include/utils.h"

// Kernel type enumeration
enum KernelType {
    NAIVE,
    COALESCED,
    TILED,
    CUDNN,
    NUM_KERNELS
};

// Kernel names
const std::string kernel_names[NUM_KERNELS] = {
    "Naive Kernel",
    "Coalesced Kernel",
    "Tiled Kernel",
    "cuDNN"
};

// Kernel function pointers
typedef void (*ConvKernelFunc)(const float*, const float*, float*, size_t, size_t, cudaStream_t);

// Array of kernel functions
const ConvKernelFunc kernel_funcs[NUM_KERNELS] = {
    launchNaiveConv3D,
    launchCoalescedConv3D,
    launchTiledConv3D,
    launchCudnnConv3D
};

// Function to measure kernel execution time
float benchmark_kernel(
    KernelType kernel_type,
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int D,
    int K,
    int runs
) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    ConvKernelFunc kernel_func = kernel_funcs[kernel_type];
    
    // Warmup run
    kernel_func(d_input, d_kernel, d_output, D, K, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark runs
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int r = 0; r < runs; r++) {
        kernel_func(d_input, d_kernel, d_output, D, K, 0);
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float total_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_time_ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return total_time_ms / runs;
}

// Main benchmark function
int main(int argc, char* argv[]) {
    // Set the random seed for reproducibility
    srand(42);
    
    // Print device information
    print_device_info();
    
    // Define test cases
    const int num_test_cases = 4;
    int sizes[num_test_cases] = {64, 96, 256, 512};
    int kernels[num_test_cases] = {3, 11, 7, 9};
    const int runs = 10;  // Number of runs for averaging
    
    // For storing results
    std::vector<BenchmarkResult> all_results;
    
    // Header for results table
    printf("=== 3D Convolution Benchmark Results ===\n");
    printf("%-20s | %-7s | %-15s | %-16s | %-15s | %-18s\n", 
           "Kernel Type", "Time", "GFLOPS", "% Peak GFLOPS", "Bandwidth", "% Peak Bandwidth");
    printf("-------------------------------------------------------------------------------------------------------\n");
    
    for (int t = 0; t < num_test_cases; t++) {
        int D = sizes[t];
        int K = kernels[t];
        
        printf("\nTest Case: (D=H=W=%d, K=%d)\n", D, K);
        printf("-------------------------------------------------------------------------------------------------------\n");
        
        // Allocate host memory
        size_t input_size = D * D * D;
        size_t kernel_size = K * K * K;
        size_t output_size = D * D * D;
        
        size_t input_bytes = input_size * sizeof(float);
        size_t kernel_bytes = kernel_size * sizeof(float);
        size_t output_bytes = output_size * sizeof(float);
        
        float* h_input = (float*)malloc(input_bytes);
        float* h_kernel = (float*)malloc(kernel_bytes);
        float* h_output_refs[NUM_KERNELS];
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            h_output_refs[k] = (float*)malloc(output_bytes);
        }
        
        // Initialize data
        initialize_data(h_input, input_size);
        initialize_data(h_kernel, kernel_size);
        
        // Allocate device memory
        float *d_input, *d_kernel, *d_output;
        CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
        CHECK_CUDA(cudaMalloc(&d_kernel, kernel_bytes));
        CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
        
        // Copy data to device
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));
        
        // Benchmark each kernel
        for (int k = 0; k < NUM_KERNELS; k++) {
            // Skip kernels that are too large for shared memory in tiled implementation
            if (k == TILED && K > 13) {
                printf("%-20s | Skipped (kernel size %d > 13 exceeds shared memory limits)\n", 
                      kernel_names[k].c_str(), K);
                
                // Add placeholder result
                BenchmarkResult result;
                result.D = D;
                result.K = K;
                result.execution_time_ms = 0;
                result.gflops = 0;
                result.bandwidth_gb_s = 0;
                result.pct_peak_gflops = 0;
                result.pct_peak_bw = 0;
                all_results.push_back(result);
                
                continue;
            }
            
            // Run the kernel and measure performance
            float kernel_time = benchmark_kernel(static_cast<KernelType>(k), d_input, d_kernel, d_output, D, K, runs);
            
            // Copy results back for validation
            CHECK_CUDA(cudaMemcpy(h_output_refs[k], d_output, output_bytes, cudaMemcpyDeviceToHost));
            
            // Calculate metrics
            BenchmarkResult result;
            calculate_metrics(kernel_names[k], D, K, kernel_time, result);
            all_results.push_back(result);
        }
        
        // Validate results (compare against naive implementation)
        printf("\nValidation results:\n");
        for (int k = 1; k < NUM_KERNELS; k++) {
            if (k == TILED && K > 13) continue;  // Skip validation for skipped kernels
            
            float max_diff = compare_results(h_output_refs[0], h_output_refs[k], output_size);
            printf("%s vs %s: Max difference = %e\n", 
                  kernel_names[0].c_str(), kernel_names[k].c_str(), max_diff);
        }
        
        // Free memory
        for (int k = 0; k < NUM_KERNELS; k++) {
            free(h_output_refs[k]);
        }
        
        free(h_input);
        free(h_kernel);
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_kernel));
        CHECK_CUDA(cudaFree(d_output));
    }
    
    // Print summary table
    printf("\n=== Performance Summary ===\n");
    printf("Size | Kernel | %-15s | %-15s | %-15s | %-15s\n",
           "Naive (GFLOPS)", "Coalesced (GFLOPS)", "Tiled (GFLOPS)", "cuDNN (GFLOPS)");
    printf("--------------------------------------------------------------------------------------\n");
    
    for (int t = 0; t < num_test_cases; t++) {
        int idx = t * NUM_KERNELS;
        printf("%4d | %6d | %-15.2f | %-15.2f | %-15.2f | %-15.2f\n",
               all_results[idx].D, all_results[idx].K,
               all_results[idx].gflops,
               all_results[idx+1].gflops,
               all_results[idx+2].gflops,
               all_results[idx+3].gflops);
    }
    
    // Save results for plotting
    save_benchmark_data(all_results, kernel_names, "benchmark_results.csv");
    
    // Reset device to clean up all memory
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
}
