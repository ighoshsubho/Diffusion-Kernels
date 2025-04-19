#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <fstream>
#include "../include/utils.h"

void initialize_data(float *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }
}

void get_theoretical_peaks(float& theoretical_gflops, float& theoretical_bandwidth_gb_s) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    // For A100
    if (prop.major == 8 && prop.minor == 0) {
        theoretical_gflops = 6912 * 1.41 * 2; // CUDA cores * clock * 2 ops
        theoretical_bandwidth_gb_s = 1555.0f;  // HBM2e bandwidth
    } else {
        // Generic calculation for other devices
        theoretical_gflops = prop.multiProcessorCount * prop.clockRate * 2.0 / 1e6;
        theoretical_bandwidth_gb_s = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    }
}

void calculate_metrics(
    const std::string& kernel_name,
    int D,
    int K,
    float execution_time_ms,
    BenchmarkResult& result
) {
    float theoretical_gflops, theoretical_bandwidth_gb_s;
    get_theoretical_peaks(theoretical_gflops, theoretical_bandwidth_gb_s);
    
    // Calculate operations and memory accesses
    long long total_ops = (long long)D * D * D * (2 * K * K * K - 1);
    long long total_memory = ((long long)D * D * D + (long long)K * K * K + (long long)D * D * D) * sizeof(float);
    
    // Store results
    result.D = D;
    result.K = K;
    result.execution_time_ms = execution_time_ms;
    result.gflops = (total_ops / 1e9) / (execution_time_ms / 1000.0f);
    result.bandwidth_gb_s = (total_memory / 1e9) / (execution_time_ms / 1000.0f);
    result.pct_peak_gflops = (result.gflops / theoretical_gflops) * 100.0f;
    result.pct_peak_bw = (result.bandwidth_gb_s / theoretical_bandwidth_gb_s) * 100.0f;
    
    // Print metrics
    printf("%-20s | %7.2f ms | %7.2f GFLOPS | %6.2f%% of peak | %7.2f GB/s | %6.2f%% of peak BW\n",
           kernel_name.c_str(),
           execution_time_ms,
           result.gflops,
           result.pct_peak_gflops,
           result.bandwidth_gb_s,
           result.pct_peak_bw);
}

void save_benchmark_data(const std::vector<BenchmarkResult>& results_by_kernel, 
                         const std::vector<std::string>& kernel_names,
                         const std::string& filename) {
    std::ofstream outfile(filename);
    
    // Write CSV header
    outfile << "D,K,MatrixSize,";
    for (const auto& name : kernel_names) {
        outfile << name << "_Time,";
        outfile << name << "_GFLOPS,";
        outfile << name << "_BW_Percent,";
    }
    outfile << std::endl;
    
    // Write each test case row
    int num_test_cases = results_by_kernel.size() / kernel_names.size();
    
    for (int i = 0; i < num_test_cases; i++) {
        int D = results_by_kernel[i * kernel_names.size()].D;
        int K = results_by_kernel[i * kernel_names.size()].K;
        long long matrixSize = (long long)D * D * D;
        
        outfile << D << "," << K << "," << matrixSize << ",";
        
        for (size_t j = 0; j < kernel_names.size(); j++) {
            const auto& result = results_by_kernel[i * kernel_names.size() + j];
            outfile << result.execution_time_ms << ",";
            outfile << result.gflops << ",";
            outfile << result.pct_peak_bw << ",";
        }
        outfile << std::endl;
    }
    
    outfile.close();
    printf("Benchmark data saved to %s\n", filename.c_str());
}

float compare_results(const float* ref, const float* test, size_t size) {
    float max_diff = 0.0f;
    size_t diff_count = 0;
    
    for (size_t i = 0; i < size; i++) {
        float diff = fabs(ref[i] - test[i]);
        max_diff = fmax(max_diff, diff);
        if (diff > 1e-5) {
            diff_count++;
        }
    }
    
    printf("Result validation: Max difference = %e, Different elements: %zu / %zu (%.2f%%)\n", 
           max_diff, diff_count, size, (100.0f * diff_count) / size);
    
    return max_diff;
}

void print_device_info() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 64);  // Approximation
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Memory Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("\n");
}
