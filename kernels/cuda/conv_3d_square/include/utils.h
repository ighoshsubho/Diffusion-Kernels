#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Error checking macro for CUDA
#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); }}

// Structure to store benchmark results
struct BenchmarkResult {
    int D;                     // Input size
    int K;                     // Kernel size
    float execution_time_ms;   // Execution time in milliseconds
    float gflops;              // Computed GFLOPS
    float bandwidth_gb_s;      // Memory bandwidth in GB/s
    float pct_peak_gflops;     // Percentage of peak GFLOPS
    float pct_peak_bw;         // Percentage of peak memory bandwidth
};

// Function to initialize arrays with random data
void initialize_data(float *data, size_t size);

// Function to calculate theoretical peaks for the current GPU
void get_theoretical_peaks(float& theoretical_gflops, float& theoretical_bandwidth_gb_s);

// Function to calculate performance metrics
void calculate_metrics(
    const std::string& kernel_name,
    int D,
    int K,
    float execution_time_ms,
    BenchmarkResult& result
);

// Function to save results to CSV for plotting
void save_benchmark_data(const std::vector<BenchmarkResult>& results_by_kernel, 
                         const std::vector<std::string>& kernel_names,
                         const std::string& filename);

// Function to compare results between two arrays
float compare_results(const float* ref, const float* test, size_t size);

// Function to print device properties
void print_device_info();

#endif // UTILS_H
