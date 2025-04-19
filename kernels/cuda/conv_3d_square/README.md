# CUDA 3D Convolution Benchmarks

This repository contains various implementations of 3D convolution kernels for CUDA, along with benchmarking tools to compare their performance.

## Kernels Included

1. **Naive Implementation**: Basic 3D convolution with straightforward thread mapping
2. **Coalesced Implementation**: Uses 1D thread blocks and vectorized (float4) memory access
3. **Tiled Implementation**: Uses shared memory tiling to reduce global memory access
4. **cuDNN Implementation**: Uses NVIDIA's cuDNN library for comparison

## Directory Structure

```
.
├── include/          # Header files
├── kernels/          # CUDA kernel implementations
├── src/              # Main source files
├── scripts/          # Python scripts for visualization
├── Makefile          # Build system
└── README.md         # This file
```

## Requirements

- CUDA Toolkit 11.0 or higher
- cuDNN 8.0 or higher
- A CUDA-capable GPU with compute capability 8.0 or higher (NVIDIA A100, RTX 3090, etc.)
- GCC/G++ compatible with your CUDA version
- Python with matplotlib (for visualization)

## Building

To build all kernels and benchmarks:

```bash
make all
```

To build specific implementations:

```bash
make naive_conv3d     # Build naive implementation
make coalesced_conv3d # Build coalesced implementation
make tiled_conv3d     # Build tiled implementation
make cudnn_conv3d     # Build cuDNN implementation
```

To build the comprehensive benchmark that compares all implementations:

```bash
make conv3d_benchmark
```

## Running Benchmarks

To run the comprehensive benchmark:

```bash
make run
```

This will run all implementations with various test cases and output performance metrics.

### Test Cases

The benchmark includes the following test cases:
- D=H=W=64, K=3
- D=H=W=96, K=11
- D=H=W=256, K=7
- D=H=W=512, K=9

## Visualization

To generate performance comparison plots:

```bash
make plot
```

This will create PNG files with GFLOPS and memory bandwidth comparisons.

## Implementation Details

### Naive Kernel

The naive kernel uses a 3D thread grid mapped directly to output elements. Each thread computes one output element by iterating through the convolution window.

### Coalesced Kernel

The coalesced kernel uses a 1D thread mapping and vectorized float4 loads to improve memory access patterns. This helps coalesce memory operations and reduces thread divergence.

### Tiled Kernel

The tiled kernel uses shared memory to cache input data in tiles. Each thread block cooperatively loads a tile (including halo regions) into shared memory, then computes output elements using the cached data. This reduces global memory accesses.

### cuDNN Implementation

The cuDNN implementation uses NVIDIA's highly optimized deep learning library. For 3D convolution, we use a slice-by-slice approach with 2D convolution functions.

## Performance Analysis

Performance metrics include:
- Execution time (ms)
- Compute throughput (GFLOPS)
- Memory bandwidth utilization (GB/s)
- Percentage of peak theoretical performance

## Contributing

Feel free to contribute by adding new kernel implementations, improving existing ones, or enhancing the benchmark framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
