CUDA_PATH ?= /usr/local/cuda
CUDNN_PATH ?= /usr/local/cudnn
HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc
CUFLAGS := -O3 -arch=sm_80

# Includes and libraries
INCLUDES := -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include -Iinclude
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib64
LDLIBS := -lcudart -lcudnn

# Executable name
BENCHMARK_EXEC := conv3d_benchmark

# Source directories
SRC_DIR := src
KERNELS_DIR := kernels
INCLUDE_DIR := include

# Source files
KERNEL_SOURCES := $(KERNELS_DIR)/naive_kernel.cu \
                 $(KERNELS_DIR)/coalesced_kernel.cu \
                 $(KERNELS_DIR)/tiled_kernel.cu \
                 $(KERNELS_DIR)/cudnn_kernel.cu

SOURCES := $(SRC_DIR)/benchmark.cu \
          $(SRC_DIR)/utils.cpp \
          $(KERNEL_SOURCES)

# All target: build the benchmark
all: $(BENCHMARK_EXEC)

# Main benchmark target
$(BENCHMARK_EXEC): $(SOURCES)
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Clean rule
clean:
	rm -f *.o $(BENCHMARK_EXEC)

# Run the benchmark
run: $(BENCHMARK_EXEC)
	./$(BENCHMARK_EXEC)

# Generate plots (requires Python with matplotlib)
plot: 
	python3 scripts/plot_results.py

.PHONY: all clean run plot