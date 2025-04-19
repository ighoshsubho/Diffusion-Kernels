CUDA_PATH ?= /usr/local/cuda
CUDNN_PATH ?= /usr/local/cudnn
HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc
CUFLAGS := -O3 -arch=sm_80

# Includes and libraries
INCLUDES := -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib64
LDLIBS := -lcudart -lcudnn

# Executable names
NAIVE_EXEC := naive_conv3d
COALESCED_EXEC := coalesced_conv3d
TILED_EXEC := tiled_conv3d
CUDNN_EXEC := cudnn_conv3d
BENCHMARK_EXEC := conv3d_benchmark

# Source files
COMMON_SRC := utils.cpp
NAIVE_SRC := naive_kernel.cu $(COMMON_SRC)
COALESCED_SRC := coalesced_kernel.cu $(COMMON_SRC)
TILED_SRC := tiled_kernel.cu $(COMMON_SRC)
CUDNN_SRC := cudnn_kernel.cu $(COMMON_SRC)
BENCHMARK_SRC := benchmark.cu $(COMMON_SRC)

# All target: build all executables
all: $(NAIVE_EXEC) $(COALESCED_EXEC) $(TILED_EXEC) $(CUDNN_EXEC) $(BENCHMARK_EXEC)

# Individual targets
$(NAIVE_EXEC): naive_main.cu $(NAIVE_SRC)
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(COALESCED_EXEC): coalesced_main.cu $(COALESCED_SRC)
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(TILED_EXEC): tiled_main.cu $(TILED_SRC)
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(CUDNN_EXEC): cudnn_main.cu $(CUDNN_SRC)
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BENCHMARK_EXEC): $(BENCHMARK_SRC) naive_kernel.cu coalesced_kernel.cu tiled_kernel.cu cudnn_kernel.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Object files for separate compilation (if needed)
%.o: %.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(HOST_COMPILER) -c $< -o $@

# Clean rule
clean:
	rm -f *.o $(NAIVE_EXEC) $(COALESCED_EXEC) $(TILED_EXEC) $(CUDNN_EXEC) $(BENCHMARK_EXEC)

# Run the full benchmark
run: $(BENCHMARK_EXEC)
	./$(BENCHMARK_EXEC)

# Generate plots (requires Python with matplotlib)
plot: run
	python3 scripts/plot_results.py

.PHONY: all clean run plot
