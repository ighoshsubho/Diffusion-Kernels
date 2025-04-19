#include <cuda_runtime.h>
#include <cudnn.h>
#include "../include/kernels.h"
#include "../include/utils.h"

// Helper function to check cuDNN status
#define CHECK_CUDNN(call) { cudnnStatus_t status = call; if (status != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
    exit(1); }}

// cuDNN context structure to avoid repeated setup
struct CudnnContext {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    void* workspace;
    size_t workspaceSize;
    bool initialized;
    
    CudnnContext() : initialized(false), workspace(nullptr), workspaceSize(0) {}
    
    ~CudnnContext() {
        if (initialized) {
            cudnnDestroyTensorDescriptor(inputDesc);
            cudnnDestroyTensorDescriptor(outputDesc);
            cudnnDestroyFilterDescriptor(filterDesc);
            cudnnDestroyConvolutionDescriptor(convDesc);
            cudnnDestroy(handle);
            if (workspace) cudaFree(workspace);
            initialized = false;
        }
    }
};

// Global context for reuse
static CudnnContext g_cudnnContext;

void launchCudnnConv3D(
    const float* A, 
    const float* B, 
    float* C, 
    size_t size, 
    size_t K,
    cudaStream_t stream
) {
    // Initialize cuDNN if not already done
    if (!g_cudnnContext.initialized) {
        CHECK_CUDNN(cudnnCreate(&g_cudnnContext.handle));
        
        // Create descriptors
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&g_cudnnContext.inputDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&g_cudnnContext.outputDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&g_cudnnContext.filterDesc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&g_cudnnContext.convDesc));
        
        g_cudnnContext.initialized = true;
    }
    
    // Set stream
    CHECK_CUDNN(cudnnSetStream(g_cudnnContext.handle, stream));
    
    // For 3D convolution, we simulate it as a 5D tensor operation
    // Batch size = 1, channels = 1
    int radius = (K - 1) / 2;
    
    // Setup tensor descriptors
    int inputDimA[5] = {1, 1, (int)size, (int)size, (int)size};  // n, c, d, h, w
    int inputStrideA[5] = {(int)(size*size*size), (int)(size*size*size), (int)(size*size), (int)size, 1};
    
    int outputDimA[5] = {1, 1, (int)size, (int)size, (int)size};  // n, c, d, h, w
    int outputStrideA[5] = {(int)(size*size*size), (int)(size*size*size), (int)(size*size), (int)size, 1};
    
    int filterDimA[5] = {1, 1, (int)K, (int)K, (int)K};  // out channels, in channels, d, h, w
    
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(g_cudnnContext.inputDesc, CUDNN_DATA_FLOAT, 5, inputDimA, inputStrideA));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(g_cudnnContext.outputDesc, CUDNN_DATA_FLOAT, 5, outputDimA, outputStrideA));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(g_cudnnContext.filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDimA));
    
    // Set convolution descriptor with padding
    int padA[3] = {radius, radius, radius};  // Same padding
    int dilationA[3] = {1, 1, 1};
    int strideA[3] = {1, 1, 1};
    
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(g_cudnnContext.convDesc, 3, padA, strideA, dilationA, 
                                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    
    // Find the best algorithm
    int requestedAlgoCount = 8;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[8];
    
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(g_cudnnContext.handle, 
                                                   g_cudnnContext.inputDesc, 
                                                   g_cudnnContext.filterDesc, 
                                                   g_cudnnContext.convDesc, 
                                                   g_cudnnContext.outputDesc, 
                                                   requestedAlgoCount, 
                                                   &returnedAlgoCount, 
                                                   perfResults));
    
    g_cudnnContext.algo = perfResults[0].algo;
    
    // Get workspace size
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(g_cudnnContext.handle, 
                                                      g_cudnnContext.inputDesc, 
                                                      g_cudnnContext.filterDesc, 
                                                      g_cudnnContext.convDesc, 
                                                      g_cudnnContext.outputDesc, 
                                                      g_cudnnContext.algo, 
                                                      &g_cudnnContext.workspaceSize));
    
    // Allocate workspace if needed or if size changed
    if (g_cudnnContext.workspace == nullptr || g_cudnnContext.workspaceSize > 0) {
        if (g_cudnnContext.workspace) {
            cudaFree(g_cudnnContext.workspace);
        }
        
        cudaMalloc(&g_cudnnContext.workspace, g_cudnnContext.workspaceSize);
    }
    
    // Launch convolution
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CHECK_CUDNN(cudnnConvolutionForward(g_cudnnContext.handle, 
                                       &alpha, 
                                       g_cudnnContext.inputDesc, A, 
                                       g_cudnnContext.filterDesc, B, 
                                       g_cudnnContext.convDesc, 
                                       g_cudnnContext.algo, 
                                       g_cudnnContext.workspace, g_cudnnContext.workspaceSize, 
                                       &beta, 
                                       g_cudnnContext.outputDesc, C));
}
