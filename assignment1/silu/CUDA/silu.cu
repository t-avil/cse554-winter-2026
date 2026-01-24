#include <cuda_runtime.h>

__global__ void silu_kernel(float* input, float* output, int n) {
    // SiLU do not need shared memory since we read and write contiguously
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

void silu(float *input, float *output, int n) {
    float *d_input, *d_output;
    size_t size = n * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input data from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(72);
    dim3 numBlocks((n + blockDim.x - 1) / blockDim.x);
    silu_kernel<<<numBlocks, blockDim>>>(d_input, d_output, n);

    // Copy output data from device to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
