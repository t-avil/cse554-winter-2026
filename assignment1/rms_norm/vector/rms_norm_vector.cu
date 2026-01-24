#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// 128 is generally safe for most GPUs, but dynamic is still better practice.
#define BLOCKS_NUM 128
#define THREADS_PER_BLOCK 128

__global__ void rms_vector_fused_kernel(float* input, double* temp, float* output, int cols, float epsilon) {
    grid_group grid = this_grid();
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    
    int idx = blockIdx.x * blockDim.x + tid;
    int grid_stride = blockDim.x * gridDim.x;
    
    // Sum of Squares
    float local_sum_sq = 0.0f;

    for (int i = idx; i < cols; i += grid_stride) {
        float val = input[i];
        local_sum_sq += val * val;
    }

    sdata[tid] = local_sum_sq;
    __syncthreads();

    // Block Reduction (Standard Tree)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds this block's total to global memory
    if (tid == 0) {
        atomicAdd(temp, sdata[0]);
    }

    // --- SYNC ENTIRE GRID ---
    // Wait for all blocks to finish adding to 'rms'
    grid.sync(); 

    // Calculate Final RMS
    // Only one thread in the entire grid needs to do the sqrt
    if (grid.thread_rank() == 0) {
        float total_sum_sq = *temp;
        *temp = sqrtf(total_sum_sq / cols + epsilon);
    }

    // --- SYNC ENTIRE GRID ---
    // Wait for the new RMS value to be written
    grid.sync();

    // Normalization ---
    float final_rms = *temp; 
    
    // Reuse the same Grid-Stride logic for applying the norm
    for (int i = idx; i < cols; i += grid_stride) {
        output[i] = input[i] / final_rms;
    }
}

void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    double* d_rms;
    cudaMalloc(&d_rms, sizeof(float));
    cudaMemset(d_rms, 0, sizeof(float));

    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(float);

    void* kernelArgs[] = { &input, &d_rms, &output, &cols, &epsilon };
    
    cudaError_t err = cudaLaunchCooperativeKernel((void*)rms_vector_fused_kernel, BLOCKS_NUM, THREADS_PER_BLOCK, kernelArgs, shared_mem_size, NULL);
    
    if (err != cudaSuccess) {
        printf("Kernel Launch Failed: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    cudaFree(d_rms);
}