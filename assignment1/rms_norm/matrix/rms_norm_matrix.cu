#include <cuda_runtime.h>
#include <stdio.h>

#define ELEMENT_PER_BLOCK 8192
#define THREADS_PER_BLOCK 128
#define ELEMENTS_PER_THREAD (ELEMENT_PER_BLOCK / THREADS_PER_BLOCK)

__global__ void rms_norm_matrix_kernel(float* input, float* output, int num_rows, int num_cols, float epsilon) {
    extern __shared__ float tile[];
    int start_row = num_cols * blockIdx.x;
    int start_col = threadIdx.x;

    // tile store sum of square of ELEMENTS_PER_THREAD elements
    float local_sum = 0.0f;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = start_row + start_col + i * blockDim.x;
        if (idx < (start_row + num_cols)) {
            float val = input[idx];
            local_sum += val * val;
        }
    }
    tile[threadIdx.x] = local_sum;
    __syncthreads();

    // reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            tile[threadIdx.x] += tile[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // tile[0] now contains the sum of squares for the entire row
        tile[0] = sqrtf(tile[0] / num_cols + epsilon);
    }
    __syncthreads();

    // apply rms norm
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = start_row + start_col + i * blockDim.x;
        if (idx < (start_row + num_cols)) {
            output[idx] = input[idx] / tile[0];
        }
    }
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    int num_blocks = rows;
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(float);

    rms_norm_matrix_kernel<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(input, output, rows, cols, epsilon);
}
