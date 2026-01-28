#include "copy_first_column.h"
#include <cuda_runtime.h>

void copy_first_column(float *h_A, float *d_A, int rows, int cols) { // 131µs
    float pinned_first_column[8192];
    for (int i = 0; i < rows; i += 1) {
        pinned_first_column[i] = h_A[cols * i];
    }
    cudaMemcpy(d_A, pinned_first_column, rows * sizeof(float), cudaMemcpyHostToDevice);
}
