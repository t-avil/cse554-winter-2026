#include "copy_first_column.h"
#include <cuda_runtime.h>

void copy_first_column(float *h_A, float *d_A, int rows, int cols) { // 131µs
    float pinned_first_column[8192];
    float* column_vector = pinned_first_column;
    for (int i = 0; i < rows; i += 4) {
        *column_vector = *h_A;
        column_vector[1] = h_A[cols];
        column_vector[2] = h_A[cols * 2];
        column_vector[3] = h_A[cols * 3];
        h_A += cols << 2;
        column_vector += 4;
    }
    cudaMemcpy(d_A, pinned_first_column, rows * sizeof(float), cudaMemcpyHostToDevice);
}
