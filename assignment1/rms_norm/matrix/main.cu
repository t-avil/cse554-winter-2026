#include<cuda_runtime.h>
#include "rms_norm_matrix.h"
#include <stdio.h>

#define MATRIX_ROWS 8192
#define MATRIX_COLS 8192
#define EPSILON 1e-8f
#define FLOAT_TOLERANCE 1e-4f

int main() {
    float *h_input, *h_output, *h_verify;
    float *d_input, *d_output;

    size_t size = MATRIX_ROWS * MATRIX_COLS * sizeof(float);
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    h_verify = (float*)malloc(size);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Initialize input matrix
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            h_input[i * MATRIX_COLS + j] = static_cast<float>(i * MATRIX_COLS + j + 1);
        }
    }

    // naive RMS normalization for verification
    for (int i = 0; i < MATRIX_ROWS; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < MATRIX_COLS; j++) {
            float val = h_input[i * MATRIX_COLS + j];
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / MATRIX_COLS + 1e-8f);
        for (int j = 0; j < MATRIX_COLS; j++) {
            h_verify[i * MATRIX_COLS + j] = h_input[i * MATRIX_COLS + j] / rms;
        }
    }

    // Copy input matrix to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    rms_norm_matrix(d_input, nullptr, d_output, MATRIX_ROWS, MATRIX_COLS, EPSILON);

    // Copy output matrix back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLS; i++) {
        if (fabs(h_output[i] - h_verify[i]) > FLOAT_TOLERANCE) {
            correct = false;
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_output[i], h_verify[i]);
            break;
        }
    }
    if (correct) {
        printf("RMS normalization successful! All values match within tolerance.\n");
    } else {
        printf("RMS normalization failed! Values do not match.\n");
    }

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}