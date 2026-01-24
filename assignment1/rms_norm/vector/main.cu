#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>

#define VECTOR_SIZE (1024*1024)
#define EPSILON 1e-8f
#define FLOAT_TOLERANCE 1e-4f

int main() {
    float *h_input, *h_output, *h_verify;
    float *d_input, *d_output;
    size_t size = VECTOR_SIZE * sizeof(float);

    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    h_verify = (float*)malloc(size);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Initialize input vector
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    rms_norm_vector(d_input, nullptr, d_output, VECTOR_SIZE, EPSILON);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verification on host
    double sum_squares = 0.0f;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        sum_squares += h_input[i] * h_input[i];
    }
    double rms = sqrtf(sum_squares / VECTOR_SIZE + EPSILON);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_verify[i] = h_input[i] / rms;
    }

    // Check results
    bool correct = true;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (fabs(h_output[i] - h_verify[i]) > FLOAT_TOLERANCE) {
            correct = false;
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_output[i], h_verify[i]);
            break;
        }
    }

    if (correct) {
        printf("RMS normalization successful!\n");
    } else {
        printf("RMS normalization failed!\n");
    }

    free(h_input);
    free(h_output);
    free(h_verify);
    cudaFree(d_input);
    cudaFree(d_output);
}