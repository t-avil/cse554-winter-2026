#include <cuda_runtime.h>
#include "silu.h"
#include <iostream>

int main() {

    int dim = 8192;
    float *h_input, *h_output;

    size_t size = dim * dim * sizeof(float);
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    
    // Initialize input matrix
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            h_input[i * dim + j] = static_cast<float>(i * dim + j);
        }
    }

    silu(h_input, h_output, dim * dim);

    // verfiy the result
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            float x = h_input[i * dim + j];
            float expected = x / (1.0f + expf(-x));
            if (fabs(h_output[i * dim + j] - expected) > 1e-5) {
                printf("Mismatch at (%d, %d): got %f, expected %f\n", i, j, h_output[i * dim + j], expected);
                return -1;
            }
        }
    }
    std::cout << "SiLU CUDA implementation is correct!" << std::endl;
}