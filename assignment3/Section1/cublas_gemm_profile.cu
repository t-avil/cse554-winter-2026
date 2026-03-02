#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

float profile_gemm(cublasHandle_t handle, int M, int N, int K,
                   int warm_up_count, int profile_count, size_t L2_size, int* clear_l2_buffer) {
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    // Allocate host memory
    __half* host_A = new __half[size_A];
    __half* host_B = new __half[size_B];

    // Initialize host arrays
    for (size_t i = 0; i < size_A; i++) host_A[i] = __float2half((float)(rand() % 10) / 10.0f);
    for (size_t i = 0; i < size_B; i++) host_B[i] = __float2half((float)(rand() % 10) / 10.0f);

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(__half));
    cudaMalloc((void**)&d_B, size_B * sizeof(__half));
    cudaMalloc((void**)&d_C, size_C * sizeof(__half));

    // Copy data from host to device
    cudaMemcpy(d_A, host_A, size_A * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, size_B * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C * sizeof(__half));

    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    // Warm up
    for (int i = 0; i < warm_up_count; ++i) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta,
                     d_C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after warmup: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] host_A; delete[] host_B;
        return -1.0f;
    }

    // Profile
    float total_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < profile_count; ++i) {
        cudaMemset(clear_l2_buffer, 0, L2_size); // Clear L2 cache
        cudaEventRecord(start);
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta,
                     d_C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float average_ms = total_ms / profile_count;

    // Compute TFLOPS: 2*M*N*K FLOPs for GEMM
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (average_ms / 1000.0)) / 1e12;

    std::cout << "M=" << M << ", N=" << N << ", K=" << K
              << " | Avg time: " << average_ms << " ms"
              << " | TFLOPS: " << tflops << std::endl;

    // Free CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] host_A;
    delete[] host_B;

    return (float)tflops;
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    int warm_up_count = 100;
    int profile_count = 100;
    size_t L2_size = 50 * 1024 * 1024;

    int* clear_l2_buffer;
    cudaMalloc(&clear_l2_buffer, L2_size);

    // M values: 128 to 2048 step 128 -> 16 values
    const int num_M = 16;
    int M_values[num_M];
    for (int i = 0; i < num_M; i++) {
        M_values[i] = (i + 1) * 128;
    }

    // (N, K) shapes
    const int num_shapes = 5;
    int shape_N[num_shapes] = {512, 4096, 14336, 4096, 1024};
    int shape_K[num_shapes] = {512, 4096,  4096, 1024, 4096};

    // Output CSV
    std::ofstream csv("gemm_perf.csv");
    csv << "batch_size,N,K,library,tflops" << std::endl;

    for (int s = 0; s < num_shapes; s++) {
        for (int i = 0; i < num_M; i++) {
            float tflops = profile_gemm(handle, M_values[i], shape_N[s], shape_K[s],
                                        warm_up_count, profile_count,
                                        L2_size, clear_l2_buffer);
            csv << M_values[i] << "," << shape_N[s] << "," << shape_K[s]
                << ",cublas," << tflops << std::endl;
        }
    }

    csv.close();

    // Free the L2 buffer
    cudaFree(clear_l2_buffer);
    cublasDestroy(handle);

    std::cout << "cuBLAS results written to gemm_perf.csv" << std::endl;
    return 0;
}