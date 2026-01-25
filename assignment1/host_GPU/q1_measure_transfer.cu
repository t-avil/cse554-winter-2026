#include <cuda_runtime.h>
#include <iostream>

int main() {
    bool HOST_TO_DEVICE = true;
    bool PINNED_MEMORY = false;

    int num_iterations = 1000;
    int num_elements = 1 << 20;

    char* data_host = new char[num_elements];
    for (int i = 0; i < num_elements; ++i) {
        data_host[i] = i * 2 + 1;
    }

    char* data_host2;
    cudaMallocHost((void**)&data_host2, num_elements * sizeof(char));
    for (int i = 0; i < num_elements; ++i) {
        data_host2[i] = i * 2 + 1;
    }
    
    char* data_device;
    cudaMalloc((void**)&data_device, num_elements * sizeof(char));
    cudaMemcpy(data_device, data_host2, num_elements * sizeof(char), cudaMemcpyHostToDevice);

    // Recording services
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    double res[21];
    for (int exp = 0; exp <= 20; ++exp) {
        size_t copy_size_bytes = (1 << exp) * sizeof(char);
        cudaEventRecord(start);
        cudaEventSynchronize(start);
        for (int i = 0; i < num_iterations; ++i) {
            
            if (HOST_TO_DEVICE) {
                cudaMemcpyAsync(data_device, PINNED_MEMORY ? data_host2 : data_host, copy_size_bytes, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpyAsync(PINNED_MEMORY ? data_host2 : data_host, data_device, copy_size_bytes, cudaMemcpyDeviceToHost);
            }

        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float duration;
        cudaEventElapsedTime(&duration, start, end);
        std::cout << "Copying 2^" << exp << " = " << copy_size_bytes << " bytes:" << std::endl;
        std::cout << "Time: " << ((double)duration / (double)num_iterations) << " ms" << std::endl;
        std::cout << "Bandwidth: " << ((double)copy_size_bytes / (double)1e9) / ((double)duration / (double)num_iterations / 1000.0) << "GB/s" << std::endl;
        std::cout << std::endl;
        res[exp] = ((double)copy_size_bytes / (double)1e9) / ((double)duration / (double)num_iterations / 1000.0);
    }
    {
        std::cout << "[" << res[0];
        for (int i = 1; i <= 20; ++i) std::cout << ", " << res[i];
        std::cout << "]" << std::endl;
    }

    cudaFree(data_device);
    delete[] data_host;
    cudaFreeHost(data_host2);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
