#include "copy_first_column.h"
#include <iostream>

int main() {
    const int ROWS = 8192, COLUMNS = 65536;

    float* host_data = new float[ROWS * COLUMNS];
    for (int x = 0; x < ROWS; ++x) {
        for (int y = 0; y < COLUMNS; ++y) {
            host_data[x * COLUMNS + y] = max(x, y);
        }
    }

    float* device_data;
    cudaMalloc((void**)&device_data, ROWS * sizeof(float));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    cudaEventSynchronize(start);
    for (int t = 0; t < 1000; ++t) {
        copy_first_column(host_data, device_data, ROWS, COLUMNS);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float duration;
    cudaEventElapsedTime(&duration, start, end);
    std::cout << "Time: " << ((double)duration / (double)1000 * (double)1000) << " µs" << std::endl;

    float host_verify_data[ROWS];
    cudaMemcpy(host_verify_data, device_data, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int x = 0; x < ROWS; ++x) {
        correct &= host_data[x * COLUMNS] == host_verify_data[x];
    }
    std::cout << (correct ? "Correct!" : "Incorrect!") << std::endl;

    cudaFree(device_data);
    delete[] host_data;

    return 0;

}