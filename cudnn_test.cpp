#include <iostream>
#include <cudnn.h>


int main() {
    cudnnHandle_t cudnn;
    cudnnStatus_t status = cudnnCreate(&cudnn);
    if (status == CUDNN_STATUS_SUCCESS) {
        std::cout << "cuDNN is installed and working correctly!" << std::endl;
    } else {
        std::cout << "cuDNN is not installed correctly. Status: " << status << std::endl;
    }
    cudnnDestroy(cudnn);
    return 0;
}

static inline uint32_t next_power_of_2(uint64_t x) {
    x--;
    x|= x >> 1;
    x|= x >> 2;
    x|= x >> 4;
    x|= x >> 8;
    x|= x >> 16;
    x|= x >> 32;
    x++;
    return x;
}