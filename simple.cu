#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
{                                                                                 \
    const cudaError_t error = call;                                               \
    if (error != cudaSuccess)                                                     \
    {                                                                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                             \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));       \
        exit(1);                                                                  \
    }                                                                             \
}

__global__ void simpleKernel() {
    printf("Simple kernel executed on device.\n");
}

int main(int argc, char const *argv[])
{
    printf("Setting up device...\n");
    int dev = 0;
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev));
    printf("Starting reduction at device %d: %s\n", dev, device_prop.name);
    CUDA_CHECK(cudaSetDevice(dev));

    printf("Running a simple kernel for debugging...\n");
    simpleKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    printf("Simple kernel completed.\n");

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
