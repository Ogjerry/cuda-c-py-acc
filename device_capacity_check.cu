#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char const *argv[])
{
    int i_dev = 0;
    cudaDeviceProp i_prop;
    cudaGetDeviceProperties(&i_prop, i_dev);

    printf("Device %d: %s\n", i_dev, i_prop.name);
    printf("Number of multiprocessors: %d\n", i_prop.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n", i_prop.totalConstMem/1024.0);
    printf("Total amount of shared memory per block: %4.2f KB \n", i_prop.sharedMemPerBlock/1024.0);
    printf("Total amount of registers available per block: %d\n", i_prop.regsPerBlock);
    printf("Warp size: %d\n", i_prop.warpSize);
    printf("Maximum number of threads per block: %d\n", i_prop.maxThreadsPerBlock);
    printf("Maximum number of therads per multiprocessor: %d\n", i_prop.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", i_prop.maxThreadsPerMultiProcessor / 32);
    return 0;
}
