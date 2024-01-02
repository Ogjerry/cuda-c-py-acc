/*
 * execution program for the whole project. Have no idea what project to carry on for now.
 * Will figure out lately. This document is a playground for cuda C/C++ programming.
 * Possible future extensions include improving computing speed of large statistical learning
 * models or computing matrix operations, which is the related field of my major study.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        printf("Thread %d: BlockIdx %d, ThreadIdx %d\n", i, blockIdx.x, threadIdx.x);
        printf("Thread %d: %f + %f = %f\n", i, a[i], b[i], a[i] + b[i]);
        c[i] = a[i] + b[i];
    }
}


int main() {
    int n = 256; // Size of the vectors
    int size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors on the host
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy input vectors from host to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    
    // Launch the vectorAdd kernel on the GPU
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Print some values from device arrays
    for (int i = 0; i < n; ++i) {
        float a_val, b_val, c_val;
        cudaMemcpy(&a_val, d_a + i, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b_val, d_b + i, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c_val, d_c + i, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Device Array Values %d: %f, %f, %f\n", i, a_val, b_val, c_val);
    }

    // Copy the result from GPU to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // Print some values from device arrays
    for (int i = 0; i < n; ++i) {
        float c_val;
        cudaMemcpy(&c_val, d_c + i, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("Device Array Values %d: %f\n", i, c_val);
        
        // Print intermediate values
        printf("Intermediate Result %d: %f\n", i, h_c[i]);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
