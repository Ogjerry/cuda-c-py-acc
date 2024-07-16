#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

typedef struct {
    float *data;
    int *dims;
    int num_dims;
    int total_size;
} Tensor;

// Function to calculate the total size of the tensor
int calculate_total_size(int *dims, int num_dims) {
    int size = 1;
    for (int i = 0; i < num_dims; i++) {
        size *= dims[i];
    }
    return size;
}

// Function to create a tensor
Tensor* create_tensor(int *dims, int num_dims) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->dims = (int *)malloc(num_dims * sizeof(int));
    for (int i = 0; i < num_dims; i++) {
        tensor->dims[i] = dims[i];
    }
    tensor->num_dims = num_dims;
    tensor->total_size = calculate_total_size(dims, num_dims);
    cudaMalloc(&tensor->data, tensor->total_size * sizeof(float));
    return tensor;
}

// Kernel to set values in the tensor
__global__ void set_values_kernel(float *data, int total_size, int dim1, int dim2, int dim3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_size) {
        int i = index / (dim2 * dim3);
        int j = (index % (dim2 * dim3)) / dim3;
        int k = index % dim3;
        data[index] = (float)(i * 100 + j * 10 + k);
    }
}

// Kernel to get values from the tensor and print them
__global__ void get_values_kernel(float *data, int total_size, int dim1, int dim2, int dim3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_size) {
        int i = index / (dim2 * dim3);
        int j = (index % (dim2 * dim3)) / dim3;
        int k = index % dim3;
        printf("tensor[%d][%d][%d] = %.2f\n", i, j, k, data[index]);
    }
}

// Function to free the tensor
void free_tensor(Tensor *tensor) {
    cudaFree(tensor->data);
    free(tensor->dims);
    free(tensor);
}

// Main function demonstrating usage of the tensor
int main() {
    int dims[3] = {2, 3, 4};
    Tensor *tensor = create_tensor(dims, 3);

    int blockSize = 256;
    int numBlocks = (tensor->total_size + blockSize - 1) / blockSize;

    // Setting values in the tensor using CUDA kernel
    set_values_kernel<<<numBlocks, blockSize>>>(tensor->data, tensor->total_size, dims[0], dims[1], dims[2]);
    cudaDeviceSynchronize();

    // Getting and printing values from the tensor using CUDA kernel
    get_values_kernel<<<numBlocks, blockSize>>>(tensor->data, tensor->total_size, dims[0], dims[1], dims[2]);
    cudaDeviceSynchronize();

    // Freeing the tensor
    free_tensor(tensor);
    return 0;
}