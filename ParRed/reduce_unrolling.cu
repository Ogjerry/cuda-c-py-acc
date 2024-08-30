#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
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




// Recursive Implementation of Interleaved Pair Approach
long int cpusum(int *data, int const size)
{
    if (size == 0) return 0; // Handle empty array

    long int *temp = (long int*) malloc( sizeof(long int) * size);
    if (!temp) return 0; // Allocation failed

    // Initialize temp array with input data
    for (int i = 0; i < size; i++) {
        temp[i] = data[i];
    }

    int isize = size;
    while (isize > 1) {
        int const stride = isize / 2;
        int remainder = isize % 2;

        for (int i = 0; i < stride; i++) {
            temp[i] = temp[i] + temp[i + stride];
        }

        // If the array size is odd, add the last element to the first element
        if (remainder != 0) {
            temp[0] += temp[isize - 1];
        }

        isize = stride + remainder;
    }

    long int result = temp[0]; // Final result
    free(temp); // Free the allocated memory
    return result;
}




__global__ void reduce_unrolling2(int *g_idata, int *g_odata, unsigned int n) {
    /*
     * Adding an element from the neighboring data block, so that
     * before reduction, the i_th thread adds two element from 
     * two indexes already.
     * 
     * In short, this kernel process two blocks in a single threadblock.
     * Meaning block 0 process block 0 and block 1, block 2 process 2 and 3.
     * because of these line of code:
     *                  blockIdx.x * blockDim.x * 2;
     *                  <<< grid.x / 2, block >>>;
     *
     *
     * blockDim.x = 8, g_idata[i] += g_idata[i + 8]
     *              ----------------- -------------------------
     * g_idata id:  |0|1|2|3|4|5|6|7| |8 |9 |10|11|12|13|14|15|...
     * threadId:    |0|1|2|3|4|5|6|7| |0 |1 |2 |3 |4 |5 |6 |7 |...
     *              | | | | | | | | | |  |  |  |  |  |  |  |  |
     *              ----------------- -------------------------
     *                   Block 0                Block 1        ...
    */
    
    // set thread id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer into the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2 data blocks, se above graphical explaination
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        //synchronize within thread block
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}




__global__ void reduce_unrolling4(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert gloal data pointer into the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    // synchronize
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    // write results from this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}




__global__ void reduce_unrolling8(int *g_idata, int *g_odata, const int n) {
    /*
     * There are two phases in reduction_unrolling: unrolling and in-place reduction.
     * Do not confuse unrolling with in-place reduction.
     * 
     * Unrolling is used to reduce the number of operations and memory accesses
     * by combining multiple iterations of a loop into a single iteration.
     * 
     * In-place reduction is used to sum up all elements within a block 
     * to a single value, efficiently using shared memory and synchronization.
     * 
     * Unrolling increases instruction-level parallelism by processing multiple 
     * elements per thread. The in-place reduction maximizes the use of available threads 
     * for summation.
    */
    
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

    // convert global data pointer into the local pointer for this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock (In-block Synchronization)
        __syncthreads();
    }
    // write results from this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}




__global__ void reduced_unrolling_intrawarp(int *g_idata, int *g_odata, unsigned int n) {
    /*
    Threads: The smallest unit of execution.
    Warps: A group of 32 threads that execute instructions in lock-step.
    Blocks: A collection of warps. Each block can contain multiple warps.
    Grid: A collection of blocks.
     * Threads within blocks might have been optimized, but warps within blocks 
     * are yet to be optimized. Instead of using for loop for inplace reduction,
     * using iner warp reduction will be more optimized.
    */

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = 8 * blockDim.x * blockIdx.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + 8 * blockDim.x * blockIdx.x;

    if (idx + 7 * blockDim.x < n) {
     int a1 = g_idata[idx];
     int a2 = g_idata[idx + blockDim.x];
     int a3 = g_idata[idx + blockDim.x * 2];
     int a4 = g_idata[idx + blockDim.x * 3];
     int a5 = g_idata[idx + blockDim.x * 4];
     int a6 = g_idata[idx + blockDim.x * 5];
     int a7 = g_idata[idx + blockDim.x * 6];
     int a8 = g_idata[idx + blockDim.x * 7];
     g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}





void check_result(long int *host_ref, long int *gpu_ref, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("CPU %ld \nGPU %ld \nat current %d\n", host_ref[i], gpu_ref[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}




int main(int argc, char const *argv[])
{
    // set up device
    int dev = 0;
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, device_prop.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // simple kernel
    printf("Running a simple kernel for debugging...\n");
    simpleKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    bool b_result = false;

    // initialization
    int size = 1 << 26; // total number of elements to reduce
    printf("    with array size %d     ", size);

    // execution configuration
    int block_size = 512;
    if (argc > 1) {
        block_size = atoi(argv[1]);
    }
    dim3 block (block_size, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int) (rand() & 0xFF);
        // h_idata[i] = i % 256;
    }
    memcpy (tmp, h_idata, bytes);

    cudaEvent_t i_start, i_elaps;
    float elapstime;
    long int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CUDA_CHECK(cudaMalloc((void **) &d_idata, bytes));
    CUDA_CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    //cpu reduction
    long int cpu_sum = cpusum(tmp, size);
    // event create
    CUDA_CHECK(cudaEventCreate(&i_start));
    CUDA_CHECK(cudaEventCreate(&i_elaps));


    // kernel 1: reduced neighbored
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_unrolling2<<<grid.x / 2, block>>> (d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int) / 2, cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu loop unrolled 2     epalsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x/2, block.x);



    // kernel 2: reduced neighbored unrolled 4
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_unrolling4<<<grid.x / 4, block>>> (d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int) / 4, cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu loop unrolled 4     epalsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x/4, block.x);




    // kernel 3: reduced neighbored unrolled 8
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_unrolling8<<<grid.x / 8, block>>> (d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int) / 8, cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu loop unrolled 8     epalsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x/8, block.x);





    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduced_unrolling_intrawarp<<<grid.x / 8, block>>> (d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int) / 8, cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu loop unrolled 8 + unrolled warp     epalsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x/8, block.x);






    // free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);

    // free device memory
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    
    // verify results
    b_result = (gpu_sum == cpu_sum);
    if (!b_result) printf("Test Fail\n");

    // verify size
    bool size_check = (grid.x * block_size == size);
    if (!size_check) printf("Wrong Size!\n");

    check_result(&cpu_sum, &gpu_sum, 1);


    return 0;


}