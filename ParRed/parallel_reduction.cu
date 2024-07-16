/*
 * Threads are passages or tunnels for operations, not for storages like arrays.

 reduce_neighbored does not syncronize between thread blocks, so the partial sum
 produced by each thread block is copied back to the host and summed sequentially there.
*/

#include <stdio.h>
#include <stdbool.h>
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
long int cpuRecursiveReduce(int *data, int const size)
{
    if (size == 0) return 0;
    long int result = 0;

    for (int i = 0; i < size; i++)
    {
        result += data[i];
    }

    return result;
}










// compute the sum of an integer array
__global__ void reduce_neighbored(int *g_idata, int *g_odata, unsigned int n) {
    /*
     *  Inputs: 
     *          'g_idata': input array for neighbor pair computation, also a pointer index
     *          'g_odata': global output array that stores the results of the reduction from each block
     *          'n'      : size of g_idata

     *  Prerequisite:
     *          gridDim.x: number of blocks in a grid.
     *          blockIdx.x: index of current block within the grid.
     *          blockDim.x: number of threads in the block.
     *          threadIdx.x: index of the thread within a block. (starting at 0)
     
     *  Operations:
     *          g_idata: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15]
     *          Block 0:  |__________  Segment 0  __________|       
     *          Block 1:  |__________  Segment 1  __________|  
     *          
     *          gpu index = threadIdx.x + blockIdx.x * blockDim.x;
     *          if the thread ID is a multiple of 2 * stride
     *              addition operation will be performed at the ID thread;
     *              Multiple operations per thread by __syncthreads();
     *          idata[0] holds the final sum

     * Graphical Ops:
     *          |3||1||7||0||4||1||6||3|
     *           |  /  |  /  |  /  |  /
     *            4     7     5     9
     *            |     /     |     /
     *              11          14
     * 
     *                    25

     *  Example:
     *          blockDim.x = 8
     *          idata: [1, 2, 3, 4, 5, 6, 7, 8]
     *          (
     *          blockIdx.x * blockDim.x calculates the starting index
     *          of the current block's segment in the global input array.
     * 
     *          idata is set to represent the portion of the gloabal input
     *          data (g_idata) that corresponds to the current block.
     *          )
     *          Iteration 1: stride = 1
     *              tid % (2 * stride) == 0: (tid = 0, 2, 4, 6)
     *                  idata[0] += idata[0 + 1] = 3
     *                  idata[2] += idata[2 + 1] = 7
     *                  idata[4] += idata[4 + 1] = 11
     *                  idata[6] += idata[6 + 1] = 15
     *          Iteration 2: stride = 2
     *              tid % (2 * stride) == 0: (tid = 0, 4)
     *                  idata[0] += idata[0 + 2] = 10
     *                  idata[4] += idata[4 + 2] = 26
     *          Iteration 3: stride = 4
     *              tid % (2 * stride) == 0: (tid = 0)
     *                  idata[0] += idata[0 + 4] = 36
     *          Assign the sum (idata[0]) to every block if the threadID is 0
     *          (
     *           To avoid race conditions where multiple threads try to write 
     *           onto the g_odata[blockIdx.x] simultaneously. Only threadIdx.x = 0
     *           is written on g_odata.
     * 
     *           Choosing thread ID 0 is a convention that ensured by reduction operation.
     *           Only thread ID 0 performs the write. Remember that a thread is a path allowing
     *           operations to go through. There is only one operation going on here and we
     *           only need one thread for this one operation. After the reduction operation, 
     *           only one thread is needed to write the final result to global memory, 
     *           hence we only use one thread for this operation.
     *          )
     * 
    */
    // set thread id
    int tid = threadIdx.x;

    // convert global (input array) data pointer to the local (GPU index) pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // gpu index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        
        if ((tid % (2 * stride)) == 0 && (tid + stride) < blockDim.x && idx + stride < n) {
            idata[tid] += idata[tid + stride];
        }
        // syncronize within block
        __syncthreads();
        // Debug output to track reductions
        // if (idx == 0 && blockIdx.x == 0) { // Only print for the first block and first thread
        //     printf("Block %d, Stride %d, Intermediate Sum: %d\n", blockIdx.x, stride, idata[0]);
        // }
    }

    // write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}





__global__ void reduce_improve(int *g_idata, int *g_odata, unsigned int n) {
    /*
     * The first version of parallel reduction can only work on even number of 
     * threads.
    */
    // set thread id
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // bound check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}




__global__ void reduce_interleaved(int *g_idata, int *g_odata, unsigned int n) {
    /*
     * 
     * Graphical Ops:
     *          |3||1||7||0|*|4||1||6||3|*
     *          |4||1||6||3|
     *          ------------
     *          |7||2|*|13||3|*
     *          |13||3|
     *          -------
     *          |20|*|5|*
     *          -------
     *          |25|
     * Operations:
     *          reduce to half of the original size each loop.
     *          if (tid < stride) ensures the addition is performed
     *          at the first half of the current thread block.
    */
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}





/*
 * Four errors I got. 
    1. CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    I wrote size instead of bytes

    2. for (int i = 0; i < grid.x / 2 ; i++) {
            gpu_sum += h_odata[i];
        }

    NOTICE grid.x / 2 not grid.x

    3. unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x

    NOTICE blockIdx.x * blockDim.x * 2

    4. int *idata = g_idata + blockIdx.x * blockDim.x * 2

    NOTICE blockIdx.x * blockDim.x * 2

*/




__global__ void reduce_unrolling(int *g_idata, int *g_odata, unsigned int n) {
    /*
     * Adding an element from the neighboring data block, so that
     * before reduction, the i_th thread adds two element from 
     * two indexes already.
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
    long int cpu_sum = cpuRecursiveReduce(tmp, size);
    // event create
    CUDA_CHECK(cudaEventCreate(&i_start));
    CUDA_CHECK(cudaEventCreate(&i_elaps));



    // kernel 1: reduced neighbored
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_neighbored<<<grid, block>>> (d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu Warmup      epalsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x, block.x);



    // kernel 1
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_neighbored<<<grid, block>>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu neighbored      elapsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x, block.x);



    // kernel 2: reduce neighbored with less divergence
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_improve<<< grid, block>>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu neighbored improved      elapsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x, block.x);



    // kernel 3: reduce interleaved
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_interleaved<<<grid, block>>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x ; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu interleaved     elapsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x, block.x);



    // kernel 4: loop unrolling, also called loop length reduction
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(i_start, 0));
    reduce_unrolling<<< grid.x / 2, block>>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaEventRecord(i_elaps, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&elapstime, i_start, i_elaps));
    // the output of this kernel is half because of two block processed per block
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2 ; i++) { // the grid.x here should be devided by 2 because of two block processed per block
        gpu_sum += h_odata[i];
    }
    printf("gpu loop unrolled     elapsed %f ms gpu_sum: %ld <<< grid %d block %d>>>\n", elapstime, gpu_sum, grid.x/2, block.x);


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