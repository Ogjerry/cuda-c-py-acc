/*
 * Demonstrating the warp thread control/divergence
 * where GPU execution is visualized and how it is not
 * fully utilized.
 * Then, by assigning threads within a warp by id, we
 * can maximize the GPU capacity.
 * Branch Efficiency = (#branches - #divergent branches) / #branches


*/

#include <stdio.h>

// 50% thread utilization
__global__ void math_kernel_threadpartition(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0)
    {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

// 100% utilization, no "warp divergence"
__global__ void math_kernel_warppartition(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void math_kernel_branch_predication(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);
    if (ipred) {
        ia = 100.f;
    }
    if (!ipred) {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}


// CPU version of the kernel
void math_cpu_threadpartition(float *c, int size) {
    for (int tid = 0; tid < size; tid++) {
        float a = 0.0f, b = 0.0f;
        if (tid % 2 == 0) {
            a = 100.0f;
        } else {
            b = 200.0f;
        }
        c[tid] = a + b;
    }
}

void math_cpu_warppartition(float *c, int size) {
    for (int tid = 0; tid < size; ++tid) {
        float a = 0.0f, b = 0.0f;
        if ((tid / 32) % 2 == 0) {
            a = 100.0f;
        } else {
            b = 200.0f;
        }
        c[tid] = a + b;
    }
}

void math_cpu_branch_predication(float *c, int size) {
    for (int tid = 0; tid < size; tid++) {
        float ia = 0.0f, ib = 0.0f;
        bool ipred = (tid % 2 == 0);
        if (ipred) {
        ia = 100.f;
        }
        if (!ipred) {
            ib = 200.0f;
        }
        c[tid] = ia + ib;
    }
}

// Verification function
void verify_result(float *host_ref, float *gpu_ref, int size) {
    const float epsilon = 1.0e-8;
    for (int i = 0; i < size; i++) {
        if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
            printf("Results do not match at index %d: host %f gpu %f\n", i, host_ref[i], gpu_ref[i]);
            return;
        }
    }
    printf("Results match.\n");
}


int main(int argc, char const *argv[])
{
    int dev = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, device_prop.name);

    // set up data size
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size      = atoi(argv[2]);

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d) \n", block.x, grid.x);

    // allocate host memory
    float *h_C = (float *)malloc(size * sizeof(float));
    float *host_ref = (float *)malloc(size * sizeof(float));
    float *gpu_ref = (float *)malloc(size * sizeof(float));

    // allocate gpu memory
    float *d_C;
    size_t n_bytes = size * sizeof(float);
    cudaMalloc((float**) &d_C, n_bytes);

    // run a warmup kernel to remove overhead, creating event
    cudaEvent_t i_start, i_elaps;
    float elapstime;

    cudaEventCreate(&i_start);
    cudaEventCreate(&i_elaps);


    ////////////////////////////////////////////////

    // measure time for thread partition kernel
    cudaEventRecord(i_start, 0);
    math_kernel_threadpartition<<<grid, block>>>(d_C);
    cudaEventRecord(i_elaps, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapstime, i_start, i_elaps);
    printf("Mathkernel THREAD partition <<<%4d, %4d  >>> elapsed %f seconds \n", grid.x, block.x, elapstime);

    // copy result from device to host
    cudaMemcpy(gpu_ref, d_C, n_bytes, cudaMemcpyDeviceToHost);

    // compute reference solution
    math_cpu_threadpartition(host_ref, size);

    // verify result
    verify_result(host_ref, gpu_ref, size);

    ////////////////////////////////////////////////

    // measure time for warp partition kernel
    cudaEventRecord(i_start, 0);
    math_kernel_warppartition<<<grid, block>>>(d_C);
    cudaEventRecord(i_elaps, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapstime, i_start, i_elaps);
    printf("Mathkernel WARP partition <<<%4d, %4d  >>> elapsed %f seconds \n", grid.x, block.x, elapstime);

    // copy result from device to host
    cudaMemcpy(gpu_ref, d_C, n_bytes, cudaMemcpyDeviceToHost);

    // compute reference solution
    math_cpu_warppartition(host_ref, size);

    // verify result
    verify_result(host_ref, gpu_ref, size);

    ////////////////////////////////////////////////

    // measure time for branch predication kernel
    cudaEventRecord(i_start, 0);
    math_kernel_branch_predication<<< grid, block >>>(d_C);
    cudaEventRecord(i_elaps, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapstime, i_start, i_elaps);
    printf("Mathkernel branch predication <<< %4d, %4d >>> elapsed %f seconds \n", grid.x, block.x, elapstime);

    // copy result from device to host
    cudaMemcpy(gpu_ref, d_C, n_bytes, cudaMemcpyDeviceToHost);

    // compute reference solution
    math_cpu_branch_predication(host_ref, size);

    // verify result
    verify_result(host_ref, gpu_ref, size);

    ////////////////////////////////////////////////

    cudaFree(d_C);
    free(h_C);
    free(host_ref);
    free(gpu_ref);
    cudaEventDestroy(i_start);
    cudaEventDestroy(i_elaps);
    cudaDeviceReset();

    return 0;
}
