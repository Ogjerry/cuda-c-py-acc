#include <stdio.h>
#include <cuda_runtime.h>



__global__ void sum_matrix_on_gpu_2d(float *A, float *B, float *C, int NX, int NY) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char const *argv[])
{
    int dev = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, device_prop.name);

    // set up data size
    int nx = 1 << 12;
    int ny = 1 << 12;
    int size = nx * ny;

    // execution configuration
    unsigned int dimx = 32, dimy = 32; // default block size
    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("Execution Configure (block %d grid %d) \n", block.x, grid.x);

    // allocate host memory
    float *h_A = (float *)malloc(size * sizeof(float)); // host array is empty needing initialization
    float *h_B = (float *)malloc(size * sizeof(float)); // host array is empty needing initialization
    float *host_ref = (float *)malloc(size * sizeof(float));
    float *gpu_ref = (float *)malloc(size * sizeof(float));

    // host A and B initialization
    for (int i = 0; i < size; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // allocate gpu memory
    float *d_A, *d_B, *d_C;
    size_t n_bytes = size * sizeof(float);
    cudaMalloc((float**) &d_A, n_bytes);
    cudaMalloc((float**) &d_B, n_bytes);
    cudaMalloc((float**) &d_C, n_bytes);

    // create time event
    cudaEvent_t i_start, i_elaps;
    float elapstime;

    cudaEventCreate(&i_start);
    cudaEventCreate(&i_elaps);

    // copy host arrays onto GPU device
    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice); // now d_A = h_A
    cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice); // now d_B = h_B
    
    // execute kernel with different configurations
    cudaEventRecord(i_start, 0);
    sum_matrix_on_gpu_2d<<< grid, block >>>(d_A, d_B, d_C, nx, ny);
    cudaEventRecord(i_elaps, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapstime, i_start, i_elaps);
    printf("Mathkernel THREAD partition <<< (%d, %d), (%d, %d) >>> elapsed %f ms \n", (nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, dimx, dimy, elapstime);

    // copy computed results from d_C to gpu_ref
    cudaMemcpy(gpu_ref, d_C, n_bytes, cudaMemcpyDeviceToHost);

    // compute CPU reference
    for (int i = 0; i < size; i++) host_ref[i] = h_A[i] + h_B[i];
    
    // verify gpu results
    int match = 1;
    for (int i = 0; i < size; i++) {
        if (abs(host_ref[i] - gpu_ref[i]) > 1e-5) {
            match = 0;
            printf("Arrays doe not match at index [%d]: host[%f] gpu[%f] \n", i, host_ref[i], gpu_ref[i]);
            break;
        }
    }
    if (match) {
        printf("Array Match!\n");
    }

    // free device and host memory
    free(h_A);
    free(h_B);
    free(host_ref);
    free(gpu_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // destroy events
    cudaEventDestroy(i_start);
    cudaEventDestroy(i_elaps);

    return 0;
}
