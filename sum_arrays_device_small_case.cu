/*
 * 1D case for matrix addition
 * Will examine 2D cases with another function



*/



#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>


// Adding back slash when #define multiple lines
#define CHECK(call) do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while (0)

void check_result(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}


void check_result_(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n", hostRef[i], gpuRef[i], i);

            // Print additional values for debugging
            for (int j = i - 5; j <= i + 5; j++) {
                if (j >= 0 && j < N) {
                    printf("host[%d] = %5.2f, gpu[%d] = %5.2f\n", j, hostRef[j], j, gpuRef[j]);
                }
            }

            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}



void initialize_data(float *ip, int size) {
    // generate seed for random number
    time_t t;
    srand( (unsigned) time(&t) );

    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>(rand() & 0xFF) / 10.0f;
    }
}


void sum_array_host(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void sum_array_gpu_1d(float *A, float *B, float *C, const int N) {
    // on cpu, arrays assign linearly. But on gpus, we need to divide linear arrays 
    // by maximum thread length of gpu to perform parallel computing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { C[idx] = A[idx] + B[idx]; }
}



double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}


__global__ void Hi (void) {
    if (threadIdx.x == 5)
        printf("Hi From %d GPU!\n", threadIdx.x);
}

int main(int argc, char const *argv[])
{
    printf("%s Starting...\n", argv[0]);

    // Setup device
    int dev = 0;
    cudaDeviceProp device_prop;
    CHECK(cudaGetDeviceProperties(&device_prop, dev));

    // setup data size of vectors
    int n_elem = 1 << 25;
    printf("vector size: %d\n", n_elem);

    // Malloc host memory
    size_t n_bytes = n_elem * sizeof(float);

    float *h_A, *h_B, *h_ref, *gpu_ref;
    h_A = (float *) malloc(n_bytes);
    h_B = (float *) malloc(n_bytes);
    h_ref = (float *) malloc(n_bytes);
    gpu_ref = (float *) malloc(n_bytes);

    double i_start, i_elaps;

    // Initialize data at host side
    i_start = cpuSecond();
    initialize_data(h_A, n_elem);
    initialize_data(h_B, n_elem);
    i_elaps = cpuSecond() - i_start;

    memset(h_ref, 0, n_bytes);
    memset(gpu_ref, 0, n_bytes);
    
    // add vector at host side for result checks
    i_start = cpuSecond();
    sum_array_host(h_A, h_B, h_ref, n_elem);
    i_elaps = cpuSecond() - i_start;


    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **) &d_A, n_bytes);
    cudaMalloc((float **) &d_B, n_bytes);
    cudaMalloc((float **) &d_C, n_bytes);

    // transfering data from host to device
    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int i_len = 1024;
    dim3 block (i_len);
    dim3 grid ((n_elem + block.x - 1) / block.x);

    i_start = cpuSecond();
    sum_array_gpu_1d<<<grid, block>>>(d_A, d_B, d_C, n_elem);
    cudaDeviceSynchronize();
    i_elaps = cpuSecond() - i_start;
    printf("sum_array_gpu<<<%d, %d>>> time elapsed %f sec\n", grid.x, block.x, i_elaps);

    // copy kernel result back to host
    cudaMemcpy(gpu_ref, d_C, n_bytes, cudaMemcpyDeviceToHost);

    // check device results
    check_result_(h_ref, gpu_ref, n_elem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_ref);
    free(gpu_ref);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        printf("Device %d: Maximum Grid Dimensions = (%d, %d, %d)\n",
               deviceId, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }


    Hi<<<1, 10>>>();
    // If this line is removed, the command line will print nothing
    cudaDeviceReset();
    return 0;
}
