#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>


// Adding back slash when #define multiple lines
#define CHECK(call) do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10 * error); \
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




void sum_matrix_host(float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ib[ix] + ia[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}




__global__ void sum_matrix_gpu_2d(float *A, float *B, float *C, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[index] = A[index] + B[index];
    }
}



int main(int argc, char const *argv[])
{
    printf("%s Starting...\n", argv[0]);

    // get device info
    int dev = 0;
    cudaDeviceProp device_prop;
    CHECK(cudaGetDeviceProperties(&device_prop, dev));
    printf("Using device %d: %s\n", dev, device_prop.name);
    CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx = 1 << 13;
    int ny = 1 << 13;
    int nxy = nx * ny;
    int n_bytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *host_ref, *gpu_ref;
    h_A = (float *) malloc(n_bytes);
    h_B = (float *) malloc(n_bytes);
    host_ref = (float *) malloc(n_bytes);
    gpu_ref = (float *) malloc(n_bytes);

    // initialize data at host side
    double i_start = cpuSecond();
    initialize_data(h_A, nxy);
    initialize_data(h_B, nxy);
    double i_elaps = cpuSecond() - i_start;

    memset(host_ref, 0, n_bytes);
    memset(gpu_ref, 0, n_bytes);

    // sum matrix at host side for result checks
    i_start = cpuSecond();
    sum_matrix_host(h_A, h_B, host_ref, nx, ny);
    i_elaps = cpuSecond() - i_start;


    // malloc global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **) &d_MatA, n_bytes);
    cudaMalloc((void **) &d_MatB, n_bytes);
    cudaMalloc((void **) &d_MatC, n_bytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, n_bytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1) / block.y);

    i_start = cpuSecond();
    sum_matrix_gpu_2d<<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);

    // invoking the kernel
    cudaDeviceSynchronize();
    i_elaps = cpuSecond() - i_start;
    printf("sum_gpu_2d <<< (%d, %d), (%d, %d) >>> elapsed %f sec \n", grid.x, grid.y, block.x, block.y, i_elaps);

    // copy gpu results back to host
    cudaMemcpy(gpu_ref, d_MatC, n_bytes, cudaMemcpyDeviceToHost);

    //check device results
    check_result_(host_ref, gpu_ref, nxy);

    // free host and device memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    free(host_ref);
    free(gpu_ref);

    // reset device
    cudaDeviceReset();

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        printf("Device %d: Maximum Grid Dimensions = (%d, %d, %d)\n",
               deviceId, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }


    return 0;
}
