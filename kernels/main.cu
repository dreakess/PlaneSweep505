#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <new>
#include <memory>
#include <chrono>
#include <stdint.h>
#include <vector>
#include <iostream>

// Macro definitions
#define MI(r, c, width) ((r) * (width) + (c))

// Parameters defining the block size and the radius of the SAD window
#define BLOCKSIZE 16
#define RAD 1

// Selector for using double precision (set to 1 to use double, 0 for float)
#define USE_DOUBLE 0

// Selection of the data type for computations based on the USE_DOUBLE flag
#if USE_DOUBLE
typedef double Real;
#else
typedef float Real;
#endif

// Macro definition for error checking after CUDA API calls
#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// Constants in GPU memory for camera parameters (invK, R_inv, t_inv, K_proj, R_proj, t_proj)

__constant__ Real c_invK[9];
__constant__ Real c_R_inv[9];
__constant__ Real c_t_inv[3];
__constant__ Real c_K_proj[9];
__constant__ Real c_R_proj[9];
__constant__ Real c_t_proj[3];

//__device__ Real c_invK[9];
//__device__ Real c_R_inv[9];
//__device__ Real c_t_inv[3];
//__device__ Real c_K_proj[9];
//__device__ Real c_R_proj[9];
//__device__ Real c_t_proj[3];


// functions for timing and FLOPS calculation

std::chrono::high_resolution_clock::time_point start_cpu_timer()
{
    return std::chrono::high_resolution_clock::now();
}

void end_cpu_timer(std::chrono::high_resolution_clock::time_point start, const char* name, double flop)
{
    auto stop = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    double gflops = flop / seconds / 1e9;

    printf("%s:\n", name);
    printf("  Processing: %.6f (s), GFLOPS: %.2f\n", seconds, gflops);
}

cudaEvent_t start_cuda_timer()
{
    cudaEvent_t start;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventRecord(start, nullptr));
    return start;
}

void end_cuda_timer(cudaEvent_t start, const char* name, double flop)
{
    cudaEvent_t stop;
    CHK(cudaEventCreate(&stop));
    CHK(cudaEventRecord(stop, nullptr));
    CHK(cudaEventSynchronize(stop));
    
    float millisec;
    CHK(cudaEventElapsedTime(&millisec, start, stop));
    double seconds = millisec / 1000.0;
    double gflops = flop / seconds / 1e9;

    printf("%s:\n", name);
    printf("  Processing: %.6f (s), GFLOPS: %.2f\n", seconds, gflops);
    
    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
}

// Calculate FLOPS based on SAD algorithm:
// SAD window size^2 * 2 (subtract + add) * pixels * planes * sensors
double calculateFLOPs(int img_w, int img_h, int ZPlanes, size_t num_sensors)
{
    int kernel_size = 2 * RAD + 1;
    // 2 ops per SAD comparison + geometry overhead
    return 2.0 * kernel_size * kernel_size * img_w * img_h * ZPlanes * num_sensors;
}

// Warmup kernel to mitigate initial overheads in timing (optional, can be removed if not needed)
__global__ void warmup(float* A, float* B, int w, int h) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= w || j >= h) return;

    int idx = MI(j, i, w);
    A[idx] = B[idx];
}

__global__ void shared_kernel(uint8_t* refY, uint8_t* sensY, float* costCube,
    int img_w, int img_h, float ZNear, float ZFar, int ZPlanes)
{
    // Shared memory tile with halo for SAD window
    extern __shared__ uint8_t tmp[];

    // indexing for shared memory tile (including halo)
    const int s_width = BLOCKSIZE + 2 * RAD;

    // indexing for global memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // indexing for Z plane directly from the grid
    int zi = blockIdx.z;

    // indedx for output cost cube
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // loading of the tile into shared memory with halo, handling borders

    // loading of the central part of the tile
    if (i < img_w && j < img_h)
        tmp[MI(ty + RAD, tx + RAD, s_width)] = refY[MI(j, i, img_w)];
    else
        tmp[MI(ty + RAD, tx + RAD, s_width)] = 0;

    // loading of the left halo
    if (tx < RAD) {
        int gx = i - RAD;
        int gy = j;
        tmp[MI(ty + RAD, tx, s_width)] = (gx >= 0 && gy >= 0 && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the right halo
    if (tx >= BLOCKSIZE - RAD) {
        int gx = i + RAD;
        int gy = j;
        tmp[MI(ty + RAD, tx + 2 * RAD, s_width)] = (gx < img_w && gy >= 0 && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the top halo
    if (ty < RAD) {
        int gx = i;
        int gy = j - RAD;
        tmp[MI(ty, tx + RAD, s_width)] = (gy >= 0 && gx >= 0 && gx < img_w) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the bottom halo
    if (ty >= BLOCKSIZE - RAD) {
        int gx = i;
        int gy = j + RAD;
        tmp[MI(ty + 2 * RAD, tx + RAD, s_width)] = (gy < img_h && gx >= 0 && gx < img_w) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the top-left corner
    if (tx < RAD && ty < RAD) {
        int gx = i - RAD;
        int gy = j - RAD;
        tmp[MI(ty, tx, s_width)] = (gx >= 0 && gy >= 0) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the top-right corner
    if (tx >= BLOCKSIZE - RAD && ty < RAD) {
        int gx = i + RAD;
        int gy = j - RAD;
        tmp[MI(ty, tx + 2 * RAD, s_width)] = (gx < img_w && gy >= 0) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the bottom-left corner
    if (tx < RAD && ty >= BLOCKSIZE - RAD) {
        int gx = i - RAD;
        int gy = j + RAD;
        tmp[MI(ty + 2 * RAD, tx, s_width)] = (gx >= 0 && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // loading of the bottom-right corner
    if (tx >= BLOCKSIZE - RAD && ty >= BLOCKSIZE - RAD) {
        int gx = i + RAD;
        int gy = j + RAD;
        tmp[MI(ty + 2 * RAD, tx + 2 * RAD, s_width)] = (gx < img_w && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // synchronization to ensure all threads have loaded their data into shared memory before any thread accesses it
    __syncthreads();

    // geometry and SAD computation, now using the tile in shared memory for refY and global memory for sensY


    // checking bounds for the current thread and Z plane
    if (i >= img_w || j >= img_h || zi >= ZPlanes) return;

    Real z = (Real)ZNear * (Real)ZFar / ((Real)ZNear + (((Real)zi / (Real)ZPlanes) * ((Real)ZFar - (Real)ZNear)));

    Real X_ref = (c_invK[0] * i + c_invK[1] * j + c_invK[2]) * z;
    Real Y_ref = (c_invK[3] * i + c_invK[4] * j + c_invK[5]) * z;
    Real Z_ref = (c_invK[6] * i + c_invK[7] * j + c_invK[8]) * z;

    Real Xw = c_R_inv[0] * X_ref + c_R_inv[1] * Y_ref + c_R_inv[2] * Z_ref - c_t_inv[0];
    Real Yw = c_R_inv[3] * X_ref + c_R_inv[4] * Y_ref + c_R_inv[5] * Z_ref - c_t_inv[1];
    Real Zw = c_R_inv[6] * X_ref + c_R_inv[7] * Y_ref + c_R_inv[8] * Z_ref - c_t_inv[2];

    Real X_p = c_R_proj[0] * Xw + c_R_proj[1] * Yw + c_R_proj[2] * Zw - c_t_proj[0];
    Real Y_p = c_R_proj[3] * Xw + c_R_proj[4] * Yw + c_R_proj[5] * Zw - c_t_proj[1];
    Real Z_p = c_R_proj[6] * Xw + c_R_proj[7] * Yw + c_R_proj[8] * Zw - c_t_proj[2];

    float x_proj = (float)(c_K_proj[0] * X_p / Z_p + c_K_proj[1] * Y_p / Z_p + c_K_proj[2]);
    float y_proj = (float)(c_K_proj[3] * X_p / Z_p + c_K_proj[4] * Y_p / Z_p + c_K_proj[5]);

    float cost = 0.0f;
    float count = 0.0f;

    // sad computation

    for (int ki = -RAD; ki <= RAD; ++ki) {
        for (int kj = -RAD; kj <= RAD; ++kj) {
            uint8_t valRef = tmp[MI(ty + RAD + kj, tx + RAD + ki, s_width)];

            int sx = roundf(x_proj) + ki;
            int sy = roundf(y_proj) + kj;

            if (sx >= 0 && sx < img_w && sy >= 0 && sy < img_h) {
                cost += fabsf((float)valRef - (float)sensY[MI(sy, sx, img_w)]);
                count += 1.0f;
            }
        }
    }

    if (count > 0) {
        int out_idx = zi * (img_w * img_h) + (j * img_w + i);
        costCube[out_idx] = fminf(cost / count, costCube[out_idx]);
    }
}


__global__ void naive_kernel(uint8_t* refY, uint8_t* sensY, float* costCube,
    int img_w, int img_h, float ZNear, float ZFar, int ZPlanes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int zi = blockIdx.z;

    if (i >= img_w || j >= img_h || zi >= ZPlanes) return;

    Real z = (Real)ZNear * (Real)ZFar / ((Real)ZNear + (((Real)zi / (Real)ZPlanes) * ((Real)ZFar - (Real)ZNear)));

    Real X_ref = (c_invK[0] * i + c_invK[1] * j + c_invK[2]) * z;
    Real Y_ref = (c_invK[3] * i + c_invK[4] * j + c_invK[5]) * z;
    Real Z_ref = (c_invK[6] * i + c_invK[7] * j + c_invK[8]) * z;

    Real Xw = c_R_inv[0] * X_ref + c_R_inv[1] * Y_ref + c_R_inv[2] * Z_ref - c_t_inv[0];
    Real Yw = c_R_inv[3] * X_ref + c_R_inv[4] * Y_ref + c_R_inv[5] * Z_ref - c_t_inv[1];
    Real Zw = c_R_inv[6] * X_ref + c_R_inv[7] * Y_ref + c_R_inv[8] * Z_ref - c_t_inv[2];

    Real X_p = c_R_proj[0] * Xw + c_R_proj[1] * Yw + c_R_proj[2] * Zw - c_t_proj[0];
    Real Y_p = c_R_proj[3] * Xw + c_R_proj[4] * Yw + c_R_proj[5] * Zw - c_t_proj[1];
    Real Z_p = c_R_proj[6] * Xw + c_R_proj[7] * Yw + c_R_proj[8] * Zw - c_t_proj[2];

    float x_proj = (float)(c_K_proj[0] * X_p / Z_p + c_K_proj[1] * Y_p / Z_p + c_K_proj[2]);
    float y_proj = (float)(c_K_proj[3] * X_p / Z_p + c_K_proj[4] * Y_p / Z_p + c_K_proj[5]);

    float cost = 0.0f;
    float count = 0.0f;

    for (int ki = -RAD; ki <= RAD; ++ki) {
        for (int kj = -RAD; kj <= RAD; ++kj) {
            int ref_x = i + ki;
            int ref_y = j + kj;

            int sx = roundf(x_proj) + ki;
            int sy = roundf(y_proj) + kj;

            if (ref_x >= 0 && ref_x < img_w && ref_y >= 0 && ref_y < img_h &&
                sx >= 0 && sx < img_w && sy >= 0 && sy < img_h)
            {
                uint8_t valRef = refY[MI(ref_y, ref_x, img_w)];
                uint8_t valSens = sensY[MI(sy, sx, img_w)];

                cost += fabsf((float)valRef - (float)valSens);
                count += 1.0f;
            }
        }
    }

    if (count > 0) {
        int out_idx = zi * (img_w * img_h) + (j * img_w + i);
        costCube[out_idx] = fminf(cost / count, costCube[out_idx]);
    }
}

// Simple kernel to initialize the cost cube with a default value (e.g., 255 for SAD)

__global__ void initCostCube(float* cube, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) cube[idx] = val;
}

// Interface function called from main.cpp to run the plane sweeping algorithm on the GPU
void runPlaneSweepingGPU(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes)
{

    // Timing and FLOPS calculation setup
    auto total_start = start_cpu_timer();
    // Calculate FLOPS for the entire plane sweeping process based on the algorithm's complexity
    double flops = calculateFLOPs(width, height, ZPlanes, sensor_images.size());

    const int imgSize = width * height;
    size_t plane = width * height * sizeof(float);

    //float* warmupA; float* warmupB;
    //CHK(cudaMalloc(&warmupA, plane));
    //CHK(cudaMalloc(&warmupB, plane));

    //dim3 warmupblock(BLOCKSIZE, BLOCKSIZE);
    //dim3 warmupgrid(((width + BLOCKSIZE - 1) / BLOCKSIZE), ((height + BLOCKSIZE - 1) / BLOCKSIZE));
    //warmup << <warmupgrid, warmupblock >> > (warmupA, warmupB, width, height);
    //CHK(cudaGetLastError());


    uint8_t* d_ref, * d_sens;
    float* d_costCube;

    CHK(cudaMalloc(&d_ref, imgSize));
    CHK(cudaMalloc(&d_sens, imgSize));
    CHK(cudaMalloc(&d_costCube, imgSize * ZPlanes * sizeof(float)));

    // Initialize the cost cube to a high value (e.g., 255 for SAD) before processing
    int totalElements = imgSize * ZPlanes;
    initCostCube << < (totalElements + 255) / 256, 256 >> > (d_costCube, totalElements, 255.0f);
    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(d_ref, ref_image, imgSize, cudaMemcpyHostToDevice));

    // Extract camera parameters for the reference image and copy them to constant memory
    const auto& ref_params = cam_params[0];
    std::vector<Real> h_invK(ref_params.begin(), ref_params.begin() + 9);
    std::vector<Real> h_R_inv(ref_params.begin() + 9, ref_params.begin() + 18);
    std::vector<Real> h_t_inv(ref_params.begin() + 18, ref_params.end());

    CHK(cudaMemcpyToSymbol(c_invK, h_invK.data(), 9 * sizeof(Real)));
    CHK(cudaMemcpyToSymbol(c_R_inv, h_R_inv.data(), 9 * sizeof(Real)));
    CHK(cudaMemcpyToSymbol(c_t_inv, h_t_inv.data(), 3 * sizeof(Real)));

    // definition of block and grid dimensions for the kernel launch
    dim3 block(BLOCKSIZE, BLOCKSIZE);

    // definition of a 3D grid 
    dim3 grid_3D((width + BLOCKSIZE - 1) / BLOCKSIZE, (height + BLOCKSIZE - 1) / BLOCKSIZE, ZPlanes);

    // calculation of shared memory size needed for the tile (including halo) for the shared_kernel
    size_t sharedMemBytes = (BLOCKSIZE + 2 * RAD) * (BLOCKSIZE + 2 * RAD) * sizeof(uint8_t);

    auto kernel_start = start_cuda_timer();

    // loop of the sensor
    for (size_t c = 0; c < sensor_images.size(); c++) {
        const auto& sens_params = cam_params[c + 1];

        // loading of the camera parameters for the current sensor and copying them to constant memory
        std::vector<Real> h_K_proj(sens_params.begin(), sens_params.begin() + 9);
        std::vector<Real> h_R_proj(sens_params.begin() + 9, sens_params.begin() + 18);
        std::vector<Real> h_t_proj(sens_params.begin() + 18, sens_params.end());

        CHK(cudaMemcpyToSymbol(c_K_proj, h_K_proj.data(), 9 * sizeof(Real)));
        CHK(cudaMemcpyToSymbol(c_R_proj, h_R_proj.data(), 9 * sizeof(Real)));
        CHK(cudaMemcpyToSymbol(c_t_proj, h_t_proj.data(), 3 * sizeof(Real)));

        CHK(cudaMemcpy(d_sens, sensor_images[c], imgSize, cudaMemcpyHostToDevice));

        // launch of the kernel for the current sensor, using shared memory optimization
        shared_kernel << <grid_3D, block, sharedMemBytes >> > (
            d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);
        
        naive_kernel << <grid_3D, block >> > (
            d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);

        CHK(cudaGetLastError());
    }


    CHK(cudaDeviceSynchronize());
    end_cuda_timer(kernel_start, "(Kernel)", flops);
    end_cpu_timer(total_start, "(Total)", flops);

    CHK(cudaMemcpy(cost_cube, d_costCube, imgSize * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

    CHK(cudaFree(d_ref)); CHK(cudaFree(d_sens)); CHK(cudaFree(d_costCube));
    //CHK(cudaFree(warmupA)); CHK(cudaFree(warmupB));
}