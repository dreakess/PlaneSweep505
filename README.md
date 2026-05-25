# CUDA Plane Sweeping — Depth Estimation via SAD

GPU implementation in CUDA of a **Plane Sweeping** algorithm for depth estimation from multi-camera images, with support for multiple memory optimization strategies and parallelism approaches.

---


## How to Configure and Select Optimizations

### 1. Select numerical precision

At the top of the file:

```cpp
#define USE_DOUBLE 0   // Float (default)
#define USE_DOUBLE 1   // Double (higher precision)
```

### 2. Configure the SAD window size

```cpp
#define RAD 1   // 3x3 window (default)
// RAD 2 → 5x5 window, more robust to noise but more expensive
// RAD 3 → 7x7 window
```

### 3. Configure the CUDA block size

```cpp
#define BLOCKSIZE 16   // 16x16 threads per block (default)
// Typical values: 8, 16, 32
```


### 4. Select where to store the matrices

```cpp
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

// now for example we are using the constant memory 

```

### 5. Select the kernel (Naive vs Shared Memory)

In `planeSweeping.cu`, inside `runPlaneSweepingGPU()`, find the two kernel launch lines and comment out the one you do **not** want to run:

```cpp
// === SHARED MEMORY VERSION ===
shared_kernel<<<grid_3D, block, sharedMemBytes>>>(
    d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);

// === NAIVE VERSION (baseline, no shared memory) ===
naive_kernel<<<grid_3D, block>>>(
    d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);
```




