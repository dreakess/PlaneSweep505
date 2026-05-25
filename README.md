# CUDA Plane Sweeping — Depth Estimation via SAD

GPU implementation in CUDA of a **Plane Sweeping** algorithm for depth estimation from multi-camera images, with support for multiple memory optimization strategies and parallelism approaches.

---




### Constant Memory

All camera parameters (K, R, t matrices) are stored in **constant memory** (`__constant__`), which is a dedicated broadcast cache: when all threads in a warp read the same address, latency is equivalent to a register access.

```cpp
__constant__ Real c_invK[9];
__constant__ Real c_R_inv[9];
__constant__ Real c_t_inv[3];
__constant__ Real c_K_proj[9];
__constant__ Real c_R_proj[9];
__constant__ Real c_t_proj[3];
```

Reference camera parameters are uploaded once before the sensor loop. Sensor parameters are updated at each iteration via `cudaMemcpyToSymbol`.


---

## How to Configure and Select Optimizations

### 1. Select the kernel (Naive vs Shared Memory)

In `planeSweeping.cu`, inside `runPlaneSweepingGPU()`, find the two kernel launch lines and comment out the one you do **not** want to run:

```cpp
// === SHARED MEMORY VERSION ===
shared_kernel<<<grid_3D, block, sharedMemBytes>>>(
    d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);

// === NAIVE VERSION (baseline, no shared memory) ===
naive_kernel<<<grid_3D, block>>>(
    d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);
```

### 2. Select numerical precision

At the top of the file:

```cpp
#define USE_DOUBLE 0   // Float (default, faster)
#define USE_DOUBLE 1   // Double (higher precision)
```

### 3. Configure the SAD window size

```cpp
#define RAD 1   // 3x3 window (default)
// RAD 2 → 5x5 window, more robust to noise but more expensive
// RAD 3 → 7x7 window
```

Increasing `RAD` improves matching robustness but raises operation count quadratically: `(2*RAD+1)^2`. It also increases the SHMEM requirement for the shared kernel.

### 4. Configure the CUDA block size

```cpp
#define BLOCKSIZE 16   // 16x16 threads per block (default)
// Typical values: 8, 16, 32
```

Higher values increase theoretical occupancy but reduce the shared memory available per block. With `RAD=1` and `BLOCKSIZE=16`, SHMEM usage is `(16+2)^2 = 324 bytes` — well below the hardware limit.




