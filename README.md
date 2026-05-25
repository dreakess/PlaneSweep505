# CUDA Plane Sweeping — Depth Estimation via SAD

GPU implementation in CUDA of a **Plane Sweeping** algorithm for depth estimation from multi-camera images, with support for multiple memory optimization strategies and parallelism approaches.

---

## Problem Overview

Given a set of images captured from multiple cameras with calibrated positions and orientations, the algorithm estimates for each pixel of the **reference camera** its **depth** (distance from the camera). The result is a dense **depth map**.

This kind of processing is fundamental in:
- 3D reconstruction of real scenes
- Multi-baseline stereo vision
- Robotics and autonomous navigation

---

## Theoretical Background

### Plane Sweeping

The core idea is to "sweep" the scene with a series of virtual planes at increasing depths (from `ZNear` to `ZFar`), project each pixel of the reference camera onto each of these planes, and verify how photometrically consistent the projection onto the secondary cameras is.

For each Z plane, a matching cost is computed: the plane that yields the minimum cost for a given pixel corresponds to the estimated depth.

Plane depths are not distributed linearly but in an **inverse** (disparity-space) fashion:

```
z(zi) = ZNear * ZFar / (ZNear + (zi / ZPlanes) * (ZFar - ZNear))
```

This ensures a denser distribution of planes close to the camera, where disparity variation is largest.

### SAD — Sum of Absolute Differences

The matching cost is computed via **SAD (Sum of Absolute Differences)** over a window of size `(2*RAD+1) x (2*RAD+1)`:

```
SAD(x, y, z) = Σ_{ki,kj} | I_ref(x+ki, y+kj) - I_sens(x'_proj+ki, y'_proj+kj) |
```

A low value indicates high photometric similarity → the estimated depth is correct.

The cost is normalized by the number of valid pixels in the window (those that fall within image boundaries) and compared against the current value in the cost cube via `fminf`, keeping the minimum across all sensors.

### Projection Geometry

For each pixel `(i, j)` of the reference camera, at depth plane `z`, the geometric pipeline is:

1. **Back-projection** into reference camera space: `X_ref = K_ref^{-1} * [i, j, 1]^T * z`
2. **World transformation**: `X_world = R_ref^{-1} * X_ref - t_ref`
3. **Projection onto the sensor camera**: `X_sens = R_sens * X_world + t_sens`
4. **2D projection**: `[x', y'] = K_sens * X_sens / Z_sens`

Each camera's parameters are passed as a vector of 21 values:
- `[0..8]` → `K_inv` (inverse of the 3x3 intrinsic matrix, for the reference) or `K` (for sensors)
- `[9..17]` → rotation matrix `R` (3x3)
- `[18..20]` → translation vector `t` (3 elements)

---

## Code Architecture

```
planeSweeping.cu
├── Macros and typedef
│   ├── MI(r, c, width)       → linear 2D/3D indexing
│   ├── BLOCKSIZE             → thread block size (x and y)
│   ├── RAD                   → SAD window radius
│   └── USE_DOUBLE            → precision selector
│
├── Constant Memory
│   ├── c_invK, c_R_inv, c_t_inv      → reference camera parameters
│   └── c_K_proj, c_R_proj, c_t_proj  → current sensor camera parameters
│
├── Timers
│   ├── start_cpu_timer() / end_cpu_timer()   → wall-clock timing (std::chrono)
│   └── start_cuda_timer() / end_cuda_timer() → GPU event-based timing
│
├── GPU Kernels
│   ├── warmup()              → stabilizes GPU clock frequencies (optional)
│   ├── initCostCube()        → initializes cost cube to 255.0f
│   ├── naive_kernel()        → baseline version, reads directly from global VRAM
│   └── shared_kernel()       → optimized version using shared memory tile with halo
│
└── runPlaneSweepingGPU()     → host interface: allocate, transfer, launch kernels
```

---

## Implemented Optimization Strategies

### Naive Kernel

**Function:** `naive_kernel`

Baseline version. Each thread reads SAD window pixels directly from **global VRAM** for every Z plane.

- **3D grid**: `(img_w / BLOCKSIZE, img_h / BLOCKSIZE, ZPlanes)` — each Z block corresponds to one depth plane.

### Shared Memory Kernel

**Function:** `shared_kernel`

Optimization via **shared memory (SHMEM)**. Before computing the SAD, each block loads a tile of the reference camera image — including the surrounding halo of width `RAD` — into shared memory. Threads then read reference pixels from SHMEM instead of global VRAM.

The tile loading is split into 9 regions to ensure full coverage of the halo:

```
┌──────────┬──────────────┬──────────┐
│ Top-Left │     Top      │ Top-Right│
├──────────┼──────────────┼──────────┤
│   Left   │    Center    │   Right  │
├──────────┼──────────────┼──────────┤
│ Bot-Left │    Bottom    │ Bot-Right│
└──────────┴──────────────┴──────────┘
```

A `__syncthreads()` barrier ensures all threads have finished loading before any SAD computation begins.

```
SHMEM size = (BLOCKSIZE + 2*RAD)^2 * sizeof(uint8_t)
```

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




