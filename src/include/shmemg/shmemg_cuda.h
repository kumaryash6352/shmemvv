/**
 * @file shmemg_cuda.h
 * @brief OpenSHMEM Auxiliary Specification Wrappers for CUDA
 *
 * This header defines macros required for tests to be performed
 * with a device using CUDA.
 */

#ifndef SHMEMGVV_DEVICE_H_
#define SHMEMGVV_DEVICE_H_

#include <stdint.h>

#if defined(__has_include)
# if __has_include(<cuda_runtime.h>)
#  include <cuda_runtime.h>
# endif
#elif defined(__CUDACC__)
# include <cuda_runtime.h>
#endif

/* Execution space qualifiers */
#ifndef SHMEMGVV_KERNEL
# if defined(__CUDACC__) || defined(__NVCC__) || defined(__CUDA__)
#  define SHMEMGVV_KERNEL __global__
#  define SHMEMGVV_DEVICE_FN __device__
# else
#  define SHMEMGVV_KERNEL
#  define SHMEMGVV_DEVICE_FN
# endif
#endif

/* Device selection */
#define SHMEMGVV_SELECT_DEVICE(idx) cudaSetDevice(idx)

/* Memory Operations */
/* Copy `len` bytes from host pointer `hptr` to device pointer `dptr`. */
#define SHMEMGVV_MEMCPY_H2D(dptr, hptr, len) cudaMemcpy((dptr), (hptr), (len), cudaMemcpyHostToDevice)
/* Copy `len` bytes from device pointer `dptr` to host pointer `hptr`. */
#define SHMEMGVV_MEMCPY_D2H(hptr, dptr, len) cudaMemcpy((hptr), (dptr), (len), cudaMemcpyDeviceToHost)
/* Set `len` bytes of memory at `dptr` on the device to byte value `val`. */
#define SHMEMGVV_MEMSET(dptr, val, len) cudaMemset((dptr), (val), (len))

/* Allocate `len` elements of type `type` in device private memory and bind to `name`. */
#define SHMEMGVV_KERNEL_MALLOC(len, type, name) \
  type* name;                                   \
  cudaMalloc((void**)&(name), (len) * sizeof(type))

/* Free device private memory allocation `name`. */
#define SHMEMGVV_KERNEL_FREE(_len, _type, name) cudaFree(name)

/* Kernel launch abstraction */
#ifndef SHMEMGVV_LAUNCH_KERNEL
# if defined(__CUDACC__) || defined(__NVCC__) || defined(__CUDA__)
#  define SHMEMGVV_LAUNCH_KERNEL(kernel, num_tgs, tg_size, ...) \
     kernel<<<(num_tgs), (tg_size)>>>(__VA_ARGS__)
# else
#  define SHMEMGVV_LAUNCH_KERNEL(kernel, num_tgs, tg_size, ...) \
     kernel(__VA_ARGS__)
# endif
#endif

/* Host-Device synchronization and error handling */
#define SHMEMGVV_DEVICE_SYNC() cudaDeviceSynchronize()
#define SHMEMGVV_CHECK_ERROR() cudaGetLastError()

/* Thread and Thread Group coordinate / indexing macros */
#define SHMEMGVV_THREAD_ID_X (threadIdx.x)
#define SHMEMGVV_THREAD_ID_Y (threadIdx.y)
#define SHMEMGVV_THREAD_ID_Z (threadIdx.z)
#define SHMEMGVV_TG_ID_X     (blockIdx.x)
#define SHMEMGVV_TG_ID_Y     (blockIdx.y)
#define SHMEMGVV_TG_ID_Z     (blockIdx.z)
#define SHMEMGVV_TG_DIM_X    (blockDim.x)
#define SHMEMGVV_TG_DIM_Y    (blockDim.y)
#define SHMEMGVV_TG_DIM_Z    (blockDim.z)
#define SHMEMGVV_GRID_DIM_X  (gridDim.x)
#define SHMEMGVV_GRID_DIM_Y  (gridDim.y)
#define SHMEMGVV_GRID_DIM_Z  (gridDim.z)

/* Backwards compatibility coordinate aliases */
#define SHMEMGVV_BLOCK_ID_X  SHMEMGVV_TG_ID_X
#define SHMEMGVV_BLOCK_ID_Y  SHMEMGVV_TG_ID_Y
#define SHMEMGVV_BLOCK_ID_Z  SHMEMGVV_TG_ID_Z
#define SHMEMGVV_BLOCK_DIM_X SHMEMGVV_TG_DIM_X
#define SHMEMGVV_BLOCK_DIM_Y SHMEMGVV_TG_DIM_Y
#define SHMEMGVV_BLOCK_DIM_Z SHMEMGVV_TG_DIM_Z

#define SHMEMGVV_GLOBAL_THREAD_ID() ((blockIdx.x) * (blockDim.x) + (threadIdx.x))
#define SHMEMGVV_IS_MASTER_THREAD() ((threadIdx.x) == 0 && (blockIdx.x) == 0)

/* Thread group (_tg) and hardware thread group (_htg) synchronization */
#define SHMEMGVV_TG_SYNC()     __syncthreads()
#define SHMEMGVV_HTG_SYNC()    __syncwarp()
#define SHMEMGVV_THREAD_SYNC() SHMEMGVV_TG_SYNC()
#define SHMEMGVV_WARP_SYNC()   SHMEMGVV_HTG_SYNC()

/* Call the OpenSHMEM aux spec routine `routine` with the provided, potentially
   empty, variadic list of parameters. */
#define SHMEMGVV_KERNEL_CALL_ROUTINE(routine, ...) routine(__VA_ARGS__)

#endif // SHMEMGVV_DEVICE_H_
