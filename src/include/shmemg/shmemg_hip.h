/**
 * @file shmemg_hip.h
 * @brief OpenSHMEM Auxiliary Specification Wrappers for HIP
 *
 * This header defines macros required for tests to be performed
 * with a device using HIP / ROCm.
 */

#ifndef SHMEMGVV_HIP_H_
#define SHMEMGVV_HIP_H_

#include <stdint.h>

#if defined(__has_include)
# if __has_include(<hip/hip_runtime.h>)
#  include <hip/hip_runtime.h>
# endif
#elif defined(__HIPCC__) || defined(__HIP__)
# include <hip/hip_runtime.h>
#endif

/* Execution space qualifiers */
#ifndef SHMEMGVV_KERNEL
# if defined(__HIPCC__) || defined(__HIP__)
#  define SHMEMGVV_KERNEL __global__
#  define SHMEMGVV_DEVICE_FN __device__
# else
#  define SHMEMGVV_KERNEL
#  define SHMEMGVV_DEVICE_FN
# endif
#endif

/* Device selection */
#define SHMEMGVV_SELECT_DEVICE(idx) hipSetDevice(idx)

/* Memory Operations */
/* Copy `len` bytes from host pointer `hptr` to device pointer `dptr`. */
#define SHMEMGVV_MEMCPY_H2D(dptr, hptr, len) hipMemcpy((dptr), (hptr), (len), hipMemcpyHostToDevice)
/* Copy `len` bytes from device pointer `dptr` to host pointer `hptr`. */
#define SHMEMGVV_MEMCPY_D2H(hptr, dptr, len) hipMemcpy((hptr), (dptr), (len), hipMemcpyDeviceToHost)
/* Set `len` bytes of memory at `dptr` on the device to byte value `val`. */
#define SHMEMGVV_MEMSET(dptr, val, len) hipMemset((dptr), (val), (len))

/* Allocate `len` elements of type `type` in device private memory and bind to `name`. */
#define SHMEMGVV_KERNEL_MALLOC(len, type, name) \
  type* name;                                   \
  hipMalloc((void**)&(name), (len) * sizeof(type))

/* Free device private memory allocation `name`. */
#define SHMEMGVV_KERNEL_FREE(_len, _type, name) hipFree(name)

/* Kernel launch abstraction */
#ifndef SHMEMGVV_LAUNCH_KERNEL
# if defined(__HIPCC__) || defined(__HIP__)
#  define SHMEMGVV_LAUNCH_KERNEL(kernel, num_tgs, tg_size, ...) \
     hipLaunchKernelGGL(kernel, dim3(num_tgs), dim3(tg_size), 0, 0, __VA_ARGS__)
# else
#  define SHMEMGVV_LAUNCH_KERNEL(kernel, num_tgs, tg_size, ...) \
     kernel(__VA_ARGS__)
# endif
#endif

/* Host-Device synchronization and error handling */
#define SHMEMGVV_DEVICE_SYNC() hipDeviceSynchronize()
#define SHMEMGVV_CHECK_ERROR() hipGetLastError()

/* Thread and Thread Group coordinate / indexing macros */
#define SHMEMGVV_THREAD_ID_X (hipThreadIdx_x)
#define SHMEMGVV_THREAD_ID_Y (hipThreadIdx_y)
#define SHMEMGVV_THREAD_ID_Z (hipThreadIdx_z)
#define SHMEMGVV_TG_ID_X     (hipBlockIdx_x)
#define SHMEMGVV_TG_ID_Y     (hipBlockIdx_y)
#define SHMEMGVV_TG_ID_Z     (hipBlockIdx_z)
#define SHMEMGVV_TG_DIM_X    (hipBlockDim_x)
#define SHMEMGVV_TG_DIM_Y    (hipBlockDim_y)
#define SHMEMGVV_TG_DIM_Z    (hipBlockDim_z)
#define SHMEMGVV_GRID_DIM_X  (hipGridDim_x)
#define SHMEMGVV_GRID_DIM_Y  (hipGridDim_y)
#define SHMEMGVV_GRID_DIM_Z  (hipGridDim_z)

/* Backwards compatibility coordinate aliases */
#define SHMEMGVV_BLOCK_ID_X  SHMEMGVV_TG_ID_X
#define SHMEMGVV_BLOCK_ID_Y  SHMEMGVV_TG_ID_Y
#define SHMEMGVV_BLOCK_ID_Z  SHMEMGVV_TG_ID_Z
#define SHMEMGVV_BLOCK_DIM_X SHMEMGVV_TG_DIM_X
#define SHMEMGVV_BLOCK_DIM_Y SHMEMGVV_TG_DIM_Y
#define SHMEMGVV_BLOCK_DIM_Z SHMEMGVV_TG_DIM_Z

#define SHMEMGVV_GLOBAL_THREAD_ID() ((hipBlockIdx_x) * (hipBlockDim_x) + (hipThreadIdx_x))
#define SHMEMGVV_IS_MASTER_THREAD() ((hipThreadIdx_x) == 0 && (hipBlockIdx_x) == 0)

/* Thread group (_tg) and hardware thread group (_htg) synchronization */
#define SHMEMGVV_TG_SYNC()     __syncthreads()
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
# define SHMEMGVV_HTG_SYNC()   __builtin_amdgcn_wave_barrier()
#else
# define SHMEMGVV_HTG_SYNC()   __syncwarp()
#endif
#define SHMEMGVV_THREAD_SYNC() SHMEMGVV_TG_SYNC()
#define SHMEMGVV_WARP_SYNC()   SHMEMGVV_HTG_SYNC()

/* Call the OpenSHMEM aux spec routine `routine` with the provided, potentially
   empty, variadic list of parameters. */
#define SHMEMGVV_KERNEL_CALL_ROUTINE(routine, ...) routine(__VA_ARGS__)

#endif // SHMEMGVV_HIP_H_
