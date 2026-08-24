/**
 * @file shmemg_interface.h
 * @brief OpenSHMEM Auxiliary Specification for GPU Support
 *
 * This header defines the interface a user must provide in
 * order to run the GPU Auxiliary Specification tests.
 */

#ifndef SHMEMG_INTERFACE_H_
#define SHMEMG_INTERFACE_H_

/**
 * Required macro definitions for runtime-agnostic GPU-centric test implementations,
 * based on the OpenSHMEM Auxiliary Specification for GPU Support:
 *
 * SHMEMGVV_KERNEL: Qualifier for device kernel entry functions
 * SHMEMGVV_DEVICE_FN: Device execution-space annotation for device-callable routines (e.g. SHMEMG_DEVICE)
 *
 * SHMEMGVV_SELECT_DEVICE(idx): Associates the calling PE with GPU device idx
 *
 * SHMEMGVV_MEMCPY_H2D(dptr, hptr, len): Copies len bytes from host memory (hptr) to device memory (dptr)
 * SHMEMGVV_MEMCPY_D2H(hptr, dptr, len): Copies len bytes from device memory (dptr) to host memory (hptr)
 * SHMEMGVV_MEMSET(dptr, val, len): Sets len bytes of device memory at dptr to byte value val
 * SHMEMGVV_KERNEL_MALLOC(len, type, name): Allocates private device memory for len elements of type
 * SHMEMGVV_KERNEL_FREE(len, type, name): Deallocates private device memory
 *
 * SHMEMGVV_LAUNCH_KERNEL(kernel, num_tgs, tg_size, ...): Launches a device kernel over num_tgs thread groups of tg_size threads
 * SHMEMGVV_DEVICE_SYNC(): Synchronizes host execution with device kernel completion
 * SHMEMGVV_CHECK_ERROR(): Retrieves device execution status or error code
 *
 * SHMEMGVV_THREAD_ID_X, SHMEMGVV_THREAD_ID_Y, SHMEMGVV_THREAD_ID_Z: Coordinates of calling thread within its thread group
 * SHMEMGVV_TG_ID_X, SHMEMGVV_TG_ID_Y, SHMEMGVV_TG_ID_Z: Coordinates of thread group within launch grid
 * SHMEMGVV_TG_DIM_X, SHMEMGVV_TG_DIM_Y, SHMEMGVV_TG_DIM_Z: Dimensions of thread group
 * SHMEMGVV_GRID_DIM_X, SHMEMGVV_GRID_DIM_Y, SHMEMGVV_GRID_DIM_Z: Dimensions of launch grid
 *
 * If your runtime does not support three dimensions in a launch grid, implement as many as your runtime
 * does support and zero the rest.
 * 
 * SHMEMGVV_GLOBAL_THREAD_ID(): Global linear thread index across the grid
 * SHMEMGVV_IS_MASTER_THREAD(): Evaluates to true for the leader thread (e.g. thread 0 in thread group 0)
 *
 * SHMEMGVV_TG_SYNC(): Synchronizes all threads within the calling thread group (_tg)
 * SHMEMGVV_HTG_SYNC(): Synchronizes all threads within the calling hardware thread group (_htg)
 *
 * SHMEMGVV_KERNEL_CALL_ROUTINE(routine, ...): Invokes an OpenSHMEM GPU routine
 */

#if defined(__CUDACC__) || defined(__CUDA__) || defined(__NVCC__) || defined(SHMEMGVV_USE_CUDA)
# include "shmemg/shmemg_cuda.h"
#elif defined(__HIPCC__) || defined(__HIP__) || defined(SHMEMGVV_USE_HIP)
# include "shmemg/shmemg_hip.h"
#endif

#endif // SHMEMG_INTERFACE_H_
