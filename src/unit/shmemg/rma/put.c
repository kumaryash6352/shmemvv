/**
 * @file put.c
 * @brief Unit test for OpenSHMEM GPU-centric shmemg put routines.
 */

#include <shmem.h>
#include <shmemg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "log.h"
#include "shmemvv.h"
#include "shmemg_interface.h"

#define NUM_ELEMS 10

SHMEMGVV_KERNEL void kernel_shmemg_putmem(void *dest, const void *source, size_t nelems, int pe) {
  if (SHMEMGVV_IS_MASTER_THREAD()) {
    shmemg_putmem(dest, source, nelems, pe);
    shmemg_quiet();
  }
}

SHMEMGVV_KERNEL void kernel_shmemg_ctx_putmem(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  if (SHMEMGVV_IS_MASTER_THREAD()) {
    shmemg_ctx_putmem(ctx, dest, source, nelems, pe);
    shmemg_ctx_quiet(ctx);
  }
}

SHMEMGVV_KERNEL void kernel_shmemg_putmem_tg(void *dest, const void *source, size_t nelems, int pe) {
  shmemg_putmem_tg(dest, source, nelems, pe);
  shmemg_quiet_tg();
}

SHMEMGVV_KERNEL void kernel_shmemg_putmem_htg(void *dest, const void *source, size_t nelems, int pe) {
  shmemg_putmem_htg(dest, source, nelems, pe);
  shmemg_quiet_htg();
}

SHMEMGVV_KERNEL void kernel_shmemg_putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
  if (SHMEMGVV_IS_MASTER_THREAD()) {
    shmemg_putmem_nbi(dest, source, nelems, pe);
    shmemg_quiet();
  }
}

#define DEFINE_KERNEL_SHMEMG_P(TYPE, TYPENAME)                          \
    SHMEMGVV_KERNEL void kernel_shmemg_##TYPENAME##_p(TYPE *dest, TYPE val, int pe) { \
        if (SHMEMGVV_IS_MASTER_THREAD()) {                              \
            shmemg_##TYPENAME##_p(dest, val, pe);                       \
            shmemg_quiet();                                             \
        }                                                               \
    }
SHMEMG_STANDARD_RMA_TYPES(DEFINE_KERNEL_SHMEMG_P)
#undef DEFINE_KERNEL_SHMEMG_P

#define TEST_SHMEMG_PUTMEM_VARIANT(ROUTINE_NAME, KERNEL_CALL, NUM_TGS, TG_SIZE) ({ \
    log_routine(ROUTINE_NAME); \
    bool success = true;                                                \
    int mype = shmem_my_pe();                                           \
    int npes = shmem_n_pes();                                           \
    log_info("Running on PE %d of %d total PEs", mype, npes);           \
    char *d_src = (char *)shmemg_malloc(NUM_ELEMS * sizeof(char));      \
    char *d_dest = (char *)shmemg_malloc(NUM_ELEMS * sizeof(char));     \
    if (!d_src || !d_dest) {                                            \
        log_fail("Failed to allocate GPU symmetric memory");            \
        return false;                                                   \
    }                                                                   \
    char h_src[NUM_ELEMS];                                              \
    char h_dest[NUM_ELEMS];                                             \
    for (int i = 0; i < NUM_ELEMS; i++) {                               \
        h_src[i] = 'a' + i + mype;                                      \
        h_dest[i] = 0;                                                  \
    }                                                                   \
    SHMEMGVV_MEMCPY_H2D(d_src, h_src, NUM_ELEMS * sizeof(char));        \
    SHMEMGVV_MEMCPY_H2D(d_dest, h_dest, NUM_ELEMS * sizeof(char));      \
    SHMEMGVV_DEVICE_SYNC();                                             \
    shmem_barrier_all();                                                \
    if (mype == 0) {                                                    \
        log_info("PE 0: Launching device kernel for %s to PE 1", ROUTINE_NAME); \
        KERNEL_CALL;                                                    \
        SHMEMGVV_DEVICE_SYNC();                                         \
        log_info("PE 0: Completed device kernel");                      \
    }                                                                   \
    shmem_barrier_all();                                                \
    if (mype == 1) {                                                    \
        log_info("PE 1: Beginning validation of received data from PE 0"); \
        SHMEMGVV_MEMCPY_D2H(h_dest, d_dest, NUM_ELEMS * sizeof(char));  \
        SHMEMGVV_DEVICE_SYNC();                                         \
        for (int i = 0; i < NUM_ELEMS; i++) {                           \
            char expected = 'a' + i;                                    \
            if (h_dest[i] != expected) {                                \
                log_fail("PE 1: Validation failed - dest[%d] = '%c', expected '%c'", \
                         i, h_dest[i], expected);                       \
                success = false;                                        \
                break;                                                  \
            }                                                           \
            log_info("PE 1: dest[%d] = '%c' (valid)", i, h_dest[i]);    \
        }                                                               \  
        if (success) {                                                  \
            log_info("PE 1: All elements validated successfully");      \
        }                                                               \
    }                                                                   \
    shmem_barrier_all();                                                \
    shmemg_free(d_src);                                                 \
    shmemg_free(d_dest);                                                \
    success;                                                            \
})

#define TEST_SHMEMG_P(TYPE, TYPENAME)\
    ({                                                                  \
        log_routine("shmemg_" #TYPENAME "_p()");                        \
        bool success = true;                                            \
        int mype = shmem_my_pe();                                       \
        int npes = shmem_n_pes();                                       \
        TYPE *d_dest = (TYPE *)shmemg_malloc(sizeof(TYPE));             \
        if (!d_dest) {                                                  \
            log_fail("Failed to allocate GPU symmetric memory");        \
            return false;                                               \
        }                                                               \
        TYPE h_dest = (TYPE)0;                                          \
        SHMEMGVV_MEMCPY_H2D(d_dest, &h_dest, sizeof(TYPE));             \
        SHMEMGVV_DEVICE_SYNC();                                         \
        shmem_barrier_all();                                            \
        if (mype == 0) {                                                \
            TYPE val = (TYPE)42;                                        \
            log_info("PE 0: Launching kernel_shmemg_" #TYPENAME "_p to PE 1"); \
            SHMEMGVV_LAUNCH_KERNEL(kernel_shmemg_##TYPENAME##_p, 1, 1, d_dest, val, 1); \
            SHMEMGVV_DEVICE_SYNC();                                     \
        }                                                               \
        shmem_barrier_all();                                            \
        if (mype == 1) {                                                \
            SHMEMGVV_MEMCPY_D2H(&h_dest, d_dest, sizeof(TYPE));         \
            SHMEMGVV_DEVICE_SYNC();                                     \
            if (h_dest != (TYPE)42) {                                   \
                log_fail("PE 1: Validation failed for shmemg_" #TYPENAME "_p"); \
                success = false;                                        \
            } else {                                                    \
                log_info("PE 1: Validated shmemg_" #TYPENAME "_p successfully"); \
            }                                                           \
        }                                                               \
        shmem_barrier_all();                                            \
        shmemg_free(d_dest);                                            \
        success;                                                        \
    })

int main(int argc, char *argv[]) {
  int requested = SHMEM_THREAD_SINGLE | SHMEM_DEVICE_KERNEL_INIT;
  int provided = 0;

  SHMEMGVV_SELECT_DEVICE(0);

  shmem_init_thread(requested, &provided);
  log_init(__FILE__);

  if (!(provided & SHMEM_DEVICE_KERNEL_INIT)) {
    if (shmem_my_pe() == 0) {
      log_fail("OpenSHMEM implementation does not support SHMEM_DEVICE_KERNEL_INIT");
    }
    shmem_finalize();
    return EXIT_FAILURE;
  }

  if (shmem_n_pes() < 2) {
    if (shmem_my_pe() == 0) {
      display_not_enough_pes("GPU-Centric RMA");
    }
    shmem_finalize();
    return EXIT_SUCCESS;
  }

  int rc = EXIT_SUCCESS;
  bool result = true;

  bool res_putmem = TEST_SHMEMG_PUTMEM_VARIANT(
    "shmemg_putmem()",
    SHMEMGVV_LAUNCH_KERNEL(kernel_shmemg_putmem, 1, 1, d_dest, d_src, NUM_ELEMS * sizeof(char), 1),
    1, 1
  );
  if (shmem_my_pe() == 0) {
    display_test_result("shmemg_putmem", res_putmem, false);
  }
  result &= res_putmem;

  bool res_tg = TEST_SHMEMG_PUTMEM_VARIANT(
    "shmemg_putmem_tg()",
    SHMEMGVV_LAUNCH_KERNEL(kernel_shmemg_putmem_tg, 1, 32, d_dest, d_src, NUM_ELEMS * sizeof(char), 1),
    1, 32
  );
  if (shmem_my_pe() == 0) {
    display_test_result("shmemg_putmem_tg", res_tg, false);
  }
  result &= res_tg;

  bool res_htg = TEST_SHMEMG_PUTMEM_VARIANT(
    "shmemg_putmem_htg()",
    SHMEMGVV_LAUNCH_KERNEL(kernel_shmemg_putmem_htg, 1, 32, d_dest, d_src, NUM_ELEMS * sizeof(char), 1),
    1, 32
  );
  if (shmem_my_pe() == 0) {
    display_test_result("shmemg_putmem_htg", res_htg, false);
  }
  result &= res_htg;

  bool res_nbi = TEST_SHMEMG_PUTMEM_VARIANT(
    "shmemg_putmem_nbi()",
    SHMEMGVV_LAUNCH_KERNEL(kernel_shmemg_putmem_nbi, 1, 1, d_dest, d_src, NUM_ELEMS * sizeof(char), 1),
    1, 1
  );
  if (shmem_my_pe() == 0) {
    display_test_result("shmemg_putmem_nbi", res_nbi, false);
  }
  result &= res_nbi;

  shmem_ctx_t dev_ctx = SHMEM_DEVICE_CTX_DEFAULT;
  bool res_ctx = TEST_SHMEMG_PUTMEM_VARIANT(
    "shmemg_ctx_putmem()",
    SHMEMGVV_LAUNCH_KERNEL(kernel_shmemg_ctx_putmem, 1, 1, dev_ctx, d_dest, d_src, NUM_ELEMS * sizeof(char), 1),
    1, 1
  );
  if (shmem_my_pe() == 0) {
    display_test_result("shmemg_ctx_putmem", res_ctx, false);
  }
  result &= res_ctx;

  bool res_p = true;
  res_p &= TEST_SHMEMG_P(float, float);
  res_p &= TEST_SHMEMG_P(double, double);
  res_p &= TEST_SHMEMG_P(char, char);
  res_p &= TEST_SHMEMG_P(short, short);
  res_p &= TEST_SHMEMG_P(int, int);
  res_p &= TEST_SHMEMG_P(long, long);
  res_p &= TEST_SHMEMG_P(long long, longlong);
  res_p &= TEST_SHMEMG_P(int32_t, int32);
  res_p &= TEST_SHMEMG_P(int64_t, int64);
  res_p &= TEST_SHMEMG_P(uint32_t, uint32);
  res_p &= TEST_SHMEMG_P(uint64_t, uint64);
  res_p &= TEST_SHMEMG_P(size_t, size);

  if (shmem_my_pe() == 0) {
    display_test_result("shmemg_<type>_p", res_p, false);
  }
  result &= res_p;

  if (!result) {
    rc = EXIT_FAILURE;
  }

  log_close(rc);
  shmem_finalize();
  return rc;
}
