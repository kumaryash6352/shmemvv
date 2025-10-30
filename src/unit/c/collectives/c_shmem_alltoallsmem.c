/**
 * @file c_shmem_alltoallsmem.c
 *
 * @brief Unit test for shmem_alltoallsmem().
 */

#include <shmem.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "shmemvv.h"

#define TEST_C_SHMEM_ALLTOALLSMEM(DST_STRIDE, SST_STRIDE, NELEMS)       \
    ({                                                                  \
      log_routine("shmem_alltoallsmem(dst=" #DST_STRIDE ", sst=" #SST_STRIDE \
                ", nelems=" #NELEMS ")");                               \
    int npes = shmem_n_pes();                                           \
    int mype = shmem_my_pe();                                           \
                                                                        \
    size_t src_size = (size_t)(SST_STRIDE) * (size_t)(NELEMS) * (size_t)npes; \
    size_t dst_size = (size_t)(DST_STRIDE) * (size_t)(NELEMS) * (size_t)npes; \
                                                                        \
    unsigned char *src = (unsigned char *)shmem_malloc(src_size);       \
    unsigned char *dest = (unsigned char *)shmem_malloc(dst_size);      \
    log_info("shmem_malloc'd %zu bytes @ &src = %p, %zu bytes @ &dest = %p", \
             src_size, (void *)src, dst_size, (void *)dest);            \
                                                                        \
    /* set stride positions to my PE id */                              \
    for (int pe = 0; pe < npes; pe++) {                                 \
        for (int i = 0; i < (NELEMS); i++) {                            \
          size_t offset = (size_t)(pe * (NELEMS) + i) * (size_t)(SST_STRIDE); \
        src[offset] = (unsigned char)mype;                              \
      }                                                                 \
    }                                                                   \
                                                                        \
    /* init dest to a known bad value */                                \
    memset(dest, 0xFF, dst_size);                                       \
                                                                        \
    shmem_barrier_all();                                                \
                                                                        \
    log_info("executing shmem_alltoallsmem: dest = %p, src = %p", (void *)dest, \
             (void *)src);                                              \
    shmem_alltoallsmem(SHMEM_TEAM_WORLD, dest, src, (DST_STRIDE), (SST_STRIDE), (NELEMS)); \
                                                                        \
    shmem_barrier_all();                                                \
                                                                        \
    log_info("validating result...");                                   \
    bool success = true;                                                \
    for (int pe = 0; pe < npes; pe++) {                                 \
        for (int i = 0; i < (NELEMS); i++) {                            \
          size_t offset = (size_t)(pe * (NELEMS) + i) * (size_t)(DST_STRIDE); \
        unsigned char expected = (unsigned char)pe;                     \
        if (dest[offset] != expected) {                                 \
            log_info("dest[%zu] failed. expected %d, got %d", offset, (int)expected, \
                   (int)dest[offset]);                                  \
          success = false;                                              \
          break;                                                        \
        }                                                               \
      }                                                                 \
      if (!success)                                                     \
          break;                                                        \
    }                                                                   \
                                                                        \
    if (success)                                                        \
        log_info("shmem_alltoallsmem with dst=" #DST_STRIDE ", sst=" #SST_STRIDE \
               ", nelems=" #NELEMS " produced expected result.");       \
    else {                                                              \
        log_fail("at least one value was unexpected in result of shmem_alltoallsmem"); \
    }                                                                   \
                                                                        \
    if (mype < 2) {                                                     \
        log_info("PE %d: Sample results:", mype);                       \
      for (int pe = 0; pe < npes && pe < 3; pe++) {                     \
          for (int i = 0; i < (NELEMS) && i < 2; i++) {                 \
            size_t offset = (size_t)(pe * (NELEMS) + i) * (size_t)(DST_STRIDE); \
          log_info("  dest[%zu] = %d", offset, (int)dest[offset]);      \
        }                                                               \
      }                                                                 \
    }                                                                   \
                                                                        \
    shmem_free(src);                                                    \
    shmem_free(dest);                                                   \
                                                                        \
    success;                                                            \
  })

int main(int argc, char *argv[]) {
  shmem_init();
  log_init(__FILE__);

  bool result = true;
  int rc = EXIT_SUCCESS;

  result &= TEST_C_SHMEM_ALLTOALLSMEM(1, 1, 1);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(2, 1, 1);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(1, 2, 1);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(2, 2, 1);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(1, 1, 2);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(2, 1, 2);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(2, 2, 2);
  result &= TEST_C_SHMEM_ALLTOALLSMEM(3, 3, 3);

  shmem_barrier_all();

  if (shmem_my_pe() == 0) {
    display_test_result("C shmem_alltoallsmem", result, false);
  }

  if (!result) {
    rc = EXIT_FAILURE;
  }

  log_close(rc);
  shmem_finalize();
  return rc;
}
