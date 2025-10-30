/**
 * @file c_shmem_collectives_stress.c
 * @brief Unit test for executing many collective operations back to back.
 * 
 * This test exercises multiple collective operations in sequence to verify
 * that the implementation can handle consecutive collective calls correctly.
 */

#include "log.h"
#include "shmemvv.h"

#define NUM_ITERATIONS 10
#define BUFFER_SIZE 8

/**
 * Test executing multiple collective operations back to back
 */
int test_collectives_stress(void) {
  log_routine("collectives_stress_test");
  bool success = true;
  int npes = shmem_n_pes();
  int mype = shmem_my_pe();

  if (npes < 2) {
    log_info("skipping stress test - requires at least 2 PEs, got %d", npes);
    return success;
  }

  // allocate symmetric memory buffers
  long *broadcast_src = (long *)shmem_malloc(BUFFER_SIZE * sizeof(long));
  long *broadcast_dest = (long *)shmem_malloc(BUFFER_SIZE * sizeof(long));
  long *reduce_src = (long *)shmem_malloc(sizeof(long));
  long *reduce_dest = (long *)shmem_malloc(sizeof(long));
  long *collect_src = (long *)shmem_malloc(sizeof(long));
  long *collect_dest = (long *)shmem_malloc(npes * sizeof(long));
  long *alltoall_src = (long *)shmem_malloc(npes * sizeof(long));
  long *alltoall_dest = (long *)shmem_malloc(npes * sizeof(long));

  if (!broadcast_src || !broadcast_dest || !reduce_src || !reduce_dest ||
      !collect_src || !collect_dest || !alltoall_src || !alltoall_dest) {
    log_fail("failed to allocate symmetric memory");
    success = false;
    goto cleanup;
  }

  log_info("starting %d iterations of collective operations", NUM_ITERATIONS);

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    int root_pe = iter % npes;
    
    // broadcast operation
    for (int i = 0; i < BUFFER_SIZE; i++) {
      broadcast_dest[i] = 0;
      broadcast_src[i] = (mype == root_pe) ? (iter * BUFFER_SIZE + i) : 0;
    }
    
    shmem_barrier_all();
    shmem_long_broadcast(SHMEM_TEAM_WORLD, broadcast_dest, broadcast_src, 
                         BUFFER_SIZE, root_pe);
    
    // verify broadcast
    for (int i = 0; i < BUFFER_SIZE; i++) {
      long expected = iter * BUFFER_SIZE + i;
      if (broadcast_dest[i] != expected) {
        log_fail("iteration %d: broadcast failed at index %d, expected %ld, got %ld",
                 iter, i, expected, broadcast_dest[i]);
        success = false;
        goto cleanup;
      }
    }

    // reduction
    *reduce_src = mype;
    *reduce_dest = 0;
    
    shmem_barrier_all();
    shmem_long_sum_reduce(SHMEM_TEAM_WORLD, reduce_dest, reduce_src, 1);
    
    // verify sum
    long expected_sum = (long)(npes * (npes - 1)) / 2;
    if (*reduce_dest != expected_sum) {
      log_fail("iteration %d: reduction failed, expected %ld, got %ld",
               iter, expected_sum, *reduce_dest);
      success = false;
      goto cleanup;
    }

    // collect 
    *collect_src = mype + iter;
    for (int i = 0; i < npes; i++) {
      collect_dest[i] = -1;
    }
    
    shmem_barrier_all();
    shmem_long_fcollect(SHMEM_TEAM_WORLD, collect_dest, collect_src, 1);
    
    // verify collect
    for (int i = 0; i < npes; i++) {
      long expected = i + iter;
      if (collect_dest[i] != expected) {
        log_fail("iteration %d: collect failed at PE %d, expected %ld, got %ld",
                 iter, i, expected, collect_dest[i]);
        success = false;
        goto cleanup;
      }
    }

    // alltoall 
    for (int i = 0; i < npes; i++) {
      alltoall_src[i] = mype * 1000 + i + iter;
      alltoall_dest[i] = -1;
    }
    
    shmem_barrier_all();
    shmem_long_alltoall(SHMEM_TEAM_WORLD, alltoall_dest, alltoall_src, 1);
    
    // verify alltoall
    for (int i = 0; i < npes; i++) {
      long expected = i * 1000 + mype + iter;
      if (alltoall_dest[i] != expected) {
        log_fail("iteration %d: alltoall failed at index %d, expected %ld, got %ld",
                 iter, i, expected, alltoall_dest[i]);
        success = false;
        goto cleanup;
      }
    }

    // sync between iterations
    // might not be needed...
    shmem_sync_all();
    
    if (mype == 0 && (iter + 1) % 5 == 0) {
      log_info("completed %d/%d iterations", iter + 1, NUM_ITERATIONS);
    }
  }

  if (success) {
    log_info("all %d iterations completed successfully", NUM_ITERATIONS);
  }

cleanup:
  shmem_free(broadcast_src);
  shmem_free(broadcast_dest);
  shmem_free(reduce_src);
  shmem_free(reduce_dest);
  shmem_free(collect_src);
  shmem_free(collect_dest);
  shmem_free(alltoall_src);
  shmem_free(alltoall_dest);

  return success;
}

int main(int argc, char *argv[]) {
  shmem_init();
  log_init(__FILE__);

  bool result = test_collectives_stress();
  int rc = EXIT_SUCCESS;

  shmem_barrier_all();

  if (shmem_my_pe() == 0) {
    display_test_result("C collectives stress test", result, false);
  }

  if (!result) {
    rc = EXIT_FAILURE;
  }

  log_close(rc);
  shmem_finalize();

  return rc;
}
