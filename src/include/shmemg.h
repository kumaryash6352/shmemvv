/**
 * @file shmemg.h
 * @brief OpenSHMEM Auxiliary Specification for GPU Support
 *
 * This header defines the GPU-aware and GPU-centric interfaces, types,
 * handles, and constants for OpenSHMEM GPU support until we have a real
 * implementation.
 */

#ifndef SHMEMG_H
#define SHMEMG_H 1

#include <shmem.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SHMEMG_DEVICE
# if defined(__CUDACC__) || defined(__HIPCC__)
#  define SHMEMG_DEVICE __device__
# elif defined(SYCL_LANGUAGE_VERSION)
#  define SHMEMG_DEVICE SYCL_EXTERNAL
# else
#  define SHMEMG_DEVICE
# endif
#endif

#ifndef SHMEM_DEVICE_HOST_INIT
#define SHMEM_DEVICE_HOST_INIT    (1 << 8)
#endif

#ifndef SHMEM_DEVICE_KERNEL_INIT
#define SHMEM_DEVICE_KERNEL_INIT  (1 << 9)
#endif

#ifndef SHMEM_DEVICE_CTX
#define SHMEM_DEVICE_CTX          (1 << 10)
#endif

extern shmem_team_t SHMEM_DEVICE_TEAM_WORLD;
extern shmem_team_t SHMEM_DEVICE_TEAM_SHARED;
extern shmem_ctx_t  SHMEM_DEVICE_CTX_DEFAULT;

#define SHMEMG_STANDARD_RMA_TYPES(X) \
  X(float, float)                    \
  X(double, double)                  \
  X(long double, longdouble)         \
  X(char, char)                      \
  X(signed char, schar)              \
  X(short, short)                    \
  X(int, int)                        \
  X(long, long)                      \
  X(long long, longlong)             \
  X(unsigned char, uchar)            \
  X(unsigned short, ushort)          \
  X(unsigned int, uint)              \
  X(unsigned long, ulong)            \
  X(unsigned long long, ulonglong)   \
  X(int8_t, int8)                    \
  X(int16_t, int16)                  \
  X(int32_t, int32)                  \
  X(int64_t, int64)                  \
  X(uint8_t, uint8)                  \
  X(uint16_t, uint16)                \
  X(uint32_t, uint32)                \
  X(uint64_t, uint64)                \
  X(size_t, size)                    \
  X(ptrdiff_t, ptrdiff)

#define SHMEMG_STANDARD_AMO_TYPES(X) \
  X(int, int)                        \
  X(long, long)                      \
  X(long long, longlong)             \
  X(unsigned int, uint)              \
  X(unsigned long, ulong)            \
  X(unsigned long long, ulonglong)   \
  X(int32_t, int32)                  \
  X(int64_t, int64)                  \
  X(uint32_t, uint32)                \
  X(uint64_t, uint64)                \
  X(size_t, size)                    \
  X(ptrdiff_t, ptrdiff)

#define SHMEMG_EXTENDED_AMO_TYPES(X) \
  X(float, float)                    \
  X(double, double)                  \
  X(int, int)                        \
  X(long, long)                      \
  X(long long, longlong)             \
  X(unsigned int, uint)              \
  X(unsigned long, ulong)            \
  X(unsigned long long, ulonglong)   \
  X(int32_t, int32)                  \
  X(int64_t, int64)                  \
  X(uint32_t, uint32)                \
  X(uint64_t, uint64)                \
  X(size_t, size)                    \
  X(ptrdiff_t, ptrdiff)

#define SHMEMG_BITWISE_AMO_TYPES(X) \
  X(unsigned int, uint)             \
  X(unsigned long, ulong)           \
  X(unsigned long long, ulonglong)  \
  X(int32_t, int32)                 \
  X(int64_t, int64)                 \
  X(uint32_t, uint32)               \
  X(uint64_t, uint64)

#define SHMEMG_SYNC_TYPES(X)        \
  X(short, short)                   \
  X(int, int)                       \
  X(long, long)                     \
  X(long long, longlong)            \
  X(unsigned short, ushort)         \
  X(unsigned int, uint)             \
  X(unsigned long, ulong)           \
  X(unsigned long long, ulonglong)  \
  X(int16_t, int16)                 \
  X(int32_t, int32)                 \
  X(int64_t, int64)                 \
  X(uint16_t, uint16)               \
  X(uint32_t, uint32)               \
  X(uint64_t, uint64)               \
  X(size_t, size)                   \
  X(ptrdiff_t, ptrdiff)

void *shmemg_malloc(size_t size);
void *shmemg_calloc(size_t count, size_t size);
void *shmemg_align(size_t alignment, size_t size);
void  shmemg_free(void *ptr);

SHMEMG_DEVICE int shmemg_my_pe(void);
SHMEMG_DEVICE int shmemg_n_pes(void);

#define SHMEMG_DECL_RMA_MEM_VARIANTS(NAME)                                            \
  SHMEMG_DEVICE void shmemg_##NAME(void *dest, const void *source, size_t nelems, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##NAME(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe); \
  SHMEMG_DEVICE void shmemg_##NAME##_tg(void *dest, const void *source, size_t nelems, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##NAME##_tg(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe); \
  SHMEMG_DEVICE void shmemg_##NAME##_htg(void *dest, const void *source, size_t nelems, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##NAME##_htg(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

SHMEMG_DECL_RMA_MEM_VARIANTS(putmem)
SHMEMG_DECL_RMA_MEM_VARIANTS(putmem_nbi)

SHMEMG_DECL_RMA_MEM_VARIANTS(getmem)
SHMEMG_DECL_RMA_MEM_VARIANTS(getmem_nbi)

#undef SHMEMG_DECL_RMA_MEM_VARIANTS

#define SHMEMG_DECL_ELEMENT_PUT(TYPE, TYPENAME)                                       \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_p(TYPE *dest, TYPE value, int pe);           \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_p(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);

SHMEMG_STANDARD_RMA_TYPES(SHMEMG_DECL_ELEMENT_PUT)
#undef SHMEMG_DECL_ELEMENT_PUT

#define SHMEMG_DECL_ELEMENT_GET(TYPE, TYPENAME)                                       \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_g(const TYPE *source, int pe);              \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_g(shmem_ctx_t ctx, const TYPE *source, int pe);

SHMEMG_STANDARD_RMA_TYPES(SHMEMG_DECL_ELEMENT_GET)
#undef SHMEMG_DECL_ELEMENT_GET

#define SHMEMG_DECL_ATOMIC_SET(TYPE, TYPENAME)                                        \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_set(TYPE *dest, TYPE value, int pe);  \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_set(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);

SHMEMG_EXTENDED_AMO_TYPES(SHMEMG_DECL_ATOMIC_SET)
#undef SHMEMG_DECL_ATOMIC_SET

#define SHMEMG_DECL_ATOMIC_ADD_INC(TYPE, TYPENAME)                                    \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_add(TYPE *dest, TYPE value, int pe);  \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_add(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_inc(TYPE *dest, int pe);              \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_inc(shmem_ctx_t ctx, TYPE *dest, int pe);

SHMEMG_STANDARD_AMO_TYPES(SHMEMG_DECL_ATOMIC_ADD_INC)
#undef SHMEMG_DECL_ATOMIC_ADD_INC

#define SHMEMG_DECL_BITWISE_AMO(TYPE, TYPENAME)                                       \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_and(TYPE *dest, TYPE value, int pe);  \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_and(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_or(TYPE *dest, TYPE value, int pe);   \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_or(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);  \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_xor(TYPE *dest, TYPE value, int pe);  \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_xor(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);

SHMEMG_BITWISE_AMO_TYPES(SHMEMG_DECL_BITWISE_AMO)
#undef SHMEMG_DECL_BITWISE_AMO

#define SHMEMG_DECL_ATOMIC_FETCH_SWAP(TYPE, TYPENAME)                                 \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_fetch(const TYPE *source, int pe);    \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_fetch(shmem_ctx_t ctx, const TYPE *source, int pe); \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_swap(TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_swap(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);

SHMEMG_EXTENDED_AMO_TYPES(SHMEMG_DECL_ATOMIC_FETCH_SWAP)
#undef SHMEMG_DECL_ATOMIC_FETCH_SWAP

#define SHMEMG_DECL_ATOMIC_FETCH_ARITH(TYPE, TYPENAME)                                \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_compare_swap(shmem_ctx_t ctx, TYPE *dest, TYPE cond, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_fetch_add(TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_fetch_add(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_fetch_inc(TYPE *dest, int pe);        \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_fetch_inc(shmem_ctx_t ctx, TYPE *dest, int pe);

SHMEMG_STANDARD_AMO_TYPES(SHMEMG_DECL_ATOMIC_FETCH_ARITH)
#undef SHMEMG_DECL_ATOMIC_FETCH_ARITH

#define SHMEMG_DECL_ATOMIC_FETCH_BITWISE(TYPE, TYPENAME)                              \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_fetch_and(TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_fetch_and(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_fetch_or(TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_fetch_or(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_##TYPENAME##_atomic_fetch_xor(TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE TYPE shmemg_ctx_##TYPENAME##_atomic_fetch_xor(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);

SHMEMG_BITWISE_AMO_TYPES(SHMEMG_DECL_ATOMIC_FETCH_BITWISE)
#undef SHMEMG_DECL_ATOMIC_FETCH_BITWISE

#define SHMEMG_DECL_ATOMIC_NBI_EXTENDED(TYPE, TYPENAME)                               \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_swap_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_swap_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_fetch_nbi(TYPE *fetch, const TYPE *source, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_fetch_nbi(shmem_ctx_t ctx, TYPE *fetch, const TYPE *source, int pe);

SHMEMG_EXTENDED_AMO_TYPES(SHMEMG_DECL_ATOMIC_NBI_EXTENDED)
#undef SHMEMG_DECL_ATOMIC_NBI_EXTENDED

#define SHMEMG_DECL_ATOMIC_NBI_STANDARD(TYPE, TYPENAME)                               \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_compare_swap_nbi(TYPE *fetch, TYPE *dest, TYPE cond, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_compare_swap_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, TYPE cond, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_fetch_add_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_fetch_add_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_fetch_inc_nbi(TYPE *fetch, TYPE *dest, int pe); \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_fetch_inc_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, int pe);

SHMEMG_STANDARD_AMO_TYPES(SHMEMG_DECL_ATOMIC_NBI_STANDARD)
#undef SHMEMG_DECL_ATOMIC_NBI_STANDARD

#define SHMEMG_DECL_ATOMIC_NBI_BITWISE(TYPE, TYPENAME)                                                                           \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_fetch_and_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);                      \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_fetch_and_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, TYPE value, int pe); \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_fetch_or_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);                       \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_fetch_or_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, TYPE value, int pe);  \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_atomic_fetch_xor_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);                      \
  SHMEMG_DEVICE void shmemg_ctx_##TYPENAME##_atomic_fetch_xor_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest, TYPE value, int pe);

SHMEMG_BITWISE_AMO_TYPES(SHMEMG_DECL_ATOMIC_NBI_BITWISE)
#undef SHMEMG_DECL_ATOMIC_NBI_BITWISE

#define SHMEMG_DECL_SIGNAL_MEM_VARIANTS(NAME)                                                                                                                         \
  SHMEMG_DEVICE void shmemg_##NAME(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);                           \
  SHMEMG_DEVICE void shmemg_ctx_##NAME(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);      \
  SHMEMG_DEVICE void shmemg_##NAME##_tg(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);                      \
  SHMEMG_DEVICE void shmemg_ctx_##NAME##_tg(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
  SHMEMG_DEVICE void shmemg_##NAME##_htg(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);                     \
  SHMEMG_DEVICE void shmemg_ctx_##NAME##_htg(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

SHMEMG_DECL_SIGNAL_MEM_VARIANTS(putmem_signal)
SHMEMG_DECL_SIGNAL_MEM_VARIANTS(putmem_signal_nbi)

#undef SHMEMG_DECL_SIGNAL_MEM_VARIANTS

SHMEMG_DEVICE uint64_t shmemg_signal_fetch(const uint64_t *sig_addr);

#define SHMEMG_DECL_SYNC_ROUTINES(TYPE, TYPENAME)                                             \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value);     \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_wait_until_tg(TYPE *ivar, int cmp, TYPE cmp_value);  \
  SHMEMG_DEVICE void shmemg_##TYPENAME##_wait_until_htg(TYPE *ivar, int cmp, TYPE cmp_value); \
  SHMEMG_DEVICE int  shmemg_##TYPENAME##_test(TYPE *ivar, int cmp, TYPE cmp_value);           \
  SHMEMG_DEVICE int  shmemg_##TYPENAME##_test_tg(TYPE *ivar, int cmp, TYPE cmp_value);        \
  SHMEMG_DEVICE int  shmemg_##TYPENAME##_test_htg(TYPE *ivar, int cmp, TYPE cmp_value);

SHMEMG_SYNC_TYPES(SHMEMG_DECL_SYNC_ROUTINES)
#undef SHMEMG_DECL_SYNC_ROUTINES

SHMEMG_DEVICE uint64_t shmemg_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_value);
SHMEMG_DEVICE uint64_t shmemg_signal_wait_until_tg(uint64_t *sig_addr, int cmp, uint64_t cmp_value);
SHMEMG_DEVICE uint64_t shmemg_signal_wait_until_htg(uint64_t *sig_addr, int cmp, uint64_t cmp_value);

#define SHMEMG_DECL_ORDERING_VARIANTS(NAME)                             \
  SHMEMG_DEVICE void shmemg_##NAME(void);                               \
  SHMEMG_DEVICE void shmemg_ctx_##NAME(shmem_ctx_t ctx);                \
  SHMEMG_DEVICE void shmemg_##NAME##_tg(void);                          \
  SHMEMG_DEVICE void shmemg_ctx_##NAME##_tg(shmem_ctx_t ctx);           \
  SHMEMG_DEVICE void shmemg_##NAME##_htg(void);                         \
  SHMEMG_DEVICE void shmemg_ctx_##NAME##_htg(shmem_ctx_t ctx);

SHMEMG_DECL_ORDERING_VARIANTS(fence)
SHMEMG_DECL_ORDERING_VARIANTS(quiet)

#undef SHMEMG_DECL_ORDERING_VARIANTS

SHMEMG_DEVICE int shmemg_sync_all(void);
SHMEMG_DEVICE int shmemg_sync_all_tg(void);
SHMEMG_DEVICE int shmemg_sync_all_htg(void);

SHMEMG_DEVICE int shmemg_team_sync(shmem_team_t team);
SHMEMG_DEVICE int shmemg_team_sync_tg(shmem_team_t team);
SHMEMG_DEVICE int shmemg_team_sync_htg(shmem_team_t team);

#ifdef __cplusplus
}
#endif

#endif /* SHMEMG_H */
