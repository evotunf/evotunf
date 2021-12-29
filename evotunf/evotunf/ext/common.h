#ifndef EVOTUNF_COMMON_H
#define EVOTUNF_COMMON_H

#include <float.h>

#define __forceinline __attribute__((always_inline)) inline

#ifdef NDEBUG
#define ASSERT_EX(cond, stmt) (0)
#else
#define ASSERT_EX(cond, stmt)			\
  do {						\
    if (!(cond)) { stmt; assert(cond); }	\
  } while (0)
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define ALIGN_DOWN(base, n) ((base) & ~((n)-1))
#define ALIGN_UP(base, n) ALIGN_DOWN((base) + (n) - 1, (n))

#ifdef __cplusplus
#define SWAP(x, y) std::swap(x, y)
#else
#define SWAP(x, y)															\
  ({																						\
    __auto_type tmp = (x);											\
    (x) = (y);																	\
    (y) = tmp;																	\
  })
#endif

#ifdef __cplusplus
#define __auto_type auto
#endif

#define ROUND_UP_TO_POW2(x)                     \
  ({                                            \
    unsigned v = (x);                        \
                                                \
    --v;                                        \
    v |= v >> 1;                                \
    v |= v >> 2;                                \
    v |= v >> 4;                                \
    v |= v >> 8;                                \
    v |= v >> 16;                               \
    ++v;                                        \
  })

#endif // EVOTUNF_COMMON_H
