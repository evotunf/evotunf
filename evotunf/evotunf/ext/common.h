#ifndef EVOTUNF_COMMON_H
#define EVOTUNF_COMMON_H

#include <float.h>

#ifdef NDEBUG
#define ASSERT_EX(cond, stmt) (0)
#else
#define ASSERT_EX(cond, stmt)                  \
    do {                                       \
        if (!(cond)) { stmt; assert(cond); }   \
    } while (0)
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define ALIGN_DOWN(base, n) ((base) & ~((n)-1))
#define ALIGN_UP(base, n) ALIGN_DOWN((base) + (n) - 1, (n))

#define SWAP(x, y)              \
    ({                          \
        __auto_type tmp = x;    \
        x = y;                  \
        y = tmp;                \
    })

#endif // EVOTUNF_COMMON_H
