#ifndef XSPARSE_ARMINTRINSIC_HPP
#define XSPARSE_ARMINTRINSIC_HPP

#include <stdint.h>

namespace xsparse {

typedef enum
{
    PREFETCH_READ,
    PREFETCH_WRITE
} prefetch_rw_t;

typedef enum
{
    PREFETCH_L1,
    PREFETCH_L2
} prefetch_level_t;

static inline void arm_prefetch(const void *p, prefetch_rw_t rw, prefetch_level_t lvl)
{
#if defined(__aarch64__)
    if (rw == PREFETCH_READ)
    {
        if (lvl == PREFETCH_L1)
        {
            __asm__ __volatile__("prfm pldl1keep, [%0]\n" ::"r"(p) : "memory");
        }
        else
        {
            __asm__ __volatile__("prfm pldl2keep, [%0]\n" ::"r"(p) : "memory");
        }
    }
    else
    {
        if (lvl == PREFETCH_L1)
        {
            __asm__ __volatile__("prfm pstl1keep, [%0]\n" ::"r"(p) : "memory");
        }
        else
        {
            __asm__ __volatile__("prfm pstl2keep, [%0]\n" ::"r"(p) : "memory");
        }
    }
#else
    // fallback: portable builtin
    __builtin_prefetch(p, (rw == PREFETCH_WRITE) ? 1 : 0, (lvl == PREFETCH_L1) ? 3 : 2);
#endif
}

}  // namespace xsparse

#endif  // XSPARSE_ARMINTRINSIC_HPP
