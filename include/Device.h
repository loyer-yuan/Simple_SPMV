#ifndef XSPARSE_DEVICE_H
#define XSPARSE_DEVICE_H

#include <cstdint>

namespace xsparse {

struct HWParams
{
    uint numCores = 1u;
};

struct RTParams
{
    uint32_t tid;
    HWParams hw;
};

}  // namespace xsparse

#endif  // XSPARSE_DEVICE_H