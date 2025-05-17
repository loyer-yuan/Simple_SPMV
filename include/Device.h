#ifndef XSPARSE_DEVICE_H
#define XSPARSE_DEVICE_H

#include <cstdint>

namespace xsparse {

struct HWParams
{
    int numCores = 1u;
};

struct RTParams
{
    int tid;
    HWParams hw;
};

}  // namespace xsparse

#endif  // XSPARSE_DEVICE_H