#ifndef XSPARSE_DEVICE_H
#define XSPARSE_DEVICE_H

#include <cstdint>

#define NumCores 8u

namespace xsparse {

struct RTParams
{
    uint32_t tid;
};

}  // namespace xsparse

#endif  // XSPARSE_DEVICE_H