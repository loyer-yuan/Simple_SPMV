#ifndef XSPARSE_ARMPL_HPP
#define XSPARSE_ARMPL_HPP

#include <armpl.h>
#include <type_traits>
#include "Matrix.hpp"

namespace xsparse {

template <EMatType::ESPMatFormat MType, typename DType>
class ArmPL
{
public:
    bool Initialize(
        const IdxType M, const IdxType N, const IdxType NNZ, const DType *vals,
        const IdxType *rowIdx, const IdxType *colIdx)
    {
        armpl_status_t info;
        if constexpr (MType == EMatType::ESPMatFormat::SPMatFormatCSR)
        {
#define CreateCSR(DType)                                            \
    {                                                               \
        info = armpl_spmat_create_csr_##DType(                      \
            &armplMat, M, N, rowIdx, colIdx, vals, creation_flags); \
    }

            if constexpr (std::is_same_v(DType, float))
                CreateCSR(s);
            else if constexpr (std::is_same_v(DType, double))
                CreateCSR(d);
            else
            {
                std::cerr << "Unsupported data type!" << std::endl;
                return false;
            }
        }
    }

    bool Run()
    {
    }

private:
    bool isInit = false;
    armpl_spmat_t armplMat;
    const double alpha = 1.0;
    const double beta = 0.0;
    int64_t creation_flags = 0;
};

}  // namespace xsparse

#endif  // XSPARSE_ARMPL_HPP