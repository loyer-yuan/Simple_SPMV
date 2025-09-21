#ifndef XSPARSE_KML_HPP
#define XSPARSE_KML_HPP

#include "Matrix.hpp"
#include "kspblas.h"

namespace xsparse {

namespace kmlwarp {

template <typename DType>
class KML
{
public:
    bool Run(
        const DType *__restrict__ csrData, const int *__restrict__ csrRowIdices,
        const int *__restrict__ csrColIdices, const DType *__restrict__ vec,
        DType *__restrict__ out, const int m, const int k)
    {
        kml_sparse_operation_t opt = KML_SPARSE_OPERATION_NON_TRANSPOSE;
        kml_sparse_status_t status =
            kml_csparse_scsrgemv(opt, m, csrData, csrRowIdices, csrColIdices, vec, out);

        if (status != KML_SPARSE_STATUS_SUCCESS)
        {
            std::cerr << "KML SPMV failed" << std::endl;
            switch (status)
            {
            case KML_SPARSE_STATUS_NOT_INITIALIZED:
                std::cerr << "矩阵句柄为空或者指向内存的指针为空" << std::endl;
                break;
            case KML_SPARSE_STATUS_ALLOC_FAILED:
                std::cerr << "内存申请失败" << std::endl;
                break;
            case KML_SPARSE_STATUS_INVALID_VALUE:
                std::cerr << "非法参数" << std::endl;
                break;
            case KML_SPARSE_STATUS_EXECUTION_FAILED:
                std::cerr << "执行失败" << std::endl;
                break;
            case KML_SPARSE_STATUS_INTERNAL_ERROR:
                std::cerr << "内部算法实现发生错误" << std::endl;
                break;
            case KML_SPARSE_STATUS_NOT_SUPPORTED:
                std::cerr << "当前参数对应的接口不支持" << std::endl;
                break;
            }
            return false;
        }

        return true;
    }
};

}  // namespace kmlwarp

}  // namespace xsparse

#endif  // XSPARSE_KML_HPP