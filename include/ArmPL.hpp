#ifndef XSPARSE_ARMPL_HPP
#define XSPARSE_ARMPL_HPP

#include <type_traits>
#include "Matrix.hpp"
#include "armpl.h"

namespace xsparse {

namespace armplwarp {

template <typename DType>
struct OpE
{
    static armpl_status_t exec(
        enum armpl_sparse_hint_value trans, DType alpha, armpl_spmat_t A,
        const DType *x, DType beta, DType *y)
    {
        if constexpr (std::is_same_v<DType, float>)
        {
            return armpl_spmv_exec_s(trans, alpha, A, x, beta, y);
        }
        else if constexpr (std::is_same_v<DType, double>)
        {
            return armpl_spmv_exec_d(trans, alpha, A, x, beta, y);
        }
        else
        {
            std::cerr << "Unsupported data type!" << std::endl;
            return ARMPL_STATUS_EXECUTION_FAILURE;
        }
    }
};

template <EMatType::ESPMatFormat MType, typename DType>
struct OpC;

template <typename DType>
struct OpC<EMatType::ESPMatFormat::SPMatFormatCSR, DType>
{
    static armpl_status_t create(
        armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz,
        const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const DType *vals,
        int32_t flags)
    {
        if constexpr (std::is_same_v<DType, float>)
        {
            return armpl_spmat_create_csr_s(A, m, n, row_ptr, col_indx, vals, flags);
        }
        else if constexpr (std::is_same_v<DType, double>)
        {
            return armpl_spmat_create_csr_d(A, m, n, row_ptr, col_indx, vals, flags);
        }
        else
        {
            std::cerr << "Unsupported data type!" << std::endl;
            return ARMPL_STATUS_INPUT_PARAMETER_ERROR;
        }
    }
};

template <typename DType>
struct OpC<EMatType::ESPMatFormat::SPMatFormatCOO, DType>
{
    static armpl_status_t create(
        armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz,
        const armpl_int_t *row_indx, const armpl_int_t *col_indx, const DType *vals,
        armpl_int_t flags)
    {
        if constexpr (std::is_same_v<DType, float>)
        {
            return armpl_spmat_create_coo_s(
                A, m, n, nnz, row_indx, col_indx, vals, flags);
        }
        else if constexpr (std::is_same_v<DType, double>)
        {
            return armpl_spmat_create_coo_d(
                A, m, n, nnz, row_indx, col_indx, vals, flags);
        }
        else
        {
            std::cerr << "Unsupported data type!" << std::endl;
            return ARMPL_STATUS_INPUT_PARAMETER_ERROR;
        }
    }
};
}  // namespace armplwarp

#define ARMPL_CEHCK(x)                   \
    if (x != ARMPL_STATUS_SUCCESS)       \
    {                                    \
        armpl_spmat_print_err(armplMat); \
        return false;                    \
    }

template <EMatType::ESPMatFormat MType, typename DType>
class ArmPL
{
public:
    bool Initialize(
        const IdxType M, const IdxType N, const IdxType NNZ, const DType *vals,
        const IdxType *rowIdx, const IdxType *colIdx)
    {
        armpl_status_t info;
        info = armplwarp::OpC<MType, DType>::create(
            &armplMat, M, N, NNZ, rowIdx, colIdx, vals, creation_flags);
        ARMPL_CEHCK(info);

        // General optimize the matrix
        // info = armpl_spmat_hint(
        //     armplMat, ARMPL_SPARSE_HINT_STRUCTURE, ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED);
        // ARMPL_CEHCK(info);

        // info = armpl_spmat_hint(
        //     armplMat, ARMPL_SPARSE_HINT_SPMV_OPERATION, ARMPL_SPARSE_OPERATION_NOTRANS);
        // ARMPL_CEHCK(info);

        // info = armpl_spmat_hint(
        //     armplMat, ARMPL_SPARSE_HINT_SPMV_INVOCATIONS,
        //     ARMPL_SPARSE_INVOCATIONS_MANY);
        // ARMPL_CEHCK(info);

        info = armpl_spmv_optimize(armplMat);
        ARMPL_CEHCK(info);

        isInit = true;
        return true;
    }

    bool Run(const DType *inputV, DType *outputV)
    {
        if (!isInit)
        {
            std::cerr << "ArmPL not initialized!" << std::endl;
            return false;
        }

        armpl_status_t info = armplwarp::OpE<DType>::exec(
            ARMPL_SPARSE_OPERATION_NOTRANS, alpha, armplMat, inputV, beta, outputV);
        ARMPL_CEHCK(info);

        return true;
    }

    bool Destroy()
    {
        if (!isInit)
        {
            std::cerr << "ArmPL not initialized!" << std::endl;
            return false;
        }

        armpl_status_t info = armpl_spmat_destroy(armplMat);
        ARMPL_CEHCK(info);

        isInit = false;
        return true;
    }

private:
    bool isInit = false;
    armpl_spmat_t armplMat;
    const double alpha = 1.0;
    const double beta = 0.0;
    int32_t creation_flags = 0;
};

#undef ARMPL_CEHCK

}  // namespace xsparse

#endif  // XSPARSE_ARMPL_HPP