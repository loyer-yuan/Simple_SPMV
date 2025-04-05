#ifndef XSPARSE_OPS_H
#define XSPARSE_OPS_H

#include "Matrix.hpp"

namespace xsparse {

/**
 * @brief COO SpMV kernel warpper
 *
 * @tparam DType data type
 * @param cooData input COO data
 * @param cooRowIndices input COO row indices
 * @param cooColIndices input COO column indices
 * @param vec input vector
 * @param out output vector
 * @param m number of rows
 * @param k number of columns
 * @param nnz number of non-zero elements
 */
template <typename DType>
void compute_spmv_coo(
    const DType *__restrict__ cooData, const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);

}  // namespace xsparse

#endif  // XSPARSE_OPS_H