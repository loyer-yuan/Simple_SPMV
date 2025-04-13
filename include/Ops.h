#ifndef XSPARSE_OPS_H
#define XSPARSE_OPS_H

#include "Matrix.hpp"

namespace xsparse {

/**
 * @brief Compute the sparse matrix-vector multiplication (SpMV) using COO format
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
void ComputeSPMVCOO(
    const DType *__restrict__ cooData, const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);

/**
 * @brief Compute the sparse matrix-vector multiplication (SpMV) using CSR format
 *
 * @tparam DType data type
 * @param csrData input CSR data
 * @param csrRowIdices input CSR row indices
 * @param csrColIdices input CSR column indices
 * @param vec input vector
 * @param out output vector
 * @param m number of rows
 * @param k number of columns
 */
template <typename DType>
void ComputeSPMVCSR(
    const DType *__restrict__ csrData, const IdxType *__restrict__ csrRowIdices,
    const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k);

/**
 * @brief Compute the sparse matrix-vector multiplication (SpMV) using ELL format
 *
 * @tparam DType data type
 * @param dataMat input ELL data
 * @param idxMat input ELL index matrix
 * @param iVec input vector
 * @param oVer output vector
 * @param m number of rows
 * @param k number of columns
 * @param colIdxMat number of columns in index matrix
 */
template <typename DType>
void ComputeSPMVELL(
    const DType *__restrict__ dataMat, const IdxType *__restrict__ idxMat,
    const DType *__restrict__ iVec, DType *__restrict__ oVer, const IdxType m,
    const IdxType k, const IdxType colIdxMat);

}  // namespace xsparse

#endif  // XSPARSE_OPS_H