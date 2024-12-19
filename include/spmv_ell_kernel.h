#ifndef SPMV_ELL_KERNEL_H
#define SPMV_ELL_KERNEL_H

/**
 * @brief Transfer the sparse matrix to the ELL format
 * 
 * @tparam T data type
 * @param mat input matrix
 * @param ell_data output data array
 * @param ell_indices output indices array
 * @param max_nnz_per_row output maximum number of non-zero elements per row
 * @param m input number of rows
 * @param k input number of columns
 * @param lda input leading dimension, default is k, means row-major
 */
template <typename T>
void mat2ell(const T* __restrict__ mat, T* __restrict__& ell_data, int* __restrict__& ell_indices, int& max_nnz_per_row, const int m, const int k, const int lda);


/**
 * @brief ELL SpMV kernel wrapper
 * 
 * @tparam T data type
 * @param ell_data input matrix's ELL format data array
 * @param ell_indices input matrix's ELL format indices array
 * @param input_vec input vector
 * @param output_vec output vector
 * @param max_nnz_per_row input maximum number of non-zero elements per row
 * @param m input number of rows
 * @param k input number of columns
 */
template <typename T>
void spmv_ell0(const T* __restrict__ ell_data, const int* __restrict__ ell_indices, const T* __restrict__ input_vec, T* __restrict__ output_vec, const int max_nnz_per_row, const int m, const int k);


/**
 * @brief ELL SpMV kernel wrapper
 * 
 * @tparam T data type
 * @param ell_data input matrix's ELL format data array
 * @param ell_indices input matrix's ELL format indices array
 * @param input_vec input vector
 * @param output_vec output vector
 * @param max_nnz_per_row input maximum number of non-zero elements per row
 * @param m input number of rows
 * @param k input number of columns
 */
template <typename T>
void compute_spmv_ell(const T* __restrict__ ell_data, const int* __restrict__ ell_indices, const T* __restrict__ input_vec, T* __restrict__ output_vec, const int max_nnz_per_row, const int m, const int k);

#endif