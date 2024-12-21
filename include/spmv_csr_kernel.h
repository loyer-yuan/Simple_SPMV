#ifndef SPMV_CSR_KERNEL_H
#define SPMV_CSR_KERNEL_H


/**
 * @brief Transfer the sparse matrix to the CSR format
 * 
 * @tparam T data type
 * @param mat input matrix
 * @param csr_data output data array
 * @param csr_ptr output pointer array
 * @param csr_indices output indices array
 * @param m input number of rows
 * @param k input number of columns
 * @param lda input leading dimension, default is k, means row-major
 */
template <typename T>
void mat2csr(const T * __restrict__ mat, T * __restrict__ &csr_data, int * __restrict__ &csr_ptr, int * __restrict__ &csr_indices, const int m, const int k, const int lda);


/**
 * @brief CSR SpMV scalar kernel warpper
 * 
 * @tparam T data type
 * @param csr_data input CSR data
 * @param csr_ptr input CSR row pointer
 * @param csr_indices input CSR column index
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 */
template <typename T>
void spmv_csr_scalar0(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k);


/**
 * @brief CSR SpMV vector kernel warpper
 * 
 * @tparam T data type
 * @param csr_data input CSR data
 * @param csr_ptr input CSR row pointer
 * @param csr_indices input CSR column index
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 */
template <typename T>
void spmv_csr_vector0(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k);


/**
 * @brief CSR SpMV scalar Entry. Including memory allocation and data transfer
 * 
 * @tparam T data type
 * @param csr_data input CSR data
 * @param csr_ptr input CSR row pointer
 * @param csr_indices input CSR column index
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 */
template <typename T>
void compute_spmv_csr_scalar(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k);


/**
 * @brief CSR SpMV vector Entry. Including memory allocation and data transfer
 * 
 * @tparam T data type
 * @param csr_data input CSR data
 * @param csr_ptr input CSR row pointer
 * @param csr_indices input CSR column index
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 */
template <typename T>
void compute_spmv_csr_vector(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k);

#endif // SPMV_CSR_KERNEL_H