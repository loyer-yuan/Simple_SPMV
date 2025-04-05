#ifndef SPMV_COO_KERNEL_H
#define SPMV_COO_KERNEL_H

/**
 * @brief Transfer the sparse matrix to the COO format
 * 
 * @tparam T data type
 * @param mat input matrix
 * @param coo_data output data array
 * @param coo_row_indices output row indices array
 * @param coo_col_indices output column indices array
 * @param m input number of rows
 * @param k input number of columns
 * @param lda input leading dimension, default is k, means row-major
 */
template <typename T>
void mat2coo(
    const T *__restrict__ mat, T *__restrict__ &coo_data,
    int *__restrict__ &coo_row_indices, int *__restrict__ &coo_col_indices, int &nnz,
    const int m, const int k, const int lda)
{
    assertm(
        m > 0 && k > 0, "Invalid matrix size, size of matrix must be greater than 0");
    assertm(k == lda, "Only support k == lda");

    vector<T> tdata;
    vector<int> trow_indices;
    vector<int> tcol_indices;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            const T val = mat[i * lda + j];
            if (val != 0)
            {
                tdata.push_back(val);
                trow_indices.push_back(i);
                tcol_indices.push_back(j);
            }
        }
    }

    nnz = tdata.size();
    // Copy to coo_data, coo_row_indices, coo_col_indices
    coo_data = new T[tdata.size()];
    coo_row_indices = new int[trow_indices.size()];
    coo_col_indices = new int[tcol_indices.size()];
    std::memcpy(coo_data, tdata.data(), sizeof(T) * tdata.size());
    std::memcpy(
        coo_row_indices, trow_indices.data(), sizeof(int) * trow_indices.size());
    std::memcpy(
        coo_col_indices, tcol_indices.data(), sizeof(int) * tcol_indices.size());
}

/**
 * @brief COO SpMV scalar kernel warpper
 * 
 * @tparam T data type
 * @param coo_data input COO data
 * @param coo_row_indices input COO row indices
 * @param coo_col_indices input COO column indices
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 * @param nnz number of non-zero elements 
 */
template <typename T>
void spmv_coo_segement0(
    const T *__restrict__ coo_data, const int *__restrict__ coo_row_indices,
    const int *__restrict__ coo_col_indices, const T *__restrict__ input_vec,
    T *__restrict__ output_vec, const int m, const int k, const int nnz);

/**
 * @brief COO SpMV scalar kernel warpper
 * 
 * @tparam T data type
 * @param coo_data input COO data
 * @param coo_row_indices input COO row indices
 * @param coo_col_indices input COO column indices
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 * @param nnz number of non-zero elements 
 */
template <typename T>
void spmv_coo_segement_naive(
    const T *__restrict__ coo_data, const int *__restrict__ coo_row_indices,
    const int *__restrict__ coo_col_indices, const T *__restrict__ input_vec,
    T *__restrict__ output_vec, const int m, const int k, const int nnz);

/**
 * @brief COO SpMV vector kernel warpper
 * 
 * @tparam T data type
 * @param coo_data input COO data
 * @param coo_row_indices input COO row indices
 * @param coo_col_indices input COO column indices
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 * @param nnz number of non-zero elements 
 */
template <typename T>
void compute_spmv_coo_segment(
    const T *__restrict__ coo_data, const int *__restrict__ coo_row_indices,
    const int *__restrict__ coo_col_indices, const T *__restrict__ input_vec,
    T *__restrict__ output_vec, const int m, const int k, const int nnz);

/**
 * @brief COO SpMV vector kernel warpper
 * 
 * @tparam T data type
 * @param coo_data input COO data
 * @param coo_row_indices input COO row indices
 * @param coo_col_indices input COO column indices
 * @param input_vec input vector
 * @param output_vec output vector
 * @param m number of rows
 * @param k number of columns
 * @param nnz number of non-zero elements 
 */
template <typename T>
void compute_spmv_coo_segment_naive(
    const T *__restrict__ coo_data, const int *__restrict__ coo_row_indices,
    const int *__restrict__ coo_col_indices, const T *__restrict__ input_vec,
    T *__restrict__ output_vec, const int m, const int k, const int nnz);

#endif  // SPMV_COO_KERNEL_H