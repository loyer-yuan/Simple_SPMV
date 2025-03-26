#ifndef SPMV_DIA_KERNEL_H
#define SPMV_DIA_KERNEL_H

#include <cassert>
#include <cstring>

#define assertm(exp, msg) assert((void(msg), exp))

/**
 * @brief Convert a matrix to DIA format
 *
 * @tparam T data type
 * @param mat input matrix
 * @param dia_data output data array
 * @param dia_offsets output offsets array
 * @param ndiags output number of diagonals
 * @param m input number of rows
 * @param k intput number of columns
 * @param lda input leading dimension, default is k, means row-major
 *
 * @todo m, k can be used as template elements
 */
template<typename T>
void mat2dia(const T* __restrict__ mat, T* __restrict__& dia_data, int* __restrict__& dia_offsets, int& ndiags, const int m, const int k, const int lda)
{
    assertm(m > 0 && k > 0, "Invalid matrix size, size of matrix must be greater than 0");
    assertm(m == lda && k == lda, "Only support square matrix");

    const int max_ndiags = 2 * m - 1;
    T** diags = new T * [max_ndiags];
    bool* is_diagonal = new bool[max_ndiags];
    for (int i = 0; i < max_ndiags; i++) {
        is_diagonal[i] = false;
    }
    int count_diags = 0;
    // Count the number of diagonals and store the diagonals
    for (int i = 0; i < m; i++) {
        int i_offset = i * lda;
        for (int j = 0; j < k; j++) {
            const int new_idx = j - i + (m - 1);
            const T value = mat[i_offset + j];
            // Check if the element is non-zero or it is on the diagonal
            if (value != 0 || is_diagonal[new_idx] == true) {
                // First time we see this diagonal
                if (!is_diagonal[new_idx]) {
                    is_diagonal[new_idx] = true;
                    diags[new_idx] = new T[m];
                    std::memset(diags[new_idx], 0, m * sizeof(T)); // TODO: Can be removed for better performance
                    count_diags++;
                }
                diags[new_idx][i] = value;
            }
        }
    }

    // Store the diagonals in the DIA format
    ndiags = count_diags;
    dia_offsets = new int[ndiags];
    dia_data = new T[ndiags * m];
    int true_idx = 0;
    for (int i = 0; i < max_ndiags; i++) {
        if (!is_diagonal[i]) {
            continue;
        }
        dia_offsets[true_idx] = i - (m - 1);
        // Store the diagonal with m-major order, note: k==m
        for (int j = 0; j < m; j++) {
            dia_data[true_idx * m + j] = diags[i][j];
        }
        true_idx++;
    }

    // Clean up
    for (int i = 0; i < max_ndiags; i++) {
        if (is_diagonal[i]) {
            delete[] diags[i];
        }
    }
    delete[] diags;
    delete[] is_diagonal;
}

/**
 * @brief DIA SpMV Entry. Including memory allocation and data transfer
 *
 * @tparam T data type
 * @param dia_data input matrix's DIA format data array
 * @param dia_offsets input matrix's DIA format offsets array
 * @param vec input vector
 * @param out output vector
 * @param ndiags intput number of diagonals
 * @param m input number of rows
 * @param k input number of columns
 */
template<typename T>
void compute_spmv_dia(const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k);


#endif // SPMV_DIA_KERNEL_H
