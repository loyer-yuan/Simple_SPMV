#include "spmv_dia_kernel.h"
#include <cassert>
#include <cstring>
#define assertm(exp, msg) assert((void(msg), exp))

#include <cstdio>

template <typename T>
void mat2dia(const T * __restrict__ mat, T * __restrict__ & dia_data, int * __restrict__ & dia_offsets, int & ndiags, const int m, const int k, const int lda)
{
    assertm(m > 0 && k > 0, "Invalid matrix size, size of matrix must be greater than 0");
    assertm(m == lda && k == lda, "Only support square matrix");

    const int max_ndiags = 2*m-1;
    T* *diags = new T*[max_ndiags];
    bool* is_diagonal = new bool[max_ndiags];
    for (int i  = 0; i < max_ndiags; i++) {
        is_diagonal[i] = false;
    }
    int count_diags = 0;
    // Count the number of diagonals and store the diagonals
    for (int i = 0; i < m; i++) {
        int i_offset = i * lda;
        for (int j = 0; j < k; j++) {
            const int new_idx = j - i + (m-1);
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
        dia_offsets[true_idx] = i - (m-1);
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

// Instantiate the template
template void mat2dia<float>(const float * __restrict__ mat, float * __restrict__ & dia_data, int * __restrict__ & dia_offsets, int & ndiags, const int m, const int k, const int lda);