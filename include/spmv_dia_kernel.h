#ifndef SPMV_DIA_KERNEL_H
#define SPMV_DIA_KERNEL_H

// #include <cuda_runtime.h>

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
void mat2dia(const T * __restrict__ mat, T * __restrict__ * dia_data, int * __restrict__ * dia_offsets, int & ndiags, const int m, const int k, const int lda);

// template<typename T>
// __global__ void spmv_dia_kernel0();

#endif