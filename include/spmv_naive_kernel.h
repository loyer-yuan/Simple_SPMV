#ifndef SPMV_NAIVE_KERNEL_H
#define SPMV_NAIVE_KERNEL_H


/**
 * @brief Naive CPU kernel for matrix-vector multiplication.
 *
 * @tparam T Data type
 * @param mat Matrix
 * @param vec Vector
 * @param out Output vector
 * @param M Number of rows
 * @param K Number of columns
 * @note Just for reference
 */
template <typename T>
void spmv_naive_kernel0(const T* __restrict__ mat, const T* __restrict__ vec, T* __restrict__ out, const int M, const int K)
{
  for (int i = 0; i < M; i++) {
    T sum = 0;
    for (int j = 0; j < K; j++) {
      sum += mat[i * K + j] * vec[j];
    }
    out[i] = sum;
  }
}


#endif // SPMV_NAIVE_KERNEL_H