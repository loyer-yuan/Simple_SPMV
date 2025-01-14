#include "spmv_dia_kernel.h"
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

#define assertm(exp, msg) assert((void(msg), exp))

template <typename T>
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
// Instantiate the template
template void mat2dia<float>(const float* __restrict__ mat, float* __restrict__& dia_data, int* __restrict__& dia_offsets, int& ndiags, const int m, const int k, const int lda);


/**
 * @brief DIA SpMV kernel
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
template <typename T>
__global__ void spmv_dia_kernel0(const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    T sum = 0;
    for (int i = 0; i < ndiags; i++) {
      const int col = row + dia_offsets[i];
      const T value = dia_data[i * m + row];
      if (col >= 0 && col < k) {
        sum += value * vec[col];
      }
    }
    out[row] = sum;
  }
}


template <typename T>
void spmv_dia0(const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k) {
  const int block_size = 128;
  const int n_blocks = (m + block_size - 1) / block_size;
  spmv_dia_kernel0<T> << <n_blocks, block_size >> > (dia_data, dia_offsets, vec, out, ndiags, m, k);
}
// Instantiate the template
template void spmv_dia0<float>(const float* __restrict__ dia_data, const int* __restrict__ dia_offsets, const float* __restrict__ vec, float* __restrict__ out, const int ndiags, const int m, const int k);


template <typename T>
void compute_spmv_dia(const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k) {
  // Prepare the device data
  float* dia_data_gpu;
  int* dia_offsets_gpu;
  float* vec_gpu;
  float* out_gpu;
  cudaMalloc(&dia_data_gpu, ndiags * m * sizeof(float));
  cudaMalloc(&dia_offsets_gpu, ndiags * sizeof(int));
  cudaMalloc(&vec_gpu, k * sizeof(float));
  cudaMalloc(&out_gpu, m * sizeof(float));
  cudaMemcpy(dia_data_gpu, dia_data, ndiags * m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dia_offsets_gpu, dia_offsets, ndiags * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_gpu, vec, k * sizeof(float), cudaMemcpyHostToDevice);

  // Run the DIA SpMV kernel
  spmv_dia0<float>(dia_data_gpu, dia_offsets_gpu, vec_gpu, out_gpu, ndiags, m, k);

  cudaMemcpy(out, out_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(dia_data_gpu);
  cudaFree(dia_offsets_gpu);
  cudaFree(vec_gpu);
  cudaFree(out_gpu);
}
// Instantiate the template
template void compute_spmv_dia<float>(const float* __restrict__ dia_data, const int* __restrict__ dia_offsets, const float* __restrict__ vec, float* __restrict__ out, const int ndiags, const int m, const int k);