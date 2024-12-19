#include "spmv_ell_kernel.h"
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cassert>

#define assertm(exp, msg) assert((void(msg), exp))

using std::vector;
using std::unique_ptr;

template <typename T>
void mat2ell(const T* __restrict__ mat, T* __restrict__& ell_data, int* __restrict__& ell_offsets, int& max_nnz_per_row, const int m, const int k, const int lda)
{
  assertm(m > 0 && k > 0, "Invalid matrix size, size of matrix must be greater than 0");
  assertm(k == lda, "Only support k == lda");

  unique_ptr<vector<T>[]> ell_data_row_ptrs = std::make_unique<vector<T>[]>(m);
  unique_ptr<vector<int>[]> ell_offsets_row_ptrs = std::make_unique<vector<int>[]>(m);
  int tmax_nnz_per_row = 0;

  for (int i = 0; i < m; i++) {
    int nnz = 0;
    for (int j = 0; j < k; j++) {
      if (mat[i * lda + j] != 0) {
        ell_data_row_ptrs[i].push_back(mat[i * lda + j]);
        ell_offsets_row_ptrs[i].push_back(j);
        nnz++;
      }
    }
    tmax_nnz_per_row = std::max(tmax_nnz_per_row, nnz);
  }

  T* tell_data = new T[m * tmax_nnz_per_row];
  memset(tell_data, 0, m * tmax_nnz_per_row * sizeof(T));
  int* tell_offsets = new int[m * tmax_nnz_per_row];
  memset(tell_offsets, 0, m * tmax_nnz_per_row * sizeof(int));

  // TODO: can be optimized
  for (int i = 0; i < m; i++) {
    int size = ell_data_row_ptrs[i].size();
    for (int j = 0; j < size; j++) {
      tell_data[j * m + i] = ell_data_row_ptrs[i].at(j);
      tell_offsets[j * m + i] = ell_offsets_row_ptrs[i].at(j);
    }
  }

  ell_data = tell_data;
  ell_offsets = tell_offsets;
  max_nnz_per_row = tmax_nnz_per_row;
}
// Instantiate the template
template void mat2ell<float>(const float* __restrict__ mat, float* __restrict__& ell_data, int* __restrict__& ell_offsets, int& max_nnz_per_row, const int m, const int k, const int lda);


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
__global__ void spmv_ell_kernel0(const float* __restrict__ ell_data, const int* __restrict__ ell_indices, const float* __restrict__ input_vec, float* __restrict__ output_vec, const int max_nnz_per_row, const int m, const int k)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < m) {
    float sum = 0;
    for (int i = 0; i < max_nnz_per_row; i++) {
      int offset = ell_indices[i * m + row];
      T val = ell_data[i * m + row];
      if (val != 0) {
        sum += val * input_vec[offset];
      }
    }
    output_vec[row] = sum;
  }
}


template <typename T>
void spmv_ell0(const T* __restrict__ ell_data, const int* __restrict__ ell_indices, const T* __restrict__ input_vec, T* __restrict__ output_vec, const int max_nnz_per_row, const int m, const int k)
{
  int block_size = 256;
  int grid_size = (m + block_size - 1) / block_size;

  spmv_ell_kernel0<T><<<grid_size, block_size>>>(ell_data, ell_indices, input_vec, output_vec, max_nnz_per_row, m, k);
}
// Instantiate the template
template void spmv_ell0<float>(const float* __restrict__ ell_data, const int* __restrict__ ell_indices, const float* __restrict__ input_vec, float* __restrict__ output_vec, const int max_nnz_per_row, const int m, const int k);


template <typename T>
void compute_spmv_ell(const T* __restrict__ ell_data, const int* __restrict__ ell_indices, const T* __restrict__ vec, T* __restrict__ out, const int max_nnz_per_row, const int m, const int k)
{
  // Prepare the device data
  float* d_ell_data;
  int* d_ell_indices;
  float* d_input_vec;
  float* d_output_vec;
  cudaMalloc(&d_ell_data, m * max_nnz_per_row * sizeof(T));
  cudaMalloc(&d_ell_indices, m * max_nnz_per_row * sizeof(int));
  cudaMalloc(&d_input_vec, k * sizeof(T));
  cudaMalloc(&d_output_vec, m * sizeof(T));
  cudaMemcpy(d_ell_data, ell_data, m * max_nnz_per_row * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ell_indices, ell_indices, m * max_nnz_per_row * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vec, vec, k * sizeof(T), cudaMemcpyHostToDevice);

  // Run the ELL SpMV kernel
  spmv_ell0(d_ell_data, d_ell_indices, d_input_vec, d_output_vec, max_nnz_per_row, m, k);

  cudaMemcpy(out, d_output_vec, m * sizeof(T), cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(d_ell_data);
  cudaFree(d_ell_indices);
  cudaFree(d_input_vec);
  cudaFree(d_output_vec);
}
// Instantiate the template
template void compute_spmv_ell<float>(const float* __restrict__ ell_data, const int* __restrict__ ell_indices, const float* __restrict__ vec, float* __restrict__ out, const int max_nnz_per_row, const int m, const int k);