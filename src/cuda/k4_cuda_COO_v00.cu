#include "spmv_coo_kernel.h"
#include <vector>
#include <cstring>
#include <cassert>

#define assertm(exp, msg) assert((void(msg), exp))

using std::vector;


template <typename T>
__device__ __forceinline__ void atomicAdd_warp(T* address, T val) {
  atomicAdd(address, val);
}


/**
 * @brief COO SpMV segment kernel. Each thread computes one element and each warp's computation may cross multiple rows.
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
__global__ void spmv_coo_segement_kernel0(const T * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k, const int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const int warp_id = tid / 32;
  const int lane = tid & 31;

  if (tid < nnz) {
    int row_idx = coo_row_indices[tid];
    int col_idx = coo_col_indices[tid];
    T result = coo_data[tid] * input_vec[col_idx];

    // reduction with shuffle
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      int down_row_idx = __shfl_down_sync(0xffffffff, row_idx, offset);
      T down_result = __shfl_down_sync(0xffffffff, result, offset);
      if (lane + offset < 32 && down_row_idx == row_idx) {
        result += down_result;
      }
    }
    
    // write result
    bool store_result = true;
    int up_row_idx = __shfl_up_sync(0xffffffff, row_idx, 1);
    if (lane - 1 >= 0 && up_row_idx == row_idx) {
      store_result = false;
    }
    if (store_result) {
      atomicAdd_warp(&output_vec[row_idx], result);
    }
  }
}


/**
 * @brief COO SpMV segment kernel. Each thread computes one element and each warp's computation may cross multiple rows.
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
template <typename T, int BLOCK_SIZE>
__global__ void spmv_coo_segement_kernel_naive(const T * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k, const int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const int warp_id = tid / 32;
  const int lane = tid & 31;
  __shared__ T shared_data[BLOCK_SIZE];
  __shared__ int shared_row_indices[BLOCK_SIZE];

  if (tid < nnz) {
    int row_idx = coo_row_indices[tid];
    int col_idx = coo_col_indices[tid];
    T result = coo_data[tid] * input_vec[col_idx];

    shared_data[threadIdx.x] = result;
    shared_row_indices[threadIdx.x] = row_idx;

    // reduction with shared memory
    if (lane >= 1 && row_idx == shared_row_indices[threadIdx.x - 1]) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x - 1];
    }
    if (lane >= 2 && row_idx == shared_row_indices[threadIdx.x - 2]) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x - 2];
    }
    if (lane >= 4 && row_idx == shared_row_indices[threadIdx.x - 4]) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x - 4];
    }
    if (lane >= 8 && row_idx == shared_row_indices[threadIdx.x - 8]) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x - 8];
    }
    if (lane >= 16 && row_idx == shared_row_indices[threadIdx.x - 16]) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x - 16];
    }
    
    // write result
    bool store_result = true;
    if (lane + 1 < 32 && row_idx == shared_row_indices[threadIdx.x + 1]) {
      store_result = false;
    }
    if (store_result) {
      atomicAdd_warp(&output_vec[row_idx], shared_data[threadIdx.x]);
    }
  }
}


template <typename T>
void spmv_coo_segement0(const T * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k, const int nnz) {
  const int block_size = 256;
  const int grid_size = (nnz + block_size - 1) / block_size;
  spmv_coo_segement_kernel0<T><<<grid_size, block_size>>>(coo_data, coo_row_indices, coo_col_indices, input_vec, output_vec, m, k, nnz);
}
// Instantiate the template
template void spmv_coo_segement0<float>(const float * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const float * __restrict__ input_vec, float * __restrict__ output_vec, const int m, const int k, const int nnz);


template <typename T>
void spmv_coo_segement_naive(const T * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k, const int nnz) {
  const int block_size = 256;
  const int grid_size = (nnz + block_size - 1) / block_size;
  spmv_coo_segement_kernel_naive<T, block_size><<<grid_size, block_size>>>(coo_data, coo_row_indices, coo_col_indices, input_vec, output_vec, m, k, nnz);
}
// Instantiate the template
template void spmv_coo_segement_naive<float>(const float * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const float * __restrict__ input_vec, float * __restrict__ output_vec, const int m, const int k, const int nnz);


template <typename T>
void compute_spmv_coo_segment(const T * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k, const int nnz) {
  // Prepare device memory
  T *d_coo_data;
  int *d_coo_row_indices;
  int *d_coo_col_indices;
  T *d_input_vec;
  T *d_output_vec;
  cudaMalloc(&d_coo_data, sizeof(T) * nnz);
  cudaMalloc(&d_coo_row_indices, sizeof(int) * nnz);
  cudaMalloc(&d_coo_col_indices, sizeof(int) * nnz);
  cudaMalloc(&d_input_vec, sizeof(T) * k);
  cudaMalloc(&d_output_vec, sizeof(T) * m);
  cudaMemcpy(d_coo_data, coo_data, sizeof(T) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_row_indices, coo_row_indices, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_col_indices, coo_col_indices, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vec, input_vec, sizeof(T) * k, cudaMemcpyHostToDevice);

  // Call kernel
  spmv_coo_segement0(d_coo_data, d_coo_row_indices, d_coo_col_indices, d_input_vec, d_output_vec, m, k, nnz);

  // Copy result back
  cudaMemcpy(output_vec, d_output_vec, sizeof(T) * m, cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_coo_data);
  cudaFree(d_coo_row_indices);
  cudaFree(d_coo_col_indices);
  cudaFree(d_input_vec);
  cudaFree(d_output_vec);
}
// Instantiate the template
template void compute_spmv_coo_segment<float>(const float * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const float * __restrict__ input_vec, float * __restrict__ output_vec, const int m, const int k, const int nnz);


template <typename T>
void compute_spmv_coo_segment_naive(const T * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k, const int nnz) {
  // Prepare device memory
  T *d_coo_data;
  int *d_coo_row_indices;
  int *d_coo_col_indices;
  T *d_input_vec;
  T *d_output_vec;
  cudaMalloc(&d_coo_data, sizeof(T) * nnz);
  cudaMalloc(&d_coo_row_indices, sizeof(int) * nnz);
  cudaMalloc(&d_coo_col_indices, sizeof(int) * nnz);
  cudaMalloc(&d_input_vec, sizeof(T) * k);
  cudaMalloc(&d_output_vec, sizeof(T) * m);
  cudaMemcpy(d_coo_data, coo_data, sizeof(T) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_row_indices, coo_row_indices, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_col_indices, coo_col_indices, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vec, input_vec, sizeof(T) * k, cudaMemcpyHostToDevice);

  // Call kernel
  spmv_coo_segement_naive(d_coo_data, d_coo_row_indices, d_coo_col_indices, d_input_vec, d_output_vec, m, k, nnz);

  // Copy result back
  cudaMemcpy(output_vec, d_output_vec, sizeof(T) * m, cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_coo_data);
  cudaFree(d_coo_row_indices);
  cudaFree(d_coo_col_indices);
  cudaFree(d_input_vec);
  cudaFree(d_output_vec);
}
// Instantiate the template
template void compute_spmv_coo_segment_naive<float>(const float * __restrict__ coo_data, const int * __restrict__ coo_row_indices, const int * __restrict__ coo_col_indices, const float * __restrict__ input_vec, float * __restrict__ output_vec, const int m, const int k, const int nnz);