#include "spmv_csr_kernel.h"
#include <vector>
#include <cstring>
#include <cassert>

#define assertm(exp, msg) assert((void(msg), exp))

using std::vector;

template <typename T>
void mat2csr(const T * __restrict__ mat, T * __restrict__ &csr_data, int * __restrict__ &csr_ptr, int * __restrict__ &csr_indices, const int m, const int k, const int lda) {
  assertm(m > 0 && k > 0, "Invalid matrix size, size of matrix must be greater than 0");
  assertm(k == lda, "Only support k == lda");
  
  vector<T> tdata;
  vector<int> tindices;
  csr_ptr = new int[m + 1];
  
  csr_ptr[0] = 0;
  int nnz_per_row = 0;
  for (int i = 0; i < m; i++) {
    nnz_per_row = 0;
    for (int j = 0; j < k; j++) {
      const T val = mat[i * lda + j];
      if (val != 0) {
        tdata.push_back(val);
        tindices.push_back(j);
        nnz_per_row++;
      }
    }
    csr_ptr[i + 1] = csr_ptr[i] + nnz_per_row;
  }

  // Copy to csr_data, csr_indices
  csr_data = new T[tdata.size()];
  csr_indices = new int[tindices.size()];
  std::memcpy(csr_data, tdata.data(), sizeof(T) * tdata.size());
  std::memcpy(csr_indices, tindices.data(), sizeof(int) * tindices.size());
}
// Instantiate the template
template void mat2csr<float>(const float * __restrict__ mat, float * __restrict__ &csr_data, int * __restrict__ &csr_ptr, int * __restrict__ &csr_indices, const int m, const int k, const int lda);


/**
 * @brief CSR SpMV scalar kernel. Each thread computes one row.
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
__global__ void spmv_csr_scalar_kernel0(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    float sum = 0;

    int start = csr_ptr[row];
    int end = csr_ptr[row + 1];
    for (int i = start; i < end; i++) {
      sum += csr_data[i] * input_vec[csr_indices[i]];
    }

    output_vec[row] = sum;
  }
}


/**
 * @brief CSR SpMV vector kernel. Each warp computes one row.
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
__global__ void spmv_csr_vector_kernel0(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane = tid & (32 - 1); // is equal to "% 32"

  int row = warp_id;
  if (row < m) {
    int start = csr_ptr[row];
    int end = csr_ptr[row + 1];

    float val = 0;
    for (int i = start + lane; i < end; i += 32) {
      val += csr_data[i] * input_vec[csr_indices[i]];
    }

    // Reduce within warp with shuffle
    if(blockDim.x >= 32) val += __shfl_down_sync(0xffffffff, val, 16);
    if(blockDim.x >= 16) val += __shfl_down_sync(0xffffffff, val, 8);
    if(blockDim.x >= 8) val += __shfl_down_sync(0xffffffff, val, 4);
    if(blockDim.x >= 4) val += __shfl_down_sync(0xffffffff, val, 2);
    if(blockDim.x >= 2) val += __shfl_down_sync(0xffffffff, val, 1);

    if (lane == 0) {
      output_vec[row] = val;
    }
  }
}


template <typename T>
void spmv_csr_scalar0(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k) {
  int block_size = 256;
  int grid_size = (m + block_size - 1) / block_size;

  spmv_csr_scalar_kernel0<T><<<grid_size, block_size>>>(csr_data, csr_ptr, csr_indices, input_vec, output_vec, m, k);
}
// Instantiate the template
template void spmv_csr_scalar0<float>(const float * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const float * __restrict__ input_vec, float * __restrict__ output_vec, const int m, const int k);


template <typename T>
void spmv_csr_vector0(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ input_vec, T * __restrict__ output_vec, const int m, const int k) {
  int block_size = 256;
  int warp_data_size = 256/32;
  int grid_size = (m + warp_data_size - 1) / warp_data_size;

  spmv_csr_vector_kernel0<T><<<grid_size, block_size>>>(csr_data, csr_ptr, csr_indices, input_vec, output_vec, m, k);
}
// Instantiate the template
template void spmv_csr_vector0<float>(const float * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const float * __restrict__ input_vec, float * __restrict__ output_vec, const int m, const int k);


template <typename T>
void compute_spmv_csr_scalar(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ vec, T * __restrict__ out, const int m, const int k) {
  // Prepare the device data
  float *d_csr_data;
  int *d_csr_ptr;
  int *d_csr_indices;
  float *d_input_vec;
  float *d_output_vec;
  cudaMalloc(&d_csr_data, csr_ptr[m] * sizeof(T));
  cudaMalloc(&d_csr_ptr, (m + 1) * sizeof(int));
  cudaMalloc(&d_csr_indices, csr_ptr[m] * sizeof(int));
  cudaMalloc(&d_input_vec, k * sizeof(T));
  cudaMalloc(&d_output_vec, m * sizeof(T));
  cudaMemcpy(d_csr_data, csr_data, csr_ptr[m] * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_ptr, csr_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_indices, csr_indices, csr_ptr[m] * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vec, vec, k * sizeof(T), cudaMemcpyHostToDevice);

  // Run the CSR SpMV scalar kernel
  spmv_csr_scalar0(d_csr_data, d_csr_ptr, d_csr_indices, d_input_vec, d_output_vec, m, k);

  cudaMemcpy(out, d_output_vec, m * sizeof(T), cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(d_csr_data);
  cudaFree(d_csr_ptr);
  cudaFree(d_csr_indices);
  cudaFree(d_input_vec);
  cudaFree(d_output_vec);
}
// Instantiate the template
template void compute_spmv_csr_scalar<float>(const float * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const float * __restrict__ vec, float * __restrict__ out, const int m, const int k);


template <typename T>
void compute_spmv_csr_vector(const T * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const T * __restrict__ vec, T * __restrict__ out, const int m, const int k) {
  // Prepare the device data
  float *d_csr_data;
  int *d_csr_ptr;
  int *d_csr_indices;
  float *d_input_vec;
  float *d_output_vec;
  cudaMalloc(&d_csr_data, csr_ptr[m] * sizeof(T));
  cudaMalloc(&d_csr_ptr, (m + 1) * sizeof(int));
  cudaMalloc(&d_csr_indices, csr_ptr[m] * sizeof(int));
  cudaMalloc(&d_input_vec, k * sizeof(T));
  cudaMalloc(&d_output_vec, m * sizeof(T));
  cudaMemcpy(d_csr_data, csr_data, csr_ptr[m] * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_ptr, csr_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_indices, csr_indices, csr_ptr[m] * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vec, vec, k * sizeof(T), cudaMemcpyHostToDevice);

  // Run the CSR SpMV vector kernel
  spmv_csr_vector0(d_csr_data, d_csr_ptr, d_csr_indices, d_input_vec, d_output_vec, m, k);

  cudaMemcpy(out, d_output_vec, m * sizeof(T), cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(d_csr_data);
  cudaFree(d_csr_ptr);
  cudaFree(d_csr_indices);
  cudaFree(d_input_vec);
  cudaFree(d_output_vec);
}
// Instantiate the template
template void compute_spmv_csr_vector<float>(const float * __restrict__ csr_data, const int * __restrict__ csr_ptr, const int * __restrict__ csr_indices, const float * __restrict__ vec, float * __restrict__ out, const int m, const int k);