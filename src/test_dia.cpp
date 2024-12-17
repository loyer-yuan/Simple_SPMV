#include "spmv_dia_kernel.h"
#include "spmv_cpu_kernel.h"
#include "utils.h"
#include <cstdio>
#include <cstring>

int main() {
  const int M = 10;
  const int K = M;
  const int NUM = M * K;
  const int DIAGS_NUM = 3;
  printf("M = %d, K = %d\n", M, K);

  // Prepare data
  float* mat_cpu = new float[NUM];
  printf("\nmat_cpu:\n");
  // Initialize mat_cpu
  std::memset(mat_cpu, 0, NUM * sizeof(float));
  random_diags_gen<float, true>(mat_cpu, M, DIAGS_NUM);
  print_data(mat_cpu, M, K);

  float* vec_cpu = new float[K];
  printf("\nvec_cpu:\n");
  prepare_data<float>(vec_cpu, K, 1.0f);
  print_data(vec_cpu, 1, K);

  // Convert the matrix to DIA format
  float* dia_data;
  int* dia_offsets;
  int ndiags = 0;
  mat2dia<float>(mat_cpu, dia_data, dia_offsets, ndiags, M, K, K);
  printf("\ndia_data:\n");
  print_data(dia_data, ndiags, M);
  printf("\ndia_offsets:\n");
  print_data(dia_offsets, 1, ndiags);

  // Run the CPU SpMV kernel
  float* out_cpu = new float[M];
  printf("\nout_cpu:\n");
  spmv_cpu_kernel0(mat_cpu, vec_cpu, out_cpu, M, K);
  print_data(out_cpu, 1, M);

  // Run the DIA SpMV kernel
  float* out_gpu_cpu = new float[M];
  compute_spmv_dia<float>(dia_data, dia_offsets, vec_cpu, out_gpu_cpu, ndiags, M, K);

  printf("\nout_gpu_cpu:\n");
  print_data(out_gpu_cpu, 1, M);

  int count = 0;
  // Compare the results
  for (int i = 0; i < M; ++i) {
    if (std::abs(out_cpu[i] - out_gpu_cpu[i]) > 1e-6 && count++ < 10) {
      printf("Results mismatch at %d: %f != %f\n", i, out_cpu[i], out_gpu_cpu[i]);
    }
  }
  if (count == 0) {
    printf("Results match!\n");
  } else {
    printf("Results mismatched %d times in total.\n", count);
  }

  // Clean up
  delete[] out_gpu_cpu;

  delete[] dia_data;
  delete[] dia_offsets;

  delete[] mat_cpu;
  delete[] vec_cpu;
  delete[] out_cpu;
}