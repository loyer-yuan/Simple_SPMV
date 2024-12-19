#include <cstdio>
#include <random>
#include <utils.h>

#include "spmv_cpu_kernel.h"
#include "spmv_gpu_kernel.h"

#define DEFAULT_M 10
#define DEFAULT_K 10

/**
 * @brief Main function
 *
 * @param argc 1 or 3, when 3, rest arguments are the M and K values
 * @param argv M and K values, or none
 */
int main(int argc, char** argv) {
  const int M = (argc != 3) ? DEFAULT_M : atoi(argv[1]);
  const int K = (argc != 3) ? DEFAULT_K : atoi(argv[2]);
  const int N = M * K;
  printf("M = %d, K = %d\n", M, K);

  float* mat_cpu = new float[N];
  prepare_data<float, true>(mat_cpu, N);
  printf("\nmat_cpu:\n");
  print_data(mat_cpu, M, K);

  float* vec_cpu = new float[K];
  prepare_data<float, true>(vec_cpu, K, 1.0);
  printf("\nvec_cpu:\n");
  print_data(vec_cpu, 1, K);

  float* out_cpu = new float[M];
  spmv_cpu_kernel0(mat_cpu, vec_cpu, out_cpu, M, K);
  printf("\nout_cpu:\n");
  print_data(out_cpu, 1, M);

  const float* cmat_cpu = mat_cpu;
  const float* cvec_cpu = vec_cpu;
  const float* cout_cpu = out_cpu;

  // ---------------------------------------------------------
  // DIA format
  // ---------------------------------------------------------
  if (M == K) {
    printf("\n===== DIA format =====\n");

    float* dia_data;
    int* dia_offsets;
    int ndiags = 0;
    mat2dia<float>(cmat_cpu, dia_data, dia_offsets, ndiags, M, K, K);
    float* out_dia = new float[M];
    compute_spmv_dia(dia_data, dia_offsets, cvec_cpu, out_dia, ndiags, M, K);

    printf("out_dia:\n");
    print_data(out_dia, 1, M);

    // Compare the results
    int count = 0;
    for (int i = 0; i < M; ++i) {
      if (std::abs(cout_cpu[i] - out_dia[i]) > 1e-6 && count++ < 10) {
        printf("Results mismatch at %d: %f != %f\n", i, cout_cpu[i], out_dia[i]);
      }
    }
    if (count == 0) {
      printf("Results match!\n");
    } else {
      printf("Results mismatched %d times in total.\n", count);
    }

    delete[] dia_data;
    delete[] dia_offsets;
    delete[] out_dia;
  } else {
    printf("\nDIA format is not supported for non-square matrices.\n");
  }
  // ---------------------------------------------------------


  // ---------------------------------------------------------
  // ELL format
  // ---------------------------------------------------------
  printf("\n===== ELL format =====\n");
  // Transfer the matrix to ELL format
  float* ell_data;
  int* ell_indices;
  int max_nnz_per_row = 0;
  mat2ell<float>(cmat_cpu, ell_data, ell_indices, max_nnz_per_row, M, K, K);
  
  // Compute the SpMV
  float* out_ell = new float[M];
  compute_spmv_ell(ell_data, ell_indices, cvec_cpu, out_ell, max_nnz_per_row, M, K);

  printf("out_ell:\n");
  print_data(out_ell, 1, M);

  // Compare the results
  int count = 0;
  for (int i = 0; i < M; ++i) {
    if (std::abs(cout_cpu[i] - out_ell[i]) > 1e-6 && count++ < 10) {
      printf("\nResults mismatch at %d: %f != %f\n", i, cout_cpu[i], out_ell[i]);
    }
  }
  if (count == 0) {
    printf("Results match!\n");
  } else {
    printf("Results mismatched %d times in total.\n", count);
  }
  
  // clean up
  delete[] ell_data;
  delete[] ell_indices;
  delete[] out_ell;
  // ---------------------------------------------------------


  // clean up
  delete[] mat_cpu;
  delete[] vec_cpu;
  delete[] out_cpu;

  return 0;
}
