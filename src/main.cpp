#include <cstdio>
#include <random>
#include <utils.h>

#include "spmv_cpu_kernel.h"
#include "spmv_gpu_kernel.h"

#define DEFAULT_M 10
#define DEFAULT_K 10
#define DEFAULT_NNZ 2

/**
 * @brief Main function
 *
 * @param argc 1 or 3, when 3, rest arguments are the M and K values
 * @param argv M and K values, or none
 */
int main(int argc, char** argv) {
  const int M = (argc != 4) ? DEFAULT_M : atoi(argv[1]);
  const int K = (argc != 4) ? DEFAULT_K : atoi(argv[2]);
  const int NNZ = (argc != 4) ? DEFAULT_NNZ : atoi(argv[3]);
  const int N = M * K;
  const float prob = (float)NNZ / (float)N;
  printf("M = %d, K = %d, NNZ = %d, prob = %f\n", M, K, NNZ, prob);

  bool print_out = true;
  if (M >= 15) {
    printf("When M >= 15, data isn't printed out.\n");
    print_out = false;
  }

  float* mat_cpu = new float[N];
  prepare_data<float, false>(mat_cpu, N, prob);
  if (print_out) {
    printf("\nmat_cpu:\n");
    print_data(mat_cpu, M, K);
  }

  float* vec_cpu = new float[K];
  prepare_data<float, false>(vec_cpu, K, 1.0);
  if (print_out) {
    printf("\nvec_cpu:\n");
    print_data(vec_cpu, 1, K);
  }

  float* out_cpu = new float[M];
  spmv_cpu_kernel0(mat_cpu, vec_cpu, out_cpu, M, K);
  if (print_out) {
    printf("\nout_cpu:\n");
    print_data(out_cpu, 1, M);
  }

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
    if (print_out) print_data(out_dia, 1, M);

    // Compare the results
    check_result<float>(cout_cpu, out_dia, M);

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
  if (print_out) print_data(out_ell, 1, M);

  // Compare the results
  check_result<float>(cout_cpu, out_ell, M);
  
  // clean up
  delete[] ell_data;
  delete[] ell_indices;
  delete[] out_ell;
  // ---------------------------------------------------------


  // ---------------------------------------------------------
  // CSR format
  // ---------------------------------------------------------
  printf("\n===== CSR format =====\n");
  // Transfer the matrix to CSR format
  float* csr_data;
  int* csr_ptr;
  int* csr_indices;
  mat2csr<float>(cmat_cpu, csr_data, csr_ptr, csr_indices, M, K, K);

  // Compute the SpMV using scalar kernel
  float* out_csr_scalar = new float[M];
  compute_spmv_csr_scalar(csr_data, csr_ptr, csr_indices, cvec_cpu, out_csr_scalar, M, K);
  float* out_csr_vector = new float[M];
  compute_spmv_csr_vector(csr_data, csr_ptr, csr_indices, cvec_cpu, out_csr_vector, M, K);

  printf("out_csr_scalar:\n");
  if (print_out) print_data(out_csr_scalar, 1, M);
  // Compare the results
  check_result<float>(cout_cpu, out_csr_scalar, M);

  printf("out_csr_vector:\n");
  if (print_out) print_data(out_csr_vector, 1, M);
  // Compare the results
  check_result<float>(cout_cpu, out_csr_vector, M);

  // clean up
  delete[] csr_data;
  delete[] csr_ptr;
  delete[] csr_indices;
  delete[] out_csr_scalar;
  delete[] out_csr_vector;
  // ---------------------------------------------------------

  // ---------------------------------------------------------
  // COO format
  // ---------------------------------------------------------
  printf("\n===== COO format =====\n");
  // Transfer the matrix to COO format
  float* coo_data;
  int* coo_row_indices;
  int* coo_col_indices;
  int nnz = 0;
  mat2coo<float>(cmat_cpu, coo_data, coo_row_indices, coo_col_indices, nnz, M, K, K);

  // Compute the SpMV
  float* out_coo_segment = new float[M];
  compute_spmv_coo_segment(coo_data, coo_row_indices, coo_col_indices, cvec_cpu, out_coo_segment, M, K, nnz);

  printf("out_coo_segment:\n");
  if (print_out) print_data(out_coo_segment, 1, M);
  // Compare the results
  check_result<float>(cout_cpu, out_coo_segment, M);

  // clean up
  delete[] coo_data;
  delete[] coo_row_indices;
  delete[] coo_col_indices;
  delete[] out_coo_segment;
  // ---------------------------------------------------------

  // clean up
  delete[] mat_cpu;
  delete[] vec_cpu;
  delete[] out_cpu;

  return 0;
}
