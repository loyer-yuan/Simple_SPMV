#include <cstdio>

#include "utils.h"
#include "spmv_cpu_kernel.h"
#include "spmv_coo_kernel.h"

#define DEFAULT_M 11397
#define DEFAULT_K 11397
#define DEFAULT_NNZ 150645


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

  // float* out_cpu = new float[M];
  // spmv_cpu_kernel0(mat_cpu, vec_cpu, out_cpu, M, K);
  // if (print_out) {
  //   printf("\nout_cpu:\n");
  //   print_data(out_cpu, 1, M);
  // }

  const float* cmat_cpu = mat_cpu;
  const float* cvec_cpu = vec_cpu;

  // Transfer the matrix to COO format
  float* coo_data;
  int* coo_row_indices;
  int* coo_col_indices;
  int nnz = 0;
  printf("mat2coo start\n");
  mat2coo<float>(cmat_cpu, coo_data, coo_row_indices, coo_col_indices, nnz, M, K, K);
  printf("mat2coo finished\n");

  printf("\n===== COO Segement naive kernel with shared memory reduction =====\n");
  float *out_coo_segment_naive = new float[M];
  compute_spmv_coo_segment_naive<float>(coo_data, coo_row_indices, coo_col_indices, cvec_cpu, out_coo_segment_naive, M, K, nnz);
  printf("out_coo_segment_naive:\n");
  if (print_out) print_data(out_coo_segment_naive, 1, M);
  // Compare the results
  // check_result<float>(out_cpu, out_coo_segment_naive, M);

  printf("\n ===== COO Segement0 kernel with shuffle reduction =====\n");
  float *out_coo_segment0 = new float[M];
  compute_spmv_coo_segment<float>(coo_data, coo_row_indices, coo_col_indices, cvec_cpu, out_coo_segment0, M, K, nnz);
  printf("out_coo_segment0:\n");
  if (print_out) print_data(out_coo_segment0, 1, M);
  // Compare the results
  // check_result<float>(out_cpu, out_coo_segment0, M);

  // Compare the results
  check_result<float>(out_coo_segment0, out_coo_segment_naive, M);

  // clean up
  delete[] mat_cpu;
  delete[] vec_cpu;
  // delete[] out_cpu;
  delete[] coo_data;
  delete[] coo_row_indices;
  delete[] coo_col_indices;
  delete[] out_coo_segment_naive;
  delete[] out_coo_segment0;
}