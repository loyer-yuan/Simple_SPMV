#include <cstdio>

#include "utils.h"
#include "spmv_cpu_kernel.h"

#define DEFAULT_M 10
#define DEFAULT_K 10
#define DEFAULT_NNZ 2

#define IS_RANDOM true

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
  prepare_data<float, IS_RANDOM>(mat_cpu, N, prob);
  if (print_out) {
    printf("\nmat_cpu:\n");
    print_data(mat_cpu, M, K);
  }

  float* vec_cpu = new float[K];
  prepare_data<float, IS_RANDOM>(vec_cpu, K, 1.0);
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


  // clean up
  delete[] mat_cpu;
  delete[] vec_cpu;
  delete[] out_cpu;

  return 0;
}