#include "spmv_dia_kernel.h"
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
    float *mat_cpu = new float[NUM];
    printf("\nmat_cpu:\n");
    // Initialize mat_cpu
    std::memset(mat_cpu, 0, NUM * sizeof(float));
    random_diags_gen<float, true>(mat_cpu, M, DIAGS_NUM);
    print_data(mat_cpu, M, K);

    float *vec_cpu = new float[K];
    printf("\nvec_cpu:\n");
    prepare_data<float>(vec_cpu, K, 1.0f);
    print_data(vec_cpu, 1, K);

    // Convert the matrix to DIA format
    float *dia_data;
    int *dia_offsets;
    int ndiags = 0;
    mat2dia<float>(mat_cpu, dia_data, dia_offsets, ndiags, M, K, K);
    printf("\ndia_data:\n");
    print_data(dia_data, ndiags, M);
    printf("\ndia_offsets:\n");
    print_data(dia_offsets, 1, ndiags);

    float *out_cpu = new float[M];
    printf("\nout_cpu:\n");

    // Clean up
    delete[] dia_data;
    delete[] dia_offsets;

    delete[] mat_cpu;
    delete[] vec_cpu;
    delete[] out_cpu;
}