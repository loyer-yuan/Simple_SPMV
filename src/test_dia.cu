#include "spmv_dia_kernel.h"
#include "spmv_cpu_kernel.h"
#include "utils.h"
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

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

    // Run the CPU SpMV kernel
    float *out_cpu = new float[M];
    printf("\nout_cpu:\n");
    spmv_cpu_kernel0(mat_cpu, vec_cpu, out_cpu, M, K);
    print_data(out_cpu, 1, M);

    // Run the DIA SpMV kernel
    float *dia_data_gpu;
    int *dia_offsets_gpu;
    float *vec_gpu;
    float *out_gpu;
    cudaMalloc(&dia_data_gpu, ndiags * M * sizeof(float));
    cudaMalloc(&dia_offsets_gpu, ndiags * sizeof(int));
    cudaMalloc(&vec_gpu, K * sizeof(float));
    cudaMalloc(&out_gpu, M * sizeof(float));
    cudaMemcpy(dia_data_gpu, dia_data, ndiags * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dia_offsets_gpu, dia_offsets, ndiags * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_gpu, vec_cpu, K * sizeof(float), cudaMemcpyHostToDevice);
    spmv_dia0<float>(dia_data_gpu, dia_offsets_gpu, vec_gpu, out_gpu, ndiags, M, K);
    float *out_gpu_cpu = new float[M];
    cudaMemcpy(out_gpu_cpu, out_gpu, M * sizeof(float), cudaMemcpyDeviceToHost);

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
    cudaFree(dia_data_gpu);
    cudaFree(dia_offsets_gpu);
    cudaFree(vec_gpu);
    cudaFree(out_gpu);

    delete[] dia_data;
    delete[] dia_offsets;

    delete[] mat_cpu;
    delete[] vec_cpu;
    delete[] out_cpu;
}