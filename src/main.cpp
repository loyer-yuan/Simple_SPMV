#include <cstdio>
#include <random>
#include <utils.h>

#include <spmv_cpu_kernel.h>

#define DEFAULT_M 10
#define DEFAULT_K 10

/**
 * @brief Main function
 * 
 * @param argc 1 or 3, when 3, rest arguments are the M and K values
 * @param argv M and K values, or none
 */
int main(int argc, char **argv)
{
    const int M = (argc != 3) ? DEFAULT_M : atoi(argv[1]);
    const int K = (argc != 3) ? DEFAULT_K : atoi(argv[2]);
    const int N = M * K;
    printf("M = %d, K = %d\n", M, K);

    float *mat_cpu = new float[N];
    prepare_data(mat_cpu, N);
    printf("\nmat_cpu:\n");
    print_data(mat_cpu, M, K);

    float *vec_cpu = new float[K];
    prepare_data(vec_cpu, K, 1.0);
    printf("\nvec_cpu:\n");
    print_data(vec_cpu, 1, K);

    float *out_cpu = new float[M];
    spmv_cpu_kernel0(mat_cpu, vec_cpu, out_cpu, M, K);
    printf("\nout_cpu:\n");
    print_data(out_cpu, 1, M);


    delete[] mat_cpu;
    delete[] vec_cpu;
}
