#include <spmv_cpu_kernel.h>

template <typename T>
void spmv_cpu_kernel0(const T * __restrict__ mat, const T * __restrict__ vec, T * __restrict__ out, const int M, const int K)
{
    for (int i = 0; i < M; i++)
    {
        T sum = 0;
        for (int j = 0; j < K; j++)
        {
            sum += mat[i * K + j] * vec[j];
        }
        out[i] = sum;
    }
}

// Instantiate the template for float
template void spmv_cpu_kernel0<float>(const float * mat, const float * vec, float * out, const int M, const int K);