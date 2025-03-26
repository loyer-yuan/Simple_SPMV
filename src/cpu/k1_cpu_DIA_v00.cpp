#include "spmv_dia_kernel.h"
#include <thread>

struct RT_Params
{
    uint32_t tid;
};

/**
 * @brief DIA SpMV kernel
 *
 * @tparam T data type
 * @param rt runtime parameters
 * @param dia_data input matrix's DIA format data array
 * @param dia_offsets input matrix's DIA format offsets array
 * @param vec input vector
 * @param out output vector
 * @param ndiags intput number of diagonals
 * @param m input number of rows
 * @param k input number of columns
 *
 * @todo use uint32_t instead of int
 */
template <typename T>
void spmv_dia_kernel0(const RT_Params rt, const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k) {
    const uint32_t row = rt.tid;
    if (row < m) {
        T sum = 0;
        for (uint32_t i = 0; i < ndiags; i++) {
            const uint32_t col = row + dia_offsets[i];
            const T value = dia_data[i * m + row];
            if (col >= 0 && col < k) {
                sum += value * vec[col];
            }
        }
        out[row] = sum;
    }
}

/**
 * @brief DIA SpMV kernel wrapper
 *
 * @tparam T data type
 * @param dia_data input matrix's DIA format data array
 * @param dia_offsets input matrix's DIA format offsets array
 * @param vec input vector
 * @param out output vector
 * @param ndiags intput number of diagonals
 * @param m input number of rows
 * @param k input number of columns
 */
template <typename T>
void spmv_dia0(const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k)
{
    std::thread threads[m];
    RT_Params rt[m];
    for (uint32_t i = 0; i < m; i++) {
        rt[i].tid = i;
        threads[i] = std::thread(&spmv_dia_kernel0<T>, rt[i], dia_data, dia_offsets, vec, out, ndiags, m, k);
        //spmv_dia_kernel0(rt, dia_data, dia_offsets, vec, out, ndiags, m, k);
    }
    for (auto& t: threads) {
        t.join();
    }
}

template<typename T>
void compute_spmv_dia(const T* __restrict__ dia_data, const int* __restrict__ dia_offsets, const T* __restrict__ vec, T* __restrict__ out, const int ndiags, const int m, const int k)
{
    spmv_dia0(dia_data, dia_offsets, vec, out, ndiags, m, k);
}
// Instantiate the template
template void compute_spmv_dia<float>(const float* __restrict__ dia_data, const int* __restrict__ dia_offsets, const float* __restrict__ vec, float* __restrict__ out, const int ndiags, const int m, const int k);
