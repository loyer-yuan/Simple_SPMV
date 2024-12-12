#ifndef SIMPLE_SPMV_UTILS_H
#define SIMPLE_SPMV_UTILS_H

#include <iostream>
#include <iomanip>


/**
 * @brief Prepare data with random values, uniformly distributed on the closed interval [0, 1].
 * The data is prepared with a probability of prob.
 * 
 * @tparam T Data type
 * @param data Data array
 * @param n Number of elements
 * @param prob Probability of non-zero values
 */
template <typename T>
void prepare_data(T *data, const int n, const float prob = 0.2)
{
    std::mt19937 gen(9658);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        if (dis(gen) < prob) {
            data[i] = dis(gen);
        } else {
            data[i] = 0.0;
        }
    }
}

/**
 * @brief Print data in a matrix format
 */
template <typename T>
void print_data(const T *data, const int m, const int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            std::cout << std::setw(6) << std::setprecision(4) << std::fixed << data[i * k + j] << " ";
        }
        std::cout << std::endl;
    }
}

#endif // SIMPLE_SPMV_UTILS_H