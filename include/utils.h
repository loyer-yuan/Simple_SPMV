#ifndef SIMPLE_SPMV_UTILS_H
#define SIMPLE_SPMV_UTILS_H

#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#define assertm(exp, msg) assert((void(msg), exp))


/**
 * @brief Prepare data with random values, uniformly distributed on the closed interval [0, 1].
 * The data is prepared with a probability of prob.
 *
 * @tparam T Data type
 * @tparam isRuntimeRandom If true, random_device is used to generate random values
 * @param data Data array
 * @param n Number of elements
 * @param prob Probability of non-zero values
 */
template <typename T = float, bool isRuntimeRandom = false>
void prepare_data(T* data, const int n, const float prob = 0.2)
{
  if constexpr (isRuntimeRandom) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
      if (dis(gen) < prob) {
        data[i] = dis(gen);
      } else {
        data[i] = 0.0;
      }
    }
  } else {
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
}


/**
 * @brief Generate a matrix with random diagonals
 */
template<typename T = float, bool isRuntimeRandom = false>
void random_diags_gen(T* mat, int m, float num_diags) {
  const int max_diags = 2 * m - 1;
  assertm(num_diags <= max_diags, "Number of diagonals should be less than or equal to 2 * m - 1");
  if constexpr (isRuntimeRandom) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < max_diags; i++) {
      // Choose as a diagonal
      if (dis(gen) < num_diags / max_diags) {
        const int distance = i - (m - 1);
        printf("distance = %d has been chosen as a diagoal.\n", distance);
        for (int x = 0; x < m; ++x) {
          for (int y = 0; y < m; ++y) {
            if (y - x == distance) {
              mat[x * m + y] = dis(gen);
            }
          }
        }
      }
    }
  } else {
    std::mt19937 gen(9658);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < max_diags; i++) {
      // Choose as a diagonal
      if (dis(gen) < num_diags / max_diags) {
        const int distance = i - (m - 1);
        printf("distance = %d has been chosen as a diagoal.\n", distance);
        for (int x = 0; x < m; ++x) {
          for (int y = 0; y < m; ++y) {
            if (y - x == distance) {
              mat[x * m + y] = dis(gen);
            }
          }
        }
      }
    }
  }
}


/**
 * @brief Print data in a matrix format
 */
template <typename T = float>
void print_data(const T* data, const int m, const int k)
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