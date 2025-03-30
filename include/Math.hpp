#ifndef XSPARSE_MATH_HPP
#define XSPARSE_MATH_HPP

#define DEPRECATED_CODE 0

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <random>

#define assertm(exp, msg) assert((void(msg), exp))

namespace xsparse {

static std::atomic<unsigned int> globalSeedCounter(2025);

/**
 * @brief Generate indices with random values, uniformly distributed on the closed 
 * interval [0, mSize * nSize].
 * The indices are prepared with a probability of prob.
 * Indices are sorted in ascending order on row-major order.
 *
 * @tparam T Data type
 * @param mIndices Row indices array
 * @param nIndices Column indices array
 * @param mSize Number of rows
 * @param nSize Number of columns
 * @param nnz Number of non-zero elements
 * @param isRuntimeRandom If true, random_device is used to generate random values
 */
template <typename T = uint32_t>
void GenerateIndices2D(
    T *mIndices, T *nIndices, const T mSize, const T nSize, const T nnz,
    bool isRuntimeRandom = false)
{
    const T maxSize = mSize * nSize;
    assertm(
        nnz <= maxSize,
        "Number of non-zero elements should be less than or equal to the dimension "
        "size");
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis(0.0f, 1.0f);
    if (isRuntimeRandom) [[likely]]
    {
        std::random_device rd;
        gen.seed(rd());
    }
    else [[unlikely]]
    {
        unsigned int seed = globalSeedCounter++;
        gen.seed(seed);
    }
    const double naiveProb = (double)nnz / (double)maxSize;

    T idx = 0;
    T remaining = nnz;

    for (T i = 0; remaining > 0 && i < maxSize; i++)
    {
        double prob = (double)remaining / (double)(maxSize - i);
        if (dis(gen) < prob)
        {
            mIndices[idx] = i / (T)nSize;
            nIndices[idx] = i % (T)nSize;
            idx++;
            --remaining;
        }
    }
    if (idx != nnz) [[unlikely]]
    {
        std::cout << "Warning: Generated indices are not equal to nnz!" << std::endl;
        // Run again
        GenerateIndices2D(mIndices, nIndices, mSize, nSize, nnz, isRuntimeRandom);
    }
}

#if DEPRECATED_CODE
/**
 * @brief Generate indices with random values, uniformly distributed on the closed interval [0, dimSize].
 * The indices are prepared with a probability of prob.
 *
 * @tparam T Data type
 * @param indices Indices array
 * @param dimSize Dimension size
 * @param nnz Number of non-zero elements
 * @param isRuntimeRandom If true, random_device is used to generate random values
 */
template <typename T = uint32_t>
void GenerateIndices1D(
    T *indices, const T dimSize, const T nnz, bool isRuntimeRandom = false)
{
    assertm(
        nnz <= dimSize,
        "Number of non-zero elements should be less than or equal to the dimension "
        "size");
    T idx = 0;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis(0.0f, 1.0f);
    if (isRuntimeRandom) [[likely]]
    {
        std::random_device rd;
        gen.seed(rd());
    }
    else [[unlikely]]
    {
        unsigned int seed = globalSeedCounter++;
        gen.seed(seed);
    }
    const double naiveProb = (nnz / dimSize);
    const double prob =
        (1.0f - naiveProb) * 0.2f + naiveProb;  // Increase the probability
    for (int i = 0; idx < nnz && i < dimSize; i++)
    {
        if (dis(gen) < prob)
        {
            indices[idx] = i;
            idx++;
        }
    }
    if (idx != (nnz - 1)) [[unlikely]]
    {
        std::cout << "Warning: Generated indices are not equal to nnz!" << std::endl;
        // Run again
        GenerateIndices(indices, dimSize, nnz, isRuntimeRandom);
    }
}
#endif  // DEPRECATED_CODE

/**
 * @brief Generate data with random values, uniformly distributed on the closed interval 
 *        [minV, maxV].
 * The data is prepared with a probability of prob.
 *
 * @tparam T Data type
 * @tparam isRuntimeRandom If true, random_device is used to generate random values
 * @param data Data array
 * @param n Number of elements
 * @param prob Probability of non-zero values
 */
template <typename T = float>
uint32_t GenerateData(
    T *data, const int n, const float prob = 0.2, const bool isRuntimeRandom = false,
    const T minV = (T)0.0, const T maxV = (T)1.0)
{
    uint32_t count = 0;
    if (isRuntimeRandom) [[likely]]
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0, 1);
        for (int i = 0; i < n; i++)
        {
            if (dis(gen) < prob)
            {
                count++;
                data[i] = (T)(((double)dis(gen)) * ((double)maxV - (double)minV) +
                              (double)minV);
            }
            else
            {
                data[i] = 0.0;
            }
        }
    }
    else [[unlikely]]
    {
        unsigned int seed = globalSeedCounter++;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> dis(0, 1);
        for (int i = 0; i < n; i++)
        {
            if (dis(gen) < prob)
            {
                count++;
                data[i] = (T)(((double)dis(gen)) * ((double)maxV - (double)minV) +
                              (double)minV);
            }
            else
            {
                data[i] = 0.0;
            }
        }
    }
    return count;
}

#if DEPRECATED_CODE
/**
 * @brief Generate a matrix with random diagonals
 */
template <typename T = float>
void GenerateDIAData(T *mat, int m, int numDiags, bool isRuntimeRandom = false)
{
    const int max_diags = 2 * m - 1;
    assertm(
        numDiags <= max_diags,
        "Number of diagonals should be less than or equal to 2 * m - 1");
    if (isRuntimeRandom)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0.0, 1.0);
        for (int i = 0; i < max_diags; i++)
        {
            // Choose as a diagonal
            if (dis(gen) < numDiags / max_diags)
            {
                const int distance = i - (m - 1);
                printf("distance = %d has been chosen as a diagoal.\n", distance);
                for (int x = 0; x < m; ++x)
                {
                    for (int y = 0; y < m; ++y)
                    {
                        if (y - x == distance)
                        {
                            mat[x * m + y] = dis(gen);
                        }
                    }
                }
            }
        }
    }
    else
    {
        unsigned int seed = globalSeedCounter++;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        for (int i = 0; i < max_diags; i++)
        {
            // Choose as a diagonal
            if (dis(gen) < numDiags / max_diags)
            {
                const int distance = i - (m - 1);
                printf("distance = %d has been chosen as a diagoal.\n", distance);
                for (int x = 0; x < m; ++x)
                {
                    for (int y = 0; y < m; ++y)
                    {
                        if (y - x == distance)
                        {
                            mat[x * m + y] = dis(gen);
                        }
                    }
                }
            }
        }
    }
}
#endif  // DEPRECATED_CODE

template <typename T = float>
inline bool AllClose(
    const T &a, const T &b, const float rtol = 1e-05, const float atol = 1e-08)
{
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

}  // namespace xsparse
#endif  // XSPARSE_MATH_HPP
