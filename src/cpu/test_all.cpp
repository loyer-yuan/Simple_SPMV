#include <vector>

#include "Math.hpp"
#include "Matrix.hpp"
#include "Ops.h"

#define DType   float

// Debugging macro
#define IsPrint 0

template <typename T>
void MVRef(
    const xsparse::BaseMatrix<T> &mat, const std::vector<T> &vec, std::vector<T> &out)
{
    const int m = mat.GetRows();
    const int k = mat.GetCols();
    assertm(k == vec.size(), "Matrix and vector size mismatch!");
    assertm(m == out.size(), "Matrix and output vector size mismatch!");

    for (int i = 0; i < m; ++i)
    {
        out[i] = 0;
        for (int j = 0; j < k; ++j)
        {
            out[i] += mat(i, j) * vec[j];
        }
    }
}

int main(int argc, char *argv[])
{
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Start of tests!" << std::endl;

    //
    // Parameters
    //
    int M = 0;
    int K = 0;
    double PROB = 0.0;
    if (argc == 4)
    {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        PROB = atof(argv[3]);
    }
    else
    {
        M = 10;
        K = 10;
        PROB = 0.2f;
    }

    if (M <= 0 || K <= 0 || PROB < 0.0f || PROB > 1.0f)
    {
        std::cerr << "Invalid parameters!" << std::endl;
        return -1;
    }

    std::cout << "M = " << M << ", K = " << K << std::endl;
    std::cout << "PROB = " << PROB << std::endl;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Test COO kernel." << std::endl;

    //
    // Prepare data
    //

    std::vector<DType> ivec(K);
    xsparse::GenerateData<DType>(
        ivec.data(), K, 1.0f, true, (DType)-10.0f, (DType)10.0f);
#if IsPrint
    std::cout << "Input vector:" << std::endl;
    for (auto i : ivec)
    {
        std::cout << i << " ";
    }
    std::cout << "\n" << std::endl;
#endif

    xsparse::SPMatrixCOO<DType> spMatCOO;
    if (!spMatCOO.CreateRamdomly(M, K, PROB, true))
    {
        std::cerr << "Failed to create SPMatrixCOO!" << std::endl;
        return -1;
    }
#if IsPrint
    spMatCOO.PrintCOO();
    spMatCOO.PrintMat();
#endif

    std::vector<DType> ovec(M);
    std::vector<DType> ovec_ref(M);

    //
    // Run the COO SpMV kernel
    //

    std::cout << "Running CPU SpMV kernel..." << std::endl;
    xsparse::compute_spmv_coo<DType>(
        spMatCOO.data.get(), spMatCOO.mInfo.rowIdx.get(), spMatCOO.mInfo.colIdx.get(),
        ivec.data(), ovec.data(), spMatCOO.m, spMatCOO.n, spMatCOO.mInfo.nnz);

    // Check the result
    std::cout << "Running reference SpMV kernel..." << std::endl;
    MVRef(spMatCOO, ivec, ovec_ref);

    std::cout << "Checking results..." << std::endl;
    int count = 0;
    for (int i = 0; i < M; ++i)
    {
        if (xsparse::AllClose(ovec[i], ovec_ref[i]) == false && count++ < 10)
        {
            std::cout << "Results mismatch at " << i << ": " << ovec[i]
                      << " != " << ovec_ref[i] << std::endl;
        }
    }
    if (count == 0)
    {
        std::cout << "Results match!" << std::endl;
    }
    else
    {
        std::cout << "Results mismatched " << count << " times in total." << std::endl;
    }

    std::cout << "End of test!" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}
