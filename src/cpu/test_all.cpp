#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Math.hpp"
#include "Matrix.hpp"
#include "Ops.h"
#include "PerfUtils.hpp"

#define DType float

// 定义固定宽度常量（根据实际需求调整）
constexpr int LABEL_WIDTH = 20;  // 标签部分占用宽度
constexpr int VALUE_WIDTH = 15;  // 数值部分占用宽度

// 辅助函数：格式化数值到字符串（固定精度）
template <typename T>
std::string format_value(T value, int precision = 6)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

#define FormatOutput(name, value, unit)                                        \
    std::cout << std::left << std::setw(LABEL_WIDTH) << name ":" << std::right \
              << std::setw(VALUE_WIDTH) << format_value((value)) << " " unit   \
              << std::endl;

#define PerfFunc(kernel, SPmat)                                                        \
    {                                                                                  \
        const int LoopCount = 1000;                                                    \
        xsparse::Timer timer;                                                          \
        for (int i = 0; i < 100; ++i)                                                  \
        {                                                                              \
            kernel;                                                                    \
        }                                                                              \
        timer.start();                                                                 \
        for (int i = 0; i < LoopCount; ++i)                                            \
        {                                                                              \
            kernel;                                                                    \
        }                                                                              \
        timer.stop();                                                                  \
        double mTime = timer.elapsed<std::chrono::nanoseconds>() / (double)1e9;        \
        double mFlops = (SPmat.GetTheoreticalFlops() * double(LoopCount)) / mTime;     \
        double mEBandwidth =                                                           \
            (SPmat.GetEffectiveSizeInBytes() * double(LoopCount)) / mTime;             \
        double mABandwidth = (SPmat.GetAllSizeInBytes() * double(LoopCount)) / mTime;  \
        FormatOutput("Time", mTime * 1000 / double(LoopCount), "ms");                  \
        FormatOutput("Flops/s", mFlops / 1024.0 / 1024.0 / 1024.0, "GFlops/s");        \
        FormatOutput(                                                                  \
            "Effective Bandwidth", mEBandwidth / 1024.0 / 1024.0 / 1024.0, "GB/s");    \
        FormatOutput(                                                                  \
            "Total GFlops", SPmat.GetTheoreticalFlops() / 1024.0 / 1024.0 / 1024.0,    \
            "GFlops");                                                                 \
        FormatOutput(                                                                  \
            "Effective Size",                                                          \
            SPmat.GetEffectiveSizeInBytes() / 1024.0 / 1024.0 / 1024.0, "GB");         \
        FormatOutput(                                                                  \
            "All Size", SPmat.GetAllSizeInBytes() / 1024.0 / 1024.0 / 1024.0, "GB");   \
        FormatOutput("All Bandwidth", mABandwidth / 1024.0 / 1024.0 / 1024.0, "GB/s"); \
    }

void PrintUsage()
{
    string usageMsg =
        "\n"
        "======================================================================\n"
        "   Usage   : test_all  [-m <int>] [-k <int>] [-prob <float>]\n"
        "                       [-p] [-f <string>] [-csrRef]\n"

        "           -m          Number of rows in the matrix (default: 5)\n"

        "           -k          Number of columns in the matrix (default: 5)\n"

        "           -prob       Probability of non-zero elements (default: 0.2)\n"

        "           -p          Print the input matrix and vector (default: false)\n"

        "           -f          Input file name. If set, M, K, prob would be\n"
        "                       overwritten \n"

        "           -csrRef   Use COO result as reference (default: false)\n"

        "           -h          Print this help message\n"
        "======================================================================\n";

    std::cout << usageMsg;
}

bool ParseArgs(
    int argc, char *argv[], int &M, int &K, double &PROB, bool &isPrint, bool &useFile,
    std::string &inputFileName, bool &useCSRRef)
{
    if (argc < 2)
    {
        PrintUsage();
        return false;
    }

    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "-m")
        {
            if (i + 1 > argc)
            {
                std::cerr << "Error: Missing value for -m option." << std::endl;
                return false;
            }
            M = std::stoi(argv[++i]);
            if (M <= 0)
            {
                std::cerr << "Error: Invalid value for -m option." << std::endl;
                return false;
            }
        }
        else if (arg == "-k")
        {
            if (i + 1 > argc)
            {
                std::cerr << "Error: Missing value for -k option." << std::endl;
                return false;
            }
            K = std::stoi(argv[++i]);
            if (K <= 0)
            {
                std::cerr << "Error: Invalid value for -k option." << std::endl;
                return false;
            }
        }
        else if (arg == "-prob")
        {
            if (i + 1 > argc)
            {
                std::cerr << "Error: Missing value for -prob option." << std::endl;
                return false;
            }
            PROB = std::stod(argv[++i]);
            if (PROB < 0.0 || PROB > 1.0)
            {
                std::cerr << "Error: Invalid value for -prob option." << std::endl;
                return false;
            }
        }
        else if (arg == "-p")
        {
            isPrint = true;
        }
        else if (arg == "-f")
        {
            if (i + 1 > argc)
            {
                std::cerr << "Error: Missing value for -f option." << std::endl;
                return false;
            }
            useFile = true;
            inputFileName = argv[++i];
        }
        else if (arg == "-csrRef")
        {
            useCSRRef = true;
        }
        else
        {
            PrintUsage();
            return false;
        }
    }
    return true;
}

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
    bool isPrint = false;
    bool useFile = false;
    bool useCSRRef = false;
    std::string inputFileName;

    if (!ParseArgs(argc, argv, M, K, PROB, isPrint, useFile, inputFileName, useCSRRef))
    {
        return -1;
    }

    if (useFile)
    {
        std::cout << "Using input file: " << inputFileName << std::endl;
    }
    else
    {
        std::cout << "M = " << M << ", K = " << K << std::endl;
        std::cout << "PROB = " << PROB << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    //
    //  Prepare data
    //

    xsparse::SPMatrixCOO<DType> spMatCOO;
    if (useFile)
    {
        if (!spMatCOO.CreateFromFile(inputFileName))
        {
            std::cerr << "Failed to create SPMatrixCOO from file!" << std::endl;
            return -1;
        }
        M = spMatCOO.GetRows();
        K = spMatCOO.GetCols();
        PROB = static_cast<double>(spMatCOO.mInfo.nnz) / static_cast<double>(M * K);

        std::cout << "----------------------------------------" << std::endl;
        std::cout << "M = " << M << ", K = " << K << ", NNZ = " << spMatCOO.mInfo.nnz
                  << std::endl;
        std::cout << "PROB = " << PROB << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
    else
    {
        if (!spMatCOO.CreateRamdomly(M, K, PROB, true))
        {
            std::cerr << "Failed to create SPMatrixCOO!" << std::endl;
            return -1;
        }
    }
    if (isPrint)
    {
        spMatCOO.PrintCOO();
    }
    if (0 && isPrint)
    {
        std::cout << "Input Matrix :" << std::endl;
        spMatCOO.PrintMat();
    }

    std::vector<DType> ivec(K);
    xsparse::GenerateData<DType>(ivec.data(), K, 1.0f, true, (DType)-10, (DType)10);
    if (isPrint)
    {
        std::cout << "Input vector:" << std::endl;
        for (auto i : ivec)
        {
            std::cout << i << " ";
        }
        std::cout << "\n" << std::endl;
    }

    std::vector<DType> ovec_ref(M);
    std::cout << "Running reference SpMV kernel..." << std::endl;
    std::cout << std::endl;

    if (!useCSRRef)
    {
        MVRef(spMatCOO, ivec, ovec_ref);

        if (isPrint)
        {
            std::cout << "Reference output vector:" << std::endl;
            for (auto i : ovec_ref)
            {
                std::cout << i << " ";
            }
            std::cout << "\n" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    //
    // Test CSR kernel
    //
    {

        std::cout << "Test CSR kernel." << std::endl;

        xsparse::SPMatrixCSR<DType> spMatCSR;
        if (!spMatCSR.CreateFromCOO(spMatCOO))
        {
            std::cerr << "Failed to create SPMatrixCSR!" << std::endl;
            return -1;
        }
        if (0 && isPrint)
        {
            spMatCSR.PrintCSR();
            spMatCSR.PrintMat();
        }

        std::cout << "Running CPU SpMV kernel..." << std::endl;

        std::vector<DType> ovec(M);
        // xsparse::ComputeSPMVCSR<DType>(
        //     spMatCSR.data.get(), spMatCSR.mInfo.rowPtr.get(),
        //     spMatCSR.mInfo.colIdx.get(), ivec.data(), ovec.data(), spMatCSR.m,
        //     spMatCSR.n);
        PerfFunc(
            xsparse::ComputeSPMVCSR<DType>(
                spMatCSR.data.get(), spMatCSR.mInfo.rowPtr.get(),
                spMatCSR.mInfo.colIdx.get(), ivec.data(), ovec.data(), spMatCSR.m,
                spMatCSR.n),
            spMatCSR);

        if (useCSRRef)
        {
            std::memcpy(ovec_ref.data(), ovec.data(), M * sizeof(DType));
            std::cout << "\nUsing CSR result as reference!\n" << std::endl;
        }

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
            std::cout << "Results mismatched " << count << " times in total."
                      << std::endl;
        }

        std::cout << "End of test!" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    //
    // Test COO kernel
    //
    {
        std::cout << "Test COO kernel." << std::endl;

        std::vector<DType> ovec(M);

        std::cout << "Running CPU SpMV kernel..." << std::endl;
        // xsparse::ComputeSPMVCOO<DType>(
        //     spMatCOO.data.get(), spMatCOO.mInfo.rowIdx.get(),
        //     spMatCOO.mInfo.colIdx.get(), ivec.data(), ovec.data(), spMatCOO.m,
        //     spMatCOO.n, spMatCOO.mInfo.nnz);
        PerfFunc(
            xsparse::ComputeSPMVCOO<DType>(
                spMatCOO.data.get(), spMatCOO.mInfo.rowIdx.get(),
                spMatCOO.mInfo.colIdx.get(), ivec.data(), ovec.data(), spMatCOO.m,
                spMatCOO.n, spMatCOO.mInfo.nnz),
            spMatCOO);

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
            std::cout << "Results mismatched " << count << " times in total."
                      << std::endl;
        }

        std::cout << "End of test!" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    //
    // Test ELL kernel
    //
    {
        std::cout << "Test ELL kernel." << std::endl;

        std::vector<DType> ovec(M);

        xsparse::SPMatrixELL<DType> spMatELL;
        if (!spMatELL.CreateFromCOO(spMatCOO))
        {
            std::cerr << "Failed to create SPMatrixELL!" << std::endl;
            return -1;
        }
        if (0 && isPrint)
        {
            spMatELL.PrintELL();
            spMatELL.PrintMat();
        }
        std::cout << "Running CPU SpMV kernel..." << std::endl;
        // xsparse::ComputeSPMVELL<DType>(
        //     spMatELL.data.get(), spMatELL.mInfo.idxMat.get(), ivec.data(), ovec.data(),
        //     spMatELL.m, spMatELL.n, spMatELL.mInfo.maxCol);
        PerfFunc(
            xsparse::ComputeSPMVELL<DType>(
                spMatELL.data.get(), spMatELL.mInfo.idxMat.get(), ivec.data(),
                ovec.data(), spMatELL.m, spMatELL.n, spMatELL.mInfo.maxCol),
            spMatELL);

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
            std::cout << "Results mismatched " << count << " times in total."
                      << std::endl;
        }

        std::cout << "End of test!" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}
