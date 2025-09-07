#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ArmPL.hpp"
#include "Device.h"
#include "Math.hpp"
#include "Matrix.hpp"
#include "Ops.h"
#include "PerfUtils.hpp"

#define TestDType float

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
        const int LoopCount = 30;                                                      \
        xsparse::Timer timer;                                                          \
        for (int i = 0; i < 5; ++i)                                                    \
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

#define CheckResult(result, ref)                                               \
    {                                                                          \
        int count = 0;                                                         \
        for (int i = 0; i < cmdOpt.M; ++i)                                     \
        {                                                                      \
            if (xsparse::AllClose(result[i], ref[i]) == false && count++ < 10) \
            {                                                                  \
                std::cout << "Results mismatch at " << i << ": " << result[i]  \
                          << " != " << ref[i] << std::endl;                    \
            }                                                                  \
        }                                                                      \
        if (count == 0)                                                        \
        {                                                                      \
            std::cout << "Results match!" << std::endl;                        \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            std::cout << "Results mismatched " << count << " times in total."  \
                      << std::endl;                                            \
        }                                                                      \
    }

struct CmdOptions
{
    int M = 0;
    int K = 0;
    double PROB = 0.0;
    bool isPrint = false;
    bool useFile = false;
    bool useCSRRef = false;
    std::string inputFileName;
    int numCores = 1;
};

void PrintUsage()
{
    string usageMsg =
        "\n"
        "======================================================================\n"
        "   Usage   : test_all  [-m <int>] [-k <int>] [-prob <float>]\n"
        "                       [-p] [-f <string>] [-csrRef] [-t <int>]\n"

        "           -m          Number of rows in the matrix (default: 5)\n"

        "           -k          Number of columns in the matrix (default: 5)\n"

        "           -prob       Probability of non-zero elements (default: 0.2)\n"

        "           -p          Print the input matrix and vector (default: false)\n"

        "           -f          Input file name. If set, M, K, prob would be\n"
        "                       overwritten \n"

        "           -csrRef     Use COO result as reference (default: false)\n"

        "           -t          Number of threads (default: 1)\n"

        "           -h          Print this help message\n"
        "======================================================================\n";

    std::cout << usageMsg;
}

bool ParseArgs(int argc, char *argv[], CmdOptions &cmdOpt)
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
            cmdOpt.M = std::stoul(argv[++i]);
            if (cmdOpt.M <= 0)
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
            cmdOpt.K = std::stoul(argv[++i]);
            if (cmdOpt.K <= 0)
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
            cmdOpt.PROB = std::stod(argv[++i]);
            if (cmdOpt.PROB < 0.0 || cmdOpt.PROB > 1.0)
            {
                std::cerr << "Error: Invalid value for -prob option." << std::endl;
                return false;
            }
        }
        else if (arg == "-p")
        {
            cmdOpt.isPrint = true;
        }
        else if (arg == "-f")
        {
            if (i + 1 > argc)
            {
                std::cerr << "Error: Missing value for -f option." << std::endl;
                return false;
            }
            cmdOpt.useFile = true;
            cmdOpt.inputFileName = argv[++i];
        }
        else if (arg == "-csrRef")
        {
            cmdOpt.useCSRRef = true;
        }
        else if (arg == "-t")
        {
            cmdOpt.numCores = std::stoul(argv[++i]);
            if (cmdOpt.numCores <= 0)
            {
                std::cerr << "Error: Invalid value for -t option." << std::endl;
                return false;
            }
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
    CmdOptions cmdOpt;

    if (!ParseArgs(argc, argv, cmdOpt))
    {
        return -1;
    }

    if (cmdOpt.useFile)
    {
        std::cout << "Using input file: " << cmdOpt.inputFileName << std::endl;
    }
    else
    {
        std::cout << "M = " << cmdOpt.M << ", K = " << cmdOpt.K << std::endl;
        std::cout << "PROB = " << cmdOpt.PROB << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    //
    //  Prepare data
    //

    xsparse::SPMatrixCOO<TestDType> spMatCOO;
    if (cmdOpt.useFile)
    {
        if (!spMatCOO.CreateFromFile(cmdOpt.inputFileName))
        {
            std::cerr << "Failed to create SPMatrixCOO from file!" << std::endl;
            return -1;
        }
        cmdOpt.M = spMatCOO.GetRows();
        cmdOpt.K = spMatCOO.GetCols();
        cmdOpt.PROB = static_cast<double>(spMatCOO.mInfo.nnz) /
                      static_cast<double>(cmdOpt.M * cmdOpt.K);

        std::cout << "----------------------------------------" << std::endl;
        std::cout << "M = " << cmdOpt.M << ", K = " << cmdOpt.K
                  << ", NNZ = " << spMatCOO.mInfo.nnz << std::endl;
        std::cout << "PROB = " << cmdOpt.PROB << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
    else
    {
        if (!spMatCOO.CreateRamdomly(cmdOpt.M, cmdOpt.K, cmdOpt.PROB, true))
        {
            std::cerr << "Failed to create SPMatrixCOO!" << std::endl;
            return -1;
        }
    }
    if (cmdOpt.isPrint)
    {
        spMatCOO.PrintCOO();
    }
    if (0 && cmdOpt.isPrint)
    {
        std::cout << "Input Matrix :" << std::endl;
        spMatCOO.PrintMat();
    }

    std::vector<TestDType> ivec(cmdOpt.K);
    xsparse::GenerateData<TestDType>(
        ivec.data(), cmdOpt.K, 1.0f, true, (TestDType)-10, (TestDType)10);
    if (cmdOpt.isPrint)
    {
        std::cout << "Input vector:" << std::endl;
        for (auto i : ivec)
        {
            std::cout << i << " ";
        }
        std::cout << "\n" << std::endl;
    }

    //
    // Test CSR kernel
    //
    {

        std::cout << "Test CSR kernel." << std::endl;

        xsparse::SPMatrixCSR<TestDType> spMatCSR;
        if (!spMatCSR.CreateFromCOO(spMatCOO))
        {
            std::cerr << "Failed to create SPMatrixCSR!" << std::endl;
            return -1;
        }
        if (0 && cmdOpt.isPrint)
        {
            spMatCSR.PrintCSR();
            spMatCSR.PrintMat();
        }

        // Test CSR kernel with ArmPL
        std::cout << "Test ArmPL CSR kernel.\n" << std::endl;
        xsparse::ArmPL<xsparse::SPMatF::SPMatFormatCSR, TestDType> armplCSRKernel;
        if (!armplCSRKernel.Initialize(
                spMatCSR.m, spMatCSR.n, spMatCSR.mInfo.nnz, spMatCSR.data.get(),
                spMatCSR.mInfo.rowPtr.get(), spMatCSR.mInfo.colIdx.get()))
        {
            std::cerr << "Failed to create ArmPL CSR matrix!" << std::endl;
            return -1;
        }
        std::vector<TestDType> ovec_armpl(cmdOpt.M);

        std::cout << "Running SpMV kernel..." << std::endl;
        PerfFunc(armplCSRKernel.Run(ivec.data(), ovec_armpl.data()), spMatCSR);
        // armplCSRKernel.Run(ivec.data(), ovec_armpl.data());

        armplCSRKernel.Destroy();
        std::cout << "End of test!" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}
