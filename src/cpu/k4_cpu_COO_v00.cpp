#include <atomic>
#include <memory>
#include <thread>
#include "Device.h"
#include "Ops.h"

namespace xsparse {

template <typename DType, int version>
struct COOKernel;

template <typename DType>
struct COOKernel<DType, 0>
{
    static void Run(
        const RTParams rt, const DType *__restrict__ cooData,
        const IdxType *__restrict__ cooRowIndices,
        const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
        std::atomic<DType> *__restrict__ out, const IdxType m, const IdxType k,
        const IdxType nnz)
    {
        int ThreadNum = rt.hw.numCores;
#pragma unroll
        for (IdxType i = rt.tid; i < nnz; i += ThreadNum)
        {
            const IdxType row = cooRowIndices[i];
            const IdxType col = cooColIndices[i];

            DType val = cooData[i] * vec[col];
            out[row].fetch_add(val, std::memory_order_relaxed);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////

template <typename DType>
void COOCompute0(
    const HWParams hw, const DType *__restrict__ cooData,
    const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz)
{
    const int ThreadNum = hw.numCores;
    std::vector<DType> atomicOutBuffer(m, 0);
    auto *atomicOut = reinterpret_cast<std::atomic<DType> *>(atomicOutBuffer.data());

    std::unique_ptr<std::thread[]> threads(new std::thread[ThreadNum]);
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        // rt[i].tid = i;
        // rt[i].hw = hw;
        RTParams rt{i, hw};
        threads[i] = std::thread(
            &COOKernel<DType, 0>::Run, rt, cooData, cooRowIndices, cooColIndices, vec,
            atomicOut, m, k, nnz);
    }
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        threads[i].join();
    }
    for (IdxType i = 0; i < m; i++)
    {
        out[i] = atomicOut[i].load(std::memory_order_relaxed);
    }
}

template <typename DType>
void ComputeSPMVCOO(
    const HWParams hw, const DType *__restrict__ cooData,
    const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz)
{
    COOCompute0<DType>(hw, cooData, cooRowIndices, cooColIndices, vec, out, m, k, nnz);
}
// Instantiation
template void ComputeSPMVCOO<float>(
    const HWParams, const float *__restrict__ cooData,
    const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const float *__restrict__ vec,
    float *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);
template void ComputeSPMVCOO<double>(
    const HWParams, const double *__restrict__ cooData,
    const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const double *__restrict__ vec,
    double *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);
template void ComputeSPMVCOO<int>(
    const HWParams, const int *__restrict__ cooData,
    const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const int *__restrict__ vec,
    int *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);
}  // namespace xsparse