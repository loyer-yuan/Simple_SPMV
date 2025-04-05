#include <atomic>
#include <thread>
#include "Device.h"
#include "Ops.h"

namespace xsparse {

template <typename DType, uint ThreadNum>
void spmv_coo_kernel0(
    const RTParams &rt, const DType *__restrict__ cooData,
    const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    std::atomic<DType> *__restrict__ out, const IdxType m, const IdxType k,
    const IdxType nnz)
{
#pragma unroll
    for (IdxType i = rt.tid; i < nnz; i += ThreadNum)
    {
        const IdxType row = cooRowIndices[i];
        const IdxType col = cooColIndices[i];

        DType val = cooData[i] * vec[col];
        out[row].fetch_add(val, std::memory_order_relaxed);
    }
}

template <typename DType, uint ThreadNum>
void spmv_coo0(
    const DType *__restrict__ cooData, const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz)
{
    std::vector<DType> atomicOutBuffer(m, 0);
    auto *atomicOut = reinterpret_cast<std::atomic<DType> *>(atomicOutBuffer.data());

    RTParams rt[ThreadNum];
    std::thread threads[ThreadNum];
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        rt[i].tid = i;
        threads[i] = std::thread(
            &spmv_coo_kernel0<DType, ThreadNum>, std::ref(rt[i]), cooData,
            cooRowIndices, cooColIndices, vec, atomicOut, m, k, nnz);
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
void compute_spmv_coo(
    const DType *__restrict__ cooData, const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz)
{
    spmv_coo0<DType, static_cast<uint>(NumCores)>(
        cooData, cooRowIndices, cooColIndices, vec, out, m, k, nnz);
}
// Instantiation
template void compute_spmv_coo<float>(
    const float *__restrict__ cooData, const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const float *__restrict__ vec,
    float *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);
template void compute_spmv_coo<double>(
    const double *__restrict__ cooData, const IdxType *__restrict__ cooRowIndices,
    const IdxType *__restrict__ cooColIndices, const double *__restrict__ vec,
    double *__restrict__ out, const IdxType m, const IdxType k, const IdxType nnz);

}  // namespace xsparse