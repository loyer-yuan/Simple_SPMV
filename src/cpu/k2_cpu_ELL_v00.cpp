#include <thread>
#include "Device.h"
#include "Ops.h"

namespace xsparse {

template <typename DType, uint ThreadNum>
void ELLKernel0(
    const RTParams &rt, const DType *__restrict__ dataMat,
    const IdxType *__restrict__ idxMat, const DType *__restrict__ iVec,
    DType *__restrict__ oVer, const IdxType m, const IdxType k, const IdxType colIdxMat)
{
#pragma unroll
    for (IdxType i = rt.tid; i < m; i += ThreadNum)
    {
        const IdxType row = i;

        DType val = 0.;
        for (IdxType j = 0; j < colIdxMat; ++j)
        {
            const IdxType col = idxMat[row * colIdxMat + j];
            if (col == ELLZeroIdxVal)
                break;
            val += dataMat[row * colIdxMat + j] * iVec[col];
        }
        oVer[row] = val;
    }
}

template <typename DType, uint ThreadNum>
void ELLCompute0(
    const DType *__restrict__ dataMat, const IdxType *__restrict__ idxMat,
    const DType *__restrict__ iVec, DType *__restrict__ oVer, const IdxType m,
    const IdxType k, const IdxType colIdxMat)
{
    RTParams rt[ThreadNum];
    std::thread threads[ThreadNum];
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        rt[i].tid = i;
        threads[i] = std::thread(
            &ELLKernel0<DType, ThreadNum>, std::ref(rt[i]), dataMat, idxMat, iVec, oVer,
            m, k, colIdxMat);
    }
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        threads[i].join();
    }
}

template <typename DType>
void ComputeSPMVELL(
    const DType *__restrict__ dataMat, const IdxType *__restrict__ idxMat,
    const DType *__restrict__ iVec, DType *__restrict__ oVer, const IdxType m,
    const IdxType k, const IdxType colIdxMat)
{
    ELLCompute0<DType, static_cast<uint>(NumCores)>(
        dataMat, idxMat, iVec, oVer, m, k, colIdxMat);
}
// Instantiation
template void ComputeSPMVELL<float>(
    const float *__restrict__, const IdxType *__restrict__, const float *__restrict__,
    float *__restrict__, const IdxType, const IdxType, const IdxType);
template void ComputeSPMVELL<double>(
    const double *__restrict__, const IdxType *__restrict__, const double *__restrict__,
    double *__restrict__, const IdxType, const IdxType, const IdxType);
template void ComputeSPMVELL<int>(
    const int *__restrict__, const IdxType *__restrict__, const int *__restrict__,
    int *__restrict__, const IdxType, const IdxType, const IdxType);

}  // namespace xsparse