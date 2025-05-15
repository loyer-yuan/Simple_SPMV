#include <thread>
#include "Device.h"
#include "Ops.h"

namespace xsparse {

template <typename DType, int version>
struct ELLKernel;

template <typename DType>
struct ELLKernel<DType, 0>
{
    static void Run(
        const RTParams &rt, const DType *__restrict__ dataMat,
        const IdxType *__restrict__ idxMat, const DType *__restrict__ iVec,
        DType *__restrict__ oVer, const IdxType m, const IdxType k,
        const IdxType colIdxMat)
    {
        const uint ThreadNum = rt.hw.numCores;
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
};

/////////////////////////////////////////////////////////////////////////////////////////

template <typename DType>
void ELLCompute0(
    const HWParams hw, const DType *__restrict__ dataMat,
    const IdxType *__restrict__ idxMat, const DType *__restrict__ iVec,
    DType *__restrict__ oVer, const IdxType m, const IdxType k, const IdxType colIdxMat)
{
    const uint ThreadNum = hw.numCores;
    std::unique_ptr<std::thread[]> threads(new std::thread[ThreadNum]);
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        RTParams rt{i, hw};
        threads[i] = std::thread(
            &ELLKernel<DType, 0>::Run, rt, dataMat, idxMat, iVec, oVer, m, k,
            colIdxMat);
    }
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        threads[i].join();
    }
}

template <typename DType>
void ComputeSPMVELL(
    const HWParams hw, const DType *__restrict__ dataMat,
    const IdxType *__restrict__ idxMat, const DType *__restrict__ iVec,
    DType *__restrict__ oVer, const IdxType m, const IdxType k, const IdxType colIdxMat)
{
    ELLCompute0<DType>(hw, dataMat, idxMat, iVec, oVer, m, k, colIdxMat);
}
// Instantiation
template void ComputeSPMVELL<float>(
    const HWParams, const float *__restrict__, const IdxType *__restrict__,
    const float *__restrict__, float *__restrict__, const IdxType, const IdxType,
    const IdxType);
template void ComputeSPMVELL<double>(
    const HWParams, const double *__restrict__, const IdxType *__restrict__,
    const double *__restrict__, double *__restrict__, const IdxType, const IdxType,
    const IdxType);
template void ComputeSPMVELL<int>(
    const HWParams, const int *__restrict__, const IdxType *__restrict__,
    const int *__restrict__, int *__restrict__, const IdxType, const IdxType,
    const IdxType);

}  // namespace xsparse