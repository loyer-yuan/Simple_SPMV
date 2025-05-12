#include <atomic>
#include <thread>
#include "Device.h"
#include "Ops.h"

namespace xsparse {

template <typename DType, uint ThreadNum, int version>
struct CSRKernel;

/**
 * @brief CSR scalar kernel
 */
template <typename DType, uint ThreadNum>
struct CSRKernel<DType, ThreadNum, 0>
{
    static void Run(
        const RTParams &rt, const DType *__restrict__ csrData,
        const IdxType *__restrict__ csrRowIdices,
        const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
        DType *__restrict__ out, const IdxType m, const IdxType k)
    {
#pragma unroll
        for (IdxType i = rt.tid; i < m; i += ThreadNum)
        {
            DType sum = 0;
            const IdxType rowStart = csrRowIdices[i];
            const IdxType rowEnd = csrRowIdices[i + 1];
            for (IdxType j = rowStart; j < rowEnd; ++j)
            {
                sum += csrData[j] * vec[csrColIdices[j]];
            }
            out[i] = sum;
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////

template <typename DType, uint ThreadNum>
void CSRCompute0(
    const DType *__restrict__ csrData, const IdxType *__restrict__ csrRowIdices,
    const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k)
{
    RTParams rt[ThreadNum];
    std::thread threads[ThreadNum];
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        rt[i].tid = i;
        threads[i] = std::thread(
            &CSRKernel<DType, ThreadNum, 0>::Run, std::ref(rt[i]), csrData,
            csrRowIdices, csrColIdices, vec, out, m, k);
    }
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        threads[i].join();
    }
}

template <typename DType>
void ComputeSPMVCSR(
    const DType *__restrict__ csrData, const IdxType *__restrict__ csrRowIdices,
    const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k)
{
    CSRCompute0<DType, NumCores>(csrData, csrRowIdices, csrColIdices, vec, out, m, k);
}
// Instantiation
template void ComputeSPMVCSR<float>(
    const float *__restrict__, const IdxType *__restrict__, const IdxType *__restrict__,
    const float *__restrict__, float *__restrict__, const IdxType, const IdxType);
template void ComputeSPMVCSR<double>(
    const double *__restrict__, const IdxType *__restrict__,
    const IdxType *__restrict__, const double *__restrict__, double *__restrict__,
    const IdxType, const IdxType);
template void ComputeSPMVCSR<int>(
    const int *__restrict__, const IdxType *__restrict__, const IdxType *__restrict__,
    const int *__restrict__, int *__restrict__, const IdxType, const IdxType);

template <typename DType>
void ComputeSPMVCSR_Ref(
    const DType *__restrict__ csrData, const IdxType *__restrict__ csrRowIdices,
    const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
    DType *__restrict__ out, const IdxType m, const IdxType k)
{
    RTParams rt[NumCores];
    std::thread threads[NumCores];
    for (IdxType i = 0; i < NumCores; i++)
    {
        rt[i].tid = i;
        threads[i] = std::thread(
            &CSRKernel<DType, NumCores, 0>::Run, std::ref(rt[i]), csrData, csrRowIdices,
            csrColIdices, vec, out, m, k);
    }
    for (IdxType i = 0; i < NumCores; i++)
    {
        threads[i].join();
    }
}
// Instantiation
template void ComputeSPMVCSR_Ref<float>(
    const float *__restrict__, const IdxType *__restrict__, const IdxType *__restrict__,
    const float *__restrict__, float *__restrict__, const IdxType, const IdxType);
template void ComputeSPMVCSR_Ref<double>(
    const double *__restrict__, const IdxType *__restrict__,
    const IdxType *__restrict__, const double *__restrict__, double *__restrict__,
    const IdxType, const IdxType);
template void ComputeSPMVCSR_Ref<int>(
    const int *__restrict__, const IdxType *__restrict__, const IdxType *__restrict__,
    const int *__restrict__, int *__restrict__, const IdxType, const IdxType);

}  // namespace xsparse
