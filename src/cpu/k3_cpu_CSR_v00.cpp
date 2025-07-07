#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include <atomic>
#include <memory>
#include <thread>
#include "Device.h"
#include "Ops.h"

namespace xsparse {

template <typename DType, int version>
struct CSRKernel;

#ifdef __ARM_FEATURE_SVE
template <typename DType, int version>
struct CSRVectorKernel;
#endif

/**
 * @brief CSR scalar kernel
 */
template <typename DType>
struct CSRKernel<DType, 0>
{
    static void Run(
        const RTParams rt, const DType *__restrict__ csrData,
        const IdxType *__restrict__ csrRowIdices,
        const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
        DType *__restrict__ out, const IdxType m, const IdxType k)
    {
        int ThreadNum = rt.hw.numCores;
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

template <typename DType>
struct CSRKernel<DType, 1>
{
    static void Run(
        const RTParams rt, const DType *__restrict__ csrData,
        const IdxType *__restrict__ csrRowIdices,
        const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
        DType *__restrict__ out, const IdxType m, const IdxType k)
    {
        int ThreadNum = rt.hw.numCores;
        const IdxType rowsPerT = m / ThreadNum;
        const IdxType rowS = rt.tid * rowsPerT;
        const IdxType rowE = rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT;
#pragma unroll
        for (IdxType i = rowS; i < rowE; ++i)
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

template <typename DType>
struct CSRKernel<DType, 2>
{
    template <int UnrollFactor = 4>
    static void Run(
        const RTParams rt, const DType *__restrict__ csrData,
        const IdxType *__restrict__ csrRowIdices,
        const IdxType *__restrict__ csrColIdices, const DType *__restrict__ vec,
        DType *__restrict__ out, const IdxType m, const IdxType k)
    {
        int ThreadNum = rt.hw.numCores;
        const IdxType rowsPerT = m / ThreadNum;
        const IdxType rowS = rt.tid * rowsPerT;
        const IdxType rowE = rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT;

        for (IdxType i = rowS; i < rowE; ++i)
        {
            DType sum = 0;
            const IdxType rowStart = csrRowIdices[i];
            const IdxType rowEnd = csrRowIdices[i + 1];
            for (IdxType j = rowStart; j < rowEnd; j += UnrollFactor)
            {
                if (j + UnrollFactor <= rowEnd)
                {
#pragma unroll
                    for (int u = 0; u < UnrollFactor; ++u)
                    {
                        sum += csrData[j + u] * vec[csrColIdices[j + u]];
                    }
                }
                else
                {
                    for (; j < rowEnd; ++j)
                    {
                        sum += csrData[j] * vec[csrColIdices[j]];
                    }
                }
            }
            out[i] = sum;
        }
    }
};

#ifdef __ARM_FEATURE_SVE
template <>
struct CSRVectorKernel<float, 0>
{
    static void Run(
        const RTParams rt, const float *__restrict__ csrData,
        const IdxType *__restrict__ csrRowIdices,
        const IdxType *__restrict__ csrColIdices, const float *__restrict__ vec,
        float *__restrict__ out, const IdxType m, const IdxType k)
    {
        int ThreadNum = rt.hw.numCores;
        const IdxType rowsPerT = m / ThreadNum;
        const IdxType rowS = rt.tid * rowsPerT;
        const IdxType rowE = rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT;

        for (IdxType i = rowS; i < rowE; ++i)
        {
            const IdxType rowStart = csrRowIdices[i];
            const IdxType rowEnd = csrRowIdices[i + 1];

            svfloat32_t sum = svdup_f32(0.0f);
            IdxType j = rowStart;
            svbool_t pg = svwhilelt_b32(j, rowEnd);
            do
            {
                svint32_t colIdx = svld1(pg, &csrColIdices[j]);
                svfloat32_t matData = svld1(pg, &csrData[j]);
                svfloat32_t vecData = svld1_gather_index(pg, vec, colIdx);
                sum = svmla_f32_m(pg, sum, matData, vecData);
                j += svcntw();
                pg = svwhilelt_b32(j, rowEnd);
            }
            while (svptest_any(svptrue_b32(), pg));
            out[i] = svaddv_f32(svptrue_b32(), sum);
        }
    }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////

template <typename DType>
void CSRCompute0(
    const HWParams hw, const DType *__restrict__ csrData,
    const IdxType *__restrict__ csrRowIdices, const IdxType *__restrict__ csrColIdices,
    const DType *__restrict__ vec, DType *__restrict__ out, const IdxType m,
    const IdxType k)
{
    const int ThreadNum = hw.numCores;
    std::vector<std::thread> threads;
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        threads.emplace_back(
            [&, i]()
            {
                RTParams rt{i, hw};
                // CSRKernel<DType, 2>::template Run<8>(
                //     rt, csrData, csrRowIdices, csrColIdices, vec, out, m, k);
                CSRVectorKernel<DType, 0>::Run(
                    rt, csrData, csrRowIdices, csrColIdices, vec, out, m, k);
            });
    }
    for (auto &t : threads)
    {
        t.join();
    }
}

template <typename DType>
void ComputeSPMVCSR(
    const HWParams hw, const DType *__restrict__ csrData,
    const IdxType *__restrict__ csrRowIdices, const IdxType *__restrict__ csrColIdices,
    const DType *__restrict__ vec, DType *__restrict__ out, const IdxType m,
    const IdxType k)
{
    CSRCompute0<DType>(hw, csrData, csrRowIdices, csrColIdices, vec, out, m, k);
}
// Instantiation
template void ComputeSPMVCSR<float>(
    const HWParams, const float *__restrict__, const IdxType *__restrict__,
    const IdxType *__restrict__, const float *__restrict__, float *__restrict__,
    const IdxType, const IdxType);
// template void ComputeSPMVCSR<double>(
//     const HWParams, const double *__restrict__, const IdxType *__restrict__,
//     const IdxType *__restrict__, const double *__restrict__, double *__restrict__,
//     const IdxType, const IdxType);
// template void ComputeSPMVCSR<int>(
//     const HWParams, const int *__restrict__, const IdxType *__restrict__,
//     const IdxType *__restrict__, const int *__restrict__, int *__restrict__,
//     const IdxType, const IdxType);

template <typename DType>
void ComputeSPMVCSR_Ref(
    const HWParams hw, const DType *__restrict__ csrData,
    const IdxType *__restrict__ csrRowIdices, const IdxType *__restrict__ csrColIdices,
    const DType *__restrict__ vec, DType *__restrict__ out, const IdxType m,
    const IdxType k)
{
    const int ThreadNum = hw.numCores;
    std::unique_ptr<std::thread[]> threads(new std::thread[ThreadNum]);
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        RTParams rt{i, hw};
        threads[i] = std::thread(
            &CSRKernel<DType, 0>::Run, rt, csrData, csrRowIdices, csrColIdices, vec,
            out, m, k);
    }
    for (IdxType i = 0; i < ThreadNum; i++)
    {
        threads[i].join();
    }
}
// Instantiation
template void ComputeSPMVCSR_Ref<float>(
    const HWParams, const float *__restrict__, const IdxType *__restrict__,
    const IdxType *__restrict__, const float *__restrict__, float *__restrict__,
    const IdxType, const IdxType);
// template void ComputeSPMVCSR_Ref<double>(
//     const HWParams, const double *__restrict__, const IdxType *__restrict__,
//     const IdxType *__restrict__, const double *__restrict__, double *__restrict__,
//     const IdxType, const IdxType);
// template void ComputeSPMVCSR_Ref<int>(
//     const HWParams, const int *__restrict__, const IdxType *__restrict__,
//     const IdxType *__restrict__, const int *__restrict__, int *__restrict__,
//     const IdxType, const IdxType);
}  // namespace xsparse
