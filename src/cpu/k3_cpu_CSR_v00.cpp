#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include <atomic>
#include <memory>
#include <thread>
#include "ArmIntrinsic.hpp"
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

// #ifdef __ARM_FEATURE_SVE
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
                // svfloat32_t vecData = svcvt_f32_s32_x(pg, colIdx);
                sum = svmla_f32_m(pg, sum, matData, vecData);
                j += svcntw();
                pg = svwhilelt_b32(j, rowEnd);
            }
            while (svptest_any(svptrue_b32(), pg));
            out[i] = svaddv_f32(svptrue_b32(), sum);
        }
    }
};

// 定义可配置的unroll factor
#ifndef CSR_UNROLL_FACTOR
#define CSR_UNROLL_FACTOR 2
#endif

// 定义宏来生成循环展开的代码
#define UNROLL_1(ACTION, ...) ACTION(0, ##__VA_ARGS__)
#define UNROLL_2(ACTION, ...) UNROLL_1(ACTION, ##__VA_ARGS__) ACTION(1, ##__VA_ARGS__)
#define UNROLL_4(ACTION, ...) \
    UNROLL_2(ACTION, ##__VA_ARGS__) ACTION(2, ##__VA_ARGS__) ACTION(3, ##__VA_ARGS__)
#define UNROLL_8(ACTION, ...)       \
    UNROLL_4(ACTION, ##__VA_ARGS__) \
    ACTION(4, ##__VA_ARGS__)        \
    ACTION(5, ##__VA_ARGS__) ACTION(6, ##__VA_ARGS__) ACTION(7, ##__VA_ARGS__)
#define UNROLL_16(ACTION, ...)      \
    UNROLL_8(ACTION, ##__VA_ARGS__) \
    ACTION(8, ##__VA_ARGS__)        \
    ACTION(9, ##__VA_ARGS__)        \
    ACTION(10, ##__VA_ARGS__)       \
    ACTION(11, ##__VA_ARGS__)       \
    ACTION(12, ##__VA_ARGS__)       \
    ACTION(13, ##__VA_ARGS__) ACTION(14, ##__VA_ARGS__) ACTION(15, ##__VA_ARGS__)

// 根据CSR_UNROLL_FACTOR选择对应的展开宏
#if CSR_UNROLL_FACTOR == 1
#define UNROLL_N(ACTION, ...) UNROLL_1(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 2
#define UNROLL_N(ACTION, ...) UNROLL_2(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 4
#define UNROLL_N(ACTION, ...) UNROLL_4(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 8
#define UNROLL_N(ACTION, ...) UNROLL_8(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 16
#define UNROLL_N(ACTION, ...) UNROLL_16(ACTION, ##__VA_ARGS__)
#else
#error "Unsupported CSR_UNROLL_FACTOR. Supported values: 1, 2, 4, 8, 16"
#endif

// 定义各种操作的宏
#define DECLARE_ACC(idx, ...) svfloat32_t acc##idx = svdup_f32(0.0f);

#define COMPUTE_ITERATION(idx, pCol, pVal, vl, vec)                                  \
    do                                                                               \
    {                                                                                \
        svint32_t colIdx##idx = svld1_s32(svptrue_b32(), pCol + (idx)*vl);           \
        svfloat32_t matData##idx = svld1(svptrue_b32(), pVal + (idx)*vl);            \
        svfloat32_t vecData##idx =                                                   \
            svld1_gather_index(svptrue_b32(), vec, colIdx##idx);                     \
        acc##idx = svmla_f32_x(svptrue_b32(), acc##idx, matData##idx, vecData##idx); \
    }                                                                                \
    while (0);

#define ADD_TO_SUM(idx, sum) sum = svadd_f32_x(svptrue_b32(), sum, acc##idx);

template <>
struct CSRVectorKernel<float, 1>
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
        const IdxType rowE = (rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT);

        const uint32_t vl = svcntw();  // runtime vector length

        for (IdxType i = rowS; i < rowE; ++i)
        {
            const IdxType rowStart = csrRowIdices[i];
            const IdxType rowEnd = csrRowIdices[i + 1];
            const IdxType nnz = rowEnd - rowStart;

            const IdxType full = nnz / vl;
            const IdxType tail = nnz % vl;

            const IdxType *pCol = &csrColIdices[rowStart];
            const float *pVal = &csrData[rowStart];

            // 使用宏声明所有累加器变量
            UNROLL_N(DECLARE_ACC)

            // 主循环：每次展开 CSR_UNROLL_FACTOR 次
            IdxType blk = 0;
            for (; blk + CSR_UNROLL_FACTOR - 1 < full; blk += CSR_UNROLL_FACTOR)
            {
                // 使用宏定义来展开循环
                UNROLL_N(COMPUTE_ITERATION, pCol, pVal, vl, vec)

                pCol += CSR_UNROLL_FACTOR * vl;
                pVal += CSR_UNROLL_FACTOR * vl;
            }

            // 处理剩余的整块
            for (; blk < full; blk++)
            {
                svint32_t idx = svld1_s32(svptrue_b32(), pCol);
                svfloat32_t a = svld1(svptrue_b32(), pVal);
                svfloat32_t x = svld1_gather_index(svptrue_b32(), vec, idx);
                acc0 = svmla_f32_x(svptrue_b32(), acc0, a, x);
                pCol += vl;
                pVal += vl;
            }

            // 处理尾巴（一次谓词）
            if (tail)
            {
                svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)tail);
                svint32_t idx = svld1_s32(pg, pCol);
                svfloat32_t a = svld1(pg, pVal);
                svfloat32_t x = svld1_gather_index(pg, vec, idx);
                acc0 = svmla_f32_m(pg, acc0, a, x);
            }

            // 合并所有累加链 - 从acc0开始，依次加上其他累加器
            svfloat32_t sum = acc0;
#if CSR_UNROLL_FACTOR >= 2
            ADD_TO_SUM(1, sum)
#endif
#if CSR_UNROLL_FACTOR >= 4
            ADD_TO_SUM(2, sum)
            ADD_TO_SUM(3, sum)
#endif
#if CSR_UNROLL_FACTOR >= 8
            ADD_TO_SUM(4, sum)
            ADD_TO_SUM(5, sum)
            ADD_TO_SUM(6, sum)
            ADD_TO_SUM(7, sum)
#endif
#if CSR_UNROLL_FACTOR >= 16
            ADD_TO_SUM(8, sum)
            ADD_TO_SUM(9, sum)
            ADD_TO_SUM(10, sum)
            ADD_TO_SUM(11, sum)
            ADD_TO_SUM(12, sum)
            ADD_TO_SUM(13, sum)
            ADD_TO_SUM(14, sum)
            ADD_TO_SUM(15, sum)
#endif

            out[i] = svaddv_f32(svptrue_b32(), sum);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// Version 2: [AI Generated] CSRVectorKernel with Data Prefetching
/////////////////////////////////////////////////////////////////////////////////////////

// 重新定义必要的宏（version 3和4需要使用）
#define DECLARE_ACC(idx, ...) svfloat32_t acc##idx = svdup_f32(0.0f);

#define COMPUTE_ITERATION(idx, pCol, pVal, vl, vec)                                  \
    do                                                                               \
    {                                                                                \
        svint32_t colIdx##idx = svld1_s32(svptrue_b32(), pCol + (idx)*vl);           \
        svfloat32_t matData##idx = svld1(svptrue_b32(), pVal + (idx)*vl);            \
        svfloat32_t vecData##idx =                                                   \
            svld1_gather_index(svptrue_b32(), vec, colIdx##idx);                     \
        acc##idx = svmla_f32_x(svptrue_b32(), acc##idx, matData##idx, vecData##idx); \
    }                                                                                \
    while (0);

#define ADD_TO_SUM(idx, sum)  sum = svadd_f32_x(svptrue_b32(), sum, acc##idx);

// 重新定义UNROLL宏
#define UNROLL_1(ACTION, ...) ACTION(0, ##__VA_ARGS__)
#define UNROLL_2(ACTION, ...) UNROLL_1(ACTION, ##__VA_ARGS__) ACTION(1, ##__VA_ARGS__)
#define UNROLL_4(ACTION, ...) \
    UNROLL_2(ACTION, ##__VA_ARGS__) ACTION(2, ##__VA_ARGS__) ACTION(3, ##__VA_ARGS__)
#define UNROLL_8(ACTION, ...)       \
    UNROLL_4(ACTION, ##__VA_ARGS__) \
    ACTION(4, ##__VA_ARGS__)        \
    ACTION(5, ##__VA_ARGS__) ACTION(6, ##__VA_ARGS__) ACTION(7, ##__VA_ARGS__)
#define UNROLL_16(ACTION, ...)      \
    UNROLL_8(ACTION, ##__VA_ARGS__) \
    ACTION(8, ##__VA_ARGS__)        \
    ACTION(9, ##__VA_ARGS__)        \
    ACTION(10, ##__VA_ARGS__)       \
    ACTION(11, ##__VA_ARGS__)       \
    ACTION(12, ##__VA_ARGS__)       \
    ACTION(13, ##__VA_ARGS__) ACTION(14, ##__VA_ARGS__) ACTION(15, ##__VA_ARGS__)

// 根据CSR_UNROLL_FACTOR选择对应的展开宏
#if CSR_UNROLL_FACTOR == 1
#define UNROLL_N(ACTION, ...) UNROLL_1(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 2
#define UNROLL_N(ACTION, ...) UNROLL_2(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 4
#define UNROLL_N(ACTION, ...) UNROLL_4(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 8
#define UNROLL_N(ACTION, ...) UNROLL_8(ACTION, ##__VA_ARGS__)
#elif CSR_UNROLL_FACTOR == 16
#define UNROLL_N(ACTION, ...) UNROLL_16(ACTION, ##__VA_ARGS__)
#else
#error "Unsupported CSR_UNROLL_FACTOR. Supported values: 1, 2, 4, 8, 16"
#endif

// 定义数据预取相关的宏
#ifndef CSR_PREFETCH_DISTANCE
#define CSR_PREFETCH_DISTANCE 64  // 预取距离，以字节为单位
#endif

#ifndef CSR_PREFETCH_ROWS_AHEAD
#define CSR_PREFETCH_ROWS_AHEAD 4  // 提前预取的行数
#endif

// 预取策略选择
#ifndef CSR_PREFETCH_STRATEGY
#define CSR_PREFETCH_STRATEGY 2  // 0: 基础预取, 1: 自适应预取, 2: 分层预取
#endif

template <>
struct CSRVectorKernel<float, 2>
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
        const IdxType rowE = (rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT);

        const uint32_t vl = svcntw();  // runtime vector length

        for (IdxType i = rowS; i < rowE; ++i)
        {
            const IdxType rowStart = csrRowIdices[i];
            const IdxType rowEnd = csrRowIdices[i + 1];
            const IdxType nnz = rowEnd - rowStart;

            // 预取未来几行的数据
#if CSR_PREFETCH_STRATEGY == 0
            // 基础预取策略：预取当前行的数据和未来行的索引
            if (i + CSR_PREFETCH_ROWS_AHEAD < rowE)
            {
                const IdxType prefetchRowStart =
                    csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD];
                const IdxType prefetchRowEnd =
                    csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD + 1];

                // 预取CSR数据和列索引
                arm_prefetch(&csrData[prefetchRowStart], PREFETCH_READ, PREFETCH_L1);
                arm_prefetch(
                    &csrColIdices[prefetchRowStart], PREFETCH_READ, PREFETCH_L1);

                // 如果行比较长，预取多个缓存行
                if (prefetchRowEnd - prefetchRowStart >
                    CSR_PREFETCH_DISTANCE / sizeof(float))
                {
                    arm_prefetch(
                        &csrData
                            [prefetchRowStart + CSR_PREFETCH_DISTANCE / sizeof(float)],
                        PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(
                        &csrColIdices
                            [prefetchRowStart +
                             CSR_PREFETCH_DISTANCE / sizeof(IdxType)],
                        PREFETCH_READ, PREFETCH_L1);
                }
            }
#elif CSR_PREFETCH_STRATEGY == 1
            // 自适应预取策略：根据行的稀疏度调整预取强度
            if (i + CSR_PREFETCH_ROWS_AHEAD < rowE)
            {
                const IdxType prefetchRowStart =
                    csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD];
                const IdxType prefetchRowEnd =
                    csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD + 1];
                const IdxType prefetchNnz = prefetchRowEnd - prefetchRowStart;

                if (prefetchNnz > 0)
                {
                    // 预取CSR数据和列索引到L1
                    arm_prefetch(
                        &csrData[prefetchRowStart], PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(
                        &csrColIdices[prefetchRowStart], PREFETCH_READ, PREFETCH_L1);

                    // 对于较长的行，使用更激进的预取
                    if (prefetchNnz > vl * 2)
                    {
                        const IdxType prefetchStep =
                            CSR_PREFETCH_DISTANCE / sizeof(float);
                        for (IdxType j = prefetchRowStart; j < prefetchRowEnd;
                             j += prefetchStep)
                        {
                            arm_prefetch(&csrData[j], PREFETCH_READ, PREFETCH_L1);
                            arm_prefetch(&csrColIdices[j], PREFETCH_READ, PREFETCH_L1);
                        }
                    }

                    // 预取向量数据（基于列索引的模式）
                    if (prefetchNnz <= vl)  // 对于短行，预取所有相关的向量元素
                    {
                        for (IdxType j = prefetchRowStart; j < prefetchRowEnd; ++j)
                        {
                            arm_prefetch(
                                &vec[csrColIdices[j]], PREFETCH_READ, PREFETCH_L1);
                        }
                    }
                }
            }
#elif CSR_PREFETCH_STRATEGY == 2
            // 分层预取策略：L1预取近期数据，L2预取远期数据
            if (i + CSR_PREFETCH_ROWS_AHEAD < rowE)
            {
                const IdxType prefetchRowStart =
                    csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD];
                const IdxType prefetchRowEnd =
                    csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD + 1];

                // L1预取：当前要处理的行
                arm_prefetch(&csrData[prefetchRowStart], PREFETCH_READ, PREFETCH_L1);
                arm_prefetch(
                    &csrColIdices[prefetchRowStart], PREFETCH_READ, PREFETCH_L1);

                // L2预取：更远的行
                if (i + CSR_PREFETCH_ROWS_AHEAD * 2 < rowE)
                {
                    const IdxType prefetchRowStart2 =
                        csrRowIdices[i + CSR_PREFETCH_ROWS_AHEAD * 2];
                    arm_prefetch(
                        &csrData[prefetchRowStart2], PREFETCH_READ, PREFETCH_L2);
                    arm_prefetch(
                        &csrColIdices[prefetchRowStart2], PREFETCH_READ, PREFETCH_L2);
                }
            }
#endif

            const IdxType full = nnz / vl;
            const IdxType tail = nnz % vl;

            const IdxType *pCol = &csrColIdices[rowStart];
            const float *pVal = &csrData[rowStart];

            // 使用与version 2相同的宏声明所有累加器变量
            UNROLL_N(DECLARE_ACC)

            // 主循环：每次展开 CSR_UNROLL_FACTOR 次，并加入预取
            IdxType blk = 0;
            for (; blk + CSR_UNROLL_FACTOR - 1 < full; blk += CSR_UNROLL_FACTOR)
            {
                // 预取下一轮循环需要的数据
                if (blk + CSR_UNROLL_FACTOR * 2 < full)
                {
                    const IdxType prefetchOffset = (blk + CSR_UNROLL_FACTOR * 2) * vl;
                    arm_prefetch(pCol + prefetchOffset, PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(pVal + prefetchOffset, PREFETCH_READ, PREFETCH_L1);
                }

                // 使用宏定义来展开循环（与version 2相同）
                UNROLL_N(COMPUTE_ITERATION, pCol, pVal, vl, vec)

                pCol += CSR_UNROLL_FACTOR * vl;
                pVal += CSR_UNROLL_FACTOR * vl;
            }

            // 处理剩余的整块
            for (; blk < full; blk++)
            {
                // 预取下一块数据
                if (blk + 1 < full)
                {
                    arm_prefetch(pCol + vl, PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(pVal + vl, PREFETCH_READ, PREFETCH_L1);
                }

                svint32_t idx = svld1_s32(svptrue_b32(), pCol);
                svfloat32_t a = svld1(svptrue_b32(), pVal);
                svfloat32_t x = svld1_gather_index(svptrue_b32(), vec, idx);
                acc0 = svmla_f32_x(svptrue_b32(), acc0, a, x);
                pCol += vl;
                pVal += vl;
            }

            // 处理尾巴（一次谓词）
            if (tail)
            {
                svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)tail);
                svint32_t idx = svld1_s32(pg, pCol);
                svfloat32_t a = svld1(pg, pVal);
                svfloat32_t x = svld1_gather_index(pg, vec, idx);
                acc0 = svmla_f32_m(pg, acc0, a, x);
            }

            // 合并所有累加链（与version 2相同）
            svfloat32_t sum = acc0;
#if CSR_UNROLL_FACTOR >= 2
            ADD_TO_SUM(1, sum)
#endif
#if CSR_UNROLL_FACTOR >= 4
            ADD_TO_SUM(2, sum)
            ADD_TO_SUM(3, sum)
#endif
#if CSR_UNROLL_FACTOR >= 8
            ADD_TO_SUM(4, sum)
            ADD_TO_SUM(5, sum)
            ADD_TO_SUM(6, sum)
            ADD_TO_SUM(7, sum)
#endif
#if CSR_UNROLL_FACTOR >= 16
            ADD_TO_SUM(8, sum)
            ADD_TO_SUM(9, sum)
            ADD_TO_SUM(10, sum)
            ADD_TO_SUM(11, sum)
            ADD_TO_SUM(12, sum)
            ADD_TO_SUM(13, sum)
            ADD_TO_SUM(14, sum)
            ADD_TO_SUM(15, sum)
#endif

            out[i] = svaddv_f32(svptrue_b32(), sum);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// Version 3: [AI Generated] Advanced CSRVectorKernel with Software Pipeline Prefetching
/////////////////////////////////////////////////////////////////////////////////////////

// 高级预取配置宏
#ifndef CSR_PIPELINE_DEPTH
#define CSR_PIPELINE_DEPTH 3  // 软件流水线深度
#endif

#ifndef CSR_VECTOR_PREFETCH_LOOKAHEAD
#define CSR_VECTOR_PREFETCH_LOOKAHEAD 8  // 向量数据预取前瞻数量
#endif

template <>
struct CSRVectorKernel<float, 3>
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
        const IdxType rowE = (rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT);

        const uint32_t vl = svcntw();  // runtime vector length

        // 软件流水线：预取多行的信息
        struct PipelineStage
        {
            IdxType rowStart;
            IdxType rowEnd;
            IdxType nnz;
            bool valid;
        };

        PipelineStage pipeline[CSR_PIPELINE_DEPTH];

        // 初始化流水线
        for (int stage = 0; stage < CSR_PIPELINE_DEPTH; ++stage)
        {
            if (rowS + stage < rowE)
            {
                pipeline[stage].rowStart = csrRowIdices[rowS + stage];
                pipeline[stage].rowEnd = csrRowIdices[rowS + stage + 1];
                pipeline[stage].nnz = pipeline[stage].rowEnd - pipeline[stage].rowStart;
                pipeline[stage].valid = true;

                // 预取流水线阶段的数据
                arm_prefetch(
                    &csrData[pipeline[stage].rowStart], PREFETCH_READ, PREFETCH_L1);
                arm_prefetch(
                    &csrColIdices[pipeline[stage].rowStart], PREFETCH_READ,
                    PREFETCH_L1);
            }
            else
            {
                pipeline[stage].valid = false;
            }
        }

        for (IdxType i = rowS; i < rowE; ++i)
        {
            // 从流水线获取当前行信息
            const IdxType rowStart = pipeline[0].rowStart;
            const IdxType rowEnd = pipeline[0].rowEnd;
            const IdxType nnz = pipeline[0].nnz;

            // 向前推进流水线
            for (int stage = 0; stage < CSR_PIPELINE_DEPTH - 1; ++stage)
            {
                pipeline[stage] = pipeline[stage + 1];
            }

            // 为流水线末端填入新行
            if (i + CSR_PIPELINE_DEPTH < rowE)
            {
                const IdxType newRowIdx = i + CSR_PIPELINE_DEPTH;
                pipeline[CSR_PIPELINE_DEPTH - 1].rowStart = csrRowIdices[newRowIdx];
                pipeline[CSR_PIPELINE_DEPTH - 1].rowEnd = csrRowIdices[newRowIdx + 1];
                pipeline[CSR_PIPELINE_DEPTH - 1].nnz =
                    pipeline[CSR_PIPELINE_DEPTH - 1].rowEnd -
                    pipeline[CSR_PIPELINE_DEPTH - 1].rowStart;
                pipeline[CSR_PIPELINE_DEPTH - 1].valid = true;

                // 预取新阶段的数据
                arm_prefetch(
                    &csrData[pipeline[CSR_PIPELINE_DEPTH - 1].rowStart], PREFETCH_READ,
                    PREFETCH_L1);
                arm_prefetch(
                    &csrColIdices[pipeline[CSR_PIPELINE_DEPTH - 1].rowStart],
                    PREFETCH_READ, PREFETCH_L1);

                // 如果行很长，预取多个缓存行
                if (pipeline[CSR_PIPELINE_DEPTH - 1].nnz >
                    CSR_PREFETCH_DISTANCE / sizeof(float))
                {
                    const IdxType midPoint = pipeline[CSR_PIPELINE_DEPTH - 1].rowStart +
                                             pipeline[CSR_PIPELINE_DEPTH - 1].nnz / 2;
                    arm_prefetch(&csrData[midPoint], PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(&csrColIdices[midPoint], PREFETCH_READ, PREFETCH_L1);
                }
            }
            else
            {
                pipeline[CSR_PIPELINE_DEPTH - 1].valid = false;
            }

            const IdxType full = nnz / vl;
            const IdxType tail = nnz % vl;

            const IdxType *pCol = &csrColIdices[rowStart];
            const float *pVal = &csrData[rowStart];

            // 智能向量预取：基于列索引模式预取向量数据
            if (nnz <= CSR_VECTOR_PREFETCH_LOOKAHEAD * vl)
            {
                // 对于相对较短的行，预取所有相关的向量元素
                svbool_t pg_prefetch = svwhilelt_b32((uint32_t)0, (uint32_t)nnz);
                svint32_t col_indices = svld1_s32(pg_prefetch, pCol);

                // 使用gather预取向量数据到L1缓存
                for (uint32_t lane = 0; lane < svcntw() && lane < nnz; ++lane)
                {
                    IdxType col_idx = pCol[lane];
                    arm_prefetch(&vec[col_idx], PREFETCH_READ, PREFETCH_L1);
                }
            }
            else
            {
                // 对于较长的行，使用采样预取策略
                const IdxType sampleStep = nnz / CSR_VECTOR_PREFETCH_LOOKAHEAD;
                for (IdxType sample = 0; sample < nnz; sample += sampleStep)
                {
                    IdxType col_idx = pCol[sample];
                    arm_prefetch(&vec[col_idx], PREFETCH_READ, PREFETCH_L1);
                }
            }

            // 使用与version 2相同的宏声明所有累加器变量
            UNROLL_N(DECLARE_ACC)

            // 主循环：每次展开 CSR_UNROLL_FACTOR 次，带智能预取
            IdxType blk = 0;
            for (; blk + CSR_UNROLL_FACTOR - 1 < full; blk += CSR_UNROLL_FACTOR)
            {
                // 多级预取策略
                if (blk + CSR_UNROLL_FACTOR * 2 < full)
                {
                    // L1预取：下一轮循环的数据
                    const IdxType prefetchOffset = (blk + CSR_UNROLL_FACTOR * 2) * vl;
                    arm_prefetch(pCol + prefetchOffset, PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(pVal + prefetchOffset, PREFETCH_READ, PREFETCH_L1);

                    // 预取向量数据
                    for (int pf_idx = 0; pf_idx < CSR_UNROLL_FACTOR &&
                                         prefetchOffset + pf_idx * vl < nnz;
                         ++pf_idx)
                    {
                        for (uint32_t lane = 0;
                             lane < vl && prefetchOffset + pf_idx * vl + lane < nnz;
                             ++lane)
                        {
                            IdxType col_idx = pCol[prefetchOffset + pf_idx * vl + lane];
                            if (col_idx < k)  // 边界检查
                            {
                                arm_prefetch(&vec[col_idx], PREFETCH_READ, PREFETCH_L1);
                            }
                        }
                    }
                }

                if (blk + CSR_UNROLL_FACTOR * 4 < full)
                {
                    // L2预取：更远的数据
                    const IdxType prefetchOffset2 = (blk + CSR_UNROLL_FACTOR * 4) * vl;
                    arm_prefetch(pCol + prefetchOffset2, PREFETCH_READ, PREFETCH_L2);
                    arm_prefetch(pVal + prefetchOffset2, PREFETCH_READ, PREFETCH_L2);
                }

                // 使用宏定义来展开循环（与version 2相同）
                UNROLL_N(COMPUTE_ITERATION, pCol, pVal, vl, vec)

                pCol += CSR_UNROLL_FACTOR * vl;
                pVal += CSR_UNROLL_FACTOR * vl;
            }

            // 处理剩余的整块，带预取
            for (; blk < full; blk++)
            {
                // 预取下一块数据
                if (blk + 2 < full)
                {
                    arm_prefetch(pCol + 2 * vl, PREFETCH_READ, PREFETCH_L1);
                    arm_prefetch(pVal + 2 * vl, PREFETCH_READ, PREFETCH_L1);
                }

                svint32_t idx = svld1_s32(svptrue_b32(), pCol);
                svfloat32_t a = svld1(svptrue_b32(), pVal);
                svfloat32_t x = svld1_gather_index(svptrue_b32(), vec, idx);
                acc0 = svmla_f32_x(svptrue_b32(), acc0, a, x);
                pCol += vl;
                pVal += vl;
            }

            // 处理尾巴（一次谓词）
            if (tail)
            {
                svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)tail);
                svint32_t idx = svld1_s32(pg, pCol);
                svfloat32_t a = svld1(pg, pVal);
                svfloat32_t x = svld1_gather_index(pg, vec, idx);
                acc0 = svmla_f32_m(pg, acc0, a, x);
            }

            // 合并所有累加链（与version 2相同）
            svfloat32_t sum = acc0;
#if CSR_UNROLL_FACTOR >= 2
            ADD_TO_SUM(1, sum)
#endif
#if CSR_UNROLL_FACTOR >= 4
            ADD_TO_SUM(2, sum)
            ADD_TO_SUM(3, sum)
#endif
#if CSR_UNROLL_FACTOR >= 8
            ADD_TO_SUM(4, sum)
            ADD_TO_SUM(5, sum)
            ADD_TO_SUM(6, sum)
            ADD_TO_SUM(7, sum)
#endif
#if CSR_UNROLL_FACTOR >= 16
            ADD_TO_SUM(8, sum)
            ADD_TO_SUM(9, sum)
            ADD_TO_SUM(10, sum)
            ADD_TO_SUM(11, sum)
            ADD_TO_SUM(12, sum)
            ADD_TO_SUM(13, sum)
            ADD_TO_SUM(14, sum)
            ADD_TO_SUM(15, sum)
#endif

            out[i] = svaddv_f32(svptrue_b32(), sum);
        }
    }
};

// 清理版本2和3使用的宏定义
#undef DECLARE_ACC
#undef COMPUTE_ITERATION
#undef ADD_TO_SUM
#undef UNROLL_N
#undef UNROLL_1
#undef UNROLL_2
#undef UNROLL_4
#undef UNROLL_8
#undef UNROLL_16

/////////////////////////////////////////////////////////////////////////////////////////
// Version 4: Advanced CSRVectorKernel with hand-crafted Software Pipeline [TODO]
/////////////////////////////////////////////////////////////////////////////////////////

template <>
struct CSRVectorKernel<float, 4>
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
        const IdxType rowE = (rt.tid == ThreadNum - 1 ? m : rowS + rowsPerT);

        const uint32_t vl = svcntw();  // runtime vector length

        for (IdxType i = rowS; i < rowE; ++i)
        {
            const IdxType rowStart = csrRowIdices[i];
            const IdxType rowEnd = csrRowIdices[i + 1];
            const IdxType nnz = rowEnd - rowStart;

            const IdxType full = nnz / vl;
            const IdxType tail = nnz % vl;

            const IdxType *pCol = &csrColIdices[rowStart];
            const float *pVal = &csrData[rowStart];

            svfloat32_t acc0 = svdup_f32(0.0f);

            // 3-Stage Pipeline
            if (full > 2)
            {
                // [Fill pipeline]
                // Stage 1: load column indices and matrix values
                svint32_t idx0 = svld1_s32(svptrue_b32(), pCol);
                svfloat32_t mat0 = svld1(svptrue_b32(), pVal);

                // Stage 2: load vector values
                svfloat32_t vec0 = svld1_gather_index(svptrue_b32(), vec, idx0);
                idx0 = svld1_s32(svptrue_b32(), pCol + 1 * vl);

                IdxType blk = 2;
                for (; blk < full; blk++)
                {
                    // Stage 3: compute and store
                    acc0 = svmla_f32_m(svptrue_b32(), acc0, mat0, vec0);  // blk-2
                    vec0 = svld1_gather_index(svptrue_b32(), vec, idx0);  // blk-1
                    idx0 = svld1_s32(svptrue_b32(), pCol + blk * vl);  // blk
                    mat0 = svld1(svptrue_b32(), pVal + (blk - 1) * vl);  // blk-1
                }

                // [Empty pipeline]
                acc0 = svmla_f32_m(svptrue_b32(), acc0, mat0, vec0);
                vec0 = svld1_gather_index(svptrue_b32(), vec, idx0);
                mat0 = svld1(svptrue_b32(), pVal + blk * vl);

                acc0 = svmla_f32_m(svptrue_b32(), acc0, mat0, vec0);
            }
            else
            {
                for (IdxType blk = 0; blk < full; blk++)
                {
                    svint32_t idx = svld1_s32(svptrue_b32(), pCol + blk * vl);
                    svfloat32_t a = svld1(svptrue_b32(), pVal + blk * vl);
                    svfloat32_t x = svld1_gather_index(svptrue_b32(), vec, idx);
                    acc0 = svmla_f32_m(svptrue_b32(), acc0, a, x);
                }
            }

            if (tail)
            {
                svbool_t pg = svwhilelt_b32(uint32_t(0), uint32_t(tail));
                svint32_t idx = svld1_s32(pg, pCol + full * vl);
                svfloat32_t a = svld1(pg, pVal + full * vl);
                svfloat32_t x = svld1_gather_index(pg, vec, idx);
                acc0 = svmla_f32_m(pg, acc0, a, x);
            }

            out[i] = svaddv_f32(svptrue_b32(), acc0);
        }
    }
};

// #endif

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
                CSRVectorKernel<DType, 4>::Run(
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
