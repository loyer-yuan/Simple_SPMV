#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

/*
SVE Intrinsic 函数说明 - 需要包含的头文件:
#include <arm_sve.h>

主要 SVE 类型:
- svbool_t      : 谓词寄存器类型
- svfloat64_t   : 64位浮点向量类型
- svuint64_t    : 64位整数向量类型

主要 SVE 函数:
- svptrue_b64()                                          : 创建全真谓词 (ptrue p0.d)
- svcntd()                                               : 获取向量中double元素数量
- svdup_f64(value)                                       : 创建重复值向量 (fmov z.d, #value)
- svld1_f64(pg, ptr)                                     : 基本加载 (ld1d {z.d}, pg/z, [ptr])
- svld1_vnum_f64(pg, ptr, offset)                        : 偏移加载 (ld1d {z.d}, pg/z, [ptr, #offset, mul vl])
- svst1_f64(pg, ptr, data)                               : 基本存储 (st1d {z.d}, pg, [ptr])
- svst1_vnum_f64(pg, ptr, offset, data)                  : 偏移存储 (st1d {z.d}, pg, [ptr, #offset, mul vl])
- svld1_gather_u64offset_f64(pg, base, offsets)          : Gather加载 (ld1d {z.d}, pg/z, [base, offsets.d])
- svst1_scatter_u64offset_f64(pg, base, offsets, data)   : Scatter存储 (st1d {z.d}, pg, [base, offsets.d])
*/

namespace xsparse {

class Timer
{
public:
    using Clock = std::chrono::steady_clock;

    Timer() = default;

    void start()
    {
        isRunning = true;
        startT = Clock::now();
    }

    // 结束计时
    void stop()
    {
        if (isRunning)
        {
            endT = Clock::now();
            isRunning = false;
        }
    }

    template <typename Duration = std::chrono::milliseconds>
    typename Duration::rep elapsed() const
    {
        return isRunning ?
                   std::chrono::duration_cast<Duration>(Clock::now() - startT).count() :
                   std::chrono::duration_cast<Duration>(endT - startT).count();
    }

    // 重置计时器
    void reset()
    {
        isRunning = false;
        startT = endT = Clock::now();
    }

    // 重启计时器
    void restart()
    {
        reset();
        start();
    }

private:
    bool isRunning = false;
    std::chrono::time_point<Clock> startT;
    std::chrono::time_point<Clock> endT;
};
}  // namespace xsparse

// clang-format off
// SVE 内存加载指令宏 - 8个向量寄存器
#define SVE_LOAD8(ptr_reg)             \
    "ld1d {z0.d}, p0/z, [" ptr_reg ", #0, mul vl]\n\t"    \
    "ld1d {z1.d}, p0/z, [" ptr_reg ", #1, mul vl]\n\t"    \
    "ld1d {z2.d}, p0/z, [" ptr_reg ", #2, mul vl]\n\t"    \
    "ld1d {z3.d}, p0/z, [" ptr_reg ", #3, mul vl]\n\t"    \
    "ld1d {z4.d}, p0/z, [" ptr_reg ", #4, mul vl]\n\t"    \
    "ld1d {z5.d}, p0/z, [" ptr_reg ", #5, mul vl]\n\t"    \
    "ld1d {z6.d}, p0/z, [" ptr_reg ", #6, mul vl]\n\t"    \
    "ld1d {z7.d}, p0/z, [" ptr_reg ", #7, mul vl]\n\t"    \
    "add " ptr_reg ", " ptr_reg ", x1\n\t"

// SVE 内存存储指令宏 - 8个向量寄存器
#define SVE_STORE8(ptr_reg)            \
    "st1d {z0.d}, p0, [" ptr_reg ", #0, mul vl]\n\t"      \
    "st1d {z1.d}, p0, [" ptr_reg ", #1, mul vl]\n\t"      \
    "st1d {z2.d}, p0, [" ptr_reg ", #2, mul vl]\n\t"      \
    "st1d {z3.d}, p0, [" ptr_reg ", #3, mul vl]\n\t"      \
    "st1d {z4.d}, p0, [" ptr_reg ", #4, mul vl]\n\t"      \
    "st1d {z5.d}, p0, [" ptr_reg ", #5, mul vl]\n\t"      \
    "st1d {z6.d}, p0, [" ptr_reg ", #6, mul vl]\n\t"      \
    "st1d {z7.d}, p0, [" ptr_reg ", #7, mul vl]\n\t"      \
    "add " ptr_reg ", " ptr_reg ", x1\n\t"

// SVE 内存拷贝指令宏 - 同时加载和存储
#define SVE_COPY8(src_reg, dst_reg)    \
    "ld1d {z0.d}, p0/z, [" src_reg ", #0, mul vl]\n\t"    \
    "ld1d {z1.d}, p0/z, [" src_reg ", #1, mul vl]\n\t"    \
    "ld1d {z2.d}, p0/z, [" src_reg ", #2, mul vl]\n\t"    \
    "ld1d {z3.d}, p0/z, [" src_reg ", #3, mul vl]\n\t"    \
    "ld1d {z4.d}, p0/z, [" src_reg ", #4, mul vl]\n\t"    \
    "ld1d {z5.d}, p0/z, [" src_reg ", #5, mul vl]\n\t"    \
    "ld1d {z6.d}, p0/z, [" src_reg ", #6, mul vl]\n\t"    \
    "ld1d {z7.d}, p0/z, [" src_reg ", #7, mul vl]\n\t"    \
    "st1d {z0.d}, p0, [" dst_reg ", #0, mul vl]\n\t"      \
    "st1d {z1.d}, p0, [" dst_reg ", #1, mul vl]\n\t"      \
    "st1d {z2.d}, p0, [" dst_reg ", #2, mul vl]\n\t"      \
    "st1d {z3.d}, p0, [" dst_reg ", #3, mul vl]\n\t"      \
    "st1d {z4.d}, p0, [" dst_reg ", #4, mul vl]\n\t"      \
    "st1d {z5.d}, p0, [" dst_reg ", #5, mul vl]\n\t"      \
    "st1d {z6.d}, p0, [" dst_reg ", #6, mul vl]\n\t"      \
    "st1d {z7.d}, p0, [" dst_reg ", #7, mul vl]\n\t"      \
    "add " src_reg ", " src_reg ", x1\n\t"               \
    "add " dst_reg ", " dst_reg ", x1\n\t"

// clang-format on

// 获取SVE向量长度（以字节为单位）
size_t get_sve_vector_length()
{
    size_t vl;
    __asm__ volatile("rdvl %[vl], #1\n\t" : [vl] "=r"(vl) : :);
    return vl;
}

// 生成随机访问索引数组
std::vector<size_t> generate_random_indices(
    size_t array_size, size_t num_accesses, size_t vector_size_bytes)
{
    // 确保索引对齐到向量边界，避免跨越向量边界的访问
    const size_t elements_per_vector = vector_size_bytes / sizeof(double);
    const size_t max_vector_index =
        (array_size / elements_per_vector) - 8;  // 减8是为了确保8个向量的连续访问不越界

    std::vector<size_t> indices;
    indices.reserve(num_accesses);

    // 使用固定种子以确保结果可重复
    std::mt19937 gen(12345);
    std::uniform_int_distribution<size_t> dis(0, max_vector_index - 1);

    for (size_t i = 0; i < num_accesses; ++i)
    {
        // 生成向量对齐的随机索引（以元素为单位）
        size_t vector_idx = dis(gen);
        size_t element_idx = vector_idx * elements_per_vector;
        indices.push_back(element_idx);
    }

    return indices;
}

// 生成用于 gather 操作的字节偏移量向量
std::vector<std::vector<uint64_t>> generate_gather_offsets(
    size_t array_size, size_t num_vectors, size_t vector_length_bytes)
{
    const size_t elements_per_vector = vector_length_bytes / sizeof(double);
    const size_t max_element_index = array_size - 1;

    std::vector<std::vector<uint64_t>> offset_vectors;
    offset_vectors.reserve(num_vectors);

    // 使用固定种子以确保结果可重复
    std::mt19937 gen(54321);
    std::uniform_int_distribution<size_t> dis(0, max_element_index);

    for (size_t v = 0; v < num_vectors; ++v)
    {
        std::vector<uint64_t> offsets(elements_per_vector);
        for (size_t i = 0; i < elements_per_vector; ++i)
        {
            // 生成随机元素索引并转换为字节偏移量
            size_t element_idx = dis(gen);
            offsets[i] = element_idx * sizeof(double);
        }
        offset_vectors.push_back(std::move(offsets));
    }

    return offset_vectors;
}

// SVE 顺序读取带宽测试
void MB_SVE_Sequential_Read(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();  // SVE向量长度（字节）
    const size_t elements_per_vector = vl / sizeof(double);  // 每个向量的double元素数
    const size_t total_vectors = 8;  // 一次操作8个向量
    const size_t stride = total_vectors * vl;  // 每轮循环的步长
    const size_t iterations = test_size / stride;  // 总迭代次数

    // 分配内存并初始化
    std::vector<double> data(test_size / sizeof(double), 1.23);
    double *ptr = data.data();

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();  // 设置谓词寄存器全真
    for (size_t iter = 0; iter < iterations; iter += 8) {
        // 每轮加载 8×8=64 个向量，每个向量 vl/8 个double
        for (int round = 0; round < 8; ++round) {
            svfloat64_t z0 = svld1_vnum_f64(pg, ptr, 0);  // ld1d {z0.d}, p0/z, [ptr, #0, mul vl]
            svfloat64_t z1 = svld1_vnum_f64(pg, ptr, 1);  // ld1d {z1.d}, p0/z, [ptr, #1, mul vl]
            svfloat64_t z2 = svld1_vnum_f64(pg, ptr, 2);  // ld1d {z2.d}, p0/z, [ptr, #2, mul vl]
            svfloat64_t z3 = svld1_vnum_f64(pg, ptr, 3);  // ld1d {z3.d}, p0/z, [ptr, #3, mul vl]
            svfloat64_t z4 = svld1_vnum_f64(pg, ptr, 4);  // ld1d {z4.d}, p0/z, [ptr, #4, mul vl]
            svfloat64_t z5 = svld1_vnum_f64(pg, ptr, 5);  // ld1d {z5.d}, p0/z, [ptr, #5, mul vl]
            svfloat64_t z6 = svld1_vnum_f64(pg, ptr, 6);  // ld1d {z6.d}, p0/z, [ptr, #6, mul vl]
            svfloat64_t z7 = svld1_vnum_f64(pg, ptr, 7);  // ld1d {z7.d}, p0/z, [ptr, #7, mul vl]
            ptr += svcntd() * 8;  // 移动指针：add ptr, ptr, stride
        }
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "mov x0, %[iterations]\n\t"         // 迭代次数
        "mov x1, %[step]\n\t"               // 设置步长到x1寄存器
        "1:\n"
        SVE_LOAD8("%[ptr]") SVE_LOAD8("%[ptr]") SVE_LOAD8("%[ptr]") SVE_LOAD8("%[ptr]")  // 4轮，共32个向量加载
        SVE_LOAD8("%[ptr]") SVE_LOAD8("%[ptr]") SVE_LOAD8("%[ptr]") SVE_LOAD8("%[ptr]")
        "subs x0, x0, #8\n\t"              // 每次内层循环处理8轮
        "b.gt 1b\n"
        : [ptr] "+r"(ptr)
        : [iterations] "r"(iterations / 8),
          [step] "r"(stride)
        : "x0", "x1", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = test_size;
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "SVE Vector Length: " << vl << " bytes (" << elements_per_vector
                  << " doubles per vector)" << std::endl;
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Sequential Read Bandwidth: " << bandwidth_gb_s << " GB/s"
                  << std::endl;
    }
}

// SVE 顺序写入带宽测试
void MB_SVE_Sequential_Write(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_vectors = 8;
    const size_t stride = total_vectors * vl;
    const size_t iterations = test_size / stride;

    // 分配内存并预填充（与读测试保持一致）
    std::vector<double> data(test_size / sizeof(double), 2.34);
    double *ptr = data.data();

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();            // 设置谓词寄存器全真
    svfloat64_t z0 = svdup_f64(1.0);        // 初始化向量寄存器为 1.0
    svfloat64_t z1 = svdup_f64(1.0);        // fmov z0-z7.d, #1.0
    svfloat64_t z2 = svdup_f64(1.0);
    svfloat64_t z3 = svdup_f64(1.0);
    svfloat64_t z4 = svdup_f64(1.0);
    svfloat64_t z5 = svdup_f64(1.0);
    svfloat64_t z6 = svdup_f64(1.0);
    svfloat64_t z7 = svdup_f64(1.0);

    for (size_t iter = 0; iter < iterations; iter += 8) {
        // 每轮存储 8×8=64 个向量到内存
        for (int round = 0; round < 8; ++round) {
            svst1_vnum_f64(pg, ptr, 0, z0);    // st1d {z0.d}, p0, [ptr, #0, mul vl]
            svst1_vnum_f64(pg, ptr, 1, z1);    // st1d {z1.d}, p0, [ptr, #1, mul vl]
            svst1_vnum_f64(pg, ptr, 2, z2);    // st1d {z2.d}, p0, [ptr, #2, mul vl]
            svst1_vnum_f64(pg, ptr, 3, z3);    // st1d {z3.d}, p0, [ptr, #3, mul vl]
            svst1_vnum_f64(pg, ptr, 4, z4);    // st1d {z4.d}, p0, [ptr, #4, mul vl]
            svst1_vnum_f64(pg, ptr, 5, z5);    // st1d {z5.d}, p0, [ptr, #5, mul vl]
            svst1_vnum_f64(pg, ptr, 6, z6);    // st1d {z6.d}, p0, [ptr, #6, mul vl]
            svst1_vnum_f64(pg, ptr, 7, z7);    // st1d {z7.d}, p0, [ptr, #7, mul vl]
            ptr += svcntd() * 8;                // 移动指针：add ptr, ptr, stride
        }
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "fmov z0.d, #1.0\n\t"              // 初始化向量寄存器
        "fmov z1.d, #1.0\n\t"
        "fmov z2.d, #1.0\n\t"
        "fmov z3.d, #1.0\n\t"
        "fmov z4.d, #1.0\n\t"
        "fmov z5.d, #1.0\n\t"
        "fmov z6.d, #1.0\n\t"
        "fmov z7.d, #1.0\n\t"
        "mov x0, %[iterations]\n\t"
        "mov x1, %[step]\n\t"               // 设置步长到x1寄存器
        "1:\n"
        SVE_STORE8("%[ptr]") SVE_STORE8("%[ptr]") SVE_STORE8("%[ptr]") SVE_STORE8("%[ptr]")  // 4轮，共32个向量存储
        SVE_STORE8("%[ptr]") SVE_STORE8("%[ptr]") SVE_STORE8("%[ptr]") SVE_STORE8("%[ptr]")
        "subs x0, x0, #8\n\t"
        "b.gt 1b\n"
        : [ptr] "+r"(ptr)
        : [iterations] "r"(iterations / 8),
          [step] "r"(stride)
        : "x0", "x1", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = test_size;
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Sequential Write Bandwidth: " << bandwidth_gb_s << " GB/s"
                  << std::endl;
    }
}

// SVE 内存拷贝带宽测试
void MB_SVE_Memory_Copy(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_vectors = 8;
    const size_t stride = total_vectors * vl;
    const size_t iterations = test_size / stride;

    // 分配源和目标内存
    std::vector<double> src_data(test_size / sizeof(double), 1.23);
    std::vector<double> dst_data(test_size / sizeof(double));
    double *src_ptr = src_data.data();
    double *dst_ptr = dst_data.data();

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();                // 设置谓词寄存器全真

    for (size_t iter = 0; iter < iterations; iter += 8) {
        // 每轮拷贝 8×8=64 个向量：先加载后存储
        for (int round = 0; round < 8; ++round) {
            // 从源地址加载8个向量
            svfloat64_t z0 = svld1_vnum_f64(pg, src_ptr, 0);  // ld1d {z0.d}, p0/z, [src, #0, mul vl]
            svfloat64_t z1 = svld1_vnum_f64(pg, src_ptr, 1);  // ld1d {z1.d}, p0/z, [src, #1, mul vl]
            svfloat64_t z2 = svld1_vnum_f64(pg, src_ptr, 2);  // ld1d {z2.d}, p0/z, [src, #2, mul vl]
            svfloat64_t z3 = svld1_vnum_f64(pg, src_ptr, 3);  // ld1d {z3.d}, p0/z, [src, #3, mul vl]
            svfloat64_t z4 = svld1_vnum_f64(pg, src_ptr, 4);  // ld1d {z4.d}, p0/z, [src, #4, mul vl]
            svfloat64_t z5 = svld1_vnum_f64(pg, src_ptr, 5);  // ld1d {z5.d}, p0/z, [src, #5, mul vl]
            svfloat64_t z6 = svld1_vnum_f64(pg, src_ptr, 6);  // ld1d {z6.d}, p0/z, [src, #6, mul vl]
            svfloat64_t z7 = svld1_vnum_f64(pg, src_ptr, 7);  // ld1d {z7.d}, p0/z, [src, #7, mul vl]

            // 存储8个向量到目标地址
            svst1_vnum_f64(pg, dst_ptr, 0, z0);               // st1d {z0.d}, p0, [dst, #0, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 1, z1);               // st1d {z1.d}, p0, [dst, #1, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 2, z2);               // st1d {z2.d}, p0, [dst, #2, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 3, z3);               // st1d {z3.d}, p0, [dst, #3, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 4, z4);               // st1d {z4.d}, p0, [dst, #4, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 5, z5);               // st1d {z5.d}, p0, [dst, #5, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 6, z6);               // st1d {z6.d}, p0, [dst, #6, mul vl]
            svst1_vnum_f64(pg, dst_ptr, 7, z7);               // st1d {z7.d}, p0, [dst, #7, mul vl]

            src_ptr += svcntd() * 8;                          // 移动源指针
            dst_ptr += svcntd() * 8;                          // 移动目标指针
        }
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "mov x0, %[iterations]\n\t"
        "mov x1, %[step]\n\t"               // 设置步长到x1寄存器
        "1:\n"
        SVE_COPY8("%[src]", "%[dst]") SVE_COPY8("%[src]", "%[dst]") SVE_COPY8("%[src]", "%[dst]") SVE_COPY8("%[src]", "%[dst]")    // 4轮，共32个向量拷贝
        SVE_COPY8("%[src]", "%[dst]") SVE_COPY8("%[src]", "%[dst]") SVE_COPY8("%[src]", "%[dst]") SVE_COPY8("%[src]", "%[dst]")
        "subs x0, x0, #8\n\t"
        "b.gt 1b\n"
        : [src] "+r"(src_ptr), [dst] "+r"(dst_ptr)
        : [iterations] "r"(iterations / 8),
          [step] "r"(stride)
        : "x0", "x1", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    // 内存拷贝涉及读取和写入，总传输量是测试大小的2倍
    const double bytes_transferred = test_size * 2;
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Memory Copy Bandwidth: " << bandwidth_gb_s << " GB/s"
                  << std::endl;
    }
}

// SVE 随机读取带宽测试
void MB_SVE_Random_Read(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, size_t num_random_accesses, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_elements = test_size / sizeof(double);
    // 随机访问次数通过参数传入

    // 分配内存并初始化
    std::vector<double> data(total_elements, 1.23);
    double *base_ptr = data.data();

    // 生成随机访问索引
    auto indices = generate_random_indices(total_elements, num_random_accesses, vl);

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();                      // 设置谓词寄存器全真

    for (size_t access = 0; access < num_random_accesses; ++access) {
        // 计算随机访问地址
        size_t random_idx = indices[access];           // 从预生成的随机索引数组获取
        double* random_addr = base_ptr + random_idx;   // 计算实际内存地址

        // 在随机地址处加载8个连续向量 (测试局部性)
        svfloat64_t z0 = svld1_vnum_f64(pg, random_addr, 0);  // ld1d {z0.d}, p0/z, [addr, #0, mul vl]
        svfloat64_t z1 = svld1_vnum_f64(pg, random_addr, 1);  // ld1d {z1.d}, p0/z, [addr, #1, mul vl]
        svfloat64_t z2 = svld1_vnum_f64(pg, random_addr, 2);  // ld1d {z2.d}, p0/z, [addr, #2, mul vl]
        svfloat64_t z3 = svld1_vnum_f64(pg, random_addr, 3);  // ld1d {z3.d}, p0/z, [addr, #3, mul vl]
        svfloat64_t z4 = svld1_vnum_f64(pg, random_addr, 4);  // ld1d {z4.d}, p0/z, [addr, #4, mul vl]
        svfloat64_t z5 = svld1_vnum_f64(pg, random_addr, 5);  // ld1d {z5.d}, p0/z, [addr, #5, mul vl]
        svfloat64_t z6 = svld1_vnum_f64(pg, random_addr, 6);  // ld1d {z6.d}, p0/z, [addr, #6, mul vl]
        svfloat64_t z7 = svld1_vnum_f64(pg, random_addr, 7);  // ld1d {z7.d}, p0/z, [addr, #7, mul vl]
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "mov x0, %[num_accesses]\n\t"       // 随机访问次数
        "mov x2, %[indices_ptr]\n\t"        // 索引数组指针
        "mov x3, %[base_ptr]\n\t"           // 基地址
        "mov x4, #8\n\t"                    // sizeof(double)
        "1:\n"
        // 加载下一个随机索引
        "ldr x5, [x2], #8\n\t"             // 加载索引并递增指针
        "mul x5, x5, x4\n\t"               // 索引 * sizeof(double) = 字节偏移
        "add x6, x3, x5\n\t"               // 计算实际地址
        // 执行8个向量的随机加载
        "ld1d {z0.d}, p0/z, [x6, #0, mul vl]\n\t"
        "ld1d {z1.d}, p0/z, [x6, #1, mul vl]\n\t"
        "ld1d {z2.d}, p0/z, [x6, #2, mul vl]\n\t"
        "ld1d {z3.d}, p0/z, [x6, #3, mul vl]\n\t"
        "ld1d {z4.d}, p0/z, [x6, #4, mul vl]\n\t"
        "ld1d {z5.d}, p0/z, [x6, #5, mul vl]\n\t"
        "ld1d {z6.d}, p0/z, [x6, #6, mul vl]\n\t"
        "ld1d {z7.d}, p0/z, [x6, #7, mul vl]\n\t"
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [num_accesses] "r"(num_random_accesses),
          [indices_ptr] "r"(indices.data()),
          [base_ptr] "r"(base_ptr)
        : "x0", "x2", "x3", "x4", "x5", "x6", "p0",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = num_random_accesses * 8 * vl;  // 每次访问8个向量
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Random Read Bandwidth: " << bandwidth_gb_s << " GB/s ("
                  << num_random_accesses << " random accesses)" << std::endl;
    }
}

// SVE 随机写入带宽测试
void MB_SVE_Random_Write(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, size_t num_random_accesses, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_elements = test_size / sizeof(double);
    // 随机访问次数通过参数传入

    // 分配内存
    std::vector<double> data(total_elements);
    double *base_ptr = data.data();

    // 生成随机访问索引
    auto indices = generate_random_indices(total_elements, num_random_accesses, vl);

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();                    // 设置谓词寄存器全真
    svfloat64_t z0 = svdup_f64(1.0);                // 初始化向量寄存器为 1.0
    svfloat64_t z1 = svdup_f64(1.0);                // fmov z0-z7.d, #1.0
    svfloat64_t z2 = svdup_f64(1.0);
    svfloat64_t z3 = svdup_f64(1.0);
    svfloat64_t z4 = svdup_f64(1.0);
    svfloat64_t z5 = svdup_f64(1.0);
    svfloat64_t z6 = svdup_f64(1.0);
    svfloat64_t z7 = svdup_f64(1.0);

    for (size_t access = 0; access < num_random_accesses; ++access) {
        // 计算随机访问地址
        size_t random_idx = indices[access];           // 从预生成的随机索引数组获取
        double* random_addr = base_ptr + random_idx;   // 计算实际内存地址

        // 在随机地址处存储8个连续向量
        svst1_vnum_f64(pg, random_addr, 0, z0);        // st1d {z0.d}, p0, [addr, #0, mul vl]
        svst1_vnum_f64(pg, random_addr, 1, z1);        // st1d {z1.d}, p0, [addr, #1, mul vl]
        svst1_vnum_f64(pg, random_addr, 2, z2);        // st1d {z2.d}, p0, [addr, #2, mul vl]
        svst1_vnum_f64(pg, random_addr, 3, z3);        // st1d {z3.d}, p0, [addr, #3, mul vl]
        svst1_vnum_f64(pg, random_addr, 4, z4);        // st1d {z4.d}, p0, [addr, #4, mul vl]
        svst1_vnum_f64(pg, random_addr, 5, z5);        // st1d {z5.d}, p0, [addr, #5, mul vl]
        svst1_vnum_f64(pg, random_addr, 6, z6);        // st1d {z6.d}, p0, [addr, #6, mul vl]
        svst1_vnum_f64(pg, random_addr, 7, z7);        // st1d {z7.d}, p0, [addr, #7, mul vl]
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        // 初始化向量寄存器为简单值
        "fmov z0.d, #1.0\n\t"
        "fmov z1.d, #1.0\n\t"
        "fmov z2.d, #1.0\n\t"
        "fmov z3.d, #1.0\n\t"
        "fmov z4.d, #1.0\n\t"
        "fmov z5.d, #1.0\n\t"
        "fmov z6.d, #1.0\n\t"
        "fmov z7.d, #1.0\n\t"
        "mov x0, %[num_accesses]\n\t"       // 随机访问次数
        "mov x2, %[indices_ptr]\n\t"        // 索引数组指针
        "mov x3, %[base_ptr]\n\t"           // 基地址
        "mov x4, #8\n\t"                    // sizeof(double)
        "1:\n"
        // 加载下一个随机索引
        "ldr x5, [x2], #8\n\t"             // 加载索引并递增指针
        "mul x5, x5, x4\n\t"               // 索引 * sizeof(double) = 字节偏移
        "add x6, x3, x5\n\t"               // 计算实际地址
        // 执行8个向量的随机存储
        "st1d {z0.d}, p0, [x6, #0, mul vl]\n\t"
        "st1d {z1.d}, p0, [x6, #1, mul vl]\n\t"
        "st1d {z2.d}, p0, [x6, #2, mul vl]\n\t"
        "st1d {z3.d}, p0, [x6, #3, mul vl]\n\t"
        "st1d {z4.d}, p0, [x6, #4, mul vl]\n\t"
        "st1d {z5.d}, p0, [x6, #5, mul vl]\n\t"
        "st1d {z6.d}, p0, [x6, #6, mul vl]\n\t"
        "st1d {z7.d}, p0, [x6, #7, mul vl]\n\t"
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [num_accesses] "r"(num_random_accesses),
          [indices_ptr] "r"(indices.data()),
          [base_ptr] "r"(base_ptr)
        : "x0", "x2", "x3", "x4", "x5", "x6", "p0",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = num_random_accesses * 8 * vl;  // 每次访问8个向量
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Random Write Bandwidth: " << bandwidth_gb_s << " GB/s ("
                  << num_random_accesses << " random accesses)" << std::endl;
    }
}

// SVE Gather 读取带宽测试
void MB_SVE_Gather_Read(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, size_t num_gather_ops, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_elements = test_size / sizeof(double);
    // Gather 操作次数通过参数传入

    // 分配内存并初始化
    std::vector<double> data(total_elements, 1.23);
    double *base_ptr = data.data();

    // 生成 gather 偏移量向量
    auto offset_vectors = generate_gather_offsets(total_elements, num_gather_ops, vl);

    // 将偏移量数据转换为连续数组以便汇编访问
    std::vector<uint64_t> flat_offsets;
    flat_offsets.reserve(num_gather_ops * elements_per_vector);
    for (const auto &vec : offset_vectors)
    {
        flat_offsets.insert(flat_offsets.end(), vec.begin(), vec.end());
    }

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();                        // 设置谓词寄存器全真

    for (size_t op = 0; op < num_gather_ops; ++op) {
        // 从预生成的偏移量数组获取当前向量的偏移量
        const uint64_t* current_offsets = &flat_offsets[op * elements_per_vector];

        // 加载偏移量向量 (字节偏移量)
        svuint64_t offset_vec = svld1_u64(pg, current_offsets);    // ld1d {z8.d}, p0/z, [offsets_ptr]

        // 执行 gather 操作：根据偏移量向量从基地址分散加载
        svfloat64_t result = svld1_gather_u64offset_f64(pg, base_ptr, offset_vec);  // ld1d {z0.d}, p0/z, [base, z8.d]

        // result 现在包含从不同内存位置聚集而来的数据
        // 在实际应用中，这里会对 result 进行处理
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "mov x0, %[num_ops]\n\t"            // Gather 操作次数
        "mov x1, %[offsets_ptr]\n\t"        // 偏移量数组指针
        "mov x2, %[base_ptr]\n\t"           // 基地址
        "mov x3, %[vec_size]\n\t"           // 每个向量的元素数量
        "mov x4, #8\n\t"                    // sizeof(uint64_t) for offset stride
        "1:\n"
        // 加载当前向量的偏移量到 z8
        "ld1d {z8.d}, p0/z, [x1]\n\t"      // 加载偏移量向量
        // 执行 gather 操作
        "ld1d {z0.d}, p0/z, [x2, z8.d]\n\t" // gather 加载
        // 移动到下一组偏移量
        "mul x5, x3, x4\n\t"               // 计算偏移量数组的步长
        "add x1, x1, x5\n\t"               // 移动到下一个偏移量向量
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [num_ops] "r"(num_gather_ops),
          [offsets_ptr] "r"(flat_offsets.data()),
          [base_ptr] "r"(base_ptr),
          [vec_size] "r"(elements_per_vector)
        : "x0", "x1", "x2", "x3", "x4", "x5", "p0", "z0", "z8", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = num_gather_ops * vl;  // 每次 gather 加载一个向量
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Gather Read Bandwidth: " << bandwidth_gb_s << " GB/s ("
                  << num_gather_ops << " gather operations)" << std::endl;
        std::cout << "Elements per vector: " << elements_per_vector << std::endl;
    }
}

void MB_SVE_Gather_Read_Unroll4(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, size_t num_gather_ops, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_elements = test_size / sizeof(double);
    // Gather 操作次数通过参数传入（确保是4的倍数）
    const size_t unroll_ops = num_gather_ops / 4;  // 展开后的循环次数

    // 分配内存并初始化
    std::vector<double> data(total_elements, 1.23);
    double *base_ptr = data.data();

    // 生成 gather 偏移量向量
    auto offset_vectors = generate_gather_offsets(total_elements, num_gather_ops, vl);

    // 将偏移量数据转换为连续数组以便汇编访问
    std::vector<uint64_t> flat_offsets;
    flat_offsets.reserve(num_gather_ops * elements_per_vector);
    for (const auto &vec : offset_vectors)
    {
        flat_offsets.insert(flat_offsets.end(), vec.begin(), vec.end());
    }

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明（展开4次版本）:
    svbool_t pg = svptrue_b64();                        // 设置谓词寄存器全真

    for (size_t op = 0; op < unroll_ops; ++op) {
        // 每次循环展开执行4个 gather 操作
        for (int unroll = 0; unroll < 4; ++unroll) {
            size_t current_op = op * 4 + unroll;
            const uint64_t* current_offsets = &flat_offsets[current_op * elements_per_vector];

            // 加载偏移量向量 (字节偏移量)
            svuint64_t offset_vec = svld1_u64(pg, current_offsets);    // ld1d {z8.d}, p0/z, [offsets_ptr]

            // 执行 gather 操作：根据偏移量向量从基地址分散加载
            svfloat64_t result = svld1_gather_u64offset_f64(pg, base_ptr, offset_vec);  // ld1d {z0-z3.d}, p0/z, [base, z8-z11.d]
        }
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "mov x0, %[num_ops]\n\t"            // 展开后的循环次数
        "mov x1, %[offsets_ptr]\n\t"        // 偏移量数组指针
        "mov x2, %[base_ptr]\n\t"           // 基地址
        "mov x3, %[vec_size]\n\t"           // 每个向量的元素数量
        "mov x4, #8\n\t"                    // sizeof(uint64_t) for offset stride
        "1:\n"
        // 第1个 gather 操作
        "ld1d {z8.d}, p0/z, [x1]\n\t"      // 加载第1组偏移量向量
        "ld1d {z0.d}, p0/z, [x2, z8.d]\n\t" // 第1个 gather 加载
        "mul x5, x3, x4\n\t"               // 计算偏移量数组的步长
        "add x1, x1, x5\n\t"               // 移动到下一个偏移量向量

        // 第2个 gather 操作
        "ld1d {z9.d}, p0/z, [x1]\n\t"      // 加载第2组偏移量向量
        "ld1d {z1.d}, p0/z, [x2, z9.d]\n\t" // 第2个 gather 加载
        "add x1, x1, x5\n\t"               // 移动到下一个偏移量向量

        // 第3个 gather 操作
        "ld1d {z10.d}, p0/z, [x1]\n\t"     // 加载第3组偏移量向量
        "ld1d {z2.d}, p0/z, [x2, z10.d]\n\t" // 第3个 gather 加载
        "add x1, x1, x5\n\t"               // 移动到下一个偏移量向量

        // 第4个 gather 操作
        "ld1d {z11.d}, p0/z, [x1]\n\t"     // 加载第4组偏移量向量
        "ld1d {z3.d}, p0/z, [x2, z11.d]\n\t" // 第4个 gather 加载
        "add x1, x1, x5\n\t"               // 移动到下一个偏移量向量

        "subs x0, x0, #1\n\t"              // 递减循环计数器
        "b.gt 1b\n"
        :
        : [num_ops] "r"(unroll_ops),
          [offsets_ptr] "r"(flat_offsets.data()),
          [base_ptr] "r"(base_ptr),
          [vec_size] "r"(elements_per_vector)
        : "x0", "x1", "x2", "x3", "x4", "x5", "p0",
          "z0", "z1", "z2", "z3", "z8", "z9", "z10", "z11", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = num_gather_ops * vl;  // 每次 gather 加载一个向量
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Gather Read Unroll4 Bandwidth: " << bandwidth_gb_s << " GB/s ("
                  << num_gather_ops << " gather operations, 4x unrolled)" << std::endl;
        std::cout << "Elements per vector: " << elements_per_vector << std::endl;
    }
}

// SVE Scatter 写入带宽测试
void MB_SVE_Scatter_Write(
    const float CPU_FREQ_GHZ, size_t memory_size_mb, size_t num_scatter_ops, bool isPrint = false)
{
    const size_t test_size = memory_size_mb * 1024 * 1024;  // 指定大小的测试数据
    const size_t vl = get_sve_vector_length();
    const size_t elements_per_vector = vl / sizeof(double);
    const size_t total_elements = test_size / sizeof(double);
    // Scatter 操作次数通过参数传入

    // 分配内存
    std::vector<double> data(total_elements, 0.0);
    double *base_ptr = data.data();

    // 生成 scatter 偏移量向量
    auto offset_vectors = generate_gather_offsets(total_elements, num_scatter_ops, vl);

    // 将偏移量数据转换为连续数组以便汇编访问
    std::vector<uint64_t> flat_offsets;
    flat_offsets.reserve(num_scatter_ops * elements_per_vector);
    for (const auto &vec : offset_vectors)
    {
        flat_offsets.insert(flat_offsets.end(), vec.begin(), vec.end());
    }

    xsparse::Timer timer;
    timer.start();

    /*
    C++ Intrinsic 等价代码说明:
    svbool_t pg = svptrue_b64();                        // 设置谓词寄存器全真
    svfloat64_t data_vec = svdup_f64(1.0);              // 初始化要写入的数据向量

    for (size_t op = 0; op < num_scatter_ops; ++op) {
        // 从预生成的偏移量数组获取当前向量的偏移量
        const uint64_t* current_offsets = &flat_offsets[op * elements_per_vector];

        // 加载偏移量向量 (字节偏移量)
        svuint64_t offset_vec = svld1_u64(pg, current_offsets);    // ld1d {z8.d}, p0/z, [offsets_ptr]

        // 执行 scatter 操作：根据偏移量向量将数据分散存储到不同位置
        svst1_scatter_u64offset_f64(pg, base_ptr, offset_vec, data_vec);  // st1d {z0.d}, p0, [base, z8.d]

        // data_vec 中的每个元素被存储到 base_ptr + offset_vec[i] 位置
    }
    */

    // clang-format off
    __asm__ volatile(
        "ptrue p0.d\n\t"                    // 设置谓词寄存器全真
        "fmov z0.d, #1.0\n\t"             // 初始化向量寄存器
        "mov x0, %[num_ops]\n\t"            // Scatter 操作次数
        "mov x1, %[offsets_ptr]\n\t"        // 偏移量数组指针
        "mov x2, %[base_ptr]\n\t"           // 基地址
        "mov x3, %[vec_size]\n\t"           // 每个向量的元素数量
        "mov x4, #8\n\t"                    // sizeof(uint64_t) for offset stride
        "1:\n"
        // 加载当前向量的偏移量到 z8
        "ld1d {z8.d}, p0/z, [x1]\n\t"      // 加载偏移量向量
        // 执行 scatter 操作
        "st1d {z0.d}, p0, [x2, z8.d]\n\t"  // scatter 存储
        // 移动到下一组偏移量
        "mul x5, x3, x4\n\t"               // 计算偏移量数组的步长
        "add x1, x1, x5\n\t"               // 移动到下一个偏移量向量
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [num_ops] "r"(num_scatter_ops),
          [offsets_ptr] "r"(flat_offsets.data()),
          [base_ptr] "r"(base_ptr),
          [vec_size] "r"(elements_per_vector)
        : "x0", "x1", "x2", "x3", "x4", "x5", "p0", "z0", "z8", "memory"
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double bytes_transferred = num_scatter_ops * vl;  // 每次 scatter 写入一个向量
    const double bandwidth_gb_s = bytes_transferred / ns;  // GB/s

    if (isPrint)
    {
        std::cout << "Test Size: " << memory_size_mb << " MB" << std::endl;
        std::cout << "Scatter Write Bandwidth: " << bandwidth_gb_s << " GB/s ("
                  << num_scatter_ops << " scatter operations)" << std::endl;
        std::cout << "Elements per vector: " << elements_per_vector << std::endl;
    }
}

int main(int argc, char **argv)
{
    float cpu_freq_ghz;
    size_t memory_size_mb = 256;  // 默认内存大小 256MB
    size_t num_random_accesses = 100000;  // 默认随机访问次数
    size_t num_gather_ops = 50000;  // 默认 Gather/Scatter 操作次数

    if (argc < 2 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << " <CPU_FREQ_GHZ> [MEMORY_SIZE_MB] [RANDOM_ACCESSES] [GATHER_OPS]"
                  << std::endl;
        std::cerr << "Example: " << argv[0] << " 2.4 512 100000 50000" << std::endl;
        std::cerr << "Defaults: memory_size=256MB, random_accesses=100000, gather_ops=50000" << std::endl;
        return 1;
    }

    try
    {
        cpu_freq_ghz = std::stof(argv[1]);
        if (argc >= 3)
        {
            memory_size_mb = std::stoull(argv[2]);
            if (memory_size_mb < 1)
            {
                std::cerr << "Memory size must be at least 1MB" << std::endl;
                return 1;
            }
        }
        if (argc >= 4)
        {
            num_random_accesses = std::stoull(argv[3]);
            if (num_random_accesses < 1)
            {
                std::cerr << "Random accesses must be at least 1" << std::endl;
                return 1;
            }
        }
        if (argc >= 5)
        {
            num_gather_ops = std::stoull(argv[4]);
            if (num_gather_ops < 1)
            {
                std::cerr << "Gather operations must be at least 1" << std::endl;
                return 1;
            }
            // 确保 gather_ops 是4的倍数，以便展开版本正常工作
            if (num_gather_ops % 4 != 0)
            {
                std::cerr << "Gather operations must be a multiple of 4 for unroll4 version" << std::endl;
                return 1;
            }
        }
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid arguments. All parameters must be valid numbers."
                  << std::endl;
        return 1;
    }

    // 热身运行
    MB_SVE_Sequential_Read(cpu_freq_ghz, memory_size_mb, false);

    std::cout << "=======================================" << std::endl;
    std::cout << "SVE Memory Bandwidth Test" << std::endl;
    std::cout << "Memory Size: " << memory_size_mb << " MB" << std::endl;
    std::cout << "Random Accesses: " << num_random_accesses << std::endl;
    std::cout << "Gather/Scatter Operations: " << num_gather_ops << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "Sequential Read Test:" << std::endl;
    MB_SVE_Sequential_Read(cpu_freq_ghz, memory_size_mb, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Sequential Write Test:" << std::endl;
    MB_SVE_Sequential_Write(cpu_freq_ghz, memory_size_mb, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Memory Copy Test:" << std::endl;
    MB_SVE_Memory_Copy(cpu_freq_ghz, memory_size_mb, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Random Access Tests:" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Random Read Test:" << std::endl;
    MB_SVE_Random_Read(cpu_freq_ghz, memory_size_mb, num_random_accesses, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Random Write Test:" << std::endl;
    MB_SVE_Random_Write(cpu_freq_ghz, memory_size_mb, num_random_accesses, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Gather/Scatter Tests:" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Gather Read Test:" << std::endl;
    MB_SVE_Gather_Read(cpu_freq_ghz, memory_size_mb, num_gather_ops, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Gather Read Unroll4 Test:" << std::endl;
    MB_SVE_Gather_Read_Unroll4(cpu_freq_ghz, memory_size_mb, num_gather_ops, true);
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Scatter Write Test:" << std::endl;
    MB_SVE_Scatter_Write(cpu_freq_ghz, memory_size_mb, num_gather_ops, true);
    std::cout << "=======================================" << std::endl;

    return 0;
}
