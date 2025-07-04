#include <chrono>
#include <iostream>
#include <string>

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

#define FMADD_2                \
    "fmadd d1, d0, d0, d0\n\t" \
    "fmadd d2, d0, d0, d0\n\t"

#define FMADD_4                \
    "fmadd d1, d0, d0, d0\n\t" \
    "fmadd d2, d0, d0, d0\n\t" \
    "fmadd d3, d0, d0, d0\n\t" \
    "fmadd d4, d0, d0, d0\n\t"

#define FMADD_16                   \
    "fmadd d1, d0, d0, d0\n\t"     \
    "fmadd d3, d2, d2, d2\n\t"     \
    "fmadd d5, d4, d4, d4\n\t"     \
    "fmadd d7, d6, d6, d6\n\t"     \
    "fmadd d9, d8, d8, d8\n\t"     \
    "fmadd d11, d10, d10, d10\n\t" \
    "fmadd d13, d12, d12, d12\n\t" \
    "fmadd d15, d14, d14, d14\n\t" \
    "fmadd d17, d16, d16, d16\n\t" \
    "fmadd d19, d18, d18, d18\n\t" \
    "fmadd d21, d20, d20, d20\n\t" \
    "fmadd d23, d22, d22, d22\n\t" \
    "fmadd d25, d24, d24, d24\n\t" \
    "fmadd d27, d26, d26, d26\n\t" \
    "fmadd d29, d28, d28, d28\n\t" \
    "fmadd d31, d30, d30, d30\n\t"

#define FMADD2                 \
    "fmadd d1, d1, d1, d1\n\t" \
    "fmadd d2, d2, d2, d2\n\t"

#define FMADD4                 \
    "fmadd d1, d1, d1, d1\n\t" \
    "fmadd d2, d2, d2, d2\n\t" \
    "fmadd d3, d3, d3, d3\n\t" \
    "fmadd d4, d4, d4, d4\n\t"

#define FMADD8                 \
    "fmadd d1, d1, d1, d1\n\t" \
    "fmadd d2, d2, d2, d2\n\t" \
    "fmadd d3, d3, d3, d3\n\t" \
    "fmadd d4, d4, d4, d4\n\t" \
    "fmadd d5, d5, d5, d5\n\t" \
    "fmadd d6, d6, d6, d6\n\t" \
    "fmadd d7, d7, d7, d7\n\t" \
    "fmadd d8, d8, d8, d8\n\t"

#define FMADD16                    \
    "fmadd d1, d1, d1, d1\n\t"     \
    "fmadd d2, d2, d2, d2\n\t"     \
    "fmadd d3, d3, d3, d3\n\t"     \
    "fmadd d4, d4, d4, d4\n\t"     \
    "fmadd d5, d5, d5, d5\n\t"     \
    "fmadd d6, d6, d6, d6\n\t"     \
    "fmadd d7, d7, d7, d7\n\t"     \
    "fmadd d8, d8, d8, d8\n\t"     \
    "fmadd d9, d9, d9, d9\n\t"     \
    "fmadd d10, d10, d10, d10\n\t" \
    "fmadd d11, d11, d11, d11\n\t" \
    "fmadd d12, d12, d12, d12\n\t" \
    "fmadd d13, d13, d13, d13\n\t" \
    "fmadd d14, d14, d14, d14\n\t" \
    "fmadd d15, d15, d15, d15\n\t" \
    "fmadd d16, d16, d16, d16\n\t"

#define FMADD32                    \
    "fmadd d0, d0, d0, d0\n\t"     \
    "fmadd d1, d1, d1, d1\n\t"     \
    "fmadd d2, d2, d2, d2\n\t"     \
    "fmadd d3, d3, d3, d3\n\t"     \
    "fmadd d4, d4, d4, d4\n\t"     \
    "fmadd d5, d5, d5, d5\n\t"     \
    "fmadd d6, d6, d6, d6\n\t"     \
    "fmadd d7, d7, d7, d7\n\t"     \
    "fmadd d8, d8, d8, d8\n\t"     \
    "fmadd d9, d9, d9, d9\n\t"     \
    "fmadd d10, d10, d10, d10\n\t" \
    "fmadd d11, d11, d11, d11\n\t" \
    "fmadd d12, d12, d12, d12\n\t" \
    "fmadd d13, d13, d13, d13\n\t" \
    "fmadd d14, d14, d14, d14\n\t" \
    "fmadd d15, d15, d15, d15\n\t" \
    "fmadd d16, d16, d16, d16\n\t" \
    "fmadd d17, d17, d17, d17\n\t" \
    "fmadd d18, d18, d18, d18\n\t" \
    "fmadd d19, d19, d19, d19\n\t" \
    "fmadd d20, d20, d20, d20\n\t" \
    "fmadd d21, d21, d21, d21\n\t" \
    "fmadd d22, d22, d22, d22\n\t" \
    "fmadd d23, d23, d23, d23\n\t" \
    "fmadd d24, d24, d24, d24\n\t" \
    "fmadd d25, d25, d25, d25\n\t" \
    "fmadd d26, d26, d26, d26\n\t" \
    "fmadd d27, d27, d27, d27\n\t" \
    "fmadd d28, d28, d28, d28\n\t" \
    "fmadd d29, d29, d29, d29\n\t" \
    "fmadd d30, d30, d30, d30\n\t" \
    "fmadd d31, d31, d31, d31\n\t"

// void MB_Througput_FMA_no_related(bool isPrint = false)
// {
//     const int64_t iter = 10000000;
//     volatile double src = 1.1;  // Avoid optimization

//     xsparse::Timer timer;
//     timer.start();

//     // clang-format off
//     __asm__ volatile(
//         "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
//         "fmov d0, %x[src]\n\t"  // Load src into d0
//         "1:\n"
//         // FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4
//         // FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4
//         FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
//         FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
//         FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
//         FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
//         "subs x0, x0, #1\n\t"
//         "b.gt 1b\n"
//         :
//         : [src] "r"(src),  // Use "r" for register input
//           [iter] "r"(iter)
//         : "x0", "d0", "d1", "d2", "d3", "d4"  // Clobbered registers
//     );
//     // clang-format on

//     timer.stop();
//     double ns = timer.elapsed<std::chrono::nanoseconds>();

//     const double cpu_freq_ghz = CPU_FREQ_GHZ;
//     const double cycles = ns * cpu_freq_ghz;
//     const double cycles_per_fmadd = cycles / (iter * 4.0 * 16.0);

//     if (isPrint)
//     {
//         std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
//         std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
//                   << " FMADDs per cycle" << std::endl;
//     }
// }

void MB_Througput_FMA_no_related(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "fmov d2, d0\n\t"  // Initialize d29 with src
        "fmov d4, d0\n\t"
        "fmov d6, d0\n\t"
        "fmov d8, d0\n\t"
        "fmov d10, d0\n\t"
        "fmov d12, d0\n\t"
        "fmov d14, d0\n\t"
        "fmov d16, d0\n\t"
        "fmov d18, d0\n\t"
        "fmov d20, d0\n\t"
        "fmov d22, d0\n\t"
        "fmov d24, d0\n\t"
        "fmov d26, d0\n\t"
        "fmov d28, d0\n\t"
        "fmov d30, d0\n\t"
        "1:\n"
        FMADD_16 FMADD_16 FMADD_16 FMADD_16
        FMADD_16 FMADD_16 FMADD_16 FMADD_16
        // FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4
        // FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4
        // FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4
        // FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4 FMADD_4
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        // FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2 FMADD_2
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
          "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d27", "d28",
          "d29", "d30", "d31" // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 16.0 * 8.0);

    if (isPrint)
    {
        std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
        std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
                  << " FMADDs per cycle" << std::endl;
    }
}

void MB_Througput_FMA2(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "fmov d1, d0\n\t"  // Initialize d1 with src
        "fmov d2, d0\n\t"  // Initialize d2 with src

        "1:\n"
        FMADD2 FMADD2 FMADD2 FMADD2 FMADD2 FMADD2 FMADD2 FMADD2
        FMADD2 FMADD2 FMADD2 FMADD2 FMADD2 FMADD2 FMADD2 FMADD2
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2"  // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 2.0 * 16.0);

    if (isPrint)
    {
        std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
        std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
                  << " FMADDs per cycle" << std::endl;
    }
}

void MB_Througput_FMA4(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "fmov d1, d0\n\t"  // Initialize d1 with src
        "fmov d2, d0\n\t"  // Initialize d2 with src
        "fmov d3, d0\n\t"  // Initialize d3 with src
        "fmov d4, d0\n\t"  // Initialize d4 with src
        "1:\n"
        FMADD4 FMADD4 FMADD4 FMADD4 FMADD4 FMADD4 FMADD4 FMADD4
        FMADD4 FMADD4 FMADD4 FMADD4 FMADD4 FMADD4 FMADD4 FMADD4
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2", "d3", "d4"  // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 4.0 * 16.0);

    if (isPrint)
    {
        std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
        std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
                  << " FMADDs per cycle" << std::endl;
    }
}

void MB_Througput_FMA8(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "fmov d1, d0\n\t"  // Initialize d1 with src
        "fmov d2, d0\n\t"  // Initialize d2 with src
        "fmov d3, d0\n\t"  // Initialize d3 with src
        "fmov d4, d0\n\t"  // Initialize d4 with src
        "fmov d5, d0\n\t"  // Initialize d5 with src
        "fmov d6, d0\n\t"  // Initialize d6 with src
        "fmov d7, d0\n\t"  // Initialize d7 with src
        "fmov d8, d0\n\t"  // Initialize d8 with src
        "1:\n"
        FMADD8 FMADD8 FMADD8 FMADD8
        FMADD8 FMADD8 FMADD8 FMADD8
        FMADD8 FMADD8 FMADD8 FMADD8
        FMADD8 FMADD8 FMADD8 FMADD8
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8" // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 8.0 * 16.0);

    if (isPrint)
    {
        std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
        std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
                  << " FMADDs per cycle" << std::endl;
    }
}

void MB_Througput_FMA16(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "fmov d1, d0\n\t"  // Initialize d1 with src
        "fmov d2, d0\n\t"  // Initialize d2 with src
        "fmov d3, d0\n\t"  // Initialize d3 with src
        "fmov d4, d0\n\t"  // Initialize d4 with src
        "fmov d5, d0\n\t"  // Initialize d5 with src
        "fmov d6, d0\n\t"  // Initialize d6 with src
        "fmov d7, d0\n\t"  // Initialize d7 with src
        "fmov d8, d0\n\t"  // Initialize d8 with src
        "fmov d9, d0\n\t"  // Initialize d9 with src
        "fmov d10, d0\n\t"  // Initialize d10 with src
        "fmov d11, d0\n\t"  // Initialize d11 with src
        "fmov d12, d0\n\t"  // Initialize d12 with src
        "fmov d13, d0\n\t"  // Initialize d13 with src
        "fmov d14, d0\n\t"  // Initialize d14 with src
        "fmov d15, d0\n\t"  // Initialize d15 with src
        "fmov d16, d0\n\t"  // Initialize d16 with src
        "fmov d17, d0\n\t"  // Initialize d17 with src
        "fmov d18, d0\n\t"  // Initialize d18 with src
        "fmov d19, d0\n\t"  // Initialize d19 with src
        "fmov d20, d0\n\t"  // Initialize d20 with src
        "fmov d21, d0\n\t"  // Initialize d21 with src
        "fmov d22, d0\n\t"  // Initialize d22 with src
        "fmov d23, d0\n\t"  // Initialize d23 with src
        "fmov d24, d0\n\t"  // Initialize d24 with src
        "1:\n"
        FMADD16 FMADD16 FMADD16 FMADD16
        FMADD16 FMADD16 FMADD16 FMADD16
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
          "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
          "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24"  // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 16.0 * 8.0);

    if (isPrint)
    {
        std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
        std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
                  << " FMADDs per cycle" << std::endl;
    }
}

void MB_Througput_FMA32(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "fmov d1, d0\n\t"  // Initialize d1 with src
        "fmov d2, d0\n\t"  // Initialize d2 with src
        "fmov d3, d0\n\t"  // Initialize d3 with src
        "fmov d4, d0\n\t"  // Initialize d4 with src
        "fmov d5, d0\n\t"  // Initialize d5 with src
        "fmov d6, d0\n\t"  // Initialize d6 with src
        "fmov d7, d0\n\t"  // Initialize d7 with src
        "fmov d8, d0\n\t"  // Initialize d8 with src
        "fmov d9, d0\n\t"  // Initialize d9 with src
        "fmov d10, d0\n\t" // Initialize d10 with src
        "fmov d11, d0\n\t" // Initialize d11 with src
        "fmov d12, d0\n\t" // Initialize d12 with src
        "fmov d13, d0\n\t" // Initialize d13 with src
        "fmov d14, d0\n\t" // Initialize d14 with src
        "fmov d15, d0\n\t" // Initialize d15 with src
        "fmov d16, d0\n\t" // Initialize d16 with src
        "fmov d17, d0\n\t" // Initialize d17 with src
        "fmov d18, d0\n\t" // Initialize d18 with src
        "fmov d19, d0\n\t" // Initialize d19 with src
        "fmov d20, d0\n\t" // Initialize d20 with src
        "fmov d21, d0\n\t" // Initialize d21 with src
        "fmov d22, d0\n\t" // Initialize d22 with src
        "fmov d23, d0\n\t" // Initialize d23 with src
        "fmov d24, d0\n\t" // Initialize d24 with src
        "fmov d25, d0\n\t" // Initialize d25 with src
        "fmov d26, d0\n\t" // Initialize d26 with src
        "fmov d27, d0\n\t" // Initialize d27 with src
        "fmov d28, d0\n\t" // Initialize d28 with src
        "fmov d29, d0\n\t" // Initialize d29 with src
        "fmov d30, d0\n\t" // Initialize d30 with src
        "fmov d31, d0\n\t" // Initialize d31 with src
        "1:\n"
        FMADD32 FMADD32 FMADD32 FMADD32
        FMADD32 FMADD32 FMADD32 FMADD32
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
          "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
          "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25",
          "d26", "d27", "d28", "d29", "d30", "d31"  // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();
    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 32.0 * 8.0);

    if (isPrint)
    {
        std::cout << "Estimated cycles per FMADD: " << cycles_per_fmadd << std::endl;
        std::cout << "Estimated throughput: " << 1.0 / cycles_per_fmadd
                  << " FMADDs per cycle" << std::endl;
    }
}

#define FMADD_CHAIN16              \
    "fmadd d1, d0, d0, d0\n\t"     \
    "fmadd d2, d1, d1, d1\n\t"     \
    "fmadd d3, d2, d2, d2\n\t"     \
    "fmadd d4, d3, d3, d3\n\t"     \
    "fmadd d5, d4, d4, d4\n\t"     \
    "fmadd d6, d5, d5, d5\n\t"     \
    "fmadd d7, d6, d6, d6\n\t"     \
    "fmadd d8, d7, d7, d7\n\t"     \
    "fmadd d9, d8, d8, d8\n\t"     \
    "fmadd d10, d9, d9, d9\n\t"    \
    "fmadd d11, d10, d10, d10\n\t" \
    "fmadd d12, d11, d11, d11\n\t" \
    "fmadd d13, d12, d12, d12\n\t" \
    "fmadd d14, d13, d13, d13\n\t" \
    "fmadd d15, d14, d14, d14\n\t" \
    "fmadd d0, d15, d15, d15\n\t"

void MB_Latency_FMA(const float CPU_FREQ_GHZ, bool isPrint = false)
{
    const int64_t iter = 10000000;
    volatile double src = 1.1;  // Avoid optimization

    xsparse::Timer timer;
    timer.start();

    // clang-format off
    __asm__ volatile(
        "mov x0, %x[iter]\n\t"  // Use %x to specify 64-bit register name
        "fmov d0, %x[src]\n\t"  // Load src into d0
        "1:\n"
        FMADD_CHAIN16 FMADD_CHAIN16 FMADD_CHAIN16 FMADD_CHAIN16
        FMADD_CHAIN16 FMADD_CHAIN16 FMADD_CHAIN16 FMADD_CHAIN16
        "subs x0, x0, #1\n\t"
        "b.gt 1b\n"
        :
        : [src] "r"(src),  // Use "r" for register input
          [iter] "r"(iter)
        : "x0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
          "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15"  // Clobbered registers
    );
    // clang-format on

    timer.stop();
    double ns = timer.elapsed<std::chrono::nanoseconds>();

    const double cpu_freq_ghz = CPU_FREQ_GHZ;
    const double cycles = ns * cpu_freq_ghz;
    const double cycles_per_fmadd = cycles / (iter * 16.0 * 8.0);

    if (isPrint)
    {
        std::cout << "Estimated latency: " << cycles_per_fmadd << " cycles per FMA"
                  << std::endl;
    }
}

int main(int argc, char **argv)
{
    float cpu_freq_ghz;
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <CPU_FREQ_GHZ>" << std::endl;
        return 1;
    }
    try
    {
        cpu_freq_ghz = std::stof(argv[1]);
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid CPU frequency: " << argv[1] << std::endl;
        return 1;
    }
    // Warm up the CPU cache
    MB_Througput_FMA_no_related(cpu_freq_ghz, false);

    std::cout << "=======================================" << std::endl;
    std::cout << "Latency Test for FMA Instructions" << std::endl;
    MB_Latency_FMA(cpu_freq_ghz, true);  // Print results
    std::cout << "=======================================" << std::endl;
    std::cout << "Throughput Test for FMA Instructions" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "No Related Instruction FMA:" << std::endl;
    MB_Througput_FMA_no_related(cpu_freq_ghz, true);
    std::cout << "----------------------------------" << std::endl;
    std::cout << "FMA2:" << std::endl;
    MB_Througput_FMA2(cpu_freq_ghz, true);  // Print results
    std::cout << "----------------------------------" << std::endl;
    std::cout << "FMA4:" << std::endl;
    MB_Througput_FMA4(cpu_freq_ghz, true);  // Print results
    std::cout << "----------------------------------" << std::endl;
    std::cout << "FMA8:" << std::endl;
    MB_Througput_FMA8(cpu_freq_ghz, true);  // Print results
    std::cout << "----------------------------------" << std::endl;
    std::cout << "FMA16:" << std::endl;
    MB_Througput_FMA16(cpu_freq_ghz, true);  // Print results
    std::cout << "----------------------------------" << std::endl;
    std::cout << "FMA32:" << std::endl;
    MB_Througput_FMA32(cpu_freq_ghz, true);  // Print results
    std::cout << "=======================================" << std::endl;
    return 0;
}