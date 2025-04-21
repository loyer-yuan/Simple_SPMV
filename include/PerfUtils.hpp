#ifndef XSPARSE_PERFUTILS_HPP
#define XSPARSE_PERFUTILS_HPP

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

#endif  // XSPARSE_PERFUTILS_HPP
