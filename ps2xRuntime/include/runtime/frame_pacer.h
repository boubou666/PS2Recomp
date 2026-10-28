#ifndef PS2_RUNTIME_FRAME_PACER_H
#define PS2_RUNTIME_FRAME_PACER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>

// Native frame pacing: a game that opts in runs its own loop on real time instead of on the PS2's vblanks. Its frame
// end calls frameEnd(), which holds the frame to the cap (frames per second, 0 = unlimited: only the host's speed
// limits it) and returns the real time the frame took, which the game feeds to its engine as its step. The cap
// starts from PS2X_FRAME_CAP (default 60) and can change while the game runs.
class FramePacer
{
public:
    static constexpr double kMaxStepSeconds = 0.1; // a longer frame (a load, a debugger stop) counts as this much

    FramePacer()
    {
        if (const char *cap = std::getenv("PS2X_FRAME_CAP"); cap && *cap)
            m_cap.store(static_cast<uint32_t>(std::strtoul(cap, nullptr, 10)), std::memory_order_relaxed);
    }

    void setCap(uint32_t framesPerSecond) noexcept { m_cap.store(framesPerSecond, std::memory_order_relaxed); }
    [[nodiscard]] uint32_t cap() const noexcept { return m_cap.load(std::memory_order_relaxed); }
    [[nodiscard]] uint64_t frames() const noexcept { return m_frames.load(std::memory_order_relaxed); }
    [[nodiscard]] double lastStepSeconds() const noexcept { return m_lastStep.load(std::memory_order_relaxed); }

    // Waits until the cap allows the next frame to start, then returns the time since the previous frame end.
    double frameEnd()
    {
        using Clock = std::chrono::steady_clock;
        const uint32_t cap = m_cap.load(std::memory_order_relaxed);
        Clock::time_point now = Clock::now();
        if (m_frames.load(std::memory_order_relaxed) == 0u)
            m_previous = now - std::chrono::microseconds(1000000 / std::max<uint32_t>(cap, 60u));
        if (cap != 0u)
        {
            const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / cap));
            const Clock::time_point due = m_previous + period;
            // Sleep most of the wait, then spin the last stretch: Windows sleeps overshoot by up to a millisecond.
            if (due - now > std::chrono::milliseconds(2))
                std::this_thread::sleep_until(due - std::chrono::milliseconds(1));
            while ((now = Clock::now()) < due)
                std::this_thread::yield();
        }
        const double step = std::min(std::chrono::duration<double>(now - m_previous).count(), kMaxStepSeconds);
        m_previous = now;
        m_lastStep.store(step, std::memory_order_relaxed);
        m_frames.fetch_add(1u, std::memory_order_relaxed);
        return step;
    }

private:
    std::atomic<uint32_t> m_cap{60};
    std::atomic<uint64_t> m_frames{0};
    std::atomic<double> m_lastStep{1.0 / 60.0};
    std::chrono::steady_clock::time_point m_previous{};
};

#endif // PS2_RUNTIME_FRAME_PACER_H
