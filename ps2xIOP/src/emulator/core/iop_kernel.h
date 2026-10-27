#pragma once

#include "iop_cpu.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
#include <string>

namespace ps2x::iop::detail
{
    class IopMemory;

    enum class IopThreadState : uint8_t
    {
        Dormant,
        Ready,
        Running,
        Sleep,
        Delay,
        Semaphore,
        EventFlag,
        Suspended,
        Dead,
    };

    struct IopThread
    {
        int id = 0;
        IopThreadState state = IopThreadState::Dormant;
        IopCpuState cpu;
        uint32_t entry = 0;
        uint32_t stackBase = 0;
        uint32_t stackSize = 0;
        uint32_t priority = 0x40;
        uint32_t initialPriority = 0x40;
        uint32_t option = 0;
        uint32_t attr = 0;
        uint64_t wakeCycle = 0;
        int waitId = 0;
        uint32_t waitBits = 0;
        uint32_t waitMode = 0;
        uint32_t waitResultAddress = 0;
        int wakeupCount = 0;
    };

    class IopKernel
    {
    public:
        // The host thread module start routines run on (hostThread), as MODLOAD's thread on the IOP.
        static constexpr int kModuleLoaderThreadId = 0x7FFF0000;

        explicit IopKernel(IopMemory &memory) noexcept;

        void reset();

        [[nodiscard]] bool dispatchThreadImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle);
        [[nodiscard]] bool dispatchSemaphoreImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] bool dispatchEventImport(uint16_t ordinal, IopCpuState &cpu);

        [[nodiscard]] int createInternalEventFlag(uint32_t attr, uint32_t option, uint32_t bits);
        [[nodiscard]] bool setInternalEventFlag(int id, uint32_t bits);

        void sleepCurrent(IopCpuState &cpu);
        void delayCurrentUntil(uint64_t wakeCycle, IopCpuState &cpu);

        // nextDelayWake (optional) receives the earliest wake cycle of the threads still delayed.
        [[nodiscard]] IopThread *beginNextReady(uint64_t currentCycle, uint64_t *nextDelayWake = nullptr);
        // The best Ready thread at `priority` or above (numerically at most), `excluded` aside: one the IOP would run
        // before a thread of that priority (no thread preempts one of equal or higher priority).
        [[nodiscard]] IopThread *readyAtOrAbove(uint32_t priority, int excluded) noexcept;
        void beginThread(IopThread &thread) noexcept;
        [[nodiscard]] uint64_t nextWakeCycle(uint64_t fallback) const;
        void endTimeslice(IopThread &thread, uint32_t returnSentinel);
        void cleanupDeadThreads();
        void terminateThreadsInRange(uint32_t base, uint32_t size);

        [[nodiscard]] size_t threadCount() const noexcept { return m_threads.size(); }
        // For running code on behalf of a thread outside its own timeslice (RPC server functions).
        [[nodiscard]] IopThread *thread(int id) noexcept
        {
            const auto it = m_threads.find(id);
            return it == m_threads.end() ? nullptr : &it->second;
        }
        [[nodiscard]] IopThread *currentThread() const noexcept { return m_currentThread; }
        // A Dormant thread the host runs guest code on (module start routines), so that code can block like it
        // would in MODLOAD's thread. Its id is outside CreateThread's range and it has no stack of its own.
        IopThread &hostThread(int id, uint32_t priority);
        void setCurrentThread(IopThread *thread) noexcept { m_currentThread = thread; }
        // One line per thread (state, pc, entry, priority, wait object) for debug snapshots.
        [[nodiscard]] std::vector<std::string> describeThreads() const;

    private:
        struct Semaphore
        {
            int id = 0;
            uint32_t attr = 0;
            uint32_t option = 0;
            int current = 0;
            int maximum = 1;
        };

        struct EventFlag
        {
            int id = 0;
            uint32_t bits = 0;
            uint32_t attr = 0;
            uint32_t option = 0;
        };

        [[nodiscard]] bool referThreadStatus(int id, uint32_t outputAddress);
        void wakeOneSemaphore(int id);
        [[nodiscard]] static bool eventSatisfied(const EventFlag &event, uint32_t bits, uint32_t mode);
        void wakeEventWaiters(EventFlag &event);
        [[nodiscard]] bool runsBefore(const IopThread &thread, const IopThread &other) const noexcept;

        IopMemory &m_memory;
        std::map<int, IopThread> m_threads;
        std::map<int, Semaphore> m_semaphores;
        std::map<int, EventFlag> m_eventFlags;
        uint32_t m_nextThreadId = 1;
        uint32_t m_nextSemaphoreId = 1;
        uint32_t m_nextEventFlagId = 1;
        IopThread *m_currentThread = nullptr;
        int m_slicedThread = 0;
    };
}
