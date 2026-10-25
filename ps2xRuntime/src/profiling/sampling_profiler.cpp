// Opt-in in-process sampling profiler: PS2X_PROFILE=<seconds> samples every thread of the process
// roughly every millisecond for that long (after an optional PS2X_PROFILE_DELAY=<seconds>), then writes
// histogram lines to PS2X_PROFILE_OUT (default ps2x_profile.txt).
// Windows: "<thread id> <leaf rva hex> <first in-exe caller rva hex> <count>", for symbolization against
// the linker map file. For threads blocked in system calls the leaf rva is 0 and the caller column says
// where in the runtime they are waiting.
// PS2X_PROFILE_GATED=1: samples only while ps2x::profiling::g_sampleGate is open (runtime/sampling_profiler_gate.h).
// Linux: "<thread id> <thread name> <leaf offset hex> <count> [<library>:<symbol>]", sampled on each
// thread's CPU-time clock (blocked threads are not sampled); offsets are relative to the executable's
// load address (addr2line -f -C -e <exe> <offset>); code outside the executable has offset 0 and is
// grouped by the shared library and exported symbol dladdr finds for it.
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#include <tlhelp32.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <tuple>

#include "runtime/sampling_profiler_gate.h"

namespace
{
    void runSamplingProfiler(double delaySeconds, double durationSeconds, std::string outPath)
    {
        std::this_thread::sleep_for(std::chrono::duration<double>(delaySeconds));
        const DWORD self = GetCurrentThreadId();
        const DWORD pid = GetCurrentProcessId();
        const auto moduleBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(moduleBase);
        const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(moduleBase + dos->e_lfanew);
        const uintptr_t moduleEnd = moduleBase + nt->OptionalHeader.SizeOfImage;
        auto inExe = [&](uintptr_t address) { return address >= moduleBase && address < moduleEnd; };

        std::map<std::tuple<DWORD, uintptr_t, uintptr_t>, uint64_t> histogram;
        std::map<DWORD, HANDLE> threads;
        uint64_t samples = 0u;

        static const bool s_gated = []
        {
            const char *value = std::getenv("PS2X_PROFILE_GATED");
            return value && *value == '1';
        }();
        timeBeginPeriod(1);
        const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(durationSeconds);
        auto nextRefresh = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() < end)
        {
            if (std::chrono::steady_clock::now() >= nextRefresh)
            {
                nextRefresh += std::chrono::milliseconds(500);
                HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                THREADENTRY32 entry{};
                entry.dwSize = sizeof(entry);
                if (snapshot != INVALID_HANDLE_VALUE && Thread32First(snapshot, &entry))
                {
                    do
                    {
                        if (entry.th32OwnerProcessID == pid && entry.th32ThreadID != self &&
                            threads.find(entry.th32ThreadID) == threads.end())
                        {
                            HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                                       FALSE, entry.th32ThreadID);
                            if (handle)
                                threads[entry.th32ThreadID] = handle;
                        }
                    } while (Thread32Next(snapshot, &entry));
                }
                if (snapshot != INVALID_HANDLE_VALUE)
                    CloseHandle(snapshot);
            }

            if (s_gated && !ps2x::profiling::g_sampleGate.load(std::memory_order_relaxed))
            {
                Sleep(1);
                continue;
            }
            for (const auto &[tid, handle] : threads)
            {
                if (SuspendThread(handle) == static_cast<DWORD>(-1))
                    continue;
                CONTEXT context{};
                context.ContextFlags = CONTEXT_FULL;
                if (GetThreadContext(handle, &context))
                {
                    const uintptr_t leaf = static_cast<uintptr_t>(context.Rip);
                    uintptr_t caller = inExe(leaf) ? leaf : 0u;
                    // Unwind until the first frame inside the exe (what a blocked thread is waiting in).
                    for (int depth = 0; depth < 48 && caller == 0u; ++depth)
                    {
                        DWORD64 imageBase = 0;
                        PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr);
                        if (function)
                        {
                            PVOID handlerData = nullptr;
                            DWORD64 establisherFrame = 0;
                            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, function, &context,
                                             &handlerData, &establisherFrame, nullptr);
                        }
                        else
                        {
                            context.Rip = *reinterpret_cast<DWORD64 *>(context.Rsp);
                            context.Rsp += 8;
                        }
                        if (context.Rip == 0)
                            break;
                        if (inExe(static_cast<uintptr_t>(context.Rip)))
                            caller = static_cast<uintptr_t>(context.Rip);
                    }
                    ++histogram[{tid, inExe(leaf) ? leaf - moduleBase : 0u, caller ? caller - moduleBase : 0u}];
                    ++samples;
                }
                ResumeThread(handle);
            }
            Sleep(1);
        }
        timeEndPeriod(1);

        if (FILE *file = std::fopen(outPath.c_str(), "w"))
        {
            for (const auto &[key, count] : histogram)
                std::fprintf(file, "%lu %llx %llx %llu\n", std::get<0>(key),
                             static_cast<unsigned long long>(std::get<1>(key)),
                             static_cast<unsigned long long>(std::get<2>(key)),
                             static_cast<unsigned long long>(count));
            std::fclose(file);
        }
        for (const auto &[tid, handle] : threads)
            CloseHandle(handle);
        std::fprintf(stderr, "[profiler] wrote %llu samples to %s\n", static_cast<unsigned long long>(samples), outPath.c_str());
    }

    struct SamplingProfilerStarter
    {
        SamplingProfilerStarter()
        {
            const char *duration = std::getenv("PS2X_PROFILE");
            if (!duration || std::atof(duration) <= 0.0)
                return;
            const char *delay = std::getenv("PS2X_PROFILE_DELAY");
            const char *out = std::getenv("PS2X_PROFILE_OUT");
            std::thread(runSamplingProfiler, delay ? std::atof(delay) : 0.0, std::atof(duration),
                        std::string(out && *out ? out : "ps2x_profile.txt"))
                .detach();
        }
    };

    const SamplingProfilerStarter g_samplingProfilerStarter;
}
#elif defined(__linux__)
#include <csignal>
#include <ctime>
#include <dirent.h>
#include <dlfcn.h>
#include <link.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>

namespace
{
    struct Sample
    {
        uint32_t tid;
        uintptr_t pc;
    };

    constexpr size_t kSampleCapacity = 1u << 22;
    Sample *g_samples = nullptr;
    std::atomic<size_t> g_sampleCount{0};
    std::atomic<bool> g_sampling{false};

    void onProfileSignal(int, siginfo_t *, void *raw)
    {
        if (!g_sampling.load(std::memory_order_relaxed))
            return;
        const size_t index = g_sampleCount.fetch_add(1u, std::memory_order_relaxed);
        if (index >= kSampleCapacity)
            return;
        const auto *context = static_cast<const ucontext_t *>(raw);
#if defined(__x86_64__)
        const auto pc = static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
        const auto pc = static_cast<uintptr_t>(context->uc_mcontext.pc);
#else
        const uintptr_t pc = 0u;
#endif
        g_samples[index] = Sample{static_cast<uint32_t>(syscall(SYS_gettid)), pc};
    }

    struct ExeRange
    {
        uintptr_t base = 0u;
        uintptr_t begin = ~uintptr_t{0};
        uintptr_t end = 0u;
    };

    ExeRange findExecutableRange()
    {
        ExeRange range;
        dl_iterate_phdr([](dl_phdr_info *info, size_t, void *data)
                        {
                            auto *out = static_cast<ExeRange *>(data);
                            if (info->dlpi_name && info->dlpi_name[0] != '\0')
                                return 0;
                            out->base = info->dlpi_addr;
                            for (int i = 0; i < info->dlpi_phnum; ++i)
                            {
                                const ElfW(Phdr) &header = info->dlpi_phdr[i];
                                if (header.p_type != PT_LOAD || (header.p_flags & PF_X) == 0)
                                    continue;
                                const uintptr_t begin = info->dlpi_addr + header.p_vaddr;
                                out->begin = std::min(out->begin, begin);
                                out->end = std::max(out->end, static_cast<uintptr_t>(begin + header.p_memsz));
                            }
                            return 1;
                        },
                        &range);
        return range;
    }

    std::string threadName(pid_t tid)
    {
        char path[64];
        std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", static_cast<int>(tid));
        std::string name = "?";
        if (FILE *file = std::fopen(path, "r"))
        {
            char buffer[64] = {};
            if (std::fgets(buffer, sizeof(buffer), file))
            {
                name = buffer;
                while (!name.empty() && (name.back() == '\n' || name.back() == ' '))
                    name.pop_back();
                for (char &c : name)
                    if (c == ' ')
                        c = '_';
            }
            std::fclose(file);
        }
        return name;
    }

    void runSamplingProfiler(double delaySeconds, double durationSeconds, long periodMicroseconds, std::string outPath)
    {
        std::this_thread::sleep_for(std::chrono::duration<double>(delaySeconds));
        const pid_t self = static_cast<pid_t>(syscall(SYS_gettid));
        std::unique_ptr<Sample[]> storage(new Sample[kSampleCapacity]);
        g_samples = storage.get();

        struct sigaction action{};
        action.sa_sigaction = onProfileSignal;
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&action.sa_mask);
        sigaction(SIGPROF, &action, nullptr);

        std::map<pid_t, timer_t> timers;
        std::map<pid_t, std::string> names;
        auto refreshThreads = [&]()
        {
            DIR *dir = opendir("/proc/self/task");
            if (!dir)
                return;
            while (const dirent *entry = readdir(dir))
            {
                const pid_t tid = static_cast<pid_t>(std::atoi(entry->d_name));
                if (tid <= 0 || tid == self || timers.count(tid) != 0u)
                    continue;
                const clockid_t clock = static_cast<clockid_t>((~static_cast<clockid_t>(tid) << 3) | 6);
                sigevent event{};
                event.sigev_notify = SIGEV_THREAD_ID;
                event.sigev_signo = SIGPROF;
                event._sigev_un._tid = tid;
                timer_t timer{};
                if (timer_create(clock, &event, &timer) != 0)
                    continue;
                itimerspec spec{};
                spec.it_interval.tv_nsec = periodMicroseconds * 1000;
                spec.it_value.tv_nsec = periodMicroseconds * 1000;
                timer_settime(timer, 0, &spec, nullptr);
                timers[tid] = timer;
                names[tid] = threadName(tid);
            }
            closedir(dir);
        };

        g_sampling.store(true, std::memory_order_release);
        const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(durationSeconds);
        while (std::chrono::steady_clock::now() < end)
        {
            refreshThreads();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        for (auto &[tid, timer] : timers)
            timer_delete(timer);
        g_sampling.store(false, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        const ExeRange exe = findExecutableRange();
        const size_t count = std::min(g_sampleCount.load(std::memory_order_acquire), kSampleCapacity);
        std::map<uintptr_t, std::string> outsideNames;
        auto outsideName = [&](uintptr_t pc) -> const std::string &
        {
            auto [it, inserted] = outsideNames.try_emplace(pc);
            if (inserted)
            {
                Dl_info info{};
                if (dladdr(reinterpret_cast<void *>(pc), &info) != 0 && info.dli_fname)
                {
                    const char *slash = std::strrchr(info.dli_fname, '/');
                    it->second = std::string(slash ? slash + 1 : info.dli_fname) + ":" +
                                 (info.dli_sname ? info.dli_sname : "?");
                }
                else
                {
                    it->second = "?:?";
                }
            }
            return it->second;
        };
        std::map<std::tuple<uint32_t, uintptr_t, std::string>, uint64_t> histogram;
        for (size_t i = 0; i < count; ++i)
        {
            const Sample &sample = g_samples[i];
            if (sample.pc >= exe.begin && sample.pc < exe.end)
                ++histogram[{sample.tid, sample.pc - exe.base, std::string()}];
            else
                ++histogram[{sample.tid, 0u, outsideName(sample.pc)}];
        }
        if (FILE *file = std::fopen(outPath.c_str(), "w"))
        {
            for (const auto &[key, samples] : histogram)
            {
                const auto name = names.find(static_cast<pid_t>(std::get<0>(key)));
                std::fprintf(file, "%u %s %llx %llu%s%s\n", std::get<0>(key),
                             name != names.end() ? name->second.c_str() : "?",
                             static_cast<unsigned long long>(std::get<1>(key)),
                             static_cast<unsigned long long>(samples),
                             std::get<2>(key).empty() ? "" : " ", std::get<2>(key).c_str());
            }
            std::fclose(file);
        }
        std::fprintf(stderr, "[profiler] wrote %zu samples to %s\n", count, outPath.c_str());
        g_samples = nullptr;
    }

    struct SamplingProfilerStarter
    {
        SamplingProfilerStarter()
        {
            const char *duration = std::getenv("PS2X_PROFILE");
            if (!duration || std::atof(duration) <= 0.0)
                return;
            const char *delay = std::getenv("PS2X_PROFILE_DELAY");
            const char *out = std::getenv("PS2X_PROFILE_OUT");
            const char *period = std::getenv("PS2X_PROFILE_PERIOD_US");
            const long periodMicroseconds = period && std::atol(period) > 0 ? std::atol(period) : 1000;
            std::thread(runSamplingProfiler, delay ? std::atof(delay) : 0.0, std::atof(duration),
                        std::min(periodMicroseconds, 999999L),
                        std::string(out && *out ? out : "ps2x_profile.txt"))
                .detach();
        }
    };

    const SamplingProfilerStarter g_samplingProfilerStarter;
}
#endif
