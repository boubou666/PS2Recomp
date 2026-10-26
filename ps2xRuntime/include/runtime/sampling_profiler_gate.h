#pragma once
// PS2X_PROFILE_GATED=1: the sampling profiler (PS2X_PROFILE, src/profiling/sampling_profiler.cpp) only samples
// while this is set, so a game layer can profile just the frames it cares about (e.g. the slow ones) by
// opening and closing it. Without PS2X_PROFILE_GATED the profiler ignores it.
#include <atomic>

namespace ps2x::profiling
{
    inline std::atomic<bool> g_sampleGate{false};

    inline void setSampleGate(bool open) { g_sampleGate.store(open, std::memory_order_relaxed); }
}
