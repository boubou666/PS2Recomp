// Replays VU1 microprogram runs recorded with PS2X_VU1_RECORD and compares the interpreter's results
// (registers, VU1 data memory, PATH1 packets, cycle count) with the recording.
//   ps2_vu1_replay <trace> [repeat] [--ignore-flags]   repeat > 1 also reports the interpreter's speed.
// --ignore-flags leaves out the MAC and status flags, which compiled microcode skips where no instruction can read
// them (VU1Interpreter::kHintFlagsUnobserved).
//   ps2_vu1_replay <trace> stats      VU cycles per microprogram (code image and start pc).
//   ps2_vu1_replay <trace> dumpcode <dir>   writes each distinct code image as <dir>/<hash>.bin.
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_vu1_trace.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOGDI
#define NOUSER
#include <windows.h>
#else
#include <time.h>
#endif

// ps2_runtime expects the generated runner to provide its function table; this tool runs no guest code.
struct R5900Context;
class PS2Runtime;
extern const uint32_t g_ps2RecompiledFunctionTableBase = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = 0u;
void (*g_ps2RecompiledFunctionTable[1])(uint8_t *, R5900Context *, PS2Runtime *) = {nullptr};

namespace
{
    bool g_ignoreFlags = false;
    bool g_tolerance = false;

    // --tolerance: how far a run's results are from the recording, word by word (as signed 32-bit integers:
    // float words then differ by their distance in ulps, FTOI results by their integer difference).
    struct ToleranceStats
    {
        size_t runs = 0, exact = 0, cycleDiffer = 0, packetShapeDiffer = 0;
        size_t words = 0, differing = 0;
        size_t buckets[5]{}; // |diff| 1, 2-4, 5-16, 17-256, more
        int64_t maxDiff = 0;
        void word(uint32_t expected, uint32_t actual)
        {
            ++words;
            if (expected == actual)
                return;
            ++differing;
            const int64_t diff = std::llabs(static_cast<int64_t>(static_cast<int32_t>(expected)) -
                                            static_cast<int64_t>(static_cast<int32_t>(actual)));
            maxDiff = std::max(maxDiff, diff);
            buckets[diff <= 1 ? 0 : diff <= 4 ? 1 : diff <= 16 ? 2 : diff <= 256 ? 3 : 4]++;
        }
        void bytes(const uint8_t *expected, const uint8_t *actual, size_t size)
        {
            for (size_t offset = 0; offset + 4u <= size; offset += 4u)
            {
                uint32_t e = 0, a = 0;
                std::memcpy(&e, expected + offset, 4);
                std::memcpy(&a, actual + offset, 4);
                word(e, a);
            }
        }
    };

    std::string describeStateDifference(const VU1State &expected, const VU1State &actual)
    {
        std::string diff;
        char line[256];
        for (int reg = 0; reg < 32; ++reg)
            if (std::memcmp(expected.vf[reg], actual.vf[reg], 16) != 0)
            {
                uint32_t e[4], a[4];
                std::memcpy(e, expected.vf[reg], 16);
                std::memcpy(a, actual.vf[reg], 16);
                std::snprintf(line, sizeof(line), " vf%d exp=%08x,%08x,%08x,%08x got=%08x,%08x,%08x,%08x", reg, e[0], e[1],
                              e[2], e[3], a[0], a[1], a[2], a[3]);
                diff += line;
            }
        for (int reg = 0; reg < 16; ++reg)
            if (expected.vi[reg] != actual.vi[reg])
            {
                std::snprintf(line, sizeof(line), " vi%d exp=%d got=%d", reg, expected.vi[reg], actual.vi[reg]);
                diff += line;
            }
        auto scalar = [&](const char *name, const void *e, const void *a, size_t size)
        {
            if (std::memcmp(e, a, size) != 0)
            {
                uint32_t ev = 0, av = 0;
                std::memcpy(&ev, e, size < 4 ? size : 4);
                std::memcpy(&av, a, size < 4 ? size : 4);
                std::snprintf(line, sizeof(line), " %s exp=%08x got=%08x", name, ev, av);
                diff += line;
            }
        };
        scalar("acc", expected.acc, actual.acc, 16);
        scalar("q", &expected.q, &actual.q, 4);
        scalar("p", &expected.p, &actual.p, 4);
        scalar("i", &expected.i, &actual.i, 4);
        scalar("r", &expected.r, &actual.r, 4);
        scalar("pc", &expected.pc, &actual.pc, 4);
        if (!g_ignoreFlags)
            scalar("mac", &expected.mac, &actual.mac, 4);
        scalar("clip", &expected.clip, &actual.clip, 4);
        if (!g_ignoreFlags)
            scalar("status", &expected.status, &actual.status, 4);
        scalar("stoppedByD", &expected.stoppedByD, &actual.stoppedByD, 1);
        scalar("stoppedByT", &expected.stoppedByT, &actual.stoppedByT, 1);
        return diff;
    }
}

// The replay thread's own CPU time: other programs on the machine inflate wall time, not this. TSC cycles on
// Windows (QueryThreadCycleTime), nanoseconds elsewhere.
static uint64_t threadCpuTicks()
{
#if defined(_WIN32)
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return cycles;
#else
    timespec now{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
    return static_cast<uint64_t>(now.tv_sec) * 1000000000ull + static_cast<uint64_t>(now.tv_nsec);
#endif
}

int main(int argc, char **argv)
{
    std::vector<const char *> args;
    for (int index = 1; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--ignore-flags") == 0)
            g_ignoreFlags = true;
        else if (std::strcmp(argv[index], "--tolerance") == 0)
            g_ignoreFlags = g_tolerance = true;
        else
            args.push_back(argv[index]);
    }
    if (args.empty())
    {
        std::fprintf(stderr, "usage: %s <trace> [repeat] [--ignore-flags]\n", argv[0]);
        return 2;
    }
    FILE *in = std::fopen(args[0], "rb");
    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", args[0]);
        return 2;
    }
    std::vector<Vu1TraceRecord> records;
    for (Vu1TraceRecord record; readVu1TraceRecord(in, record);)
        records.push_back(record);
    std::fclose(in);
    if (args.size() > 2 && std::string_view(args[1]) == "dumpcode")
    {
        std::map<size_t, const std::vector<uint8_t> *> images;
        for (const Vu1TraceRecord &record : records)
            if (!record.code.empty())
                images.emplace(std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char *>(record.code.data()), record.code.size())),
                               &record.code);
        for (const auto &[hash, code] : images)
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%016zx.bin", args[2], hash);
            if (FILE *out = std::fopen(path, "wb"))
            {
                std::fwrite(code->data(), 1, code->size(), out);
                std::fclose(out);
            }
            std::printf("%s\n", path);
        }
        return 0;
    }
    if (args.size() > 1 && std::string_view(args[1]) == "stats")
    {
        struct Program
        {
            uint64_t runs = 0;
            uint64_t cycles = 0;
        };
        std::map<std::pair<size_t, uint32_t>, Program> programs;
        size_t codeHash = 0;
        uint64_t total = 0;
        for (const Vu1TraceRecord &record : records)
        {
            if (!record.code.empty())
                codeHash = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char *>(record.code.data()), record.code.size()));
            Program &program = programs[{codeHash, record.startPC}];
            const uint64_t cycles = record.stateAfter.cycles - record.stateBefore.cycles;
            ++program.runs;
            program.cycles += cycles;
            total += cycles;
        }
        std::vector<std::pair<std::pair<size_t, uint32_t>, Program>> sorted(programs.begin(), programs.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.second.cycles > b.second.cycles; });
        std::printf("programs=%zu runs=%zu vu-cycles=%llu\n", sorted.size(), records.size(), static_cast<unsigned long long>(total));
        for (size_t i = 0; i < sorted.size() && i < 40u; ++i)
            std::printf("code=%016zx pc=0x%04x runs=%llu cycles=%llu (%.1f%%) avg=%llu\n", sorted[i].first.first, sorted[i].first.second,
                        static_cast<unsigned long long>(sorted[i].second.runs), static_cast<unsigned long long>(sorted[i].second.cycles),
                        100.0 * static_cast<double>(sorted[i].second.cycles) / static_cast<double>(std::max<uint64_t>(total, 1u)),
                        static_cast<unsigned long long>(sorted[i].second.cycles / std::max<uint64_t>(sorted[i].second.runs, 1u)));
        return 0;
    }
    // "parallel": could two threads run consecutive microprograms at once? For each run u (an MSCAL with its
    // MSCNT continuations) after a run p, replays u from the state a second thread would have had: the real state
    // before u with every register and data byte p changed put back as it was before p (unless VIF1 rewrote it in
    // between), then compares what u produced with the recording: its GS packets, the data bytes and registers
    // either version wrote, and its cycles. Equal: u did not depend on p, so both could have run in parallel and
    // been merged. Then estimates, with each run's recorded cycles, the time two workers would take.
    if (args.size() > 1 && std::string_view(args[1]) == "parallel")
    {
        struct Unit
        {
            size_t first = 0, last = 0;
            uint64_t cycles = 0;
        };
        // PS2X_PARALLEL_RECORDS=1: every record (MSCNT included) is a run of its own.
        const bool perRecord = std::getenv("PS2X_PARALLEL_RECORDS") != nullptr;
        std::vector<Unit> units;
        std::vector<const std::vector<uint8_t> *> codeAt(records.size(), nullptr);
        const std::vector<uint8_t> *currentCode = nullptr;
        for (size_t index = 0; index < records.size(); ++index)
        {
            if (!records[index].code.empty())
                currentCode = &records[index].code;
            codeAt[index] = currentCode;
            if (!records[index].resume || units.empty() || perRecord)
                units.push_back({index, index, 0});
            units.back().last = index;
            units.back().cycles += records[index].stateAfter.cycles - records[index].stateBefore.cycles;
        }

        // Register slots compared and restored one by one (cycles, pc and the branch/halt state are run bookkeeping).
        struct Slot
        {
            size_t offset, size;
            std::string name;
        };
        std::vector<Slot> slots;
        for (int reg = 0; reg < 32; ++reg)
            slots.push_back({offsetof(VU1State, vf) + reg * 16u, 16u, "vf" + std::to_string(reg)});
        for (int reg = 0; reg < 16; ++reg)
            slots.push_back({offsetof(VU1State, vi) + reg * 4u, 4u, "vi" + std::to_string(reg)});
        slots.push_back({offsetof(VU1State, acc), 16u, "acc"});
        slots.push_back({offsetof(VU1State, q), 4u, "q"});
        slots.push_back({offsetof(VU1State, p), 4u, "p"});
        slots.push_back({offsetof(VU1State, i), 4u, "i"});
        slots.push_back({offsetof(VU1State, r), 4u, "r"});
        slots.push_back({offsetof(VU1State, mac), 4u, "mac"});
        slots.push_back({offsetof(VU1State, clip), 4u, "clip"});
        slots.push_back({offsetof(VU1State, status), 4u, "status"});
        const auto slotOf = [](const VU1State &state, const Slot &slot)
        { return reinterpret_cast<const uint8_t *>(&state) + slot.offset; };
        const auto slotOfMut = [](VU1State &state, const Slot &slot)
        { return reinterpret_cast<uint8_t *>(&state) + slot.offset; };

        GS gs;
        VU1Interpreter vu;
        std::vector<std::vector<uint8_t>> got;
        vu.setPacketObserver([&got](const uint8_t *packet, uint32_t size) { got.emplace_back(packet, packet + size); });
        std::vector<uint8_t> runCode(Vu1TraceRecord::kMemorySize), runData(Vu1TraceRecord::kMemorySize);

        // Replays unit u from (state, data); returns packets, final data and state, and cycles.
        const auto runUnit = [&](const Unit &u, VU1State state, std::vector<uint8_t> data, std::vector<uint8_t> &dataOut,
                                 VU1State &stateOut, uint64_t &cycles)
        {
            got.clear();
            cycles = 0;
            for (size_t index = u.first; index <= u.last; ++index)
            {
                const Vu1TraceRecord &record = records[index];
                if (codeAt[index] && (index == u.first || !record.code.empty()))
                {
                    runCode = *codeAt[index];
                    vu.invalidateDecodedCode();
                }
                if (index == u.first)
                {
                    vu.state() = state;
                    runData = data;
                }
                else
                {
                    // VIF1 writes between the slices of this run (bytes changed between them), applied as they are.
                    const Vu1TraceRecord &before = records[index - 1];
                    for (size_t b = 0; b < runData.size(); ++b)
                        if (record.dataBefore[b] != before.dataAfter[b])
                            runData[b] = record.dataBefore[b];
                }
                const uint64_t start = vu.state().cycles;
                if (record.resume)
                    vu.resume(runCode.data(), Vu1TraceRecord::kMemorySize, runData.data(), Vu1TraceRecord::kMemorySize, gs,
                              nullptr, record.top, record.itop, record.maxCycles);
                else
                    vu.execute(runCode.data(), Vu1TraceRecord::kMemorySize, runData.data(), Vu1TraceRecord::kMemorySize,
                               gs, nullptr, record.startPC, record.top, record.itop, record.maxCycles);
                cycles += vu.state().cycles - start;
            }
            dataOut = runData;
            stateOut = vu.state();
        };

        // PS2X_PARALLEL_CONTROL=1: nothing put back, every run should come out independent (checks the method).
        const bool control = std::getenv("PS2X_PARALLEL_CONTROL") != nullptr;
        size_t independent = 0, dependent = 0, timingOnly = 0;
        uint64_t independentCycles = 0, dependentCycles = 0;
        std::map<std::string, size_t> reasons;
        std::vector<bool> dependsOnPrevious(units.size(), true);
        for (size_t n = 1; n < units.size(); ++n)
        {
            const Unit &p = units[n - 1], &u = units[n];
            const Vu1TraceRecord &pFirst = records[p.first], &pLast = records[p.last];
            const Vu1TraceRecord &uFirst = records[u.first], &uLast = records[u.last];

            VU1State spec = uFirst.stateBefore;
            for (const Slot &slot : slots)
            {
                const bool pChanged = std::memcmp(slotOf(pLast.stateAfter, slot), slotOf(pFirst.stateBefore, slot), slot.size) != 0;
                const bool keptSince = std::memcmp(slotOf(uFirst.stateBefore, slot), slotOf(pLast.stateAfter, slot), slot.size) == 0;
                if (pChanged && keptSince && !control)
                    std::memcpy(slotOfMut(spec, slot), slotOf(pFirst.stateBefore, slot), slot.size);
            }
            std::vector<uint8_t> specData = uFirst.dataBefore;
            for (size_t b = 0; b < specData.size(); ++b)
                if (!control && pLast.dataAfter[b] != pFirst.dataBefore[b] && uFirst.dataBefore[b] == pLast.dataAfter[b])
                    specData[b] = pFirst.dataBefore[b];

            std::vector<uint8_t> dataAfter;
            VU1State stateAfter{};
            uint64_t cycles = 0;
            runUnit(u, spec, specData, dataAfter, stateAfter, cycles);

            std::vector<std::vector<uint8_t>> expectedPackets;
            for (size_t index = u.first; index <= u.last; ++index)
                for (const auto &packet : records[index].packets)
                    expectedPackets.push_back(packet);
            std::string why;
            if (got != expectedPackets)
                why += " packets";
            bool dataDiffers = false;
            for (size_t b = 0; b < dataAfter.size() && !dataDiffers; ++b)
            {
                const bool written = dataAfter[b] != specData[b] || uLast.dataAfter[b] != uFirst.dataBefore[b];
                dataDiffers = written && dataAfter[b] != uLast.dataAfter[b];
            }
            if (dataDiffers)
                why += " data";
            for (const Slot &slot : slots)
            {
                const bool written = std::memcmp(slotOf(stateAfter, slot), slotOf(spec, slot), slot.size) != 0 ||
                                     std::memcmp(slotOf(uLast.stateAfter, slot), slotOf(uFirst.stateBefore, slot), slot.size) != 0;
                if (written && std::memcmp(slotOf(stateAfter, slot), slotOf(uLast.stateAfter, slot), slot.size) != 0)
                {
                    why += " " + slot.name.substr(0, 2);
                    break;
                }
            }
            if (why.empty())
            {
                ++independent;
                independentCycles += u.cycles;
                dependsOnPrevious[n] = false;
                if (cycles != u.cycles)
                    ++timingOnly;
            }
            else
            {
                ++dependent;
                dependentCycles += u.cycles;
                ++reasons[why];
            }
        }
        uint64_t total = 0;
        for (const Unit &u : units)
            total += u.cycles;
        std::printf("runs=%zu (records %zu), cycles=%llu\n", units.size(), records.size(), static_cast<unsigned long long>(total));
        size_t vifBetween = 0, resumes = 0;
        for (size_t index = 1; index < records.size(); ++index)
            if (records[index].resume)
            {
                ++resumes;
                vifBetween += records[index].dataBefore != records[index - 1].dataAfter ? 1u : 0u;
            }
        std::printf("MSCNT records: %zu, %zu of them after VIF1 wrote VU memory (new batches, not time slices)\n", resumes,
                    vifBetween);
        std::printf("independent of the run before: %zu runs, %.1f%% of cycles (%zu of them with other cycle counts)\n",
                    independent, 100.0 * independentCycles / std::max<uint64_t>(total, 1u), timingOnly);
        std::printf("dependent: %zu runs, %.1f%% of cycles\n", dependent, 100.0 * dependentCycles / std::max<uint64_t>(total, 1u));
        for (const auto &[why, count] : reasons)
            std::printf("   differs in%s: %zu\n", why.c_str(), count);

        // Size of the runs: the share of cycles by run length, as threads cost microseconds to hand work over.
        const uint64_t bounds[] = {256, 1024, 4096, 16384, 65536, ~0ull};
        uint64_t lower = 0;
        for (uint64_t bound : bounds)
        {
            uint64_t count = 0, cycles = 0, independentHere = 0;
            for (size_t n = 0; n < units.size(); ++n)
                if (units[n].cycles >= lower && units[n].cycles < bound)
                {
                    ++count;
                    cycles += units[n].cycles;
                    independentHere += dependsOnPrevious[n] ? 0u : units[n].cycles;
                }
            std::printf("runs of %llu..%llu cycles: %llu runs, %.1f%% of cycles, %.1f%% of those independent\n",
                        static_cast<unsigned long long>(lower), static_cast<unsigned long long>(bound == ~0ull ? 0 : bound),
                        static_cast<unsigned long long>(count), 100.0 * cycles / std::max<uint64_t>(total, 1u),
                        100.0 * independentHere / std::max<uint64_t>(cycles, 1u));
            lower = bound;
        }


        // Which earlier runs each run depends on: for j = 1..kDepth, only the effects of run n-j still visible when
        // run n starts (changed by it, not changed since by a later run or VIF1) are put back; a different result
        // means run n read them. Then the runs as a graph on two workers and on unlimited ones.
        constexpr size_t kDepth = 6;
        std::vector<std::vector<size_t>> deps(units.size());
        std::vector<bool> deeper(units.size(), false); // depends on something we did not test (at kDepth, all differ)
        std::map<std::string, size_t> slotReasons;
        uint64_t regionHits[Vu1TraceRecord::kMemorySize / 256] = {};
        for (size_t n = 1; n < units.size(); ++n)
        {
            const Unit &u = units[n];
            const Vu1TraceRecord &uFirst = records[u.first], &uLast = records[u.last];
            std::vector<std::vector<uint8_t>> expectedPackets;
            for (size_t index = u.first; index <= u.last; ++index)
                for (const auto &packet : records[index].packets)
                    expectedPackets.push_back(packet);
            for (size_t j = 1; j <= kDepth && j <= n; ++j)
            {
                const Unit &r = units[n - j];
                const Vu1TraceRecord &rFirst = records[r.first], &rLast = records[r.last];
                VU1State spec = uFirst.stateBefore;
                for (const Slot &slot : slots)
                {
                    const bool changed = std::memcmp(slotOf(rLast.stateAfter, slot), slotOf(rFirst.stateBefore, slot), slot.size) != 0;
                    const bool visible = std::memcmp(slotOf(uFirst.stateBefore, slot), slotOf(rLast.stateAfter, slot), slot.size) == 0;
                    if (changed && visible)
                        std::memcpy(slotOfMut(spec, slot), slotOf(rFirst.stateBefore, slot), slot.size);
                }
                std::vector<uint8_t> specData = uFirst.dataBefore;
                for (size_t b = 0; b < specData.size(); ++b)
                    if (rLast.dataAfter[b] != rFirst.dataBefore[b] && uFirst.dataBefore[b] == rLast.dataAfter[b])
                        specData[b] = rFirst.dataBefore[b];
                if (specData == uFirst.dataBefore && std::memcmp(&spec, &uFirst.stateBefore, sizeof(spec)) == 0)
                    continue; // nothing of run n-j is still visible
                std::vector<uint8_t> dataAfter;
                VU1State stateAfter{};
                uint64_t cycles = 0;
                runUnit(u, spec, specData, dataAfter, stateAfter, cycles);
                bool differs = got != expectedPackets;
                for (size_t b = 0; b < dataAfter.size(); ++b)
                {
                    const bool written = dataAfter[b] != specData[b] || uLast.dataAfter[b] != uFirst.dataBefore[b];
                    if (written && dataAfter[b] != uLast.dataAfter[b])
                    {
                        differs = true;
                        break;
                    }
                }
                for (const Slot &slot : slots)
                {
                    const bool written = std::memcmp(slotOf(stateAfter, slot), slotOf(spec, slot), slot.size) != 0 ||
                                         std::memcmp(slotOf(uLast.stateAfter, slot), slotOf(uFirst.stateBefore, slot), slot.size) != 0;
                    if (written && std::memcmp(slotOf(stateAfter, slot), slotOf(uLast.stateAfter, slot), slot.size) != 0)
                        differs = true;
                }
                if (differs)
                {
                    deps[n].push_back(n - j);
                    if (j == 1)
                    {
                        // What of the run before mattered: registers it left that differ, data pages it left.
                        for (const Slot &slot : slots)
                            if (std::memcmp(slotOf(spec, slot), slotOf(uFirst.stateBefore, slot), slot.size) != 0)
                                ++slotReasons[slot.name];
                        for (size_t b = 0; b < specData.size(); b += 256)
                            if (std::memcmp(&specData[b], &uFirst.dataBefore[b], 256) != 0)
                                ++regionHits[b / 256];
                    }
                }
            }
            deeper[n] = deps[n].size() == std::min(kDepth, n) && n > kDepth;
        }
        std::map<size_t, uint64_t> byNearest; // cycles of runs by the distance to the nearest run they depend on
        uint64_t noDep = 0;
        for (size_t n = 1; n < units.size(); ++n)
        {
            if (deps[n].empty())
                noDep += units[n].cycles;
            else
                byNearest[n - deps[n].front()] += units[n].cycles;
        }
        std::printf("\nnearest run depended on (share of cycles): none within %zu %.1f%%", kDepth, 100.0 * noDep / total);
        for (const auto &[distance, cycles] : byNearest)
            std::printf(", %zu back %.1f%%", distance, 100.0 * cycles / total);
        std::printf("\nregisters of the run before that a run reads (runs):");
        for (const auto &[name, count] : slotReasons)
            if (count >= 20)
                std::printf(" %s:%zu", name.c_str(), count);
        std::printf("\ndata pages (256 B) of the run before that a run reads (runs):");
        for (size_t page = 0; page < std::size(regionHits); ++page)
            if (regionHits[page] >= 20)
                std::printf(" 0x%04zx:%llu", page * 256, static_cast<unsigned long long>(regionHits[page]));
        std::printf("\n");

        // List scheduling of the graph (edges to the runs within kDepth each depends on; one deeper than that is
        // treated as depending on every run before it), in order of the trace. Runs that depend on the run just
        // before join its task (one handover per task); a handover costs `handoff` VU cycles of the worker's time
        // (PS2X_PARALLEL_HANDOFF, several values separated by commas).
        std::vector<size_t> taskOf(units.size(), 0);
        size_t tasks = 0;
        for (size_t n = 0; n < units.size(); ++n)
        {
            const bool joins = n > 0 && !deps[n].empty() && deps[n].front() == n - 1;
            taskOf[n] = joins ? taskOf[n - 1] : tasks++;
        }
        std::printf("tasks (chains of runs that each depend on the one before): %zu, %.1f runs each\n", tasks,
                    static_cast<double>(units.size()) / static_cast<double>(std::max<size_t>(tasks, 1u)));
        std::vector<double> handoffs = {0.0};
        if (const char *list = std::getenv("PS2X_PARALLEL_HANDOFF"))
        {
            handoffs.clear();
            for (const char *at = list; *at;)
            {
                handoffs.push_back(std::atof(at));
                const char *comma = std::strchr(at, ',');
                at = comma ? comma + 1 : at + std::strlen(at);
            }
        }
        for (double handoff : handoffs)
            for (int workersCount : {1, 2, 3})
            {
                std::vector<double> finish(units.size(), 0.0), free(workersCount, 0.0);
                std::vector<int> taskWorker(tasks, -1);
                double latestDone = 0.0;
                for (size_t n = 0; n < units.size(); ++n)
                {
                    double ready = 0.0;
                    for (size_t d : deps[n])
                        ready = std::max(ready, finish[d]);
                    if (deeper[n])
                        ready = std::max(ready, latestDone);
                    // A task stays on its worker; a new task goes to the first free one and pays the handover.
                    int w = taskWorker[taskOf[n]];
                    double cost = static_cast<double>(units[n].cycles);
                    if (w < 0)
                    {
                        w = static_cast<int>(std::min_element(free.begin(), free.end()) - free.begin());
                        taskWorker[taskOf[n]] = w;
                        cost += workersCount > 1 ? handoff : 0.0;
                    }
                    const double start = std::max(free[w], ready);
                    finish[n] = start + cost;
                    free[w] = finish[n];
                    latestDone = std::max(latestDone, finish[n]);
                }
                std::printf("handoff %5.0f cycles, %d worker(s): %.2fx\n", handoff, workersCount,
                            static_cast<double>(total) / latestDone);
            }

        // Two workers, in order: a run starts once a worker is free, once the run two before it is done (its state
        // includes everything up to there), and, if it depends on the run before, once that one is done.
        // handoff: host cost per run handed to a worker, in VU cycles (args[2], default 0).
        const double handoff = args.size() > 2 ? std::atof(args[2]) : 0.0;
        std::vector<double> end(units.size(), 0.0);
        double workers[2] = {0.0, 0.0};
        for (size_t n = 0; n < units.size(); ++n)
        {
            int w = workers[0] <= workers[1] ? 0 : 1;
            double start = workers[w];
            if (n >= 2)
                start = std::max(start, end[n - 2]);
            if (n >= 1 && dependsOnPrevious[n])
                start = std::max(start, end[n - 1]);
            end[n] = start + static_cast<double>(units[n].cycles) + handoff;
            workers[w] = end[n];
        }
        const double makespan = *std::max_element(end.begin(), end.end());
        std::printf("two workers (handoff %.0f cycles per run): %.0f against %llu cycles serial, %.2fx\n", handoff, makespan,
                    static_cast<unsigned long long>(total), static_cast<double>(total) / makespan);
        return 0;
    }
    const int repeat = args.size() > 1 ? std::atoi(args[1]) : 1;

    GS gs; // never initialized: packets are only observed
    VU1Interpreter vu1;
    std::vector<uint8_t> code(Vu1TraceRecord::kMemorySize), data(Vu1TraceRecord::kMemorySize);
    std::vector<std::vector<uint8_t>> packets;
    vu1.setPacketObserver([&packets](const uint8_t *packet, uint32_t size)
                          { packets.emplace_back(packet, packet + size); });

    size_t mismatches = 0, resumeMismatches = 0, timingMismatches = 0;
    ToleranceStats packetStats, dataStats, registerStats;
    size_t toleranceRuns = 0, toleranceExact = 0, cycleDiffer = 0, packetShapeDiffer = 0;
    uint64_t totalCycles = 0;
    double seconds = 0.0;
    uint64_t cpuTicks = 0;
    for (int pass = 0; pass < repeat; ++pass)
    {
        const auto start = std::chrono::steady_clock::now();
        const uint64_t startTicks = threadCpuTicks();
        for (size_t index = 0; index < records.size(); ++index)
        {
            const Vu1TraceRecord &record = records[index];
            if (!record.code.empty())
            {
                code = record.code;
                vu1.invalidateDecodedCode();
            }
            const uint64_t cyclesBefore = vu1.state().cycles; // the interpreter's own clock after the previous run
            vu1.state() = record.stateBefore;
            data = record.dataBefore;
            packets.clear();
            if (record.resume)
                vu1.resume(code.data(), Vu1TraceRecord::kMemorySize, data.data(), Vu1TraceRecord::kMemorySize, gs, nullptr,
                           record.top, record.itop, record.maxCycles);
            else
                vu1.execute(code.data(), Vu1TraceRecord::kMemorySize, data.data(), Vu1TraceRecord::kMemorySize, gs,
                            nullptr, record.startPC, record.top, record.itop, record.maxCycles);
            if (pass != 0)
                continue;

            const VU1State &actual = vu1.state();
            const uint64_t expectedCycles = record.stateAfter.cycles - record.stateBefore.cycles;
            const uint64_t actualCycles = actual.cycles - cyclesBefore;
            totalCycles += expectedCycles;
            std::string diff = describeStateDifference(record.stateAfter, actual);
            const bool dataDiffers = data != record.dataAfter;
            const bool packetsDiffer = packets != record.packets;
            if (g_tolerance)
            {
                ++toleranceRuns;
                if (diff.empty() && !dataDiffers && !packetsDiffer && expectedCycles == actualCycles)
                {
                    ++toleranceExact;
                    continue;
                }
                if (expectedCycles != actualCycles)
                    ++cycleDiffer;
                bool sameShape = packets.size() == record.packets.size();
                for (size_t p = 0; sameShape && p < packets.size(); ++p)
                    sameShape = packets[p].size() == record.packets[p].size();
                if (!sameShape)
                    ++packetShapeDiffer;
                else
                    for (size_t p = 0; p < packets.size(); ++p)
                        packetStats.bytes(record.packets[p].data(), packets[p].data(), packets[p].size());
                dataStats.bytes(record.dataAfter.data(), data.data(), data.size());
                registerStats.bytes(reinterpret_cast<const uint8_t *>(record.stateAfter.vf),
                                    reinterpret_cast<const uint8_t *>(actual.vf), sizeof(actual.vf));
                registerStats.bytes(reinterpret_cast<const uint8_t *>(record.stateAfter.acc),
                                    reinterpret_cast<const uint8_t *>(actual.acc), sizeof(actual.acc));
                continue;
            }
            if (diff.empty() && !dataDiffers && !packetsDiffer)
            {
                if (expectedCycles != actualCycles && timingMismatches++ < 10)
                    std::printf("#%zu pc=0x%x cycles exp=%llu got=%llu\n", index, record.startPC,
                                static_cast<unsigned long long>(expectedCycles),
                                static_cast<unsigned long long>(actualCycles));
                continue;
            }
            size_t &counter = record.resume ? resumeMismatches : mismatches;
            if (counter++ < 10)
            {
                std::printf("#%zu %s pc=0x%x:%s%s%s\n", index, record.resume ? "resume" : "mscal", record.startPC,
                            diff.c_str(), dataDiffers ? " data" : "", packetsDiffer ? " packets" : "");
                if (packetsDiffer)
                    std::printf("   packets exp=%zu got=%zu\n", record.packets.size(), packets.size());
                if (dataDiffers)
                    for (size_t offset = 0; offset < data.size(); offset += 4)
                        if (std::memcmp(&data[offset], &record.dataAfter[offset], 4) != 0)
                        {
                            uint32_t e, a;
                            std::memcpy(&e, &record.dataAfter[offset], 4);
                            std::memcpy(&a, &data[offset], 4);
                            std::printf("   first data diff at 0x%zx exp=%08x got=%08x\n", offset, e, a);
                            break;
                        }
            }
        }
        if (pass != 0 || repeat == 1)
        {
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            cpuTicks += threadCpuTicks() - startTicks;
        }
    }

    if (g_tolerance)
    {
        std::printf("tolerance: runs=%zu exact=%zu cycles-differ=%zu packet-shape-differs=%zu\n", toleranceRuns,
                    toleranceExact, cycleDiffer, packetShapeDiffer);
        const auto report = [](const char *name, const ToleranceStats &stats)
        {
            std::printf("  %-9s words=%zu differing=%zu (|d|<=1 %zu, <=4 %zu, <=16 %zu, <=256 %zu, more %zu) max=%lld\n", name,
                        stats.words, stats.differing, stats.buckets[0], stats.buckets[1], stats.buckets[2],
                        stats.buckets[3], stats.buckets[4], static_cast<long long>(stats.maxDiff));
        };
        report("packets", packetStats);
        report("data", dataStats);
        report("registers", registerStats);
    }
    std::printf("records=%zu mismatches=%zu resume-mismatches=%zu timing-only=%zu vu-cycles=%llu\n", records.size(),
                mismatches, resumeMismatches, timingMismatches, static_cast<unsigned long long>(totalCycles));
    const int timedPasses = repeat > 1 ? repeat - 1 : 1;
    std::printf("time per pass=%.2f ms  (%.1f ns per VU cycle)\n", seconds * 1000.0 / timedPasses,
                totalCycles ? seconds * 1e9 / timedPasses / static_cast<double>(totalCycles) : 0.0);
#if defined(_WIN32)
    std::printf("thread cpu per pass=%.2f Mcycles\n", static_cast<double>(cpuTicks) / 1e6 / timedPasses);
#else
    std::printf("thread cpu per pass=%.2f ms\n", static_cast<double>(cpuTicks) / 1e6 / timedPasses);
#endif
    return mismatches == 0 ? 0 : 1;
}
