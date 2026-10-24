// Replays VU1 microprogram runs recorded with PS2X_VU1_RECORD and compares the interpreter's results
// (registers, VU1 data memory, PATH1 packets, cycle count) with the recording.
//   ps2_vu1_replay <trace> [repeat] [--ignore-flags]   repeat > 1 also reports the interpreter's speed.
// --ignore-flags leaves out the MAC and status flags, which compiled microcode skips where no instruction can read
// them (VU1Interpreter::kHintFlagsUnobserved).
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_vu1_trace.h"

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
    for (int pass = 0; pass < repeat; ++pass)
    {
        const auto start = std::chrono::steady_clock::now();
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
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
    return mismatches == 0 ? 0 : 1;
}
