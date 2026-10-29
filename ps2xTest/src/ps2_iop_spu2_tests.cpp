#include "MiniTest.h"
#include "../../ps2xIOP/src/emulator/core/iop_memory.h"
#include "../../ps2xIOP/src/emulator/core/iop_spu2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    using ps2x::iop::detail::IopMemory;
    using ps2x::iop::detail::IopSpu2;

    constexpr uint32_t kBase = 0x1F900000u;
    constexpr uint32_t kDma4Madr = 0x1F8010C0u;
    constexpr uint32_t kDma4Bcr = 0x1F8010C4u;
    constexpr uint32_t kDma4Chcr = 0x1F8010C8u;
    constexpr uint32_t kStream = 0x40000u;
    constexpr uint32_t kChunks = 64u;
    constexpr int16_t kUnwritten = 0x7000;

    void setTestEnv(const char *name, const char *value)
    {
#if defined(_WIN32)
        _putenv_s(name, value ? value : "");
#else
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
#endif
    }

    struct WallClockStream
    {
        std::unique_ptr<IopMemory> memory;
        std::unique_ptr<IopSpu2> spu2;
        uint64_t cycle = 0;
        uint32_t written = 0;

        WallClockStream()
        {
            const char *previous = std::getenv("PS2X_SPU2_WALLCLOCK");
            const std::string saved = previous ? previous : "";
            setTestEnv("PS2X_SPU2_WALLCLOCK", "1");
            memory = std::make_unique<IopMemory>();
            spu2 = std::make_unique<IopSpu2>(*memory);
            setTestEnv("PS2X_SPU2_WALLCLOCK", previous ? saved.c_str() : nullptr);
            memory->setSpu2(spu2.get());
            spu2->setClock(&cycle);
            spu2->setInterruptSink([](int) {});

            memory->write16(kBase + 0x198u, 0x0FF0u);
            memory->write16(kBase + 0x400u + 0x198u, 0x0FFCu);
            for (uint32_t core = 0; core < 2u; ++core)
            {
                memory->write16(kBase + 0x760u + core * 0x28u, 0x3FFFu);
                memory->write16(kBase + 0x762u + core * 0x28u, 0x3FFFu);
            }
            memory->write16(kBase + 0x788u + 0x08u, 0x7FFFu);
            memory->write16(kBase + 0x788u + 0x0Au, 0x7FFFu);
            memory->write16(kBase + 0x76Cu, 0x7FFFu);
            memory->write16(kBase + 0x76Eu, 0x7FFFu);

            const std::vector<int16_t> unwritten(kChunks * 512u, kUnwritten);
            (void)memory->writeRam(kStream, unwritten.data(), unwritten.size() * 2u);
        }

        void start()
        {
            memory->write16(kBase + 0x1B0u, 1u);
            memory->write32(kDma4Madr, kStream);
            memory->write32(kDma4Bcr, ((kChunks * 1024u / 64u) << 16u) | 16u);
            memory->write32(kDma4Chcr, 0x01000201u);
        }

        void writeAhead(uint64_t lead)
        {
            while (written < kChunks && written * 256u <= cycle / IopSpu2::CyclesPerSample + lead)
            {
                std::vector<int16_t> chunk(512u);
                for (uint32_t i = 0; i < 256u; ++i)
                {
                    const double t = static_cast<double>(written * 256u + i) / 48000.0;
                    chunk[i] = chunk[256u + i] = static_cast<int16_t>(4000.0 * std::sin(2.0 * 3.14159265358979 * 440.0 * t));
                }
                (void)memory->writeRam(kStream + written * 1024u, chunk.data(), 1024u);
                ++written;
            }
        }

        void advance(uint64_t samples)
        {
            cycle += samples * IopSpu2::CyclesPerSample;
            spu2->advanceTo(cycle);
        }
    };

    enum class Feed
    {
        Interrupt,
        Poll,
    };

    struct AdmaStream
    {
        int core = 0;
        uint32_t ring = 0;
        uint32_t half = 0;
        Feed feed = Feed::Interrupt;
        int dmaHalf = 0;
        uint64_t nextChunk = 0;
        uint32_t writePos = 0;

        [[nodiscard]] int16_t left(uint64_t frame) const
        {
            const int64_t step = core == 0 ? 5 : 3;
            return static_cast<int16_t>(static_cast<int64_t>((frame * step) % 12000u) - 6000);
        }

        [[nodiscard]] int16_t right(uint64_t frame) const
        {
            const uint64_t stair = core == 0 ? 2400u : 1700u;
            return static_cast<int16_t>(static_cast<int64_t>(((frame / stair) * 37u) % 12000u) - 6000 + (core == 0 ? 0 : 1));
        }
    };

    struct AdmaRig
    {
        std::unique_ptr<IopMemory> memory;
        std::unique_ptr<IopSpu2> spu2;
        uint64_t cycle = 0;
        std::vector<int> irqs;
        std::vector<int16_t> output;
        std::vector<AdmaStream> streams;

        AdmaRig(bool wallClock, std::vector<AdmaStream> feeds) : streams(std::move(feeds))
        {
            const char *previous = std::getenv("PS2X_SPU2_WALLCLOCK");
            const std::string saved = previous ? previous : "";
            setTestEnv("PS2X_SPU2_WALLCLOCK", wallClock ? "1" : nullptr);
            memory = std::make_unique<IopMemory>();
            spu2 = std::make_unique<IopSpu2>(*memory);
            setTestEnv("PS2X_SPU2_WALLCLOCK", previous ? saved.c_str() : nullptr);
            memory->setSpu2(spu2.get());
            spu2->setClock(&cycle);
            spu2->setInterruptSink([this](int irq) { irqs.push_back(irq); });
            spu2->setOutputSink([this](const int16_t *frames, size_t count)
                                { output.insert(output.end(), frames, frames + count * 2u); });

            memory->write16(kBase + 0x198u, 0x0FF0u);
            memory->write16(kBase + 0x400u + 0x198u, 0x0FFCu);
            for (uint32_t core = 0; core < 2u; ++core)
            {
                memory->write16(kBase + 0x760u + core * 0x28u, 0x3FFFu);
                memory->write16(kBase + 0x762u + core * 0x28u, 0x3FFFu);
                memory->write16(kBase + 0x76Cu + core * 0x28u, 0x7FFFu);
                memory->write16(kBase + 0x76Eu + core * 0x28u, 0x7FFFu);
            }
            memory->write16(kBase + 0x788u + 0x08u, 0x7FFFu);
            memory->write16(kBase + 0x788u + 0x0Au, 0x7FFFu);

            for (AdmaStream &stream : streams)
            {
                writeChunks(stream, 0u, 2u * stream.half / 1024u);
                stream.writePos = 0u;
                memory->write16(kBase + static_cast<uint32_t>(stream.core) * 0x400u + 0x1B0u,
                                static_cast<uint16_t>(1u << stream.core));
                startHalf(stream, 0);
            }
        }

        void writeChunks(AdmaStream &stream, uint32_t offset, uint32_t chunks)
        {
            const uint32_t ringBytes = 2u * stream.half;
            for (uint32_t i = 0; i < chunks; ++i)
            {
                std::array<int16_t, 512> chunk{};
                for (uint32_t s = 0; s < 256u; ++s)
                {
                    const uint64_t frame = stream.nextChunk * 256u + s;
                    chunk[s] = stream.left(frame);
                    chunk[256u + s] = stream.right(frame);
                }
                (void)memory->writeRam(stream.ring + (offset + i * 1024u) % ringBytes, chunk.data(), 1024u);
                ++stream.nextChunk;
            }
        }

        void startHalf(AdmaStream &stream, int half)
        {
            const uint32_t dma = IopSpu2::DmaBase[stream.core];
            stream.dmaHalf = half;
            memory->write32(dma, stream.ring + static_cast<uint32_t>(half) * stream.half);
            memory->write32(dma + 4u, ((stream.half / 64u) << 16u) | 16u);
            memory->write32(dma + 8u, 0x01000201u);
        }

        void service()
        {
            std::vector<int> pending;
            pending.swap(irqs);
            for (const int irq : pending)
            {
                for (AdmaStream &stream : streams)
                {
                    if (irq != IopSpu2::DmaIrq[stream.core])
                        continue;
                    const int finished = stream.dmaHalf;
                    startHalf(stream, 1 - finished);
                    if (stream.feed == Feed::Interrupt)
                        writeChunks(stream, static_cast<uint32_t>(finished) * stream.half, stream.half / 1024u);
                }
            }
        }

        void poll()
        {
            for (AdmaStream &stream : streams)
            {
                if (stream.feed != Feed::Poll)
                    continue;
                const uint32_t ringBytes = 2u * stream.half;
                uint32_t pos = (memory->read32(IopSpu2::DmaBase[stream.core]) & 0x00FFFFFFu) - stream.ring;
                pos -= pos % 1024u;
                pos = (pos + ringBytes - 1024u) % ringBytes;
                const uint32_t bytes = (pos + ringBytes - stream.writePos) % ringBytes;
                writeChunks(stream, stream.writePos, bytes / 1024u);
                stream.writePos = pos;
            }
        }

        void runTo(uint64_t target)
        {
            while (cycle < target)
            {
                cycle = std::min(target, std::max(cycle + 1u, spu2->nextEventCycle()));
                spu2->advanceTo(cycle);
                service();
            }
        }
    };

    struct WallClockRun
    {
        size_t pull = 1024u;
        double speed = 1.0;
        uint64_t slice = 15u;
        uint64_t frames = 48000u;
    };

    std::vector<int16_t> referenceOutput(const std::vector<AdmaStream> &streams, uint64_t samples)
    {
        AdmaRig rig(false, streams);
        constexpr uint64_t kEeFrame = 800u;
        for (uint64_t at = kEeFrame; at <= samples; at += kEeFrame)
        {
            rig.runTo(at * IopSpu2::CyclesPerSample);
            rig.poll();
        }
        rig.spu2->flush();
        return rig.output;
    }

    std::vector<int16_t> wallClockOutput(const std::vector<AdmaStream> &streams, const WallClockRun &run, IopSpu2::Stats &stats)
    {
        AdmaRig rig(true, streams);
        constexpr uint64_t kEeFrame = 800u;
        std::vector<int16_t> played;
        std::vector<int16_t> frames(run.pull * 2u);
        uint64_t nextPull = run.pull;
        uint64_t nextIop = run.slice;
        uint64_t nextEe = kEeFrame;
        while (nextPull <= run.frames)
        {
            const uint64_t host = std::min({nextPull, nextIop, nextEe});
            if (host == nextIop)
            {
                rig.runTo(static_cast<uint64_t>(static_cast<double>(host) * run.speed * IopSpu2::CyclesPerSample));
                nextIop += run.slice;
            }
            if (host == nextEe)
            {
                rig.poll();
                nextEe += kEeFrame;
            }
            if (host == nextPull)
            {
                rig.spu2->pull(frames.data(), run.pull);
                played.insert(played.end(), frames.begin(), frames.end());
                nextPull += run.pull;
            }
        }
        stats = rig.spu2->stats();
        return played;
    }

    struct Comparison
    {
        size_t matched = 0;
        size_t silent = 0;
        size_t unmatched = 0;
        size_t backward = 0;
        size_t firstUnmatched = SIZE_MAX;
    };

    Comparison compareToReference(const std::vector<int16_t> &reference, const std::vector<int16_t> &played)
    {
        constexpr size_t kWindow = 4u;
        const auto frameAt = [](const std::vector<int16_t> &samples, size_t index)
        {
            return (static_cast<uint32_t>(static_cast<uint16_t>(samples[index * 2u])) << 16u) |
                   static_cast<uint16_t>(samples[index * 2u + 1u]);
        };
        const auto windowKey = [&](const std::vector<int16_t> &samples, size_t index)
        {
            uint64_t key = 1469598103934665603ull;
            for (size_t i = 0; i < kWindow; ++i)
                key = (key ^ frameAt(samples, index + i)) * 1099511628211ull;
            return key;
        };
        const size_t referenceFrames = reference.size() / 2u;
        const size_t playedFrames = played.size() / 2u;
        std::unordered_map<uint64_t, size_t> windows;
        for (size_t i = 0; i + kWindow <= referenceFrames; ++i)
            if (frameAt(reference, i) != 0u)
                windows.emplace(windowKey(reference, i), i);

        Comparison result;
        size_t position = SIZE_MAX;
        size_t furthest = 0;
        for (size_t j = 0; j < playedFrames; ++j)
        {
            const uint32_t frame = frameAt(played, j);
            if (position < referenceFrames && frameAt(reference, position) == frame)
            {
                ++result.matched;
                ++position;
                furthest = std::max(furthest, position);
                continue;
            }
            if (frame == 0u)
            {
                ++result.silent;
                continue;
            }
            if (j + kWindow <= playedFrames)
            {
                const auto found = windows.find(windowKey(played, j));
                if (found != windows.end())
                {
                    if (found->second + 1u < furthest)
                        ++result.backward;
                    else
                        ++result.matched;
                    position = found->second + 1u;
                    furthest = std::max(furthest, position);
                    continue;
                }
            }
            ++result.unmatched;
            result.firstUnmatched = std::min(result.firstUnmatched, j);
            position = SIZE_MAX;
        }
        return result;
    }
}

void register_ps2_iop_spu2_tests()
{
    MiniTest::Case("PS2IopSpu2WallClock", [](TestCase &tc)
    {
        tc.Run("core 0 AutoDMA PCM is not mixed ahead of a stream filled on a slower IOP clock", [](TestCase &t)
        {
            WallClockStream stream;
            stream.writeAhead(2048u);
            stream.start();
            std::vector<int16_t> played;
            std::vector<int16_t> frames(1024u * 2u);
            while (stream.cycle / IopSpu2::CyclesPerSample < (kChunks - 4u) * 256u)
            {
                stream.spu2->pull(frames.data(), 1024u);
                played.insert(played.end(), frames.begin(), frames.end());
                for (int burst = 0; burst < 6; ++burst)
                {
                    stream.advance(128u);
                    stream.writeAhead(2048u);
                }
            }
            const size_t unwritten = static_cast<size_t>(std::count_if(played.begin(), played.end(), [](int16_t sample)
                                                                      { return std::abs(sample) > 6000; }));
            const size_t sound = static_cast<size_t>(std::count_if(played.begin(), played.end(), [](int16_t sample)
                                                                  { return std::abs(sample) > 2000; }));
            t.Equals(unwritten, size_t{0}, "the input never reaches stream data the IOP has not written yet");
            t.IsTrue(sound > played.size() / 4u, "the stream itself plays");
            t.IsTrue(stream.spu2->stats().gapFrames > 0u, "the device's extra frames are silence, not mixed ahead");
        });

        tc.Run("core 0 AutoDMA input keeps the IOP clock while the device takes nothing", [](TestCase &t)
        {
            WallClockStream stream;
            stream.writeAhead(kChunks * 256u);
            stream.start();
            stream.advance(32u * 256u);
            const uint32_t consumed = (stream.memory->read32(kDma4Madr) - kStream) / 1024u;
            t.IsTrue(consumed >= 30u && consumed <= 33u, "about one 1 KB chunk per 256 samples of IOP time, got " + std::to_string(consumed));
            t.IsTrue(stream.spu2->stats().droppedFrames > 0u, "the frames no one takes are dropped from the queue");
        });
    });

    MiniTest::Case("PS2IopSpu2WallClockMatchesIopClock", [](TestCase &tc)
    {
        const auto check = [](TestCase &t, const std::vector<AdmaStream> &streams, const std::vector<WallClockRun> &runs)
        {
            for (const WallClockRun &run : runs)
            {
                IopSpu2::Stats stats;
                const std::vector<int16_t> played = wallClockOutput(streams, run, stats);
                const std::vector<int16_t> reference =
                    referenceOutput(streams, static_cast<uint64_t>(static_cast<double>(run.frames) * run.speed) + 8192u);
                const Comparison result = compareToReference(reference, played);
                const std::string where = "pull " + std::to_string(run.pull) + ", IOP speed " +
                                          std::to_string(run.speed) + ", IOP slice " + std::to_string(run.slice) +
                                          " (matched " + std::to_string(result.matched) + ", silent " +
                                          std::to_string(result.silent) + ", unmatched " + std::to_string(result.unmatched) +
                                          " from frame " + std::to_string(result.firstUnmatched) + ", backward " +
                                          std::to_string(result.backward) + ", late " + std::to_string(stats.lateFrames) +
                                          ", gap " + std::to_string(stats.gapFrames) + ", drop " +
                                          std::to_string(stats.droppedFrames) + ", underruns " +
                                          std::to_string(stats.admaUnderruns) + ")";
                t.Equals(result.unmatched, size_t{0}, "every frame played is one the IOP clock mixes: " + where);
                t.Equals(result.backward, size_t{0}, "no stretch of the stream plays twice: " + where);
                t.IsTrue(result.matched > played.size() / 4u, "the stream plays: " + where);
            }
        };
        std::vector<WallClockRun> runs;
        for (const size_t pull : {256u, 480u, 1024u})
            for (const double speed : {0.75, 1.0, 1.25})
                for (const uint64_t slice : {15u, 480u, 1920u})
                    runs.push_back({pull, speed, slice, 72000u});

        tc.Run("core 0 AutoDMA PCM, refilled at each transfer's interrupt", [=](TestCase &t)
        {
            check(t, {AdmaStream{0, 0x40000u, 24u * 1024u, Feed::Interrupt}}, runs);
            check(t, {AdmaStream{0, 0x40000u, 4u * 1024u, Feed::Interrupt}}, runs);
        });

        tc.Run("core 0 AutoDMA PCM, refilled behind the transfer address polled every frame", [=](TestCase &t)
        {
            check(t, {AdmaStream{0, 0x40000u, 24u * 1024u, Feed::Poll}}, runs);
        });

        tc.Run("core 0 and core 1 AutoDMA PCM at once", [=](TestCase &t)
        {
            check(t, {AdmaStream{0, 0x40000u, 24u * 1024u, Feed::Interrupt}, AdmaStream{1, 0x60000u, 24u * 1024u, Feed::Interrupt}},
                  runs);
        });
    });
}
