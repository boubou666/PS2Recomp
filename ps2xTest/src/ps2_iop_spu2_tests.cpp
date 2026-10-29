#include "MiniTest.h"
#include "../../ps2xIOP/src/emulator/core/iop_memory.h"
#include "../../ps2xIOP/src/emulator/core/iop_spu2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
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
}
