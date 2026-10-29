#include "MiniTest.h"
#include "runtime/native/native_vif1.h"
#include "runtime/native/native_vu1_stream.h"

#include <cstdint>
#include <cstring>
#include <vector>

using ps2x::native::Vu1StreamAdapter;

namespace
{
    uint32_t vif(uint32_t cmd, uint32_t num, uint32_t imm) { return (cmd << 24) | (num << 16) | imm; }

    // MPG at 0: a NOP pair with the E bit, then the NOP pair it runs before ending.
    void mpgEndingProgram(std::vector<uint32_t> &stream)
    {
        stream.push_back(0);
        stream.push_back(vif(0x4a, 2, 0));
        stream.insert(stream.end(), {0x8000033cu, 0x400002ffu, 0x8000033cu, 0x000002ffu});
    }

    void send(std::vector<uint32_t> stream)
    {
        while (stream.size() % 4u)
            stream.push_back(0);
        Vu1StreamAdapter::instance().observe(reinterpret_cast<const uint8_t *>(stream.data()), stream.size() * 4u);
    }

    bool g_active = true;
    std::vector<uint32_t> g_tops;
    bool g_translate = false;

    void install()
    {
        Vu1StreamAdapter::Config config;
        config.active = [] { return g_active; };
        config.choose = [](Vu1StreamAdapter &, uint32_t)
        { return g_translate ? Vu1StreamAdapter::Choice::Port : Vu1StreamAdapter::Choice::Sandbox; };
        config.draw = [](Vu1StreamAdapter &adapter, uint32_t, bool) { g_tops.push_back(adapter.top()); };
        Vu1StreamAdapter::instance().install(config);
    }
}

void register_native_vu1_stream_tests()
{
    MiniTest::Case("NativeVu1Stream", [](TestCase &tc)
    {
        tc.Run("the VU1 gate gets the adapter's decision for each MSCAL, in stream order", [](TestCase &t)
        {
            install();
            g_translate = false;
            std::vector<uint32_t> stream;
            mpgEndingProgram(stream);
            g_active = false;
            send(stream);
            send({vif(0x14, 0, 0)});
            g_active = true;
            send({vif(0x14, 0, 0)});
            const uint8_t *code = Vu1StreamAdapter::instance().code();
            t.IsFalse(ps2x::native::vu1Skipped(code, 0), "inactive: VU1 runs it");
            t.IsTrue(ps2x::native::vu1Skipped(code, 0), "active: the sandbox ran it, VU1 skips it");
            t.IsFalse(ps2x::native::vu1Skipped(code, 0), "nothing more decided: VU1 runs it");
        });

        tc.Run("each MPG moves the code version on, nothing else does", [](TestCase &t)
        {
            install();
            const uint64_t before = Vu1StreamAdapter::instance().codeVersion();
            send({vif(0x01, 0, 0x0101), vif(0x6c, 1, 5), 1, 2, 3, 4});
            t.Equals(Vu1StreamAdapter::instance().codeVersion(), before, "an UNPACK leaves it");
            std::vector<uint32_t> stream;
            mpgEndingProgram(stream);
            mpgEndingProgram(stream);
            send(stream);
            t.Equals(Vu1StreamAdapter::instance().codeVersion(), before + 2u, "one per MPG");
        });

        tc.Run("UNPACK fills the adapter's copy of VU1 data memory at its address", [](TestCase &t)
        {
            install();
            send({vif(0x01, 0, 0x0101), vif(0x6c, 1, 5), 1, 2, 3, 4}); // STCYCL 1,1; UNPACK V4-32 at 5
            const uint32_t *data = Vu1StreamAdapter::instance().data();
            t.Equals(data[5 * 4 + 0], 1u, "x");
            t.Equals(data[5 * 4 + 3], 4u, "w");
        });

        tc.Run("MSCALs see TOP flip between BASE and BASE + OFFSET", [](TestCase &t)
        {
            install();
            g_translate = true;
            g_active = true;
            g_tops.clear();
            send({vif(0x03, 0, 0x10), vif(0x02, 0, 0x20), vif(0x14, 0, 0), vif(0x14, 0, 0), vif(0x14, 0, 0)});
            g_translate = false;
            t.Equals(g_tops.size(), size_t{3}, "three programs drawn by the port");
            if (g_tops.size() == 3u)
            {
                t.Equals(g_tops[0], 0x10u, "first at BASE");
                t.Equals(g_tops[1], 0x30u, "then BASE + OFFSET");
                t.Equals(g_tops[2], 0x10u, "then BASE again");
            }
            for (int i = 0; i < 3; ++i)
                ps2x::native::vu1Skipped(Vu1StreamAdapter::instance().code(), 0);
        });

        tc.Run("a DIRECT that only draws is drawn natively, one carrying an image is left to the GS", [](TestCase &t)
        {
            install();
            g_active = true;
            // PACKED tag, NLOOP 1, EOP, one register (A+D), then an XYZ2 write (a vertex kick).
            const uint32_t draws[] = {0x00008001u, 0x10000000u, 0x0000000eu, 0u, 0x00100010u, 0u, 0x00000005u, 0u};
            // IMAGE tag (FLG 2), NLOOP 1, EOP, then its quadword.
            const uint32_t image[] = {0x00008001u, 0x08000000u, 0u, 0u, 1u, 2u, 3u, 4u};
            std::vector<uint32_t> stream = {0, 0, 0, vif(0x50, 0, 2)};
            stream.insert(stream.end(), std::begin(draws), std::end(draws));
            stream.insert(stream.end(), {0, 0, 0, vif(0x50, 0, 2)});
            stream.insert(stream.end(), std::begin(image), std::end(image));
            send(stream);
            t.IsTrue(ps2x::native::directSkipped(2), "a vertex kick: drawn natively");
            t.IsFalse(ps2x::native::directSkipped(2), "an image: the GS gets it");
        });

        tc.Run("an image continued in the next DIRECT is not read as GIF tags", [](TestCase &t)
        {
            install();
            g_active = true;
            // DIRECT 1: an IMAGE tag of 2 quadwords and no data; DIRECT 2: its pixels, which read as a PACKED A+D
            // vertex kick; then a real vertex kick.
            const uint32_t imageTag[] = {0x00008002u, 0x08000000u, 0u, 0u};
            const uint32_t pixels[] = {0x00008001u, 0x10000000u, 0x0000000eu, 0u, 0x00100010u, 0u, 0x00000005u, 0u};
            std::vector<uint32_t> stream = {0, 0, 0, vif(0x50, 0, 1)};
            stream.insert(stream.end(), std::begin(imageTag), std::end(imageTag));
            stream.insert(stream.end(), {0, 0, 0, vif(0x50, 0, 2)});
            stream.insert(stream.end(), std::begin(pixels), std::end(pixels));
            stream.insert(stream.end(), {0, 0, 0, vif(0x50, 0, 2)});
            stream.insert(stream.end(), std::begin(pixels), std::end(pixels));
            send(stream);
            // The VIF1 interpreter asks about the tag's DIRECT and the last one only (the pixels go to the image).
            t.IsFalse(ps2x::native::directSkipped(1), "the image tag: the GS gets it");
            t.IsTrue(ps2x::native::directSkipped(2), "the vertex kick after the image: drawn natively");
        });
    });
}
