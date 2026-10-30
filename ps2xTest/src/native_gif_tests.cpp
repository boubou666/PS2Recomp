#include "MiniTest.h"
#include "runtime/native/native_gif.h"
#include "runtime/native/native_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using ps2x::native::GifToNative;
using ps2x::native::Material;
using ps2x::native::Renderer;
using ps2x::native::Vertex;

namespace
{
    std::vector<Vertex> g_taken;

    void take(void *, const Material &, const float[16], float, const Vertex *vertices, uint32_t count)
    {
        g_taken.insert(g_taken.end(), vertices, vertices + count);
    }

    uint32_t bits(float f)
    {
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    }

    void quad(std::vector<uint32_t> &packet, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
    {
        packet.insert(packet.end(), {a, b, c, d});
    }

    // A textured sprite with perspective coordinates: PRIM, then per vertex RGBAQ, ST (with Q) and XYZ2.
    std::vector<uint32_t> stqSprite(float s0, float t0, float q0, float s1, float t1, float q1)
    {
        std::vector<uint32_t> packet;
        const uint64_t tag = 1u | (1ull << 15) | (7ull << 60);
        const uint64_t regs = 0x0ull | (0x1ull << 4) | (0x2ull << 8) | (0x5ull << 12) | (0x1ull << 16) | (0x2ull << 20) | (0x5ull << 24);
        quad(packet, static_cast<uint32_t>(tag), static_cast<uint32_t>(tag >> 32), static_cast<uint32_t>(regs),
             static_cast<uint32_t>(regs >> 32));
        quad(packet, 0x6u | 0x10u, 0, 0, 0);
        quad(packet, 0x80, 0x80, 0x80, 0x80);
        quad(packet, bits(s0), bits(t0), bits(q0), 0);
        quad(packet, 100u * 16u, 50u * 16u, 0, 0);
        quad(packet, 0x80, 0x80, 0x80, 0x80);
        quad(packet, bits(s1), bits(t1), bits(q1), 0);
        quad(packet, 200u * 16u, 150u * 16u, 0, 0);
        return packet;
    }

    std::vector<uint32_t> testWrite(uint64_t test)
    {
        std::vector<uint32_t> packet;
        const uint64_t tag = 1u | (1ull << 60);
        quad(packet, static_cast<uint32_t>(tag), static_cast<uint32_t>(tag >> 32), 0xeu, 0);
        quad(packet, static_cast<uint32_t>(test), static_cast<uint32_t>(test >> 32), 0x47u, 0);
        return packet;
    }

    std::vector<uint32_t> regionClamp()
    {
        std::vector<uint32_t> packet;
        const uint64_t tag = 2u | (1ull << 60);
        const uint64_t tex0 = (8ull << 26) | (8ull << 30);
        const uint64_t clamp = 2u | (2u << 2) | (16ull << 4) | (207ull << 14) | (32ull << 24) | (95ull << 34);
        quad(packet, static_cast<uint32_t>(tag), static_cast<uint32_t>(tag >> 32), 0xeeu, 0);
        quad(packet, static_cast<uint32_t>(tex0), static_cast<uint32_t>(tex0 >> 32), 0x06u, 0);
        quad(packet, static_cast<uint32_t>(clamp), static_cast<uint32_t>(clamp >> 32), 0x08u, 0);
        return packet;
    }

    std::vector<Vertex> drawSprite(bool gsWrites, const std::vector<uint32_t> &before = {})
    {
        Renderer &renderer = Renderer::instance();
        renderer.setStreamDraws(true);
        renderer.takeGsContextBatches(~0ull, &take, nullptr);
        g_taken.clear();
        renderer.beginGroup(1);
        GifToNative gif;
        gif.setGsWrites(gsWrites);
        std::vector<uint32_t> packet = before;
        const std::vector<uint32_t> sprite = stqSprite(0.5f, 0.25f, 2.0f, 1.0f, 1.0f, 4.0f);
        packet.insert(packet.end(), sprite.begin(), sprite.end());
        gif.packet(reinterpret_cast<const uint8_t *>(packet.data()), packet.size() * 4u);
        gif.flush();
        renderer.takeGsContextBatches(~0ull, &take, nullptr);
        renderer.setStreamDraws(false);
        return g_taken;
    }

    bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }
}

void register_native_gif_tests()
{
    MiniTest::Case("NativeGif", [](TestCase &tc)
    {
        tc.Run("a sprite's four corners take the second vertex's Q, as the GS does", [](TestCase &t)
        {
            const std::vector<Vertex> vertices = drawSprite(true);
            t.Equals(vertices.size(), size_t{6}, "two triangles");
            float left = vertices.empty() ? 0.0f : vertices[0].x;
            for (const Vertex &v : vertices)
                left = std::min(left, v.x);
            bool sameQ = true, uv = true;
            for (const Vertex &v : vertices)
            {
                sameQ = sameQ && near(v.q, 4.0f);
                const bool first = near(v.x, left);
                uv = uv && near(v.u * v.q, first ? 0.5f : 1.0f) && (near(v.v * v.q, 0.25f) || near(v.v * v.q, 1.0f));
            }
            t.IsTrue(sameQ, "Q of the second vertex on every corner");
            t.IsTrue(uv, "S and T over that Q");
        });
        tc.Run("a REGION_CLAMP the GS applies leaves the texture coordinates over the whole texture", [](TestCase &t)
        {
            const std::vector<Vertex> vertices = drawSprite(true, regionClamp());
            t.Equals(vertices.size(), size_t{6}, "two triangles");
            float left = vertices.empty() ? 0.0f : vertices[0].x;
            for (const Vertex &v : vertices)
                left = std::min(left, v.x);
            bool uv = true;
            for (const Vertex &v : vertices)
            {
                const bool first = near(v.x, left);
                uv = uv && near(v.u * v.q, first ? 0.5f : 1.0f) && (near(v.v * v.q, 0.25f) || near(v.v * v.q, 1.0f));
            }
            t.IsTrue(uv, "S and T as the packet sent them");
        });
        tc.Run("an alpha test that never passes still draws what its AFAIL writes", [](TestCase &t)
        {
            const uint64_t never = 1u | (1ull << 16) | (2ull << 17);
            t.Equals(drawSprite(true, testWrite(never | (3ull << 12))).size(), size_t{6}, "RGB_ONLY drawn");
            t.Equals(drawSprite(true, testWrite(never | (1ull << 12))).size(), size_t{6}, "FB_ONLY drawn");
            t.Equals(drawSprite(true, testWrite(never | (2ull << 12))).size(), size_t{6}, "ZB_ONLY drawn by the GS");
            t.Equals(drawSprite(true, testWrite(never)).size(), size_t{0}, "KEEP skipped");
        });
    });
}
