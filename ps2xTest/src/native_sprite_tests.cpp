#include "MiniTest.h"
#include "runtime/gs/gs_native_backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace
{
    using Span = GSNativeBackend::SpriteSpan;

    constexpr float kVertexShift = 31.0f / 64.0f;

    struct Strip
    {
        float p0, p1, t0, t1;
    };

    // Soulcalibur III's post copy: 64-pixel strips ending at x + 63.5, texel coordinates half a texel in.
    std::vector<Strip> copyStrips(uint32_t count, float width, float overhang)
    {
        std::vector<Strip> strips;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float x = static_cast<float>(i) * width;
            strips.push_back({x, x + width - overhang, x + 0.5f, x + width});
        }
        return strips;
    }

    struct Coverage
    {
        uint32_t hits = 0;
        float texel = 0.0f;
    };

    // What the vertex shader and the fragment shader's sample shift make of scaled pixel h along one axis, for a
    // source target at the same scale: the strips that cover it and the source texel its bilinear lookup centres on.
    Coverage cover(const std::vector<Span> &spans, uint32_t scale, uint32_t h)
    {
        Coverage c;
        const float s = static_cast<float>(scale);
        const float g = (static_cast<float>(h) + 0.5f) / s - kVertexShift;
        for (const Span &span : spans)
        {
            const float low = std::fmin(span.p0, span.p1), high = std::fmax(span.p0, span.p1);
            if (g < low || g >= high)
                continue;
            const float slope = (span.t1 - span.t0) / (span.p1 - span.p0);
            const float tc = span.t0 + (g - span.p0) * slope - slope / 64.0f;
            c.texel = tc * s - 0.5f;
            ++c.hits;
        }
        return c;
    }

    std::vector<Span> spans(const std::vector<Strip> &strips, uint32_t scale, uint32_t &empty)
    {
        std::vector<Span> out;
        empty = 0;
        for (const Strip &strip : strips)
        {
            Span span;
            if (GSNativeBackend::spriteSpan(strip.p0, strip.p1, strip.t0, strip.t1, scale, span))
                out.push_back(span);
            else
                ++empty;
        }
        return out;
    }

    void expectCopy(TestCase &t, const std::vector<Strip> &strips, uint32_t scale, uint32_t pixels)
    {
        uint32_t empty = 0;
        const std::vector<Span> drawn = spans(strips, scale, empty);
        t.Equals(empty, 0u, "every strip covers GS pixels");
        uint32_t holes = 0, overlaps = 0, offTexel = 0;
        for (uint32_t h = 0; h < pixels * scale; ++h)
        {
            const Coverage c = cover(drawn, scale, h);
            if (c.hits == 0u)
                ++holes;
            else if (c.hits > 1u)
                ++overlaps;
            else if (std::fabs(c.texel - static_cast<float>(h)) > 1.0f / 256.0f)
                ++offTexel;
        }
        const std::string at = " at " + std::to_string(scale) + "x";
        t.Equals(holes, 0u, "no scaled pixel is left to no strip" + at);
        t.Equals(overlaps, 0u, "no scaled pixel is drawn by two strips" + at);
        t.Equals(offTexel, 0u, "each scaled pixel reads its own source texel" + at);
    }
}

void register_native_sprite_tests()
{
    MiniTest::Case("NativeSprite", [](TestCase &tc)
    {
        tc.Run("a 64x16 bilinear strip copy ending at x + 63.5 fills every column at each scale", [](TestCase &t)
        {
            for (uint32_t scale = 1u; scale <= 4u; ++scale)
                expectCopy(t, copyStrips(10u, 64.0f, 0.5f), scale, 640u);
        });

        tc.Run("its 16-row strips fill every row at each scale", [](TestCase &t)
        {
            for (uint32_t scale = 1u; scale <= 4u; ++scale)
            {
                std::vector<Strip> rows;
                for (uint32_t i = 0; i < 28u; ++i)
                {
                    const float y = static_cast<float>(i) * 16.0f;
                    rows.push_back({y, y + 16.0f, y + 0.5f, y + 16.5f});
                }
                expectCopy(t, rows, scale, 448u);
            }
        });

        tc.Run("the second strip's first scaled column at 2x is drawn from the second strip", [](TestCase &t)
        {
            uint32_t empty = 0;
            const std::vector<Span> drawn = spans(copyStrips(2u, 64.0f, 0.5f), 2u, empty);
            t.Equals(static_cast<uint32_t>(drawn.size()), 2u, "both strips are drawn");
            const Coverage first = cover({drawn[1]}, 2u, 128u);
            t.Equals(first.hits, 1u, "host column 128 belongs to the strip starting at GS pixel 64");
            t.IsTrue(std::fabs(first.texel - 128.0f) <= 1.0f / 256.0f, "and samples host texel 128");
            t.Equals(cover({drawn[0]}, 2u, 127u).hits, 1u, "host column 127 is the first strip's last");
        });

        tc.Run("1x keeps the corners as given", [](TestCase &t)
        {
            Span span;
            t.IsTrue(GSNativeBackend::spriteSpan(64.0f, 127.5f, 64.5f, 128.0f, 1u, span), "drawn");
            t.IsTrue(span.p0 == 64.0f && span.p1 == 127.5f && span.t0 == 64.5f && span.t1 == 128.0f, "unchanged");
        });

        tc.Run("a sprite given right to left keeps its texel coordinates at its corners", [](TestCase &t)
        {
            Span span;
            t.IsTrue(GSNativeBackend::spriteSpan(127.5f, 64.0f, 128.0f, 64.5f, 2u, span), "drawn");
            t.IsTrue(span.p0 == 127.5f && span.p1 == 63.5f, "edges on the GS pixel boundaries of 64..127");
            t.IsTrue(span.t0 == 128.0f && span.t1 == 64.0f, "texel coordinates on the same line");
        });

        tc.Run("a sprite between two GS pixels draws nothing above 1x", [](TestCase &t)
        {
            Span span;
            t.IsFalse(GSNativeBackend::spriteSpan(63.5f, 64.0f, 0.0f, 1.0f, 2u, span), "covers no GS pixel");
            t.IsFalse(GSNativeBackend::spriteSpan(10.0f, 10.0f, 0.0f, 1.0f, 2u, span), "empty");
        });
    });
}
