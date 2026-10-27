#include "MiniTest.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "runtime/ps2_memory.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
    constexpr uint32_t kZbufPage = 64u;
    constexpr uint32_t kTextureBlock = 64u;
    constexpr uint32_t kMipBlock = 128u;

    uint64_t vertex(uint32_t x16, uint32_t y16, uint32_t z = 0u)
    {
        return static_cast<uint64_t>(x16) | (static_cast<uint64_t>(y16) << 16) | (static_cast<uint64_t>(z) << 32);
    }

    uint64_t rgbaq(uint32_t r, uint32_t g, uint32_t b, uint32_t a, float q = 1.0f)
    {
        uint32_t qBits = 0u;
        std::memcpy(&qBits, &q, sizeof(qBits));
        return static_cast<uint64_t>(r) | (static_cast<uint64_t>(g) << 8) | (static_cast<uint64_t>(b) << 16) |
               (static_cast<uint64_t>(a) << 24) | (static_cast<uint64_t>(qBits) << 32);
    }

    uint64_t st(float s, float t)
    {
        uint32_t sBits = 0u, tBits = 0u;
        std::memcpy(&sBits, &s, sizeof(sBits));
        std::memcpy(&tBits, &t, sizeof(tBits));
        return static_cast<uint64_t>(sBits) | (static_cast<uint64_t>(tBits) << 32);
    }

    uint64_t uv(uint32_t u16, uint32_t v16)
    {
        return static_cast<uint64_t>(u16) | (static_cast<uint64_t>(v16) << 16);
    }

    uint64_t alpha(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t fix)
    {
        return static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << 2) | (static_cast<uint64_t>(c) << 4) |
               (static_cast<uint64_t>(d) << 6) | (static_cast<uint64_t>(fix) << 32);
    }

    // A 64x64 frame at page 0 (FBW 1) in the given format, Z at page 64, everything drawn, COLCLAMP set as games
    // leave it, PRIM supplying the primitive attributes.
    struct Target
    {
        std::vector<uint8_t> vram;
        GS gs;

        explicit Target(uint32_t framePsm = GS_PSM_CT32, uint32_t zPsm = 0u, bool zMask = true)
            : vram(PS2_GS_VRAM_SIZE, 0u)
        {
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);
            gs.writeRegister(GS_REG_FRAME_1, (1ull << 16) | (static_cast<uint64_t>(framePsm) << 24));
            gs.writeRegister(GS_REG_ZBUF_1, kZbufPage | (static_cast<uint64_t>(zPsm & 0xFu) << 24) | (zMask ? (1ull << 32) : 0ull));
            gs.writeRegister(GS_REG_SCISSOR_1, (63ull << 16) | (63ull << 48));
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_COLCLAMP, 1ull);
            gs.writeRegister(GS_REG_PRMODECONT, 1ull);
        }

        uint32_t frame32(uint32_t x, uint32_t y) const
        {
            return gs.ReadVram(GS_PSM_CT32, 0u, 1u, x, y);
        }

        uint32_t frame16(uint32_t x, uint32_t y) const
        {
            return gs.ReadVram(GS_PSM_CT16, 0u, 1u, x, y) & 0xFFFFu;
        }

        uint32_t z(uint32_t psm, uint32_t x, uint32_t y) const
        {
            return gs.ReadVram(psm, kZbufPage * 32u, 1u, x, y);
        }

        void sprite(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t z = 0u)
        {
            gs.writeRegister(GS_REG_XYZ2, vertex(x0, y0, z));
            gs.writeRegister(GS_REG_XYZ2, vertex(x1, y1, z));
        }

        // A CT32 texture at kTextureBlock (TBW 1), sampled with DECAL and its own alpha.
        void texture(uint32_t tw, uint32_t th, uint32_t tbp = kTextureBlock)
        {
            gs.writeRegister(GS_REG_TEX0_1, static_cast<uint64_t>(tbp) | (1ull << 14) | (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                                                (static_cast<uint64_t>(tw) << 26) | (static_cast<uint64_t>(th) << 30) | (1ull << 34) |
                                                (1ull << 35));
            gs.writeRegister(GS_REG_TEXFLUSH, 0ull);
        }
    };

    uint32_t ct16(uint32_t channel5, bool alpha)
    {
        return channel5 | (channel5 << 5) | (channel5 << 10) | (alpha ? 0x8000u : 0u);
    }
}

void register_ps2_gs_reference_tests()
{
    MiniTest::Case("PS2GSReference", [](TestCase &tc)
    {
        tc.Run("fog is one lerp toward FOGCOL: (F * C + (256 - F) * FOGCOL) >> 8", [](TestCase &t)
        {
            auto fogged = [](uint32_t f, uint32_t colour, uint64_t fogcol)
            {
                Target target;
                target.gs.writeRegister(GS_REG_FOGCOL, fogcol);
                target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT) | (1ull << 5));
                target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(colour, colour, colour, 0x80u));
                target.gs.writeRegister(GS_REG_FOG, static_cast<uint64_t>(f) << 56);
                target.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
                return target.frame32(0u, 0u);
            };
            t.Equals(fogged(255u, 128u, 0x808080ull), 0x80808080u, "F=255 keeps C=128 over FOGCOL=128 at 128");
            t.Equals(fogged(128u, 200u, 0x646464ull), 0x80969696u, "F=128: (128 * 200 + 128 * 100) >> 8 = 150");
            t.Equals(fogged(0u, 255u, 0x302010ull), 0x80302010u, "F=0 gives FOGCOL exactly");
            t.Equals(fogged(1u, 0u, 0xFFFFFFull), 0x80FEFEFEu, "F=1 over C=0: (255 * 255) >> 8 = 254");
        });

        tc.Run("COLCLAMP=1 clamps the blended colour, COLCLAMP=0 keeps its low 8 bits", [](TestCase &t)
        {
            auto blended = [](uint64_t colclamp, uint64_t alphaReg)
            {
                Target target;
                target.gs.WriteVram(GS_PSM_CT32, 0u, 1u, 0u, 0u, 0x80646464u);
                target.gs.writeRegister(GS_REG_COLCLAMP, colclamp);
                target.gs.writeRegister(GS_REG_ALPHA_1, alphaReg);
                target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE) | (1ull << 6));
                target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(200u, 200u, 200u, 0x80u));
                target.sprite(0u, 0u, 16u, 16u);
                return target.frame32(0u, 0u);
            };
            const uint64_t add = alpha(0u, 2u, 2u, 1u, 0x80u);      // Cs * 128 >> 7 + Cd = 300
            const uint64_t subtract = alpha(2u, 0u, 2u, 1u, 0x80u); // -Cs * 128 >> 7 + Cd = -100
            t.Equals(blended(1ull, add), 0x80FFFFFFu, "300 clamps to 255");
            t.Equals(blended(0ull, add), 0x802C2C2Cu, "300 wraps to 44");
            t.Equals(blended(1ull, subtract), 0x80000000u, "-100 clamps to 0");
            t.Equals(blended(0ull, subtract), 0x809C9C9Cu, "-100 wraps to 156");
        });

        tc.Run("DTHE adds the DIMX entry of the pixel's window position to colours written to 16-bit frames", [](TestCase &t)
        {
            // DIMX rows: (-4, -1, 0, 1), (2, 3, -2, -3), (0, 1, 2, 3), (-1, -2, -3, -4) as 3-bit two's complement.
            const int dimx[4][4] = {{-4, -1, 0, 1}, {2, 3, -2, -3}, {0, 1, 2, 3}, {-1, -2, -3, -4}};
            uint64_t dimxReg = 0u;
            for (int row = 0; row < 4; ++row)
                for (int column = 0; column < 4; ++column)
                    dimxReg |= static_cast<uint64_t>(dimx[row][column] & 7) << (row * 16 + column * 4);

            auto draw = [&](uint32_t framePsm, uint64_t dthe, uint64_t colclamp, uint32_t colour)
            {
                auto target = std::make_unique<Target>(framePsm);
                target->gs.writeRegister(GS_REG_DIMX, dimxReg);
                target->gs.writeRegister(GS_REG_DTHE, dthe);
                target->gs.writeRegister(GS_REG_COLCLAMP, colclamp);
                target->gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE));
                target->gs.writeRegister(GS_REG_RGBAQ, rgbaq(colour, colour, colour, 0x80u));
                target->sprite(0u, 0u, 128u, 128u);
                return target;
            };

            const auto dithered = draw(GS_PSM_CT16, 1ull, 1ull, 71u);
            bool matches = true;
            for (uint32_t y = 0; y < 8u; ++y)
                for (uint32_t x = 0; x < 8u; ++x)
                    matches = matches && dithered->frame16(x, y) == ct16(static_cast<uint32_t>(71 + dimx[y & 3u][x & 3u]) >> 3, true);
            t.IsTrue(matches, "71 + DIMX[y & 3][x & 3], truncated to 5 bits, in every pixel of an 8x8 sprite");
            t.Equals(dithered->frame16(1u, 0u), ct16(8u, true), "71 - 1 = 70 stores 8");
            t.Equals(dithered->frame16(3u, 0u), ct16(9u, true), "71 + 1 = 72 stores 9");

            t.Equals(draw(GS_PSM_CT16, 0ull, 1ull, 71u)->frame16(3u, 0u), ct16(8u, true), "DTHE=0 does not dither");
            t.Equals(draw(GS_PSM_CT32, 1ull, 1ull, 71u)->frame32(3u, 0u), 0x80474747u, "32-bit frames are not dithered");
            t.Equals(draw(GS_PSM_CT16, 1ull, 1ull, 254u)->frame16(1u, 1u), ct16(31u, true), "254 + 3 clamps to 255 with COLCLAMP=1");
            t.Equals(draw(GS_PSM_CT16, 1ull, 0ull, 254u)->frame16(1u, 1u), ct16(0u, true), "254 + 3 wraps to 1 with COLCLAMP=0");
            t.Equals(draw(GS_PSM_CT16, 1ull, 0ull, 2u)->frame16(0u, 0u), ct16(31u, true), "2 - 4 wraps to 254 with COLCLAMP=0");
        });

        tc.Run("Z past the Z buffer format's range saturates before the depth test and the write", [](TestCase &t)
        {
            {
                Target target(GS_PSM_CT32, 1u, false); // Z24
                target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
                target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(1u, 2u, 3u, 0x80u));
                target.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u, 0x01000005u));
                t.Equals(target.z(GS_PSM_Z24, 0u, 0u) & 0xFFFFFFu, 0xFFFFFFu, "Z24 stores 0x1000005 as 0xFFFFFF");

                target.gs.writeRegister(GS_REG_TEST_1, 0x10000ull | (3ull << 17)); // GREATER
                target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(9u, 9u, 9u, 0x80u));
                target.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u, 0x02000000u));
                t.Equals(target.frame32(0u, 0u), 0x80030201u, "0x2000000 saturates to the stored 0xFFFFFF: not GREATER");

                target.gs.writeRegister(GS_REG_TEST_1, 0x10000ull | (2ull << 17)); // GEQUAL
                target.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u, 0x02000000u));
                t.Equals(target.frame32(0u, 0u), 0x80090909u, "0x2000000 saturates to the stored 0xFFFFFF: GEQUAL");
            }
            {
                Target target(GS_PSM_CT32, 2u, false); // Z16
                target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE));
                target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(1u, 2u, 3u, 0x80u));
                target.sprite(0u, 0u, 16u, 16u, 0x12345u);
                t.Equals(target.z(GS_PSM_Z16, 0u, 0u) & 0xFFFFu, 0xFFFFu, "Z16 stores 0x12345 as 0xFFFF");
            }
        });

        tc.Run("triangles sharing an edge cover each whole-pixel sample once (top-left rule)", [](TestCase &t)
        {
            Target target;
            target.gs.writeRegister(GS_REG_ALPHA_1, alpha(0u, 2u, 2u, 1u, 0x80u)); // Cs + Cd
            target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_TRIANGLE) | (1ull << 6));
            target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(1u, 0u, 0u, 0x80u));
            const uint64_t corners[4] = {vertex(0u, 0u), vertex(64u, 0u), vertex(64u, 64u), vertex(0u, 64u)};
            for (const int index : {0, 1, 2, 0, 2, 3})
                target.gs.writeRegister(GS_REG_XYZ2, corners[index]);
            bool once = true;
            for (uint32_t y = 0; y <= 5u; ++y)
                for (uint32_t x = 0; x <= 5u; ++x)
                    once = once && (target.frame32(x, y) & 0xFFu) == ((x < 4u && y < 4u) ? 1u : 0u);
            t.IsTrue(once, "pixels 0..3 on both axes drawn exactly once, the right and bottom edges left out");
        });

        tc.Run("triangles sample coverage and attributes at whole pixel positions", [](TestCase &t)
        {
            Target target;
            target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_TRIANGLE));
            target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(255u, 255u, 255u, 0x80u));
            target.gs.writeRegister(GS_REG_XYZ2, vertex(8u, 0u));
            target.gs.writeRegister(GS_REG_XYZ2, vertex(128u, 0u));
            target.gs.writeRegister(GS_REG_XYZ2, vertex(8u, 128u));
            t.Equals(target.frame32(0u, 1u), 0u, "x = 0 lies left of the edge at 0.5");
            t.Equals(target.frame32(1u, 1u), 0x80FFFFFFu, "x = 1 lies inside");
            t.Equals(target.frame32(1u, 0u), 0x80FFFFFFu, "the top edge at y = 0 is drawn");

            Target gouraud;
            gouraud.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_TRIANGLE) | (1ull << 3));
            gouraud.gs.writeRegister(GS_REG_RGBAQ, rgbaq(0u, 0u, 0u, 0x80u));
            gouraud.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
            gouraud.gs.writeRegister(GS_REG_RGBAQ, rgbaq(128u, 100u, 0u, 0x80u));
            gouraud.gs.writeRegister(GS_REG_XYZ2, vertex(128u, 0u));
            gouraud.gs.writeRegister(GS_REG_RGBAQ, rgbaq(0u, 0u, 0u, 0x80u));
            gouraud.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 128u));
            t.Equals(gouraud.frame32(3u, 1u), 0x80002530u, "R = 16 * 3 = 48, G = 12.5 * 3 = 37.5 truncated to 37");
            t.Equals(gouraud.frame32(5u, 2u), 0x80003E50u, "R = 80, G = 62.5 truncated to 62");
        });

        tc.Run("sprites cover [x0, x1) and take texture coordinates at whole pixel positions", [](TestCase &t)
        {
            Target target;
            target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE));
            target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(255u, 255u, 255u, 0x80u));
            target.sprite(8u, 0u, 40u, 16u);
            t.Equals(target.frame32(0u, 0u), 0u, "0 < 0.5");
            t.Equals(target.frame32(1u, 0u), 0x80FFFFFFu, "0.5 <= 1 < 2.5");
            t.Equals(target.frame32(2u, 0u), 0x80FFFFFFu, "0.5 <= 2 < 2.5");
            t.Equals(target.frame32(3u, 0u), 0u, "3 >= 2.5");
            target.sprite(160u, 0u, 160u, 64u);
            t.Equals(target.frame32(10u, 1u), 0u, "a zero-width sprite draws nothing");

            Target textured;
            for (uint32_t x = 0; x < 4u; ++x)
                textured.gs.WriteVram(GS_PSM_CT32, kTextureBlock, 1u, x, 0u, (x & 1u) ? 0x80FFFFFFu : 0x80000000u);
            textured.texture(2u, 0u);
            textured.gs.writeRegister(GS_REG_TEX1_1, 1ull << 5);           // MMAG linear
            textured.gs.writeRegister(GS_REG_CLAMP_1, 1ull | (1ull << 2)); // CLAMP
            textured.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE) | (1ull << 4) | (1ull << 8));
            textured.gs.writeRegister(GS_REG_RGBAQ, rgbaq(128u, 128u, 128u, 0x80u));
            textured.gs.writeRegister(GS_REG_UV, uv(0u, 0u));
            textured.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
            textured.gs.writeRegister(GS_REG_UV, uv(64u, 16u));
            textured.gs.writeRegister(GS_REG_XYZ2, vertex(64u, 16u));
            t.Equals(textured.frame32(1u, 0u), 0x80808080u, "pixel 1 samples U = 1.0: half of texel 0, half of texel 1");

            textured.gs.writeRegister(GS_REG_TEX1_1, 0ull);
            textured.gs.writeRegister(GS_REG_UV, uv(0u, 0u));
            textured.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
            textured.gs.writeRegister(GS_REG_UV, uv(64u, 16u));
            textured.gs.writeRegister(GS_REG_XYZ2, vertex(64u, 16u));
            t.Equals(textured.frame32(1u, 0u), 0x80FFFFFFu, "nearest: pixel 1 samples texel 1");
        });

        tc.Run("points round their position halves up, lines step [first, last) along the major axis", [](TestCase &t)
        {
            Target target;
            target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(255u, 0u, 0u, 0x80u));
            target.gs.writeRegister(GS_REG_XYZ2, vertex(24u, 0u));
            target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(0u, 255u, 0u, 0x80u));
            target.gs.writeRegister(GS_REG_XYZ2, vertex(87u, 0u));
            t.Equals(target.frame32(2u, 0u), 0x800000FFu, "1.5 rounds to 2");
            t.Equals(target.frame32(1u, 0u), 0u, "1.5 does not round to 1");
            t.Equals(target.frame32(5u, 0u), 0x8000FF00u, "5.4375 rounds to 5");

            Target line;
            line.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_LINE));
            line.gs.writeRegister(GS_REG_RGBAQ, rgbaq(255u, 255u, 255u, 0x80u));
            line.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
            line.gs.writeRegister(GS_REG_XYZ2, vertex(64u, 32u));
            const bool expected = line.frame32(0u, 0u) != 0u && line.frame32(1u, 1u) != 0u && line.frame32(2u, 1u) != 0u &&
                                  line.frame32(3u, 2u) != 0u;
            t.IsTrue(expected, "(0,0) to (4,2): y = x / 2 rounded halves up at x = 0..3");
            t.Equals(line.frame32(1u, 0u), 0u, "y = 0.5 at x = 1 rounds up");
            t.Equals(line.frame32(4u, 2u), 0u, "the last point is not drawn");
        });

        tc.Run("the LOD chooses MMAG at or below 0 and MMIN above it", [](TestCase &t)
        {
            auto drawWithK = [](uint32_t k)
            {
                Target target;
                for (uint32_t x = 0; x < 4u; ++x)
                    target.gs.WriteVram(GS_PSM_CT32, kTextureBlock, 1u, x, 0u, (x & 1u) ? 0x80FFFFFFu : 0x80000000u);
                target.texture(2u, 0u);
                // LCM = 1, MMAG nearest, MMIN linear, K in sixteenths.
                target.gs.writeRegister(GS_REG_TEX1_1, 1ull | (1ull << 6) | (static_cast<uint64_t>(k & 0xFFFu) << 32));
                target.gs.writeRegister(GS_REG_CLAMP_1, 1ull | (1ull << 2));
                target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE) | (1ull << 4) | (1ull << 8));
                target.gs.writeRegister(GS_REG_UV, uv(0u, 0u));
                target.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
                target.gs.writeRegister(GS_REG_UV, uv(64u, 16u));
                target.gs.writeRegister(GS_REG_XYZ2, vertex(64u, 16u));
                return target.frame32(1u, 0u);
            };
            t.Equals(drawWithK(0u), 0x80FFFFFFu, "LOD 0 magnifies: nearest texel 1");
            t.Equals(drawWithK(0xFF0u), 0x80FFFFFFu, "LOD -1 magnifies: nearest texel 1");
            t.Equals(drawWithK(1u), 0x80808080u, "LOD 1/16 minifies: bilinear between texels 0 and 1");
        });

        tc.Run("mipmap levels come from MIPTBP1, nearest or blended by the LOD's fraction", [](TestCase &t)
        {
            auto draw = [](uint64_t mmin, uint32_t k, bool fromQ, float q)
            {
                Target target;
                for (uint32_t y = 0; y < 8u; ++y)
                    for (uint32_t x = 0; x < 8u; ++x)
                        target.gs.WriteVram(GS_PSM_CT32, kTextureBlock, 1u, x, y, 0x800000FFu);
                for (uint32_t y = 0; y < 4u; ++y)
                    for (uint32_t x = 0; x < 4u; ++x)
                        target.gs.WriteVram(GS_PSM_CT32, kMipBlock, 1u, x, y, 0x8000FF00u);
                target.texture(3u, 3u);
                target.gs.writeRegister(GS_REG_MIPTBP1_1, static_cast<uint64_t>(kMipBlock) | (1ull << 14));
                // MXL 1, MMAG nearest, LCM unless the LOD comes from Q.
                target.gs.writeRegister(GS_REG_TEX1_1, (fromQ ? 0ull : 1ull) | (1ull << 2) | (mmin << 6) |
                                                           (static_cast<uint64_t>(k & 0xFFFu) << 32));
                target.gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE) | (1ull << 4));
                target.gs.writeRegister(GS_REG_RGBAQ, rgbaq(128u, 128u, 128u, 0x80u, q));
                target.gs.writeRegister(GS_REG_ST, st(0.0f, 0.0f));
                target.gs.writeRegister(GS_REG_XYZ2, vertex(0u, 0u));
                target.gs.writeRegister(GS_REG_ST, st(0.25f * q, 0.25f * q));
                target.gs.writeRegister(GS_REG_XYZ2, vertex(32u, 32u));
                return target.frame32(1u, 1u);
            };
            constexpr uint32_t kRed = 0x800000FFu;
            constexpr uint32_t kGreen = 0x8000FF00u;
            t.Equals(draw(2u, 7u, false, 1.0f), kRed, "LOD 7/16 rounds to level 0");
            t.Equals(draw(2u, 8u, false, 1.0f), kGreen, "LOD 8/16 rounds to level 1");
            t.Equals(draw(2u, 40u, false, 1.0f), kGreen, "LOD 2.5 stops at MXL 1");
            t.Equals(draw(3u, 8u, false, 1.0f), 0x80007F7Fu, "LOD 0.5 blends: 255 + (-255 * 8 >> 4) = 127, 255 * 8 >> 4 = 127");
            t.Equals(draw(2u, 0u, true, 0.5f), kGreen, "Q = 0.5: LOD = log2(1 / Q) = 1");
            t.Equals(draw(2u, 0u, true, 1.0f), kRed, "Q = 1: LOD 0, level 0");
            t.Equals(draw(2u, 0u, true, 2.0f), kRed, "Q = 2: LOD -1, level 0");
        });
    });
}
