#include "MiniTest.h"
#include "runtime/native/native_texture_cache.h"

#include <cstdint>
#include <vector>

using ps2x::native::TextureCache;

namespace
{
    uint64_t bitbltbuf(uint32_t dbp, uint32_t dbw, uint32_t dpsm)
    {
        return (uint64_t{dbp} << 32) | (uint64_t{dbw} << 48) | (uint64_t{dpsm} << 56);
    }

    uint64_t trxreg(uint32_t w, uint32_t h) { return uint64_t{w} | (uint64_t{h} << 32); }

    uint64_t tex0(uint32_t tbp, uint32_t tbw, uint32_t psm, uint32_t tw, uint32_t th, uint32_t cbp = 0, uint32_t csa = 0)
    {
        return uint64_t{tbp} | (uint64_t{tbw} << 14) | (uint64_t{psm} << 20) | (uint64_t{tw} << 26) |
               (uint64_t{th} << 30) | (uint64_t{1} << 34) | (uint64_t{cbp} << 37) | (uint64_t{csa} << 56);
    }

    // A 256-colour CT32 palette in CSM1 order, entry i = colour(i), uploaded as the 16 x 16 image the game sends.
    uint32_t colour(uint32_t i) { return i | ((255u - i) << 8) | ((i * 7u & 0xffu) << 16) | (0x40u << 24); }

    std::vector<uint8_t> csm1Palette()
    {
        std::vector<uint8_t> image(16 * 16 * 4);
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t x = 0, y = 0;
            ps2x::native::csm1Position(i, x, y);
            const uint32_t c = colour(i);
            for (int b = 0; b < 4; ++b)
                image[(y * 16 + x) * 4 + b] = static_cast<uint8_t>(c >> (8 * b));
        }
        return image;
    }

    uint32_t opaque(uint32_t c) { return (c & 0x00ffffffu) | ((c >> 24) * 2u << 24); } // GS 0x80 alpha -> 0xff scale
}

void register_native_texture_cache_tests()
{
    MiniTest::Case("NativeTextureCache", [](TestCase &tc)
    {
        tc.Run("CT32 texture decodes as uploaded, alpha scaled from 0x80 to 0xff", [](TestCase &t)
        {
            TextureCache cache;
            std::vector<uint8_t> pixels(8 * 8 * 4);
            for (size_t i = 0; i < pixels.size(); ++i)
                pixels[i] = static_cast<uint8_t>(i * 13u);
            for (size_t i = 3; i < pixels.size(); i += 4)
                pixels[i] = 0x80;
            cache.upload(bitbltbuf(0, 1, 0x00), 0, trxreg(8, 8), pixels.data(), pixels.size());
            const TextureCache::Texture &texture = *cache.texture(tex0(0, 1, 0x00, 3, 3));
            t.Equals(texture.width, 8u, "width 2^TW");
            bool same = true;
            for (size_t i = 0; i < 64; ++i)
            {
                const uint32_t c = pixels[i * 4] | (pixels[i * 4 + 1] << 8) | (pixels[i * 4 + 2] << 16) | (0xffu << 24);
                same &= texture.rgba[i] == c;
            }
            t.IsTrue(same, "every texel equals the uploaded pixel, opaque");
        });

        tc.Run("T8 texture looks its indices up in a CSM1 CT32 palette", [](TestCase &t)
        {
            TextureCache cache;
            std::vector<uint8_t> indices(16 * 16);
            for (uint32_t i = 0; i < 256; ++i)
                indices[i] = static_cast<uint8_t>(i);
            const std::vector<uint8_t> palette = csm1Palette();
            cache.upload(bitbltbuf(64, 1, 0x13), 0, trxreg(16, 16), indices.data(), indices.size());
            cache.upload(bitbltbuf(200, 1, 0x00), 0, trxreg(16, 16), palette.data(), palette.size());
            const TextureCache::Texture &texture = *cache.texture(tex0(64, 1, 0x13, 4, 4, 200));
            bool same = true;
            for (uint32_t i = 0; i < 256; ++i)
                same &= texture.rgba[i] == opaque(colour(i));
            t.IsTrue(same, "texel i shows palette entry i, wherever CSM1 stored it");
        });

        tc.Run("T4 texture uses the 16 palette entries CSA selects", [](TestCase &t)
        {
            TextureCache cache;
            std::vector<uint8_t> indices(8 * 2 / 2); // 16 texels, two per byte, low nibble first
            for (uint32_t i = 0; i < 16; ++i)
                indices[i / 2] |= static_cast<uint8_t>(i << ((i & 1u) * 4u));
            const std::vector<uint8_t> palette = csm1Palette();
            cache.upload(bitbltbuf(128, 1, 0x14), 0, trxreg(16, 1), indices.data(), indices.size());
            cache.upload(bitbltbuf(200, 1, 0x00), 0, trxreg(16, 16), palette.data(), palette.size());
            const TextureCache::Texture &texture = *cache.texture(tex0(128, 1, 0x14, 4, 0, 200, 3));
            bool same = true;
            for (uint32_t i = 0; i < 16; ++i)
                same &= texture.rgba[i] == opaque(colour(48u + i));
            t.IsTrue(same, "texel i shows entry CSA * 16 + i");
        });

        tc.Run("an upload over a cached texture's pages decodes it again", [](TestCase &t)
        {
            TextureCache cache;
            std::vector<uint8_t> pixels(8 * 8 * 4, 0x10);
            cache.upload(bitbltbuf(0, 1, 0x00), 0, trxreg(8, 8), pixels.data(), pixels.size());
            const uint64_t tex = tex0(0, 1, 0x00, 3, 3);
            t.Equals(cache.texture(tex)->rgba[0] & 0xffu, 0x10u, "first upload");
            std::vector<uint8_t> again(8 * 8 * 4, 0x22);
            cache.upload(bitbltbuf(0, 1, 0x00), 0, trxreg(8, 8), again.data(), again.size());
            t.Equals(cache.texture(tex)->rgba[0] & 0xffu, 0x22u, "the second upload replaces the cached image");
        });
    });
}
