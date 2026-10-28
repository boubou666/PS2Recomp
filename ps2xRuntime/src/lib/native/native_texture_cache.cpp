#include "runtime/native/native_texture_cache.h"

#include "runtime/gs/ps2_gs_memory.h"

#include <algorithm>
#include <mutex>

namespace ps2x::native
{
    namespace
    {
        enum Psm : uint32_t
        {
            CT32 = 0x00,
            CT24 = 0x01,
            CT16 = 0x02,
            CT16S = 0x0a,
            T8 = 0x13,
            T4 = 0x14,
            T8H = 0x1b,
            T4HL = 0x24,
            T4HH = 0x2c,
        };

        uint32_t bitsPerPixel(uint32_t psm)
        {
            switch (psm)
            {
            case CT32: return 32;
            case CT24: return 24;
            case CT16:
            case CT16S: return 16;
            case T8: return 8;
            case T4: return 4;
            case T8H: return 8; // uploaded as 8-bit indices, stored in bits 24-31 of 32-bit words
            case T4HL:
            case T4HH: return 4;
            default: return 32;
            }
        }

        void write(uint8_t *memory, uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
        {
            switch (psm)
            {
            case CT32: GSMem::WriteCT32(memory, bp, bw, x, y, value); break;
            case CT24: GSMem::WriteCT24(memory, bp, bw, x, y, value); break;
            case CT16: GSMem::WriteCT16(memory, bp, bw, x, y, value); break;
            case CT16S: GSMem::WriteCT16S(memory, bp, bw, x, y, value); break;
            case T8: GSMem::WriteP8(memory, bp, bw, x, y, value); break;
            case T4: GSMem::WriteP4(memory, bp, bw, x, y, value); break;
            case T8H: GSMem::WriteP8H(memory, bp, bw, x, y, value); break;
            case T4HL: GSMem::WriteP4HL(memory, bp, bw, x, y, value); break;
            case T4HH: GSMem::WriteP4HH(memory, bp, bw, x, y, value); break;
            default: break;
            }
        }

        uint32_t read(uint8_t *memory, uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
        {
            switch (psm)
            {
            case CT32: return GSMem::ReadCT32(memory, bp, bw, x, y);
            case CT24: return GSMem::ReadCT24(memory, bp, bw, x, y);
            case CT16: return GSMem::ReadCT16(memory, bp, bw, x, y);
            case CT16S: return GSMem::ReadCT16S(memory, bp, bw, x, y);
            case T8: return GSMem::ReadP8(memory, bp, bw, x, y);
            case T4: return GSMem::ReadP4(memory, bp, bw, x, y);
            case T8H: return GSMem::ReadP8H(memory, bp, bw, x, y);
            case T4HL: return GSMem::ReadP4HL(memory, bp, bw, x, y);
            case T4HH: return GSMem::ReadP4HH(memory, bp, bw, x, y);
            default: return 0;
            }
        }

        // GS colours to RGBA8: alpha 0x80 is opaque, so it doubles (clamped).
        uint32_t fromCt32(uint32_t c)
        {
            const uint32_t a = std::min<uint32_t>(255u, (c >> 24) * 2u);
            return (c & 0x00ffffffu) | (a << 24);
        }

        uint32_t fromCt16(uint32_t c, uint64_t texa)
        {
            const uint32_t r = (c & 0x1fu) << 3, g = ((c >> 5) & 0x1fu) << 3, b = ((c >> 10) & 0x1fu) << 3;
            const uint32_t ta0 = texa & 0xffu, ta1 = (texa >> 32) & 0xffu;
            const bool aem = (texa >> 15) & 1u;
            uint32_t a = (c & 0x8000u) ? ta1 : ((aem && (c & 0x7fffu) == 0u) ? 0u : ta0);
            a = std::min<uint32_t>(255u, a * 2u);
            return r | (g << 8) | (b << 16) | (a << 24);
        }

        uint32_t fromCt24(uint32_t c, uint64_t texa)
        {
            const bool aem = (texa >> 15) & 1u;
            const uint32_t rgb = c & 0x00ffffffu;
            const uint32_t a = (aem && rgb == 0u) ? 0u : std::min<uint32_t>(255u, (texa & 0xffu) * 2u);
            return rgb | (a << 24);
        }

        bool indexed(uint32_t psm)
        {
            return psm == T8 || psm == T4 || psm == T8H || psm == T4HL || psm == T4HH;
        }

        void initTables()
        {
            static std::once_flag once;
            std::call_once(once, [] { GSMem::InitLookupTables(); });
        }
    }

    TextureCache::TextureCache()
        : m_memory(kMemoryBytes, 0u), m_pageGeneration(kMemoryBytes / kPageBytes, 0u)
    {
        initTables();
    }

    void TextureCache::forEachPage(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w,
                                   uint32_t h, std::vector<std::pair<uint32_t, uint32_t>> *record)
    {
        // Every 8-pixel-aligned run of a row lies in one 256-byte block, so 8 x 8 steps visit every page touched.
        auto visit = [&](uint32_t x, uint32_t y)
        {
            const uint32_t page = GSMem::PageOf(psm, bp, bw, x, y) % m_pageGeneration.size();
            if (record)
                record->emplace_back(page, m_pageGeneration[page]);
            else
                ++m_pageGeneration[page];
        };
        for (uint32_t y = y0; y < y0 + h; y += 8u)
            for (uint32_t x = x0; x < x0 + w; x += 8u)
                visit(x, y);
        visit(x0 + w - 1u, y0 + h - 1u);
    }

    void TextureCache::upload(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, const uint8_t *data, size_t bytes)
    {
        const uint32_t dbp = (bitbltbuf >> 32) & 0x3fffu;
        const uint32_t dbw = (bitbltbuf >> 48) & 0x3fu;
        const uint32_t dpsm = (bitbltbuf >> 56) & 0x3fu;
        const uint32_t dx = (trxpos >> 32) & 0x7ffu;
        const uint32_t dy = (trxpos >> 48) & 0x7ffu;
        const uint32_t w = trxreg & 0xfffu;
        const uint32_t h = (trxreg >> 32) & 0xfffu;
        if (w == 0u || h == 0u)
            return;
        const uint32_t bpp = bitsPerPixel(dpsm);
        const size_t pixels = std::min<size_t>(static_cast<size_t>(w) * h, bytes * 8u / bpp);
        for (size_t i = 0; i < pixels; ++i)
        {
            const uint32_t x = dx + static_cast<uint32_t>(i % w);
            const uint32_t y = dy + static_cast<uint32_t>(i / w);
            uint32_t value = 0u;
            switch (bpp)
            {
            case 32: value = static_cast<uint32_t>(data[i * 4]) | (data[i * 4 + 1] << 8) | (data[i * 4 + 2] << 16) |
                             (static_cast<uint32_t>(data[i * 4 + 3]) << 24); break;
            case 24: value = static_cast<uint32_t>(data[i * 3]) | (data[i * 3 + 1] << 8) | (data[i * 3 + 2] << 16); break;
            case 16: value = static_cast<uint32_t>(data[i * 2]) | (data[i * 2 + 1] << 8); break;
            case 8: value = data[i]; break;
            default: value = (data[i >> 1] >> ((i & 1u) * 4u)) & 0xfu; break;
            }
            write(m_memory.data(), dpsm, dbp, dbw, x, y, value);
        }
        forEachPage(dpsm, dbp, dbw, dx, dy, w, h, nullptr);
        ++m_uploads;
    }

    bool TextureCache::stillValid(const Texture &texture) const
    {
        for (const auto &[page, generation] : texture.pages)
            if (m_pageGeneration[page] != generation)
                return false;
        return true;
    }

    uint32_t TextureCache::decodeTexel(uint32_t psm, uint32_t tbp, uint32_t tbw, uint32_t x, uint32_t y, uint64_t tex0,
                                       uint64_t texa, const uint32_t *palette) const
    {
        uint8_t *memory = const_cast<uint8_t *>(m_memory.data());
        const uint32_t value = read(memory, psm, tbp, tbw, x, y);
        if (indexed(psm))
            return palette[value & 0xffu];
        (void)tex0;
        switch (psm)
        {
        case CT32: return fromCt32(value);
        case CT24: return fromCt24(value, texa);
        default: return fromCt16(value, texa);
        }
    }

    const TextureCache::Texture &TextureCache::texture(uint64_t tex0, uint64_t texa)
    {
        // TBP0 TBW PSM TW TH TCC, CBP CPSM CSM CSA (TFX and CLD do not change the image).
        const uint64_t selected = tex0 & ~((uint64_t{3} << 35) | (uint64_t{7} << 61));
        const uint32_t psm = (tex0 >> 20) & 0x3fu;
        const uint64_t key = selected ^ (indexed(psm) || psm == CT32 ? 0u : (texa * 0x9e3779b97f4a7c15ull));
        auto found = m_textures.find(key);
        if (found != m_textures.end() && stillValid(found->second))
            return found->second;

        Texture texture;
        texture.key = key;
        texture.version = ++m_decodes;
        const uint32_t tbp = tex0 & 0x3fffu;
        const uint32_t tbw = (tex0 >> 14) & 0x3fu;
        texture.width = 1u << std::min<uint32_t>(10u, (tex0 >> 26) & 0xfu);
        texture.height = 1u << std::min<uint32_t>(10u, (tex0 >> 30) & 0xfu);

        uint32_t palette[256] = {};
        if (indexed(psm))
        {
            const uint32_t cbp = (tex0 >> 37) & 0x3fffu;
            const uint32_t cpsm = (tex0 >> 51) & 0xfu;
            const uint32_t csa = (tex0 >> 56) & 0x1fu;
            const bool eight = psm == T8 || psm == T8H;
            const uint32_t entries = eight ? 256u : 16u;
            uint8_t *memory = m_memory.data();
            for (uint32_t i = 0; i < entries; ++i)
            {
                uint32_t px = 0, py = 0;
                csm1Position(eight ? i : (csa * 16u + i) & 0xffu, px, py);
                palette[i] = cpsm == CT32 ? fromCt32(GSMem::ReadCT32(memory, cbp, 1, px, py))
                                          : fromCt16(GSMem::ReadCT16(memory, cbp, 1, px, py), texa);
            }
            forEachPage(cpsm == CT32 ? CT32 : CT16, cbp, 1, 0, 0, 16, 16, &texture.pages);
        }

        texture.rgba.resize(static_cast<size_t>(texture.width) * texture.height);
        for (uint32_t y = 0; y < texture.height; ++y)
            for (uint32_t x = 0; x < texture.width; ++x)
                texture.rgba[static_cast<size_t>(y) * texture.width + x] = decodeTexel(psm, tbp, tbw, x, y, tex0, texa, palette);
        forEachPage(psm, tbp, tbw, 0, 0, texture.width, texture.height, &texture.pages);
        std::sort(texture.pages.begin(), texture.pages.end());
        texture.pages.erase(std::unique(texture.pages.begin(), texture.pages.end()), texture.pages.end());
        return m_textures[key] = std::move(texture);
    }
}
