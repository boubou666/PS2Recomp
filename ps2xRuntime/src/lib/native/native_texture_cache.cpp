#include "runtime/native/native_texture_cache.h"

#include "runtime/gs/ps2_gs_memory.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    namespace
    {
        std::atomic<bool> g_uploadsWanted{false};
    }

    TextureCache &textures()
    {
        static TextureCache cache;
        return cache;
    }

    bool uploadsWanted() { return g_uploadsWanted.load(std::memory_order_relaxed); }
    void setUploadsWanted(bool wanted) { g_uploadsWanted.store(wanted, std::memory_order_relaxed); }

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
        std::lock_guard<std::mutex> lock(m_mutex);
        uploadLocked(bitbltbuf, trxpos, trxreg, data, bytes);
    }

    void TextureCache::scanGifPacket(const uint8_t *packet, size_t bytes)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t i = 0;
        auto u64 = [&](size_t at) { uint64_t v; std::memcpy(&v, packet + at, 8); return v; };
        while (i + 16u <= bytes)
        {
            const uint64_t lo = u64(i), hi = u64(i + 8u);
            i += 16u;
            const uint32_t nloop = lo & 0x7fffu;
            const uint32_t flg = (lo >> 58) & 3u;
            const uint32_t nreg = ((lo >> 60) & 0xfu) ? static_cast<uint32_t>((lo >> 60) & 0xfu) : 16u;
            if (flg == 0u) // PACKED: only A+D can address the transfer registers
            {
                for (uint32_t n = 0; n < nloop; ++n)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (i + 16u > bytes)
                            return;
                        const uint32_t reg = (hi >> (4u * r)) & 0xfu;
                        const uint64_t data = u64(i), addr = u64(i + 8u) & 0xffu;
                        i += 16u;
                        if (reg != 0xeu)
                            continue;
                        if (addr == 0x3bu) // TEXA, FOGCOL: the game's values for materials built from its data
                            m_gameTexa = data;
                        else if (addr == 0x3du)
                            m_gameFogColour = static_cast<uint32_t>(data & 0xffffffu);
                        else if (addr == 0x50u)
                            m_bitbltbuf = data;
                        else if (addr == 0x51u)
                            m_trxpos = data;
                        else if (addr == 0x52u)
                            m_trxreg = data;
                        else if (addr == 0x53u && (data & 3u) == 0u) // host to local
                        {
                            const uint64_t w = m_trxreg & 0xfffu, h = (m_trxreg >> 32) & 0xfffu;
                            m_pending.clear();
                            m_pendingWanted = static_cast<size_t>((w * h * bitsPerPixel((m_bitbltbuf >> 56) & 0x3fu) + 7u) / 8u);
                        }
                    }
            }
            else if (flg == 1u) // REGLIST
                i += ((static_cast<size_t>(nloop) * nreg + 1u) & ~size_t{1}) * 8u;
            else // IMAGE
            {
                const size_t size = std::min<size_t>(static_cast<size_t>(nloop) * 16u, bytes - i);
                if (m_pendingWanted != 0u)
                {
                    m_pending.insert(m_pending.end(), packet + i, packet + i + size);
                    if (m_pending.size() >= m_pendingWanted)
                    {
                        uploadLocked(m_bitbltbuf, m_trxpos, m_trxreg, m_pending.data(), m_pending.size());
                        m_pending.clear();
                        m_pendingWanted = 0u;
                    }
                }
                i += size;
            }
        }
    }

    void TextureCache::setStoreMode(bool on)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storeMode = on;
    }

    void TextureCache::storeUpload(uint32_t dbp, uint32_t dbw, uint32_t dpsm, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h,
                                   const uint8_t *data, size_t bytes)
    {
        const size_t wanted = (static_cast<size_t>(w) * h * bitsPerPixel(dpsm) + 7u) / 8u;
        StoredBase &base = m_store[(static_cast<uint64_t>(dbp) << 6) | dpsm];
        for (auto it = base.uploads.begin(); it != base.uploads.end();)
        {
            if (it->dx == dx && it->dy == dy && it->w == w && it->h == h && it->dbw == dbw && it->data.size() == std::min(bytes, wanted) &&
                std::memcmp(it->data.data(), data, it->data.size()) == 0)
            {
                it->seq = ++m_uploadSeq; // the same pixels again: newest at its place, nothing to decode again
                return;
            }
            // An older upload this one covers whole is gone.
            if (it->dx >= dx && it->dy >= dy && it->dx + it->w <= dx + w && it->dy + it->h <= dy + h)
                it = base.uploads.erase(it);
            else
                ++it;
        }
        StoredUpload upload;
        upload.dx = dx;
        upload.dy = dy;
        upload.w = w;
        upload.h = h;
        upload.dbw = dbw;
        upload.data.assign(data, data + std::min(bytes, wanted));
        upload.seq = ++m_uploadSeq;
        upload.id = ++m_contentSerial;
        base.uploads.push_back(std::move(upload));
        base.serial = m_contentSerial;
    }

    const TextureCache::StoredUpload *TextureCache::laidOutSource(uint32_t base, uint32_t &sourceBase, uint64_t &serial) const
    {
        // The newest CT32 upload whose span (pages of 64 x 32 pixels, dbw across) holds `base`.
        const StoredUpload *source = nullptr;
        for (const auto &[key, stored] : m_store)
        {
            if ((key & 0x3fu) != CT32)
                continue;
            const uint32_t dbp = static_cast<uint32_t>(key >> 6);
            for (const StoredUpload &u : stored.uploads)
            {
                const uint32_t end = dbp + ((u.dy + u.h + 31u) / 32u) * std::max(u.dbw, 1u) * 32u;
                if (base >= dbp && base < end && (!source || u.seq > source->seq))
                {
                    source = &u;
                    sourceBase = dbp;
                    serial = u.id;
                }
            }
        }
        return source;
    }

    std::shared_ptr<const TextureCache::Texture> TextureCache::textureFromStore(uint64_t tex0, uint64_t texa)
    {
        const uint64_t selected = tex0 & ~((uint64_t{3} << 35) | (uint64_t{7} << 61));
        const uint32_t psm = (tex0 >> 20) & 0x3fu;
        const uint64_t key = selected ^ (indexed(psm) || psm == CT32 ? 0u : (texa * 0x9e3779b97f4a7c15ull));
        const uint32_t tbp = tex0 & 0x3fffu, tbw = (tex0 >> 14) & 0x3fu;
        const uint32_t cbp = (tex0 >> 37) & 0x3fffu, cpsm = (tex0 >> 51) & 0xfu;
        const uint32_t width = 1u << std::min<uint32_t>(10u, (tex0 >> 26) & 0xfu);
        const uint32_t height = 1u << std::min<uint32_t>(10u, (tex0 >> 30) & 0xfu);

        // Where the image and the palette come from: an upload in their own format at their base, or a CT32 upload
        // holding them in the GS's layout (laidOut). Each source's serial says when it changed.
        const auto direct = [&](uint32_t bp, uint32_t format) -> const StoredBase * {
            const auto it = m_store.find((static_cast<uint64_t>(bp) << 6) | format);
            return it == m_store.end() || it->second.uploads.empty() ? nullptr : &it->second;
        };
        // The newer of the two wins: the game reuses its texture area for one asset after another.
        const auto newest = [](const StoredBase *b) {
            uint64_t seq = 0;
            for (const StoredUpload &u : b->uploads)
                seq = std::max(seq, u.seq);
            return seq;
        };
        const auto pick = [&](uint32_t bp, uint32_t format, const StoredBase *&base, const StoredUpload *&laid,
                              uint32_t &laidBase, uint64_t &serial) {
            base = direct(bp, format);
            uint64_t laidSerial = 0;
            laid = format == CT32 ? nullptr : laidOutSource(bp, laidBase, laidSerial);
            if (base && laid && newest(base) > laid->seq)
                laid = nullptr;
            if (laid)
                base = nullptr;
            serial = laid ? laidSerial : base ? base->serial : 0u;
        };
        const StoredBase *image = nullptr, *palette = nullptr;
        const StoredUpload *imageLaid = nullptr, *paletteLaid = nullptr;
        uint32_t imageLaidBase = 0, paletteLaidBase = 0;
        uint64_t imageSerial = 0u, paletteSerial = 0u;
        pick(tbp, psm, image, imageLaid, imageLaidBase, imageSerial);
        if (indexed(psm))
        {
            if (cpsm == CT32)
            {
                // A CT32 palette is either its own upload or a part of a CT32 asset: the newest upload holding it.
                palette = direct(cbp, cpsm);
                uint64_t laidSerial = 0;
                paletteLaid = laidOutSource(cbp, paletteLaidBase, laidSerial);
                if (paletteLaid && palette && newest(palette) >= paletteLaid->seq && paletteLaidBase == cbp)
                    paletteLaid = nullptr;
                if (paletteLaid)
                    palette = nullptr;
                paletteSerial = paletteLaid ? laidSerial : palette ? palette->serial : 0u;
            }
            else
            {
                palette = direct(cbp, cpsm);
                paletteSerial = palette ? palette->serial : 0u;
            }
        }
        if (const auto it = m_storeTextures.find(key);
            it != m_storeTextures.end() && it->second.imageSerial == imageSerial && it->second.paletteSerial == paletteSerial)
        {
            m_lastSource = 3;
            return it->second.texture;
        }
        m_lastSource = imageLaid ? 2 : image ? 1 : 0;
        if (!image && !imageLaid)
        {
            ++m_storeMisses;
            m_storeTextures[key] = {nullptr, imageSerial, paletteSerial};
            return nullptr;
        }

        uint32_t colours[256] = {};
        if (indexed(psm))
        {
            const uint32_t csa = (tex0 >> 56) & 0x1fu;
            const bool eight = psm == T8 || psm == T8H;
            std::vector<uint32_t> laid;
            if (paletteLaid)
                laidOut(*paletteLaid, cbp - paletteLaidBase, CT32, 1, 16, 16, laid);
            for (uint32_t i = 0; i < (eight ? 256u : 16u); ++i)
            {
                uint32_t px = 0, py = 0;
                csm1Position(eight ? i : (csa * 16u + i) & 0xffu, px, py);
                if (paletteLaid)
                {
                    colours[i] = fromCt32(laid[static_cast<size_t>(py) * 16u + px]);
                    continue;
                }
                if (!palette)
                    continue;
                for (auto u = palette->uploads.rbegin(); u != palette->uploads.rend(); ++u)
                    if (px >= u->dx && py >= u->dy && px < u->dx + u->w && py < u->dy + u->h)
                    {
                        const size_t at = static_cast<size_t>(py - u->dy) * u->w + (px - u->dx);
                        if (cpsm == CT32 && at * 4u + 4u <= u->data.size())
                        {
                            uint32_t c;
                            std::memcpy(&c, u->data.data() + at * 4u, 4u);
                            colours[i] = fromCt32(c);
                        }
                        else if (cpsm != CT32 && at * 2u + 2u <= u->data.size())
                            colours[i] = fromCt16(static_cast<uint32_t>(u->data[at * 2u]) | (u->data[at * 2u + 1u] << 8), texa);
                        break;
                    }
            }
        }
        const auto colourOf = [&](uint32_t value) {
            if (indexed(psm))
                return colours[value & 0xffu];
            return psm == CT32 ? fromCt32(value) : psm == CT24 ? fromCt24(value, texa) : fromCt16(value, texa);
        };

        Texture texture;
        texture.key = key;
        texture.version = ++m_decodes;
        texture.width = width;
        texture.height = height;
        texture.rgba.assign(static_cast<size_t>(width) * height, 0u);
        if (imageLaid)
        {
            std::vector<uint32_t> values;
            laidOut(*imageLaid, tbp - imageLaidBase, psm, tbw, width, height, values);
            for (size_t i = 0; i < values.size(); ++i)
                texture.rgba[i] = colourOf(values[i]);
        }
        else
        {
            const uint32_t bpp = bitsPerPixel(psm);
            for (const StoredUpload &u : image->uploads) // oldest first: newer ones overwrite
            {
                const uint32_t x1 = std::min(u.dx + u.w, width), y1 = std::min(u.dy + u.h, height);
                for (uint32_t y = u.dy; y < y1; ++y)
                    for (uint32_t x = u.dx; x < x1; ++x)
                    {
                        const size_t i = static_cast<size_t>(y - u.dy) * u.w + (x - u.dx);
                        uint32_t value = 0u;
                        if (bpp == 32 && i * 4u + 4u <= u.data.size())
                            std::memcpy(&value, u.data.data() + i * 4u, 4u);
                        else if (bpp == 24 && i * 3u + 3u <= u.data.size())
                            value = u.data[i * 3u] | (u.data[i * 3u + 1u] << 8) | (u.data[i * 3u + 2u] << 16);
                        else if (bpp == 16 && i * 2u + 2u <= u.data.size())
                            value = u.data[i * 2u] | (u.data[i * 2u + 1u] << 8);
                        else if (bpp == 8 && i < u.data.size())
                            value = u.data[i];
                        else if (bpp == 4 && (i >> 1) < u.data.size())
                            value = (u.data[i >> 1] >> ((i & 1u) * 4u)) & 0xfu;
                        else
                            continue;
                        texture.rgba[static_cast<size_t>(y) * width + x] = colourOf(value);
                    }
            }
        }
        auto shared = std::make_shared<const Texture>(std::move(texture));
        m_storeTextures[key] = {shared, imageSerial, paletteSerial};
        return shared;
    }

    void TextureCache::laidOut(const StoredUpload &source, uint32_t relativeBase, uint32_t psm, uint32_t bw, uint32_t width,
                               uint32_t height, std::vector<uint32_t> &values)
    {
        // The asset as the CT32 data it was uploaded as, in a scratch image of its pages alone, read back at
        // `relativeBase` in `psm`: the GS layout's formula, once per asset decode (no GS memory is kept).
        static std::vector<uint8_t> scratch(kMemoryBytes);
        const uint32_t pagesAcross = std::max(source.dbw, 1u);
        const size_t used = std::min<size_t>(scratch.size(), static_cast<size_t>((source.dy + source.h + 31u) / 32u) * pagesAcross * kPageBytes +
                                                                   static_cast<size_t>(relativeBase) * 256u + kPageBytes * 4u);
        std::memset(scratch.data(), 0, used);
        for (uint32_t y = 0; y < source.h; ++y)
            for (uint32_t x = 0; x < source.w; ++x)
            {
                const size_t i = static_cast<size_t>(y) * source.w + x;
                if (i * 4u + 4u > source.data.size())
                    break;
                uint32_t value;
                std::memcpy(&value, source.data.data() + i * 4u, 4u);
                GSMem::WriteCT32(scratch.data(), 0, pagesAcross, source.dx + x, source.dy + y, value);
            }
        values.assign(static_cast<size_t>(width) * height, 0u);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                values[static_cast<size_t>(y) * width + x] = read(scratch.data(), psm, relativeBase, std::max(bw, 1u), x, y);
    }

    void TextureCache::uploadLocked(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, const uint8_t *data, size_t bytes)
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
        if (m_storeMode)
        {
            storeUpload(dbp, dbw, dpsm, dx, dy, w, h, data, bytes);
            if (!compareStore())
            {
                ++m_uploads;
                return;
            }
        }
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

    bool TextureCache::compareStore()
    {
        // PS2X_NATIVE_TEX_COMPARE=<n>: store mode also keeps the GS memory copy, and logs the first n textures whose
        // store decode differs from the GS memory decode (a diagnosis of the store, never a drawing path).
        static const int limit = [] { const char *v = std::getenv("PS2X_NATIVE_TEX_COMPARE"); return v ? std::atoi(v) : 0; }();
        return limit > 0;
    }

    void TextureCache::compareWithMemory(uint64_t tex0, uint64_t texa, const Texture *stored)
    {
        static int left = [] { const char *v = std::getenv("PS2X_NATIVE_TEX_COMPARE"); return v ? std::atoi(v) : 0; }();
        static uint64_t compared = 0, differing = 0;
        ++compared;
        if (left <= 0)
            return;
        const auto reference = memoryTexture(tex0, texa);
        size_t wrong = 0, first = 0;
        if (stored)
            for (size_t i = 0; i < reference->rgba.size() && i < stored->rgba.size(); ++i)
                if (reference->rgba[i] != stored->rgba[i] && wrong++ == 0)
                    first = i;
        if (stored && wrong == 0)
            return;
        ++differing;
        --left;
        const uint32_t width = reference->width;
        std::fprintf(stderr, "[tex-compare] tex0=%016llx psm=%02x source=%d %s wrong=%zu/%zu first=(%zu,%zu) store=%08x gs=%08x (%llu/%llu differ)\n",
                     static_cast<unsigned long long>(tex0), static_cast<unsigned>((tex0 >> 20) & 0x3fu), m_lastSource,
                     stored ? "decoded" : "missing", wrong, reference->rgba.size(), first % width, first / width,
                     stored && first < stored->rgba.size() ? stored->rgba[first] : 0u, reference->rgba[first],
                     static_cast<unsigned long long>(differing), static_cast<unsigned long long>(compared));
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

    std::shared_ptr<const TextureCache::Texture> TextureCache::texture(uint64_t tex0, uint64_t texa)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_storeMode)
            return memoryTexture(tex0, texa);
        auto stored = textureFromStore(tex0, texa);
        if (compareStore())
            compareWithMemory(tex0, texa, stored.get());
        return stored;
    }

    std::shared_ptr<const TextureCache::Texture> TextureCache::memoryTexture(uint64_t tex0, uint64_t texa)
    {
        // TBP0 TBW PSM TW TH TCC, CBP CPSM CSM CSA (TFX and CLD do not change the image).
        const uint64_t selected = tex0 & ~((uint64_t{3} << 35) | (uint64_t{7} << 61));
        const uint32_t psm = (tex0 >> 20) & 0x3fu;
        const uint64_t key = selected ^ (indexed(psm) || psm == CT32 ? 0u : (texa * 0x9e3779b97f4a7c15ull));
        auto found = m_textures.find(key);
        if (found != m_textures.end() && stillValid(*found->second))
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
        auto shared = std::make_shared<const Texture>(std::move(texture));
        m_textures[key] = shared;
        return shared;
    }
}
