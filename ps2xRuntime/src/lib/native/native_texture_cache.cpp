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
            if ((lo >> 46) & 1u) // PRE
                m_gameRegisters[0] = (lo >> 47) & 0x7ffu;
            if (flg == 0u) // PACKED: only A+D can address the transfer registers
            {
                bool drew = false; // a picture drawn by this tag: its target noted once, with FRAME and SCISSOR then
                for (uint32_t n = 0; n < nloop; ++n)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (i + 16u > bytes)
                            return;
                        const uint32_t reg = (hi >> (4u * r)) & 0xfu;
                        const uint64_t data = u64(i), addr = u64(i + 8u) & 0xffu;
                        i += 16u;
                        if (reg == 0x0u)
                            m_gameRegisters[0] = data & 0x7ffu;
                        if ((reg == 0x4u || reg == 0x5u) && !drew)
                        {
                            drew = true;
                            const bool second = (m_gameRegisters[0] >> 9) & 1u;
                            noteTarget(m_gameRegisters[second ? 0x4du : 0x4cu], m_gameRegisters[second ? 0x41u : 0x40u]);
                        }
                        if (reg != 0xeu)
                            continue;
                        if (addr < m_gameRegisters.size())
                            m_gameRegisters[addr] = data;
                        if ((addr == 0x04u || addr == 0x05u) && !drew)
                        {
                            drew = true;
                            const bool second = (m_gameRegisters[0] >> 9) & 1u;
                            noteTarget(m_gameRegisters[second ? 0x4du : 0x4cu], m_gameRegisters[second ? 0x41u : 0x40u]);
                        }
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

    namespace
    {
        // A page of `psm` in pixels: 8 KiB, 32 blocks.
        void pageSize(uint32_t psm, uint32_t &w, uint32_t &h)
        {
            switch (psm)
            {
            case CT16:
            case CT16S: w = 64u; h = 64u; break;
            case T8: w = 128u; h = 64u; break;
            case T4: w = 128u; h = 128u; break;
            default: w = 64u; h = 32u; break; // CT32, CT24 and the formats kept in a CT32's bits
            }
        }

        // The blocks [begin, end) of the pages a rectangle of `psm` at bp, bw touches.
        void blockSpan(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                       uint32_t &begin, uint32_t &end)
        {
            uint32_t pw = 0, ph = 0;
            pageSize(psm, pw, ph);
            const uint32_t across = std::max(1u, std::max(bw, 1u) * 64u / pw);
            begin = bp + ((y / ph) * across + x / pw) * 32u;
            end = bp + (((y + h - 1u) / ph) * across + (x + w - 1u) / pw + 1u) * 32u;
        }
    }

    void TextureCache::noteTarget(uint64_t frame, uint64_t scissor)
    {
        // FRAME: FBP (pages), FBW (64 pixels), PSM; SCISSOR the rectangle drawn: its pages are the picture.
        if (frame == 0u)
            return;
        const uint32_t fbp = static_cast<uint32_t>(frame & 0x1FFu), fbw = static_cast<uint32_t>((frame >> 16) & 0x3Fu);
        const uint32_t psm = static_cast<uint32_t>((frame >> 24) & 0x3Fu);
        const uint32_t x0 = static_cast<uint32_t>(scissor & 0x7FFu), x1 = static_cast<uint32_t>((scissor >> 16) & 0x7FFu);
        const uint32_t y0 = static_cast<uint32_t>((scissor >> 32) & 0x7FFu), y1 = static_cast<uint32_t>((scissor >> 48) & 0x7FFu);
        if (x1 < x0 || y1 < y0)
            return;
        uint32_t begin = 0, end = 0;
        blockSpan(psm == CT16 || psm == CT16S ? psm : CT32, fbp * 32u, std::max(fbw, 1u), x0, y0, x1 - x0 + 1u, y1 - y0 + 1u, begin, end);
        for (Target &target : m_targets)
            if (target.begin == begin && target.end == end)
            {
                target.seq = ++m_uploadSeq;
                return;
            }
        m_targets.push_back({begin, end, ++m_uploadSeq});
    }

    void TextureCache::storeUpload(uint32_t dbp, uint32_t dbw, uint32_t dpsm, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h,
                                   const uint8_t *data, size_t bytes)
    {
        const size_t size = std::min(bytes, (static_cast<size_t>(w) * h * bitsPerPixel(dpsm) + 7u) / 8u);
        // A block of `psm` in pixels, and whether its uploads write all the bits of the memory they touch.
        uint32_t blockW = 8u, blockH = 8u;
        switch (dpsm)
        {
        case CT16:
        case CT16S: blockW = 16u; break;
        case T8: blockW = 16u; blockH = 16u; break;
        case T4: blockW = 32u; blockH = 16u; break;
        default: break;
        }
        const bool allBits = dpsm != CT24 && dpsm != T8H && dpsm != T4HL && dpsm != T4HH;
        const auto blockOf = [&](uint32_t x, uint32_t y) { return GSMem::BlockOf(dpsm, dbp, std::max(dbw, 1u), x, y) & 0x3fffu; };

        StoredUpload *placed = nullptr;
        for (StoredUpload &u : m_stored)
            if (u.dbp == dbp && u.psm == dpsm && u.dx == dx && u.dy == dy && u.w == w && u.h == h && u.dbw == dbw &&
                u.data.size() == size && std::memcmp(u.data.data(), data, size) == 0)
            {
                u.seq = ++m_uploadSeq; // the same pixels again: newest at its place, nothing to decode again
                placed = &u;
                break;
            }
        if (!placed)
        {
            StoredUpload upload;
            upload.dbp = dbp;
            upload.psm = dpsm;
            upload.dx = dx;
            upload.dy = dy;
            upload.w = w;
            upload.h = h;
            upload.dbw = dbw;
            blockSpan(dpsm, dbp, dbw, dx, dy, w, h, upload.begin, upload.end);
            for (uint32_t y = dy / blockH * blockH; y < dy + h; y += blockH)
                for (uint32_t x = dx / blockW * blockW; x < dx + w; x += blockW)
                    upload.blocks.push_back(static_cast<uint16_t>(blockOf(x, y)));
            std::sort(upload.blocks.begin(), upload.blocks.end());
            upload.blocks.erase(std::unique(upload.blocks.begin(), upload.blocks.end()), upload.blocks.end());
            if (allBits)
                for (uint32_t y = (dy + blockH - 1u) / blockH * blockH; y + blockH <= dy + h; y += blockH)
                    for (uint32_t x = (dx + blockW - 1u) / blockW * blockW; x + blockW <= dx + w; x += blockW)
                        upload.full.push_back(static_cast<uint16_t>(blockOf(x, y)));
            upload.data.assign(data, data + size);
            upload.seq = ++m_uploadSeq;
            upload.id = ++m_contentSerial;
            // Laid out once, as the GS keeps it: its blocks, and a mask of the bits it writes in them (a 4-bit or
            // high-bits format writes part of a byte, a rectangle part of a block). A decode copies blocks.
            {
                thread_local std::vector<uint8_t> lay(kMemoryBytes, 0u), bits(kMemoryBytes, 0u);
                const uint32_t bpp = bitsPerPixel(dpsm);
                const uint32_t ones = bpp >= 32u ? 0xffffffffu : (1u << bpp) - 1u;
                const size_t pixels = std::min<size_t>(static_cast<size_t>(w) * h, size * 8u / bpp);
                const uint8_t *d = upload.data.data();
                for (size_t i = 0; i < pixels; ++i)
                {
                    uint32_t value = 0u;
                    switch (bpp)
                    {
                    case 32: std::memcpy(&value, d + i * 4u, 4u); break;
                    case 24: value = d[i * 3u] | (d[i * 3u + 1u] << 8) | (d[i * 3u + 2u] << 16); break;
                    case 16: value = d[i * 2u] | (d[i * 2u + 1u] << 8); break;
                    case 8: value = d[i]; break;
                    default: value = (d[i >> 1] >> ((i & 1u) * 4u)) & 0xfu; break;
                    }
                    const uint32_t x = dx + static_cast<uint32_t>(i % w), y = dy + static_cast<uint32_t>(i / w);
                    write(lay.data(), dpsm, dbp, dbw, x, y, value);
                    write(bits.data(), dpsm, dbp, dbw, x, y, ones);
                }
                upload.laid.resize(upload.blocks.size() * 256u);
                upload.mask.resize(upload.blocks.size() * 256u);
                for (size_t k = 0; k < upload.blocks.size(); ++k)
                {
                    const size_t at = static_cast<size_t>(upload.blocks[k]) * 256u;
                    std::memcpy(upload.laid.data() + k * 256u, lay.data() + at, 256u);
                    std::memcpy(upload.mask.data() + k * 256u, bits.data() + at, 256u);
                    std::memset(lay.data() + at, 0, 256u);
                    std::memset(bits.data() + at, 0, 256u);
                }
            }
            // The same rectangle or less at the same place in the same format: the same bits again.
            m_stored.erase(std::remove_if(m_stored.begin(), m_stored.end(), [&](const StoredUpload &u) {
                               return u.dbp == dbp && u.psm == dpsm && u.dbw == dbw && u.dx >= dx && u.dy >= dy &&
                                      u.dx + u.w <= dx + w && u.dy + u.h <= dy + h;
                           }),
                           m_stored.end());
            m_stored.push_back(std::move(upload));
            static size_t s_reported = 256u;
            if (m_stored.size() >= s_reported * 2u)
            {
                s_reported = m_stored.size();
                std::fprintf(stderr, "[native-textures] %zu uploads kept\n", m_stored.size());
            }
            placed = &m_stored.back();
        }

        // The blocks it fills whole are its own from now on; an older upload left without a block of its own is gone.
        if (!allBits)
            return;
        bool filled = false;
        for (uint32_t y = (dy + blockH - 1u) / blockH * blockH; y + blockH <= dy + h; y += blockH)
            for (uint32_t x = (dx + blockW - 1u) / blockW * blockW; x + blockW <= dx + w; x += blockW)
            {
                m_blockSeq[blockOf(x, y)] = placed->seq;
                filled = true;
            }
        if (!filled)
            return;
        const uint64_t seq = placed->seq;
        const uint32_t begin = placed->begin, end = placed->end;
        m_stored.erase(std::remove_if(m_stored.begin(), m_stored.end(), [&](const StoredUpload &u) {
                           if (u.seq >= seq || u.end <= begin || end <= u.begin)
                               return false;
                           for (const uint16_t block : u.blocks)
                               if (m_blockSeq[block] <= u.seq)
                                   return false;
                           return true;
                       }),
                       m_stored.end());
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

        // The uploads the texture is made of, in the order the game sent them (an asset is often sent as one format and
        // read as another, a palette in parts): newest first, those still showing in a block the texture or its
        // palette reads, until every such block is filled whole. The game cycles assets through the same area, so
        // what is under the newest one does not count.
        uint32_t imageBegin = 0, imageEnd = 0, paletteBegin = 0, paletteEnd = 0;
        blockSpan(psm, tbp, tbw, 0, 0, width, height, imageBegin, imageEnd);
        if (indexed(psm))
            blockSpan(cpsm == CT32 ? CT32 : CT16, cbp, 1, 0, 0, 16, 16, paletteBegin, paletteEnd);
        // Marks of the blocks read: 1 the image, 2 the palette, 4 filled whole by a newer upload.
        thread_local std::vector<uint8_t> marks(kMemoryBytes / 256u, 0u);
        thread_local std::vector<uint16_t> needed;
        needed.clear();
        const auto need = [&](uint32_t block, uint8_t what) {
            block &= 0x3fffu;
            if (marks[block] == 0u)
                needed.push_back(static_cast<uint16_t>(block));
            marks[block] |= what;
        };
        // The blocks a TEX0 reads, worked out once per TEX0 (the layout formulas are the costly part of a lookup).
        thread_local std::unordered_map<uint64_t, std::vector<uint32_t>> blocksRead; // block | what << 16
        auto found = blocksRead.find(selected);
        if (found == blocksRead.end())
        {
            if (blocksRead.size() > 16384u)
                blocksRead.clear();
            std::vector<uint32_t> list;
            // Every block is at least 8 x 8 pixels.
            for (uint32_t y = 0; y < height; y += 8u)
                for (uint32_t x = 0; x < width; x += 8u)
                    list.push_back((GSMem::BlockOf(psm, tbp, std::max(tbw, 1u), x, y) & 0x3fffu) | (1u << 16));
            if (indexed(psm))
            {
                const uint32_t csa = (tex0 >> 56) & 0x1fu;
                const bool eight = psm == T8 || psm == T8H;
                for (uint32_t i = 0; i < (eight ? 256u : 16u); i += 4u)
                {
                    uint32_t px = 0, py = 0;
                    csm1Position(eight ? i : (csa * 16u + i) & 0xffu, px, py);
                    list.push_back((GSMem::BlockOf(cpsm == CT32 ? CT32 : CT16, cbp, 1, px, py) & 0x3fffu) | (2u << 16));
                }
            }
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
            found = blocksRead.emplace(selected, std::move(list)).first;
        }
        for (const uint32_t entry : found->second)
            need(entry & 0xffffu, static_cast<uint8_t>(entry >> 16));
        thread_local std::vector<const StoredUpload *> all, parts;
        all.clear();
        parts.clear();
        for (const StoredUpload &u : m_stored)
            if ((u.begin < imageEnd && imageBegin < u.end) || (paletteEnd != 0u && u.begin < paletteEnd && paletteBegin < u.end))
                all.push_back(&u);
        std::sort(all.begin(), all.end(), [](const StoredUpload *a, const StoredUpload *b) { return a->seq > b->seq; });
        size_t open = needed.size();
        bool imageCovered = false;
        for (const StoredUpload *u : all)
        {
            if (open == 0u)
                break;
            uint8_t shows = 0u;
            for (const uint16_t block : u->blocks)
                if (marks[block] != 0u && !(marks[block] & 4u))
                    shows |= marks[block];
            if (shows == 0u)
                continue;
            parts.push_back(u);
            imageCovered |= (shows & 1u) != 0u;
            for (const uint16_t block : u->full)
                if (marks[block] != 0u && !(marks[block] & 4u))
                {
                    marks[block] |= 4u;
                    --open;
                }
        }
        for (const uint16_t block : needed)
            marks[block] = 0u;
        std::reverse(parts.begin(), parts.end()); // oldest first
        // A picture the game drew there since its last upload (a copy of the frame it samples): not an asset, no
        // texture from the store.
        {
            uint64_t newestUpload = 0;
            for (const StoredUpload *u : parts)
                if (u->begin < imageEnd && imageBegin < u->end)
                    newestUpload = std::max(newestUpload, u->seq);
            for (const Target &target : m_targets)
                if (target.begin < imageEnd && imageBegin < target.end && target.seq > newestUpload)
                {
                    ++m_storeMisses;
                    m_lastSource = 0;
                    return nullptr;
                }
        }
        uint64_t signature = 1469598103934665603ull;
        for (const StoredUpload *u : parts)
            signature = (signature ^ u->id) * 1099511628211ull;

        // PS2X_NATIVE_TEX_STATS=1: every 2000 decodes, the lookups, the uploads kept and the textures decoded most.
        static const bool s_stats = std::getenv("PS2X_NATIVE_TEX_STATS") != nullptr;
        static uint64_t s_lookups = 0;
        static std::unordered_map<uint64_t, uint32_t> s_decoded;
        ++s_lookups;
        // Each content a TEX0 shows is kept (two assets drawn in turn with one TEX0 are decoded once each).
        const uint64_t cached = key ^ (signature * 0x9e3779b97f4a7c15ull);
        if (m_storeTextures.size() > 8192u)
            m_storeTextures.clear();
        if (const auto it = m_storeTextures.find(cached); it != m_storeTextures.end() && it->second.signature == signature)
        {
            m_lastSource = 3;
            return it->second.texture;
        }
        m_lastSource = parts.empty() ? 0 : parts.size() == 1u ? 1 : 2;
        if (s_stats && ++s_decoded[tex0 & 0x3fffffffffull] && (m_decodes + 1u) % 2000u == 0u)
        {
            std::vector<std::pair<uint32_t, uint64_t>> top;
            for (const auto &[t, n] : s_decoded)
                top.emplace_back(n, t);
            std::sort(top.rbegin(), top.rend());
            std::fprintf(stderr, "[native-textures] %llu decodes / %llu lookups, %zu uploads kept, %zu parts now; most:",
                         static_cast<unsigned long long>(m_decodes + 1u), static_cast<unsigned long long>(s_lookups),
                         m_stored.size(), parts.size());
            for (size_t i = 0; i < top.size() && i < 6u; ++i)
                std::fprintf(stderr, " %010llx x%u", static_cast<unsigned long long>(top[i].second), top[i].first);
            std::fprintf(stderr, "\n");
            s_decoded.clear();
        }
        if (!imageCovered)
        {
            ++m_storeMisses;
            m_storeTextures[cached] = {nullptr, signature};
            return nullptr;
        }

        // The parts laid out in a scratch image of the pages they fill, by the GS's layout formulas (once per decode;
        // nothing of it is kept), then the texture read back from it.
        static std::vector<uint8_t> scratch(kMemoryBytes);
        const auto clear = [&](uint32_t begin, uint32_t end) {
            const size_t from = static_cast<size_t>(begin) * 256u % scratch.size();
            std::memset(scratch.data() + from, 0, std::min(scratch.size() - from, static_cast<size_t>(end - begin) * 256u));
        };
        clear(imageBegin, imageEnd);
        if (paletteEnd != 0u)
            clear(paletteBegin, paletteEnd);
        // Each part's blocks as storeUpload laid them out, over what is there, only the bits the upload wrote.
        for (const StoredUpload *u : parts)
            for (size_t k = 0; k < u->blocks.size(); ++k)
            {
                uint8_t *to = scratch.data() + static_cast<size_t>(u->blocks[k]) * 256u;
                const uint8_t *from = u->laid.data() + k * 256u, *mask = u->mask.data() + k * 256u;
                for (size_t b = 0; b < 256u; b += 8u)
                {
                    uint64_t t, f, m;
                    std::memcpy(&t, to + b, 8u);
                    std::memcpy(&f, from + b, 8u);
                    std::memcpy(&m, mask + b, 8u);
                    t = (t & ~m) | (f & m);
                    std::memcpy(to + b, &t, 8u);
                }
            }

        uint32_t colours[256] = {};
        if (indexed(psm))
        {
            const uint32_t csa = (tex0 >> 56) & 0x1fu;
            const bool eight = psm == T8 || psm == T8H;
            for (uint32_t i = 0; i < (eight ? 256u : 16u); ++i)
            {
                uint32_t px = 0, py = 0;
                csm1Position(eight ? i : (csa * 16u + i) & 0xffu, px, py);
                colours[i] = cpsm == CT32 ? fromCt32(GSMem::ReadCT32(scratch.data(), cbp, 1, px, py))
                                          : fromCt16(GSMem::ReadCT16(scratch.data(), cbp, 1, px, py), texa);
            }
        }
        Texture texture;
        texture.key = key;
        texture.version = ++m_decodes;
        texture.width = width;
        texture.height = height;
        texture.rgba.resize(static_cast<size_t>(width) * height);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
            {
                const uint32_t value = read(scratch.data(), psm, tbp, tbw, x, y);
                texture.rgba[static_cast<size_t>(y) * width + x] =
                    indexed(psm) ? colours[value & 0xffu]
                                 : psm == CT32 ? fromCt32(value) : psm == CT24 ? fromCt24(value, texa) : fromCt16(value, texa);
            }
        auto shared = std::make_shared<const Texture>(std::move(texture));
        m_storeTextures[cached] = {shared, signature};
        return shared;
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
        // One texture in 16 (the GS memory decode is slow): enough to find a store that goes wrong.
        if (++compared % 16u != 0u || left <= 0)
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
