#include "runtime/native/native_gif.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2x::native
{
    namespace
    {
        constexpr float kScreenSpace[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

        float asFloat(uint32_t bits)
        {
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        bool sameMaterial(const Material &a, const Material &b)
        {
            return a.tex0 == b.tex0 && a.texa == b.texa && a.textured == b.textured && a.blend == b.blend && a.alphaRef == b.alphaRef &&
                   a.alphaTest == b.alphaTest && a.depthTest == b.depthTest && a.depthWrite == b.depthWrite && a.fog == b.fog &&
                   a.fogColour == b.fogColour && a.clampU == b.clampU && a.clampV == b.clampV;
        }
    }

    void GifToNative::packet(const uint8_t *data, size_t bytes) { parse(data, bytes, true); }

    void GifToNative::registers(const uint8_t *data, size_t bytes) { parse(data, bytes, false); }

    void GifToNative::parse(const uint8_t *data, size_t bytes, bool draw)
    {
        const size_t qwords = bytes / 16u;
        size_t q = 0;
        while (q < qwords)
        {
            uint64_t tag, regs;
            std::memcpy(&tag, data + q * 16u, 8);
            std::memcpy(&regs, data + q * 16u + 8u, 8);
            ++q;
            const uint32_t nloop = tag & 0x7fffu, flg = (tag >> 58) & 3u;
            const uint32_t nreg = ((tag >> 60) & 0xfu) ? static_cast<uint32_t>((tag >> 60) & 0xfu) : 16u;
            const bool eop = (tag >> 15) & 1u;
            static const bool s_logTags = []
            {
                const char *value = std::getenv("PS2X_NATIVE_LOG_GIF");
                return value && *value == '2';
            }();
            static int s_tags = 0;
            if (s_logTags && (m_source >> 16) == 0x3270u && s_tags < 24)
            {
                ++s_tags;
                std::fprintf(stderr, "[native] gif ribbon tag %s: nloop %u eop %d pre %d prim %#llx flg %u nreg %u regs %#llx (queued %d strip %d)\n",
                             draw ? "draw" : "registers", nloop, eop ? 1 : 0, static_cast<int>((tag >> 46) & 1u),
                             static_cast<unsigned long long>((tag >> 47) & 0x7ffu), flg, nreg, static_cast<unsigned long long>(regs), m_queued, m_strip);
            }
            if (((tag >> 46) & 1u) && flg != 2u) // PRE: the tag's PRIM field
                write(0x00, (tag >> 47) & 0x7ffu, draw);
            if (flg == 0u) // PACKED
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r, ++q)
                    {
                        if (q >= qwords)
                            return;
                        uint32_t w[4];
                        std::memcpy(w, data + q * 16u, 16);
                        const uint64_t lo = w[0] | (static_cast<uint64_t>(w[1]) << 32);
                        switch ((regs >> (4u * r)) & 0xfu)
                        {
                        case 0x0: write(0x00, w[0] & 0x7ffu, draw); break;
                        case 0x1: // RGBAQ: Q from the last ST
                            m_rgbaq = (w[0] & 0xffu) | ((w[1] & 0xffu) << 8) | ((w[2] & 0xffu) << 16) | (static_cast<uint64_t>(w[3] & 0xffu) << 24) |
                                      (m_rgbaq & 0xffffffff00000000ull);
                            break;
                        case 0x2: // ST, and Q
                            m_s = asFloat(w[0]);
                            m_t = asFloat(w[1]);
                            m_rgbaq = (m_rgbaq & 0xffffffffull) | (static_cast<uint64_t>(w[2]) << 32);
                            break;
                        case 0x3: m_uv = (w[0] & 0x3fffu) | ((w[1] & 0x3fffu) << 16); break;
                        case 0x4: // XYZF2
                        case 0xc: // XYZF3
                        {
                            const uint64_t value = (w[0] & 0xffffu) | ((w[1] & 0xffffu) << 16) | (static_cast<uint64_t>((w[2] >> 4) & 0xffffffu) << 32) |
                                                   (static_cast<uint64_t>((w[3] >> 4) & 0xffu) << 56);
                            const bool adc = (w[3] >> 15) & 1u;
                            vertex(value, true, !adc && ((regs >> (4u * r)) & 0xfu) == 0x4, draw);
                            break;
                        }
                        case 0x5: // XYZ2
                        case 0xd: // XYZ3
                        {
                            const uint64_t value = (w[0] & 0xffffu) | ((w[1] & 0xffffu) << 16) | (static_cast<uint64_t>(w[2]) << 32);
                            const bool adc = (w[3] >> 15) & 1u;
                            vertex(value, false, !adc && ((regs >> (4u * r)) & 0xfu) == 0x5, draw);
                            break;
                        }
                        case 0x6: write(0x06, lo, draw); break;
                        case 0x7: write(0x07, lo, draw); break;
                        case 0x8: write(0x08, lo, draw); break;
                        case 0x9: write(0x09, lo, draw); break;
                        case 0xa: m_fog = (w[3] >> 4) & 0xffu; break;
                        case 0xe: write(w[2] & 0xffu, lo, draw); break; // A+D
                        default: break; // NOP
                        }
                    }
            }
            else if (flg == 1u) // REGLIST: 64-bit register values, two per quadword
            {
                const size_t words = static_cast<size_t>(nloop) * nreg;
                for (size_t i = 0; i < words; ++i)
                {
                    if (q * 16u + i * 8u + 8u > bytes)
                        return;
                    uint64_t value;
                    std::memcpy(&value, data + q * 16u + i * 8u, 8);
                    const uint32_t reg = (regs >> (4u * (i % nreg))) & 0xfu;
                    if (reg == 0xeu || reg == 0xfu)
                        continue;
                    write(reg == 0xau ? 0x0au : reg, value, draw);
                }
                q += (words + 1u) / 2u;
            }
            else // IMAGE (a texture upload: the texture cache has it from the GS front end)
                q += nloop;
            (void)eop; // a DIRECT packet can hold several GIF packets: all of it is read
        }
    }

    void GifToNative::write(uint32_t reg, uint64_t value, bool draw)
    {
        // The drawing state a packet sets (not the vertex queue's registers), for the GS to set too (setGsWrites).
        if (m_gsWrites && draw && reg >= 0x06u && reg != 0x0cu && reg != 0x0du)
            m_pendingWrites.emplace_back(static_cast<uint8_t>(reg), value);
        switch (reg)
        {
        case 0x00: // PRIM: a new primitive
            m_prim = value & 0x7ffu;
            m_queued = 0;
            m_strip = 0;
            break;
        case 0x01: m_rgbaq = value; break;
        case 0x02:
            m_s = asFloat(static_cast<uint32_t>(value));
            m_t = asFloat(static_cast<uint32_t>(value >> 32));
            break;
        case 0x03: m_uv = value; break;
        case 0x04: vertex(value, true, true, draw); break;
        case 0x05: vertex(value, false, true, draw); break;
        case 0x0c: vertex(value, true, false, draw); break;
        case 0x0d: vertex(value, false, false, draw); break;
        case 0x06: m_tex0[0] = value; break;
        case 0x07: m_tex0[1] = value; break;
        case 0x08: m_clamp[0] = value; break;
        case 0x09: m_clamp[1] = value; break;
        case 0x0a: m_fog = (value >> 56) & 0xffu; break;
        case 0x1a: m_prmodecont = value & 1u; break;
        case 0x1b: m_prmode = value; break;
        case 0x3b: m_texa = value; break;
        case 0x3d: m_fogcol = value; break;
        case 0x42: m_alpha[0] = value; break;
        case 0x43: m_alpha[1] = value; break;
        case 0x47: m_test[0] = value; break;
        case 0x48: m_test[1] = value; break;
        case 0x4e: m_zbuf[0] = value; break;
        case 0x4f: m_zbuf[1] = value; break;
        default: break;
        }
    }

    void GifToNative::vertex(uint64_t xyz, bool withFog, bool kick, bool draw)
    {
        if (!draw)
            return; // the GS draws it
        const uint64_t attributes = m_prmodecont ? m_prim : m_prmode;
        const int context = (attributes >> 9) & 1u;
        Kick k;
        k.x = static_cast<float>(xyz & 0xffffu) / 16.0f;
        k.y = static_cast<float>((xyz >> 16) & 0xffffu) / 16.0f;
        const uint32_t z = withFog ? static_cast<uint32_t>((xyz >> 32) & 0xffffffu) : static_cast<uint32_t>(xyz >> 32);
        k.z = static_cast<float>(z) / Renderer::instance().gsZScale();
        k.fog = static_cast<float>(withFog ? (xyz >> 56) & 0xffu : m_fog) / 255.0f;
        k.rgba = static_cast<uint32_t>(m_rgbaq);
        const uint64_t tex0 = m_tex0[context];
        if ((attributes >> 8) & 1u) // FST: UV in texels (10.4)
        {
            const float tw = static_cast<float>(1u << ((tex0 >> 26) & 0xfu)), th = static_cast<float>(1u << ((tex0 >> 30) & 0xfu));
            k.u = static_cast<float>(m_uv & 0x3fffu) / 16.0f / tw;
            k.v = static_cast<float>((m_uv >> 16) & 0x3fffu) / 16.0f / th;
        }
        else
        {
            const float q = asFloat(static_cast<uint32_t>(m_rgbaq >> 32));
            k.u = q != 0.0f ? m_s / q : m_s;
            k.v = q != 0.0f ? m_t / q : m_t;
            k.q = q != 0.0f ? q : 1.0f;
        }

        const uint32_t type = m_prim & 7u;
        switch (type)
        {
        case 0: // point
            if (kick)
                emit(&k, 1);
            break;
        case 1: // line
        case 3: // triangle
        case 6: // sprite
        {
            const int needed = type == 3 ? 3 : 2;
            m_queue[m_queued++] = k;
            if (m_queued == needed)
            {
                if (kick)
                    emit(m_queue, needed);
                m_queued = 0;
            }
            break;
        }
        case 2: // line strip
            if (m_queued == 1 && kick)
            {
                const Kick pair[2] = {m_queue[0], k};
                emit(pair, 2);
            }
            m_queue[0] = k;
            m_queued = 1;
            break;
        case 4: // triangle strip
        case 5: // triangle fan
            if (m_strip >= 2 && kick)
            {
                const Kick triangle[3] = {m_queue[0], m_queue[1], k};
                emit(triangle, 3);
            }
            if (m_strip < 2)
                m_queue[m_strip] = k;
            else if (type == 4)
            {
                m_queue[0] = m_queue[1];
                m_queue[1] = k;
            }
            else
                m_queue[1] = k;
            ++m_strip;
            break;
        default: break;
        }
    }

    Material GifToNative::material() const
    {
        const uint64_t attributes = m_prmodecont ? m_prim : m_prmode;
        const int context = (attributes >> 9) & 1u;
        Material m;
        m.textured = (attributes >> 4) & 1u;
        m.tex0 = m.textured ? m_tex0[context] : 0u;
        m.texa = m_texa;
        const uint32_t wms = m_clamp[context] & 3u, wmt = (m_clamp[context] >> 2) & 3u;
        m.clampU = wms == 1u || wms == 2u; // CLAMP, REGION_CLAMP (its region taken as the texture)
        m.clampV = wmt == 1u || wmt == 2u;
        m.blend = Material::Blend::Opaque;
        if ((attributes >> 6) & 1u) // ABE
        {
            const uint64_t alpha = m_alpha[context];
            const uint32_t a = alpha & 3u, b = (alpha >> 2) & 3u, d = (alpha >> 6) & 3u;
            m.blend = a == 0u && b == 2u && d == 1u ? Material::Blend::Additive : Material::Blend::Alpha;
        }
        const uint64_t test = m_test[context];
        const bool alphaTest = test & 1u;
        const uint32_t atst = (test >> 1) & 7u, aref = (test >> 4) & 0xffu, afail = (test >> 12) & 3u;
        m.depthWrite = !((m_zbuf[context] >> 32) & 1u);
        if (alphaTest && atst == 0u && afail == 1u) // never passes, the colour still written (FB_ONLY): no depth
            m.depthWrite = false;
        else if (alphaTest && (atst == 5u || atst == 6u)) // GEQUAL, GREATER
        {
            m.alphaTest = atst == 6u || aref > 0u;
            m.alphaRef = static_cast<uint8_t>(atst == 6u ? aref : aref - 1u);
        }
        const uint32_t ztst = (test >> 17) & 3u;
        m.depthTest = ((test >> 16) & 1u) && ztst != 1u;
        m.fog = (attributes >> 5) & 1u;
        m.fogColour = static_cast<uint32_t>(m_fogcol & 0xffffffu);
        return m;
    }

    void GifToNative::emit(const Kick *kicks, int count)
    {
        // PS2X_NATIVE_LOG_GIF=2: per source and primitive type, how many and their average size on screen (pixels),
        // every 20000 primitives.
        static const bool s_stats = []
        {
            const char *value = std::getenv("PS2X_NATIVE_LOG_GIF");
            return value && *value == '2';
        }();
        if (s_stats)
        {
            struct Stat
            {
                uint64_t count = 0;
                double width = 0, height = 0;
            };
            static std::unordered_map<uint64_t, Stat> s_byKind;
            static uint64_t s_total = 0;
            float x0 = kicks[0].x, x1 = x0, y0 = kicks[0].y, y1 = y0;
            for (int i = 1; i < count; ++i)
            {
                x0 = std::min(x0, kicks[i].x), x1 = std::max(x1, kicks[i].x);
                y0 = std::min(y0, kicks[i].y), y1 = std::max(y1, kicks[i].y);
            }
            static int s_dumps = 0;
            if ((m_source & 0xffffu) == 0u && s_dumps < 12)
            {
                ++s_dumps;
                std::fprintf(stderr, "[native] gif source %#x prim %#llx:", m_source, static_cast<unsigned long long>(m_prim));
                for (int i = 0; i < count; ++i)
                    std::fprintf(stderr, " (%.2f %.2f %.1f uv %.3f %.3f %08x)", kicks[i].x, kicks[i].y, kicks[i].z, kicks[i].u, kicks[i].v, kicks[i].rgba);
                std::fprintf(stderr, "\n");
            }
            static int s_ribbons = 0;
            if ((m_source >> 16) == 0x3270u && s_ribbons < 6)
            {
                ++s_ribbons;
                const int ctx = ((m_prmodecont ? m_prim : m_prmode) >> 9) & 1u;
                std::fprintf(stderr, "[native] gif ribbon tex0 %#llx clamp %#llx alpha %#llx test %#llx:", static_cast<unsigned long long>(m_tex0[ctx]),
                             static_cast<unsigned long long>(m_clamp[ctx]), static_cast<unsigned long long>(m_alpha[ctx]),
                             static_cast<unsigned long long>(m_test[ctx]));
                for (int i = 0; i < count; ++i)
                    std::fprintf(stderr, " (%.1f %.1f uv %.3f %.3f)", kicks[i].x, kicks[i].y, kicks[i].u, kicks[i].v);
                std::fprintf(stderr, "\n");
            }
            Stat &stat = s_byKind[(static_cast<uint64_t>(m_source) << 8) | (m_prim & 0x7fu)];
            ++stat.count;
            stat.width += x1 - x0;
            stat.height += y1 - y0;
            if (++s_total % 20000u == 0u)
            {
                for (const auto &[key, value] : s_byKind)
                    std::fprintf(stderr, "[native] gif source %#llx prim %#llx (type %llu, abe %llu): %llu, average %.1f x %.1f\n",
                                 static_cast<unsigned long long>(key >> 8), static_cast<unsigned long long>(key & 0xffu),
                                 static_cast<unsigned long long>(key & 7u), static_cast<unsigned long long>((key >> 6) & 1u),
                                 static_cast<unsigned long long>(value.count), value.width / value.count, value.height / value.count);
                s_byKind.clear();
            }
        }
        const uint64_t attributes = m_prmodecont ? m_prim : m_prmode;
        const int context = (attributes >> 9) & 1u;
        const uint64_t test = m_test[context];
        if ((test & 1u) && ((test >> 1) & 7u) == 0u && ((test >> 12) & 3u) != 1u)
            return; // the alpha test never passes and nothing is written (KEEP), or only depth (ZB_ONLY)
        if (((test >> 16) & 1u) && ((test >> 17) & 3u) == 0u)
            return; // depth test NEVER
        Material m = material();
        if (m_gsWrites)
        {
            m.fromGsContext = true;
            m.context = static_cast<uint8_t>(context);
            m.alphaBlend = (attributes >> 6) & 1u;
        }
        if (!m_batchOpen || !sameMaterial(m, m_batchMaterial) || !m_pendingWrites.empty())
        {
            flush();
            m_batchMaterial = m;
            m_batchOpen = true;
            if (!m_pendingWrites.empty())
            {
                m_batchMaterial.gsWrites = std::make_shared<const std::vector<std::pair<uint8_t, uint64_t>>>(std::move(m_pendingWrites));
                m_pendingWrites.clear();
            }
        }
        // A constant alpha (ALPHA.C = FIX) stands in for the source alpha.
        const uint64_t alpha = m_alpha[context];
        const bool fixedAlpha = m.blend != Material::Blend::Opaque && ((alpha >> 4) & 3u) == 2u;
        const bool flat = !((attributes >> 3) & 1u); // IIP = 0: the last vertex's colour
        auto out = [&](const Kick &k, uint32_t rgba)
        {
            Vertex v;
            v.x = k.x;
            v.y = k.y;
            v.z = k.z;
            v.u = k.u;
            v.v = k.v;
            v.q = m_gsWrites ? k.q : 1.0f;
            v.fog = k.fog;
            v.rgba = fixedAlpha ? (rgba & 0x00ffffffu) | (static_cast<uint32_t>((alpha >> 32) & 0xffu) << 24) : rgba;
            m_triangles.push_back(v);
        };
        const uint32_t last = kicks[count - 1].rgba;
        if (count == 3)
        {
            for (int i = 0; i < 3; ++i)
                out(kicks[i], flat ? last : kicks[i].rgba);
            return;
        }
        // Quads: a sprite's two corners, a line one pixel wide, a point one pixel square.
        Kick corner[4];
        const uint32_t type = m_prim & 7u;
        if (count == 2 && type == 6)
        {
            const Kick &a = kicks[0], &b = kicks[1];
            const float au = a.u * a.q / b.q, av = a.v * a.q / b.q;
            corner[0] = b;
            corner[0].x = a.x, corner[0].y = a.y, corner[0].u = au, corner[0].v = av;
            corner[1] = b;
            corner[1].y = a.y, corner[1].v = av;
            corner[2] = b;
            corner[2].x = a.x, corner[2].u = au;
            corner[3] = b;
            for (Kick &c : corner)
                c.rgba = last;
        }
        else if (count == 2)
        {
            const Kick &a = kicks[0], &b = kicks[1];
            const float dx = b.x - a.x, dy = b.y - a.y, length = std::sqrt(dx * dx + dy * dy);
            const float nx = length > 0.0f ? -dy / length * 0.5f : 0.5f, ny = length > 0.0f ? dx / length * 0.5f : 0.0f;
            corner[0] = a, corner[1] = a, corner[2] = b, corner[3] = b;
            corner[0].x += nx, corner[0].y += ny, corner[1].x -= nx, corner[1].y -= ny;
            corner[2].x += nx, corner[2].y += ny, corner[3].x -= nx, corner[3].y -= ny;
            if (flat)
                for (Kick &c : corner)
                    c.rgba = last;
        }
        else
        {
            for (Kick &c : corner)
                c = kicks[0];
            corner[1].x += 1.0f, corner[2].y += 1.0f, corner[3].x += 1.0f, corner[3].y += 1.0f;
        }
        const int order[6] = {0, 1, 2, 1, 2, 3};
        for (int i : order)
            out(corner[i], corner[i].rgba);
    }

    void GifToNative::flush()
    {
        // PS2X_NATIVE_LOG_GIF=1: now and then, a batch's size and first vertex.
        static const bool s_log = std::getenv("PS2X_NATIVE_LOG_GIF") != nullptr;
        static uint32_t s_batches = 0;
        if (s_log && !m_triangles.empty() && ++s_batches % 500u == 1u)
            std::fprintf(stderr, "[native] gif batch %u: %zu vertices, first (%.1f %.1f %.1f) uv (%.3f %.3f) rgba %08x, tex0 %#llx textured %d blend %d\n",
                         s_batches, m_triangles.size(), m_triangles[0].x, m_triangles[0].y, m_triangles[0].z, m_triangles[0].u, m_triangles[0].v,
                         m_triangles[0].rgba, static_cast<unsigned long long>(m_batchMaterial.tex0), m_batchMaterial.textured ? 1 : 0,
                         static_cast<int>(m_batchMaterial.blend));
        if (!m_triangles.empty())
            Renderer::instance().draw(m_batchMaterial, kScreenSpace, m_triangles.data(), static_cast<uint32_t>(m_triangles.size()));
        m_triangles.clear();
        m_batchOpen = false;
    }
}
