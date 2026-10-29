#include "runtime/native/native_vu1_stream.h"

#include "runtime/native/native_vif1.h"
#include "runtime/native/native_vu1_sandbox.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>

namespace ps2x::native
{
    namespace
    {
        // What the adapter decided for each MSCAL and DIRECT as it read the stream (at the transfer's start, on the EE
        // thread), for the gates when VIF1 gets to it, in the same order.
        struct Decision
        {
            uint32_t pc = 0; // an MSCAL's start, or a DIRECT's size in quadwords
            bool skipped = false;
            bool direct = false;
            uint64_t cycles = 0; // a sandboxed program's VU1 cycles
        };
        std::mutex g_decisionsMutex;
        std::deque<Decision> g_decisions;

        void noteDecision(uint32_t pc, bool skipped, bool direct, uint64_t cycles = 0)
        {
            std::lock_guard<std::mutex> lock(g_decisionsMutex);
            g_decisions.push_back({pc, skipped, direct, cycles});
            if (g_decisions.size() > 65536u) // nobody takes them (no gate)
                g_decisions.pop_front();
        }

        bool programGate(const uint8_t *vu1Code, uint32_t startPc)
        {
            (void)vu1Code;
            Decision found;
            {
                std::lock_guard<std::mutex> lock(g_decisionsMutex);
                while (!g_decisions.empty())
                {
                    const Decision decision = g_decisions.front();
                    g_decisions.pop_front();
                    if (!decision.direct && decision.pc == startPc)
                    {
                        found = decision;
                        break;
                    }
                }
            }
            if (found.skipped)
                advanceVif1Clock(found.cycles);
            return found.skipped; // not found: a transfer the adapter did not see, VU1 draws it
        }

        // A DIRECT the VIF1 interpreter does not ask about (cut, or finishing an image) leaves its entry for
        // programGate to drop.
        bool directGate(uint32_t qwords)
        {
            std::lock_guard<std::mutex> lock(g_decisionsMutex);
            while (!g_decisions.empty() && g_decisions.front().direct)
            {
                const Decision decision = g_decisions.front();
                g_decisions.pop_front();
                if (decision.pc == qwords)
                    return decision.skipped;
            }
            return false;
        }

        // A DIRECT packet that only draws (vertex kicks, no image transfer).
        bool drawsOnly(const uint32_t *words, size_t qwords)
        {
            bool draws = false;
            size_t q = 0;
            while (q < qwords)
            {
                const uint64_t lo = words[q * 4] | (static_cast<uint64_t>(words[q * 4 + 1]) << 32);
                const uint64_t regs = words[q * 4 + 2] | (static_cast<uint64_t>(words[q * 4 + 3]) << 32);
                ++q;
                const uint32_t nloop = lo & 0x7fffu, flg = (lo >> 58) & 3u;
                const uint32_t nreg = ((lo >> 60) & 0xfu) ? static_cast<uint32_t>((lo >> 60) & 0xfu) : 16u;
                if (flg >= 2u)
                    return false; // an image transfer: the GS (and the texture cache) must have it
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    const uint32_t reg = (regs >> (4u * r)) & 0xfu;
                    draws = draws || reg == 0x4u || reg == 0x5u;
                }
                if (flg == 0u)
                {
                    for (uint32_t i = 0; i < nloop * nreg && q < qwords; ++i, ++q)
                        if (((regs >> (4u * (i % nreg))) & 0xfu) == 0xeu)
                        {
                            const uint32_t address = words[q * 4 + 2] & 0xffu;
                            if (address == 0x04u || address == 0x05u)
                                draws = true;
                            else if (address >= 0x50u && address <= 0x54u)
                                return false; // a transfer set up (BITBLTBUF, TRXPOS, TRXREG, TRXDIR, HWREG)
                        }
                }
                else
                    q += (static_cast<size_t>(nloop) * nreg + 1u) / 2u;
            }
            return draws;
        }

        // The quadwords of a GIF IMAGE a packet leaves unfinished (0: none), as the VIF1 interpreter reckons them.
        uint32_t pendingImageQwc(const uint32_t *words, size_t qwords)
        {
            size_t q = 0;
            while (q < qwords)
            {
                const uint64_t lo = words[q * 4] | (static_cast<uint64_t>(words[q * 4 + 1]) << 32);
                ++q;
                const uint64_t nloop = lo & 0x7fffu, flg = (lo >> 58) & 3u;
                const uint64_t nreg = ((lo >> 60) & 0xfu) ? ((lo >> 60) & 0xfu) : 16u;
                const uint64_t payload = flg == 0u ? nloop * nreg : flg == 1u ? (nloop * nreg + 1u) / 2u : nloop;
                if (flg >= 2u && payload > qwords - q)
                    return static_cast<uint32_t>(payload - (qwords - q));
                if (payload > qwords - q)
                    return 0u;
                q += payload;
            }
            return 0u;
        }

        // A DIRECT(HL) next (alone, or after a NOP): the pending image goes on in its data.
        bool directCodeFollows(const uint32_t *words, size_t count)
        {
            const auto isDirect = [](uint32_t code) { return ((code >> 24) & 0x7eu) == 0x50u; };
            return (count >= 1u && isDirect(words[0])) || (count >= 2u && words[0] == 0u && isDirect(words[1]));
        }

        void onVif1(const uint8_t *stream, size_t bytes) { Vu1StreamAdapter::instance().observe(stream, bytes); }

        struct Stats
        {
            std::map<uint32_t, uint64_t> sandboxed, port, onVu1; // by start address
            uint64_t directsNative = 0, directsGs = 0;
        };
        Stats g_stats;
        // Read at the first frame's end: a port may set it when it installs the adapter.
        bool logging()
        {
            static const bool s_log = []
            {
                const char *value = std::getenv("PS2X_NATIVE_STREAM_LOG");
                return value && *value == '1';
            }();
            return s_log;
        }
    }

    Vu1StreamAdapter &Vu1StreamAdapter::instance()
    {
        static Vu1StreamAdapter s_adapter;
        return s_adapter;
    }

    Vu1StreamAdapter::Vu1StreamAdapter() = default;

    Vu1StreamAdapter::~Vu1StreamAdapter() { delete m_sandbox; }

    void Vu1StreamAdapter::install(Config config)
    {
        m_config = std::move(config);
        if (!m_sandbox)
            m_sandbox = new Vu1Sandbox();
        m_sandbox->gif().setGsWrites(m_config.gsWrites);
        Renderer &renderer = Renderer::instance();
        renderer.setEnabled(true);
        renderer.beginFrame(m_config.screen);
        // In the GS stream where VU1's output would have been; with renderers keeping GS memory themselves, present()
        // draws over the GS frame, clipped per group.
        renderer.setStreamDraws(true);
        setVif1Observer(&onVif1, m_config.onVif1Thread);
        setVu1Gate(&programGate);
        setDirectGate(&directGate);
        m_installed = true;
    }

    void Vu1StreamAdapter::publish()
    {
        Renderer &renderer = Renderer::instance();
        static uint32_t s_frames = 0;
        if (logging() && ++s_frames % 250u == 0u)
        {
            std::string line;
            for (const auto *map : {&g_stats.sandboxed, &g_stats.port, &g_stats.onVu1})
            {
                line += map == &g_stats.sandboxed ? " sandbox:" : map == &g_stats.port ? " port:" : " VU1:";
                for (const auto &[pc, count] : *map)
                {
                    char item[40];
                    std::snprintf(item, sizeof(item), " %#x x%llu", pc, static_cast<unsigned long long>(count));
                    line += item;
                }
            }
            std::fprintf(stderr, "[native:stream] per 250 frames:%s; DIRECT native %llu, GS %llu\n", line.c_str(),
                         static_cast<unsigned long long>(g_stats.directsNative), static_cast<unsigned long long>(g_stats.directsGs));
            g_stats = Stats{};
        }
        renderer.endFrame();
        renderer.beginFrame(m_config.screen);
    }

    void Vu1StreamAdapter::frameEnd()
    {
        if (!m_installed)
            return;
        if (m_config.onVif1Thread)
            runOnVif1Thread([this] { publish(); });
        else
            publish();
    }

    void Vu1StreamAdapter::observe(const uint8_t *stream, size_t bytes)
    {
        // VIF registers the game set in a transfer the adapter did not see (at boot) come from VIF1's own.
        if (const VIFRegisters *regs = vif1Registers())
        {
            if (!m_sawMask)
                m_vif.mask = regs->mask;
            if (!m_sawRow)
                std::memcpy(m_vif.row, regs->row, sizeof(m_vif.row));
            if (!m_sawCol)
                std::memcpy(m_vif.col, regs->col, sizeof(m_vif.col));
            if (!m_sawMode)
                m_vif.mode = regs->mode;
        }
        run(reinterpret_cast<const uint32_t *>(stream), bytes / 4u);
    }

    void Vu1StreamAdapter::program(uint32_t cmd, uint32_t pc)
    {
        Renderer &renderer = Renderer::instance();
        if (cmd == 0x17u) // MSCNT: continues the program where it ended, where it runs
        {
            // Its draws are a group of their own, at the MSCNT's place in the GS stream (after what the stream sent
            // since the MSCAL: a texture, a CLUT, registers its packets draw with); VIF1 numbers it the same way.
            if (m_running != Running::None)
            {
                renderer.beginGroup(++m_group);
                renderer.setKey(0);
            }
            if (m_running == Running::Sandbox)
                m_sandbox->resume(m_code, sizeof(m_code), reinterpret_cast<uint8_t *>(m_data), sizeof(m_data), m_top, m_itops);
            else if (m_running == Running::Port)
                m_config.draw(*this, m_runningPc, true);
            return;
        }
        Choice choice = Choice::Vu1;
        if (!m_config.active || m_config.active())
            choice = m_config.choose ? m_config.choose(*this, pc) : Choice::Sandbox;
        if (choice == Choice::Port && !m_config.draw)
            choice = Choice::Sandbox;
        m_runningPc = pc;
        if (choice == Choice::Vu1)
        {
            noteDecision(pc, false, false);
            m_running = Running::None;
            ++g_stats.onVu1[pc];
            return;
        }
        // A program VU1 skips: its draws are the next group of the GS stream (the VIF1 interpreter numbers the
        // programs it skips the same way).
        renderer.beginGroup(++m_group);
        renderer.setKey(0);
        if (choice == Choice::Port)
        {
            noteDecision(pc, true, false);
            m_running = Running::Port;
            ++g_stats.port[pc];
            m_config.draw(*this, pc, false);
            return;
        }
        m_running = Running::Sandbox;
        ++g_stats.sandboxed[pc];
        m_sandbox->gif().setSource(pc);
        const uint64_t cyclesBefore = m_sandbox->cycles();
        m_sandbox->call(m_code, sizeof(m_code), reinterpret_cast<uint8_t *>(m_data), sizeof(m_data), pc, m_top, m_itops);
        noteDecision(pc, true, false, m_sandbox->cycles() - cyclesBefore);
    }

    void Vu1StreamAdapter::direct(const uint32_t *words, size_t qwords, size_t size)
    {
        if (m_pendingImageQwc != 0u)
        {
            // The rest of an image: the VIF1 interpreter sends it to the GS without asking the gate, then what follows
            // in the packet as GIF data.
            const size_t image = std::min<size_t>(m_pendingImageQwc, qwords);
            m_pendingImageQwc -= static_cast<uint32_t>(image);
            words += image * 4u;
            qwords -= image;
            if (qwords != 0u)
            {
                m_sandbox->gif().registers(reinterpret_cast<const uint8_t *>(words), qwords * 16u);
                if (const uint32_t pending = pendingImageQwc(words, qwords))
                {
                    m_pendingImageQwc = pending;
                    m_pendingImageRaw = qwords + image < size;
                }
            }
            return;
        }
        const bool active = !m_config.active || m_config.active();
        const bool whole = qwords == size;
        Renderer &renderer = Renderer::instance();
        // The port's own packets first: its group begins before its hook draws.
        if (active && whole && m_config.drawDirect)
        {
            renderer.beginGroup(m_group + 1u);
            renderer.setKey(0);
            if (m_config.drawDirect(*this, words, qwords))
            {
                ++m_group;
                ++g_stats.directsNative;
                noteDecision(static_cast<uint32_t>(size), true, true);
                return;
            }
        }
        const bool native = m_config.directs && active && whole && drawsOnly(words, qwords);
        noteDecision(static_cast<uint32_t>(size), native, true);
        if (native)
        {
            ++g_stats.directsNative;
            renderer.beginGroup(++m_group);
            renderer.setKey(0);
            m_sandbox->gif().setSource(0xd1ec7u);
            m_sandbox->gif().packet(reinterpret_cast<const uint8_t *>(words), qwords * 16u);
            m_sandbox->gif().flush();
        }
        else
        {
            ++g_stats.directsGs;
            // The GS draws it; the sandbox's packets draw with the registers it sets.
            m_sandbox->gif().registers(reinterpret_cast<const uint8_t *>(words), qwords * 16u);
            if (const uint32_t pending = pendingImageQwc(words, qwords))
            {
                m_pendingImageQwc = pending;
                m_pendingImageRaw = !whole;
            }
        }
    }

    void Vu1StreamAdapter::run(const uint32_t *words, size_t count)
    {
        size_t i = 0;
        while (i < count)
        {
            if (m_pendingImageQwc != 0u && (m_pendingImageRaw || !directCodeFollows(words + i, count - i)))
            {
                // Image quadwords straight in the stream (no VIFcode before them): the GS's, as VIF1 takes them.
                const size_t image = std::min<size_t>(m_pendingImageQwc, (count - i) / 4u);
                if (image == 0u)
                    break;
                m_pendingImageQwc -= static_cast<uint32_t>(image);
                i += image * 4u;
                continue;
            }
            const uint32_t code = words[i++];
            const uint32_t cmd = (code >> 24) & 0x7fu, num = (code >> 16) & 0xffu, imm = code & 0xffffu;
            if ((cmd & 0x60u) == 0x60u) // UNPACK, as VIF1 does it (masks, row / column, add modes)
            {
                m_vif.cycle = m_cl | (m_wl << 8);
                m_vif.tops = m_tops;
                const uint32_t used = vifUnpack(m_vif, reinterpret_cast<uint8_t *>(m_data), sizeof(m_data), code,
                                                reinterpret_cast<const uint8_t *>(words + i), static_cast<uint32_t>((count - i) * 4u));
                i += (used + 3u) / 4u;
                continue;
            }
            switch (cmd)
            {
            case 0x01: // STCYCL
                m_cl = imm & 0xffu;
                m_wl = (imm >> 8) & 0xffu;
                if (m_wl == 0)
                    m_wl = 1;
                break;
            case 0x02: // OFFSET
                m_offset = imm & 0x3ffu;
                m_tops = m_base;
                break;
            case 0x03: // BASE
                m_base = imm & 0x3ffu;
                m_tops = m_base;
                break;
            case 0x04: // ITOP
                m_itops = imm & 0x3ffu;
                break;
            case 0x05: // STMOD
                m_vif.mode = imm & 3u;
                m_sawMode = true;
                break;
            case 0x14: // MSCAL
            case 0x15: // MSCALF
            case 0x17: // MSCNT
                // TOP is TOPS at the start; the double buffer then flips.
                m_top = m_tops;
                m_tops = m_tops == m_base ? m_base + m_offset : m_base;
                program(cmd, imm * 8u);
                break;
            case 0x20: // STMASK
                m_sawMask = true;
                if (i < count)
                    m_vif.mask = words[i];
                i += 1;
                break;
            case 0x30: // STROW
            case 0x31: // STCOL
                (cmd == 0x30 ? m_sawRow : m_sawCol) = true;
                if (i + 4u <= count)
                    std::memcpy(cmd == 0x30 ? m_vif.row : m_vif.col, words + i, 16);
                i += 4;
                break;
            case 0x4a: // MPG
            {
                const size_t size = static_cast<size_t>(num ? num : 256u) * 2u;
                const size_t at = static_cast<size_t>(imm) * 8u, bytes = std::min(size, count - i) * 4u;
                if (at < sizeof(m_code))
                    std::memcpy(m_code + at, words + i, std::min(bytes, sizeof(m_code) - at));
                m_sandbox->codeChanged();
                ++m_codeVersion;
                i += size;
                break;
            }
            case 0x50: // DIRECT
            case 0x51: // DIRECTHL
            {
                const size_t size = static_cast<size_t>(imm ? imm : 65536u);
                const size_t have = std::min(size * 4u, count - i) / 4u;
                direct(words + i, have, size);
                i += size * 4u;
                break;
            }
            default:
                break;
            }
        }
    }
}
