// Based on Blackline Interactive implementation
#include "ThreadNaming.h"
#include "runtime/ps2_memory.h"
#include "runtime/vif1_command_stats.h"
#include <atomic>
#include <chrono>
#include <cstring>

namespace ps2_vif1_command_stats
{
    namespace
    {
        std::atomic<uint64_t> g_commands[0x80]{};
    }

    void noteCommand(uint8_t opcode)
    {
        g_commands[opcode & 0x7Fu].fetch_add(1u, std::memory_order_relaxed);
    }

    uint64_t commands(uint8_t opcode)
    {
        return g_commands[opcode & 0x7Fu].load(std::memory_order_relaxed);
    }
}

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
    constexpr uint8_t kGifFmtImage = 2u;

    uint32_t pendingGifImageQwc(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u)
            return 0u;

        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            uint64_t tagLo = 0u;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            offset += 16u;

            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t payloadBytes = 0u;
            if (flg == 0u) // PACKED
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 16ull;
            }
            else if (flg == 1u) // REGLIST, padded to a quadword
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 8ull;
                payloadBytes = (payloadBytes + 15ull) & ~15ull;
            }
            else // IMAGE, or FLG 3 which behaves as IMAGE
            {
                payloadBytes = static_cast<uint64_t>(nloop) * 16ull;
                const uint64_t availableBytes = sizeBytes - offset;
                if (payloadBytes > availableBytes)
                {
                    return static_cast<uint32_t>((payloadBytes - availableBytes) / 16ull);
                }
            }

            if (payloadBytes > static_cast<uint64_t>(sizeBytes - offset))
                return 0u;
            offset += static_cast<uint32_t>(payloadBytes);
        }

        return 0u;
    }

    // Decodes one UNPACK source vector (no mask, no row/col handling) into `out`; false for formats
    // without a decoding (V2-5, V3-5).
    inline bool decodeUnpackVector(uint8_t vl, uint8_t vn, int components, bool zeroExtend, const uint8_t *srcVec,
                                   const uint8_t *payloadEnd, uint32_t out[4])
    {
        auto extend16 = [zeroExtend](uint16_t raw) -> uint32_t
        {
            if (zeroExtend)
                return static_cast<uint32_t>(raw);
            return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
        };

        auto extend8 = [zeroExtend](uint8_t raw) -> uint32_t
        {
            if (zeroExtend)
                return static_cast<uint32_t>(raw);
            return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
        };

        if (vl == 0u)
        {
            if (components == 1)
            {
                uint32_t scalar = 0;
                std::memcpy(&scalar, srcVec, sizeof(scalar));
                out[0] = scalar;
                out[1] = scalar;
                out[2] = scalar;
                out[3] = scalar;
            }
            else
            {
                const uint32_t limit = components == 3 ? 4u : static_cast<uint32_t>(components);
                for (uint32_t c = 0; c < limit; ++c)
                {
                    uint32_t scalar = 0;
                    if (srcVec + c * 4u + 4u <= payloadEnd)
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                    out[c] = scalar;
                }
                if (components == 2)
                {
                    out[2] = out[0];
                    out[3] = out[1];
                }
            }
        }
        else if (vl == 1u)
        {
            if (components == 1)
            {
                uint16_t raw = 0;
                std::memcpy(&raw, srcVec, sizeof(raw));
                const uint32_t scalar = extend16(raw);
                out[0] = scalar;
                out[1] = scalar;
                out[2] = scalar;
                out[3] = scalar;
            }
            else
            {
                const uint32_t limit = components == 3 ? 4u : static_cast<uint32_t>(components);
                for (uint32_t c = 0; c < limit; ++c)
                {
                    uint16_t raw = 0;
                    if (srcVec + c * 2u + 2u <= payloadEnd)
                        std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                    out[c] = extend16(raw);
                }
                if (components == 2)
                {
                    out[2] = out[0];
                    out[3] = out[1];
                }
            }
        }
        else if (vl == 2u)
        {
            if (components == 1)
            {
                const uint32_t scalar = extend8(srcVec[0]);
                out[0] = scalar;
                out[1] = scalar;
                out[2] = scalar;
                out[3] = scalar;
            }
            else
            {
                const uint32_t limit = components == 3 ? 4u : static_cast<uint32_t>(components);
                for (uint32_t c = 0; c < limit; ++c)
                {
                    out[c] = srcVec + c < payloadEnd ? extend8(srcVec[c]) : 0u;
                }
                if (components == 2)
                {
                    out[2] = out[0];
                    out[3] = out[1];
                }
            }
        }
        else if (vl == 3u && vn == 3u)
        {
            // V4-5: packed color-like format in a single 16-bit value.
            uint16_t packed = 0;
            std::memcpy(&packed, srcVec, sizeof(packed));
            out[0] = (packed & 0x001Fu) << 3;
            out[1] = (packed & 0x03E0u) >> 2;
            out[2] = (packed & 0x7C00u) >> 7;
            out[3] = (packed & 0x8000u) >> 8;
        }
        else
        {
            return false;
        }
        return true;
    }

    // Executes the UNPACK VIFcode `cmd` (VIF0 or VIF1) into `vuData`, reading its payload from `payload`
    // (`available` bytes left in the stream). Returns the payload size in bytes, rounded up to a word.
    uint32_t executeVifUnpack(VIFRegisters &regs, uint8_t *vuData, uint32_t vuDataSize, uint32_t addrMask, uint32_t cmd,
                              const uint8_t *payload, uint32_t available)
    {
        const uint8_t opcode = (cmd >> 24) & 0x7F;
        const uint16_t imm = cmd & 0xFFFF;
        const uint8_t num = (cmd >> 16) & 0xFF;
        uint8_t vn = (opcode >> 2) & 0x3;
        uint8_t vl = opcode & 0x3;
        const bool maskEnable = (opcode & 0x10u) != 0u;
        int components = vn + 1;
        int bitsPerComponent = 32;
        switch (vl)
        {
        case 0:
            bitsPerComponent = 32;
            break;
        case 1:
            bitsPerComponent = 16;
            break;
        case 2:
            bitsPerComponent = 8;
            break;
        case 3:
            bitsPerComponent = (vn == 3) ? 4 : 16;
            break;
        default:
            break;
        }
        int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
        uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
        // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
        const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

        // STCYCL controls write cycles for UNPACK.
        uint32_t cl = regs.cycle & 0xFFu;
        uint32_t wl = (regs.cycle >> 8) & 0xFFu;
        if (cl == 0u)
            cl = 1u;
        if (wl == 0u)
            wl = 1u;

        uint32_t sourceVectorCount = writeVectorCount;
        if (cl < wl)
        {
            const uint32_t fullBlocks = writeVectorCount / wl;
            uint32_t remainder = writeVectorCount % wl;
            if (remainder > cl)
                remainder = cl;
            sourceVectorCount = fullBlocks * cl + remainder;
        }

        uint32_t totalBytes = sourceVectorCount * bytesPerVector;
        totalBytes = (totalBytes + 3) & ~3u;

        uint32_t vuAddr = (uint32_t)imm & addrMask;
        if ((imm & 0x8000u) != 0u)
            vuAddr = (vuAddr + (regs.tops & 0x3FFu)) & addrMask;

        const bool zeroExtend = (imm & 0x4000u) != 0u;

        if (vuData && totalBytes > 0 && totalBytes <= available)
        {
            const uint8_t *srcBase = payload;
            const uint8_t *payloadEnd = payload + available;
            const uint32_t addMode = regs.mode & 3u;
            if (!maskEnable && addMode != 1u && addMode != 2u && cl >= wl && !(vl == 3u && vn != 3u))
            {
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t destOff = ((vuAddr + (writeIndex / wl) * cl + writeIndex % wl) & addrMask) * 16u;
                    if (destOff + 16u > vuDataSize)
                        continue;
                    uint32_t lanes[4];
                    decodeUnpackVector(vl, vn, components, zeroExtend, srcBase + writeIndex * bytesPerVector, payloadEnd, lanes);
                    std::memcpy(vuData + destOff, lanes, sizeof(lanes));
                }
                return totalBytes;
            }
            uint32_t srcIndex = 0u;
            for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
            {
                const uint32_t cyclePos = writeIndex % wl;
                const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                uint32_t destVec = 0;
                if (cl >= wl)
                {
                    destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & addrMask;
                }
                else
                {
                    destVec = (vuAddr + writeIndex) & addrMask;
                }

                uint32_t destOff = destVec * 16u;
                if (destOff + 16u > vuDataSize)
                {
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                        ++srcIndex;
                    continue;
                }

                uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                std::memcpy(lanes, vuData + destOff, sizeof(lanes));
                uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                bool decoded = false;

                const uint8_t *srcVec = nullptr;
                if (sourceAvailable && srcIndex < sourceVectorCount)
                {
                    srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    decoded = true;
                }

                const bool handledFormat =
                    decoded && decodeUnpackVector(vl, vn, components, zeroExtend, srcVec, payloadEnd, decompressed);

                // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                if (!handledFormat && decoded && !maskEnable && (regs.mode == 0u || regs.mode == 3u))
                {
                    uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                    std::memcpy(vuData + destOff, srcVec, copyBytes);
                    continue;
                }

                const bool canAdd = (vl != 3u || vn != 3u);
                const uint32_t mode = regs.mode & 3u;
                const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                for (uint32_t field = 0u; field < 4u; ++field)
                {
                    uint32_t maskSpec = 0u;
                    if (maskEnable)
                    {
                        const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                        maskSpec = (regs.mask >> shift) & 0x3u;
                    }

                    // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                    if (!decoded && maskSpec == 0u)
                        maskSpec = 1u;

                    uint32_t writeVal = lanes[field];
                    if (maskSpec == 0u)
                    {
                        if (handledFormat)
                        {
                            writeVal = decompressed[field];
                            if (canAdd && (mode == 1u || mode == 2u))
                            {
                                writeVal = writeVal + regs.row[field];
                                if (mode == 2u)
                                    regs.row[field] = writeVal;
                            }
                        }
                    }
                    else if (maskSpec == 1u)
                    {
                        writeVal = regs.row[field];
                    }
                    else if (maskSpec == 2u)
                    {
                        writeVal = regs.col[colIdx];
                    }
                    else
                    {
                        continue; // write-protect
                    }

                    lanes[field] = writeVal;
                }

                std::memcpy(vuData + destOff, lanes, sizeof(lanes));
            }
        }
        return totalBytes;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (m_vu0Code && destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu0Code + destAddr, data + pos, copyBytes);
                    markVU0CodeModified();
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF || opcode == VIF_MSCNT)
        {
            if (m_vu0MscalCallback)
                m_vu0MscalCallback(opcode == VIF_MSCNT ? ~0u : static_cast<uint32_t>(imm) * 8u);
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            pos += executeVifUnpack(vif0_regs, m_vu0Data, PS2_VU0_DATA_SIZE, 0xFFu, cmd, data + pos, sizeBytes - pos);
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

uint32_t PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return 0u;

    uint32_t pos = 0;

    while (pos + 4 <= sizeBytes)
    {
        if (m_vif1Direct.qwLeft != 0u)
        {
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
                break;
            const uint32_t bytes = std::min<uint32_t>(m_vif1Direct.qwLeft, availableQw) * 16u;
            const uint32_t stop = m_vif1Segments
                                      ? findSignalStall(m_vif1Direct.scan, data + pos, bytes, 0x10009000u, *m_vif1Segments, pos)
                                      : bytes;
            if (stop != 0u)
                submitGifPacket(GifPathId::Path2, data + pos, stop, true, m_vif1Direct.directHl, true);
            m_vif1Direct.qwLeft -= stop / 16u;
            pos += stop;
            if (stop < bytes)
            {
                m_vif1StoppedAtSignal = true;
                return pos;
            }
            continue;
        }

        if (m_vif1PendingPath2ImageQwc != 0u)
        {
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
            {
                break;
            }

            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, availableQw);
            std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
            const uint64_t imageTag =
                static_cast<uint64_t>(chunkQw & 0x7FFFu) |
                ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
                (static_cast<uint64_t>(kGifFmtImage) << 58);
            std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
            std::memcpy(imagePacket.data() + 16u, data + pos, static_cast<size_t>(chunkQw) * 16u);
            submitGifPacket(GifPathId::Path2, imagePacket.data(), static_cast<uint32_t>(imagePacket.size()), true, m_vif1PendingPath2DirectHl);

            pos += chunkQw * 16u;
            m_vif1PendingPath2ImageQwc -= chunkQw;
            if (m_vif1PendingPath2ImageQwc == 0u)
            {
                m_vif1PendingPath2DirectHl = false;
            }
            continue;
        }

        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        ps2_vif1_command_stats::noteCommand(opcode);
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            uint32_t startPC = (uint32_t)imm * 8u;

            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscalCallback)
                m_vu1MscalCallback(startPC, runTop, runItop);
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscntCallback)
                m_vu1MscntCallback(runTop, runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            uint32_t destAddr = (uint32_t)imm * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
                    markVU1CodeModified();
                }
            }
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            if (qwCount > 0)
            {
                const bool directHl = (opcode == VIF_DIRECTHL);
                if (m_vif1Segments)
                {
                    GifSignalScan scan;
                    const uint32_t stop = findSignalStall(scan, data + pos, qwCount * 16u, 0x10009000u, *m_vif1Segments, pos);
                    if (stop < qwCount * 16u)
                    {
                        if (stop != 0u)
                            submitGifPacket(GifPathId::Path2, data + pos, stop, true, directHl, true);
                        m_vif1Direct.qwLeft = qwCount - stop / 16u;
                        m_vif1Direct.directHl = directHl;
                        m_vif1Direct.scan = scan;
                        m_vif1StoppedAtSignal = true;
                        return pos + stop;
                    }
                }
                submitGifPacket(GifPathId::Path2, data + pos, qwCount * 16, true, directHl);

                const uint32_t pendingImageQw = pendingGifImageQwc(data + pos, qwCount * 16u);
                if (pendingImageQw != 0u)
                {
                    m_vif1PendingPath2ImageQwc = pendingImageQw;
                    m_vif1PendingPath2DirectHl = directHl;
                }
            }

            pos += qwCount * 16;
            if (truncated)
            {
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            pos += executeVifUnpack(vif1_regs, m_vu1Data, PS2_VU1_DATA_SIZE, 0x3FFu, cmd, data + pos, sizeBytes - pos);
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
    return std::min(pos, sizeBytes);
}

uint64_t ps2CurrentEeCycle();

namespace
{
    thread_local bool t_onVif1Worker = false;
}

void PS2Memory::setVif1Threaded(bool enabled)
{
    if (enabled == m_vif1Threaded)
        return;
    if (!enabled)
    {
        stopVif1Worker();
        return;
    }
    m_vif1Stop = false;
    m_vif1Threaded = true;
    m_vif1Thread = std::thread([this]() { vif1WorkerMain(); });
}

bool PS2Memory::onVif1Worker() const
{
    return t_onVif1Worker;
}

void PS2Memory::waitForVif1() const
{
    if (!m_vif1Threaded || t_onVif1Worker)
        return;
    const uint64_t queued = m_vif1Queued.load(std::memory_order_acquire);
    waitForVif1Until(queued);
}

void PS2Memory::markVif1Frame()
{
    m_vif1FrameMark = m_vif1Queued.load(std::memory_order_acquire);
    ++m_vif1FramesMarked;
}

void PS2Memory::waitForVif1PreviousFrame()
{
    if (!m_vif1Threaded || t_onVif1Worker)
        return;
    waitForVif1Until(m_vif1FrameMark);
    if (!m_presentedFramesProvider || m_vif1FramesMarked < 2u)
        return;
    // The GS thread presents the frames in order; give up after a second rather than hang if it stopped.
    const uint64_t target = m_vif1FramesMarked - 1u;
    const auto waitStart = std::chrono::steady_clock::now();
    const auto deadline = waitStart + std::chrono::seconds(1);
    for (uint32_t spin = 0; m_presentedFramesProvider() < target; ++spin)
    {
        if (spin < 64u)
            std::this_thread::yield();
        else if (std::chrono::steady_clock::now() > deadline)
            break;
        else
            std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    m_vif1WaitNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - waitStart).count()),
                           std::memory_order_relaxed);
}

void PS2Memory::waitForVif1Until(uint64_t queued) const
{
    if (m_vif1Done.load(std::memory_order_acquire) >= queued)
        return;

    const auto waitStart = std::chrono::steady_clock::now();
    const auto noteWait = [&]()
    {
        m_vif1WaitNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - waitStart).count()),
                               std::memory_order_relaxed);
    };
    // Most waits are short (the frame's last transfers): spin a little before sleeping.
    for (uint32_t spin = 0; spin < 4096u; ++spin)
    {
        if (m_vif1Done.load(std::memory_order_acquire) >= queued)
        {
            noteWait();
            return;
        }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_pause();
#endif
    }
    std::unique_lock<std::mutex> lock(m_vif1Mutex);
    m_vif1IdleCv.wait(lock, [&]() { return m_vif1Done.load(std::memory_order_acquire) >= queued; });
    noteWait();
}

void PS2Memory::advanceVif1WorkerClock(uint64_t cycles)
{
    const uint64_t cycle = m_vif1WorkerCycle.load(std::memory_order_relaxed) + cycles;
    m_vif1WorkerCycle.store(cycle, std::memory_order_release);
    uint64_t horizon = m_vif1Horizon.load(std::memory_order_relaxed);
    while (horizon != ~0ull && horizon < cycle &&
           !m_vif1Horizon.compare_exchange_weak(horizon, cycle, std::memory_order_acq_rel))
    {
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (m_vif1ClockWaiter.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(m_vif1Mutex);
        m_vif1IdleCv.notify_all();
    }
}

void PS2Memory::waitForVif1Clock(uint64_t eeCycle) const
{
    if (!vif1BehindEe(eeCycle))
        return;
    const auto waitStart = std::chrono::steady_clock::now();
    for (uint32_t spin = 0; vif1BehindEe(eeCycle); ++spin)
    {
        if (spin < 256u)
        {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
#endif
            continue;
        }
        m_vif1ClockWaiter.store(true, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::unique_lock<std::mutex> lock(m_vif1Mutex);
        m_vif1IdleCv.wait_for(lock, std::chrono::milliseconds(1), [&]() { return !vif1BehindEe(eeCycle); });
    }
    m_vif1ClockWaiter.store(false, std::memory_order_relaxed);
    m_vif1WaitNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - waitStart).count()),
                           std::memory_order_relaxed);
}

void PS2Memory::queueVif1Job(std::vector<uint8_t> data)
{
    if (data.empty())
        return;
    Vif1Job job;
    job.data = std::move(data);
    queueVif1Job(std::move(job));
}

void PS2Memory::queueVif1Callback(std::function<void()> callback)
{
    Vif1Job job;
    job.kind = Vif1Job::Kind::Callback;
    job.callback = std::move(callback);
    queueVif1Job(std::move(job));
}

void PS2Memory::queueVif1Job(Vif1Job job)
{
    job.vu0Fbrst = m_vu0FbrstProvider ? m_vu0FbrstProvider() : 0u;
    job.queueCycle = ps2CurrentEeCycle();
    {
        std::lock_guard<std::mutex> lock(m_vif1Mutex);
        if (gsLag() && m_vif1Horizon.load(std::memory_order_relaxed) == ~0ull)
            m_vif1Horizon.store(std::max(m_vif1WorkerCycle.load(std::memory_order_relaxed), job.queueCycle),
                                std::memory_order_release);
        m_vif1Jobs.push_back(std::move(job));
        m_vif1Queued.fetch_add(1u, std::memory_order_release);
    }
    m_vif1WorkCv.notify_one();
}

void PS2Memory::vif1WorkerMain()
{
    ThreadNaming::SetCurrentThreadName("vif1");
    t_onVif1Worker = true;
    for (;;)
    {
        Vif1Job job;
        {
            std::unique_lock<std::mutex> lock(m_vif1Mutex);
            m_vif1WorkCv.wait(lock, [&]() { return m_vif1Stop || !m_vif1Jobs.empty(); });
            if (m_vif1Jobs.empty())
                return;
            job = std::move(m_vif1Jobs.front());
            m_vif1Jobs.pop_front();
            if (gsLag())
            {
                const uint64_t start = std::max(m_vif1WorkerCycle.load(std::memory_order_relaxed), job.queueCycle);
                m_vif1WorkerCycle.store(start, std::memory_order_release);
                m_vif1Horizon.store(start, std::memory_order_release);
            }
        }
        m_vif1JobVu0Fbrst = job.vu0Fbrst;
        switch (job.kind)
        {
        case Vif1Job::Kind::Vif1:
            processVIF1Data(job.data.data(), static_cast<uint32_t>(job.data.size()));
            break;
        case Vif1Job::Kind::Path3:
            submitGifPacket(GifPathId::Path3, job.data.data(), static_cast<uint32_t>(job.data.size()), false, false, true);
            break;
        case Vif1Job::Kind::PrivWrite32:
            write32(job.address, static_cast<uint32_t>(job.value));
            break;
        case Vif1Job::Kind::PrivWrite64:
            write64(job.address, job.value);
            break;
        case Vif1Job::Kind::Callback:
            job.callback();
            break;
        }
        if (m_gifArbiter)
            m_gifArbiter->drain();
        {
            std::lock_guard<std::mutex> lock(m_vif1Mutex);
            m_vif1Done.fetch_add(1u, std::memory_order_release);
            if (m_vif1Jobs.empty())
                m_vif1Horizon.store(~0ull, std::memory_order_release);
        }
        m_vif1IdleCv.notify_all();
    }
}

void PS2Memory::stopVif1Worker()
{
    if (!m_vif1Threaded)
        return;
    {
        std::lock_guard<std::mutex> lock(m_vif1Mutex);
        m_vif1Stop = true;
    }
    m_vif1WorkCv.notify_all();
    if (m_vif1Thread.joinable())
        m_vif1Thread.join();
    m_vif1Threaded = false;
}
