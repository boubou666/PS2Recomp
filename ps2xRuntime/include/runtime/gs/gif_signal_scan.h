#ifndef PS2_GIF_SIGNAL_SCAN_H
#define PS2_GIF_SIGNAL_SCAN_H

#include <cstdint>
#include <cstring>

struct GifSignalScan
{
    uint64_t regs = 0;
    uint32_t nreg = 0;
    uint32_t reg = 0;
    uint32_t packedQw = 0;
    uint32_t skipQw = 0;

    bool inTag() const { return packedQw != 0u || skipQw != 0u; }

    uint32_t nextSignal(const uint8_t *data, uint32_t size, uint32_t offset)
    {
        constexpr uint64_t kAddressData = 0xEu;
        constexpr uint64_t kSignal = 0x60u;
        while (offset + 16u <= size)
        {
            if (packedQw != 0u)
            {
                uint64_t hi = 0;
                std::memcpy(&hi, data + offset + 8u, sizeof(hi));
                if (((regs >> (reg * 4u)) & 0xFu) == kAddressData && (hi & 0xFFu) == kSignal)
                    return offset;
                passQuadword();
                offset += 16u;
                continue;
            }
            if (skipQw != 0u)
            {
                const uint32_t available = (size - offset) / 16u;
                const uint32_t skipped = skipQw < available ? skipQw : available;
                skipQw -= skipped;
                offset += skipped * 16u;
                continue;
            }
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            std::memcpy(&regs, data + offset + 8u, sizeof(regs));
            offset += 16u;
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            reg = 0u;
            if (flg == 0u)
                packedQw = nloop * nreg;
            else if (flg == 1u)
                skipQw = (nloop * nreg + 1u) / 2u;
            else
                skipQw = nloop;
        }
        return size;
    }

    void passQuadword()
    {
        --packedQw;
        if (++reg == nreg)
            reg = 0u;
    }
};

#endif
