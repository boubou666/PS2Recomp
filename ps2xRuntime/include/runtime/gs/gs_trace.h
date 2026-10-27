#ifndef PS2_GS_TRACE_H
#define PS2_GS_TRACE_H

// GS command streams recorded in a game with PS2X_GS_RECORD and replayed by the ps2_gs_replay tool, to time a
// renderer, compare two renderers and check that a change leaves VRAM bit for bit the same.
//   GSR2 header: magic, VRAM size, VRAM, state image size, state image (serializeGsStateImage)
//   GSR1 header: magic, VRAM size, VRAM, drawing state size, drawing state (GS::saveDrawingState, CPU path only)
//   then commands: kind, four registers, payload size, payload
// Little endian, as the hosts that record and replay them.
#include "runtime/gs/gs_state_image.h"

#include <cstdint>
#include <cstdio>
#include <vector>

struct GsTraceCommand
{
    enum Kind : uint8_t
    {
        GifPacket = 0,
        NativePackedGifPacket = 1,
        NativeImageUpload = 2, // registers: BITBLTBUF, TRXPOS, TRXREG, TRXDIR
        PresentationLatch = 3, // payload: kPresentationLatchQwords display registers (GSR2 only)
        Register = 4,          // registers[0] address, registers[1] value (GSR2 only)
    };
    // PMODE, SMODE1, SMODE2, DISPFB1, DISPLAY1, DISPFB2, DISPLAY2, BGCOLOR, vsync tick.
    static constexpr uint32_t kPresentationLatchQwords = 9u;
    uint8_t kind = GifPacket;
    uint64_t registers[4]{};
    std::vector<uint8_t> bytes;
};

struct GsTraceHeader
{
    static constexpr uint32_t kMagicV1 = 0x31525347u; // "GSR1"
    static constexpr uint32_t kMagicV2 = 0x32525347u; // "GSR2"
    uint32_t version = 2;
    std::vector<uint8_t> vram;
    GSStateImage image;               // GSR2
    std::vector<uint8_t> legacyState; // GSR1
};

inline void writeGsTraceHeader(FILE *out, const std::vector<uint8_t> &vram, const GSStateImage &image)
{
    const uint32_t magic = GsTraceHeader::kMagicV2;
    const uint32_t vramSize = static_cast<uint32_t>(vram.size());
    const std::vector<uint8_t> state = serializeGsStateImage(image);
    const uint32_t stateSize = static_cast<uint32_t>(state.size());
    std::fwrite(&magic, 4, 1, out);
    std::fwrite(&vramSize, 4, 1, out);
    std::fwrite(vram.data(), 1, vram.size(), out);
    std::fwrite(&stateSize, 4, 1, out);
    std::fwrite(state.data(), 1, state.size(), out);
}

inline void writeGsTraceCommand(FILE *out, uint8_t kind, const uint64_t registers[4], const uint8_t *data, uint32_t size)
{
    std::fwrite(&kind, 1, 1, out);
    std::fwrite(registers, 8, 4, out);
    std::fwrite(&size, 4, 1, out);
    if (size != 0u)
        std::fwrite(data, 1, size, out);
}

inline bool readGsTraceHeader(FILE *in, GsTraceHeader &header)
{
    uint32_t magic = 0, vramSize = 0, stateSize = 0;
    if (std::fread(&magic, 4, 1, in) != 1 || (magic != GsTraceHeader::kMagicV1 && magic != GsTraceHeader::kMagicV2) ||
        std::fread(&vramSize, 4, 1, in) != 1)
        return false;
    header.version = magic == GsTraceHeader::kMagicV1 ? 1u : 2u;
    header.vram.resize(vramSize);
    if (std::fread(header.vram.data(), 1, vramSize, in) != vramSize || std::fread(&stateSize, 4, 1, in) != 1)
        return false;
    std::vector<uint8_t> state(stateSize);
    if (std::fread(state.data(), 1, stateSize, in) != stateSize)
        return false;
    if (header.version == 1u)
    {
        header.legacyState = std::move(state);
        return true;
    }
    return deserializeGsStateImage(state.data(), state.size(), header.image);
}

inline bool readGsTraceCommand(FILE *in, GsTraceCommand &command)
{
    uint32_t size = 0;
    if (std::fread(&command.kind, 1, 1, in) != 1 || std::fread(command.registers, 8, 4, in) != 4 ||
        std::fread(&size, 4, 1, in) != 1)
        return false;
    command.bytes.resize(size);
    return size == 0u || std::fread(command.bytes.data(), 1, size, in) == size;
}

#endif
