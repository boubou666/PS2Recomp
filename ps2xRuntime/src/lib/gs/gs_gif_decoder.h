#pragma once

// GIF decoding without rendering, for renderers that take the raw GIF stream: GIF tags (PACKED, REGLIST, IMAGE,
// FLG 3 as IMAGE), A+D, the general register file, PRMODE / PRMODECONT and the vertex queue with its kicks. The
// state it builds is a GSStateImage (no CLUT buffer: loading the CLUT is the renderer's job). It decodes the way
// the frontend's own decoder (GS::processGIFPacket) does, which ps2_gs_gif_decoder tests check packet by packet.
#include "runtime/gs/gs_state_image.h"

#include <array>
#include <cstdint>

class GSGifDecoder
{
public:
    struct Primitive
    {
        uint64_t prim = 0; // PRIM as drawn: PRMODE's attributes when PRMODECONT.AC is 0, PRIM's type
        uint32_t vertexCount = 0;
        std::array<GSStateImage::Vertex, 3> vertices{};
    };

    class Sink
    {
    public:
        virtual ~Sink() = default;
        // A drawing kick completed a primitive; the decoder's state is the one it is drawn with.
        virtual void primitive(const GSGifDecoder &, const Primitive &) {}
        // An A+D or REGLIST register write, after it took effect (also vertex kicks, SIGNAL, FINISH, TEXFLUSH...).
        virtual void registerWritten(const GSGifDecoder &, uint8_t, uint64_t) {}
        // IMAGE data or a HWREG write, for the transfer TRXDIR started.
        virtual void imageData(const GSGifDecoder &, const uint8_t *, uint32_t) {}
    };

    explicit GSGifDecoder(Sink &sink);

    // A whole packet: tags and their data. Data missing at the end of the packet is dropped with its tag.
    void processPacket(const uint8_t *data, uint32_t sizeBytes);
    void writeRegister(uint8_t address, uint64_t value);

    const GSStateImage &state() const { return m_state; }
    void loadState(const GSStateImage &image);
    void reset();
    // PRIM as drawn now.
    uint64_t effectivePrim() const;

private:
    void writePacked(uint8_t descriptor, uint64_t lo, uint64_t hi);
    void kick(uint64_t xyz, uint8_t fog, bool drawing);
    void setQ(uint32_t qBits);

    Sink &m_sink;
    GSStateImage m_state;
};
