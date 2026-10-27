#include "gs_gif_decoder.h"

#include "runtime/gs/gs_types.h"

#include <cstring>

namespace
{
    constexpr uint32_t kOneBits = 0x3F800000u; // 1.0f

    uint64_t load64(const uint8_t *p)
    {
        uint64_t value = 0;
        std::memcpy(&value, p, sizeof(value));
        return value;
    }

    uint32_t verticesNeeded(uint64_t prim)
    {
        switch (prim & 0x7u)
        {
        case GS_PRIM_POINT:
            return 1u;
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
        case GS_PRIM_SPRITE:
            return 2u;
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
            return 3u;
        default:
            return 0u;
        }
    }
}

GSGifDecoder::GSGifDecoder(Sink &sink) : m_sink(sink)
{
    reset();
}

void GSGifDecoder::reset()
{
    m_state = GSStateImage{};
    m_state.flags = GSStateImage::HasVertexQueue;
    m_state.setReg(GS_REG_PRMODECONT, 1u);
    m_state.setReg(GS_REG_RGBAQ, 0x80808080ull | (uint64_t(kOneBits) << 32));
    m_state.setReg(GS_REG_COLCLAMP, 1u);
    m_state.setReg(GS_REG_TRXDIR, 3u);
    for (uint8_t i = 0; i < 2u; ++i)
    {
        m_state.setReg(GS_REG_FRAME_1 + i, 10ull << 16);
        m_state.setReg(GS_REG_SCISSOR_1 + i, (639ull << 16) | (447ull << 48));
    }
}

void GSGifDecoder::loadState(const GSStateImage &image)
{
    m_state = image;
    m_state.flags = GSStateImage::HasVertexQueue;
    if ((image.flags & GSStateImage::HasVertexQueue) == 0u)
        m_state.vertexCount = 0u;
}

uint64_t GSGifDecoder::effectivePrim() const
{
    const uint64_t prim = m_state.reg(GS_REG_PRIM);
    if ((m_state.reg(GS_REG_PRMODECONT) & 1u) != 0u)
        return prim & 0x7FFu;
    return (m_state.reg(GS_REG_PRMODE) & 0x7F8u) | (prim & 0x7u);
}

void GSGifDecoder::setQ(uint32_t qBits)
{
    const uint64_t rgbaq = m_state.reg(GS_REG_RGBAQ);
    m_state.setReg(GS_REG_RGBAQ, (rgbaq & 0xFFFFFFFFull) | (uint64_t(qBits) << 32));
}

void GSGifDecoder::processPacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data)
        return;
    uint32_t offset = 0;
    while (offset + 16u <= sizeBytes)
    {
        const uint64_t tagLo = load64(data + offset);
        const uint64_t tagHi = load64(data + offset + 8u);
        offset += 16u;
        setQ(kOneBits);

        const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
        if (nreg == 0u)
            nreg = 16u;
        if (((tagLo >> 46) & 1u) != 0u)
            writeRegister(GS_REG_PRIM, (tagLo >> 47) & 0x7FFu);

        uint8_t regs[16];
        for (uint32_t i = 0; i < nreg; ++i)
            regs[i] = static_cast<uint8_t>((tagHi >> (i * 4u)) & 0xFu);

        if (flg == GIF_FMT_PACKED)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 16u > sizeBytes)
                        return;
                    writePacked(regs[r], load64(data + offset), load64(data + offset + 8u));
                    offset += 16u;
                }
        }
        else if (flg == GIF_FMT_REGLIST)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 8u > sizeBytes)
                        return;
                    writeRegister(regs[r], load64(data + offset));
                    offset += 8u;
                }
            if (((nloop * nreg) & 1u) != 0u)
                offset += 8u;
        }
        else
        {
            uint32_t imageBytes = nloop * 16u;
            if (offset + imageBytes > sizeBytes)
                imageBytes = sizeBytes - offset;
            m_sink.imageData(*this, data + offset, imageBytes);
            offset += imageBytes;
        }
    }
}

void GSGifDecoder::writePacked(uint8_t descriptor, uint64_t lo, uint64_t hi)
{
    switch (descriptor)
    {
    case 0x00:
        writeRegister(GS_REG_PRIM, lo & 0x7FFu);
        break;
    case 0x01:
    {
        const uint64_t rgba = (lo & 0xFFu) | (((lo >> 32) & 0xFFu) << 8) | ((hi & 0xFFu) << 16) |
                              (((hi >> 32) & 0xFFu) << 24);
        m_state.setReg(GS_REG_RGBAQ, (m_state.reg(GS_REG_RGBAQ) & ~0xFFFFFFFFull) | rgba);
        break;
    }
    case 0x02:
        m_state.setReg(GS_REG_ST, lo);
        setQ(static_cast<uint32_t>(hi));
        break;
    case 0x03:
        m_state.setReg(GS_REG_UV, (lo & 0x3FFFu) | (((lo >> 32) & 0x3FFFu) << 16));
        break;
    case 0x04:
    case 0x0C:
    {
        const uint64_t xyz = (lo & 0xFFFFu) | (((lo >> 32) & 0xFFFFu) << 16) | (((hi >> 4) & 0xFFFFFFu) << 32);
        kick(xyz, static_cast<uint8_t>(hi >> 36), descriptor == 0x04 && ((hi >> 47) & 1u) == 0u);
        break;
    }
    case 0x05:
    case 0x0D:
    {
        const uint64_t xyz = (lo & 0xFFFFu) | (((lo >> 32) & 0xFFFFu) << 16) | ((hi & 0xFFFFFFFFu) << 32);
        kick(xyz, static_cast<uint8_t>(m_state.reg(GS_REG_FOG) >> 56),
             descriptor == 0x05 && ((hi >> 47) & 1u) == 0u);
        break;
    }
    case 0x0A:
        m_state.setReg(GS_REG_FOG, ((hi >> 36) & 0xFFu) << 56);
        break;
    case 0x0E:
        writeRegister(static_cast<uint8_t>(hi & 0xFFu), lo);
        break;
    case 0x0F:
        break;
    default:
        writeRegister(descriptor, lo);
        break;
    }
}

void GSGifDecoder::writeRegister(uint8_t address, uint64_t value)
{
    switch (address)
    {
    case GS_REG_PRIM:
        m_state.setReg(GS_REG_PRIM, value);
        m_state.vertexCount = 0u;
        break;
    case GS_REG_XYZF2:
    case GS_REG_XYZF3:
        kick((value & 0xFFFFFFFFull) | (((value >> 32) & 0xFFFFFFu) << 32), static_cast<uint8_t>(value >> 56),
             address == GS_REG_XYZF2);
        break;
    case GS_REG_XYZ2:
    case GS_REG_XYZ3:
        kick(value, static_cast<uint8_t>(m_state.reg(GS_REG_FOG) >> 56), address == GS_REG_XYZ2);
        break;
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    {
        // TEX2 rewrites TEX0's PSM and CLUT fields.
        constexpr uint64_t kMask = (0x3Full << 20) | (0x7FFFFFFull << 37);
        const uint8_t tex0 = address == GS_REG_TEX2_1 ? GS_REG_TEX0_1 : GS_REG_TEX0_2;
        m_state.setReg(tex0, (m_state.reg(tex0) & ~kMask) | (value & kMask));
        break;
    }
    case GS_REG_HWREG:
    {
        uint8_t bytes[8];
        std::memcpy(bytes, &value, sizeof(bytes));
        m_sink.imageData(*this, bytes, sizeof(bytes));
        break;
    }
    default:
        if (address < m_state.registers.size())
            m_state.setReg(address, value);
        break;
    }
    m_sink.registerWritten(*this, address, value);
}

void GSGifDecoder::kick(uint64_t xyz, uint8_t fog, bool drawing)
{
    GSStateImage::Vertex &vertex = m_state.vertexQueue[m_state.vertexCount % GSStateImage::kMaxQueuedVertices];
    vertex.xyz = xyz;
    vertex.rgbaq = m_state.reg(GS_REG_RGBAQ);
    vertex.st = m_state.reg(GS_REG_ST);
    vertex.uv = m_state.reg(GS_REG_UV);
    vertex.fog = fog;
    ++m_state.vertexCount;

    const uint64_t prim = effectivePrim();
    const uint32_t needed = verticesNeeded(prim);
    if (needed == 0u || m_state.vertexCount < needed)
        return;

    if (drawing)
    {
        Primitive primitive;
        primitive.prim = prim;
        primitive.vertexCount = needed;
        for (uint32_t i = 0; i < needed; ++i)
            primitive.vertices[i] = m_state.vertexQueue[i];
        m_sink.primitive(*this, primitive);
    }

    std::array<GSStateImage::Vertex, GSStateImage::kMaxQueuedVertices> &queue = m_state.vertexQueue;
    switch (prim & 0x7u)
    {
    case GS_PRIM_LINESTRIP:
        queue[0] = queue[1];
        m_state.vertexCount = 1u;
        break;
    case GS_PRIM_TRISTRIP:
        queue[0] = queue[1];
        queue[1] = queue[2];
        m_state.vertexCount = 2u;
        break;
    case GS_PRIM_TRIFAN:
        queue[1] = queue[2];
        m_state.vertexCount = 2u;
        break;
    default:
        m_state.vertexCount = 0u;
        break;
    }
}
