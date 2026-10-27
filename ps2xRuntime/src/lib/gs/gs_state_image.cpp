#include "runtime/gs/gs_state_image.h"

namespace
{
    // Little endian whatever the host is.
    void put(std::vector<uint8_t> &out, uint64_t value, uint32_t bytes)
    {
        for (uint32_t i = 0; i < bytes; ++i)
            out.push_back(static_cast<uint8_t>(value >> (8u * i)));
    }

    struct Reader
    {
        const uint8_t *data;
        size_t size;
        size_t offset = 0;

        bool get(uint64_t &value, uint32_t bytes)
        {
            if (offset + bytes > size)
                return false;
            value = 0;
            for (uint32_t i = 0; i < bytes; ++i)
                value |= static_cast<uint64_t>(data[offset + i]) << (8u * i);
            offset += bytes;
            return true;
        }
    };
}

std::vector<uint8_t> serializeGsStateImage(const GSStateImage &image)
{
    std::vector<uint8_t> out;
    put(out, GSStateImage::kVersion, 4);
    put(out, image.flags, 4);
    put(out, GSStateImage::kRegisters.size(), 4);
    for (const uint8_t address : GSStateImage::kRegisters)
    {
        put(out, address, 1);
        put(out, image.reg(address), 8);
    }
    put(out, image.vertexCount, 4);
    for (const GSStateImage::Vertex &vertex : image.vertexQueue)
    {
        put(out, vertex.xyz, 8);
        put(out, vertex.rgbaq, 8);
        put(out, vertex.st, 8);
        put(out, vertex.uv, 8);
        put(out, vertex.fog, 1);
    }
    for (const uint16_t entry : image.clut)
        put(out, entry, 2);
    for (const uint32_t cbp : image.clutCbp)
        put(out, cbp, 4);
    return out;
}

bool deserializeGsStateImage(const uint8_t *data, size_t size, GSStateImage &image, size_t *consumed)
{
    Reader in{data, size};
    uint64_t version = 0, flags = 0, registerCount = 0, vertexCount = 0;
    if (!data || !in.get(version, 4) || version != GSStateImage::kVersion || !in.get(flags, 4) ||
        !in.get(registerCount, 4))
        return false;
    GSStateImage result;
    result.flags = static_cast<uint32_t>(flags);
    for (uint64_t i = 0; i < registerCount; ++i)
    {
        uint64_t address = 0, value = 0;
        if (!in.get(address, 1) || !in.get(value, 8) || address >= result.registers.size())
            return false;
        result.setReg(static_cast<uint8_t>(address), value);
    }
    if (!in.get(vertexCount, 4))
        return false;
    result.vertexCount = static_cast<uint32_t>(vertexCount);
    for (GSStateImage::Vertex &vertex : result.vertexQueue)
    {
        uint64_t fog = 0;
        if (!in.get(vertex.xyz, 8) || !in.get(vertex.rgbaq, 8) || !in.get(vertex.st, 8) || !in.get(vertex.uv, 8) ||
            !in.get(fog, 1))
            return false;
        vertex.fog = static_cast<uint8_t>(fog);
    }
    for (uint16_t &entry : result.clut)
    {
        uint64_t value = 0;
        if (!in.get(value, 2))
            return false;
        entry = static_cast<uint16_t>(value);
    }
    for (uint32_t &cbp : result.clutCbp)
    {
        uint64_t value = 0;
        if (!in.get(value, 4))
            return false;
        cbp = static_cast<uint32_t>(value);
    }
    image = result;
    if (consumed)
        *consumed = in.offset;
    return true;
}
