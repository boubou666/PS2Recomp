#pragma once

// The GS drawing state as raw register values, independent of which renderer holds it: what a GS trace starts
// from (GSR2) and what moves between renderers. Taken between GIF packets with no image transfer in progress.
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct GSStateImage
{
    static constexpr uint32_t kVersion = 1u;
    static constexpr uint32_t kMaxQueuedVertices = 6u;

    enum Flags : uint32_t
    {
        HasVertexQueue = 1u << 0,
        HasClut = 1u << 1,
    };

    // A vertex waiting in the queue, as the registers that made it.
    struct Vertex
    {
        uint64_t xyz = 0;   // XYZ2 layout: X and Y 12.4, Z 32 bits
        uint64_t rgbaq = 0; // RGBAQ layout, Q as float bits
        uint64_t st = 0;
        uint64_t uv = 0;
        uint8_t fog = 0;
    };

    // Every general register the image holds, by GS address. TEX2 is folded into TEX0; RGBAQ, ST, UV and FOG are
    // the values the next vertex takes. TRXDIR is the last direction written.
    static constexpr std::array<uint8_t, 43> kRegisters = {
        0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x14, 0x15, 0x18, 0x19, 0x1A, 0x1B,
        0x1C, 0x22, 0x34, 0x35, 0x36, 0x37, 0x3B, 0x3D, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46,
        0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53};

    uint32_t flags = 0;
    std::array<uint64_t, 0x80> registers{};
    // Vertices kicked since the last primitive was drawn; slot i holds the vertex at position i % kMaxQueuedVertices.
    uint32_t vertexCount = 0;
    std::array<Vertex, kMaxQueuedVertices> vertexQueue{};
    // The GS CLUT buffer: 512 16-bit entries (CT32 palettes as low halves then high halves), and the CBP0/CBP1
    // values the last CLD 4/5 loads compared against.
    std::array<uint16_t, 512> clut{};
    std::array<uint32_t, 2> clutCbp{};

    uint64_t reg(uint8_t address) const { return registers[address & 0x7Fu]; }
    void setReg(uint8_t address, uint64_t value) { registers[address & 0x7Fu] = value; }
};

std::vector<uint8_t> serializeGsStateImage(const GSStateImage &image);
bool deserializeGsStateImage(const uint8_t *data, size_t size, GSStateImage &image, size_t *consumed = nullptr);
