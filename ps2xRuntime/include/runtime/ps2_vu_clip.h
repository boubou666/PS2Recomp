#ifndef PS2_VU_CLIP_H
#define PS2_VU_CLIP_H

#include <cstdint>
#include <cstring>

// CLIP judgement of fs.xyz against |w|: bit 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z. A w with exponent 0 is a zero
// limit, and only operands with a nonzero exponent exceed it.
inline uint32_t ps2VuClipJudge(const float *fs, float w)
{
    uint32_t wBits = 0u;
    std::memcpy(&wBits, &w, sizeof(wBits));
    const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

    const auto exceeds = [limit](float value, uint32_t signMask)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        bits ^= signMask;
        int32_t orderedBits = 0;
        std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
        return orderedBits > limit;
    };

    uint32_t flags = 0u;
    for (uint32_t component = 0; component < 3u; ++component)
    {
        if (exceeds(fs[component], 0x00000000u))
            flags |= 1u << (component * 2u);
        if (exceeds(fs[component], 0x80000000u))
            flags |= 2u << (component * 2u);
    }
    return flags;
}

#endif
