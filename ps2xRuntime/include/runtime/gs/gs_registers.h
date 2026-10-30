#pragma once

#include "runtime/gs/gs_types.h"

#include <cstdint>

namespace ps2x::gs
{
    inline GSPrimReg decodePrim(uint64_t value)
    {
        GSPrimReg prim{};
        prim.type = static_cast<GSPrimType>(value & 0x7u);
        prim.iip = ((value >> 3) & 1u) != 0u;
        prim.tme = ((value >> 4) & 1u) != 0u;
        prim.fge = ((value >> 5) & 1u) != 0u;
        prim.abe = ((value >> 6) & 1u) != 0u;
        prim.aa1 = ((value >> 7) & 1u) != 0u;
        prim.fst = ((value >> 8) & 1u) != 0u;
        prim.ctxt = ((value >> 9) & 1u) != 0u;
        prim.fix = ((value >> 10) & 1u) != 0u;
        return prim;
    }

    // TEX0 / TEX2 PSM. A value the GS does not define is read as PSMCT32 (GSdx's format table defaults every entry
    // to it; paraLLEl-GS shows such textures): The Third Age draws its movie pictures with PSM 0x20.
    inline uint8_t textureFormat(uint64_t tex0)
    {
        const uint8_t psm = static_cast<uint8_t>((tex0 >> 20) & 0x3F);
        switch (psm)
        {
        case 0x00: case 0x01: case 0x02: case 0x0A: case 0x13: case 0x14: case 0x1B: case 0x24: case 0x2C:
        case 0x30: case 0x31: case 0x32: case 0x3A:
            return psm;
        default:
            return 0x00;
        }
    }

    inline GSTex0Reg decodeTex0(uint64_t value)
    {
        GSTex0Reg t{};
        t.tbp0 = static_cast<uint32_t>(value & 0x3FFF);
        t.tbw = static_cast<uint8_t>((value >> 14) & 0x3F);
        t.psm = textureFormat(value);
        t.tw = static_cast<uint8_t>((value >> 26) & 0xF);
        t.th = static_cast<uint8_t>((value >> 30) & 0xF);
        t.tcc = static_cast<uint8_t>((value >> 34) & 0x1);
        t.tfx = static_cast<uint8_t>((value >> 35) & 0x3);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        return t;
    }

    inline bool linearFilter(uint64_t tex1)
    {
        const uint8_t mmag = static_cast<uint8_t>((tex1 >> 5u) & 0x1u);
        const uint8_t mmin = static_cast<uint8_t>((tex1 >> 6u) & 0x7u);
        return mmag != 0u || mmin == 1u || (mmin & 0x4u) != 0u;
    }
}
