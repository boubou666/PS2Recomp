#pragma once

// GS register values (as written through A+D or the register file) to the frontend's decoded fields and back.
// encode(decode(v)) keeps every bit the frontend uses; decode(encode(f)) == f.
#include "runtime/gs/gs_types.h"

#include <cstdint>

namespace GSReg
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

    inline GSTex0Reg decodeTex0(uint64_t value)
    {
        GSTex0Reg t{};
        t.tbp0 = static_cast<uint32_t>(value & 0x3FFF);
        t.tbw = static_cast<uint8_t>((value >> 14) & 0x3F);
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
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

    // TEX2 rewrites the PSM and CLUT fields of TEX0.
    inline void applyTex2(GSTex0Reg &t, uint64_t value)
    {
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
    }

    inline GSXYOffsetReg decodeXyoffset(uint64_t value)
    {
        GSXYOffsetReg o{};
        o.ofx = static_cast<uint16_t>(value & 0xFFFF);
        o.ofy = static_cast<uint16_t>((value >> 32) & 0xFFFF);
        return o;
    }

    inline GSTexClutReg decodeTexclut(uint64_t value)
    {
        GSTexClutReg c{};
        c.cbw = static_cast<uint8_t>(value & 0x3Fu);
        c.cou = static_cast<uint8_t>((value >> 6) & 0x3Fu);
        c.cov = static_cast<uint16_t>((value >> 12) & 0x3FFu);
        return c;
    }

    inline GSScissorReg decodeScissor(uint64_t value)
    {
        GSScissorReg s{};
        s.x0 = static_cast<uint16_t>(value & 0x7FF);
        s.x1 = static_cast<uint16_t>((value >> 16) & 0x7FF);
        s.y0 = static_cast<uint16_t>((value >> 32) & 0x7FF);
        s.y1 = static_cast<uint16_t>((value >> 48) & 0x7FF);
        return s;
    }

    inline GSFrameReg decodeFrame(uint64_t value)
    {
        GSFrameReg f{};
        f.fbp = static_cast<uint32_t>(value & 0x1FF);
        f.fbw = static_cast<uint32_t>((value >> 16) & 0x3F);
        f.psm = static_cast<uint8_t>((value >> 24) & 0x3F);
        f.fbmsk = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        return f;
    }

    inline GSZbufReg decodeZbuf(uint64_t value)
    {
        GSZbufReg z{};
        z.zbp = value & 0x1FF;
        z.psm = ((value >> 24) & 0xF) | 0x30;
        z.zmask = (value >> 32) & 1;
        return z;
    }

    inline GSBitBltBuf decodeBitbltbuf(uint64_t value)
    {
        GSBitBltBuf b{};
        b.sbp = static_cast<uint32_t>(value & 0x3FFF);
        b.sbw = static_cast<uint8_t>((value >> 16) & 0x3F);
        b.spsm = static_cast<uint8_t>((value >> 24) & 0x3F);
        b.dbp = static_cast<uint32_t>((value >> 32) & 0x3FFF);
        b.dbw = static_cast<uint8_t>((value >> 48) & 0x3F);
        b.dpsm = static_cast<uint8_t>((value >> 56) & 0x3F);
        return b;
    }

    inline GSTrxPos decodeTrxpos(uint64_t value)
    {
        GSTrxPos p{};
        p.ssax = static_cast<uint16_t>(value & 0x7FF);
        p.ssay = static_cast<uint16_t>((value >> 16) & 0x7FF);
        p.dsax = static_cast<uint16_t>((value >> 32) & 0x7FF);
        p.dsay = static_cast<uint16_t>((value >> 48) & 0x7FF);
        p.dir = static_cast<uint8_t>((value >> 59) & 0x3);
        return p;
    }

    inline GSTrxReg decodeTrxreg(uint64_t value)
    {
        GSTrxReg r{};
        r.rrw = static_cast<uint16_t>(value & 0xFFF);
        r.rrh = static_cast<uint16_t>((value >> 32) & 0xFFF);
        return r;
    }

    inline GSTexaReg decodeTexa(uint64_t value)
    {
        GSTexaReg t{};
        t.ta0 = static_cast<uint8_t>(value & 0xFFu);
        t.aem = ((value >> 15) & 0x1u) != 0u;
        t.ta1 = static_cast<uint8_t>((value >> 32) & 0xFFu);
        return t;
    }

    inline uint64_t encodePrim(const GSPrimReg &prim)
    {
        return (static_cast<uint64_t>(prim.type) & 0x7u) | (uint64_t(prim.iip) << 3) | (uint64_t(prim.tme) << 4) |
               (uint64_t(prim.fge) << 5) | (uint64_t(prim.abe) << 6) | (uint64_t(prim.aa1) << 7) |
               (uint64_t(prim.fst) << 8) | (uint64_t(prim.ctxt) << 9) | (uint64_t(prim.fix) << 10);
    }

    inline uint64_t encodeTex0(const GSTex0Reg &t)
    {
        return (uint64_t(t.tbp0) & 0x3FFF) | ((uint64_t(t.tbw) & 0x3F) << 14) | ((uint64_t(t.psm) & 0x3F) << 20) |
               ((uint64_t(t.tw) & 0xF) << 26) | ((uint64_t(t.th) & 0xF) << 30) | ((uint64_t(t.tcc) & 0x1) << 34) |
               ((uint64_t(t.tfx) & 0x3) << 35) | ((uint64_t(t.cbp) & 0x3FFF) << 37) | ((uint64_t(t.cpsm) & 0xF) << 51) |
               ((uint64_t(t.csm) & 0x1) << 55) | ((uint64_t(t.csa) & 0x1F) << 56) | ((uint64_t(t.cld) & 0x7) << 61);
    }

    inline uint64_t encodeXyoffset(const GSXYOffsetReg &o)
    {
        return uint64_t(o.ofx) | (uint64_t(o.ofy) << 32);
    }

    inline uint64_t encodeTexclut(const GSTexClutReg &c)
    {
        return (uint64_t(c.cbw) & 0x3F) | ((uint64_t(c.cou) & 0x3F) << 6) | ((uint64_t(c.cov) & 0x3FF) << 12);
    }

    inline uint64_t encodeScissor(const GSScissorReg &s)
    {
        return (uint64_t(s.x0) & 0x7FF) | ((uint64_t(s.x1) & 0x7FF) << 16) | ((uint64_t(s.y0) & 0x7FF) << 32) |
               ((uint64_t(s.y1) & 0x7FF) << 48);
    }

    inline uint64_t encodeFrame(const GSFrameReg &f)
    {
        return (uint64_t(f.fbp) & 0x1FF) | ((uint64_t(f.fbw) & 0x3F) << 16) | ((uint64_t(f.psm) & 0x3F) << 24) |
               (uint64_t(f.fbmsk) << 32);
    }

    inline uint64_t encodeZbuf(const GSZbufReg &z)
    {
        return (uint64_t(z.zbp) & 0x1FF) | ((uint64_t(z.psm) & 0xF) << 24) | (uint64_t(z.zmask) << 32);
    }

    inline uint64_t encodeBitbltbuf(const GSBitBltBuf &b)
    {
        return (uint64_t(b.sbp) & 0x3FFF) | ((uint64_t(b.sbw) & 0x3F) << 16) | ((uint64_t(b.spsm) & 0x3F) << 24) |
               ((uint64_t(b.dbp) & 0x3FFF) << 32) | ((uint64_t(b.dbw) & 0x3F) << 48) | ((uint64_t(b.dpsm) & 0x3F) << 56);
    }

    inline uint64_t encodeTrxpos(const GSTrxPos &p)
    {
        return (uint64_t(p.ssax) & 0x7FF) | ((uint64_t(p.ssay) & 0x7FF) << 16) | ((uint64_t(p.dsax) & 0x7FF) << 32) |
               ((uint64_t(p.dsay) & 0x7FF) << 48) | ((uint64_t(p.dir) & 0x3) << 59);
    }

    inline uint64_t encodeTrxreg(const GSTrxReg &r)
    {
        return (uint64_t(r.rrw) & 0xFFF) | ((uint64_t(r.rrh) & 0xFFF) << 32);
    }

    inline uint64_t encodeTexa(const GSTexaReg &t)
    {
        return uint64_t(t.ta0) | (uint64_t(t.aem) << 15) | (uint64_t(t.ta1) << 32);
    }
}
