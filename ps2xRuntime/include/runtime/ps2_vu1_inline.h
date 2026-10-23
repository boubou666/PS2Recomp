#ifndef PS2_VU1_INLINE_H
#define PS2_VU1_INLINE_H

// VU0/VU1 instruction semantics, inline so that run() and the microcode compiled ahead of time (ps2_vu_aot)
// share one implementation. Compiled code calls stepPair() with a constexpr decodePair(): once inlined, the
// decode, the opcode switches and the operand bookkeeping fold away. Only VU sources and generated microcode
// include this header.
#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <limits>

#if defined(_MSC_VER) && !defined(__clang__)
#define PS2X_VU_INLINE __forceinline
#define PS2X_VU_LAMBDA_INLINE [[msvc::forceinline]]
#else
#define PS2X_VU_INLINE inline __attribute__((always_inline))
#define PS2X_VU_LAMBDA_INLINE __attribute__((always_inline))
#endif

namespace ps2_vu1_detail
{
    // Instruction field extraction helpers
    constexpr uint8_t DEST(uint32_t i) { return (uint8_t)((i >> 21) & 0xF); }
    constexpr uint8_t FT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
    constexpr uint8_t FS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
    constexpr uint8_t FD(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
    constexpr uint8_t BC(uint32_t i) { return (uint8_t)(i & 0x3); }

    // Lower instruction field helpers
    constexpr uint8_t LIT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
    constexpr uint8_t LIS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
    constexpr uint8_t LID(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
    constexpr uint8_t VIT(uint32_t i) { return (uint8_t)((i >> 16) & 0xF); }
    constexpr uint8_t VIS(uint32_t i) { return (uint8_t)((i >> 11) & 0xF); }
    constexpr uint8_t VID(uint32_t i) { return (uint8_t)((i >> 6) & 0xF); }
    constexpr int16_t IMM11(uint32_t i) { return (int16_t)(int32_t)((int32_t)(i << 21) >> 21); }
    constexpr int16_t IMM15(uint32_t i)
    {
        uint32_t lo11 = i & 0x7FF;
        uint32_t hi4 = (i >> 21) & 0xF;
        uint32_t raw = (hi4 << 11) | lo11;
        return (int16_t)(int32_t)((int32_t)(raw << 17) >> 17);
    }

    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    // The exact FMAC result calculateFmacExactResult computes in long double, as a double. Products of two floats
    // are exact in double; sums are rounded toward zero (the run's rounding mode) in both, so the zero test, the
    // sign and the comparison with FLT_MIN agree. Only a sum within a double ulp above FLT_MAX could compare
    // differently with FLT_MAX: callers fall back to long double for sums at FLT_MAX or above.
    enum class FmacKind : uint8_t
    {
        None,
        Add,
        Sub,
        Mul,
        Madd,
        Msub,
        CrossMsub, // OPMSUB: acc - vs.yzx * vt.zxy, w = 0
        CrossMul   // OPMULA: vs.yzx * vt.zxy, w = 0
    };

    struct FmacOperation
    {
        FmacKind kind = FmacKind::None;
        enum class Right : uint8_t
        {
            Vector,
            Broadcast,
            Q,
            I
        } right = Right::Vector;
        uint8_t broadcast = 0;
    };

    constexpr FmacOperation decodeFmacOperation(uint32_t upper)
    {
        const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
        const bool special = op >= 0x3Cu;
        const uint8_t code = special ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
        FmacOperation operation;
        using Right = FmacOperation::Right;
        if (code <= 0x1Bu && (code <= 0x0Fu || code >= 0x18u))
        {
            constexpr FmacKind kinds[4] = {FmacKind::Add, FmacKind::Sub, FmacKind::Madd, FmacKind::Msub};
            operation.kind = code >= 0x18u ? FmacKind::Mul : kinds[code >> 2];
            operation.right = Right::Broadcast;
            operation.broadcast = static_cast<uint8_t>(code & 3u);
            return operation;
        }
        switch (code)
        {
        case 0x1Cu: operation = {FmacKind::Mul, Right::Q}; break;
        case 0x1Eu: operation = {FmacKind::Mul, Right::I}; break;
        case 0x20u: operation = {FmacKind::Add, Right::Q}; break;
        case 0x21u: operation = {FmacKind::Madd, Right::Q}; break;
        case 0x22u: operation = {FmacKind::Add, Right::I}; break;
        case 0x23u: operation = {FmacKind::Madd, Right::I}; break;
        case 0x24u: operation = {FmacKind::Sub, Right::Q}; break;
        case 0x25u: operation = {FmacKind::Msub, Right::Q}; break;
        case 0x26u: operation = {FmacKind::Sub, Right::I}; break;
        case 0x27u: operation = {FmacKind::Msub, Right::I}; break;
        case 0x28u: operation = {FmacKind::Add, Right::Vector}; break;
        case 0x29u: operation = {FmacKind::Madd, Right::Vector}; break;
        case 0x2Au: operation = {FmacKind::Mul, Right::Vector}; break;
        case 0x2Cu: operation = {FmacKind::Sub, Right::Vector}; break;
        case 0x2Du: operation = {FmacKind::Msub, Right::Vector}; break;
        case 0x2Eu: operation = {special ? FmacKind::CrossMul : FmacKind::CrossMsub, Right::Vector}; break;
        default: break;
        }
        return operation;
    }


    inline int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // PS2 FMAC addition. Before adding, the smaller operand loses the mantissa bits below the
    // larger one's last bit (no guard bits); the sum then rounds toward zero, which the
    // interpreter's FE_TOWARDZERO mode does. A plain IEEE sum differs by one ulp when the small
    // operand has the opposite sign: Valkyrie Profile 2 derives its file decryption key on VU0
    // and depends on the exact PS2 result (PCSX2 hardcodes these additions as a "TriAce hack").
    inline float vuAdd(float a, float b)
    {

        uint32_t ua = 0u;
        uint32_t ub = 0u;
        std::memcpy(&ua, &a, sizeof(ua));
        std::memcpy(&ub, &b, sizeof(ub));
        const uint32_t ea = (ua >> 23) & 0xFFu;
        const uint32_t eb = (ub >> 23) & 0xFFu;
        if (ea != 0u && eb != 0u && ea != eb)
        {
            uint32_t &small = ea > eb ? ub : ua;
            const uint32_t shift = ea > eb ? ea - eb : eb - ea;
            small &= shift >= 24u ? 0x80000000u : ~((1u << shift) - 1u);
            std::memcpy(&a, &ua, sizeof(a));
            std::memcpy(&b, &ub, sizeof(b));
        }
        return a + b;
    }

    inline float vuSub(float a, float b)
    {
        return vuAdd(a, -b);
    }

    // vuAdd on four lanes: with AVX2 the same operand alignment, then one SSE add under the
    // interpreter's round-toward-zero mode; without it, lane by lane.
    inline __m128 vuAdd4(__m128 a, __m128 b)
    {
#if !defined(__AVX2__)
        alignas(16) float la[4];
        alignas(16) float lb[4];
        _mm_store_ps(la, a);
        _mm_store_ps(lb, b);
        for (int c = 0; c < 4; ++c)
            la[c] = vuAdd(la[c], lb[c]);
        return _mm_load_ps(la);
#else
        const __m128i ia = _mm_castps_si128(a);
        const __m128i ib = _mm_castps_si128(b);
        const __m128i exponentMask = _mm_set1_epi32(0xFF);
        const __m128i allOnes = _mm_set1_epi32(-1);
        const __m128i ea = _mm_and_si128(_mm_srli_epi32(ia, 23), exponentMask);
        const __m128i eb = _mm_and_si128(_mm_srli_epi32(ib, 23), exponentMask);
        const __m128i zero = _mm_setzero_si128();
        const __m128i inactive = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi32(ea, zero), _mm_cmpeq_epi32(eb, zero)),
                                              _mm_cmpeq_epi32(ea, eb));
        const __m128i aLarger = _mm_cmpgt_epi32(ea, eb);
        const __m128i shift = _mm_abs_epi32(_mm_sub_epi32(ea, eb));
        const __m128i one = _mm_set1_epi32(1);
        const __m128i lowBits = _mm_sub_epi32(_mm_sllv_epi32(one, shift), one);
        const __m128i far = _mm_cmpgt_epi32(shift, _mm_set1_epi32(23));
        const __m128i keep = _mm_or_si128(_mm_andnot_si128(far, _mm_xor_si128(lowBits, allOnes)),
                                          _mm_and_si128(far, _mm_set1_epi32(static_cast<int>(0x80000000u))));
        // The operand with the smaller exponent loses its low bits.
        const __m128i truncateA = _mm_andnot_si128(inactive, _mm_andnot_si128(aLarger, allOnes));
        const __m128i truncateB = _mm_andnot_si128(inactive, aLarger);
        const __m128i newA = _mm_and_si128(ia, _mm_or_si128(_mm_andnot_si128(truncateA, allOnes), keep));
        const __m128i newB = _mm_and_si128(ib, _mm_or_si128(_mm_andnot_si128(truncateB, allOnes), keep));
        return _mm_add_ps(_mm_castsi128_ps(newA), _mm_castsi128_ps(newB));
#endif
    }

    inline __m128 negate4(__m128 value)
    {
        return _mm_xor_ps(value, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000u))));
    }

    // MAX/MINI compare raw sign-magnitude bit patterns and pass the chosen operand through unchanged, so
    // denormals (often integer data, e.g. colors unpacked by VIF) and exponent-255 values survive.
    inline int64_t vuMinMaxKey(float value)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        const int64_t magnitude = static_cast<int64_t>(bits & 0x7FFFFFFFu);
        return (bits & 0x80000000u) != 0u ? -magnitude : magnitude;
    }

    inline float vuMax(float a, float b) { return vuMinMaxKey(a) > vuMinMaxKey(b) ? a : b; }
    inline float vuMini(float a, float b) { return vuMinMaxKey(a) < vuMinMaxKey(b) ? a : b; }

    // normalizeOperand() on four lanes: denormals become signed zero, exponent 255 becomes +-FLT_MAX.
    inline void normalizeVector(const float *source, float *target)
    {
        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
        const __m128i exponentMask = _mm_set1_epi32(0x7F800000);
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        const __m128i exponent = _mm_and_si128(bits, exponentMask);
        const __m128i zeroExponent = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
        const __m128i maxExponent = _mm_cmpeq_epi32(exponent, exponentMask);
        __m128i result = _mm_or_si128(_mm_andnot_si128(zeroExponent, bits), _mm_and_si128(zeroExponent, sign));
        result = _mm_or_si128(_mm_andnot_si128(maxExponent, result),
                              _mm_and_si128(maxExponent, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF))));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(target), result);
    }

    inline float vuEatan(float value)
    {
        constexpr float coefficients[] = {
            0.999999344348907f,
            -0.333298563957214f,
            0.199465364217758f,
            -0.13085337519646f,
            0.096420042216778f,
            -0.055909886956215f,
            0.021861229091883f,
            -0.004054057877511f};
        constexpr float quarterPi = 0.785398185253143f;

        const float squared = value * value;
        float polynomial = coefficients[7];
        for (int index = 6; index >= 0; --index)
            polynomial = coefficients[index] + squared * polynomial;
        return quarterPi + value * polynomial;
    }

    inline float vuEsin(float value)
    {
        constexpr float coefficients[] = {
            1.0f,
            -0.166666567325592f,
            0.008333025500178f,
            -0.000198074136279f,
            0.000002601886990f};

        const float squared = value * value;
        float polynomial = coefficients[4];
        for (int index = 3; index >= 0; --index)
            polynomial = coefficients[index] + squared * polynomial;
        return value * polynomial;
    }

    inline float vuEexp(float value)
    {
        constexpr float coefficients[] = {
            0.249998688697815f,
            0.031257584691048f,
            0.002591371303424f,
            0.000171562001924f,
            0.000005430199963f,
            0.000000690600018f};

        float polynomial = coefficients[5];
        for (int index = 4; index >= 0; --index)
            polynomial = coefficients[index] + value * polynomial;
        polynomial = 1.0f + value * polynomial;
        polynomial *= polynomial;
        polynomial *= polynomial;
        return polynomial != 0.0f ? 1.0f / polynomial : std::numeric_limits<float>::max();
    }
}

using namespace ps2_vu1_detail;

constexpr void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

constexpr void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

constexpr uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

constexpr VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper)
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

constexpr VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(Unit unit, uint32_t lower)
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

constexpr VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodePair(Unit unit, uint32_t lower, uint32_t upper)
{
    DecodedInstructionPair decoded;
    decoded.lower = lower;
    decoded.upper = upper;
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(unit, decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }
    return decoded;
}

inline VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    uint32_t lower = 0u, upper = 0u;
    std::memcpy(&lower, vuCode + pc, sizeof(lower));
    std::memcpy(&upper, vuCode + pc + sizeof(lower), sizeof(upper));
    return decodePair(m_unit, lower, upper);
}

PS2X_VU_INLINE float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

PS2X_VU_INLINE float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

PS2X_VU_INLINE float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

PS2X_VU_INLINE uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

PS2X_VU_INLINE int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

PS2X_VU_INLINE void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

PS2X_VU_INLINE void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

PS2X_VU_INLINE void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

// MAC/status flags of an FMAC result. A written lane with exponent 2..253 is far from zero, the denormal range and
// FLT_MAX: only its sign flag can be set, and products of product-sums in that range only add sticky sign and zero
// flags. Anything else needs the exact result: fmacFlagsSlow.
PS2X_VU_INLINE bool VU1Interpreter::fmacFlagsFast(float *result, uint8_t dest, uint32_t &mac, uint32_t &status,
                                                  uint32_t &sticky) const
{
    // dest bit 3 is x (lane 0); movemask bit c is lane c.
    static constexpr uint8_t kDestLanes[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                               0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
    // Lane c's flags are bit 3 - c of each MAC nibble.
    static constexpr uint8_t kReverse4[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                              0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
    const uint32_t active = kDestLanes[dest & 0xFu];
    const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(result));
    const __m128i exponent = _mm_and_si128(_mm_srli_epi32(bits, 23), _mm_set1_epi32(0xFF));
    const __m128i inRange = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1)),
                                          _mm_cmpgt_epi32(_mm_set1_epi32(254), exponent));
    const uint32_t outOfRange = active & ~static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(inRange)));

    const uint32_t upper = m_currentUpperInstruction;
    const FmacOperation operation = decodeFmacOperation(upper);
    const FmacKind kind = operation.kind;
    const bool productSum = kind == FmacKind::Madd || kind == FmacKind::Msub || kind == FmacKind::CrossMsub;
    const bool cross = kind == FmacKind::CrossMsub || kind == FmacKind::CrossMul;

    sticky = 0u;
    uint32_t zeroLanes = 0u, underflowLanes = 0u, overflowLanes = 0u;
    uint32_t signLanes = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(bits))) & active & ~outOfRange;
    if (outOfRange != 0u || kind == FmacKind::None)
        return false;
    if (productSum)
    {
        alignas(16) float left[4];
        alignas(16) float right[4];
        normalizeVector(m_state.vf[FS(upper)], left);
        switch (operation.right)
        {
        case FmacOperation::Right::Broadcast:
            _mm_store_ps(right, _mm_set1_ps(normalizeOperand(m_state.vf[FT(upper)][operation.broadcast])));
            break;
        case FmacOperation::Right::Q:
            _mm_store_ps(right, _mm_set1_ps(normalizeOperand(m_state.q)));
            break;
        case FmacOperation::Right::I:
            _mm_store_ps(right, _mm_set1_ps(normalizeOperand(m_state.i)));
            break;
        default:
            normalizeVector(m_state.vf[FT(upper)], right);
            break;
        }
        __m128 leftVector = _mm_load_ps(left);
        __m128 rightVector = _mm_load_ps(right);
        if (cross)
        {
            leftVector = _mm_shuffle_ps(leftVector, leftVector, _MM_SHUFFLE(3, 0, 2, 1));     // y z x w
            rightVector = _mm_shuffle_ps(rightVector, rightVector, _MM_SHUFFLE(3, 1, 0, 2)); // z x y w
        }

        if (productSum)
        {
            // calculateFmacProductSticky: every product condition accumulates into the sticky flags.
            const __m128i leftBits = _mm_castps_si128(leftVector);
            const __m128i rightBits = _mm_castps_si128(rightVector);
            const __m128i exponentMask = _mm_set1_epi32(0xFF);
            const __m128i leftExponent = _mm_and_si128(_mm_srli_epi32(leftBits, 23), exponentMask);
            const __m128i rightExponent = _mm_and_si128(_mm_srli_epi32(rightBits, 23), exponentMask);
            const __m128i zero = _mm_or_si128(_mm_cmpeq_epi32(leftExponent, _mm_setzero_si128()),
                                              _mm_cmpeq_epi32(rightExponent, _mm_setzero_si128()));
            // The product's exponent is leftExponent + rightExponent - 127, or one more: 2..253 needs no exact check.
            const __m128i productExponent =
                _mm_sub_epi32(_mm_add_epi32(leftExponent, rightExponent), _mm_set1_epi32(127));
            const __m128i normal = _mm_and_si128(_mm_cmpgt_epi32(productExponent, _mm_set1_epi32(1)),
                                                 _mm_cmpgt_epi32(_mm_set1_epi32(253), productExponent));
            const uint32_t zeroProducts = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(zero))) & active;
            const uint32_t normalProducts = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(normal))) & active;
            if ((zeroProducts | normalProducts) != active)
                return false;
            const uint32_t productNegative =
                static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(_mm_xor_si128(leftBits, rightBits)))) & active;
            sticky = (zeroProducts != 0u ? 0x1u : 0u) | (productNegative != 0u ? 0x2u : 0u);
        }

    }

    mac = static_cast<uint32_t>(kReverse4[zeroLanes]) | (static_cast<uint32_t>(kReverse4[signLanes]) << 4) |
          (static_cast<uint32_t>(kReverse4[underflowLanes]) << 8) |
          (static_cast<uint32_t>(kReverse4[overflowLanes]) << 12);
    status = (zeroLanes != 0u ? 0x1u : 0u) | (signLanes != 0u ? 0x2u : 0u) | (underflowLanes != 0u ? 0x4u : 0u) |
             (overflowLanes != 0u ? 0x8u : 0u);
    return true;
}

PS2X_VU_INLINE void VU1Interpreter::fmacFlags(float *result, uint8_t dest)
{
    // dest 0 writes no flags (updateFmacFlags).
    if (dest == 0u)
        return;
    uint32_t mac = 0u, status = 0u, sticky = 0u;
    if (!fmacFlagsFast(result, dest, mac, status, sticky))
    {
        fmacFlagsSlow(result, dest);
        return;
    }
    FlagPipelineEntry *entry = pushFlagEntry(true, 0xFFFFFFFFu);
    if (!entry)
        return;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = sticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

PS2X_VU_INLINE void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    fmacFlags(result, dest);
    applyDest(dst, result, dest);
}

PS2X_VU_INLINE void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    fmacFlags(result, dest);
    applyDestAcc(result, dest);
}

PS2X_VU_INLINE void VU1Interpreter::queueFsset(uint16_t immediate)
{
    // FSSET overrides the status flags an upper instruction of the same pair would write.
    forEachFlagEntryIssuedThisCycle([](FlagPipelineEntry &entry) { entry.writesStatus = false; });
    if (FlagPipelineEntry *entry = pushFlagEntry(false, 0xFFFFFFFEu))
    {
        entry->status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry->writesSticky = true;
    }
}

PS2X_VU_INLINE void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (FlagPipelineEntry *entry = pushFlagEntry(true, 0xFFFFFFFDu))
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
    }
}

PS2X_VU_INLINE void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    forEachFlagEntryIssuedThisCycle([](FlagPipelineEntry &entry) { entry.writesClip = false; });
    if (FlagPipelineEntry *entry = pushFlagEntry(false, 0xFFFFFFFAu))
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
    }
}

PS2X_VU_INLINE VU1Interpreter::FlagPipelineEntry *VU1Interpreter::pushFlagEntry(bool upper, uint32_t overflowCode)
{
    // Every flag write has the FMAC latency, so the queue is ordered by readiness and commits in issue order.
    if (m_flagCount == kMaxFlagEntries)
        commitFlags(); // entries whose latency elapsed are only waiting for a reader
    if (m_flagCount == kMaxFlagEntries)
    {
        reportReservedInstruction(upper, overflowCode);
        return nullptr;
    }
    FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + m_flagCount) % kMaxFlagEntries];
    ++m_flagCount;
    entry.readyCycle = m_cycle + kFmacLatency;
    entry.issueCycle = m_cycle;
    entry.mac = 0u;
    entry.status = 0u;
    entry.extraSticky = 0u;
    entry.clip = 0u;
    entry.valid = true;
    entry.writesMac = false;
    entry.writesStatus = false;
    entry.writesSticky = false;
    entry.writesClip = false;
    return &entry;
}

PS2X_VU_INLINE void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    noteCommitCycle(m_nextScalarCommit, m_fdiv.readyCycle);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

PS2X_VU_INLINE void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            noteCommitCycle(m_nextScalarCommit, entry.readyCycle);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

PS2X_VU_INLINE void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    // A store becomes visible at the next cycle boundary, before any later instruction or PATH1 transfer
    // reads VU memory, so writing it now is indistinguishable.
    if (!m_activeVuData || address + 16u > m_activeVuDataSize)
        return;
    if (m_cycle + 1u > m_lastWritebackCycle)
        m_lastWritebackCycle = m_cycle + 1u;
    uint32_t *target = reinterpret_cast<uint32_t *>(m_activeVuData + address);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((laneMask & laneForComponent(component)) != 0u)
            target[component] = words[component];
    }
}

PS2X_VU_INLINE void VU1Interpreter::commitReadyPipelines()
{
    // Register and memory results are written at issue (see run()); only flags, Q and P are delayed,
    // and flags wait for a reader (commitFlags).
    if (m_cycle >= m_nextScalarCommit)
        commitScalarPipelines();
}

PS2X_VU_INLINE void VU1Interpreter::resetXgkick()
{
    // Everything but the packet buffer, which is only read up to copiedBytes.
    m_xgkick.sourceAddress = 0u;
    m_xgkick.totalBytes = 0u;
    m_xgkick.copiedBytes = 0u;
    m_xgkick.currentTagEnd = 0u;
    m_xgkick.cycleCredit = 0u;
    m_xgkick.issueCycle = 0u;
    m_xgkick.active = false;
    m_xgkick.currentTagEop = false;
}

PS2X_VU_INLINE void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    resetXgkick();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

PS2X_VU_INLINE void VU1Interpreter::advanceOneCycle()
{
    advanceTo(m_cycle + 1u);
}

PS2X_VU_INLINE void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    if (targetCycle <= m_cycle)
        return;
    // Nothing executes while the clock jumps, so pending results and PATH1 progress can be applied in one step.
    const uint64_t elapsed = targetCycle - m_cycle;
    m_cycle = targetCycle;
    m_state.cycles = m_cycle;
    commitReadyPipelines();
    progressXgkick(elapsed);
}

PS2X_VU_INLINE uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    const auto laneReady = [&ready](const std::array<uint64_t, 4> &laneCycles, uint8_t lanes) PS2X_VU_LAMBDA_INLINE
    {
        if (lanes & 0x8u)
            ready = std::max(ready, laneCycles[0]);
        if (lanes & 0x4u)
            ready = std::max(ready, laneCycles[1]);
        if (lanes & 0x2u)
            ready = std::max(ready, laneCycles[2]);
        if (lanes & 0x1u)
            ready = std::max(ready, laneCycles[3]);
    };
    // Written out for each usage (no loop over pointers) so that compiled microcode folds the decode.
    const auto usageReady = [&](const InstructionUsage &usage) PS2X_VU_LAMBDA_INLINE
    {
        for (uint32_t index = 0; index < usage.vfReadCount; ++index)
        {
            const VfAccess &read = usage.vfRead[index];
            if (m_vfReadyMax[read.reg] > ready)
                laneReady(m_vfReady[read.reg], read.lanes);
        }
        for (uint32_t mask = usage.viRead & 0xFFFEu; mask != 0u; mask &= mask - 1u)
            ready = std::max(ready, m_viReady[static_cast<uint32_t>(_tzcnt_u32(mask))]);
        if (usage.accRead != 0u && m_accReadyMax > ready)
            laneReady(m_accReady, usage.accRead);
    };
    usageReady(decoded.upperUsage);
    usageReady(decoded.lowerUsage);

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

PS2X_VU_INLINE void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    // Writeback completion of everything issued so far; the end-of-program flush waits for it.
    const auto noteWriteback = [this](uint64_t readyCycle)
    {
        if (readyCycle > m_lastWritebackCycle)
            m_lastWritebackCycle = readyCycle;
    };
    const auto setLanes = [](std::array<uint64_t, 4> &laneCycles, uint8_t lanes, uint64_t readyCycle)
    {
        if (lanes & 0x8u)
            laneCycles[0] = readyCycle;
        if (lanes & 0x4u)
            laneCycles[1] = readyCycle;
        if (lanes & 0x2u)
            laneCycles[2] = readyCycle;
        if (lanes & 0x1u)
            laneCycles[3] = readyCycle;
    };

    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        setLanes(m_vfReady[lowerWrite.reg], lowerWrite.lanes, m_cycle + latency);
        m_vfReadyMax[lowerWrite.reg] = std::max(m_vfReadyMax[lowerWrite.reg], m_cycle + latency);
        noteWriteback(m_cycle + latency);
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        setLanes(m_vfReady[upperWrite.reg], upperWrite.lanes, m_cycle + latency);
        m_vfReadyMax[upperWrite.reg] = std::max(m_vfReadyMax[upperWrite.reg], m_cycle + latency);
        noteWriteback(m_cycle + latency);
    }

    if (const uint32_t viWrites = decoded.lowerUsage.viWrite & 0xFFFEu; viWrites != 0u)
    {
        const uint64_t readyCycle = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency
                                                                                   : decoded.lowerUsage.latency);
        for (uint32_t mask = viWrites; mask != 0u; mask &= mask - 1u)
            m_viReady[static_cast<uint32_t>(_tzcnt_u32(mask))] = readyCycle;
        noteWriteback(readyCycle);
    }
    if (decoded.upperUsage.accWrite != 0u)
    {
        setLanes(m_accReady, decoded.upperUsage.accWrite, m_cycle + kAccForwardLatency);
        m_accReadyMax = std::max(m_accReadyMax, m_cycle + kAccForwardLatency);
        noteWriteback(m_cycle + kAccForwardLatency);
    }
}

PS2X_VU_INLINE void VU1Interpreter::logInFlightWrite(InFlightWrite::Kind kind, uint8_t reg, uint8_t lanes, uint64_t readyCycle,
                                      const float oldValue[4], const float newValue[4])
{
    // Writes become ready at most a few cycles after issue, so the oldest entry of the ring has
    // always landed when it is overwritten.
    InFlightWrite &entry = m_inFlight[m_inFlightNext];
    m_inFlightNext = (m_inFlightNext + 1u) % kInFlightWrites;
    if (m_inFlightCount < kInFlightWrites)
        ++m_inFlightCount;
    entry.kind = kind;
    entry.reg = reg;
    entry.lanes = lanes;
    entry.readyCycle = readyCycle;
    std::memcpy(entry.oldValue, oldValue, sizeof(entry.oldValue));
    std::memcpy(entry.newValue, newValue, sizeof(entry.newValue));
}

PS2X_VU_INLINE void VU1Interpreter::progressXgkick(uint64_t cycles)
{
    if (m_xgkick.active)
        progressXgkickActive(cycles);
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
PS2X_VU_INLINE void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    // NOP (special 0x2F/0x30) is the most common upper instruction.
    const uint32_t low11 = instr & 0x7FFu;
    if (low11 == 0x2FFu || low11 == 0x33Cu)
        return;

    float *vd = m_state.vf[fd];
    alignas(16) float normalizedVs[4];
    alignas(16) float normalizedVt[4];
    alignas(16) float normalizedAcc[4];
    normalizeVector(m_state.vf[fs], normalizedVs);
    normalizeVector(m_state.vf[ft], normalizedVt);
    normalizeVector(m_state.acc, normalizedAcc);
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(bc)));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(bc))));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc))));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc)))));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        const float bc = m_state.vf[ft][op & 3];
        for (int c = 0; c < 4; c++)
            result[c] = vuMax(m_state.vf[fs][c], bc);
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        const float bc = m_state.vf[ft][op & 3];
        for (int c = 0; c < 4; c++)
            result[c] = vuMini(m_state.vf[fs][c], bc);
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = vuMax(m_state.vf[fs][c], m_state.i);
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = vuMini(m_state.vf[fs][c], m_state.i);
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(q)));
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(i)));
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(q))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q)))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(i))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i)))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_load_ps(vt)));
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = vuMax(m_state.vf[fs][c], m_state.vf[ft][c]);
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_load_ps(vt))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt)))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = vuSub(acc[0], vs[1] * vt[2]);
        result[1] = vuSub(acc[1], vs[2] * vt[0]);
        result[2] = vuSub(acc[2], vs[0] * vt[1]);
        result[3] = 0.0f;
        applyFmacDest(vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = vuMini(m_state.vf[fs][c], m_state.vf[ft][c]);
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(bc)));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(bc))));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc))));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc)))));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1D: // ABS (clears the sign bit; the magnitude bits pass through)
            for (int c = 0; c < 4; c++)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[fs][c], sizeof(bits));
                bits &= 0x7FFFFFFFu;
                std::memcpy(&result[c], &bits, sizeof(bits));
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(q)));
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(i)));
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(q))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q)))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(i))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i)))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_load_ps(vt)));
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_load_ps(vt))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt)))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestAcc(result, dest);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}

// ============================================================================
// Lower instructions
// ============================================================================
PS2X_VU_INLINE void VU1Interpreter::execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr)
{
    (void)upperInstr;
    if (instr == 0x00000000 || instr == 0x8000033C) // NOP
        return;

    uint8_t opHi = (instr >> 25) & 0x7F;
    const uint32_t pcMask = microAddressMask();

    // The lower instruction encoding uses bits 31:25 for the primary opcode
    switch (opHi)
    {
    case 0x00: // LQ (Load Quadword from VU data memory)
    {
        uint8_t it = FT(instr);  // VF destination
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            applyDest(m_state.vf[it], tmp, dest);
        }
        return;
    }
    case 0x01: // SQ (Store Quadword to VU data memory)
    {
        uint8_t is = FS(instr);  // VF source
        uint8_t it = VIT(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[it] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            uint32_t words[4]{};
            std::memcpy(words, m_state.vf[is], sizeof(words));
            queueStore(addr, words, dest);
        }
        return;
    }
    case 0x04: // ILW (Integer Load Word from VU data memory)
    {
        uint8_t it = VIT(instr); // VI destination
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            int comp = 0;
            if (dest & 0x8)
                comp = 0;
            else if (dest & 0x4)
                comp = 1;
            else if (dest & 0x2)
                comp = 2;
            else
                comp = 3;
            uint32_t v;
            std::memcpy(&v, vuData + addr + comp * 4, 4);
            if (it != 0)
                m_state.vi[it] = (int32_t)(int16_t)(v & 0xFFFF);
        }
        return;
    }
    case 0x05: // ISW (Integer Store Word to VU data memory)
    {
        uint8_t it = VIT(instr); // VI source
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            const uint32_t val = static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[it] & 0xFFFF));
            const uint32_t words[4] = {val, val, val, val};
            queueStore(addr, words, dest);
        }
        return;
    }
    case 0x08: // IADDIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] + imm);
        return;
    }
    case 0x09: // ISUBIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] - imm);
        return;
    }
    case 0x10: // FCEQ
    {
        commitFlags();
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & 0xFFFFFF) == imm24) ? 1 : 0;
        return;
    }
    case 0x11: // FCSET
    {
        queueFcset(instr & 0xFFFFFFu);
        return;
    }
    case 0x12: // FCAND
    {
        commitFlags();
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & imm24) != 0) ? 1 : 0;
        return;
    }
    case 0x13: // FCOR
    {
        commitFlags();
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip | imm24) == 0xFFFFFF) ? 1 : 0;
        return;
    }
    case 0x14: // FSEQ
    {
        commitFlags();
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = ((m_state.status & 0xFFFu) == imm12) ? 1 : 0;
        return;
    }
    case 0x15: // FSSET
    {
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        queueFsset(imm12);
        return;
    }
    case 0x16: // FSAND
    {
        commitFlags();
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>((m_state.status & 0xFFFu) & imm12);
        return;
    }
    case 0x17: // FSOR
    {
        commitFlags();
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>((m_state.status & 0xFFFu) | imm12);
        return;
    }
    case 0x18: // FMEQ
    {
        commitFlags();
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = ((m_state.mac & 0xFFFF) == (uint32_t)(uint16_t)m_state.vi[is]) ? 1 : 0;
        return;
    }
    case 0x1A: // FMAND
    {
        commitFlags();
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac & (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1B: // FMOR
    {
        commitFlags();
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac | (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1C: // FCGET
    {
        commitFlags();
        const uint8_t it = VIT(instr);
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>(m_state.clip & 0x0FFFu);
        return;
    }
    case 0x20: // B (unconditional branch)
    {
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x21: // BAL (Branch and link)
    {
        uint8_t it = VIT(instr);
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    // JR/JALR read the register as written by the previous instruction: only the conditional
    // branches see the old value (PCSX2 microVU analyses the VI delay for those alone). VP2 does
    // IADDIU vi1, vi0, 1301 followed by JALR vi15, vi1 to call its loader at 0x28A8.
    case 0x24: // JR
    {
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)m_state.vi[is] * 8u) & pcMask;
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x25: // JALR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)m_state.vi[is] * 8u) & pcMask;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x28: // IBEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) == (int16_t)readBranchVi(it))
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x29: // IBNE
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) != (int16_t)readBranchVi(it))
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2C: // IBLTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) < 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2D: // IBGTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) > 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2E: // IBLEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) <= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2F: // IBGEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) >= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }

    case 0x40: // Lower1 / lower special. Bit31 set; low 6 bits select integer or special op.
    {
        const uint8_t funct = instr & 0x3Fu;
        const uint8_t vfT = FT(instr);
        const uint8_t vfS = FS(instr);
        const uint8_t viT = VIT(instr);
        const uint8_t viS = VIS(instr);
        const uint8_t viD = VID(instr);
        const uint8_t dest = (instr >> 21) & 0xF;

        switch (funct)
        {
        case 0x30: // IADD
            if (viD != 0)
                m_state.vi[viD] = (int16_t)(m_state.vi[viS] + m_state.vi[viT]);
            return;
        case 0x31: // ISUB
            if (viD != 0)
                m_state.vi[viD] = (int16_t)(m_state.vi[viS] - m_state.vi[viT]);
            return;
        case 0x32: // IADDI
        {
            int16_t imm5 = (int16_t)((int32_t)((instr >> 6) & 0x1F) << 27 >> 27);
            if (viT != 0)
                m_state.vi[viT] = (int16_t)(m_state.vi[viS] + imm5);
            return;
        }
        case 0x34: // IAND
            if (viD != 0)
                m_state.vi[viD] = m_state.vi[viS] & m_state.vi[viT];
            return;
        case 0x35: // IOR
            if (viD != 0)
                m_state.vi[viD] = m_state.vi[viS] | m_state.vi[viT];
            return;

        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F: // Lower1 special. Dobie decodes this as (instr & 3) | ((instr >> 4) & 0x7C).
        {
            const uint8_t funct2 = (uint8_t)((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
            switch (funct2)
            {
            case 0x30: // MOVE
            {
                float tmp[4];
                std::memcpy(tmp, m_state.vf[vfS], 16);
                applyDest(m_state.vf[vfT], tmp, dest);
                return;
            }
            case 0x31: // MR32 (rotate right by 32 bits = shift xyzw -> yzwx)
            {
                float tmp[4] = {m_state.vf[vfS][1], m_state.vf[vfS][2], m_state.vf[vfS][3], m_state.vf[vfS][0]};
                applyDest(m_state.vf[vfT], tmp, dest);
                return;
            }
            case 0x34: // LQI (Load Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[vfT], tmp, dest);
                }
                if (viS != 0)
                    m_state.vi[viS] = (int16_t)(m_state.vi[viS] + 1);
                return;
            }
            case 0x35: // SQI (Store Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, m_state.vf[vfS], sizeof(words));
                    queueStore(addr, words, dest);
                }
                if (viT != 0)
                    m_state.vi[viT] = (int16_t)(m_state.vi[viT] + 1);
                return;
            }
            case 0x36: // LQD (Load Quadword, pre-decrement)
            {
                if (viS != 0)
                    m_state.vi[viS] = (int16_t)(m_state.vi[viS] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[vfT], tmp, dest);
                }
                return;
            }
            case 0x37: // SQD (Store Quadword, pre-decrement)
            {
                if (viT != 0)
                    m_state.vi[viT] = (int16_t)(m_state.vi[viT] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, m_state.vf[vfS], sizeof(words));
                    queueStore(addr, words, dest);
                }
                return;
            }
            case 0x38: // DIV
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                const float num = normalizeOperand(m_state.vf[vfS][fsf]);
                const float den = normalizeOperand(m_state.vf[vfT][ftf]);
                uint32_t statusDi = 0u;
                float result = 0.0f;
                if (den == 0.0f)
                {
                    statusDi = num == 0.0f ? 0x10u : 0x20u;
                    result = std::signbit(num) != std::signbit(den)
                                 ? -std::numeric_limits<float>::max()
                                 : std::numeric_limits<float>::max();
                }
                else
                {
                    result = num / den;
                }
                uint32_t ignoredFlags = 0u;
                result = normalizeResult(result, ignoredFlags);
                queueQ(result, 7u, statusDi);
                return;
            }
            case 0x39: // SQRT
            {
                int ftf = (instr >> 23) & 0x3;
                const float val = normalizeOperand(m_state.vf[vfT][ftf]);
                queueQ(std::sqrt(std::fabs(val)), 7u,
                       val < 0.0f ? 0x10u : 0u);
                return;
            }
            case 0x3A: // RSQRT
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                const float num = normalizeOperand(m_state.vf[vfS][fsf]);
                const float radicand = normalizeOperand(m_state.vf[vfT][ftf]);
                const float den = std::sqrt(std::fabs(radicand));
                uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
                float result = 0.0f;
                if (den != 0.0f)
                    result = num / den;
                else
                {
                    statusDi = num == 0.0f ? 0x10u : 0x20u;
                    result = std::signbit(num)
                                 ? -std::numeric_limits<float>::max()
                                 : std::numeric_limits<float>::max();
                }
                uint32_t ignoredFlags = 0u;
                result = normalizeResult(result, ignoredFlags);
                queueQ(result, 13u, statusDi);
                return;
            }
            case 0x3B: // WAITQ
                return;
            case 0x3C: // MTIR (Move To Integer Register)
            {
                // MTIR encodes a two-bit fsf component selector in bits
                // 22:21. It is not a four-bit destination mask.
                const uint32_t comp = (instr >> 21) & 0x3u;
                uint32_t fval;
                std::memcpy(&fval, &m_state.vf[vfS][comp], 4);
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(int16_t)(fval & 0xFFFF);
                return;
            }
            case 0x3D: // MFIR (Move From Integer Register)
            {
                float result[4];
                int32_t val = (int32_t)(int16_t)(m_state.vi[viS] & 0xFFFF);
                std::memcpy(&result[0], &val, 4);
                result[1] = result[0];
                result[2] = result[0];
                result[3] = result[0];
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x3E: // ILWR - integer load word from address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    int comp = 0;
                    if (dest & 0x8)
                        comp = 0;
                    else if (dest & 0x4)
                        comp = 1;
                    else if (dest & 0x2)
                        comp = 2;
                    else
                        comp = 3;
                    uint32_t v;
                    std::memcpy(&v, vuData + addr + comp * 4, 4);
                    if (viT != 0)
                        m_state.vi[viT] = (int32_t)(int16_t)(v & 0xFFFF);
                }
                return;
            }
            case 0x3F: // ISWR - integer store word to address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    const uint32_t val =
                        static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viT] & 0xFFFF));
                    const uint32_t words[4] = {val, val, val, val};
                    queueStore(addr, words, dest);
                }
                return;
            }
            case 0x40: // RNEXT
            {
                const uint32_t x = (m_state.r >> 4) & 1u;
                const uint32_t y = (m_state.r >> 22) & 1u;
                m_state.r = ((m_state.r << 1) ^ x ^ y) & 0x007FFFFFu;
                m_state.r |= 0x3F800000u;
                float value = 0.0f;
                std::memcpy(&value, &m_state.r, sizeof(value));
                const float result[4] = {value, value, value, value};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x41: // RGET
            {
                float value = 0.0f;
                std::memcpy(&value, &m_state.r, sizeof(value));
                const float result[4] = {value, value, value, value};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x42: // RINIT
            {
                const uint32_t component = (instr >> 21) & 3u;
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[vfS][component], sizeof(bits));
                m_state.r = 0x3F800000u | (bits & 0x007FFFFFu);
                return;
            }
            case 0x43: // RXOR
            {
                const uint32_t component = (instr >> 21) & 3u;
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[vfS][component], sizeof(bits));
                m_state.r = 0x3F800000u | ((m_state.r ^ bits) & 0x007FFFFFu);
                return;
            }
            case 0x64: // MFP (Move From P register)
            {
                float result[4] = {m_state.p, m_state.p, m_state.p, m_state.p};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x68: // XTOP - move current VIF1 TOP into VI register
            {
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(m_state.top & 0x3FFu);
                return;
            }
            case 0x69: // XITOP - move current VIF1 ITOP into VI register
            {
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(m_state.itop & 0x3FFu);
                return;
            }
            case 0x6C: // XGKICK - send GIF packet from VU1 data memory
                startXgkick(static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viS])));
                return;
            case 0x70: // ESADD
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                queueP(x * x + y * y + z * z, 11u);
                return;
            }
            case 0x71: // ERSADD
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                const float sum = x * x + y * y + z * z;
                queueP(sum != 0.0f ? 1.0f / sum : sum, 18u);
                return;
            }
            case 0x72: // ELENG
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                queueP(std::sqrt(x * x + y * y + z * z), 18u);
                return;
            }
            case 0x73: // ERLENG
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                const float len = std::sqrt(x * x + y * y + z * z);
                queueP(len != 0.0f ? 1.0f / len : len, 24u);
                return;
            }
            case 0x74: // EATANxy
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float y = normalizeOperand(m_state.vf[vfS][1]);
                queueP(x != 0.0f ? vuEatan(y / x) : 0.0f, 54u);
                return;
            }
            case 0x75: // EATANxz
            {
                const float x = normalizeOperand(m_state.vf[vfS][0]);
                const float z = normalizeOperand(m_state.vf[vfS][2]);
                queueP(x != 0.0f ? vuEatan(z / x) : 0.0f, 54u);
                return;
            }
            case 0x76: // ESUM
            {
                float sum = 0.0f;
                for (uint32_t component = 0; component < 4u; ++component)
                    sum += normalizeOperand(m_state.vf[vfS][component]);
                queueP(sum, 12u);
                return;
            }
            case 0x77: // ERSQRT
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                float result = value;
                if (result >= 0.0f)
                {
                    result = std::sqrt(result);
                    if (result != 0.0f)
                        result = 1.0f / result;
                }
                queueP(result, 18u);
                return;
            }
            case 0x78: // ESQRT
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                queueP(value >= 0.0f ? std::sqrt(value) : value, 12u);
                return;
            }
            case 0x79: // ESIN
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                queueP(vuEsin(value), 29u);
                return;
            }
            case 0x7A: // ERCPR
            {
                const uint32_t component = (instr >> 21) & 3u;
                const float value = normalizeOperand(m_state.vf[vfS][component]);
                queueP(value != 0.0f ? 1.0f / value : value, 12u);
                return;
            }
            case 0x7B: // WAITP
                return;
            case 0x7C: // EATAN
            {
                const uint32_t component = (instr >> 21) & 3u;
                queueP(vuEatan(normalizeOperand(m_state.vf[vfS][component])), 54u);
                return;
            }
            case 0x7D: // EEXP
            {
                const uint32_t component = (instr >> 21) & 3u;
                queueP(vuEexp(normalizeOperand(m_state.vf[vfS][component])), 44u);
                return;
            }
            default:
                reportReservedInstruction(false, instr);
                return;
            }
        }
        default:
            reportReservedInstruction(false, instr);
            return;
        }
    }
    default:
        reportReservedInstruction(false, instr);
        break;
    }
}

PS2X_VU_INLINE VU1Interpreter::StepResult VU1Interpreter::stepPair(const DecodedInstructionPair &decoded,
                                                                  const RunContext &context)
{
    // Time-sliced runs keep the previous value of every register they write (logInFlightWrite): out of line.
    if (m_logInFlight)
        return stepPairLogged(decoded, context);
    return stepPairImpl<false>(decoded, context);
}

template <bool kLogInFlight>
PS2X_VU_INLINE VU1Interpreter::StepResult VU1Interpreter::stepPairImpl(const DecodedInstructionPair &decoded,
                                                                      const RunContext &context)
{
    if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
    {
        reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
        return StepResult::Stop;
    }

    const uint64_t budgetEnd = context.budgetEnd;
    // The caller checked m_cycle < budgetEnd (canIssue): only a stall can reach the budget.
    const uint64_t readyCycle = calculatePairReadyCycle(decoded);
    if (readyCycle > m_cycle)
    {
        stallUntilReady(decoded, readyCycle, budgetEnd);
        if (m_cycle >= budgetEnd)
            return StepResult::Stop;
    }

    uint8_t writtenVi = 0u;
    int32_t oldVi = 0;
    if (decoded.lowerUsage.viWrite != 0u)
    {
        for (uint32_t reg = 1; reg < 16u; ++reg)
        {
            if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
            {
                writtenVi = static_cast<uint8_t>(reg);
                oldVi = m_state.vi[reg];
                break;
            }
        }
    }

    // Results are written immediately. Every reader stalls until its operands' latency has elapsed
    // (calculatePairReadyCycle), so the delayed writeback is unobservable inside a run; only the same-pair
    // case (lower reads the register upper writes) needs the old value. A run that stops on its cycle
    // budget hides writes still in flight (logInFlightWrites / hideInFlightWrites).
    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    const bool logLowerVf = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg &&
                            lowerWrite.reg != upperWrite.reg;
    alignas(16) float oldUpperVf[4];
    alignas(16) float oldLowerVf[4];
    alignas(16) float oldAcc[4];
    if constexpr (kLogInFlight)
    {
        if (upperWrite.reg != 0u)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (logLowerVf)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
        if (decoded.upperUsage.accWrite != 0u)
            std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));
    }

    if (decoded.iBit)
    {
        execUpper(decoded.upper);
        float immediate = 0.0f;
        std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
        m_state.i = normalizeOperand(immediate);
    }
    else if (decoded.upperVfShadowReg != 0u)
    {
        float oldVf[4]{};
        float upperVf[4]{};
        std::memcpy(oldVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(oldVf));
        execUpper(decoded.upper);
        std::memcpy(upperVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(upperVf));
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    oldVf,
                    sizeof(oldVf));
        execLower(decoded.lower, context.vuData, context.dataSize, *context.gs, context.memory, decoded.upper);
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    upperVf,
                    sizeof(upperVf));
    }
    else
    {
        execUpper(decoded.upper);
        execLower(decoded.lower, context.vuData, context.dataSize, *context.gs, context.memory, decoded.upper);
    }

    m_viBranchBackupValid = false;

    markPairWrites(decoded);
    if (kLogInFlight && upperWrite.reg != 0u)
    {
        const uint32_t latency =
            decoded.upperUsage.vfLatency != 0u ? decoded.upperUsage.vfLatency : decoded.upperUsage.latency;
        logInFlightWrite(InFlightWrite::Kind::Vf, upperWrite.reg, upperWrite.lanes, m_cycle + latency, oldUpperVf,
                         m_state.vf[upperWrite.reg]);
    }
    if (kLogInFlight && logLowerVf)
    {
        const uint32_t latency =
            decoded.lowerUsage.vfLatency != 0u ? decoded.lowerUsage.vfLatency : decoded.lowerUsage.latency;
        logInFlightWrite(InFlightWrite::Kind::Vf, lowerWrite.reg, lowerWrite.lanes, m_cycle + latency, oldLowerVf,
                         m_state.vf[lowerWrite.reg]);
    }
    if (kLogInFlight && decoded.upperUsage.accWrite != 0u)
        logInFlightWrite(InFlightWrite::Kind::Acc, 0u, decoded.upperUsage.accWrite, m_cycle + kAccForwardLatency,
                         oldAcc, m_state.acc);
    if (kLogInFlight && writtenVi != 0u)
    {
        const uint32_t latency =
            decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency;
        const float oldValue[4] = {static_cast<float>(oldVi), 0.0f, 0.0f, 0.0f};
        const float newValue[4] = {static_cast<float>(m_state.vi[writtenVi]), 0.0f, 0.0f, 0.0f};
        logInFlightWrite(InFlightWrite::Kind::Vi, writtenVi, 0x8u, m_cycle + latency, oldValue, newValue);
    }
    if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
        recordViWriteForBranch(writtenVi, oldVi);

    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    m_state.vi[0] = 0;

    uint32_t nextPc = m_state.pc + 8u;
    if (nextPc >= context.codeSize)
        nextPc = 0u;
    m_state.pc = nextPc;

    if (m_state.branchPending)
    {
        if (m_state.branchDelay == 0u)
        {
            m_state.pc = m_state.branchTarget & microAddressMask();
            m_state.branchPending = false;
        }
        else
        {
            --m_state.branchDelay;
        }
    }

    const bool dHalt = decoded.dBit && m_state.dBitEnabled;
    const bool tHalt = decoded.tBit && m_state.tBitEnabled;
    const bool haltBit = dHalt || tHalt;
    const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

    bool programEnded = false;
    if (m_state.haltAfterDelaySlot)
    {
        m_state.stoppedByD = m_pendingHaltD;
        m_state.stoppedByT = m_pendingHaltT;
        programEnded = true;
    }
    else if (m_state.ebit)
        programEnded = true;
    else if (haltBit && !haltBranch)
    {
        m_state.stoppedByD = dHalt;
        m_state.stoppedByT = tHalt;
        programEnded = true;
    }
    else if (decoded.eBit)
        m_state.ebit = true;
    else if (haltBranch)
    {
        m_state.haltAfterDelaySlot = true;
        m_pendingHaltD = dHalt;
        m_pendingHaltT = tHalt;
    }

    advanceOneCycle();
    return programEnded ? StepResult::Ended : StepResult::Continue;
}


#endif
