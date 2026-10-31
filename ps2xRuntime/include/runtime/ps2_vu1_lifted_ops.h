#pragma once
// The operations and pipeline helpers the C++ that tools/vu1_lift.py writes for a VU program is made of: each
// instruction as the operation it does on registers held in locals, the timing VU1Interpreter's pipeline gives it
// already worked out by the generator.

#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_lifted.h"

#include <immintrin.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// MSVC (x64: SSE only, no x87 code to round) or GCC / Clang.
#if defined(_MSC_VER) && !defined(__clang__)
#define PS2X_LIFTED_COLD __declspec(noinline)
#define PS2X_LIFTED_X87 0
#else
#define PS2X_LIFTED_COLD __attribute__((noinline, cold))
#define PS2X_LIFTED_X87 1
#endif

namespace ps2x::vu1lift
{
    // Where a lifted program hands each XGKICK packet.
    using Vu1Kick = void (*)(void *user, const uint8_t *packet, uint32_t bytes);

    constexpr uint64_t kUnknownTarget = ~0ull;

    constexpr int lanes(int dest) { return ((dest >> 3) & 1) | (((dest >> 2) & 1) << 1) | (((dest >> 1) & 1) << 2) | ((dest & 1) << 3); }

    template <int kDest>
    inline __m128 put(__m128 old, __m128 value)
    {
        if constexpr ((kDest & 15) == 15)
            return value;
        else if constexpr ((kDest & 15) == 0)
            return old;
        else
            return _mm_blend_ps(old, value, lanes(kDest));
    }

    inline __m128i bits(__m128 value) { return _mm_castps_si128(value); }
    inline __m128 floats(__m128i value) { return _mm_castsi128_ps(value); }
    inline __m128 load(const float *source) { return _mm_loadu_ps(source); }
    inline void store(float *target, __m128 value) { _mm_storeu_ps(target, value); }
    inline int32_t i16(int32_t value) { return static_cast<int16_t>(value); }

    inline __m128 norm(__m128 value)
    {
        const __m128i b = bits(value);
        const __m128i magnitudeMask = _mm_set1_epi32(0x7FFFFFFF);
        const __m128i magnitude = _mm_and_si128(b, magnitudeMask);
        const __m128i normal = _mm_cmpgt_epi32(magnitude, _mm_set1_epi32(0x007FFFFF));
        const __m128i clamped = _mm_and_si128(_mm_min_epi32(magnitude, _mm_set1_epi32(0x7F7FFFFF)), normal);
        return floats(_mm_or_si128(_mm_andnot_si128(magnitudeMask, b), clamped));
    }

    inline float normalizeOperand(float value)
    {
        uint32_t b = std::bit_cast<uint32_t>(value);
        const uint32_t exponent = (b >> 23) & 0xFFu;
        if (exponent == 0u)
            b &= 0x80000000u;
        else if (exponent == 0xFFu)
            b = (b & 0x80000000u) | 0x7F7FFFFFu;
        return std::bit_cast<float>(b);
    }

    inline uint32_t laneBits(__m128 value, int lane)
    {
        alignas(16) uint32_t words[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(words), bits(value));
        return words[lane];
    }

    inline float laneValue(__m128 value, int lane) { return std::bit_cast<float>(laneBits(value, lane)); }

    template <int kLane>
    inline __m128 spread(__m128 value)
    {
        return _mm_shuffle_ps(value, value, _MM_SHUFFLE(kLane, kLane, kLane, kLane));
    }

    enum class Op
    {
        Add,
        Sub,
        Mul,
        Madd,
        Msub,
        CrossMsub,
        CrossMul,
    };

    enum class R
    {
        Vec,
        Bc0,
        Bc1,
        Bc2,
        Bc3,
        Scalar,
    };

    struct Fmac
    {
        uint32_t sticky = 0u;
        __m128i stickySign = _mm_setzero_si128(), stickyZero = _mm_setzero_si128();
        uint32_t di = 0u, diSticky = 0u;
        bool hasDi = false;

        uint32_t status(uint32_t initial, bool hasFmac, uint32_t lastCur) const
        {
            uint32_t value = initial;
            const uint32_t gathered = sticky | (_mm_movemask_ps(floats(stickySign)) != 0 ? 0x2u : 0u) |
                                      (_mm_movemask_ps(floats(stickyZero)) != 0 ? 0x1u : 0u);
            if (hasFmac)
                value = (value & 0xFF0u) | lastCur;
            value |= (gathered & 0xFu) << 6;
            if (hasDi)
                value = (value & 0xFCFu) | di;
            value |= diSticky << 6;
            return value;
        }
    };

    inline void landDi(Fmac &f, uint32_t di)
    {
        f.di = di;
        f.diSticky |= di;
        f.hasDi = true;
    }

    constexpr ps2_vu1_detail::FmacKind fmacKind(Op op)
    {
        using K = ps2_vu1_detail::FmacKind;
        switch (op)
        {
        case Op::Add: return K::Add;
        case Op::Sub: return K::Sub;
        case Op::Mul: return K::Mul;
        case Op::Madd: return K::Madd;
        case Op::Msub: return K::Msub;
        case Op::CrossMsub: return K::CrossMsub;
        case Op::CrossMul: return K::CrossMul;
        }
        return K::None;
    }

    PS2X_LIFTED_COLD inline __m128 fmacCold(Fmac &f, __m128 value, int dest, ps2_vu1_detail::FmacKind kind, __m128 left,
                                                         __m128 right, __m128 acc, uint32_t &mac, uint32_t &cur)
    {
        alignas(16) float result[4];
        _mm_store_ps(result, value);
        ps2_vu1_detail::FmacOperation operation;
        operation.kind = kind;
        uint32_t status = 0u, sticky = 0u;
        Vu1Lifted::fmacFlags(result, static_cast<uint8_t>(dest), operation, left, right, acc, mac, status, sticky);
        f.sticky |= status | sticky;
        cur = status;
        return _mm_load_ps(result);
    }

    template <Op kOp, R kRight, int kDest, bool kSticky, bool kLive, typename Right>
    inline __m128 fmac(Fmac &f, __m128 target, __m128 vs, Right vt, __m128 accIn, uint32_t &mac, uint32_t &cur)
    {
        constexpr int kActive = lanes(kDest);
        constexpr bool kProductSum = kOp == Op::Madd || kOp == Op::Msub || kOp == Op::CrossMsub;
        constexpr bool kCross = kOp == Op::CrossMsub || kOp == Op::CrossMul;
        const __m128 left = norm(vs);
        __m128 right;
        if constexpr (kRight == R::Scalar)
            right = _mm_set1_ps(normalizeOperand(vt));
        else if constexpr (kRight == R::Vec)
            right = norm(vt);
        else
            right = spread<static_cast<int>(kRight) - static_cast<int>(R::Bc0)>(norm(vt));
        const __m128 a = kProductSum ? norm(accIn) : _mm_setzero_ps();
        __m128 l = left, r = right;
        if constexpr (kCross)
        {
            l = _mm_shuffle_ps(l, l, _MM_SHUFFLE(3, 0, 2, 1));
            r = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 1, 0, 2));
        }
        __m128 value;
        if constexpr (kOp == Op::Add)
            value = ps2_vu1_detail::vuAdd4(l, r);
        else if constexpr (kOp == Op::Sub)
            value = ps2_vu1_detail::vuAdd4(l, ps2_vu1_detail::negate4(r));
        else if constexpr (kOp == Op::Mul || kOp == Op::CrossMul)
            value = _mm_mul_ps(l, r);
        else if constexpr (kOp == Op::Madd)
            value = ps2_vu1_detail::vuAdd4(a, _mm_mul_ps(l, r));
        else
            value = ps2_vu1_detail::vuAdd4(a, ps2_vu1_detail::negate4(_mm_mul_ps(l, r)));
        if constexpr (kCross)
            value = _mm_blend_ps(value, _mm_setzero_ps(), 0x8);

        const __m128i activeLanes = _mm_setr_epi32((kActive & 1) ? -1 : 0, (kActive & 2) ? -1 : 0, (kActive & 4) ? -1 : 0,
                                                   (kActive & 8) ? -1 : 0);
        const __m128i b = bits(value);
        const __m128i exponentMask = _mm_set1_epi32(0xFF);
        const __m128i exponent = _mm_and_si128(_mm_srli_epi32(b, 23), exponentMask);
        const __m128i inRange = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1)), _mm_cmpgt_epi32(_mm_set1_epi32(254), exponent));
        const __m128i zeroResult = _mm_cmpeq_epi32(_mm_and_si128(b, _mm_set1_epi32(0x7FFFFFFF)), _mm_setzero_si128());
        __m128i zeroExact = _mm_set1_epi32(-1);
        __m128i negative = b;
        __m128i zeroProduct = _mm_setzero_si128();
        __m128i oddProduct = _mm_setzero_si128();
        if constexpr (kOp != Op::Add && kOp != Op::Sub)
        {
            const __m128i lb = bits(l), rb = bits(r);
            const __m128i le = _mm_and_si128(_mm_srli_epi32(lb, 23), exponentMask);
            const __m128i re = _mm_and_si128(_mm_srli_epi32(rb, 23), exponentMask);
            zeroProduct = _mm_or_si128(_mm_cmpeq_epi32(le, _mm_setzero_si128()), _mm_cmpeq_epi32(re, _mm_setzero_si128()));
            zeroExact = kCross ? _mm_or_si128(zeroProduct, _mm_setr_epi32(0, 0, 0, -1)) : zeroProduct;
            if constexpr (kProductSum)
            {
                const __m128i pe = _mm_sub_epi32(_mm_add_epi32(le, re), _mm_set1_epi32(127));
                const __m128i normalProduct = _mm_and_si128(_mm_cmpgt_epi32(pe, _mm_set1_epi32(1)), _mm_cmpgt_epi32(_mm_set1_epi32(253), pe));
                oddProduct = _mm_andnot_si128(_mm_or_si128(zeroProduct, normalProduct), activeLanes);
                negative = _mm_or_si128(negative, _mm_xor_si128(lb, rb));
            }
        }
        const __m128i exact =
            _mm_or_si128(_mm_andnot_si128(_mm_or_si128(inRange, _mm_and_si128(zeroResult, zeroExact)), activeLanes), oddProduct);
        if (_mm_movemask_ps(floats(exact)) != 0) [[unlikely]]
        {
            uint32_t m = 0u, c = 0u;
            value = fmacCold(f, value, kDest, fmacKind(kOp), left, right, a, m, c);
            if constexpr (kLive)
                mac = m, cur = c;
            return put<kDest>(target, value);
        }
        if constexpr (kSticky)
        {
            f.stickySign = _mm_or_si128(f.stickySign, _mm_and_si128(negative, activeLanes));
            f.stickyZero = _mm_or_si128(f.stickyZero, _mm_and_si128(kProductSum ? _mm_or_si128(zeroResult, zeroProduct) : zeroResult, activeLanes));
        }
        if constexpr (kLive)
        {
            static constexpr uint8_t kReverse4[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
            const uint32_t zeros = static_cast<uint32_t>(_mm_movemask_ps(floats(zeroResult))) & kActive;
            const uint32_t signs = static_cast<uint32_t>(_mm_movemask_ps(value)) & kActive;
            mac = kReverse4[zeros] | (static_cast<uint32_t>(kReverse4[signs]) << 4);
            cur = (zeros ? 0x1u : 0u) | (signs ? 0x2u : 0u);
        }
        return put<kDest>(target, value);
    }

    inline __m128i minMaxKey(__m128 value)
    {
        const __m128i b = bits(value);
        const __m128i sign = _mm_srai_epi32(b, 31);
        return _mm_sub_epi32(_mm_xor_si128(_mm_and_si128(b, _mm_set1_epi32(0x7FFFFFFF)), sign), sign);
    }

    template <R kRight, typename Right>
    inline __m128 rawRight(Right vt)
    {
        if constexpr (kRight == R::Scalar)
            return _mm_set1_ps(vt);
        else if constexpr (kRight == R::Vec)
            return vt;
        else
            return spread<static_cast<int>(kRight) - static_cast<int>(R::Bc0)>(vt);
    }

    template <R kRight, int kDest, typename Right>
    inline __m128 maxv(__m128 target, __m128 vs, Right vt)
    {
        const __m128 t = rawRight<kRight>(vt);
        const __m128 pickS = floats(_mm_cmpgt_epi32(minMaxKey(vs), minMaxKey(t)));
        return put<kDest>(target, _mm_blendv_ps(t, vs, pickS));
    }

    template <R kRight, int kDest, typename Right>
    inline __m128 miniv(__m128 target, __m128 vs, Right vt)
    {
        const __m128 t = rawRight<kRight>(vt);
        const __m128 pickS = floats(_mm_cmpgt_epi32(minMaxKey(t), minMaxKey(vs)));
        return put<kDest>(target, _mm_blendv_ps(t, vs, pickS));
    }

    template <int kScaleBits, int kDest>
    inline __m128 ftoi(__m128 target, __m128 vs)
    {
        __m128 scaled = norm(vs);
        if constexpr (kScaleBits != 0)
            scaled = _mm_mul_ps(scaled, _mm_set1_ps(static_cast<float>(1u << kScaleBits)));
        const __m128i truncated = _mm_cvttps_epi32(scaled);
        const __m128 overflow = _mm_cmpge_ps(scaled, _mm_set1_ps(2147483648.0f));
        return put<kDest>(target, _mm_blendv_ps(floats(truncated), floats(_mm_set1_epi32(0x7FFFFFFF)), overflow));
    }

    template <int kScaleBits, int kDest>
    inline __m128 itof(__m128 target, __m128 vs)
    {
        __m128 value = _mm_cvtepi32_ps(bits(vs));
        if constexpr (kScaleBits != 0)
            value = _mm_div_ps(value, _mm_set1_ps(static_cast<float>(1u << kScaleBits)));
        return put<kDest>(target, value);
    }

    inline __m128 rotate(__m128 value) { return _mm_shuffle_ps(value, value, _MM_SHUFFLE(0, 3, 2, 1)); }

    inline uint32_t clipFlags(__m128 vs, __m128 vt)
    {
        const uint32_t wBits = laneBits(vt, 3);
        const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
        alignas(16) uint32_t w[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(w), bits(vs));
        uint32_t flags = 0u;
        flags |= static_cast<int32_t>(w[0]) > limit ? 0x01u : 0u;
        flags |= static_cast<int32_t>(w[0] ^ 0x80000000u) > limit ? 0x02u : 0u;
        flags |= static_cast<int32_t>(w[1]) > limit ? 0x04u : 0u;
        flags |= static_cast<int32_t>(w[1] ^ 0x80000000u) > limit ? 0x08u : 0u;
        flags |= static_cast<int32_t>(w[2]) > limit ? 0x10u : 0u;
        flags |= static_cast<int32_t>(w[2] ^ 0x80000000u) > limit ? 0x20u : 0u;
        return flags;
    }

    inline float divide(float num, float den, uint32_t &di)
    {
        num = normalizeOperand(num);
        den = normalizeOperand(den);
        float result;
        if (den == 0.0f)
        {
            di = num == 0.0f ? 0x10u : 0x20u;
            result = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        }
        else
        {
            di = 0u;
            result = num / den;
        }
        return normalizeOperand(result);
    }

    inline float squareRoot(float value, uint32_t &di)
    {
        value = normalizeOperand(value);
        di = value < 0.0f ? 0x10u : 0u;
        return normalizeOperand(std::sqrt(std::fabs(value)));
    }

    inline float reciprocalRoot(float num, float radicand, uint32_t &di)
    {
        num = normalizeOperand(num);
        radicand = normalizeOperand(radicand);
        const float den = std::sqrt(std::fabs(radicand));
        di = radicand < 0.0f ? 0x10u : 0u;
        float result;
        if (den != 0.0f)
            result = num / den;
        else
        {
            di = num == 0.0f ? 0x10u : 0x20u;
            result = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        }
        return normalizeOperand(result);
    }

    template <uint32_t kQwordMask>
    struct Memory
    {
        uint8_t *data;

        uint8_t *at(int32_t qword) const { return data + ((static_cast<uint32_t>(qword) & kQwordMask) << 4); }
        __m128 lq(int32_t qword) const { return _mm_loadu_ps(reinterpret_cast<const float *>(at(qword))); }
        template <int kDest>
        void sq(int32_t qword, __m128 value) const
        {
            float *target = reinterpret_cast<float *>(at(qword));
            _mm_storeu_ps(target, put<kDest>(_mm_loadu_ps(target), value));
        }
        int32_t ilw(int32_t qword, int component) const
        {
            uint32_t value;
            std::memcpy(&value, at(qword) + component * 4, 4);
            return static_cast<int16_t>(value & 0xFFFFu);
        }
        template <int kDest>
        void isw(int32_t qword, int32_t value) const
        {
            sq<kDest>(qword, floats(_mm_set1_epi32(static_cast<int32_t>(static_cast<uint16_t>(value & 0xFFFF)))));
        }
    };

    struct Path1
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        static constexpr uint32_t kMemory = 16u * 1024u;

        bool pending = false, done = false, eop = false;
        uint32_t source = 0u, copied = 0u, tagEnd = 0u, total = 0u;
        int64_t start = 0;

        static std::array<uint8_t, kBufferSize> &buffer()
        {
            static thread_local std::array<uint8_t, kBufferSize> s_buffer;
            return s_buffer;
        }

        void begin(int32_t vi, int64_t t)
        {
            pending = true;
            done = eop = false;
            source = (static_cast<uint32_t>(static_cast<uint16_t>(vi)) * 16u) % kMemory;
            copied = tagEnd = total = 0u;
            start = t;
        }

        void step(const uint8_t *data)
        {
            std::array<uint8_t, kBufferSize> &packet = buffer();
            if (copied > kBufferSize - 16u)
            {
                done = true;
                total = 0u;
                return;
            }
            const uint32_t offset = copied;
            std::memcpy(packet.data() + offset, data + (source + copied) % kMemory, 16u);
            copied += 16u;
            if (tagEnd == 0u)
            {
                uint64_t tag;
                std::memcpy(&tag, packet.data() + offset, 8);
                const uint32_t nloop = static_cast<uint32_t>(tag & 0x7FFFu), format = static_cast<uint32_t>((tag >> 58) & 3u);
                const uint32_t nreg = ((tag >> 60) & 0xFu) ? static_cast<uint32_t>((tag >> 60) & 0xFu) : 16u;
                uint64_t bytes = 16u;
                if (format == 0u)
                    bytes += static_cast<uint64_t>(nloop) * nreg * 16u;
                else if (format == 1u)
                    bytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
                else
                    bytes += static_cast<uint64_t>(nloop) * 16u;
                if (bytes > kBufferSize - offset)
                {
                    done = true;
                    total = 0u;
                    return;
                }
                tagEnd = offset + static_cast<uint32_t>(bytes);
                eop = ((tag >> 15) & 1u) != 0u;
                if (eop)
                    total = tagEnd;
            }
            if (copied >= tagEnd)
            {
                if (eop)
                    done = true;
                else
                    tagEnd = 0u;
            }
        }

        void advance(const uint8_t *data, int64_t t)
        {
            if (!pending)
                return;
            const int64_t want = (1 + t - start) / 2;
            while (!done && static_cast<int64_t>(copied / 16u) < want)
                step(data);
        }

        int64_t finish(const uint8_t *data, Vu1Kick out, void *user)
        {
            while (!done)
                step(data);
            pending = false;
            if (total != 0u)
                out(user, buffer().data(), total);
            return start + 2 * static_cast<int64_t>(total / 16u) - 1;
        }
    };

    // RNEXT: the next value of the R register's 23-bit generator, as a float in [1, 2).
    inline uint32_t rnext(uint32_t r)
    {
        const uint32_t x = (r >> 4) & 1u;
        const uint32_t y = (r >> 22) & 1u;
        return (((r << 1) ^ x ^ y) & 0x007FFFFFu) | 0x3F800000u;
    }

    // The EFU's results, as VU1Interpreter computes them (the P value before it lands).
    namespace efu
    {
        inline float x(__m128 v) { return normalizeOperand(laneValue(v, 0)); }
        inline float y(__m128 v) { return normalizeOperand(laneValue(v, 1)); }
        inline float z(__m128 v) { return normalizeOperand(laneValue(v, 2)); }
        inline float w(__m128 v) { return normalizeOperand(laneValue(v, 3)); }
        inline float squares(__m128 v) { return x(v) * x(v) + y(v) * y(v) + z(v) * z(v); }

        inline float atanPolynomial(float value)
        {
            constexpr float c[] = {0.999999344348907f, -0.333298563957214f, 0.199465364217758f, -0.13085337519646f,
                                   0.096420042216778f, -0.055909886956215f, 0.021861229091883f, -0.004054057877511f};
            const float squared = value * value;
            float polynomial = c[7];
            for (int index = 6; index >= 0; --index)
                polynomial = c[index] + squared * polynomial;
            return 0.785398185253143f + value * polynomial;
        }

        inline float esadd(__m128 v) { return normalizeOperand(squares(v)); }
        inline float ersadd(__m128 v)
        {
            const float sum = squares(v);
            return normalizeOperand(sum != 0.0f ? 1.0f / sum : sum);
        }
        inline float eleng(__m128 v) { return normalizeOperand(std::sqrt(squares(v))); }
        inline float erleng(__m128 v)
        {
            const float length = std::sqrt(squares(v));
            return normalizeOperand(length != 0.0f ? 1.0f / length : length);
        }
        inline float eatanxy(__m128 v) { return normalizeOperand(x(v) != 0.0f ? atanPolynomial(y(v) / x(v)) : 0.0f); }
        inline float eatanxz(__m128 v) { return normalizeOperand(x(v) != 0.0f ? atanPolynomial(z(v) / x(v)) : 0.0f); }
        inline float esum(__m128 v) { return normalizeOperand(x(v) + y(v) + z(v) + w(v)); }
        inline float esqrt(float value)
        {
            value = normalizeOperand(value);
            return normalizeOperand(value >= 0.0f ? std::sqrt(value) : value);
        }
        inline float ersqrt(float value)
        {
            value = normalizeOperand(value);
            float result = value;
            if (result >= 0.0f)
            {
                result = std::sqrt(result);
                if (result != 0.0f)
                    result = 1.0f / result;
            }
            return normalizeOperand(result);
        }
        inline float ercpr(float value)
        {
            value = normalizeOperand(value);
            return normalizeOperand(value != 0.0f ? 1.0f / value : value);
        }
        inline float esin(float value)
        {
            value = normalizeOperand(value);
            constexpr float c[] = {1.0f, -0.166666567325592f, 0.008333025500178f, -0.000198074136279f, 0.000002601886990f};
            const float squared = value * value;
            float polynomial = c[4];
            for (int index = 3; index >= 0; --index)
                polynomial = c[index] + squared * polynomial;
            return normalizeOperand(value * polynomial);
        }
        inline float eatan(float value) { return normalizeOperand(atanPolynomial(normalizeOperand(value))); }
        inline float eexp(float value)
        {
            value = normalizeOperand(value);
            constexpr float c[] = {0.249998688697815f, 0.031257584691048f, 0.002591371303424f,
                                   0.000171562001924f, 0.000005430199963f, 0.000000690600018f};
            float polynomial = c[5];
            for (int index = 4; index >= 0; --index)
                polynomial = c[index] + value * polynomial;
            polynomial = 1.0f + value * polynomial;
            polynomial *= polynomial;
            polynomial *= polynomial;
            return normalizeOperand(polynomial != 0.0f ? 1.0f / polynomial : std::numeric_limits<float>::max());
        }
    }

    struct RoundTowardZero
    {
        unsigned int csr = _mm_getcsr();
        uint16_t cw = 0u;
        RoundTowardZero()
        {
#if PS2X_LIFTED_X87
            __asm__ volatile("fnstcw %0" : "=m"(cw));
#endif
            _mm_setcsr(csr | 0x6000u);
#if PS2X_LIFTED_X87
            const uint16_t towardZero = static_cast<uint16_t>(cw | 0x0C00u);
            __asm__ volatile("fldcw %0" : : "m"(towardZero));
#endif
        }
        ~RoundTowardZero()
        {
            _mm_setcsr(csr);
#if PS2X_LIFTED_X87
            __asm__ volatile("fldcw %0" : : "m"(cw));
#endif
        }
    };
}
