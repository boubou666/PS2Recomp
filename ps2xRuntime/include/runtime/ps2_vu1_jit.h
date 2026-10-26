#pragma once
// VU1 recompiler support: ps2_vu_aot turns each block (VU1Interpreter::planBlock) into one function where the VU
// registers are local variables (host registers) and each instruction is a call to one of these inline helpers.
// They do what VU1Interpreter's execUpper()/execLower() do for that instruction, on values instead of m_state:
// same normalizations, same PS2 addition, same results, so a recompiled block matches the interpreter bit for bit
// (checked with ps2_vu1_replay). Timing is the block's static schedule: the clock only moves where an instruction
// depends on it (Q/P/clip/flag queues, stores while PATH1 reads VU memory) and at the block's end. FMAC flags are
// only skipped where no reader can see them (kHintFlagsUnobserved); pairs ps2_vu_aot does not recompile run
// through the interpreter's pair body in the middle of the block (state written back and reloaded around them).
#include "runtime/ps2_vu1_inline.h"

#include <bit>

#if defined(__AVX2__)
#define PS2X_VU1_JIT 1
#else
#define PS2X_VU1_JIT 0
#endif

struct Vu1Jit
{
    using Vu = VU1Interpreter;
    using FmacKind = ps2_vu1_detail::FmacKind;
    using FmacOperation = ps2_vu1_detail::FmacOperation;

    // ---- What an upper instruction is, for ps2_vu_aot ----
    enum class Upper : uint8_t
    {
        Nop,
        Fmac,    // result to vf[fd]
        FmacAcc, // result to ACC
        Max,
        Mini,
        Itof, // to vf[ft]
        Ftoi, // to vf[ft]
        Abs,  // to vf[ft]
        Clip,
        Unsupported
    };
    struct UpperInfo
    {
        Upper kind = Upper::Unsupported;
        uint8_t target = 0; // the vf register written (Fmac, Max, Mini: fd; Itof, Ftoi, Abs: ft)
    };
    static constexpr UpperInfo upperInfo(uint32_t upper)
    {
        const uint32_t op = upper & 0x3Fu;
        const uint8_t fd = static_cast<uint8_t>((upper >> 6) & 0x1Fu);
        const uint8_t ft = static_cast<uint8_t>((upper >> 16) & 0x1Fu);
        const uint32_t low11 = upper & 0x7FFu;
        if (low11 == 0x2FFu || low11 == 0x33Cu)
            return {Upper::Nop, 0};
        if (op >= 0x3Cu)
        {
            const uint32_t special = (upper & 3u) | ((upper >> 4) & 0x7Cu);
            if (special >= 0x10u && special <= 0x13u)
                return {Upper::Itof, ft};
            if (special >= 0x14u && special <= 0x17u)
                return {Upper::Ftoi, ft};
            if (special == 0x1Du)
                return {Upper::Abs, ft};
            if (special == 0x1Fu)
                return {Upper::Clip, 0};
            if (special == 0x2Fu || special == 0x30u)
                return {Upper::Nop, 0};
            if (ps2_vu1_detail::decodeFmacOperation(upper).kind != FmacKind::None)
                return {Upper::FmacAcc, 0};
            return {Upper::Unsupported, 0};
        }
        if ((op >= 0x10u && op <= 0x13u) || op == 0x1Du || op == 0x2Bu)
            return {Upper::Max, fd};
        if ((op >= 0x14u && op <= 0x17u) || op == 0x1Fu || op == 0x2Fu)
            return {Upper::Mini, fd};
        if (ps2_vu1_detail::decodeFmacOperation(upper).kind != FmacKind::None)
            return {Upper::Fmac, fd};
        return {Upper::Unsupported, 0};
    }

    // ---- Lower instructions recompiled (the others run through the interpreter) ----
    enum class Lower : uint8_t
    {
        Nop,
        Lq,
        Sq,
        Ilw,
        Isw,
        Iaddiu,
        Isubiu,
        Iadd,
        Isub,
        Iaddi,
        Iand,
        Ior,
        Move,
        Mr32,
        Lqi,
        Sqi,
        Lqd,
        Sqd,
        Div,
        Sqrt,
        Rsqrt,
        Waitq,
        Mtir,
        Mfir,
        Ilwr,
        Iswr,
        Mfp,
        Xtop,
        Xitop,
        Waitp,
        FlagRead, // FCEQ, FCAND, FCOR, FCGET, FSEQ, FSAND, FSOR, FMEQ, FMAND, FMOR
        Fcset,
        Fsset,
        Rnext,
        Rget,
        Rinit,
        Rxor,
        Efu, // ESADD..EEXP (not WAITP)
        Branch, // B, BAL, JR, JALR, IBxx: ps2_vu_aot handles them
        Unsupported
    };
    static constexpr Lower lowerKind(uint32_t lower)
    {
        if (lower == 0u || lower == 0x8000033Cu)
            return Lower::Nop;
        switch (lower >> 25)
        {
        case 0x00u: return Lower::Lq;
        case 0x01u: return Lower::Sq;
        case 0x04u: return Lower::Ilw;
        case 0x05u: return Lower::Isw;
        case 0x08u: return Lower::Iaddiu;
        case 0x09u: return Lower::Isubiu;
        case 0x10u: case 0x12u: case 0x13u: case 0x14u: case 0x16u: case 0x17u: case 0x18u: case 0x1Au: case 0x1Bu:
        case 0x1Cu:
            return Lower::FlagRead;
        case 0x11u: return Lower::Fcset;
        case 0x15u: return Lower::Fsset;
        case 0x20u: case 0x21u: case 0x24u: case 0x25u: case 0x28u: case 0x29u:
        case 0x2Cu: case 0x2Du: case 0x2Eu: case 0x2Fu:
            return Lower::Branch;
        case 0x40u:
            break;
        default:
            return Lower::Unsupported;
        }
        switch (lower & 0x3Fu)
        {
        case 0x30u: return Lower::Iadd;
        case 0x31u: return Lower::Isub;
        case 0x32u: return Lower::Iaddi;
        case 0x34u: return Lower::Iand;
        case 0x35u: return Lower::Ior;
        case 0x3Cu: case 0x3Du: case 0x3Eu: case 0x3Fu:
            break;
        default:
            return Lower::Unsupported;
        }
        switch ((lower & 0x3u) | ((lower >> 4) & 0x7Cu))
        {
        case 0x30u: return Lower::Move;
        case 0x31u: return Lower::Mr32;
        case 0x34u: return Lower::Lqi;
        case 0x35u: return Lower::Sqi;
        case 0x36u: return Lower::Lqd;
        case 0x37u: return Lower::Sqd;
        case 0x38u: return Lower::Div;
        case 0x39u: return Lower::Sqrt;
        case 0x3Au: return Lower::Rsqrt;
        case 0x3Bu: return Lower::Waitq;
        case 0x3Cu: return Lower::Mtir;
        case 0x3Du: return Lower::Mfir;
        case 0x3Eu: return Lower::Ilwr;
        case 0x3Fu: return Lower::Iswr;
        case 0x40u: return Lower::Rnext;
        case 0x41u: return Lower::Rget;
        case 0x42u: return Lower::Rinit;
        case 0x43u: return Lower::Rxor;
        case 0x70u: case 0x71u: case 0x72u: case 0x73u: case 0x74u: case 0x75u: case 0x76u: case 0x78u: case 0x79u:
        case 0x7Au: case 0x7Cu: case 0x7Du: case 0x7Eu:
            return Lower::Efu;
        case 0x64u: return Lower::Mfp;
        case 0x68u: return Lower::Xtop;
        case 0x69u: return Lower::Xitop;
        case 0x7Bu: return Lower::Waitp;
        default: return Lower::Unsupported;
        }
    }

    // ---- State ----
    static VU1State &state(Vu &vu) { return vu.m_state; }
    static __m128 load(const float *source) { return _mm_loadu_ps(source); }
    static void store(float *target, __m128 value) { _mm_storeu_ps(target, value); }
    static __m128 vf0() { return _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f); }
    // The clock to cycle: what runBlock() does before each pair (pending Q/P results, PATH1 progress).
    static PS2X_VU_INLINE void advance(Vu &vu, uint64_t cycle) { vu.advanceTo(cycle); }
    static PS2X_VU_INLINE uint32_t wrapPc(const Vu::RunContext &context, uint32_t pc)
    {
        return pc >= context.codeSize ? pc - context.codeSize : pc;
    }
    // What markPairWritesAt() on each pair of a block, in order, leaves behind, as cycles after the block's base
    // (-1: not written): each lane's ready cycle is its last writer's (the lower write before the upper one in a
    // pair), and the per-register and writeback maxima are maxima over all the writes.
    struct BlockWrites
    {
        std::array<std::array<int32_t, 4>, 32> vfLane{};
        std::array<int32_t, 32> vfMax{};
        std::array<int32_t, 16> vi{};
        std::array<int32_t, 4> accLane{};
        int32_t accMax = -1;
        int32_t writebackMax = -1;
    };
    template <const Vu::BlockPlan &kPlan, const auto &kPairs>
    static constexpr BlockWrites blockWrites()
    {
        BlockWrites writes;
        for (auto &lanes : writes.vfLane)
            lanes = {-1, -1, -1, -1};
        writes.vfMax.fill(-1);
        writes.vi.fill(-1);
        writes.accLane = {-1, -1, -1, -1};
        const auto setLanes = [](std::array<int32_t, 4> &laneCycles, uint8_t lanes, int32_t ready)
        {
            for (uint32_t lane = 0; lane < 4u; ++lane)
                if ((lanes & (0x8u >> lane)) != 0u)
                    laneCycles[lane] = ready;
        };
        const auto vfWrite = [&](Vu::VfAccess write, const Vu::InstructionUsage &usage, int32_t issue)
        {
            const int32_t ready = issue + (usage.vfLatency != 0u ? usage.vfLatency : usage.latency);
            setLanes(writes.vfLane[write.reg], write.lanes, ready);
            writes.vfMax[write.reg] = std::max(writes.vfMax[write.reg], ready);
            writes.writebackMax = std::max(writes.writebackMax, ready);
        };
        for (uint32_t index = 0; index < kPlan.pairCount; ++index)
        {
            const auto &decoded = kPairs[index];
            const int32_t issue = kPlan.issue[index];
            if (const Vu::VfAccess write = decoded.lowerUsage.vfWrite; write.reg != 0u && decoded.suppressedLowerVf != write.reg)
                vfWrite(write, decoded.lowerUsage, issue);
            if (const Vu::VfAccess write = decoded.upperUsage.vfWrite; write.reg != 0u)
                vfWrite(write, decoded.upperUsage, issue);
            if (const uint32_t viWrites = decoded.lowerUsage.viWrite & 0xFFFEu; viWrites != 0u)
            {
                const int32_t ready = issue + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency
                                                                                 : decoded.lowerUsage.latency);
                for (uint32_t reg = 1; reg < 16u; ++reg)
                    if ((viWrites & (1u << reg)) != 0u)
                        writes.vi[reg] = ready;
                writes.writebackMax = std::max(writes.writebackMax, ready);
            }
            if (decoded.upperUsage.accWrite != 0u)
            {
                const int32_t ready = issue + static_cast<int32_t>(Vu::kAccForwardLatency);
                setLanes(writes.accLane, decoded.upperUsage.accWrite, ready);
                writes.accMax = std::max(writes.accMax, ready);
                writes.writebackMax = std::max(writes.writebackMax, ready);
            }
        }
        return writes;
    }
    template <std::array<int32_t, 4> kLanes>
    static PS2X_VU_INLINE void setReadyLanes(std::array<uint64_t, 4> &laneCycles, uint64_t base)
    {
        if constexpr (kLanes[0] >= 0)
            laneCycles[0] = base + static_cast<uint32_t>(kLanes[0]);
        if constexpr (kLanes[1] >= 0)
            laneCycles[1] = base + static_cast<uint32_t>(kLanes[1]);
        if constexpr (kLanes[2] >= 0)
            laneCycles[2] = base + static_cast<uint32_t>(kLanes[2]);
        if constexpr (kLanes[3] >= 0)
            laneCycles[3] = base + static_cast<uint32_t>(kLanes[3]);
    }
    template <const BlockWrites &kWrites, size_t kReg>
    static PS2X_VU_INLINE void leaveVf(Vu &vu, uint64_t base)
    {
        if constexpr (kWrites.vfMax[kReg] >= 0)
        {
            setReadyLanes<kWrites.vfLane[kReg]>(vu.m_vfReady[kReg], base);
            vu.m_vfReadyMax[kReg] = std::max(vu.m_vfReadyMax[kReg], base + static_cast<uint32_t>(kWrites.vfMax[kReg]));
        }
        if constexpr (kReg < 16u && kWrites.vi[kReg] >= 0)
            vu.m_viReady[kReg] = base + static_cast<uint32_t>(kWrites.vi[kReg]);
    }
    template <const Vu::BlockPlan &kPlan, const auto &kPairs>
    static constexpr BlockWrites kBlockWrites = blockWrites<kPlan, kPairs>();

    // The block's end: when its results land (markPairWritesAt on each pair, folded at compile time into one
    // store per written lane), the clock after its last pair, and the VI value a branch right after the block
    // would read (recordViWriteForBranch).
    template <const Vu::BlockPlan &kPlan, const auto &kPairs>
    static PS2X_VU_INLINE void leave(Vu &vu, uint64_t base)
    {
        constexpr const BlockWrites &kWrites = kBlockWrites<kPlan, kPairs>;
        [&]<size_t... kReg>(std::index_sequence<kReg...>) PS2X_VU_LAMBDA_INLINE
        {
            (leaveVf<kWrites, kReg>(vu, base), ...);
        }(std::make_index_sequence<32>{});
        if constexpr (kWrites.accMax >= 0)
        {
            setReadyLanes<kWrites.accLane>(vu.m_accReady, base);
            vu.m_accReadyMax = std::max(vu.m_accReadyMax, base + static_cast<uint32_t>(kWrites.accMax));
        }
        if constexpr (kWrites.writebackMax >= 0)
            vu.m_lastWritebackCycle =
                std::max(vu.m_lastWritebackCycle, base + static_cast<uint32_t>(kWrites.writebackMax));
        vu.m_viBranchBackupValid = false;
        vu.advanceTo(base + kPlan.issue[kPlan.pairCount - 1u] + 1u);
    }
    static PS2X_VU_INLINE void recordViForBranch(Vu &vu, uint8_t reg, int32_t oldValue)
    {
        vu.recordViWriteForBranch(reg, oldValue);
    }
    static PS2X_VU_INLINE int32_t readBranchVi(const Vu &vu, uint8_t reg) { return vu.readBranchVi(reg); }
    // An I bit: the lower word is the new I, after the upper instruction read the old one.
    static PS2X_VU_INLINE void setI(Vu &vu, uint32_t bits) { vu.m_state.i = vu.normalizeOperand(std::bit_cast<float>(bits)); }

    // ---- Helpers ----
    template <uint8_t kDest>
    static PS2X_VU_INLINE __m128 blend(__m128 old, __m128 value)
    {
        if constexpr ((kDest & 0xFu) == 0xFu)
            return value;
        else if constexpr ((kDest & 0xFu) == 0u)
            return old;
        else
            return _mm_blend_ps(old, value,
                                ((kDest >> 3) & 1) | (((kDest >> 2) & 1) << 1) | (((kDest >> 1) & 1) << 2) |
                                    ((kDest & 1) << 3));
    }
    template <bool kKnown>
    static PS2X_VU_INLINE __m128 normalized(__m128 value)
    {
        if constexpr (kKnown)
            return value;
        alignas(16) float source[4], target[4];
        _mm_store_ps(source, value);
        ps2_vu1_detail::normalizeVector(source, target);
        return _mm_load_ps(target);
    }
    static PS2X_VU_INLINE float lane(__m128 value, uint32_t index)
    {
        alignas(16) float lanes[4];
        _mm_store_ps(lanes, value);
        return lanes[index];
    }
    static PS2X_VU_INLINE uint32_t laneBits(__m128 value, uint32_t index)
    {
        return std::bit_cast<uint32_t>(lane(value, index));
    }

    // ---- Upper ----
    // An FMAC instruction whose flags no reader sees (kHintFlagsUnobserved): execUpper's result, normalized as
    // fmacFlags() does (its fast check first, then fmacFlagsWith on the operands), into target.
    // execUpper's FMAC result, with the normalized operands fmacFlagsWith takes.
    template <uint32_t kUpper, uint8_t kHints>
    static PS2X_VU_INLINE void fmacResult(Vu &vu, float *result, __m128 vs, __m128 vt, __m128 acc, __m128 &left,
                                          __m128 &right, __m128 &a)
    {
        constexpr FmacOperation kOperation = ps2_vu1_detail::decodeFmacOperation(kUpper);
        left = normalized<(kHints & Vu::kKnownNormalFs) != 0u>(vs);
        if constexpr (kOperation.right == FmacOperation::Right::Broadcast)
            right = _mm_set1_ps(lane(normalized<(kHints & Vu::kKnownNormalFt) != 0u>(vt), kOperation.broadcast));
        else if constexpr (kOperation.right == FmacOperation::Right::Q)
            right = _mm_set1_ps(vu.normalizeOperand(vu.m_state.q));
        else if constexpr (kOperation.right == FmacOperation::Right::I)
            right = _mm_set1_ps(vu.normalizeOperand(vu.m_state.i));
        else
            right = normalized<(kHints & Vu::kKnownNormalFt) != 0u>(vt);
        constexpr bool kProductSum = kOperation.kind == FmacKind::Madd || kOperation.kind == FmacKind::Msub ||
                                     kOperation.kind == FmacKind::CrossMsub;
        a = kProductSum ? normalized<(kHints & Vu::kKnownNormalAcc) != 0u>(acc) : _mm_setzero_ps();

        if constexpr (kOperation.kind == FmacKind::Add)
            _mm_store_ps(result, ps2_vu1_detail::vuAdd4(left, right));
        else if constexpr (kOperation.kind == FmacKind::Sub)
            _mm_store_ps(result, ps2_vu1_detail::vuAdd4(left, ps2_vu1_detail::negate4(right)));
        else if constexpr (kOperation.kind == FmacKind::Mul)
            _mm_store_ps(result, _mm_mul_ps(left, right));
        else if constexpr (kOperation.kind == FmacKind::Madd)
            _mm_store_ps(result, ps2_vu1_detail::vuAdd4(a, _mm_mul_ps(left, right)));
        else if constexpr (kOperation.kind == FmacKind::Msub)
            _mm_store_ps(result, ps2_vu1_detail::vuAdd4(a, ps2_vu1_detail::negate4(_mm_mul_ps(left, right))));
        else
        {
            alignas(16) float l[4], r[4], c[4];
            _mm_store_ps(l, left);
            _mm_store_ps(r, right);
            _mm_store_ps(c, a);
            if constexpr (kOperation.kind == FmacKind::CrossMul)
            {
                result[0] = l[1] * r[2];
                result[1] = l[2] * r[0];
                result[2] = l[0] * r[1];
            }
            else
            {
                result[0] = ps2_vu1_detail::vuSub(c[0], l[1] * r[2]);
                result[1] = ps2_vu1_detail::vuSub(c[1], l[2] * r[0]);
                result[2] = ps2_vu1_detail::vuSub(c[2], l[0] * r[1]);
            }
            result[3] = 0.0f;
        }
    }

    // An FMAC instruction whose flags no reader sees (kHintFlagsUnobserved): execUpper's result, normalized as
    // fmacFlags() does (its fast check first, then fmacFlagsWith on the operands), into target.
    template <uint32_t kUpper, uint8_t kHints>
    static PS2X_VU_INLINE void fmac(Vu &vu, __m128 &target, __m128 vs, __m128 vt, __m128 acc)
    {
        static_assert((kHints & Vu::kHintFlagsUnobserved) != 0u, "fmacLive computes the flags a reader can see");
        constexpr FmacOperation kOperation = ps2_vu1_detail::decodeFmacOperation(kUpper);
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        if constexpr (kDest == 0u)
            return;
        alignas(16) float result[4];
        __m128 left, right, a;
        fmacResult<kUpper, kHints>(vu, result, vs, vt, acc, left, right, a);
        normalizeUnobserved<kOperation>(vu, result, kDest, left, right, a);
        target = blend<kDest>(target, _mm_load_ps(result));
    }

    // An FMAC instruction whose flags a reader can see: fmacFlags() on the operands, the entry queued at its issue
    // cycle (the caller advanced the clock to it). kDead: BlockPlan::deadFlags (only the sticky bits go on).
    template <uint32_t kUpper, uint8_t kHints, bool kDead>
    static PS2X_VU_INLINE void fmacLive(Vu &vu, __m128 &target, __m128 vs, __m128 vt, __m128 acc)
    {
        constexpr FmacOperation kOperation = ps2_vu1_detail::decodeFmacOperation(kUpper);
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        if constexpr (kDest == 0u)
            return;
        alignas(16) float result[4];
        __m128 left, right, a;
        fmacResult<kUpper, kHints>(vu, result, vs, vt, acc, left, right, a);
        uint32_t mac = 0u, status = 0u, sticky = 0u;
        vu.fmacFlagsWith(
            result, kDest, kOperation, [&]() { return left; }, [&]() { return right; }, [&]() { return a; }, mac,
            status, sticky);
        if constexpr (kDead)
        {
            vu.m_deadSticky |= status | sticky;
        }
        else if (Vu::FlagPipelineEntry *entry = vu.pushFlagEntry(true, 0xFFFFFFFFu))
        {
            entry->mac = mac;
            entry->status = status;
            entry->extraSticky = sticky | vu.m_deadSticky;
            vu.m_deadSticky = 0u;
            entry->writesMac = true;
            entry->writesStatus = true;
        }
        target = blend<kDest>(target, _mm_load_ps(result));
    }

    // ---- Relaxed float mode (VU1Interpreter::setRelaxedFloat) ----
    static bool enableRelaxedFloat()
    {
        Vu::setRelaxedFloat(true);
        return true;
    }
    template <uint32_t kIndex>
    static PS2X_VU_INLINE __m128 spread(__m128 value)
    {
        return _mm_shuffle_ps(value, value, _MM_SHUFFLE(kIndex, kIndex, kIndex, kIndex));
    }
    // An FMAC instruction as plain SSE: operands as they are (denormals are zero under DAZ), the sum of IEEE
    // arithmetic instead of the PS2's (no operand truncation), the result clamped to +-FLT_MAX. kLiveFlags: the MAC
    // and status flags a reader can see, from the result (zero, sign, overflow before the clamp), queued at the issue
    // cycle (the caller advanced the clock to it).
    template <uint32_t kUpper, bool kLiveFlags>
    static PS2X_VU_INLINE void fmacRelaxed(Vu &vu, __m128 &target, __m128 vs, __m128 vt, __m128 acc)
    {
        constexpr FmacOperation kOperation = ps2_vu1_detail::decodeFmacOperation(kUpper);
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        if constexpr (kDest == 0u)
            return;
        __m128 right;
        if constexpr (kOperation.right == FmacOperation::Right::Broadcast)
            right = spread<kOperation.broadcast>(vt);
        else if constexpr (kOperation.right == FmacOperation::Right::Q)
            right = _mm_set1_ps(vu.m_state.q);
        else if constexpr (kOperation.right == FmacOperation::Right::I)
            right = _mm_set1_ps(vu.m_state.i);
        else
            right = vt;
        __m128 left = vs;
        if constexpr (kOperation.kind == FmacKind::CrossMul || kOperation.kind == FmacKind::CrossMsub)
        {
            left = _mm_shuffle_ps(left, left, _MM_SHUFFLE(3, 0, 2, 1));     // y z x w
            right = _mm_shuffle_ps(right, right, _MM_SHUFFLE(3, 1, 0, 2)); // z x y w
        }
        __m128 result;
        if constexpr (kOperation.kind == FmacKind::Add)
            result = _mm_add_ps(left, right);
        else if constexpr (kOperation.kind == FmacKind::Sub)
            result = _mm_sub_ps(left, right);
        else if constexpr (kOperation.kind == FmacKind::Mul || kOperation.kind == FmacKind::CrossMul)
            result = _mm_mul_ps(left, right);
        else if constexpr (kOperation.kind == FmacKind::Madd)
            result = _mm_add_ps(acc, _mm_mul_ps(left, right));
        else
            result = _mm_sub_ps(acc, _mm_mul_ps(left, right));
        if constexpr (kOperation.kind == FmacKind::CrossMul || kOperation.kind == FmacKind::CrossMsub)
            result = _mm_blend_ps(result, _mm_setzero_ps(), 0x8);
        const __m128 maximum = _mm_set1_ps(std::numeric_limits<float>::max());
        const __m128 clamped = _mm_max_ps(_mm_min_ps(result, maximum), _mm_xor_ps(maximum, _mm_set1_ps(-0.0f)));
        if constexpr (kLiveFlags)
        {
            static constexpr uint8_t kReverse4[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                                      0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
            constexpr uint32_t kActive = ((kDest >> 3) & 1u) | (((kDest >> 2) & 1u) << 1) | (((kDest >> 1) & 1u) << 2) |
                                         ((kDest & 1u) << 3);
            const uint32_t zero = static_cast<uint32_t>(_mm_movemask_ps(_mm_cmpeq_ps(clamped, _mm_setzero_ps()))) & kActive;
            const uint32_t sign = static_cast<uint32_t>(_mm_movemask_ps(clamped)) & kActive;
            const uint32_t overflow =
                static_cast<uint32_t>(_mm_movemask_ps(_mm_cmpneq_ps(clamped, result))) & kActive;
            if (Vu::FlagPipelineEntry *entry = vu.pushFlagEntry(true, 0xFFFFFFFFu))
            {
                entry->mac = static_cast<uint32_t>(kReverse4[zero]) | (static_cast<uint32_t>(kReverse4[sign]) << 4) |
                             (static_cast<uint32_t>(kReverse4[overflow]) << 12);
                entry->status = (zero ? 0x1u : 0u) | (sign ? 0x2u : 0u) | (overflow ? 0x8u : 0u);
                entry->extraSticky = 0u;
                entry->writesMac = true;
                entry->writesStatus = true;
            }
        }
        target = blend<kDest>(target, clamped);
    }

    template <FmacOperation kOperation>
    static PS2X_VU_INLINE void normalizeUnobserved(Vu &vu, float *result, uint8_t dest, __m128 left, __m128 right,
                                                   __m128 acc)
    {
        // As fmacFlags() with kHintFlagsUnobserved: lanes with exponent 2..253, and zeros whose exact result is a
        // zero of the same sign, stay as they are.
        static constexpr uint8_t kDestLanes[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                                   0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
        const __m128i bits = _mm_load_si128(reinterpret_cast<const __m128i *>(result));
        const __m128i exponent = _mm_and_si128(_mm_srli_epi32(bits, 23), _mm_set1_epi32(0xFF));
        const __m128i inRange = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1)),
                                              _mm_cmpgt_epi32(_mm_set1_epi32(254), exponent));
        uint32_t pending = kDestLanes[dest & 0xFu] & ~static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(inRange)));
        if (pending == 0u)
            return;
        const auto zeroLanes = [](__m128i value)
        {
            return static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(
                _mm_cmpeq_epi32(_mm_and_si128(value, _mm_set1_epi32(0x7FFFFFFF)), _mm_setzero_si128()))));
        };
        if constexpr (kOperation.kind == FmacKind::Add || kOperation.kind == FmacKind::Sub ||
                      kOperation.kind == FmacKind::Mul)
            pending &= ~zeroLanes(bits);
        else if constexpr (kOperation.kind == FmacKind::Madd || kOperation.kind == FmacKind::Msub)
            pending &= ~(zeroLanes(bits) &
                         (zeroLanes(_mm_castps_si128(left)) | zeroLanes(_mm_castps_si128(right))));
        if (pending == 0u)
            return;
        uint32_t mac = 0u, status = 0u, sticky = 0u;
        vu.fmacFlagsWith(
            result, dest, kOperation, [&]() { return left; }, [&]() { return right; }, [&]() { return acc; }, mac,
            status, sticky);
    }

    // MAX/MINI: raw sign-magnitude comparison (vuMax/vuMini: the larger or smaller key, else the second operand),
    // bits passed through. The key is the magnitude, negated when the sign is set.
    template <uint32_t kUpper, bool kMax>
    static PS2X_VU_INLINE void maxMini(Vu &vu, __m128 &target, __m128 vs, __m128 vt)
    {
        constexpr uint32_t kOp = kUpper & 0x3Fu;
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        __m128 t;
        if constexpr (kOp == 0x1Du || kOp == 0x1Fu) // MAXi, MINIi
            t = _mm_set1_ps(vu.m_state.i);
        else if constexpr (kOp < 0x18u) // bc
            t = spread<kOp & 3u>(vt);
        else
            t = vt;
        const auto key = [](__m128 value)
        {
            const __m128i bits = _mm_castps_si128(value);
            const __m128i sign = _mm_srai_epi32(bits, 31);
            return _mm_sub_epi32(_mm_xor_si128(_mm_and_si128(bits, _mm_set1_epi32(0x7FFFFFFF)), sign), sign);
        };
        const __m128i ks = key(vs), kt = key(t);
        const __m128 pickS = _mm_castsi128_ps(kMax ? _mm_cmpgt_epi32(ks, kt) : _mm_cmpgt_epi32(kt, ks));
        target = blend<kDest>(target, _mm_blendv_ps(t, vs, pickS));
    }

    template <uint32_t kUpper>
    static PS2X_VU_INLINE void itof(__m128 &target, __m128 vs)
    {
        constexpr uint32_t kSpecial = (kUpper & 3u) | ((kUpper >> 4) & 0x7Cu);
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        constexpr float kScale = kSpecial == 0x10u ? 1.0f : kSpecial == 0x11u ? 16.0f : kSpecial == 0x12u ? 4096.0f : 32768.0f;
        alignas(16) int32_t s[4];
        alignas(16) float result[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(s), _mm_castps_si128(vs));
        for (int c = 0; c < 4; ++c)
            result[c] = kSpecial == 0x10u ? static_cast<float>(s[c]) : static_cast<float>(s[c]) / kScale;
        target = blend<kDest>(target, _mm_load_ps(result));
    }

    template <uint32_t kUpper, uint8_t kHints>
    static PS2X_VU_INLINE void ftoi(__m128 &target, __m128 vs)
    {
        constexpr uint32_t kSpecial = (kUpper & 3u) | ((kUpper >> 4) & 0x7Cu);
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        constexpr float kScale = kSpecial == 0x14u ? 1.0f : kSpecial == 0x15u ? 16.0f : kSpecial == 0x16u ? 4096.0f : 32768.0f;
        alignas(16) float s[4];
        alignas(16) int32_t result[4];
        _mm_store_ps(s, normalized<(kHints & Vu::kKnownNormalFs) != 0u>(vs));
        for (int c = 0; c < 4; ++c)
            result[c] = ps2_vu1_detail::vuFloatToInt(s[c], kScale);
        target = blend<kDest>(target, _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(result))));
    }

    template <uint32_t kUpper>
    static PS2X_VU_INLINE void abs(__m128 &target, __m128 vs)
    {
        constexpr uint8_t kDest = (kUpper >> 21) & 0xFu;
        target = blend<kDest>(target, _mm_and_ps(vs, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))));
    }

    // CLIP: its clip flags enter the flag queue at its issue cycle (the caller advanced the clock to it).
    static PS2X_VU_INLINE void clip(Vu &vu, __m128 vs, __m128 vt)
    {
        const uint32_t wBits = laneBits(vt, 3);
        const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
        const auto exceeds = [limit](uint32_t bits, uint32_t signMask)
        { return static_cast<int32_t>(bits ^ signMask) > limit; };
        uint32_t flags = 0u;
        const uint32_t x = laneBits(vs, 0), y = laneBits(vs, 1), z = laneBits(vs, 2);
        if (exceeds(x, 0u))
            flags |= 0x01u;
        if (exceeds(x, 0x80000000u))
            flags |= 0x02u;
        if (exceeds(y, 0u))
            flags |= 0x04u;
        if (exceeds(y, 0x80000000u))
            flags |= 0x08u;
        if (exceeds(z, 0u))
            flags |= 0x10u;
        if (exceeds(z, 0x80000000u))
            flags |= 0x20u;
        vu.queueClip(flags);
    }

    // ---- Lower ----
    static constexpr int16_t imm11(uint32_t lower)
    {
        return static_cast<int16_t>(static_cast<int32_t>((lower & 0x7FFu) << 21) >> 21);
    }
    static constexpr int16_t imm15(uint32_t lower)
    {
        return static_cast<int16_t>(static_cast<int16_t>(lower & 0x7FFu) | static_cast<int16_t>((lower >> 10) & 0x7800u));
    }
    static PS2X_VU_INLINE bool inData(const Vu::RunContext &context, uint32_t &address)
    {
        address &= context.dataSize - 1u;
        return address + 16u <= context.dataSize;
    }
    template <uint8_t kDest>
    static PS2X_VU_INLINE void loadQuad(const Vu::RunContext &context, __m128 &target, uint32_t address)
    {
        if (inData(context, address))
            target = blend<kDest>(target, _mm_loadu_ps(reinterpret_cast<const float *>(context.vuData + address)));
    }
    // queueStore(): written now; the program's writeback completes the cycle after (m_lastWritebackCycle). A PATH1
    // transfer in flight reads VU memory as the clock moves: it gets to the store's cycle first.
    template <uint8_t kDest>
    static PS2X_VU_INLINE void storeWords(Vu &vu, const Vu::RunContext &context, uint64_t cycle, uint32_t address,
                                          __m128i words)
    {
        if (vu.m_xgkick.active)
            vu.advanceTo(cycle);
        if (!inData(context, address) || !vu.m_activeVuData || address + 16u > vu.m_activeVuDataSize)
            return;
        if (cycle + 1u > vu.m_lastWritebackCycle)
            vu.m_lastWritebackCycle = cycle + 1u;
        float *target = reinterpret_cast<float *>(vu.m_activeVuData + address);
        _mm_storeu_ps(target, blend<kDest>(_mm_loadu_ps(target), _mm_castsi128_ps(words)));
    }
    static PS2X_VU_INLINE int32_t integerWord(const Vu::RunContext &context, uint32_t address, uint8_t dest, bool &loaded)
    {
        loaded = inData(context, address);
        if (!loaded)
            return 0;
        const int component = (dest & 0x8u) ? 0 : (dest & 0x4u) ? 1 : (dest & 0x2u) ? 2 : 3;
        uint32_t value = 0u;
        std::memcpy(&value, context.vuData + address + component * 4, 4);
        return static_cast<int32_t>(static_cast<int16_t>(value & 0xFFFFu));
    }

    template <uint32_t kLower>
    static PS2X_VU_INLINE void lq(const Vu::RunContext &context, __m128 &target, int32_t base)
    {
        loadQuad<(kLower >> 21) & 0xFu>(context, target, static_cast<uint32_t>(static_cast<int32_t>(base + imm11(kLower))) * 16u);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void sq(Vu &vu, const Vu::RunContext &context, uint64_t cycle, __m128 source, int32_t base)
    {
        storeWords<(kLower >> 21) & 0xFu>(vu, context, cycle,
                                          static_cast<uint32_t>(static_cast<int32_t>(base + imm11(kLower))) * 16u,
                                          _mm_castps_si128(source));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void ilw(const Vu::RunContext &context, int32_t &target, int32_t base)
    {
        bool loaded = false;
        const int32_t value = integerWord(context, static_cast<uint32_t>(static_cast<int32_t>(base + imm11(kLower))) * 16u,
                                          (kLower >> 21) & 0xFu, loaded);
        if (loaded)
            target = value;
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void isw(Vu &vu, const Vu::RunContext &context, uint64_t cycle, int32_t source, int32_t base)
    {
        const uint32_t value = static_cast<uint32_t>(static_cast<uint16_t>(source & 0xFFFF));
        storeWords<(kLower >> 21) & 0xFu>(vu, context, cycle,
                                          static_cast<uint32_t>(static_cast<int32_t>(base + imm11(kLower))) * 16u,
                                          _mm_set1_epi32(static_cast<int32_t>(value)));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void ilwr(const Vu::RunContext &context, int32_t &target, int32_t base)
    {
        bool loaded = false;
        const int32_t value =
            integerWord(context, static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u, (kLower >> 21) & 0xFu, loaded);
        if (loaded)
            target = value;
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void iswr(Vu &vu, const Vu::RunContext &context, uint64_t cycle, int32_t source, int32_t base)
    {
        const uint32_t value = static_cast<uint32_t>(static_cast<uint16_t>(source & 0xFFFF));
        storeWords<(kLower >> 21) & 0xFu>(vu, context, cycle, static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u,
                                          _mm_set1_epi32(static_cast<int32_t>(value)));
    }
    // LQI/SQI read the address before the increment, LQD/SQD after the decrement (vi0 never changes).
    template <uint32_t kLower>
    static PS2X_VU_INLINE void lqi(const Vu::RunContext &context, __m128 &target, int32_t &base, bool baseIsZero)
    {
        loadQuad<(kLower >> 21) & 0xFu>(context, target, static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u);
        if (!baseIsZero)
            base = static_cast<int16_t>(base + 1);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void sqi(Vu &vu, const Vu::RunContext &context, uint64_t cycle, __m128 source, int32_t &base,
                                   bool baseIsZero)
    {
        storeWords<(kLower >> 21) & 0xFu>(vu, context, cycle, static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u,
                                          _mm_castps_si128(source));
        if (!baseIsZero)
            base = static_cast<int16_t>(base + 1);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void lqd(const Vu::RunContext &context, __m128 &target, int32_t &base, bool baseIsZero)
    {
        if (!baseIsZero)
            base = static_cast<int16_t>(base - 1);
        loadQuad<(kLower >> 21) & 0xFu>(context, target, static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void sqd(Vu &vu, const Vu::RunContext &context, uint64_t cycle, __m128 source, int32_t &base,
                                   bool baseIsZero)
    {
        if (!baseIsZero)
            base = static_cast<int16_t>(base - 1);
        storeWords<(kLower >> 21) & 0xFu>(vu, context, cycle, static_cast<uint32_t>(static_cast<uint16_t>(base)) * 16u,
                                          _mm_castps_si128(source));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void move(__m128 &target, __m128 source)
    {
        target = blend<(kLower >> 21) & 0xFu>(target, source);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void mr32(__m128 &target, __m128 source)
    {
        target = blend<(kLower >> 21) & 0xFu>(target, _mm_shuffle_ps(source, source, _MM_SHUFFLE(0, 3, 2, 1)));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void mtir(int32_t &target, __m128 source)
    {
        target = static_cast<int32_t>(static_cast<int16_t>(laneBits(source, (kLower >> 21) & 0x3u) & 0xFFFFu));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void mfir(__m128 &target, int32_t source)
    {
        target = blend<(kLower >> 21) & 0xFu>(
            target, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(static_cast<int16_t>(source & 0xFFFF)))));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void mfp(Vu &vu, __m128 &target)
    {
        target = blend<(kLower >> 21) & 0xFu>(target, _mm_set1_ps(vu.m_state.p));
    }

    // DIV, SQRT, RSQRT: queueQ at the issue cycle (the caller advanced the clock to it).
    template <uint32_t kLower>
    static PS2X_VU_INLINE void div(Vu &vu, __m128 vs, __m128 vt)
    {
        const float num = vu.normalizeOperand(lane(vs, (kLower >> 21) & 0x3u));
        const float den = vu.normalizeOperand(lane(vt, (kLower >> 23) & 0x3u));
        uint32_t statusDi = 0u;
        float result = 0.0f;
        if (den == 0.0f)
        {
            statusDi = num == 0.0f ? 0x10u : 0x20u;
            result = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max()
                                                            : std::numeric_limits<float>::max();
        }
        else
        {
            result = num / den;
        }
        uint32_t ignoredFlags = 0u;
        result = vu.normalizeResult(result, ignoredFlags);
        vu.queueQ(result, 7u, statusDi);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void sqrt(Vu &vu, __m128 vt)
    {
        const float value = vu.normalizeOperand(lane(vt, (kLower >> 23) & 0x3u));
        vu.queueQ(std::sqrt(std::fabs(value)), 7u, value < 0.0f ? 0x10u : 0u);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void rsqrt(Vu &vu, __m128 vs, __m128 vt)
    {
        const float num = vu.normalizeOperand(lane(vs, (kLower >> 21) & 0x3u));
        const float radicand = vu.normalizeOperand(lane(vt, (kLower >> 23) & 0x3u));
        const float den = std::sqrt(std::fabs(radicand));
        uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
        float result = 0.0f;
        if (den != 0.0f)
            result = num / den;
        else
        {
            statusDi = num == 0.0f ? 0x10u : 0x20u;
            result = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        }
        uint32_t ignoredFlags = 0u;
        result = vu.normalizeResult(result, ignoredFlags);
        vu.queueQ(result, 13u, statusDi);
    }

    // Flag readers: the flags landed by their issue cycle (commitFlags after advancing the clock to it).
    // FCEQ/FCAND/FCOR write vi1: the caller passes it as target.
    template <uint32_t kLower>
    static PS2X_VU_INLINE void flagRead(Vu &vu, uint64_t cycle, int32_t &target, int32_t source)
    {
        vu.advanceTo(cycle);
        vu.commitFlags();
        const VU1State &state = vu.m_state;
        constexpr uint32_t kOp = kLower >> 25;
        constexpr uint32_t kImm24 = kLower & 0xFFFFFFu;
        constexpr uint32_t kImm12 = (((kLower >> 21) & 0x1u) << 11) | (kLower & 0x7FFu);
        if constexpr (kOp == 0x10u) // FCEQ
            target = ((state.clip & 0xFFFFFFu) == kImm24) ? 1 : 0;
        else if constexpr (kOp == 0x12u) // FCAND
            target = ((state.clip & kImm24) != 0u) ? 1 : 0;
        else if constexpr (kOp == 0x13u) // FCOR
            target = ((state.clip | kImm24) == 0xFFFFFFu) ? 1 : 0;
        else if constexpr (kOp == 0x1Cu) // FCGET
            target = static_cast<int32_t>(state.clip & 0x0FFFu);
        else if constexpr (kOp == 0x14u) // FSEQ
            target = ((state.status & 0xFFFu) == kImm12) ? 1 : 0;
        else if constexpr (kOp == 0x16u) // FSAND
            target = static_cast<int32_t>((state.status & 0xFFFu) & kImm12);
        else if constexpr (kOp == 0x17u) // FSOR
            target = static_cast<int32_t>((state.status & 0xFFFu) | kImm12);
        else if constexpr (kOp == 0x18u) // FMEQ
            target = ((state.mac & 0xFFFFu) == static_cast<uint32_t>(static_cast<uint16_t>(source))) ? 1 : 0;
        else if constexpr (kOp == 0x1Au) // FMAND
            target = static_cast<int32_t>(state.mac & static_cast<uint32_t>(static_cast<uint16_t>(source)));
        else // FMOR
            target = static_cast<int32_t>(state.mac | static_cast<uint32_t>(static_cast<uint16_t>(source)));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void fcset(Vu &vu) { vu.queueFcset(kLower & 0xFFFFFFu); }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void fsset(Vu &vu)
    {
        vu.queueFsset(static_cast<uint16_t>((((kLower >> 21) & 0x1u) << 11) | (kLower & 0x7FFu)));
    }

    // R register.
    template <uint32_t kLower>
    static PS2X_VU_INLINE void rnext(Vu &vu, __m128 &target)
    {
        uint32_t &r = vu.m_state.r;
        const uint32_t x = (r >> 4) & 1u;
        const uint32_t y = (r >> 22) & 1u;
        r = ((r << 1) ^ x ^ y) & 0x007FFFFFu;
        r |= 0x3F800000u;
        target = blend<(kLower >> 21) & 0xFu>(target, _mm_set1_ps(std::bit_cast<float>(r)));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void rget(Vu &vu, __m128 &target)
    {
        target = blend<(kLower >> 21) & 0xFu>(target, _mm_set1_ps(std::bit_cast<float>(vu.m_state.r)));
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void rinit(Vu &vu, __m128 source)
    {
        vu.m_state.r = 0x3F800000u | (laneBits(source, (kLower >> 21) & 3u) & 0x007FFFFFu);
    }
    template <uint32_t kLower>
    static PS2X_VU_INLINE void rxor(Vu &vu, __m128 source)
    {
        vu.m_state.r = 0x3F800000u | ((vu.m_state.r ^ laneBits(source, (kLower >> 21) & 3u)) & 0x007FFFFFu);
    }

    // EFU: queueP at the issue cycle (the caller advanced the clock to it).
    template <uint32_t kLower>
    static PS2X_VU_INLINE void efu(Vu &vu, __m128 vs)
    {
        constexpr uint32_t kFunct = (kLower & 0x3u) | ((kLower >> 4) & 0x7Cu);
        alignas(16) float v[4];
        _mm_store_ps(v, vs);
        const auto n = [&](uint32_t component) { return vu.normalizeOperand(v[component]); };
        constexpr uint32_t kComponent = (kLower >> 21) & 3u;
        if constexpr (kFunct == 0x70u) // ESADD
        {
            const float x = n(0), y = n(1), z = n(2);
            vu.queueP(x * x + y * y + z * z, 11u);
        }
        else if constexpr (kFunct == 0x71u) // ERSADD
        {
            const float x = n(0), y = n(1), z = n(2);
            const float sum = x * x + y * y + z * z;
            vu.queueP(sum != 0.0f ? 1.0f / sum : sum, 18u);
        }
        else if constexpr (kFunct == 0x72u) // ELENG
        {
            const float x = n(0), y = n(1), z = n(2);
            vu.queueP(std::sqrt(x * x + y * y + z * z), 18u);
        }
        else if constexpr (kFunct == 0x73u) // ERLENG
        {
            const float x = n(0), y = n(1), z = n(2);
            const float len = std::sqrt(x * x + y * y + z * z);
            vu.queueP(len != 0.0f ? 1.0f / len : len, 24u);
        }
        else if constexpr (kFunct == 0x74u) // EATANxy
        {
            const float x = n(0), y = n(1);
            vu.queueP(x != 0.0f ? ps2_vu1_detail::vuEatan(y / x) : 0.0f, 54u);
        }
        else if constexpr (kFunct == 0x75u) // EATANxz
        {
            const float x = n(0), z = n(2);
            vu.queueP(x != 0.0f ? ps2_vu1_detail::vuEatan(z / x) : 0.0f, 54u);
        }
        else if constexpr (kFunct == 0x76u) // ESUM
        {
            float sum = 0.0f;
            for (uint32_t component = 0; component < 4u; ++component)
                sum += n(component);
            vu.queueP(sum, 12u);
        }
        else if constexpr (kFunct == 0x79u) // ERSQRT
        {
            float result = n(kComponent);
            if (result >= 0.0f)
            {
                result = std::sqrt(result);
                if (result != 0.0f)
                    result = 1.0f / result;
            }
            vu.queueP(result, 18u);
        }
        else if constexpr (kFunct == 0x78u) // ESQRT
        {
            const float value = n(kComponent);
            vu.queueP(value >= 0.0f ? std::sqrt(value) : value, 12u);
        }
        else if constexpr (kFunct == 0x7Cu) // ESIN
            vu.queueP(ps2_vu1_detail::vuEsin(n(kComponent)), 29u);
        else if constexpr (kFunct == 0x7Au) // ERCPR
        {
            const float value = n(kComponent);
            vu.queueP(value != 0.0f ? 1.0f / value : value, 12u);
        }
        else if constexpr (kFunct == 0x7Du) // EATAN
            vu.queueP(ps2_vu1_detail::vuEatan(n(kComponent)), 54u);
        else // EEXP
            vu.queueP(ps2_vu1_detail::vuEexp(n(kComponent)), 44u);
    }
};
