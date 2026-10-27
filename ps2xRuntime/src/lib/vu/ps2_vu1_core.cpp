#include "runtime/ps2_vu1_inline.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <ps2_log.h>

namespace
{
    // The VU rounds toward zero: run() sets it for the SSE float math (MXCSR) and the x87 long double exact checks
    // (control word), as std::fesetround(FE_TOWARDZERO) does, and puts the caller's mode back. Written directly:
    // MinGW's fegetround/fesetround cost about 3% of the VU1 thread, called on every run.
    class VuRoundingScope
    {
    public:
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        VuRoundingScope()
        {
            m_csr = _mm_getcsr();
            __asm__ volatile("fnstcw %0" : "=m"(m_cw));
            if ((m_csr & kCsrTowardZero) != kCsrTowardZero)
                _mm_setcsr(m_csr | kCsrTowardZero);
            if ((m_cw & kCwTowardZero) != kCwTowardZero)
            {
                const uint16_t towardZero = static_cast<uint16_t>(m_cw | kCwTowardZero);
                __asm__ volatile("fldcw %0" : : "m"(towardZero));
            }
        }
        ~VuRoundingScope()
        {
            if ((m_csr & kCsrTowardZero) != kCsrTowardZero)
                _mm_setcsr((_mm_getcsr() & ~kCsrTowardZero) | (m_csr & kCsrTowardZero));
            if ((m_cw & kCwTowardZero) != kCwTowardZero)
                __asm__ volatile("fldcw %0" : : "m"(m_cw));
        }

    private:
        static constexpr unsigned int kCsrTowardZero = 0x6000u;
        static constexpr uint16_t kCwTowardZero = 0x0C00u;
        unsigned int m_csr = 0u;
        uint16_t m_cw = 0u;
#else
        VuRoundingScope() : m_previous(std::fegetround()), m_set(std::fesetround(FE_TOWARDZERO) == 0) {}
        ~VuRoundingScope()
        {
            if (m_set && m_previous != -1)
                std::fesetround(m_previous);
        }

    private:
        int m_previous;
        bool m_set;
#endif
    };
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

void VU1Interpreter::resetScheduler()
{
    m_flagPipeline = {};
    m_flagHead = 0u;
    m_flagCount = 0u;
    m_inFlightCount = 0u;
    m_inFlightHidden = false;
    m_fdiv = {};
    m_efu = {};
    resetXgkick();
    m_vfReady = {};
    m_vfReadyMax = {};
    m_accReadyMax = 0;
    m_viReady = {};
    m_accReady = {};
    m_efuResourceReady = 0;
    m_lastWritebackCycle = 0;
    m_nextScalarCommit = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        // A result with exponent 2..253 is far from zero, the denormal range and FLT_MAX: the exact
        // result has the same sign and raises no zero/underflow/overflow flag.
        uint32_t bits = 0u;
        std::memcpy(&bits, &result[component], sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent >= 2u && exponent <= 253u)
        {
            laneFlags[component] = (bits & 0x80000000u) != 0u ? 0x2u : 0u;
            continue;
        }

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        uint32_t leftBits = 0u, rightBits = 0u;
        std::memcpy(&leftBits, &left, sizeof(leftBits));
        std::memcpy(&rightBits, &right, sizeof(rightBits));
        const uint32_t productSign = (leftBits ^ rightBits) & 0x80000000u;
        const int32_t leftExponent = static_cast<int32_t>((leftBits >> 23) & 0xFFu);
        const int32_t rightExponent = static_cast<int32_t>((rightBits >> 23) & 0xFFu);
        if (leftExponent == 0 || rightExponent == 0)
        {
            // Normalized operands: exponent 0 is a signed zero.
            extraSticky |= 0x1u | (productSign != 0u ? 0x2u : 0u);
            continue;
        }
        // The product's exponent is leftExponent + rightExponent - 127, or one more.
        const int32_t productExponent = leftExponent + rightExponent - 127;
        if (productExponent >= 2 && productExponent + 1 <= 253)
        {
            extraSticky |= productSign != 0u ? 0x2u : 0u;
            continue;
        }

        float product = left * right;
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    // A lane's Z/S/U/O flags land at bits lane, lane+4, lane+8 and lane+12 of MAC (x is bit 3).
    static constexpr std::array<uint16_t, 16> kSpread = []
    {
        std::array<uint16_t, 16> spread{};
        for (uint32_t flags = 0; flags < 16u; ++flags)
            spread[flags] = static_cast<uint16_t>((flags & 1u) | ((flags & 2u) << 3) | ((flags & 4u) << 6) | ((flags & 8u) << 9));
        return spread;
    }();
    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        const uint32_t flags = laneFlags[component] & 0xFu;
        mac |= static_cast<uint32_t>(kSpread[flags]) << (3u - component);
        status |= laneFlags[component];
    }

    if (m_flagWritesDead)
    {
        m_deadSticky |= status | extraSticky;
        return;
    }
    FlagPipelineEntry *entry = pushFlagEntry(true, 0xFFFFFFFFu);
    if (!entry)
        return;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky | m_deadSticky;
    m_deadSticky = 0u;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::foldDeadSticky()
{
    // The newest pending status write takes them, else the status register.
    for (uint32_t index = m_flagCount; index-- > 0u;)
    {
        FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + index) % kMaxFlagEntries];
        if (entry.writesStatus)
        {
            entry.extraSticky |= m_deadSticky;
            m_deadSticky = 0u;
            return;
        }
    }
    m_state.status |= (m_deadSticky & 0xFu) << 6;
    m_deadSticky = 0u;
}

void VU1Interpreter::commitLandedFlags()
{
    while (m_flagCount != 0u && m_flagPipeline[m_flagHead].readyCycle <= m_cycle)
    {
        FlagPipelineEntry &entry = m_flagPipeline[m_flagHead];
        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry.valid = false;
        m_flagHead = (m_flagHead + 1u) % kMaxFlagEntries;
        --m_flagCount;
    }
}

void VU1Interpreter::commitScalarPipelines()
{
    {
        m_nextScalarCommit = UINT64_MAX;
        if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
        {
            // Q's status bits land after every flag write ready by now, as with per-cycle commits.
            commitFlags();
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
        else if (m_fdiv.valid)
        {
            noteCommitCycle(m_nextScalarCommit, m_fdiv.readyCycle);
        }

        for (ScalarPipelineEntry &entry : m_efu)
        {
            if (!entry.valid)
                continue;
            if (entry.readyCycle <= m_cycle)
            {
                m_state.p = entry.value;
                entry = {};
            }
            else
            {
                noteCommitCycle(m_nextScalarCommit, entry.readyCycle);
            }
        }
    }

}

namespace
{
    // normalizeFmacExactResult for a double: returns the Z/S/U/O lane flags and updates value.
    uint8_t classifyExactResult(float &value, double exact)
    {
        const bool negative = std::signbit(exact);
        const double magnitude = std::fabs(exact);
        uint8_t flags = negative ? 0x2u : 0u;
        const uint32_t bits = negative ? 0x80000000u : 0u;
        if (magnitude == 0.0)
        {
            flags |= 0x1u;
            std::memcpy(&value, &bits, sizeof(value));
        }
        else if (magnitude > static_cast<double>(std::numeric_limits<float>::max()))
        {
            flags |= 0x8u;
            const uint32_t maxBits = bits | 0x7F7FFFFFu;
            std::memcpy(&value, &maxBits, sizeof(value));
        }
        else if (magnitude < static_cast<double>(std::numeric_limits<float>::min()))
        {
            flags |= 0x5u;
            std::memcpy(&value, &bits, sizeof(value));
        }
        return flags;
    }
}

// Lane flags of a sum at FLT_MAX or above (sum: as a double), from the long double result as fmacFlagsSlow does.
// calculateFmacExactResult's expressions, from the normalized operands (cross products: shuffled).
uint8_t VU1Interpreter::fmacSumAtFloatMaxFlags(float &value, uint32_t lane, double sum, FmacKind kind,
                                               const float left[4], const float right[4], const float acc[4]) const
{
    const long double l = left[lane], r = right[lane], a = acc[lane];
    long double exactResult = 0.0L;
    switch (kind)
    {
    case FmacKind::Add: exactResult = l + r; break;
    case FmacKind::Sub: exactResult = l - r; break;
    case FmacKind::Madd: exactResult = a + l * r; break;
    case FmacKind::Msub: exactResult = a - l * r; break;
    case FmacKind::CrossMsub: exactResult = lane == 3u ? 0.0L : a - l * r; break;
    default: return classifyExactResult(value, sum); // products are exact in double and never get here
    }
    return normalizeFmacExactResult(value, exactResult);
}

void VU1Interpreter::fmacFlagsSlow(float *result, uint8_t dest)
{
    const FmacOperation operation = decodeFmacOperation(m_currentUpperInstruction);
    if (operation.kind != FmacKind::None)
    {
        const uint8_t fs = FS(m_currentUpperInstruction);
        const uint8_t ft = FT(m_currentUpperInstruction);
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const bool cross = operation.kind == FmacKind::CrossMsub || operation.kind == FmacKind::CrossMul;
        const auto left = [&](uint32_t lane) -> double
        { return normalizeOperand(m_state.vf[fs][cross ? crossLeft[lane] : lane]); };
        const auto right = [&](uint32_t lane) -> double
        {
            switch (operation.right)
            {
            case FmacOperation::Right::Broadcast:
                return normalizeOperand(m_state.vf[ft][operation.broadcast]);
            case FmacOperation::Right::Q:
                return normalizeOperand(m_state.q);
            case FmacOperation::Right::I:
                return normalizeOperand(m_state.i);
            default:
                return normalizeOperand(m_state.vf[ft][cross ? crossRight[lane] : lane]);
            }
        };

        float values[4];
        std::memcpy(values, result, sizeof(values));
        uint8_t laneFlags[4]{};
        bool exact = true;
        uint32_t extraSticky = 0u;
        const bool productSum = operation.kind == FmacKind::Madd || operation.kind == FmacKind::Msub ||
                                operation.kind == FmacKind::CrossMsub;
        for (uint32_t lane = 0; lane < 4u && exact; ++lane)
        {
            if ((dest & laneForComponent(lane)) == 0u)
                continue;
            uint32_t bits = 0u;
            std::memcpy(&bits, &values[lane], sizeof(bits));
            const uint32_t exponent = (bits >> 23) & 0xFFu;
            if (exponent >= 2u && exponent <= 253u)
                laneFlags[lane] = (bits & 0x80000000u) != 0u ? 0x2u : 0u;
            else
            {
                double sum = 0.0;
                switch (operation.kind)
                {
                case FmacKind::Add: sum = left(lane) + right(lane); break;
                case FmacKind::Sub: sum = left(lane) - right(lane); break;
                case FmacKind::Mul: sum = left(lane) * right(lane); break;
                case FmacKind::Madd: sum = static_cast<double>(normalizeOperand(m_state.acc[lane])) + left(lane) * right(lane); break;
                case FmacKind::Msub: sum = static_cast<double>(normalizeOperand(m_state.acc[lane])) - left(lane) * right(lane); break;
                case FmacKind::CrossMsub:
                    sum = lane == 3u ? 0.0 : static_cast<double>(normalizeOperand(m_state.acc[lane])) - left(lane) * right(lane);
                    break;
                case FmacKind::CrossMul: sum = lane == 3u ? 0.0 : left(lane) * right(lane); break;
                default: break;
                }
                const bool product = operation.kind == FmacKind::Mul || operation.kind == FmacKind::CrossMul;
                long double exactResult = 0.0L;
                if (!product && std::fabs(sum) >= static_cast<double>(std::numeric_limits<float>::max()) &&
                    calculateFmacExactResult(lane, exactResult))
                    laneFlags[lane] = normalizeFmacExactResult(values[lane], exactResult); // see FmacKind
                else
                    laneFlags[lane] = classifyExactResult(values[lane], sum);
            }

            if (productSum && exact)
            {
                // calculateFmacProductSticky: every product condition accumulates into the sticky flags.
                const float leftValue = static_cast<float>(left(lane));
                const float rightValue = static_cast<float>(right(lane));
                uint32_t leftBits = 0u, rightBits = 0u;
                std::memcpy(&leftBits, &leftValue, sizeof(leftBits));
                std::memcpy(&rightBits, &rightValue, sizeof(rightBits));
                const uint32_t productSign = (leftBits ^ rightBits) & 0x80000000u;
                const int32_t leftExponent = static_cast<int32_t>((leftBits >> 23) & 0xFFu);
                const int32_t rightExponent = static_cast<int32_t>((rightBits >> 23) & 0xFFu);
                const int32_t productExponent = leftExponent + rightExponent - 127;
                if (leftExponent == 0 || rightExponent == 0)
                    extraSticky |= 0x1u | (productSign != 0u ? 0x2u : 0u);
                else if (productExponent >= 2 && productExponent + 1 <= 253)
                    extraSticky |= productSign != 0u ? 0x2u : 0u;
                else
                {
                    float productValue = leftValue * rightValue;
                    extraSticky |= classifyExactResult(productValue, static_cast<double>(leftValue) * rightValue) & 0xFu;
                }
            }
        }
        if (exact)
        {
            std::memcpy(result, values, sizeof(values));
            updateFmacFlags(laneFlags, dest, extraSticky);
            return;
        }
    }

    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
}

void VU1Interpreter::stallUntilReady(const DecodedInstructionPair &decoded, uint64_t readyCycle, uint64_t budgetEnd)
{
    while (readyCycle > m_cycle)
    {
        if (readyCycle >= budgetEnd)
        {
            advanceTo(budgetEnd);
            break;
        }
        advanceTo(readyCycle);
        if (decoded.lowerUsage.pipeline == PipelineXgkick)
            catchUpXgkick();
        readyCycle = calculatePairReadyCycle(decoded);
    }
}

namespace
{
    std::atomic<bool> &relaxedFloatFlag()
    {
        static std::atomic<bool> flag{false};
        return flag;
    }
}

void VU1Interpreter::setRelaxedFloat(bool enabled)
{
    relaxedFloatFlag().store(enabled, std::memory_order_relaxed);
}

bool VU1Interpreter::relaxedFloat()
{
    return relaxedFloatFlag().load(std::memory_order_relaxed);
}

VU1Interpreter::StepResult VU1Interpreter::stepPairInterpreted(const DecodedInstructionPair &decoded,
                                                               const RunContext &context)
{
    return stepPair(decoded, context);
}

VU1Interpreter::StepResult VU1Interpreter::stepPairLogged(const DecodedInstructionPair &decoded, const RunContext &context)
{
    return stepPairImpl<true>(decoded, context);
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const long double magnitude = std::fabs(exactResult);
    const long double maximum = static_cast<long double>(std::numeric_limits<float>::max());
    const long double minimum = static_cast<long double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

void VU1Interpreter::progressXgkickActive(uint64_t cycles)
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    m_xgkick.cycleCredit += static_cast<uint32_t>(std::min<uint64_t>(cycles, 0x40000000u));
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        // Inside a GIFtag's data, a qword every two cycles in one copy, up to the tag's end or the end of VU memory
        // (the tag's size was checked against the buffer when it was read).
        if (m_xgkick.currentTagEnd != 0u && m_xgkick.copiedBytes < m_xgkick.currentTagEnd)
        {
            const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
            const uint32_t qwords = std::min({m_xgkick.cycleCredit / 2u,
                                              (m_xgkick.currentTagEnd - m_xgkick.copiedBytes) / 16u,
                                              (m_activeVuDataSize - source) / 16u});
            if (qwords > 1u)
            {
                std::memcpy(m_xgkick.packet.data() + m_xgkick.copiedBytes, m_activeVuData + source, qwords * 16u);
                m_xgkick.copiedBytes += qwords * 16u;
                m_xgkick.cycleCredit -= qwords * 2u;
                if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
                {
                    if (m_xgkick.currentTagEop)
                        finishXgkick();
                    else
                    {
                        m_xgkick.currentTagEnd = 0u;
                        m_xgkick.currentTagEop = false;
                    }
                }
                continue;
            }
        }
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
        if (source + 16u <= m_activeVuDataSize)
        {
            std::memcpy(m_xgkick.packet.data() + qwordOffset, m_activeVuData + source, 16u);
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
                m_xgkick.packet[qwordOffset + i] = m_activeVuData[(source + i) % m_activeVuDataSize];
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else // IMAGE; FLG 3 ("disable") behaves as IMAGE on the GS
                tagBytes += static_cast<uint64_t>(nloop) * 16u;

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_packetObserver)
        m_packetObserver(m_xgkick.packet.data(), m_xgkick.totalBytes);
    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active || m_cycle < m_lastWritebackCycle)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    // The queue is ordered by readiness: flags are pending while the newest one is.
    if (m_flagCount != 0u && m_flagPipeline[(m_flagHead + m_flagCount - 1u) % kMaxFlagEntries].readyCycle > m_cycle)
        return true;
    return false;
}

void VU1Interpreter::flushPipelines()
{
    // Nothing executes, so the clock jumps from one completion to the next (advanceTo applies everything ready),
    // ending on the same cycle as ticking would: PATH1 to the end of the current GIFtag's data, else to the
    // last pending result.
    catchUpXgkick();
    while (pipelinesPending())
    {
        uint64_t target = m_cycle + 1u;
        if (m_xgkick.active)
        {
            if (m_xgkick.currentTagEnd != 0u && m_xgkick.copiedBytes < m_xgkick.currentTagEnd)
            {
                const uint64_t needed = static_cast<uint64_t>(m_xgkick.currentTagEnd - m_xgkick.copiedBytes) / 16u * 2u;
                if (needed > m_xgkick.cycleCredit)
                    target = m_cycle + (needed - m_xgkick.cycleCredit);
            }
        }
        else
        {
            if (m_fdiv.valid)
                target = std::max(target, m_fdiv.readyCycle);
            for (const ScalarPipelineEntry &entry : m_efu)
                if (entry.valid)
                    target = std::max(target, entry.readyCycle);
            target = std::max(target, m_lastWritebackCycle);
            if (m_flagCount != 0u)
                target = std::max(target, m_flagPipeline[(m_flagHead + m_flagCount - 1u) % kMaxFlagEntries].readyCycle);
        }
        advanceTo(target);
        catchUpXgkick();
    }
}

const VU1Interpreter::DecodedInstructionPair &VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    // Without a PS2Memory the caller owns the code buffer and reports changes via invalidateDecodedCode().
    const bool trackedVuCode = memory == nullptr ||
                               (m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                               (m_unit == Unit::VU0 && vuCode == memory->getVU0Code());
    const uint32_t pairIndex = pc / 8u;
    if ((pc & 7u) != 0u || !trackedVuCode || pairIndex >= kMaxDecodedPairs)
    {
        m_uncachedDecodedPair = decodeInstructionPair(vuCode, pc);
        return m_uncachedDecodedPair;
    }

    // Pairs are decoded on first use and stamped with the code generation they were decoded from, so a
    // microcode upload only costs the pairs that actually run afterwards.
    if (m_cachedVuCode != vuCode || m_cachedMemory != memory || m_cachedCodeSize != codeSize)
    {
        m_decodedPairStamp.fill(0u);
        m_cachedVuCode = vuCode;
        m_cachedMemory = memory;
        m_cachedCodeSize = codeSize;
    }
    const uint64_t stamp = memory == nullptr ? m_externalCodeGeneration + 1u
                           : (m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration()) + 1u;
    if (m_decodedPairStamp[pairIndex] != stamp)
    {
        m_decodedCodeCache[pairIndex] = decodeInstructionPair(vuCode, pc);
        m_decodedPairStamp[pairIndex] = stamp;
    }
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::applyInFlightWrite(const InFlightWrite &entry, bool newValue)
{
    const float *value = newValue ? entry.newValue : entry.oldValue;
    if (entry.kind == InFlightWrite::Kind::Vi)
    {
        m_state.vi[entry.reg] = static_cast<int32_t>(value[0]);
        return;
    }
    float *target = entry.kind == InFlightWrite::Kind::Acc ? m_state.acc : m_state.vf[entry.reg];
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((entry.lanes & laneForComponent(component)) != 0u)
            target[component] = value[component];
    }
}

void VU1Interpreter::hideInFlightWrites()
{
    // The run stopped before these results' writeback: show the registers as the pipeline would,
    // undoing the newest writes first.
    for (uint32_t back = 1u; back <= m_inFlightCount; ++back)
    {
        const InFlightWrite &entry = m_inFlight[(m_inFlightNext + kInFlightWrites - back) % kInFlightWrites];
        if (entry.readyCycle > m_cycle)
        {
            applyInFlightWrite(entry, false);
            m_inFlightHidden = true;
        }
    }
}

void VU1Interpreter::revealInFlightWrites()
{
    if (!m_inFlightHidden)
        return;
    m_inFlightHidden = false;
    for (uint32_t forward = m_inFlightCount; forward > 0u; --forward)
    {
        const InFlightWrite &entry = m_inFlight[(m_inFlightNext + kInFlightWrites - forward) % kInFlightWrites];
        if (entry.readyCycle > m_cycle) // hidden at the end of the previous run (the clock has not moved)
            applyInFlightWrite(entry, true);
    }
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles, bool runToCompletion)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles, runToCompletion);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

namespace
{
    std::vector<VU1Interpreter::CompiledFragment> &compiledFragmentRegistry()
    {
        static std::vector<VU1Interpreter::CompiledFragment> registry;
        return registry;
    }

    // PS2X_VU_AOT=0 runs every pair through the interpreter.
    bool compiledFragmentsEnabled()
    {
        static const bool enabled = []
        {
            const char *value = std::getenv("PS2X_VU_AOT");
            return !(value && value[0] == '0');
        }();
        return enabled;
    }
}

namespace
{
    // Bumped by every registration: images resolved before it are resolved again.
    std::atomic<uint64_t> &registryGeneration()
    {
        static std::atomic<uint64_t> generation{1u};
        return generation;
    }

    std::vector<VU1Interpreter::LiftedEntry> &liftedEntryRegistry()
    {
        static std::vector<VU1Interpreter::LiftedEntry> registry;
        return registry;
    }

    // PS2X_VU_LIFTED=0 leaves lifted microcode out (the compiled blocks run instead).
    bool liftedEntriesEnabled()
    {
        static const bool enabled = []
        {
            const char *value = std::getenv("PS2X_VU_LIFTED");
            return !(value && value[0] == '0');
        }();
        return enabled;
    }
}

void VU1Interpreter::registerLiftedEntry(const LiftedEntry &lifted)
{
    liftedEntryRegistry().push_back(lifted);
    registryGeneration().fetch_add(1u, std::memory_order_relaxed);
}

void VU1Interpreter::registerCompiledFragment(const CompiledFragment &fragment)
{
    // By length: refreshCompiledFragments() applies the longest matching fragment last (a whole program over the
    // separately uploaded pieces it contains, whose blocks stop at the piece boundaries).
    auto &registry = compiledFragmentRegistry();
    registry.insert(std::upper_bound(registry.begin(), registry.end(), fragment,
                                     [](const CompiledFragment &left, const CompiledFragment &right)
                                     { return left.pairCount < right.pairCount; }),
                    fragment);
    registryGeneration().fetch_add(1u, std::memory_order_relaxed);
}

void VU1Interpreter::refreshCompiledFragments(const uint8_t *vuCode, uint32_t codeSize, uint64_t stamp)
{
    m_compiledChecked = true;
    m_compiledStamp = stamp;
    m_compiledCode = vuCode;
    const uint64_t generation = registryGeneration().load(std::memory_order_relaxed);
    CompiledImage *image = nullptr;
    for (CompiledImage &cached : m_compiledImages)
        if (cached.registryGeneration == generation && cached.code.size() == codeSize &&
            std::memcmp(cached.code.data(), vuCode, codeSize) == 0)
        {
            image = &cached;
            break;
        }
    if (image)
    {
        for (const auto &[replaced, entry] : image->replaced)
            *replaced = entry;
    }
    else
    {
        image = &*std::min_element(m_compiledImages.begin(), m_compiledImages.end(),
                                   [](const CompiledImage &left, const CompiledImage &right)
                                   { return left.lastUse < right.lastUse; });
        image->code.assign(vuCode, vuCode + codeSize);
        image->registryGeneration = generation;
        image->byPair.clear();
        image->replaced.clear();
        resolveCompiledImage(*image);
    }
    image->lastUse = ++m_compiledImageUses;
    m_compiledImage = image;
}

void VU1Interpreter::resolveCompiledImage(CompiledImage &image) const
{
    const uint8_t *vuCode = image.code.data();
    const uint32_t pairCount = static_cast<uint32_t>(image.code.size() / 8u);
    std::vector<CompiledEntry> &byPair = image.byPair;
    // Versions of one program share their ranges: hash each range once per refresh (games upload microcode often).
    struct RangeHash
    {
        uint32_t firstPair;
        uint32_t pairCount;
        uint64_t hash;
    };
    std::vector<RangeHash> rangeHashes;
    for (const CompiledFragment &fragment : compiledFragmentRegistry())
    {
        if (fragment.unit != m_unit || fragment.firstPair + fragment.pairCount > pairCount)
            continue;
        auto range = std::find_if(rangeHashes.begin(), rangeHashes.end(), [&](const RangeHash &entry)
                                  { return entry.firstPair == fragment.firstPair && entry.pairCount == fragment.pairCount; });
        if (range == rangeHashes.end())
            range = rangeHashes.insert(rangeHashes.end(),
                                       {fragment.firstPair, fragment.pairCount,
                                        codeHash(vuCode + fragment.firstPair * 8u, fragment.pairCount * 8u)});
        if (range->hash != fragment.codeHash)
            continue;
        if (byPair.empty())
            byPair.assign(pairCount, nullptr);
        std::copy_n(fragment.pairs, fragment.pairCount, byPair.begin() + fragment.firstPair);
    }
    if (byPair.empty() || !liftedEntriesEnabled())
        return;
    for (const LiftedEntry &lifted : liftedEntryRegistry())
    {
        if (lifted.unit != m_unit || lifted.firstPair + lifted.pairCount > pairCount || lifted.pair >= pairCount)
            continue;
        const auto range = std::find_if(rangeHashes.begin(), rangeHashes.end(), [&](const RangeHash &entry)
                                        { return entry.firstPair == lifted.firstPair && entry.pairCount == lifted.pairCount; });
        if (range == rangeHashes.end() || range->hash != lifted.codeHash || !byPair[lifted.pair])
            continue;
        *lifted.replaced = byPair[lifted.pair];
        image.replaced.push_back({lifted.replaced, byPair[lifted.pair]});
        byPair[lifted.pair] = lifted.entry;
    }
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles, bool runToCompletion)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const VuRoundingScope rounding;
    const bool relaxed = m_unit == Unit::VU1 && relaxedFloat();
    const unsigned int previousCsr = relaxed ? _mm_getcsr() : 0u;
    if (relaxed)
        _mm_setcsr(previousCsr | 0x8040u); // flush to zero, denormals are zero
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    commitReadyPipelines(); // afterwards every clock advance commits what became ready
    revealInFlightWrites();
    // Only short time slices can stop with writes in flight that a caller then inspects; run-to-completion
    // budgets (VIF MSCAL/MSCNT, VU0 microprograms) skip the logging.
    const bool timeSlice = !runToCompletion && maxCycles < (1u << 20);
    m_logInFlight = timeSlice;
    if (!m_logInFlight)
        m_inFlightCount = 0u;

    RunContext context;
    context.vuData = vuData;
    context.dataSize = dataSize;
    context.codeSize = codeSize;
    context.gs = &gs;
    context.memory = memory;
    context.budgetEnd = budgetEnd;

    // Microcode compiled ahead of time replaces the interpreter on the pairs whose bytes it was compiled from.
    const CompiledEntry *compiled = nullptr;
    const bool vu1 = m_unit == Unit::VU1;
    const bool trackedVuCode = memory == nullptr || vuCode == (vu1 ? memory->getVU1Code() : memory->getVU0Code());
    if (trackedVuCode && !m_compiledCodeDisabled && compiledFragmentsEnabled() && !compiledFragmentRegistry().empty())
    {
        const uint64_t stamp = memory == nullptr ? m_externalCodeGeneration + 1u
                                                 : (vu1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration()) + 1u;
        if (!m_compiledChecked || stamp != m_compiledStamp || vuCode != m_compiledCode)
            refreshCompiledFragments(vuCode, codeSize, stamp);
        if (!m_compiledImage->byPair.empty())
            compiled = m_compiledImage->byPair.data();
    }

    while (canIssue(context))
    {
        StepResult result;
        if (compiled && (m_state.pc & 7u) == 0u && compiled[m_state.pc >> 3])
            result = compiled[m_state.pc >> 3](*this, context);
        else
            result = stepPair(getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc), context);
        if (result == StepResult::Continue)
            continue;
        programEnded = result == StepResult::Ended;
        break;
    }
    catchUpXgkick();

    if (!programEnded)
        hideInFlightWrites();

    // Small budgets are deliberate time slices; a run-to-completion budget that expires means a
    // runaway program.
    if (!programEnded && m_cycle >= budgetEnd && !timeSlice)
    {
        static uint32_t s_budgetWarnings = 0u;
        if (s_budgetWarnings < 8u)
        {
            ++s_budgetWarnings;
            std::fprintf(stderr, "[VU%u] program still running after %u cycles at pc=0x%x\n",
                         m_unit == Unit::VU1 ? 1u : 0u, maxCycles, m_state.pc);
        }
    }
    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    commitFlags();
    m_state.cycles = m_cycle;
    if (relaxed)
        _mm_setcsr((_mm_getcsr() & ~0x8040u) | (previousCsr & 0x8040u));
}
