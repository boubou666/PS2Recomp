// Compiles the VU1 microcode found in a PS2 ELF ahead of time.
//   ps2_vu_aot <elf> <output-dir> [file-prefix]
//
// Games upload microcode with VIF MPG commands from DMA packets that usually sit in the ELF's data. The tool
// scans the loadable segments for MPG VIFcodes, keeps the fragments whose pairs decode as VU1 instructions, and
// writes one C++ file per fragment. Each pair becomes a call to VU1Interpreter::stepPair() with a constexpr
// decodePair(), so the compiler folds the decode and the opcode switches; the semantics stay the interpreter's.
// Straight-line runs of pairs also become one function each (VU1Interpreter::runBlock), which issues every pair at
// the cycle planBlock() works out at compile time.
// At run time a fragment replaces the interpreter only while VU1 code memory holds its exact bytes at its
// address (VU1Interpreter::refreshCompiledFragments), so stray matches in the scan are harmless.
#include "runtime/ps2_vu1_inline.h"
#include "runtime/ps2_vu1_jit.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <deque>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    constexpr uint32_t kVu1CodePairs = 0x4000u / 8u;

    struct Fragment
    {
        uint32_t fileOffset = 0; // of the code, after the VIFcode (the first one's, for a chain of MPGs)
        uint32_t firstPair = 0;
        uint32_t pairCount = 0;
        uint64_t hash = 0;
        const uint8_t *code = nullptr;
        uint32_t chunks = 1; // MPG commands it was uploaded with
    };

    uint32_t readU32(const std::vector<uint8_t> &file, size_t offset)
    {
        uint32_t value = 0;
        std::memcpy(&value, file.data() + offset, sizeof(value));
        return value;
    }

    uint16_t readU16(const std::vector<uint8_t> &file, size_t offset)
    {
        uint16_t value = 0;
        std::memcpy(&value, file.data() + offset, sizeof(value));
        return value;
    }

    bool decodesAsMicrocode(const uint8_t *code, uint32_t pairCount)
    {
        uint32_t reserved = 0, zero = 0;
        for (uint32_t pair = 0; pair < pairCount; ++pair)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, code + pair * 8u, 4);
            std::memcpy(&upper, code + pair * 8u + 4u, 4);
            if (lower == 0u && upper == 0u)
                ++zero;
            const VU1Interpreter::DecodedInstructionPair decoded =
                VU1Interpreter::decodePair(VU1Interpreter::Unit::VU1, lower, upper);
            if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
                ++reserved;
        }
        // MIPS code with a 0x4A top byte (COP2) passes the VIFcode test; its "pairs" are mostly reserved.
        return reserved * 4u <= pairCount && zero < pairCount;
    }

    std::string hex(uint64_t value, int digits)
    {
        char text[32];
        std::snprintf(text, sizeof(text), "%0*llx", digits, static_cast<unsigned long long>(value));
        return text;
    }

    // MAC/status flag readers among lower instructions: FSEQ, FSSET, FSAND, FSOR, FMEQ, FMAND, FMOR.
    bool readsMacStatus(const VU1Interpreter::DecodedInstructionPair &pair)
    {
        const uint32_t op = pair.lower >> 25;
        return !pair.iBit && op >= 0x14u && op <= 0x1Bu && op != 0x19u;
    }

    // A reader that can see the sticky status bits (or FSSET, which writes them).
    bool readsStickyStatus(const VU1Interpreter::DecodedInstructionPair &pair)
    {
        if (pair.iBit)
            return false;
        const uint32_t op = pair.lower >> 25;
        if (op == 0x16u) // FSAND
            return (((((pair.lower >> 21) & 1u) << 11) | (pair.lower & 0x7FFu)) & 0xFC0u) != 0u;
        return op == 0x14u || op == 0x15u || op == 0x17u;
    }

    // Pairs whose FMAC MAC/status writes no flag reader can see (VU1Interpreter::kHintFlagsUnobserved). A write
    // lands 4 cycles after it issues and a reader sees the last write landed: a reader is blocked from an earlier
    // write by any write at least 4 pairs before it (each pair takes a cycle or more). Backwards over the fragment's
    // control flow, each pair keeps the distances (in pairs, 4 meaning 4 or more) at which it may reach a reader
    // nothing blocks; a jump register, the end of a program, a D/T bit or leaving the fragment count as a reader
    // right there.
    std::vector<bool> findUnobservedFlagWrites(const std::vector<VU1Interpreter::DecodedInstructionPair> &pairs,
                                               uint32_t firstPair)
    {
        using Vu = VU1Interpreter;
        const uint32_t count = static_cast<uint32_t>(pairs.size());
        const uint32_t kExit = count; // virtual successor: a reader at distance 0
        std::vector<std::vector<uint32_t>> successors(count);
        const auto writesFlags = [&](uint32_t index)
        {
            const Vu::DecodedInstructionPair &pair = pairs[index];
            return ps2_vu1_detail::decodeFmacOperation(pair.upper).kind != ps2_vu1_detail::FmacKind::None && ((pair.upper >> 21) & 0xFu) != 0u;
        };
        const auto isBranch = [&](uint32_t index)
        { return !pairs[index].iBit && pairs[index].lowerUsage.pipeline == Vu::PipelineBranch; };
        for (uint32_t index = 0; index < count; ++index)
        {
            std::vector<uint32_t> &next = successors[index];
            const Vu::DecodedInstructionPair &pair = pairs[index];
            if (pair.dBit || pair.tBit)
                next.push_back(kExit);
            const bool delaySlot = index > 0u && (isBranch(index - 1u) || pairs[index - 1u].eBit);
            if (!delaySlot)
            {
                next.push_back(index + 1u < count ? index + 1u : kExit);
                continue;
            }
            if (isBranch(index) || pairs[index - 1u].eBit)
            {
                next.push_back(kExit); // a branch in a delay slot, or the end of the program
                continue;
            }
            const Vu::DecodedInstructionPair &branch = pairs[index - 1u];
            const uint32_t opcode = branch.lower >> 25;
            const bool unconditional = opcode == 0x20u || opcode == 0x21u;
            const bool staticTarget = unconditional || opcode == 0x28u || opcode == 0x29u || (opcode >= 0x2Cu && opcode <= 0x2Fu);
            if (!staticTarget)
            {
                next.push_back(kExit); // JR, JALR
                continue;
            }
            int32_t offset = static_cast<int32_t>(branch.lower & 0x7FFu);
            if (offset & 0x400)
                offset -= 0x800;
            const int64_t target = static_cast<int64_t>(index) + offset; // the branch's pc + 8 + offset * 8
            next.push_back(target >= 0 && target < static_cast<int64_t>(count) ? static_cast<uint32_t>(target) : kExit);
            if (!unconditional)
                next.push_back(index + 1u < count ? index + 1u : kExit);
        }

        // reach[i]: bit d set when pair i reaches an unblocked reader d pairs on (bit 4: 4 or more).
        std::vector<uint8_t> reach(count, 0u);
        const auto blocked = [&](uint32_t index) -> uint8_t
        {
            if (index == kExit)
                return 0x1u;
            return writesFlags(index) ? static_cast<uint8_t>(reach[index] & 0x0Fu) : reach[index];
        };
        const auto shift = [](uint8_t distances) { return static_cast<uint8_t>(((distances << 1) | (distances & 0x10u)) & 0x1Fu); };
        for (bool changed = true; changed;)
        {
            changed = false;
            for (uint32_t index = count; index-- > 0u;)
            {
                uint8_t value = readsMacStatus(pairs[index]) ? 0x1u : 0x0u;
                for (uint32_t next : successors[index])
                    value = static_cast<uint8_t>(value | shift(blocked(next)));
                if (value != reach[index])
                {
                    reach[index] = value;
                    changed = true;
                }
            }
        }
        std::vector<bool> unobserved(count, false);
        for (uint32_t index = 0; index < count; ++index)
        {
            if (!writesFlags(index))
                continue;
            uint8_t seen = 0u;
            for (uint32_t next : successors[index])
                seen = static_cast<uint8_t>(seen | blocked(next));
            unobserved[index] = seen == 0u;
        }
        (void)firstPair;
        return unobserved;
    }


    // ---- Recompiled blocks (ps2_vu1_jit.h) ----
    struct JitPairInfo
    {
        Vu1Jit::UpperInfo upper;
        Vu1Jit::Lower lower = Vu1Jit::Lower::Nop;
        bool fallback = false;  // through the interpreter's pair body
        bool liveFlags = false; // an FMAC whose flags a reader can see (Vu1Jit::fmacLive)
        bool stickyFlags = false; // an FMAC whose sticky status bits are all a reader can see (Vu1Jit::fmacSticky)
    };

    // PS2X_VU_AOT_RELAXED=1: FMAC instructions of recompiled blocks as plain SSE (Vu1Jit::fmacRelaxed).
    bool relaxedFloat()
    {
        static const bool relaxed = std::getenv("PS2X_VU_AOT_RELAXED") != nullptr;
        return relaxed;
    }

    // PS2X_VU_AOT_FAST=1: VU1 microcode accurate enough rather than exact (Vu1Jit::fastEnter): relaxed floats, the
    // blocks' static schedule without readiness checks or tables. VU0 microcode stays exact.
    bool fastVu1()
    {
        static const bool fast = std::getenv("PS2X_VU_AOT_FAST") != nullptr;
        return fast;
    }

    std::string vfName(uint32_t reg) { return "vf" + std::to_string(reg); }
    std::string viName(uint32_t reg) { return reg == 0u ? std::string("vi0") : "vi" + std::to_string(reg); }

    // One block as one function: VU registers in locals, each instruction a Vu1Jit call. Returns false when the
    // block keeps the per-pair path (a branch or E bit anywhere but at its end, a delay slot it cannot recompile).
    bool writeJitBlock(std::ostream &out, const std::vector<VU1Interpreter::DecodedInstructionPair> &decoded,
                       uint32_t first, uint32_t end, uint32_t firstPair, const std::vector<bool> &unobserved,
                       const std::vector<bool> &stickyOnly, const VU1Interpreter::BlockPlan &plan, uint32_t pcMask,
                       bool fast = false)
    {
        using Vu = VU1Interpreter;
        using J = Vu1Jit;
        const uint32_t count = end - first;
        const auto upperOf = [&](uint32_t k) { return decoded[first + k].upper; };
        const auto lowerOf = [&](uint32_t k) { return decoded[first + k].lower; };
        const auto isBranch = [&](uint32_t k)
        { return !decoded[first + k].iBit && J::lowerKind(lowerOf(k)) == J::Lower::Branch; };
        const bool branchEnd = count >= 2u && isBranch(count - 2u);
        const bool eBitEnd = count >= 2u && decoded[first + count - 2u].eBit;
        // A branch or E bit on the last pair (its delay slot past the block, e.g. the fragment's end): the block
        // leaves it pending, as the interpreter does after that pair.
        const bool branchLast = !branchEnd && !eBitEnd && isBranch(count - 1u);
        const bool eBitLast = !branchEnd && !eBitEnd && decoded[first + count - 1u].eBit;
        if (branchLast && eBitLast)
            return false;
        for (uint32_t k = 0; k < count; ++k)
        {
            const bool tail = (k + 2u == count && (branchEnd || eBitEnd)) || (k + 1u == count && (branchLast || eBitLast));
            if ((isBranch(k) || decoded[first + k].eBit) && !tail)
                return false;
        }

        std::vector<JitPairInfo> info(count);
        for (uint32_t k = 0; k < count; ++k)
        {
            const Vu::DecodedInstructionPair &pair = decoded[first + k];
            JitPairInfo &pi = info[k];
            pi.upper = J::upperInfo(pair.upper);
            if ((pi.upper.kind == J::Upper::Fmac || pi.upper.kind == J::Upper::FmacAcc) && ((pair.upper >> 21) & 0xFu) == 0u)
                pi.upper = {J::Upper::Nop, 0}; // no lane written, no flag either
            pi.lower = pair.iBit ? J::Lower::Nop : J::lowerKind(pair.lower);
            const bool fmac = pi.upper.kind == J::Upper::Fmac || pi.upper.kind == J::Upper::FmacAcc;
            const bool relaxed = relaxedFloat() || fast;
            pi.liveFlags = fmac && ((pair.upper >> 21) & 0xFu) != 0u && !unobserved[first + k] &&
                           (relaxed || !stickyOnly[first + k]);
            pi.stickyFlags = fmac && !relaxed && stickyOnly[first + k];
            pi.fallback = pi.upper.kind == J::Upper::Unsupported || pi.lower == J::Lower::Unsupported;
        }
        // The last two pairs end the block: the interpreter cannot run them in the middle of it.
        for (uint32_t k = count >= 2u ? count - 2u : 0u; k < count; ++k)
            if (info[k].fallback && (branchEnd || eBitEnd || branchLast || eBitLast))
                return false;

        // Registers the block touches (loaded on entry) and writes (stored on exit).
        std::vector<bool> vfUsed(32, false), vfWritten(32, false), viUsed(16, false), viWritten(16, false);
        bool accUsed = false, accWritten = false;
        for (uint32_t k = 0; k < count; ++k)
        {
            const Vu::DecodedInstructionPair &pair = decoded[first + k];
            const uint32_t upper = pair.upper, lower = pair.lower;
            vfUsed[ps2_vu1_detail::FS(upper)] = vfUsed[ps2_vu1_detail::FT(upper)] = true;
            vfUsed[ps2_vu1_detail::FD(upper)] = true;
            if (pair.upperUsage.accRead || pair.upperUsage.accWrite)
                accUsed = true;
            if (pair.upperUsage.accWrite)
                accWritten = true;
            if (pair.upperUsage.vfWrite.reg)
                vfWritten[pair.upperUsage.vfWrite.reg] = true;
            if (info[k].upper.kind != J::Upper::Nop && info[k].upper.kind != J::Upper::Clip &&
                info[k].upper.kind != J::Upper::FmacAcc)
                vfWritten[info[k].upper.target] = true;
            if (!pair.iBit)
            {
                vfUsed[ps2_vu1_detail::FS(lower)] = vfUsed[ps2_vu1_detail::FT(lower)] = true;
                viUsed[ps2_vu1_detail::VIS(lower)] = viUsed[ps2_vu1_detail::VIT(lower)] = true;
                viUsed[ps2_vu1_detail::VID(lower)] = true;
                if (pair.lowerUsage.vfWrite.reg)
                    vfWritten[pair.lowerUsage.vfWrite.reg] = true;
                for (uint32_t reg = 1; reg < 16u; ++reg)
                    if (pair.lowerUsage.viWrite & (1u << reg))
                        viWritten[reg] = true;
                // Recompiled lower instructions write their decoded targets; mark them in case the usage omits one.
                switch (info[k].lower)
                {
                case J::Lower::Lq: case J::Lower::Move: case J::Lower::Mr32: case J::Lower::Lqi: case J::Lower::Lqd:
                case J::Lower::Mfir: case J::Lower::Mfp: case J::Lower::Rnext: case J::Lower::Rget:
                    vfWritten[ps2_vu1_detail::FT(lower)] = true;
                    break;
                case J::Lower::FlagRead:
                {
                    const uint32_t op = lower >> 25;
                    if (op == 0x10u || op == 0x12u || op == 0x13u) // FCEQ, FCAND, FCOR write vi1
                        viUsed[1] = viWritten[1] = true;
                    else if (ps2_vu1_detail::VIT(lower) != 0u)
                        viWritten[ps2_vu1_detail::VIT(lower)] = true;
                    break;
                }
                default:
                    break;
                }
            }
        }
        vfWritten[0] = false;
        viWritten[0] = false;

        const uint32_t id = firstPair + first;
        const auto cycle = [&](uint32_t k) { return "base + " + std::to_string(plan.issue[k]) + "u"; };
        const auto hex32 = [](uint32_t value) { return "0x" + hex(value, 8) + "u"; };
        std::ostringstream body;
        const auto storeAll = [&](const char *indent)
        {
            for (uint32_t reg = 1; reg < 32u; ++reg)
                if (vfWritten[reg])
                    body << indent << "J::store(s.vf[" << reg << "], " << vfName(reg) << ");\n";
            if (accWritten)
                body << indent << "J::store(s.acc, acc);\n";
            for (uint32_t reg = 1; reg < 16u; ++reg)
                if (viWritten[reg])
                    body << indent << "s.vi[" << reg << "] = " << viName(reg) << ";\n";
        };
        const auto loadAll = [&](const char *indent)
        {
            for (uint32_t reg = 1; reg < 32u; ++reg)
                if (vfUsed[reg])
                    body << indent << vfName(reg) << " = J::load(s.vf[" << reg << "]);\n";
            if (accUsed)
                body << indent << "acc = J::load(s.acc);\n";
            for (uint32_t reg = 1; reg < 16u; ++reg)
                if (viUsed[reg])
                    body << indent << viName(reg) << " = s.vi[" << reg << "];\n";
        };

        // A conditional branch reads the old value of a VI the pair before it wrote (readBranchVi).
        std::string branchOldVi;
        int branchOldReg = -1;
        if (branchEnd || branchLast)
        {
            const uint32_t b = branchEnd ? count - 2u : count - 1u;
            const uint32_t opcode = lowerOf(b) >> 25;
            if (opcode >= 0x28u && b > 0u)
            {
                const Vu::DecodedInstructionPair &before = decoded[first + b - 1u];
                if (!before.iBit && before.lowerUsage.delaysNextBranchRead)
                    for (uint32_t reg = 1; reg < 16u; ++reg)
                        if (before.lowerUsage.viWrite & (1u << reg))
                            branchOldReg = static_cast<int>(reg);
            }
        }
        // The VI a branch right after the block would read as it was (recordViWriteForBranch).
        int lastViReg = -1;
        {
            const Vu::DecodedInstructionPair &last = decoded[first + count - 1u];
            if (!last.iBit && last.lowerUsage.delaysNextBranchRead)
                for (uint32_t reg = 1; reg < 16u; ++reg)
                    if (last.lowerUsage.viWrite & (1u << reg))
                        lastViReg = static_cast<int>(reg);
        }

        bool usesSticky = false;
        for (uint32_t k = 0; k < count; ++k)
            usesSticky = usesSticky || info[k].stickyFlags;
        const auto foldSticky = [&](const char *indent)
        {
            if (usesSticky)
                body << indent << "J::foldSticky(vu, stickySign, stickyZero);\n";
        };

        bool needsBodies = false;
        for (uint32_t k = 0; k < count; ++k)
        {
            const Vu::DecodedInstructionPair &pair = decoded[first + k];
            const JitPairInfo &pi = info[k];
            const uint32_t pc = (firstPair + first + k) * 8u;
            const uint32_t upper = pair.upper, lower = pair.lower;
            const uint8_t hints = static_cast<uint8_t>(plan.knownNormal[k] | (unobserved[first + k] ? Vu::kHintFlagsUnobserved : 0u));
            body << "        // pair " << (firstPair + first + k) << " (pc 0x" << hex(pc, 4) << ")\n";
            if (branchOldReg >= 0 && k + (branchEnd ? 3u : 2u) == count)
                body << "        const int32_t viBranchOld = " << viName(static_cast<uint32_t>(branchOldReg)) << ";\n";
            if (lastViReg >= 0 && k + 1u == count)
                body << "        const int32_t viLastOld = " << viName(static_cast<uint32_t>(lastViReg)) << ";\n";
            if (pi.fallback)
            {
                needsBodies = true;
                body << "        {\n";
                foldSticky("            ");
                storeAll("            ");
                body << "            J::advance(vu, " << cycle(k) << ");\n"
                     << "            jitBody" << (firstPair + first + k) << "(vu, context);\n";
                loadAll("            ");
                body << "        }\n";
                continue;
            }
            const auto fs = vfName(ps2_vu1_detail::FS(upper)), ft = vfName(ps2_vu1_detail::FT(upper));
            const bool needsClock = pi.upper.kind == J::Upper::Clip ||
                                    ((pi.upper.kind == J::Upper::Fmac || pi.upper.kind == J::Upper::FmacAcc) &&
                                     (ps2_vu1_detail::decodeFmacOperation(upper).right == ps2_vu1_detail::FmacOperation::Right::Q ||
                                      pi.liveFlags)) ||
                                    pi.lower == J::Lower::Div || pi.lower == J::Lower::Sqrt || pi.lower == J::Lower::Rsqrt ||
                                    pi.lower == J::Lower::Mfp || pi.lower == J::Lower::Fcset || pi.lower == J::Lower::Fsset ||
                                    pi.lower == J::Lower::Efu;
            if (needsClock)
                body << "        J::advance(vu, " << cycle(k) << ");\n";
            if ((pi.liveFlags && !relaxedFloat() && !fast) || pi.lower == J::Lower::Fsset)
                foldSticky("        ");

            std::ostringstream up;
            switch (pi.upper.kind)
            {
            case J::Upper::Fmac:
            case J::Upper::FmacAcc:
                if (relaxedFloat() || fast)
                    up << "J::fmacRelaxed<" << hex32(upper) << ", " << (pi.liveFlags ? "true" : "false") << ">(vu, ";
                else if (pi.liveFlags)
                    up << "J::fmacLive<" << hex32(upper) << ", " << static_cast<unsigned>(hints) << "u, "
                       << (plan.deadFlags[k] ? "true" : "false") << ">(vu, ";
                else if (pi.stickyFlags)
                    up << "J::fmacSticky<" << hex32(upper) << ", " << static_cast<unsigned>(hints) << "u>(vu, ";
                else
                    up << "J::fmac<" << hex32(upper) << ", " << static_cast<unsigned>(hints) << "u>(vu, ";
                up
                   << (pi.upper.kind == J::Upper::FmacAcc ? std::string("acc") : vfName(pi.upper.target)) << ", " << fs
                   << ", " << ft << ", acc" << (pi.stickyFlags ? ", stickySign, stickyZero" : "") << ");";
                break;
            case J::Upper::Max:
            case J::Upper::Mini:
                up << "J::maxMini<" << hex32(upper) << ", " << (pi.upper.kind == J::Upper::Max ? "true" : "false")
                   << ">(vu, " << vfName(pi.upper.target) << ", " << fs << ", " << ft << ");";
                break;
            case J::Upper::Itof:
                up << "J::itof<" << hex32(upper) << ">(" << vfName(pi.upper.target) << ", " << fs << ");";
                break;
            case J::Upper::Ftoi:
                up << "J::ftoi<" << hex32(upper) << ", 0u>(" << vfName(pi.upper.target) << ", " << fs << ");";
                break;
            case J::Upper::Abs:
                up << "J::abs<" << hex32(upper) << ">(" << vfName(pi.upper.target) << ", " << fs << ");";
                break;
            case J::Upper::Clip:
                up << "J::clip(vu, " << fs << ", " << ft << ");";
                break;
            default:
                break;
            }

            std::ostringstream lo;
            if (!pair.iBit)
            {
                const uint8_t lft = ps2_vu1_detail::FT(lower), lfs = ps2_vu1_detail::FS(lower);
                const uint8_t vit = ps2_vu1_detail::VIT(lower), vis = ps2_vu1_detail::VIS(lower), vid = ps2_vu1_detail::VID(lower);
                const std::string L = hex32(lower);
                const std::string viT = vit ? viName(vit) : std::string("viDiscard");
                const std::string viD = vid ? viName(vid) : std::string("viDiscard");
                switch (pi.lower)
                {
                case J::Lower::Lq: lo << "J::lq<" << L << ">(context, " << vfName(lft) << ", " << viName(vis) << ");"; break;
                case J::Lower::Sq: lo << "J::sq<" << L << ">(vu, context, " << cycle(k) << ", " << vfName(lfs) << ", " << viName(vit) << ");"; break;
                case J::Lower::Ilw: lo << "J::ilw<" << L << ">(context, " << viT << ", " << viName(vis) << ");"; break;
                case J::Lower::Isw: lo << "J::isw<" << L << ">(vu, context, " << cycle(k) << ", " << viName(vit) << ", " << viName(vis) << ");"; break;
                case J::Lower::Iaddiu: lo << viT << " = static_cast<int16_t>(" << viName(vis) << " + (" << J::imm15(lower) << "));"; break;
                case J::Lower::Isubiu: lo << viT << " = static_cast<int16_t>(" << viName(vis) << " - (" << J::imm15(lower) << "));"; break;
                case J::Lower::Iadd: lo << viD << " = static_cast<int16_t>(" << viName(vis) << " + " << viName(vit) << ");"; break;
                case J::Lower::Isub: lo << viD << " = static_cast<int16_t>(" << viName(vis) << " - " << viName(vit) << ");"; break;
                case J::Lower::Iaddi:
                    lo << viT << " = static_cast<int16_t>(" << viName(vis) << " + ("
                       << static_cast<int>(static_cast<int16_t>(static_cast<int32_t>(((lower >> 6) & 0x1Fu) << 27) >> 27)) << "));";
                    break;
                case J::Lower::Iand: lo << viD << " = " << viName(vis) << " & " << viName(vit) << ";"; break;
                case J::Lower::Ior: lo << viD << " = " << viName(vis) << " | " << viName(vit) << ";"; break;
                case J::Lower::Move: lo << "J::move<" << L << ">(" << vfName(lft) << ", " << vfName(lfs) << ");"; break;
                case J::Lower::Mr32: lo << "J::mr32<" << L << ">(" << vfName(lft) << ", " << vfName(lfs) << ");"; break;
                case J::Lower::Lqi: lo << "J::lqi<" << L << ">(context, " << vfName(lft) << ", " << viName(vis) << ", " << (vis == 0 ? "true" : "false") << ");"; break;
                case J::Lower::Sqi: lo << "J::sqi<" << L << ">(vu, context, " << cycle(k) << ", " << vfName(lfs) << ", " << viName(vit) << ", " << (vit == 0 ? "true" : "false") << ");"; break;
                case J::Lower::Lqd: lo << "J::lqd<" << L << ">(context, " << vfName(lft) << ", " << viName(vis) << ", " << (vis == 0 ? "true" : "false") << ");"; break;
                case J::Lower::Sqd: lo << "J::sqd<" << L << ">(vu, context, " << cycle(k) << ", " << vfName(lfs) << ", " << viName(vit) << ", " << (vit == 0 ? "true" : "false") << ");"; break;
                case J::Lower::Div: lo << "J::div<" << L << ">(vu, " << vfName(lfs) << ", " << vfName(lft) << ");"; break;
                case J::Lower::Sqrt: lo << "J::sqrt<" << L << ">(vu, " << vfName(lft) << ");"; break;
                case J::Lower::Rsqrt: lo << "J::rsqrt<" << L << ">(vu, " << vfName(lfs) << ", " << vfName(lft) << ");"; break;
                case J::Lower::Mtir: lo << "J::mtir<" << L << ">(" << viT << ", " << vfName(lfs) << ");"; break;
                case J::Lower::Mfir: lo << "J::mfir<" << L << ">(" << vfName(lft) << ", " << viName(vis) << ");"; break;
                case J::Lower::Ilwr: lo << "J::ilwr<" << L << ">(context, " << viT << ", " << viName(vis) << ");"; break;
                case J::Lower::Iswr: lo << "J::iswr<" << L << ">(vu, context, " << cycle(k) << ", " << viName(vit) << ", " << viName(vis) << ");"; break;
                case J::Lower::Mfp: lo << "J::mfp<" << L << ">(vu, " << vfName(lft) << ");"; break;
                case J::Lower::Xtop: lo << viT << " = static_cast<int32_t>(s.top & 0x3FFu);"; break;
                case J::Lower::Xitop: lo << viT << " = static_cast<int32_t>(s.itop & 0x3FFu);"; break;
                case J::Lower::FlagRead:
                {
                    const uint32_t op = lower >> 25;
                    const std::string flagTarget = (op == 0x10u || op == 0x12u || op == 0x13u) ? viName(1) : viT;
                    lo << "J::flagRead<" << L << ">(vu, " << cycle(k) << ", " << flagTarget << ", " << viName(vis) << ");";
                    break;
                }
                case J::Lower::Fcset: lo << "J::fcset<" << L << ">(vu);"; break;
                case J::Lower::Fsset: lo << "J::fsset<" << L << ">(vu);"; break;
                case J::Lower::Rnext: lo << "J::rnext<" << L << ">(vu, " << vfName(lft) << ");"; break;
                case J::Lower::Rget: lo << "J::rget<" << L << ">(vu, " << vfName(lft) << ");"; break;
                case J::Lower::Rinit: lo << "J::rinit<" << L << ">(vu, " << vfName(lfs) << ");"; break;
                case J::Lower::Rxor: lo << "J::rxor<" << L << ">(vu, " << vfName(lfs) << ");"; break;
                case J::Lower::Efu: lo << "J::efu<" << L << ">(vu, " << vfName(lfs) << ");"; break;
                case J::Lower::Branch:
                {
                    // Evaluated where the branch issues; the delay slot follows, then the pc moves.
                    const uint32_t opcode = lower >> 25;
                    const int32_t imm = J::imm11(lower);
                    const std::string mask = "0x" + hex(pcMask, 4) + "u";
                    const std::string targetStatic = "((" + std::to_string(pc + 8u) + "u + " + std::to_string(imm * 8) + "u) & " + mask + ")";
                    const auto readCond = [&](uint8_t reg) -> std::string
                    {
                        if (reg == 0u)
                            return "0";
                        if (k == 0u)
                            return "J::readBranchVi(vu, " + std::to_string(reg) + ")";
                        if (static_cast<int>(reg) == branchOldReg)
                            return "viBranchOld";
                        return viName(reg);
                    };
                    const auto i16 = [](const std::string &v) { return "static_cast<int16_t>(" + v + ")"; };
                    switch (opcode)
                    {
                    case 0x20u: lo << "taken = true; target = " << targetStatic << ";"; break;
                    case 0x21u:
                        lo << "taken = true; target = " << targetStatic << ";";
                        if (vit)
                            lo << " " << viName(vit) << " = " << ((pc + 16u) / 8u) << ";";
                        break;
                    case 0x24u: lo << "taken = true; target = (static_cast<uint32_t>(static_cast<uint16_t>(" << viName(vis) << ")) * 8u) & " << mask << ";"; break;
                    case 0x25u:
                        lo << "taken = true; target = (static_cast<uint32_t>(static_cast<uint16_t>(" << viName(vis) << ")) * 8u) & " << mask << ";";
                        if (vit)
                            lo << " " << viName(vit) << " = " << ((pc + 16u) / 8u) << ";";
                        break;
                    case 0x28u: lo << "taken = " << i16(readCond(vis)) << " == " << i16(readCond(vit)) << "; target = " << targetStatic << ";"; break;
                    case 0x29u: lo << "taken = " << i16(readCond(vis)) << " != " << i16(readCond(vit)) << "; target = " << targetStatic << ";"; break;
                    case 0x2Cu: lo << "taken = " << i16(readCond(vis)) << " < 0; target = " << targetStatic << ";"; break;
                    case 0x2Du: lo << "taken = " << i16(readCond(vis)) << " > 0; target = " << targetStatic << ";"; break;
                    case 0x2Eu: lo << "taken = " << i16(readCond(vis)) << " <= 0; target = " << targetStatic << ";"; break;
                    case 0x2Fu: lo << "taken = " << i16(readCond(vis)) << " >= 0; target = " << targetStatic << ";"; break;
                    default: return false;
                    }
                    break;
                }
                default:
                    break;
                }
            }

            const uint8_t shadow = pair.upperVfShadowReg;
            if (pair.iBit)
            {
                body << "        " << up.str() << "\n";
                body << "        J::setI(vu, " << hex32(lower) << ");\n";
            }
            else if (shadow != 0u)
            {
                body << "        {\n            const __m128 shadowOld = " << vfName(shadow) << ";\n"
                     << "            " << up.str() << "\n"
                     << "            const __m128 shadowUpper = " << vfName(shadow) << ";\n"
                     << "            " << vfName(shadow) << " = shadowOld;\n"
                     << "            " << lo.str() << "\n"
                     << "            " << vfName(shadow) << " = shadowUpper;\n        }\n";
            }
            else
            {
                if (!up.str().empty())
                    body << "        " << up.str() << "\n";
                if (!lo.str().empty())
                    body << "        " << lo.str() << "\n";
            }
            const bool writesVf0 = (pi.upper.kind != J::Upper::Nop && pi.upper.kind != J::Upper::Clip &&
                                    pi.upper.kind != J::Upper::FmacAcc && pi.upper.target == 0u) ||
                                   (!pair.iBit && ps2_vu1_detail::FT(lower) == 0u &&
                                    (pi.lower == J::Lower::Lq || pi.lower == J::Lower::Move || pi.lower == J::Lower::Mr32 ||
                                     pi.lower == J::Lower::Lqi || pi.lower == J::Lower::Lqd || pi.lower == J::Lower::Mfir ||
                                     pi.lower == J::Lower::Mfp || pi.lower == J::Lower::Rnext || pi.lower == J::Lower::Rget));
            if (writesVf0)
                body << "        vf0 = J::vf0();\n";
        }

        // Pair bodies for the pairs the interpreter runs.
        if (needsBodies)
            for (uint32_t k = 0; k < count; ++k)
                if (info[k].fallback)
                {
                    const uint8_t hints = static_cast<uint8_t>(plan.knownNormal[k] | (unobserved[first + k] ? Vu::kHintFlagsUnobserved : 0u));
                    out << "    Vu::StepResult jitBody" << (firstPair + first + k) << "(Vu &vu, const Vu::RunContext &context)\n"
                        << "    {\n        return vu.stepPairBodyEntry<" << (plan.deadFlags[k] ? "true" : "false") << ", "
                        << static_cast<unsigned>(hints) << "u, Vu::BlockInterior>(kPair" << (firstPair + first + k)
                        << ", context);\n    }\n";
                }

        out << "    Vu::StepResult jit" << id << "(Vu &vu, const Vu::RunContext &context)\n    {\n"
            << "        using J = Vu1Jit;\n"
            << "        uint64_t base = 0u;\n"
            << (fast ? "        if (!J::fastEnter<kBlockPlan" : "        if (!vu.blockReady<kBlockPlan") << id
            << (fast ? ">(vu, context, base))\n" : ">(context, base))\n")
            << "            return pair" << id << "(vu, context);\n"
            << "        VU1State &s = J::state(vu);\n"
            << "        [[maybe_unused]] __m128 vf0 = J::vf0();\n"
            << "        [[maybe_unused]] int32_t vi0 = 0, viDiscard = 0;\n"
            << "        [[maybe_unused]] bool taken = false;\n"
            << "        [[maybe_unused]] uint32_t target = 0u;\n";
        if (usesSticky)
            out << "        __m128i stickySign = _mm_setzero_si128(), stickyZero = _mm_setzero_si128();\n";
        for (uint32_t reg = 1; reg < 32u; ++reg)
            if (vfUsed[reg])
                out << "        __m128 " << vfName(reg) << " = J::load(s.vf[" << reg << "]);\n";
        out << "        [[maybe_unused]] __m128 acc = " << (accUsed ? "J::load(s.acc)" : "_mm_setzero_ps()") << ";\n";
        for (uint32_t reg = 1; reg < 16u; ++reg)
            if (viUsed[reg])
                out << "        int32_t " << viName(reg) << " = s.vi[" << reg << "];\n";
        out << body.str();
        body.str("");
        foldSticky("        ");
        storeAll("        ");
        out << body.str();
        const uint32_t lastPc = (firstPair + end - 1u) * 8u;
        if (branchEnd)
            out << "        s.pc = taken ? target : J::wrapPc(context, " << (lastPc + 8u) << "u);\n";
        else
            out << "        s.pc = J::wrapPc(context, " << (lastPc + 8u) << "u);\n";
        if (branchLast) // stepPairBody after a branch pair: pending, its delay already counted down
            out << "        if (taken)\n        {\n            s.branchPending = true;\n            s.branchTarget = target;\n"
                << "            s.branchDelay = 0;\n        }\n";
        if (eBitLast)
            out << "        s.ebit = true;\n";
        if (fast)
            out << "        J::fastLeave<kBlockPlan" << id << ">(vu, base);\n";
        else
            out << "        J::leave<kBlockPlan" << id << ", kBlockPairs" << id << ">(vu, base);\n";
        if (lastViReg >= 0)
            out << "        J::recordViForBranch(vu, " << lastViReg << ", viLastOld);\n";
        out << "        return Vu::StepResult::" << (eBitEnd ? "Ended" : "Continue") << ";\n    }\n";
        return true;
    }

    // Straight-line runs [first, end) of the fragment: a block ends after a branch's delay slot and after the delay
    // slot of an E bit, starts at every static branch target, and leaves out XGKICK and reserved pairs (they run on
    // their own, as does any pair a jump lands on in the middle of a block).
    std::vector<std::pair<uint32_t, uint32_t>> findBlocks(const std::vector<VU1Interpreter::DecodedInstructionPair> &pairs,
                                                         uint32_t firstPair, const std::set<uint32_t> &entryPairs = {})
    {
        using Vu = VU1Interpreter;
        const uint32_t count = static_cast<uint32_t>(pairs.size());
        std::vector<bool> starts(count + 2u, false), alone(count, false);
        starts[0] = true;
        for (uint32_t entry : entryPairs) // where microprograms start (VCALLMS targets)
            if (entry >= firstPair && entry < firstPair + count)
                starts[entry - firstPair] = true;
        for (uint32_t index = 0; index < count; ++index)
        {
            const Vu::DecodedInstructionPair &pair = pairs[index];
            if (pair.lowerUsage.pipeline == Vu::PipelineBranch)
            {
                const uint32_t opcode = pair.lower >> 25;
                const bool staticTarget = opcode == 0x20u || opcode == 0x21u || opcode == 0x28u || opcode == 0x29u ||
                                          (opcode >= 0x2Cu && opcode <= 0x2Fu);
                if (staticTarget)
                {
                    int32_t offset = static_cast<int32_t>(pair.lower & 0x7FFu);
                    if (offset & 0x400)
                        offset -= 0x800;
                    const int64_t target = static_cast<int64_t>(firstPair) + index + 1 + offset;
                    if (target >= firstPair && target < static_cast<int64_t>(firstPair) + count)
                        starts[static_cast<uint32_t>(target - firstPair)] = true;
                }
                starts[index + 2u] = true;
            }
            if (pair.eBit)
                starts[index + 2u] = true;
            if (pair.lowerUsage.pipeline == Vu::PipelineXgkick || pair.lowerUsage.reserved || pair.upperUsage.reserved)
            {
                alone[index] = true;
                starts[index] = true;
                starts[index + 1u] = true;
            }
        }
        std::vector<std::pair<uint32_t, uint32_t>> blocks;
        for (uint32_t first = 0; first < count;)
        {
            uint32_t end = first + 1u;
            while (end < count && !starts[end] && end - first < Vu::kMaxBlockPairs)
                ++end;
            if (!alone[first] && Vu::planBlock(pairs.data() + first, end - first).valid)
                blocks.push_back({first, end});
            first = end;
        }
        return blocks;
    }

    void writeFragment(const std::filesystem::path &path, const std::string &prefix, const std::string &elfName,
                       const Fragment &fragment, const uint8_t *code, bool stickyUnobserved,
                       VU1Interpreter::Unit unit = VU1Interpreter::Unit::VU1, const std::set<uint32_t> &entryPairs = {})
    {
        const bool vu0 = unit == VU1Interpreter::Unit::VU0;
        const std::string unitName = vu0 ? "Vu::Unit::VU0" : "Vu::Unit::VU1";
        const uint32_t pcMask = vu0 ? 0x0FFFu : 0x3FFFu;
        // One small function per pair (inlining stepPair() into one function per fragment takes GCC minutes), and
        // one per straight-line run (VU1Interpreter::runBlock), which replaces the first pair's entry.
        // A namespace per fragment keeps the names apart when a build merges sources (unity builds).
        using Vu = VU1Interpreter;
        const std::string tag = hex(fragment.firstPair, 3) + "_" + hex(fragment.hash, 16);
        std::vector<Vu::DecodedInstructionPair> decoded;
        std::vector<std::pair<uint32_t, uint32_t>> words;
        for (uint32_t pair = 0; pair < fragment.pairCount; ++pair)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, code + pair * 8u, 4);
            std::memcpy(&upper, code + pair * 8u + 4u, 4);
            words.push_back({lower, upper});
            decoded.push_back(Vu::decodePair(unit, lower, upper));
        }
        const auto blocks = findBlocks(decoded, fragment.firstPair, entryPairs);
        // Flags no reader sees are skipped only if nothing in the program can read the sticky bits they add to.
        // Otherwise the recompiled blocks keep only their sticky bits (Vu1Jit::fmacSticky).
        std::vector<bool> unobserved = findUnobservedFlagWrites(decoded, fragment.firstPair);
        for (const auto &[first, end] : blocks) // writes a later write in the block hides (BlockPlan::deadFlags)
        {
            const Vu::BlockPlan plan = Vu::planBlock(decoded.data() + first, end - first);
            for (uint32_t pair = first; pair < end; ++pair)
                if (plan.deadFlags[pair - first])
                    unobserved[pair] = true;
        }
        std::vector<bool> stickyOnly(fragment.pairCount, false);
        if (vu0)
            unobserved.assign(fragment.pairCount, false);
        else if (!stickyUnobserved)
            stickyOnly.swap(unobserved);
        const auto hints = [&](uint32_t pair) { return unobserved[pair] ? std::string(" | Vu::kHintFlagsUnobserved") : std::string(); };

        const auto decodeCall = [&](uint32_t pair)
        { return "Vu::decodePair(" + unitName + ", 0x" + hex(words[pair].first, 8) + "u, 0x" + hex(words[pair].second, 8) + "u)"; };
        std::vector<bool> blockStart(fragment.pairCount, false), jitStart(fragment.pairCount, false);
        for (const auto &[first, end] : blocks)
            blockStart[first] = true;
        const bool jitEnabled = std::getenv("PS2X_VU_AOT_NO_JIT") == nullptr;
        const bool fast = fastVu1() && unit == VU1Interpreter::Unit::VU1;
        // Recompiled blocks first: their pairs only run alone when a block cannot start (or a jump lands inside
        // one), through the interpreter; compiling them pair by pair takes most of the build.
        std::vector<std::string> jitText(fragment.pairCount);
        if (jitEnabled)
            for (const auto &[first, end] : blocks)
            {
                std::ostringstream jit;
                const Vu::BlockPlan plan = Vu::planBlock(decoded.data() + first, end - first);
                if (!writeJitBlock(jit, decoded, first, end, fragment.firstPair, unobserved, stickyOnly, plan, pcMask, fast))
                    continue;
                jitText[first] = jit.str();
                jitStart[first] = true;
            }
        std::vector<std::string> pairText(fragment.pairCount), blockText(fragment.pairCount);
        for (uint32_t pair = 0; pair < fragment.pairCount; ++pair)
        {
            const uint32_t id = fragment.firstPair + pair;
            std::ostringstream out;
            out << "\n    constexpr Vu::DecodedInstructionPair kPair" << id << " = " << decodeCall(pair) << ";\n"
                << "    Vu::StepResult pair" << id << "(Vu &vu, const Vu::RunContext &context)\n"
                << "    {\n";
            // Under the JIT every pair alone goes through the interpreter: an inlined stepPair() for each pair outside
            // the recompiled blocks took most of the build (MSVC: about 15 minutes for 64 of them).
            if (jitEnabled)
                out << "#if PS2X_VU1_JIT\n        return vu.stepPairInterpreted(kPair" << id << ", context);\n#else\n";
            out << "        return vu.stepPair" << (unobserved[pair] ? "<Vu::kHintFlagsUnobserved>" : "") << "(kPair" << id
                << ", context);\n";
            if (jitEnabled)
                out << "#endif\n";
            out << "    }\n";
            pairText[pair] = out.str();
        }
        std::vector<bool> insideBlock(fragment.pairCount, false); // a pair after a block's first one
        for (const auto &[first, end] : blocks)
        {
            for (uint32_t pair = first + 1u; pair < end; ++pair)
                insideBlock[pair] = true;
            const uint32_t id = fragment.firstPair + first;
            std::ostringstream out;
            out << "\n    constexpr Vu::DecodedInstructionPair kBlockPairs" << id << "[] = {";
            for (uint32_t pair = first; pair < end; ++pair)
                out << (pair == first ? "" : ",") << ((pair - first) % 6u == 0u ? "\n        " : " ") << "kPair"
                    << (fragment.firstPair + pair);
            out << "};\n"
                << "    constexpr Vu::BlockPlan kBlockPlan" << id << " = Vu::planBlock(kBlockPairs" << id << ", " << (end - first)
                << "u);\n";
            if (jitStart[first])
                out << "#if PS2X_VU1_JIT\n" << jitText[first] << "#endif\n";
            // Without the JIT the block runs its pairs inlined in turn. Under the JIT a block it does not take runs
            // its pairs alone (inlining them took minutes of the build for a few short blocks).
            if (jitEnabled)
                out << "#if !PS2X_VU1_JIT\n";
            for (uint32_t pair = first; pair < end; ++pair)
                out << "    Vu::StepResult body" << (fragment.firstPair + pair) << "(Vu &vu, const Vu::RunContext &context)\n"
                    << "    {\n"
                    << "        return vu.stepPairBodyEntry<kBlockPlan" << id << ".deadFlags[" << (pair - first) << "], kBlockPlan"
                    << id << ".knownNormal[" << (pair - first) << "]" << hints(pair) << ", Vu::"
                    << (pair - first + 2u < end - first ? "BlockInterior" : "BlockTail") << ">(kPair"
                    << (fragment.firstPair + pair) << ", context);\n"
                    << "    }\n";
            out << "    constexpr Vu::CompiledEntry kBlockBodies" << id << "[] = {";
            for (uint32_t pair = first; pair < end; ++pair)
                out << (pair == first ? "" : ",") << ((pair - first) % 8u == 0u ? "\n        " : " ") << "&body"
                    << (fragment.firstPair + pair);
            out << "};\n"
                << "    Vu::StepResult block" << id << "(Vu &vu, const Vu::RunContext &context)\n"
                << "    {\n"
                << "        return vu.runBlock<kBlockPlan" << id << ", kBlockBodies" << id << ", kBlockPairs" << id
                << ">(context, &pair" << id << ");\n"
                << "    }\n";
            if (jitEnabled)
                out << "#endif\n";
            blockText[first] = out.str();
        }

        // A long fragment (a chain of MPGs) is split into files of about kPartPairs pairs, between blocks: one
        // translation unit of thousands of pairs takes the compiler too long and too much memory. The functions
        // have external linkage in the fragment's namespace; the last part declares the others' and holds the table.
        constexpr uint32_t kPartPairs = 256u;
        std::vector<uint32_t> partStarts{0u};
        for (uint32_t pair = 1u; pair < fragment.pairCount; ++pair)
            if (pair - partStarts.back() >= kPartPairs && !insideBlock[pair] && fragment.pairCount - pair >= kPartPairs / 4u)
                partStarts.push_back(pair);
        for (size_t part = 0; part < partStarts.size(); ++part)
        {
            const uint32_t partBegin = partStarts[part];
            const uint32_t partEnd = part + 1u < partStarts.size() ? partStarts[part + 1u] : fragment.pairCount;
            const bool last = part + 1u == partStarts.size();
            std::filesystem::path partPath = path;
            if (partStarts.size() > 1u)
                partPath.replace_filename(path.stem().string() + "_p" + std::to_string(part) + path.extension().string());
            std::ofstream out(partPath);
            out << "// Generated by ps2_vu_aot from " << elfName << " (file offset 0x" << hex(fragment.fileOffset, 0);
            if (fragment.chunks > 1u)
                out << ", " << fragment.chunks << " MPG commands";
            out << "): VU1 microcode pairs " << fragment.firstPair << ".." << (fragment.firstPair + fragment.pairCount - 1)
                << ", " << blocks.size() << " blocks";
            if (partStarts.size() > 1u)
                out << "; part " << (part + 1u) << " of " << partStarts.size() << ": pairs " << (fragment.firstPair + partBegin)
                    << ".." << (fragment.firstPair + partEnd - 1u);
            out << ".\n"
                << "#include \"runtime/ps2_vu1_jit.h\"\n\n"
                << "#if PS2X_VU1_JIT\n#define JIT_ENTRY(n) &jit##n\n#define BLOCK_ENTRY(n) &pair##n\n#else\n"
                   "#define JIT_ENTRY(n) &block##n\n#define BLOCK_ENTRY(n) &block##n\n#endif\n\n"
                << "namespace " << prefix << tag << "\n{\n"
                << "    using Vu = VU1Interpreter;\n";
            if ((relaxedFloat() || fast) && part == 0u)
                out << "#if PS2X_VU1_JIT\n    [[maybe_unused]] const bool kRelaxedFloat = Vu1Jit::enableRelaxedFloat();\n#endif\n";
            for (uint32_t pair = partBegin; pair < partEnd; ++pair)
                out << pairText[pair];
            for (uint32_t pair = partBegin; pair < partEnd; ++pair)
                out << blockText[pair];
            if (!last)
            {
                out << "}\n";
                continue;
            }
            if (partBegin > 0u)
                out << "\n";
            for (uint32_t pair = 0; pair < partBegin; ++pair) // the other parts' entries
            {
                const std::string id = std::to_string(fragment.firstPair + pair);
                const std::string signature = "(Vu &vu, const Vu::RunContext &context);\n";
                if (jitStart[pair])
                    out << "#if PS2X_VU1_JIT\n    Vu::StepResult jit" << id << signature << "#else\n    Vu::StepResult block" << id
                        << signature << "#endif\n";
                else if (blockStart[pair] && jitEnabled)
                    out << "#if PS2X_VU1_JIT\n    Vu::StepResult pair" << id << signature << "#else\n    Vu::StepResult block" << id
                        << signature << "#endif\n";
                else
                    out << "    Vu::StepResult " << (blockStart[pair] ? "block" : "pair") << id << signature;
            }
            out << "\n    constexpr Vu::CompiledEntry kPairs[] = {";
            for (uint32_t pair = 0; pair < fragment.pairCount; ++pair)
                out << (pair % 8u == 0u ? "\n        " : " ")
                    << (jitStart[pair] ? "JIT_ENTRY(" : !blockStart[pair] ? "&pair" : jitEnabled ? "BLOCK_ENTRY(" : "&block")
                    << (fragment.firstPair + pair) << (jitStart[pair] || (blockStart[pair] && jitEnabled) ? ")" : "") << ",";
            out << "\n    };\n\n"
                << "    const bool registered = (Vu::registerCompiledFragment({" << fragment.firstPair << "u, "
                << fragment.pairCount << "u, 0x" << hex(fragment.hash, 16) << "ull, kPairs, " << unitName << "}), true);\n"
                << "}\n";
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <elf> <output-dir> [file-prefix]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path elfPath = argv[1];
    const std::filesystem::path outDir = argv[2];
    const std::string prefix = argc > 3 ? argv[3] : "vu1_aot_";

    std::ifstream in(elfPath, std::ios::binary);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (file.size() < 0x34 || readU32(file, 0) != 0x464C457Fu)
    {
        std::fprintf(stderr, "%s is not an ELF file\n", argv[1]);
        return 1;
    }

    const uint32_t phoff = readU32(file, 0x1C);
    const uint16_t phentsize = readU16(file, 0x2A);
    const uint16_t phnum = readU16(file, 0x2C);
    // Every MPG whose pairs decode as VU1 microcode, by file position of its VIFcode.
    std::map<size_t, Fragment> uploads;
    for (uint32_t index = 0; index < phnum; ++index)
    {
        const size_t header = phoff + static_cast<size_t>(index) * phentsize;
        if (header + 32u > file.size() || readU32(file, header) != 1u) // PT_LOAD
            continue;
        const uint32_t offset = readU32(file, header + 4u);
        const uint32_t fileSize = readU32(file, header + 16u);
        const size_t end = std::min<size_t>(file.size(), static_cast<size_t>(offset) + fileSize);
        for (size_t position = offset; position + 8u <= end; position += 4u)
        {
            const uint32_t vifcode = readU32(file, position);
            if (((vifcode >> 24) & 0x7Fu) != 0x4Au) // MPG (bit 7 is the interrupt flag)
                continue;
            const uint32_t pairCount = ((vifcode >> 16) & 0xFFu) == 0u ? 256u : ((vifcode >> 16) & 0xFFu);
            const uint32_t firstPair = vifcode & 0xFFFFu;
            const size_t code = position + 4u;
            if ((code & 7u) != 0u || firstPair + pairCount > kVu1CodePairs || code + pairCount * 8u > end)
                continue;
            if (!decodesAsMicrocode(file.data() + code, pairCount))
                continue;
            uploads[position] = {static_cast<uint32_t>(code), firstPair, pairCount,
                                 VU1Interpreter::codeHash(file.data() + code, pairCount * 8u), file.data() + code};
        }
    }

    // A program longer than one MPG (256 pairs) comes as consecutive MPGs of one packet, NOP-padded so that each
    // one's code stays 8-byte aligned. Each whole chain becomes one fragment: compiled chunk by chunk, a loop across
    // a chunk boundary would stay interpreted. Chunks of a chain are not compiled alone (the game uploads the
    // chain); MPGs outside chains are.
    std::map<size_t, size_t> nextUpload;
    for (const auto &[position, upload] : uploads)
    {
        size_t next = upload.fileOffset + upload.pairCount * 8u;
        for (int nops = 0; nops < 4 && next + 4u <= file.size(); ++nops)
        {
            const auto found = uploads.find(next);
            if (found != uploads.end() && found->second.firstPair == upload.firstPair + upload.pairCount)
            {
                nextUpload[position] = next;
                break;
            }
            if (((readU32(file, next) >> 24) & 0x7Fu) != 0u) // NOP
                break;
            next += 4u;
        }
    }
    std::set<size_t> chained;
    for (const auto &[position, next] : nextUpload)
    {
        chained.insert(position);
        chained.insert(next);
    }
    std::set<size_t> chainStarts = chained;
    for (const auto &[position, next] : nextUpload)
        chainStarts.erase(next);
    std::vector<Fragment> fragments;
    std::deque<std::vector<uint8_t>> chainCode;
    std::set<std::tuple<uint32_t, uint32_t, uint64_t>> seen;
    const auto addFragment = [&](const Fragment &fragment)
    {
        if (seen.insert({fragment.firstPair, fragment.pairCount, fragment.hash}).second)
            fragments.push_back(fragment);
    };
    for (const size_t start : chainStarts)
    {
        Fragment chain = uploads[start];
        std::vector<uint8_t> &code = chainCode.emplace_back(chain.code, chain.code + chain.pairCount * 8u);
        for (auto next = nextUpload.find(start); next != nextUpload.end(); next = nextUpload.find(next->second))
        {
            const Fragment &chunk = uploads[next->second];
            code.insert(code.end(), chunk.code, chunk.code + chunk.pairCount * 8u);
            chain.pairCount += chunk.pairCount;
            ++chain.chunks;
        }
        chain.code = code.data();
        chain.hash = VU1Interpreter::codeHash(chain.code, chain.pairCount * 8u);
        addFragment(chain);
    }
    for (const auto &[position, upload] : uploads)
        if (!chained.count(position))
            addFragment(upload);

    // VU0 microprograms come by MPG too (through VIF0): fragments that fit VU0 code memory and decode as VU0
    // microcode (no EFU, XGKICK or XTOP) also get a VU0 build.
    const std::string vu0Prefix = "vu0_aot_";
    std::filesystem::create_directories(outDir);
    for (const auto &entry : std::filesystem::directory_iterator(outDir))
    {
        const std::string name = entry.path().filename().string();
        if ((name.rfind(prefix, 0) == 0 || name.rfind(vu0Prefix, 0) == 0) && entry.path().extension() == ".cpp")
            std::filesystem::remove(entry.path());
    }
    // Sticky status bits gather every FMAC write: if any microcode can read them, every write stays computed.
    bool stickyUnobserved = true;
    for (const Fragment &fragment : fragments)
        for (uint32_t pair = 0; pair < fragment.pairCount && stickyUnobserved; ++pair)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, fragment.code + pair * 8u, 4);
            std::memcpy(&upper, fragment.code + pair * 8u + 4u, 4);
            if (readsStickyStatus(VU1Interpreter::decodePair(VU1Interpreter::Unit::VU1, lower, upper)))
                stickyUnobserved = false;
        }
    uint32_t totalPairs = 0;
    for (const Fragment &fragment : fragments)
    {
        const std::filesystem::path path =
            outDir / (prefix + hex(fragment.firstPair, 3) + "_" + hex(fragment.hash, 16) + ".cpp");
        writeFragment(path, prefix, elfPath.filename().string(), fragment, fragment.code,
                      stickyUnobserved);
        totalPairs += fragment.pairCount;
    }
    constexpr uint32_t kVu0CodePairs = 0x1000u / 8u;
    size_t vu0Fragments = 0;
    // VU0 microprograms start where the EE's VCALLMS points (COP2, funct 0x38, immediate in bits 6..20): blocks
    // start there too, so a call does not begin with interpreted pairs.
    std::set<uint32_t> vu0Entries;
    for (size_t offset = 0; offset + 4u <= file.size(); offset += 4u)
    {
        const uint32_t word = readU32(file, offset);
        if ((word >> 26) == 0x12u && ((word >> 25) & 1u) != 0u && (word & 0x3Fu) == 0x38u && ((word >> 21) & 0xFu) == 0u)
            vu0Entries.insert((word >> 6) & 0x7FFFu);
    }
    for (const Fragment &fragment : fragments)
    {
        if (fragment.firstPair + fragment.pairCount > kVu0CodePairs)
            continue;
        bool vu0Code = true;
        for (uint32_t pair = 0; pair < fragment.pairCount && vu0Code; ++pair)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, fragment.code + pair * 8u, 4);
            std::memcpy(&upper, fragment.code + pair * 8u + 4u, 4);
            const auto decoded = VU1Interpreter::decodePair(VU1Interpreter::Unit::VU0, lower, upper);
            vu0Code = !decoded.upperUsage.reserved && !decoded.lowerUsage.reserved;
        }
        if (!vu0Code)
            continue;
        // The EE reads VU0's flags (CFC2), sticky bits included: every VU0 flag write is computed.
        const std::filesystem::path path =
            outDir / (vu0Prefix + hex(fragment.firstPair, 3) + "_" + hex(fragment.hash, 16) + ".cpp");
        writeFragment(path, vu0Prefix, elfPath.filename().string(), fragment, fragment.code, false,
                      VU1Interpreter::Unit::VU0, vu0Entries);
        ++vu0Fragments;
    }
    std::printf("ps2_vu_aot: %zu microcode fragments, %u instruction pairs (%zu also as VU0 microcode) -> %s\n",
                fragments.size(), totalPairs, vu0Fragments, outDir.string().c_str());
    return 0;
}
