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

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    constexpr uint32_t kVu1CodePairs = 0x4000u / 8u;

    struct Fragment
    {
        uint32_t fileOffset = 0; // of the code, after the VIFcode
        uint32_t firstPair = 0;
        uint32_t pairCount = 0;
        uint64_t hash = 0;
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

    // Straight-line runs [first, end) of the fragment: a block ends after a branch's delay slot and after the delay
    // slot of an E bit, starts at every static branch target, and leaves out XGKICK and reserved pairs (they run on
    // their own, as does any pair a jump lands on in the middle of a block).
    std::vector<std::pair<uint32_t, uint32_t>> findBlocks(const std::vector<VU1Interpreter::DecodedInstructionPair> &pairs,
                                                         uint32_t firstPair)
    {
        using Vu = VU1Interpreter;
        const uint32_t count = static_cast<uint32_t>(pairs.size());
        std::vector<bool> starts(count + 2u, false), alone(count, false);
        starts[0] = true;
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
            if (!alone[first] && end - first >= 2u && Vu::planBlock(pairs.data() + first, end - first).valid)
                blocks.push_back({first, end});
            first = end;
        }
        return blocks;
    }

    void writeFragment(const std::filesystem::path &path, const std::string &prefix, const std::string &elfName,
                       const Fragment &fragment, const uint8_t *code, bool stickyUnobserved)
    {
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
            decoded.push_back(Vu::decodePair(Vu::Unit::VU1, lower, upper));
        }
        const auto blocks = findBlocks(decoded, fragment.firstPair);
        // Flags no reader sees are skipped only if nothing in the program can read the sticky bits they add to.
        std::vector<bool> unobserved(fragment.pairCount, false);
        if (stickyUnobserved)
        {
            unobserved = findUnobservedFlagWrites(decoded, fragment.firstPair);
            for (const auto &[first, end] : blocks) // writes a later write in the block hides (BlockPlan::deadFlags)
            {
                const Vu::BlockPlan plan = Vu::planBlock(decoded.data() + first, end - first);
                for (uint32_t pair = first; pair < end; ++pair)
                    if (plan.deadFlags[pair - first])
                        unobserved[pair] = true;
            }
        }
        const auto hints = [&](uint32_t pair) { return unobserved[pair] ? std::string(" | Vu::kHintFlagsUnobserved") : std::string(); };

        std::ofstream out(path);
        out << "// Generated by ps2_vu_aot from " << elfName << " (file offset 0x" << hex(fragment.fileOffset, 0)
            << "): VU1 microcode pairs " << fragment.firstPair << ".." << (fragment.firstPair + fragment.pairCount - 1)
            << ", " << blocks.size() << " blocks.\n"
            << "#include \"runtime/ps2_vu1_inline.h\"\n\n"
            << "namespace " << prefix << tag << "\n{\n"
            << "    using Vu = VU1Interpreter;\n";
        const auto decodeCall = [&](uint32_t pair)
        { return "Vu::decodePair(Vu::Unit::VU1, 0x" + hex(words[pair].first, 8) + "u, 0x" + hex(words[pair].second, 8) + "u)"; };
        std::vector<bool> blockStart(fragment.pairCount, false);
        for (const auto &[first, end] : blocks)
            blockStart[first] = true;
        for (uint32_t pair = 0; pair < fragment.pairCount; ++pair)
        {
            const uint32_t id = fragment.firstPair + pair;
            out << "\n    constexpr Vu::DecodedInstructionPair kPair" << id << " = " << decodeCall(pair) << ";\n"
                << "    Vu::StepResult pair" << id << "(Vu &vu, const Vu::RunContext &context)\n"
                << "    {\n"
                << "        return vu.stepPair" << (unobserved[pair] ? "<Vu::kHintFlagsUnobserved>" : "") << "(kPair" << id
                << ", context);\n"
                << "    }\n";
        }
        for (const auto &[first, end] : blocks)
        {
            const uint32_t id = fragment.firstPair + first;
            out << "\n    constexpr Vu::DecodedInstructionPair kBlockPairs" << id << "[] = {";
            for (uint32_t pair = first; pair < end; ++pair)
                out << (pair == first ? "" : ",") << ((pair - first) % 6u == 0u ? "\n        " : " ") << "kPair"
                    << (fragment.firstPair + pair);
            out << "};\n"
                << "    constexpr Vu::BlockPlan kBlockPlan" << id << " = Vu::planBlock(kBlockPairs" << id << ", " << (end - first)
                << "u);\n";
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
        }
        out << "\n    constexpr Vu::CompiledEntry kPairs[] = {";
        for (uint32_t pair = 0; pair < fragment.pairCount; ++pair)
            out << (pair % 8u == 0u ? "\n        " : " ") << (blockStart[pair] ? "&block" : "&pair") << (fragment.firstPair + pair)
                << ",";
        out << "\n    };\n\n"
            << "    const bool registered = (Vu::registerCompiledFragment({" << fragment.firstPair << "u, "
            << fragment.pairCount << "u, 0x" << hex(fragment.hash, 16) << "ull, kPairs}), true);\n"
            << "}\n";
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
    std::vector<Fragment> fragments;
    std::set<std::tuple<uint32_t, uint32_t, uint64_t>> seen;
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
            const uint64_t hash = VU1Interpreter::codeHash(file.data() + code, pairCount * 8u);
            if (!seen.insert({firstPair, pairCount, hash}).second)
                continue;
            fragments.push_back({static_cast<uint32_t>(code), firstPair, pairCount, hash});
        }
    }

    std::filesystem::create_directories(outDir);
    for (const auto &entry : std::filesystem::directory_iterator(outDir))
    {
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && entry.path().extension() == ".cpp")
            std::filesystem::remove(entry.path());
    }
    // Sticky status bits gather every FMAC write: if any microcode can read them, every write stays computed.
    bool stickyUnobserved = true;
    for (const Fragment &fragment : fragments)
        for (uint32_t pair = 0; pair < fragment.pairCount && stickyUnobserved; ++pair)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, file.data() + fragment.fileOffset + pair * 8u, 4);
            std::memcpy(&upper, file.data() + fragment.fileOffset + pair * 8u + 4u, 4);
            if (readsStickyStatus(VU1Interpreter::decodePair(VU1Interpreter::Unit::VU1, lower, upper)))
                stickyUnobserved = false;
        }
    uint32_t totalPairs = 0;
    for (const Fragment &fragment : fragments)
    {
        const std::filesystem::path path =
            outDir / (prefix + hex(fragment.firstPair, 3) + "_" + hex(fragment.hash, 16) + ".cpp");
        writeFragment(path, prefix, elfPath.filename().string(), fragment, file.data() + fragment.fileOffset,
                      stickyUnobserved);
        totalPairs += fragment.pairCount;
    }
    std::printf("ps2_vu_aot: %zu microcode fragments, %u instruction pairs -> %s\n", fragments.size(), totalPairs,
                outDir.string().c_str());
    return 0;
}
