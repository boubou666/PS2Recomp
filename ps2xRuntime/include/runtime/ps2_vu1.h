#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

class GS;
class PS2Memory;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t r;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    uint64_t cycles;
    bool ebit;
    bool haltAfterDelaySlot;
    bool dBitEnabled;
    bool tBitEnabled;
    bool stoppedByD;
    bool stoppedByT;
    uint32_t top;  // VIF TOP visible to XTOP
    uint32_t itop; // VIF ITOP visible to XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

namespace ps2_vu1_detail
{
    enum class FmacKind : uint8_t;
    struct FmacOperation;
}
struct Vu1Jit;

class VU1Interpreter
{
    friend struct Vu1Jit; // the recompiled blocks (ps2_vu1_jit.h) work on the interpreter's state directly
public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Interpreter(Unit unit = Unit::VU1);

    void reset();

    // A budget under 2^20 cycles is a time slice: the run can stop with writes in flight, which it keeps out of the
    // visible state until the next run. runToCompletion makes maxCycles a safety limit instead (no in-flight
    // bookkeeping, which slows every instruction pair).
    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536, bool runToCompletion = false);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    // ---- Pair decoding and stepping, shared by run() and microcode compiled ahead of time ----
    // (ps2_vu_aot generates C++ that calls stepPair() with a constexpr decodePair() of each instruction pair;
    // see ps2_vu1_inline.h.)
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        InstructionUsage lowerUsage{};
        InstructionUsage upperUsage{};
        bool iBit = false;
        bool eBit = false;
        bool mBit = false;
        bool dBit = false;
        bool tBit = false;
        uint8_t upperVfShadowReg = 0;
        uint8_t suppressedLowerVf = 0;
    };

    enum class StepResult : uint8_t
    {
        Continue, // next pair
        Stop,     // cycle budget spent, reserved instruction or stop request
        Ended     // E bit (after its delay slot) or D/T halt
    };

    struct RunContext
    {
        uint8_t *vuData = nullptr;
        uint32_t dataSize = 0;
        uint32_t codeSize = 0;
        GS *gs = nullptr;
        PS2Memory *memory = nullptr;
        uint64_t budgetEnd = 0;
    };

    static constexpr InstructionUsage decodeUpperUsage(uint32_t upper);
    static constexpr InstructionUsage decodeLowerUsage(Unit unit, uint32_t lower);
    static constexpr DecodedInstructionPair decodePair(Unit unit, uint32_t lower, uint32_t upper);
    // One instruction pair at m_state.pc, as one iteration of run()'s loop.
    // kUpperHints: kKnownNormal* and kHintFlagsUnobserved.
    template <uint8_t kUpperHints = 0>
    StepResult stepPair(const DecodedInstructionPair &decoded, const RunContext &context);
    // stepPair(), out of line: for compiled pairs that rarely run alone (inside recompiled blocks).
    StepResult stepPairInterpreted(const DecodedInstructionPair &decoded, const RunContext &context);
    // run()'s loop condition, checked before every pair.
    bool canIssue(const RunContext &context) const
    {
        return m_cycle < context.budgetEnd && !m_stopRequested && m_state.pc + 8u <= context.codeSize;
    }

    // A fragment of microcode compiled ahead of time: pairs [firstPair, firstPair + pairCount) of code memory, valid
    // while those bytes hash to codeHash. pairs[i] runs pair firstPair + i, as stepPair() does.
    using CompiledEntry = StepResult (*)(VU1Interpreter &vu, const RunContext &context);
    struct CompiledFragment
    {
        uint32_t firstPair = 0;
        uint32_t pairCount = 0;
        uint64_t codeHash = 0;
        const CompiledEntry *pairs = nullptr;
        Unit unit = Unit::VU1; // the VU whose code memory it runs from
    };
    static void registerCompiledFragment(const CompiledFragment &fragment);
    // Relaxed float mode of the recompiled VU1 microcode (ps2_vu_aot PS2X_VU_AOT_RELAXED=1, ps2_vu1_jit.h): VU1 runs
    // with the SSE denormals-are-zero and flush-to-zero modes, which give the PS2's zero for denormal operands and
    // results, and FMAC instructions are plain SSE operations clamped to +-FLT_MAX.
    static void setRelaxedFloat(bool enabled);
    static bool relaxedFloat();
    // Runs this instance through the interpreter only (checks of the compiled code against it).
    void setCompiledCodeDisabled(bool disabled) { m_compiledCodeDisabled = disabled; }

    // ---- Blocks: straight-line runs of pairs that ps2_vu_aot compiles as one function ----
    // planBlock() schedules a block at compile time, assuming every value written before the block is ready by the
    // time the block first reads it: each pair's issue cycle after entry. runBlock() checks that assumption on entry
    // (the live-ins: what the block reads before writing it, with the cycle it first reads it) and then issues each
    // pair at its planned cycle instead of working out its operands' readiness; otherwise it runs the first pair as
    // stepPair() does and leaves the rest to the per-pair functions. Results, flags and cycle counts are the
    // interpreter's: the clock goes through the same cycles, and pairs still record when their results land.
    static constexpr uint32_t kMaxBlockPairs = 64u;
    static constexpr uint8_t kKnownNormalFs = 0x1u;
    static constexpr uint8_t kKnownNormalFt = 0x2u;
    static constexpr uint8_t kKnownNormalAcc = 0x4u;
    // With kKnownNormal* in a pair's upper-instruction hints (ps2_vu_aot): no MAC/status flag reader can see the
    // FMAC flags this pair writes (a later write lands first on every path), and none reads the sticky bits: they
    // are not computed. PS2X_VU1_EXACT_FLAGS=1 computes them anyway.
    static constexpr uint8_t kHintFlagsUnobserved = 0x8u;
    struct BlockLiveIn
    {
        enum Kind : uint8_t
        {
            Vf,
            Vi,
            Acc,
            Fdiv,        // a pending DIV/SQRT/RSQRT the block waits for
            EfuResource, // the EFU still busy with an earlier instruction
            EfuEntries   // WAITP before the block's own EFU instructions
        };
        Kind kind = Vf;
        uint8_t reg = 0;
        uint8_t lanes = 0;
        uint16_t issue = 0;
    };
    struct BlockPlan
    {
        uint32_t pairCount = 0;
        std::array<uint16_t, kMaxBlockPairs> issue{};
        std::array<BlockLiveIn, 64> liveIns{};
        uint32_t liveInCount = 0;
        // Pairs whose MAC/status writes no instruction can see: a later write in the block lands before any reader
        // runs. They only add their sticky bits, to the next write that does land (m_deadSticky).
        std::array<bool, kMaxBlockPairs> deadFlags{};
        // Per pair, the upper instruction's operands that an FMAC instruction of the block wrote (kUpperHints*):
        // FMAC results are normalized already, so they are read without normalizeOperand().
        std::array<uint8_t, kMaxBlockPairs> knownNormal{};
        bool usesDtBits = false;
        bool valid = false; // false: not a block runBlock() handles (reserved instruction, XGKICK, too long...)
    };
    static constexpr BlockPlan planBlock(const DecodedInstructionPair *pairs, uint32_t count);
    // kBodies[i] runs pair i as stepPairBody() (a function of its own per pair: inlining a whole block into one
    // function costs GCC gigabytes); first runs the block's first pair as stepPair() does.
    // kPairs: the block's decoded pairs. The block records when their results land (markPairWritesAt) once,
    // as it leaves; its pairs before the last two leave the pc and the branch and halt handling to it too.
    template <const BlockPlan &kPlan, const auto &kBodies, const auto &kPairs>
    StepResult runBlock(const RunContext &context, CompiledEntry first);
    // runBlock()'s entry check: whether the planned schedule holds, from base (the first pair's issue cycle).
    template <const BlockPlan &kPlan>
    bool blockReady(const RunContext &context, uint64_t &base) const;
    enum BlockPosition : uint8_t
    {
        NotInBlock,
        BlockTail,    // one of a block's last two pairs (a branch, its delay slot, an E bit and its delay slot)
        BlockInterior // the others: no branch, halt or E bit to handle
    };
    template <bool kDeadFlags, uint8_t kUpperHints = 0, BlockPosition kPosition = NotInBlock>
    StepResult stepPairBodyEntry(const DecodedInstructionPair &decoded, const RunContext &context);

    static constexpr uint64_t codeHash(const uint8_t *code, uint32_t size)
    {
        uint64_t hash = 0xCBF29CE484222325ull; // FNV-1a
        for (uint32_t index = 0; index < size; ++index)
            hash = (hash ^ code[index]) * 0x100000001B3ull;
        return hash;
    }

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

    // Called with every PATH1 packet an XGKICK completes (recording and replay tooling).
    using PacketObserver = std::function<void(const uint8_t *data, uint32_t size)>;
    void setPacketObserver(PacketObserver observer) { m_packetObserver = std::move(observer); }
    // For callers running code that PS2Memory does not track: drop decoded instructions after changing it.
    void invalidateDecodedCode() { ++m_externalCodeGeneration; }

private:
    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool valid = false;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;
    };

    static constexpr uint32_t kFmacLatency = 4u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    static constexpr uint32_t kMaxFlagEntries = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    std::array<uint64_t, kMaxDecodedPairs> m_decodedPairStamp{};
    DecodedInstructionPair m_uncachedDecodedPair{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;

    // Earliest readyCycle pending in the Q/P queues; commitReadyPipelines() skips the rest.
    uint64_t m_nextScalarCommit = 0;
    static void noteCommitCycle(uint64_t &next, uint64_t readyCycle)
    {
        if (readyCycle < next)
            next = readyCycle;
    }
    std::array<FlagPipelineEntry, kMaxFlagEntries> m_flagPipeline{}; // ring buffer in issue order
    uint32_t m_flagHead = 0;
    uint32_t m_flagCount = 0;
    FlagPipelineEntry *pushFlagEntry(bool upper, uint32_t overflowCode);
    // Applies the flag writes whose latency has elapsed. Flags are committed lazily: only their
    // readers (FMxxx, FSxxx, FCxxx), the Q status write and the end of a run need them current.
    void commitFlags();
    template <typename Fn>
    void forEachFlagEntryIssuedThisCycle(Fn &&fn)
    {
        for (uint32_t index = m_flagCount; index-- > 0u;)
        {
            FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + index) % kMaxFlagEntries];
            if (entry.issueCycle != m_cycle)
                break;
            fn(entry);
        }
    }
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    XgkickPipeline m_xgkick{};
    std::array<std::array<uint64_t, 4>, 32> m_vfReady{};
    // Latest ready cycle ever set on any lane of each VF register (and of ACC): a read whose
    // register is past this needs no per-lane check.
    std::array<uint64_t, 32> m_vfReadyMax{};
    uint64_t m_accReadyMax = 0;
    std::array<uint64_t, 16> m_viReady{};
    std::array<uint64_t, 4> m_accReady{};

    // Register writes of the last few cycles with their previous values: a run that stops on its cycle
    // budget hides those still in flight, and the next run reveals them again.
    struct InFlightWrite
    {
        enum class Kind : uint8_t
        {
            Vf,
            Vi,
            Acc
        };
        uint64_t readyCycle = 0;
        float oldValue[4]{};
        float newValue[4]{};
        Kind kind = Kind::Vf;
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };
    static constexpr uint32_t kInFlightWrites = 16u;
    std::array<InFlightWrite, kInFlightWrites> m_inFlight{};
    uint32_t m_inFlightNext = 0;
    uint32_t m_inFlightCount = 0;
    bool m_inFlightHidden = false;
    bool m_logInFlight = false;
    void logInFlightWrite(InFlightWrite::Kind kind, uint8_t reg, uint8_t lanes, uint64_t readyCycle,
                          const float oldValue[4], const float newValue[4]);
    void applyInFlightWrite(const InFlightWrite &entry, bool newValue);
    void hideInFlightWrites();
    void revealInFlightWrites();

    uint64_t m_cycle = 0;
    uint64_t m_lastWritebackCycle = 0;
    uint64_t m_efuResourceReady = 0;
    uint32_t m_workingClip = 0;
    uint32_t m_currentUpperInstruction = 0;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;
    PacketObserver m_packetObserver;
    uint64_t m_externalCodeGeneration = 0;

    // Compiled fragments matching the current code memory, by pair index (empty: none apply).
    std::vector<CompiledEntry> m_compiledByPair;
    const uint8_t *m_compiledCode = nullptr;
    uint64_t m_compiledStamp = 0;
    bool m_compiledChecked = false;
    bool m_compiledCodeDisabled = false;
    void refreshCompiledFragments(const uint8_t *vuCode, uint32_t codeSize, uint64_t stamp);

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles, bool runToCompletion = false);
    // stepPair's body; kLogInFlight keeps the previous value of the registers each pair writes (time-sliced runs).
    template <bool kLogInFlight, uint8_t kUpperHints = 0>
#if defined(_MSC_VER) && !defined(__clang__)
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    StepResult stepPairImpl(const DecodedInstructionPair &decoded, const RunContext &context);
    // Everything stepPairImpl does once the pair can issue. kDeadFlags: see BlockPlan::deadFlags.
    template <bool kLogInFlight, bool kDeadFlags = false, uint8_t kUpperHints = 0, BlockPosition kPosition = NotInBlock>
#if defined(_MSC_VER) && !defined(__clang__)
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    StepResult stepPairBody(const DecodedInstructionPair &decoded, const RunContext &context);
    template <BlockLiveIn kLiveIn>
    uint64_t blockLiveInReadyCycle() const;
    void foldDeadSticky();
    bool m_flagWritesDead = false; // the pair being executed has dead MAC/status writes (BlockPlan::deadFlags)
    uint32_t m_deadSticky = 0;     // their sticky bits (Z/S/U/O), for the next flag write that lands
    StepResult stepPairLogged(const DecodedInstructionPair &decoded, const RunContext &context);
    void stallUntilReady(const DecodedInstructionPair &decoded, uint64_t readyCycle, uint64_t budgetEnd);

    static constexpr void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static constexpr void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static constexpr uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    const DecodedInstructionPair &getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory,
                                                                 uint32_t pc);

    // kUpperHints: kKnownNormal* (BlockPlan::knownNormal) and kHintFlagsUnobserved.
    template <uint8_t kUpperHints = 0>
    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    template <uint8_t kUpperHints = 0>
    void applyFmacDest(float *dst, float *result, uint8_t dest);
    template <uint8_t kUpperHints = 0>
    void applyFmacDestAcc(float *result, uint8_t dest);
    template <uint8_t kUpperHints = 0>
    bool fmacFlagsFast(float *result, uint8_t dest, uint32_t &mac, uint32_t &status, uint32_t &sticky) const;
    // fmacFlagsFast with the operands given: each callable returns one, normalized (right: broadcast, Q or I
    // already spread; cross products unshuffled).
    template <typename Left, typename Right, typename Acc>
    bool fmacFlagsWith(float *result, uint8_t dest, ps2_vu1_detail::FmacOperation operation, const Left &left,
                       const Right &right, const Acc &acc, uint32_t &mac, uint32_t &status, uint32_t &sticky) const;
    template <uint8_t kUpperHints = 0>
    void fmacFlags(float *result, uint8_t dest);
    void fmacFlagsSlow(float *result, uint8_t dest);
    uint8_t fmacSumAtFloatMaxFlags(float &value, uint32_t lane, double sum, ps2_vu1_detail::FmacKind kind,
                                   const float left[4], const float right[4], const float acc[4]) const;
    void normalizeFmacResult(float *result, uint8_t dest, uint8_t laneFlags[4]);
    bool calculateFmacExactResult(uint32_t component, long double &result) const;
    uint8_t normalizeFmacExactResult(float &value, long double exactResult) const;
    uint32_t calculateFmacProductSticky(uint8_t dest) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    void commitScalarPipelines();
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick(uint64_t cycles);
    void progressXgkickActive(uint64_t cycles);
    void resetXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);
    void markPairWritesAt(const DecodedInstructionPair &decoded, uint64_t issueCycle);
    bool pipelinesPending() const;

    float normalizeOperand(float value) const;
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const;
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
