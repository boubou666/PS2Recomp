#ifndef PS2_RUNTIME_H
#define PS2_RUNTIME_H

#include <cstring>
#include <cstdint>
#include <vector>
#include <deque>
#include <string>
#include <string_view>
#include <functional>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX instructions
#include <smmintrin.h> // For SSE4.1 instructions
#endif
#include <atomic>
#include <array>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "ps2_log.h"
#include "runtime/ps2_address.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_audio.h"
#include "runtime/ps2_pad.h"
#include "runtime/ps2_rom_device.h"
#include "runtime/ps2_vfs.h"
#include "runtime/frame_pacer.h"
#include "ps2x/iop/iop_types.h"
#include "ps2_recomp_abi.h"

namespace ps2x::iop
{
    class IopSubsystem;
}

class PS2IopHostAdapter;
class PS2IopTransport;
class EeScheduler;
struct EeEvent;

class PS2Runtime
{
public:
    struct IoPaths
    {
        std::filesystem::path elfPath;
        std::filesystem::path elfDirectory;
        std::filesystem::path hostRoot;
        std::filesystem::path cdRoot;
        std::filesystem::path mcRoot;
        std::filesystem::path cdImage;
    };

    PS2Runtime();
    ~PS2Runtime();

    bool initialize(const char *title = "PS2 Game");
    bool syncCoreSubsystems();
    bool loadELF(const std::string &elfPath);
    void run();

    [[nodiscard]] ps2x::iop::ModuleLoadResult loadIopModule(std::string_view path, const void *arguments = nullptr, uint32_t argumentSize = 0);
    [[nodiscard]] ps2x::iop::ModuleLoadResult loadIopModuleBuffer(uint32_t guestAddress, const void *arguments = nullptr, uint32_t argumentSize = 0);
    [[nodiscard]] ps2x::iop::ModuleLoadResult loadIopModuleBufferFromIop(uint32_t iopAddress, const void *arguments = nullptr, uint32_t argumentSize = 0);
    // EE -> IOP SIF command for a handler registered on the IOP with sceSifAddCmdHandler.
    [[nodiscard]] bool sendIopSifCommand(uint32_t cid, const void *packet, uint32_t size);
    [[nodiscard]] bool stopIopModule(int32_t moduleId, int32_t *result = nullptr);
    [[nodiscard]] ps2x::iop::DebugSnapshot iopDebugSnapshot() const;
    // The IOP on its own thread: the EE cycles the recompiled code accounts are owed to the IOP, which runs them
    // in slices on that thread instead of in lockstep on the EE thread. The IOP stays behind the EE; every other
    // IOP entry point (RPC, SIF transfers and commands, IOP memory) runs between two slices, and the EE waits only
    // if the IOP falls more than two fields behind. IOP -> EE SIF commands are queued to the EE thread
    // (EeScheduler::postTask). PS2X_IOP_THREAD=1 enables it.
    void setIopThreaded(bool enabled);
    [[nodiscard]] bool iopThreaded() const noexcept { return m_iopThreaded.load(std::memory_order_relaxed); }
    // With the IOP thread, runs an RPC to a server an IOP module registered on that thread and calls done with
    // the result on the EE thread (EeScheduler::postTask); send is the send buffer copied at the call. False
    // (nothing queued) without the IOP thread or for an HLE service: the caller runs the RPC itself.
    bool queueIopRpc(ps2x::iop::RpcRequest request, std::vector<uint8_t> send,
                     std::function<void(const ps2x::iop::RpcResult &)> done);
    uint32_t allocateIopMemory(uint32_t size, uint32_t alignment = 16u);
    bool freeIopMemory(uint32_t address);
    bool readIopMemory(uint32_t address, void *destination, size_t size) const;
    bool writeIopMemory(uint32_t address, const void *source, size_t size);
    bool zeroIopMemory(uint32_t address, size_t size);
    bool isIopMemoryRange(uint32_t address, size_t size) const;

    using DebugUiCallback = void (*)(PS2Runtime &runtime, void *userData);
    void setDebugUiCallbacks(DebugUiCallback initCallback,
                             DebugUiCallback drawCallback,
                             DebugUiCallback shutdownCallback,
                             void *userData);

    using RecompiledFunction = Ps2RecompiledFunction;
    using GuestBranchKind = Ps2GuestBranchKind;

    enum class MissingFunctionPolicy : uint32_t
    {
        // Strict mode for tests/CI: log the bad target and request the runtime to stop.
        Stop = 0,

        // Debug mode: log once, leave ctx->pc on the bad target, and let the caller unwind.
        ContinueToTarget = 1,

        // Debug mode: same as ContinueToTarget, but triggers a debugger break once on MSVC.
        BreakOnce = 2,

        // Escape hatch only: skip missing calls by returning to fallthrough (it can hide guest bugs)
        SkipCallDebug = 3,
    };

    bool replaceFunction(uint32_t address, RecompiledFunction func);
    // TODO remove this later need to update all tests
    bool registerFunction(uint32_t address, RecompiledFunction func);
    RecompiledFunction lookupFunction(uint32_t address);
    bool hasFunction(uint32_t address) const;
    // An address no recompiled function covers but holding `jr $ra` with a nop delay slot: an
    // empty routine to return from (compilers reuse a function's final jr ra as an empty method,
    // and games build such pointers at run time, out of reach of static entry discovery).
    bool isTrivialReturnStub(uint32_t address) const;
    bool dispatchGuestBranch(uint8_t *rdram,
                             R5900Context *ctx,
                             uint32_t targetPc,
                             uint32_t sourcePc,
                             uint32_t fallthroughPc,
                             GuestBranchKind kind,
                             const char *debugName);
    void reportMissingFunction(uint8_t *rdram,
                               R5900Context *ctx,
                               uint32_t targetPc,
                               uint32_t sourcePc,
                               GuestBranchKind kind,
                               const char *debugName);
    void setMissingFunctionPolicy(MissingFunctionPolicy policy);
    MissingFunctionPolicy missingFunctionPolicy() const;
    void resetMissingFunctionReportOnce();

    static const IoPaths &getIoPaths();
    static void setIoPaths(const IoPaths &paths);
    static void configureIoPathsFromElf(const std::string &elfPath);

    void SignalException(R5900Context *ctx, PS2Exception exception);

    void executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address);
    void vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address);
    // COP0 Count (MFC0/MTC0 $9): the EE clock, one tick per EE cycle the scheduler has accounted.
    uint32_t readCop0Count() const noexcept;
    void writeCop0Count(uint32_t value) noexcept;
    // COP0 $25 (MFPS/MFPC, MTPS/MTPC): select is the instruction's low bits, bit 0 clear for PCCR, set for a
    // counter (bit 1: PCR0 or PCR1). While PCCR.CTE (bit 31) is set the counters count EE cycles.
    uint32_t readCop0Perf(uint32_t select) const noexcept;
    void writeCop0Perf(uint32_t select, uint32_t value) noexcept;
    // Count and the counters on the host's clock (in EE cycles) instead of the EE's accounted cycles: for games that
    // time their frames with Count (native frame pacing). Off by default.
    void setCop0WallClock(bool on) noexcept;

public:
    void handleSyscall(uint8_t *rdram, R5900Context *ctx);
    void handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId);
    void handleBreak(uint8_t *rdram, R5900Context *ctx);

    void handleTrap(uint8_t *rdram, R5900Context *ctx);
    void handleTLBR(uint8_t *rdram, R5900Context *ctx);
    void handleTLBWI(uint8_t *rdram, R5900Context *ctx);
    void handleTLBWR(uint8_t *rdram, R5900Context *ctx);
    void handleTLBP(uint8_t *rdram, R5900Context *ctx);
    void clearLLBit(R5900Context *ctx);
    void configureGuestHeap(uint32_t guestBase, uint32_t guestLimit = PS2_RAM_SIZE);
    // A game that runs its own sbrk owns guest RAM from the end of its BSS up to EndOfHeap, the same span the runtime's
    // guest heap (HLE DMA packets, thread stacks, ...) takes by default. pinGuestHeap moves the runtime's heap to
    // [guestBase, guestLimit) for good: SetupHeap no longer moves it and EndOfHeap returns guestBase, so the game's sbrk
    // stops below it. For game overrides.
    void pinGuestHeap(uint32_t guestBase, uint32_t guestLimit);
    bool guestHeapPinned() const;
    uint32_t guestMalloc(uint32_t size, uint32_t alignment = 16u);
    uint32_t guestCalloc(uint32_t count, uint32_t size, uint32_t alignment = 16u);
    uint32_t guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment = 16u);
    void guestFree(uint32_t guestAddr);
    uint32_t guestHeapBase() const;
    uint32_t guestHeapEnd() const;
    uint32_t guestHeapLimit() const;
    uint32_t guestHeapHardLimit() const;
    uint32_t reserveAsyncCallbackStack(uint32_t size, uint32_t alignment = 16u);

    // Guest RAM kept by the runtime: HLE pools (SIF RPC packets and servers, TLS, boot mode)
    // in [base, base + kReservedPoolBytes) and invocation stacks growing down from top.
    // Default: pools at 0x01F00000, stacks below 32 MiB, guest heap ending at 0x01F00000.
    // Games that use all user RAM can move the region (e.g. into kernel memory below the
    // ELF) and raise the heap limit. Call from a game override, before the guest runs.
    static constexpr uint32_t kReservedPoolBytes = 0x40000u;
    void configureReservedMemory(uint32_t base, uint32_t top, uint32_t guestHeapHardLimit);
    static uint32_t reservedMemoryBase();
    // SetupThread's main stack [stackBase, stackTop). Unless configureReservedMemory placed the region, a stack that
    // overlaps it moves the region to the spare kernel RAM below the ELF, or below the stack when the ELF is there,
    // and the guest heap ends at the stack or the region.
    void reserveMainThreadStack(uint32_t stackBase, uint32_t stackTop);

    // Host-rendered frames shown instead of the GS output while the callback returns true
    // (e.g. a game override playing a movie on the host). Called on the render thread.
    using PresentationOverride = std::function<bool(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height)>;
    void setPresentationOverride(PresentationOverride presentationOverride);
    bool presentOverride(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height);

    void drainCompletedDmacHandlers(uint8_t *rdram);
    // Raise INTC cause 0 (GS) for unmasked CSR SIGNAL/FINISH rising edges recorded by the GS.
    void pollGsInterrupt(uint8_t *rdram);

    void requestStop();
    bool isStopRequested() const;

    // A game layer that knows where the game's frame ends reports each one here: the fps overlay (PS2X_SHOW_FPS)
    // counts them, and counts display buffer flips (GS::displayFlips) for games that report nothing.
    void noteGameFrame() { m_gameFrames.fetch_add(1u, std::memory_order_relaxed); }
    uint64_t gameFrames() const { return m_gameFrames.load(std::memory_order_relaxed); }

    EeScheduler &eeScheduler();
    const EeScheduler &eeScheduler() const;
    // Native frame pacing for games that run their loop on real time (runtime/frame_pacer.h).
    FramePacer &framePacer() { return m_framePacer; }
    void postEeEvent(EeEvent event);
    bool eeCheckpointDue(uint32_t cycles = 32u) noexcept;
    friend bool ps2rCheckpointDue(PS2Runtime *runtime, uint32_t cycles) noexcept;
    [[noreturn]] void eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc);

    struct EeExitHandlerRegistration
    {
        uint32_t function = 0;
        uint32_t argument = 0;
    };
    void addEeExitHandler(int threadId, uint32_t function, uint32_t argument);
    std::vector<EeExitHandlerRegistration> takeEeExitHandlers(int threadId);
    void removeEeExitHandlers(int threadId);
    bool findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const;
    void setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler);
    void initializeEeKernelState(uint8_t *rdram);

    uint8_t Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint16_t Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint32_t Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint64_t Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    __m128i Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);

    void Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value);
    void Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value);
    void Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value);
    void Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value);
    void Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value);
    void kickGifDmaChainFromMMIO(uint8_t *rdram,
                                 R5900Context *ctx,
                                 uint32_t dPcrValue,
                                 uint32_t dStatValue,
                                 uint32_t tadr,
                                 uint32_t chcr);

    static inline bool isSpecialAddress(uint32_t addr)
    {
        return Ps2IsSpecialAddress(addr);
    }

public:
    inline R5900Context &cpu() { return m_cpuContext; }
    inline const R5900Context &cpu() const { return m_cpuContext; }

    // The EE server's reply to an IOP -> EE RPC (PS2IopHostAdapter::callEeRpc), delivered to the IRX under the IOP lock.
    void completeIopEeRpc(uint64_t token, const std::vector<uint8_t> &reply);
    inline PS2Memory &memory() { return m_memory; }
    inline const PS2Memory &memory() const { return m_memory; }

    // The GS as the guest sees it: after the queued VIF1 transfers (PS2Memory::waitForVif1).
    inline GS &gs()
    {
        m_memory.waitForVif1();
        return m_gs;
    }
    inline const GS &gs() const
    {
        m_memory.waitForVif1();
        return m_gs;
    }
    // Without waiting: presentation, which shows whatever is drawn by then.
    inline GS &gsForPresentation() { return m_gs; }
    inline GifArbiter &gifArbiter() { return m_gifArbiter; }
    inline const GifArbiter &gifArbiter() const { return m_gifArbiter; }
    inline VU1Interpreter &vu0() { return m_vu0; }
    inline const VU1Interpreter &vu0() const { return m_vu0; }
    inline VU1Interpreter &vu1() { return m_vu1; }
    inline const VU1Interpreter &vu1() const { return m_vu1; }

    inline PS2AudioBackend &audioBackend() { return m_audioBackend; }
    inline const PS2AudioBackend &audioBackend() const { return m_audioBackend; }
    inline PSPadBackend &padBackend() { return m_padBackend; }
    inline const PSPadBackend &padBackend() const { return m_padBackend; }
    inline PS2RomDevice &romDevice() { return m_romDevice; }
    inline const PS2RomDevice &romDevice() const { return m_romDevice; }
    inline PS2Vfs &vfs() { return m_vfs; }
    inline const PS2Vfs &vfs() const { return m_vfs; }

private:
    struct GuestHeapBlock
    {
        uint32_t addr = 0;
        uint32_t size = 0;
        bool free = true;
    };

    static uint32_t alignGuestHeapValue(uint32_t value, uint32_t alignment);
    static bool isGuestHeapAlignmentValid(uint32_t alignment);
    static uint32_t normalizeGuestHeapAlignment(uint32_t alignment);
    uint32_t clampGuestHeapBase(uint32_t guestBase) const;
    uint32_t clampGuestHeapLimit(uint32_t guestLimit) const;
    void resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit);
    void ensureGuestHeapInitializedLocked();
    int32_t findGuestHeapBlockIndexLocked(uint32_t guestAddr) const;
    uint32_t allocateGuestBlockLocked(uint32_t size, uint32_t alignment);
    void freeGuestBlockLocked(uint32_t guestAddr);
    void coalesceGuestHeapLocked();
    void placeReservedMemory(uint32_t base, uint32_t top, uint32_t guestHeapHardLimit);
    void HandleIntegerOverflow(R5900Context *ctx);

    [[nodiscard]] ps2x::iop::RpcAbi selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const;
    [[nodiscard]] bool canBindIopRpc(uint32_t sid) const noexcept;
    [[nodiscard]] ps2x::iop::RpcResult handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request);
    void notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer);
    void advanceIopEeCycles(uint64_t eeCycles) noexcept;
    void resetIop();

    friend class PS2IopTransport;
    friend class EeScheduler;

private:
    PS2Memory m_memory;
    GifArbiter m_gifArbiter;
    GS m_gs;
    std::unique_ptr<PS2IopHostAdapter> m_iopHost;
    std::unique_ptr<ps2x::iop::IopSubsystem> m_iopSubsystem;
    void iopThreadMain();
    mutable std::recursive_mutex m_iopMutex;
    std::thread m_iopThread;
    std::mutex m_iopWorkMutex;
    std::condition_variable m_iopWorkCv;
    std::condition_variable m_iopDrainCv;
    std::atomic<uint64_t> m_iopOwedCycles{0};
    std::atomic<bool> m_iopThreadStop{false};
    std::atomic<bool> m_iopThreaded{false};
    mutable std::atomic<int> m_iopWanted{0};
    std::deque<std::function<void()>> m_iopJobs; // guarded by m_iopWorkMutex
    PS2AudioBackend m_audioBackend;
    PSPadBackend m_padBackend;
    PS2RomDevice m_romDevice;
    PS2Vfs m_vfs;
    VU1Interpreter m_vu0{VU1Interpreter::Unit::VU0};
    VU1Interpreter m_vu1{VU1Interpreter::Unit::VU1};
    R5900Context m_cpuContext;
    std::unique_ptr<EeScheduler> m_eeScheduler;
    FramePacer m_framePacer;
    mutable std::mutex m_eeKernelStateMutex;
    std::unordered_map<int, std::vector<EeExitHandlerRegistration>> m_eeExitHandlers;
    std::unordered_map<uint32_t, uint32_t> m_eeSyscallOverrides;
    std::unordered_set<uint32_t> m_eeSyscallMirrorAddresses;
    mutable std::mutex m_guestHeapMutex;
    mutable std::mutex m_asyncCallbackStackMutex;
    std::vector<GuestHeapBlock> m_guestHeapBlocks;
    uint32_t m_guestHeapBase = 0x00100000u;
    uint32_t m_guestHeapEnd = 0x00100000u;
    uint32_t m_guestHeapLimit = PS2_RAM_SIZE;
    uint32_t m_guestHeapSuggestedBase = 0x00100000u;
    uint32_t m_guestHeapHardLimit = 0x01F00000u;
    bool m_guestHeapConfigured = false;
    bool m_guestHeapPinned = false;
    uint64_t cop0Clock() const noexcept;
    bool m_cop0WallClock = false;
    uint64_t m_cop0CountBase = 0; // EE cycle at which Count read 0 (MTC0 Count moves it)
    uint32_t m_cop0Pccr = 0;
    uint64_t m_cop0PcrBase[2] = {}; // EE cycle at which the counter read 0, while counting
    uint32_t m_cop0PcrValue[2] = {}; // the counter while stopped
    uint32_t m_asyncCallbackStackFloor = 0x01F00000u;
    uint32_t m_asyncCallbackStackTop = PS2_RAM_SIZE;
    uint32_t m_reservedMemoryTop = PS2_RAM_SIZE;
    bool m_reservedMemoryExplicit = false;

    std::atomic<uint32_t> m_missingFunctionPolicy{static_cast<uint32_t>(MissingFunctionPolicy::ContinueToTarget)};
    std::atomic<bool> m_missingFunctionReported{false};
    std::atomic<bool> m_stopRequested{false};
    std::atomic<uint64_t> m_gameFrames{0};
    DebugUiCallback m_debugUiInitCallback = nullptr;
    DebugUiCallback m_debugUiDrawCallback = nullptr;
    DebugUiCallback m_debugUiShutdownCallback = nullptr;
    void *m_debugUiUserData = nullptr;
    bool m_debugUiInitialized = false;
    bool m_gsIrqLine = false;
    std::mutex m_presentationOverrideMutex;
    PresentationOverride m_presentationOverride;

public:
    std::atomic<uint32_t> m_debugPc{0};
    std::atomic<uint32_t> m_debugRa{0};
    std::atomic<uint32_t> m_debugSp{0};
    std::atomic<uint32_t> m_debugGp{0};

private:
    struct LoadedModule
    {
        std::string name;
        uint32_t baseAddress;
        size_t size;
        bool active;
    };

    std::vector<LoadedModule> m_loadedModules;
    uint8_t *m_boundRdram = nullptr;
    uint8_t *m_boundGSVram = nullptr;
};

// Generated by ps2xRecomp in ps2xRuntime/src/runner/register_functions.cpp.
extern const uint32_t g_ps2RecompiledFunctionTableBase;
extern const uint32_t g_ps2RecompiledFunctionTableEnd;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount;
extern PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[];

#endif // PS2_RUNTIME_H
