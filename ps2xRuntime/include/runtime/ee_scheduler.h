#pragma once

#include "ps2_runtime.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

// This exception is the EE equivalent of a longjmp to the dispatcher.  It is
// not an error and must only be caught at EeScheduler::run().
struct EeDispatcherTransfer final
{
};

enum class EeThreadStatus : uint8_t
{
    Running,
    Ready,
    Waiting,
    WaitingSuspended,
    Suspended,
    Dormant,
};

enum class EeWaitReason : uint8_t
{
    None,
    Sleep,
    Semaphore,
    EventFlag,
    VSync,
    External,
    Mpeg,
};

struct EeSemaphoreWait
{
    int id = 0;
};

struct EeEventFlagWait
{
    int id = 0;
    uint32_t bits = 0;
    uint32_t mode = 0;
    uint32_t resultAddress = 0;
};

struct EeVSyncWait
{
    uint64_t afterTick = 0;
    int fixedResult = -1;
};

struct EeExternalWait
{
    uint32_t type = 0;
    uint64_t token = 0;
};

using EeWaitPayload = std::variant<std::monostate,
                                   EeSemaphoreWait,
                                   EeEventFlagWait,
                                   EeVSyncWait,
                                   EeExternalWait>;

struct EeWaitState
{
    EeWaitReason reason = EeWaitReason::None;
    EeWaitPayload payload{};
    std::function<void(R5900Context &)> completion;
};

enum class GuestInvocationKind : uint8_t
{
    Interrupt,
    Alarm,
    GsCallback,
    RpcCallback,
    SyscallOverride,
    ExitHandler,
    HleCall,
    SifCommand,
};

struct GuestInvocation
{
    GuestInvocationKind kind = GuestInvocationKind::Interrupt;
    uint64_t sequence = 0;
    uint64_t tag = 0;
    R5900Context context{};
    std::function<void(const R5900Context &, R5900Context &)> onComplete;
};

struct GuestThread
{
    int id = 0;
    R5900Context context{};
    uint32_t entry = 0;
    uint64_t cycles = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t arg = 0;
    int initialPriority = 0;
    int currentPriority = 0;
    EeThreadStatus status = EeThreadStatus::Dormant;
    int suspendCount = 0;
    uint32_t wakeupCount = 0;
    bool ownsStack = false;
    uint32_t tlsBase = 0;
    EeWaitState wait{};
    std::function<void(R5900Context &)> resumeCompletion;
    std::vector<GuestInvocation> invocations;

    [[nodiscard]] R5900Context &activeContext()
    {
        return invocations.empty() ? context : invocations.back().context;
    }

    [[nodiscard]] const R5900Context &activeContext() const
    {
        return invocations.empty() ? context : invocations.back().context;
    }
};

struct EeSemaphore
{
    int id = 0;
    int count = 0;
    int maxCount = 0;
    int initCount = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    std::deque<int> waiters;
};

struct EeEventFlag
{
    int id = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t initBits = 0;
    uint32_t bits = 0;
    std::deque<int> waiters;
};

struct EeAlarm
{
    int id = 0;
    uint16_t ticks = 0;
    uint32_t handler = 0;
    uint32_t argument = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
};

struct EeIrqHandler
{
    int id = 0;
    uint32_t cause = 0;
    uint32_t handler = 0;
    uint32_t argument = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
    bool enabled = true;
    int order = 0;
};

struct EeThreadSnapshot
{
    int id = 0;
    uint32_t pc = 0;
    uint32_t ra = 0;
    uint32_t sp = 0;
    uint32_t contextGp = 0;
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    int initialPriority = 0;
    int currentPriority = 0;
    EeThreadStatus status = EeThreadStatus::Dormant;
    EeWaitReason waitReason = EeWaitReason::None;
    int waitId = 0;
    int suspendCount = 0;
    uint32_t wakeupCount = 0;
    uint32_t invocationDepth = 0;
    uint64_t cycles = 0; // EE cycles accounted while this thread ran
};

struct EeSemaphoreSnapshot
{
    int id = 0;
    int count = 0;
    int maxCount = 0;
    uint32_t waiters = 0;
};

struct EeEventFlagSnapshot
{
    int id = 0;
    uint32_t bits = 0;
    uint32_t initBits = 0;
    uint32_t attr = 0;
    uint32_t waiters = 0;
};

struct EeKernelSnapshot
{
    uint64_t sequence = 0;
    uint64_t eeCycle = 0;
    uint64_t sliceEndCycle = 0;
    uint64_t nextEventCycle = 0;
    uint64_t idleCycles = 0; // EE cycles skipped while no thread was ready
    int runningThreadId = 0;
    uint32_t enabledIntcMask = 0;
    uint32_t pendingIntcMask = 0;   // causes raised while disabled
    uint32_t enabledDmacMask = 0;
    uint32_t pendingDmacMask = 0;
    std::vector<EeIrqHandler> dmacHandlers;
    uint32_t pendingInvocations = 0; // queued handlers not started yet
    bool heldUntilEi = false;        // ... or a thread switch, because the running thread executed DI
    std::vector<EeThreadSnapshot> threads;
    std::vector<EeSemaphoreSnapshot> semaphores;
    std::vector<EeEventFlagSnapshot> eventFlags;
};

enum class EeEventType : uint8_t
{
    Stop,
    VBlankStart,
    VBlankEnd,
    Dmac,
    ExternalWake,
    Alarm,
    GsInterrupt, // GS raised SIGNAL/FINISH; dispatch INTC cause 0 now rather than at the next VBlank
    Task,        // postTask: run the next queued host task on the executor thread
};

struct EeEvent
{
    EeEventType type = EeEventType::ExternalWake;
    uint32_t id = 0;
    uint64_t value = 0;
};

struct EeThreadCreateParams
{
    uint32_t attr = 0;
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    int priority = 0;
    uint32_t option = 0;
};

class EeScheduler
{
public:
    static constexpr int kMainThreadId = 1;
    static constexpr int kFirstThreadId = 2;
    static constexpr int kLastThreadId = 255;
    static constexpr int kPriorityCount = 128;
    static constexpr uint64_t kEeClockHz = 294912000ull;
    static constexpr uint32_t kGeneratedCheckpointCycles = 32u;
    static constexpr uint32_t kGuestDispatchCycles = 8u;
    static constexpr uint64_t kDefaultTimeSliceCycles = 65536ull;

    explicit EeScheduler(PS2Runtime &runtime);
    ~EeScheduler();

    EeScheduler(const EeScheduler &) = delete;
    EeScheduler &operator=(const EeScheduler &) = delete;

    void reset(uint8_t *rdram, const R5900Context &mainContext);
    void run();
    void requestStop();
    void postEvent(EeEvent event);
    // Runs task on the executor thread at its next checkpoint (or wakes it). Thread-safe; for host threads
    // (the IOP thread) that must reach EE state only the executor may touch.
    void postTask(std::function<void()> task);
    [[nodiscard]] bool onExecutorThread() const noexcept { return m_executorThread == std::this_thread::get_id(); }
    // Host time the executor spent idle, every guest thread blocked (waiting for a vblank, an interrupt...).
    [[nodiscard]] uint64_t idleWaitNanoseconds() const noexcept { return m_idleWaitNs.load(std::memory_order_relaxed); }
    // One idle wait, for a game layer that wants to know what its frames wait on (setIdleObserver).
    struct IdleRecord
    {
        uint64_t nanoseconds = 0;
        // What ended it: the first posted event ("ev:<type>") or the deadline that came ("deadline:<type>",
        // "timer" for an EE timer interrupt).
        std::string wokenBy;
        // The blocked guest threads, by id: "t<id>p<priority>:<what>@<return address>", what being S<id>
        // (semaphore), E<id> (event flag), V (vsync), Z (sleep), X<type> (external), M (MPEG).
        std::string blocked;
    };
    // Called on the executor thread after each idle wait (nullptr: off, nothing is collected).
    void setIdleObserver(std::function<void(const IdleRecord &)> observer) { m_idleObserver = std::move(observer); }
    // Called on the executor thread with its id when a thread ends (returns from its entry or calls ExitThread), once
    // it is dormant, so an HLE that owns a thread can start it again (libcdvd's callback thread). Nothing runs then
    // (current thread 0): startThread on the ended id just makes it ready, and the dispatcher picks it up next. After
    // ExitDeleteThread the record is already gone, so thread(id) is null there.
    void addThreadEndObserver(std::function<void(int)> observer) { m_threadEndObservers.push_back(std::move(observer)); }
    // EE cycles skipped by wall-clock catch-up (see WallClockCatchUp).
    [[nodiscard]] uint64_t catchUpCycles() const noexcept { return m_catchUpCycles.load(std::memory_order_relaxed); }
    [[nodiscard]] bool checkpointDue(uint32_t cycles = kGeneratedCheckpointCycles) noexcept;
    // Set while guest code unwinds to the dispatcher (a checkpoint fired or a jump left the
    // function), cleared before each dispatch: the enclosing dispatchGuestBranch frames must not
    // read ctx->pc as the result of their call (see dispatchGuestBranch).
    void noteTransferToDispatcher() noexcept { m_transferToDispatcher = true; }
    [[nodiscard]] bool transferToDispatcherPending() const noexcept { return m_transferToDispatcher; }
    void accountCycles(uint32_t cycles) noexcept;
    [[nodiscard]] bool isExecutingGuest() const noexcept;

    // Kernel object API. All calls except postEvent/requestStop execute on the
    // EE executor and therefore need no host synchronization.
    void setupCurrentThread(uint32_t stack, uint32_t stackSize, uint32_t gp);
    int createThread(const EeThreadCreateParams &params);
    int deleteThread(int id, uint32_t &ownedStack);
    int startThread(int id, uint32_t arg, const R5900Context &caller, bool interruptSafe);
    [[noreturn]] void exitCurrent(bool deleteThread);
    int terminateThread(int id, uint32_t &ownedStack, bool interruptSafe);
    int suspendThread(int id, bool interruptSafe);
    int resumeThread(int id, bool interruptSafe);
    void sleepCurrent();
    int wakeupThread(int id, bool interruptSafe);
    int cancelWakeup(int id);
    int changePriority(int id, int priority, bool interruptSafe, int &oldPriority);
    int rotateReadyQueue(int priority, bool interruptSafe);
    int releaseWait(int id, bool interruptSafe);
    void transferIfRequested(bool interruptSafe);

    int createSemaphore(int initCount, int maxCount, uint32_t attr, uint32_t option);
    int deleteSemaphore(int id, bool interruptSafe);
    int signalSemaphore(int id, bool interruptSafe);
    int pollSemaphore(int id);
    void waitSemaphore(int id);

    int createEventFlag(uint32_t initialBits, uint32_t attr, uint32_t option);
    int deleteEventFlag(int id, bool interruptSafe);
    int setEventFlag(int id, uint32_t bits, bool interruptSafe);
    int clearEventFlag(int id, uint32_t mask);
    int pollEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t &observedBits);
    void waitEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t resultAddress);

    int setAlarm(uint16_t ticks, uint32_t handler, uint32_t argument, uint32_t gp, uint32_t sp);
    int cancelAlarm(int id);
    void queueInvocation(GuestInvocation invocation);
    [[nodiscard]] size_t pendingInvocationCount() const noexcept { return m_pendingInvocations.size(); }
    [[noreturn]] void invokeCurrent(GuestInvocation invocation);
    [[noreturn]] void invokeCurrentSequence(std::vector<GuestInvocation> invocations);
    [[nodiscard]] bool hasInvocation(GuestInvocationKind kind, uint64_t tag) const;
    [[nodiscard]] uint32_t invocationStackTop();

    int addIrqHandler(bool dmac,
                      uint32_t cause,
                      uint32_t handler,
                      bool append,
                      uint32_t argument,
                      uint32_t gp,
                      uint32_t sp);
    int removeIrqHandler(bool dmac, uint32_t cause, int id);
    int setIrqHandlerEnabled(bool dmac, int id, bool enabled);
    int setIrqCauseEnabled(bool dmac, uint32_t cause, bool enabled);
    void dispatchIrq(bool dmac, uint32_t cause);
    void setVSyncFlag(uint32_t flagAddress, uint32_t tickAddress);
    [[nodiscard]] uint64_t currentVSyncTick() const noexcept;
    [[nodiscard]] uint64_t currentCycle() const noexcept { return m_eeCycle; }
    // Re-checks the GS interrupt once `delayCycles` EE cycles have passed, and not before the host
    // time those cycles map to on the vblank timeline: an EE running behind real time is not held back
    // further, one running ahead waits as it would for the next vblank.
    void scheduleGsInterruptCheck(uint64_t delayCycles);
    // The same at an absolute EE cycle; callable from other host threads (the VIF1 worker).
    void scheduleGsInterruptCheckAt(uint64_t cycle);
    // PS2X_EE_PROFILE=1: EE cycles accounted at each checkpoint, by the guest pc it resumes at
    // (loop heads); returns and clears the counts gathered since the last call.
    [[nodiscard]] std::vector<std::pair<uint32_t, uint64_t>> takeGuestProfile();
    // Vblanks per second (numerator / denominator): NTSC fields until the game resets the GS
    // in PAL mode (sceGsResetGraph), which switches to 50 Hz.
    struct FieldRate
    {
        uint64_t numerator;
        uint64_t denominator;
    };
    static constexpr FieldRate kNtscFieldRate{60000u, 1001u};
    static constexpr FieldRate kPalFieldRate{50u, 1u};
    void setFieldRate(FieldRate rate) noexcept;
    [[nodiscard]] FieldRate fieldRate() const noexcept;
    // Guest code slower than real time normally fast-forwards to the next event once its host
    // deadline has passed, so busy-wait loops do not stretch VBlank and timer events. A game whose
    // EE is stalled by host work (a saturated GS) loses EE time that way while the IOP keeps its
    // pace: its EE-produced audio can underrun. Disabling it lets emulated time slow down instead.
    // OncePerField skips to at most one event per field: a game never sees more than one field go by
    // at once, which its late-frame paths may not expect. PS2X_EE_CATCHUP=0|field|1 overrides the mode.
    // Off also never lets the EE run faster than real time to pay back a slow stretch: a vblank more than
    // a field late starts the next field from the time it was handled.
    enum class WallClockCatchUp : uint8_t
    {
        Off,
        OncePerField,
        Full,
    };
    void setWallClockCatchUp(bool enabled) noexcept { m_wallClockCatchUp = enabled ? WallClockCatchUp::Full : WallClockCatchUp::Off; }
    void setWallClockCatchUp(WallClockCatchUp mode) noexcept { m_wallClockCatchUp = mode; }
    [[nodiscard]] WallClockCatchUp wallClockCatchUp() const noexcept;
    uint32_t setGsVSyncCallback(uint32_t callback, uint32_t gp, uint32_t sp);
    // Variable refresh, for a game override: while the hold is on, a VBlankStart that falls due waits, in steps of
    // 250 us, until the hold is released or `limit` has passed since its time; the next field then counts from when it
    // fired. A frame that overruns the field by a little takes that little longer instead of a whole second field.
    void setVBlankHold(bool held) noexcept { m_vblankHold.store(held, std::memory_order_relaxed); }
    void setVBlankHoldLimit(std::chrono::microseconds limit) noexcept { m_vblankHoldLimit = limit; }
    [[nodiscard]] uint64_t vblanksHeld() const noexcept { return m_vblanksHeld.load(std::memory_order_relaxed); }
    // A game that waits for its vblank by spinning on a counter instead of blocking (sceGsSyncV, WaitVSync): thread
    // `threadId` is in that wait loop while its pc is in [begin, end). Its frame counts as complete there, so vblanks
    // latch the picture as they do for blocked waiters (otherwise only every third vblank does). With holdVBlank, a
    // due VBlankStart also waits (up to the hold limit) while the thread runs outside the loop. end == 0 turns it off.
    void setGuestFrameWait(int threadId, uint32_t begin, uint32_t end, bool holdVBlank) noexcept
    {
        m_frameWaitThread = threadId;
        m_frameWaitPcBegin = begin;
        m_frameWaitPcEnd = end;
        m_frameWaitHoldsVBlank = holdVBlank;
    }
    // Native frames: a game that runs its own loop on real time (FramePacer) shows each frame when it ends it
    // (presentGuestFrame, from the game's frame end on the EE thread) instead of at vblanks, which then latch
    // nothing, and keeps EE time on real time itself: events fire once EE time reaches them, never held for their
    // host deadline. The vblank timer keeps running for whatever else of the game counts on it.
    void setNativeFrames(bool enabled) noexcept { m_nativeFrames = enabled; }
    [[nodiscard]] bool nativeFrames() const noexcept { return m_nativeFrames; }
    void presentGuestFrame();
    // Whether a thread of equal priority takes over at the end of a time slice (default yes; PS2X_EE_SLICE_EQUAL=0
    // no). The real EE kernel never round-robins equal priorities: a thread runs until it blocks or a higher
    // priority one becomes ready.
    void setEqualPrioritySlicing(bool enabled) noexcept { m_equalPrioritySlicing.store(enabled, std::memory_order_relaxed); }
    [[nodiscard]] bool equalPrioritySlicing() const noexcept { return m_equalPrioritySlicing.load(std::memory_order_relaxed); }

    [[noreturn]] void waitVSync(uint64_t afterTick, int fixedResult = -1, std::function<void(R5900Context &)> completion = {});
    void completeVSync(uint64_t tick);
    void completeExternalWait(uint32_t type, uint64_t token, int result);
    [[noreturn]] void waitExternal(EeWaitReason reason, uint32_t type, uint64_t token, std::function<void(R5900Context &)> completion = {});
    // Blocks the current guest thread until deadline on the host clock while the other guest threads run: a native
    // frame end for a game whose frame wait sleeps until its vblank (JC2, SCTPT2), where lower-priority threads work in
    // that time; holding the executor itself (FramePacer::frameEnd) starves them. At the deadline onWake runs on the
    // executor (charge EE time, present the frame), then completion resumes the thread (return value, pc).
    [[noreturn]] void waitHostDeadline(std::chrono::steady_clock::time_point deadline, std::function<void()> onWake,
                                       std::function<void(R5900Context &)> completion = {});

    [[nodiscard]] GuestThread *thread(int id);
    [[nodiscard]] const GuestThread *thread(int id) const;
    [[nodiscard]] EeSemaphore *semaphore(int id);
    [[nodiscard]] const EeSemaphore *semaphore(int id) const;
    [[nodiscard]] EeEventFlag *eventFlag(int id);
    [[nodiscard]] const EeEventFlag *eventFlag(int id) const;
    [[nodiscard]] GuestThread *currentThread();
    [[nodiscard]] const GuestThread *currentThread() const;
    [[nodiscard]] int currentThreadId() const noexcept;
    [[nodiscard]] R5900Context *currentContext();
    [[nodiscard]] uint8_t *rdram() const noexcept;

    // Direct syscall tests use the same main-thread record without starting a
    // second executor. Production execution calls reset() before run().
    void bindMainContextForSyscall(R5900Context &ctx, uint8_t *rdram);

    [[nodiscard]] EeKernelSnapshot snapshot() const;
    // Pending deadlines by event type, with how far ahead the furthest is (state dump).
    [[nodiscard]] std::string deadlineSummary() const;
    void publishSnapshot();

private:
    struct ScheduledEvent
    {
        uint64_t deadlineCycle = 0;
        std::chrono::steady_clock::time_point hostDeadline{};
        EeEvent event{};
        uint64_t sequence = 0;
    };

    void assertExecutor() const;
    [[nodiscard]] int allocateThreadId();
    GuestThread &acquireInvocationThread();
    void enqueueReady(GuestThread &thread, bool front = false);
    void removeReady(GuestThread &thread);
    [[nodiscard]] GuestThread *selectReady();
    void makeRunning(GuestThread &thread);
    void makeDormant(GuestThread &thread);
    void removeFromWaitObject(GuestThread &thread);
    [[noreturn]] void blockCurrent(EeWaitState wait);
    void makeReady(GuestThread &thread, int result, bool interruptSafe);
    void requestPreemptionIfHigher(const GuestThread &readyThread, bool interruptSafe);
    void applyPendingPreemption();
    void processPendingEvents();
    void processDueDeadlines();
    [[nodiscard]] bool interruptInvocationPending() const;
    // The running thread executed DI outside a handler, or runs an interrupt-context handler: on the
    // PS2 the thread switch waits for its EI or for the handler's return (ERET).
    [[nodiscard]] bool threadSwitchDeferred() const noexcept;
    void processEvent(const EeEvent &event);
    void finishEventWaiters(EeEventFlag &flag, bool interruptSafe);
    [[nodiscard]] static bool eventCondition(uint32_t current, uint32_t requested, uint32_t mode);
    static int waitObjectId(const EeWaitState &wait);
    void writeGuestU32(uint32_t address, uint32_t value);
    void waitForEvent();
    void scheduleEvent(uint64_t deadlineCycle, std::chrono::steady_clock::time_point hostDeadline, EeEvent event);
    [[nodiscard]] std::chrono::steady_clock::time_point timelineHostTime(uint64_t cycle) const;
    void updateNextDeadline();
    [[nodiscard]] bool hasReadyAtOrAbovePriority(int priority) const;
    void renewTimeSlice();
    void copyMainContextToRuntime();
    void publishDebugContext(const R5900Context &context);
    void publishIdleDebugContext();

    PS2Runtime &m_runtime;
    uint8_t *m_rdram = nullptr;
    std::array<std::deque<int>, kPriorityCount> m_readyQueues{};
    std::unordered_map<int, GuestThread> m_threads;
    std::unordered_map<int, EeSemaphore> m_semaphores;
    std::unordered_map<int, EeEventFlag> m_eventFlags;
    std::unordered_map<int, EeAlarm> m_alarms;
    std::unordered_map<int, EeIrqHandler> m_intcHandlers;
    std::unordered_map<int, EeIrqHandler> m_dmacHandlers;
    int m_nextThreadId = kFirstThreadId;
    int m_nextInvocationThreadId = -1;
    int m_nextSemaphoreId = 1;
    int m_nextEventFlagId = 1;
    int m_nextAlarmId = 1;
    int m_nextIntcHandlerId = 1;
    int m_nextDmacHandlerId = 1;
    int m_intcHeadOrder = 0;
    int m_intcTailOrder = 1000;
    int m_dmacHeadOrder = 0;
    int m_dmacTailOrder = 1000;
    uint32_t m_enabledIntcMask = 0xFFFFFFFFu;
    uint32_t m_enabledDmacMask = 0xFFFFFFFFu;
    // Causes raised while disabled stay pending (INTC_STAT / D_STAT) until enabled again; a DMAC
    // cause is dropped if the game cleared its D_STAT CIS meanwhile.
    uint32_t m_pendingIntcMask = 0u;
    uint32_t m_pendingDmacMask = 0u;
    int m_currentThreadId = 0;
    bool m_rescheduleRequested = false;
    bool m_timeSliceExpired = false;
    bool m_transferToDispatcher = false;
    bool m_insideInterrupt = false;
    // A pending invocation or a thread switch waits because the running thread executed DI
    // (Status.EIE clear) or runs a handler (see threadSwitchDeferred). checkpointDue() returns to
    // the dispatcher once it executes EI; a handler's return reaches the dispatcher anyway.
    bool m_heldUntilEi = false;
    uint32_t m_pendingEeTimerInterrupts = 0;
    uint64_t m_eeCycle = 0;
    uint64_t m_idleCycles = 0;
    std::mutex m_guestProfileMutex;
    std::unordered_map<uint32_t, uint64_t> m_guestProfile;
    uint64_t m_iopEeCycles = 0; // EE cycles handed to the IOP (accounted ones plus vblank catch-up)
    uint64_t m_iopVblankEnd = 0; // IOP cycle at which the current vblank's frame ends
    FieldRate m_fieldRate = kNtscFieldRate;
    uint64_t m_vblankPeriodCycles = 0;
    // The vblank timeline (cycle and host time of the last vblank start), read by other host threads.
    std::atomic<uint64_t> m_timelineCycle{0};
    std::atomic<std::chrono::steady_clock::rep> m_timelineHostTicks{0};
    std::chrono::nanoseconds m_vblankPeriod{0};
    std::atomic<bool> m_vblankHold{false};
    std::chrono::microseconds m_vblankHoldLimit{6000};
    std::chrono::steady_clock::time_point m_vblankHeldSince{}; // the held vblank's own time (zero: none held)
    uint64_t m_vblankHeldCycle = 0;
    std::atomic<uint64_t> m_vblanksHeld{0};
    int m_frameWaitThread = 0;
    uint32_t m_frameWaitPcBegin = 0;
    uint32_t m_frameWaitPcEnd = 0;
    bool m_frameWaitHoldsVBlank = false;
    bool m_nativeFrames = false;
    void latchGuestFrame();
    // The setGuestFrameWait thread, when it is running or ready: whether it sits in its wait loop.
    [[nodiscard]] std::optional<bool> guestFrameWaiting();
    std::atomic<bool> m_equalPrioritySlicing{[]
    {
        const char *value = std::getenv("PS2X_EE_SLICE_EQUAL");
        return !(value && *value == '0');
    }()};
    uint64_t m_sliceEndCycle = kDefaultTimeSliceCycles;
    std::thread::id m_executorThread{};
    std::deque<std::function<void()>> m_tasks; // guarded by m_eventMutex
    std::atomic<uint64_t> m_idleWaitNs{0};
    std::function<void(const IdleRecord &)> m_idleObserver;
    std::vector<std::function<void(int)>> m_threadEndObservers;
    void notifyThreadEnd(int id);
    [[nodiscard]] std::string describeBlockedThreads() const;
    void reportIdle(std::chrono::steady_clock::time_point start, std::string wokenBy, std::string blocked);
    std::atomic<uint64_t> m_catchUpCycles{0};
    void noteIdleWait(std::chrono::steady_clock::time_point start) noexcept
    {
        m_idleWaitNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - start).count()),
                               std::memory_order_relaxed);
    }
    // waitHostDeadline: a host thread (started on first use) posts each wake-up at its deadline.
    static constexpr uint32_t kHostDeadlineWaitType = 0x48445750u; // 'HDWP'
    void hostDeadlineLoop();
    std::mutex m_hostDeadlineMutex;
    std::condition_variable m_hostDeadlineCv;
    std::vector<std::pair<std::chrono::steady_clock::time_point, uint64_t>> m_hostDeadlines; // guarded by the mutex
    std::unordered_map<uint64_t, std::function<void()>> m_hostDeadlineWakes; // executor only
    uint64_t m_hostDeadlineToken = 0; // executor only
    bool m_hostDeadlineQuit = false; // guarded by the mutex; set by the destructor
    std::thread m_hostDeadlineThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_guestExecuting{false};
    std::atomic<bool> m_stopRequested{false};
    std::atomic<bool> m_checkpointPending{false};
    uint32_t m_debugPublishCountdown = 0u;

    mutable std::mutex m_eventMutex;
    std::condition_variable m_eventCv;
    std::deque<EeEvent> m_events;
    std::vector<ScheduledEvent> m_deadlines;
    std::deque<GuestInvocation> m_pendingInvocations;
    uint64_t m_eventSequence = 0;
    uint64_t m_invocationSequence = 0;
    uint64_t m_vsyncTick = 0;
    uint32_t m_vsyncFlagAddress = 0;
    uint32_t m_vsyncTickAddress = 0;
    uint32_t m_gsVSyncCallback = 0;
    uint32_t m_gsVSyncCallbackGp = 0;
    uint32_t m_gsVSyncCallbackSp = 0;
    std::unordered_map<uint64_t, uint32_t> m_invocationStackTops;
    std::atomic<uint64_t> m_nextDeadlineCycle{0};
    // Host deadline of the event at m_nextDeadlineCycle; lets slow guest code catch up to wall-clock time.
    std::atomic<std::chrono::steady_clock::rep> m_nextDeadlineHostTicks{0};
    uint32_t m_hostCatchUpPoll = 0;
    WallClockCatchUp m_wallClockCatchUp = WallClockCatchUp::Full;
    uint64_t m_lastCatchUpTick = ~0ull;
    std::chrono::steady_clock::time_point m_deadlinesHeldUntil{};

    mutable std::mutex m_snapshotMutex;
    EeKernelSnapshot m_snapshot;
    uint64_t m_snapshotSequence = 0;
};
