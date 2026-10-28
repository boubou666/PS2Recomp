#ifndef PS2_MEMORY_H
#define PS2_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <iostream>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "gs/gif_signal_scan.h"
#include "gs/ps2_gif_arbiter.h"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX instructions
#include <smmintrin.h> // For SSE4.1 instructions
#endif

class GS;

#include "runtime/ps2_memory_map.h"

// PS2 GS (Graphics Synthesizer) registers
struct GSRegisters
{
    uint64_t pmode;    // Pixel mode
    uint64_t smode1;   // Sync mode 1
    uint64_t smode2;   // Sync mode 2
    uint64_t srfsh;    // Refresh control
    uint64_t synch1;   // Synchronization control 1
    uint64_t synch2;   // Synchronization control 2
    uint64_t syncv;    // Synchronization control V
    uint64_t dispfb1;  // Display buffer 1
    uint64_t display1; // Display area 1
    uint64_t dispfb2;  // Display buffer 2
    uint64_t display2; // Display area 2
    uint64_t extbuf;   // External buffer
    uint64_t extdata;  // External data
    uint64_t extwrite; // External write
    uint64_t bgcolor;  // Background color
    std::atomic<uint64_t> csr;
    std::atomic<uint64_t> vsyncTick;
    uint64_t imr;      // Interrupt mask
    uint64_t busdir;   // Bus direction
    uint64_t siglblid; // Signal label ID
};
static_assert(sizeof(GSRegisters) == (20u * sizeof(uint64_t)), "GSRegisters layout changed unexpectedly");
static_assert(alignof(GSRegisters) == alignof(uint64_t), "GSRegisters alignment must remain 64-bit");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "GS CSR atomic must be lock-free on all supported targets");

// CSR SIGNAL (bit0) / FINISH (bit1) 0->1 transitions raised by the GS, consumed as INTC cause 0 edges.
inline std::atomic<uint32_t> g_gsCsrRaisedEdges{0};

// GS register writes the CPU observes (SIGNAL, FINISH, LABEL) take effect in order; the GS holds
// everything after a SIGNAL until the CPU clears CSR.SIGNAL. Each returns the CSR event bits that
// rose (SIGNAL 0x1, FINISH 0x2).
enum class GsCpuEvent : uint8_t
{
    Signal,
    Finish,
    Label
};
// Events take effect from EE cycle `due` (the GS needs time to reach them); `now` is the current
// EE cycle.
uint32_t ps2GsPostEvent(GSRegisters &regs, GsCpuEvent event, uint32_t id, uint32_t mask, uint64_t due, uint64_t now);
uint32_t ps2GsApplyDueEvents(GSRegisters &regs, uint64_t now);
void ps2GsResetEvents();

// PS2 VIF (VPU Interface) registers
struct VIFRegisters
{
    uint32_t stat;   // Status
    uint32_t fbrst;  // VIF Force Break
    uint32_t err;    // Error status
    uint32_t mark;   // Interrupt control
    uint32_t cycle;  // Transfer mode
    uint32_t mode;   // Mode control
    uint32_t num;    // Data amount counter
    uint32_t mask;   // Data mask
    uint32_t code;   // VIFcode
    uint32_t itops;  // ITOP save
    uint32_t base;   // Base address
    uint32_t ofst;   // Offset
    uint32_t tops;   // TOPS
    uint32_t itop;   // ITOP
    uint32_t top;    // TOP
    uint32_t row[4]; // Transfer row data
    uint32_t col[4]; // Transfer column data
};
static_assert(sizeof(VIFRegisters) == (23u * sizeof(uint32_t)), "VIFRegisters layout changed unexpectedly");

// PS2 DMA registers
struct DMARegisters
{
    uint32_t chcr; // Channel control
    uint32_t madr; // Memory address
    uint32_t qwc;  // Quadword count
    uint32_t tadr; // Tag address
    uint32_t asr0; // Address stack 0
    uint32_t asr1; // Address stack 1
    uint32_t sadr; // Source address
};
static_assert(sizeof(DMARegisters) == (7u * sizeof(uint32_t)), "DMARegisters layout changed unexpectedly");

struct JumpTable
{
    uint32_t address = 0;          // Base address of the jump table
    uint32_t baseRegister = 0;     // Register used for index
    std::vector<uint32_t> targets; // Jump targets
};

class PS2Memory
{
public:
    PS2Memory();
    ~PS2Memory();

    PS2Memory(const PS2Memory &) = delete;
    PS2Memory &operator=(const PS2Memory &) = delete;
    PS2Memory(PS2Memory &&) = delete;
    PS2Memory &operator=(PS2Memory &&) = delete;

    // Initialize memory
    bool initialize(size_t ramSize = PS2_RAM_SIZE);

    // Memory access methods
    uint8_t *getRDRAM() { return m_rdram; }
    const uint8_t *getRDRAM() const { return m_rdram; }
    uint8_t *getScratchpad() { return m_scratchpad; }
    uint8_t *getIOPRAM() { return iop_ram; }
    uint64_t dmaStartCount() const { return m_dmaStartCount.load(std::memory_order_relaxed); }
    // The last GIF channel events, for diagnostics: 'S' start (CHCR with STR: chcr, madr, qwc), 'Q' end interrupt
    // queued, 'C' end interrupts consumed (count, 1 = by a sceDmaSend stub), 'D' dispatched (handlers started,
    // 1 = held because the cause is disabled).
    struct GifEvent
    {
        uint32_t kind = 0, a = 0, b = 0, c = 0;
    };
    static constexpr uint32_t kGifEvents = 32u;
    void noteGifEvent(uint32_t kind, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0)
    {
        const uint32_t slot = m_gifEventNext.fetch_add(1, std::memory_order_relaxed) % kGifEvents;
        m_gifEvents[slot] = {kind, a, b, c};
    }
    std::array<GifEvent, kGifEvents> gifEvents(uint32_t &next) const
    {
        next = m_gifEventNext.load(std::memory_order_relaxed);
        return m_gifEvents;
    }
    uint64_t gifCopyCount() const { return m_gifCopyCount.load(std::memory_order_relaxed); }
    uint64_t gsWriteCount() const { return m_gsWriteCount.load(std::memory_order_relaxed); }
    uint64_t vifWriteCount() const { return m_vifWriteCount.load(std::memory_order_relaxed); }
    uint64_t getVU0CodeGeneration() const { return m_vu0CodeGeneration.load(std::memory_order_relaxed); }
    uint64_t getVU1CodeGeneration() const { return m_vu1CodeGeneration.load(std::memory_order_relaxed); }

    // Read/write memory
    uint8_t read8(uint32_t address);
    uint16_t read16(uint32_t address);
    uint32_t read32(uint32_t address);
    uint64_t read64(uint32_t address);
    __m128i read128(uint32_t address);

    void write8(uint32_t address, uint8_t value);
    void write16(uint32_t address, uint16_t value);
    void write32(uint32_t address, uint32_t value);
    void write64(uint32_t address, uint64_t value);
    void write128(uint32_t address, __m128i value);

    // TLB handling
    uint32_t translateAddress(uint32_t virtualAddress);
    bool tlbRead(uint32_t index, uint32_t &vpn, uint32_t &pfn, uint32_t &mask, bool &valid) const;
    bool tlbWrite(uint32_t index, uint32_t vpn, uint32_t pfn, uint32_t mask, bool valid);
    int32_t tlbProbe(uint32_t vpn) const;
    size_t tlbEntryCount() const { return m_tlbEntries.size(); }

    // Hardware register interface
    bool writeIORegister(uint32_t address, uint32_t value);
    uint32_t readIORegister(uint32_t address);
    void raiseIntcStat(uint32_t bits) noexcept { m_intcStat.fetch_or(bits, std::memory_order_acq_rel); }

    // EE timers advance from the scheduler's emulated EE-cycle clock. The
    // returned mask uses bits 0..3 for newly raised TIM0..TIM3 interrupts.
    uint32_t advanceEeTimers(uint64_t eeCycles) noexcept;
    // EE cycles owed to timer busy-waits (see readIORegister): the scheduler accounts them so a
    // loop polling a timer faster than it ticks finishes in game time, not in host time.
    uint64_t takeTimerSpinCycles() noexcept { return m_timerSpinCycles.exchange(0u, std::memory_order_acq_rel); }
    uint64_t pendingTimerSpinCycles() const noexcept { return m_timerSpinCycles.load(std::memory_order_acquire); }
    [[nodiscard]] uint64_t cyclesUntilNextEeTimerInterrupt() const noexcept;
    // Timer registers as last updated (no flush, no side effects): for diagnostics from any thread.
    struct EeTimerState
    {
        uint32_t count = 0;
        uint32_t mode = 0;
        uint32_t compare = 0;
        uint32_t compareBacklog = 0;
        uint32_t overflowBacklog = 0;
    };
    [[nodiscard]] EeTimerState eeTimerState(size_t index) const noexcept
    {
        if (index >= m_eeTimers.size())
            return {};
        return {m_eeTimers[index].count, m_eeTimers[index].mode, m_eeTimers[index].compare,
                m_timerCompareBacklog[index], m_timerOverflowBacklog[index]};
    }
    void resetEeTimers() noexcept;
    // Applies the cycles advanceEeTimers() has been holding back (timer registers read the result).
    uint32_t flushEeTimers() noexcept;

    // GS -> EE image data (TRXDIR = 1) that the VIF1 FIFO hands to the EE: VIF1 DMA with CHCR.DIR = 0, quadword
    // reads of VIF1_FIFO, and VIF1_STAT.FQC while the EE has set VIF1_STAT.FDR.
    using LocalToHostReader = std::function<uint32_t(uint8_t *, uint32_t)>;
    void setLocalToHostReader(LocalToHostReader reader) { m_localToHostReader = std::move(reader); }

    using GifPacketCallback = std::function<void(const uint8_t *, uint32_t)>;
    void setGifPacketCallback(GifPacketCallback cb) { m_gifPacketCallback = std::move(cb); }

    // DMA channel 4 (toIPU) for the HLE MPEG decoder, enabled while an HLE stream is fed as elementary stream. A start leaves the
    // channel busy and the decoder pulls the source data as it needs it, as the IPU does; a game may loop a whole movie
    // held in memory with a chain that never ends. A guest that polls CHCR while nothing pulls (it drives the IPU
    // itself) sees the transfer complete, as with the channel disabled.
    void setIpuInputEnabled(bool enabled);
    // Code that drives the IPU itself (its registers) gets the transfer completed instead of parked.
    void releaseIpuInput();
    std::vector<uint8_t> pullIpuInput(size_t minBytes);
    void setGifArbiter(GifArbiter *arbiter) { m_gifArbiter = arbiter; }

    using Vu1MscalCallback = std::function<void(uint32_t startPC, uint32_t top, uint32_t itop)>;
    void setVu1MscalCallback(Vu1MscalCallback cb) { m_vu1MscalCallback = std::move(cb); }
    // VIF0 MSCAL/MSCALF (start pc in bytes) and MSCNT (~0u: continue at the current pc).
    using Vu0MscalCallback = std::function<void(uint32_t startPC)>;
    void setVu0MscalCallback(Vu0MscalCallback cb) { m_vu0MscalCallback = std::move(cb); }
    using Vu1MscntCallback = std::function<void(uint32_t top, uint32_t itop)>;
    void setVu1MscntCallback(Vu1MscntCallback cb) { m_vu1MscntCallback = std::move(cb); }

    uint8_t *getVU1Code() { return m_vu1Code; }
    const uint8_t *getVU1Code() const { return m_vu1Code; }
    uint8_t *getVU1Data() { return m_vu1Data; }
    const uint8_t *getVU1Data() const { return m_vu1Data; }
    uint8_t *getVU0Code() { return m_vu0Code; }
    const uint8_t *getVU0Code() const { return m_vu0Code; }
    uint8_t *getVU0Data() { return m_vu0Data; }
    const uint8_t *getVU0Data() const { return m_vu0Data; }

    bool isPath3Masked() const { return m_path3Masked; }
    void flushMaskedPath3Packets(bool drainImmediately = true);

    void submitPendingPath2Image(const uint8_t *data, uint32_t chunkQw, bool directHl);
    // dmaTransfer: PATH3 data of a GIF DMA transfer, whose last GIF tag may continue in the next transfer.
    void submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately = true, bool path2DirectHl = false,
                         bool dmaTransfer = false);
    void processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount);
    void processGIFPacket(const uint8_t *data, uint32_t sizeBytes);
    bool tryProcessNativeGifImageUploadChain(GS &gs, uint32_t tadr, uint32_t chcr);
    bool tryProcessNativeGifPackedChain(GS &gs, uint32_t tadr, uint32_t chcr);
    void processVIF0Data(uint32_t srcPhysAddr, uint32_t sizeBytes);
    void processVIF0Data(const uint8_t *data, uint32_t sizeBytes);
    void processVIF1Data(uint32_t srcPhysAddr, uint32_t sizeBytes);
    uint32_t processVIF1Data(const uint8_t *data, uint32_t sizeBytes);
    void processPendingTransfers();
    // A GIF channel transfer (normal mode, PATH3) of a packet the host built, as if the EE had started it: in order
    // with the rest of the GIF stream, and the channel ends with its DMAC interrupt. For HLE calls that send GS
    // packets (sceGsExecLoadImage) without taking guest memory for them.
    void submitHostGifTransfer(const uint8_t *data, uint32_t size);
    // A SIGNAL that reaches the GS while CSR.SIGNAL is set stops the GS until the CPU clears the bit. The GIF and
    // VIF1 DMA channels stop at such a SIGNAL (CHCR.STR stays set, MADR/QWC/TADR point at it) and continue from
    // memory once CSR.SIGNAL is acknowledged. PS2X_GS_SIGNAL_STALL=0 disables it.
    void setSignalStallEnabled(bool enabled) { m_signalStallEnabled = enabled; }
    bool dmaChannelSignalStalled(uint32_t channelBase) const;
    void resumeSignalStalls();
    std::vector<uint32_t> consumeCompletedDmacCauses();

    // VIF1 DMA (unpacks, VU1 microprograms, PATH1/PATH2 packets) on a worker thread, as the VIF and VU1 run beside
    // the EE on the hardware. Each transfer still completes at once for the EE: its data is copied when it starts,
    // and anything that observes VIF1, VU1 or the GS (their registers, VU1 memory, other GIF paths, GS kernel calls)
    // first waits for the queued transfers (waitForVif1). PS2X_VIF1_THREAD=1 enables it.
    void setVif1Threaded(bool enabled);
    bool vif1Threaded() const { return m_vif1Threaded; }
    void waitForVif1(const char *site = __builtin_FUNCTION(), int line = __builtin_LINE()) const;
    // Diagnostics: where the EE waited for the VIF1 worker since the last call (site:line ms, ...).
    std::string takeVif1WaitSites() const;
    // GS lag (with the worker): the GS and VU1 may finish a frame while the EE runs the next one, as on the
    // hardware. PATH3 packets and GS privileged register writes then join the worker's queue in order instead of
    // waiting for it (CSR, live state, is read and written at once), the presented frame is latched by the worker
    // once it has drawn it (queueVif1Callback), and sceGsSyncPath only waits for the frame before
    // (waitForVif1PreviousFrame). PS2X_GS_LAG=1 enables it. A game whose threads depend on the EE being held back
    // by the GS may behave differently with it.
    void setGsLag(bool enabled) { m_gsLag = enabled; }
    bool gsLag() const { return m_gsLag && m_vif1Threaded; }
    // With the GS lag the worker keeps its own EE-cycle clock: a job starts no earlier than the EE cycle it was
    // queued at, and the VU1 microprograms it runs advance it by their VU cycles. The EE may run ahead of the
    // worker's clock only while the worker is idle (vif1Horizon is then unbounded): past it, waitForVif1Clock
    // blocks until the worker gets there, so events the worker raises (SIGNAL, FINISH) reach the EE at the cycle
    // they belong to instead of after the EE has run past them.
    // PS2X_VIF1_SLACK=<EE cycles> lets the EE run that far past the horizon (the worker's events may reach it up to
    // that late) instead of stopping at the start of every microprogram the worker runs.
    [[nodiscard]] bool vif1BehindEe(uint64_t eeCycle) const
    {
        const uint64_t horizon = m_vif1Horizon.load(std::memory_order_acquire);
        return horizon != ~0ull && eeCycle > horizon + m_vif1Slack;
    }
    void waitForVif1Clock(uint64_t eeCycle) const;
    [[nodiscard]] uint64_t vif1WorkerCycle() const { return m_vif1WorkerCycle.load(std::memory_order_acquire); }
    void advanceVif1WorkerClock(uint64_t cycles);
    void queueVif1Callback(std::function<void()> callback);
    // Marks the end of a frame in the queue (the presentation latch).
    void markVif1Frame();
    // Waits for the transfers queued before the last frame mark, and for the frames before that one to be presented
    // (setPresentedFramesProvider): the GS is at most one frame behind, which bounds the added input latency.
    void waitForVif1PreviousFrame();
    // Host time threads other than the worker spent waiting for it (waitForVif1*).
    [[nodiscard]] uint64_t vif1WaitNanoseconds() const noexcept { return m_vif1WaitNs.load(std::memory_order_relaxed); }
    using PresentedFramesProvider = std::function<uint64_t()>;
    void setPresentedFramesProvider(PresentedFramesProvider provider) { m_presentedFramesProvider = std::move(provider); }
    bool onVif1Worker() const;
    // VU0 FBRST of the EE when the transfer being processed was queued (D and T bit enables of VU1).
    using Vu0FbrstProvider = std::function<uint32_t()>;
    void setVu0FbrstProvider(Vu0FbrstProvider provider) { m_vu0FbrstProvider = std::move(provider); }
    uint32_t vif1JobVu0Fbrst() const { return m_vif1JobVu0Fbrst; }

    int pollDmaRegisters();

    // Track code modifications for self-modifying code
    void registerCodeRegion(uint32_t start, uint32_t end);
    bool isCodeAddress(uint32_t address) const;
    bool isCodeModified(uint32_t address, uint32_t size);
    void clearModifiedFlag(uint32_t address, uint32_t size);

    // GS register accessors
    GSRegisters &gs() { return gs_regs; }
    const GSRegisters &gs() const { return gs_regs; }
    uint8_t *getGSVRAM() { return m_gsVRAM; }
    const uint8_t *getGSVRAM() const { return m_gsVRAM; }
    bool hasSeenGifCopy() const { return m_seenGifCopy; }
    // Main RAM (32MB)
    uint8_t *m_rdram;

    // Scratchpad memory (16KB)
    uint8_t *m_scratchpad;

    // IOP RAM (2MB)
    uint8_t *iop_ram;

    bool m_seenGifCopy;
    std::atomic<uint64_t> m_dmaStartCount{0};
    std::array<GifEvent, kGifEvents> m_gifEvents{};
    std::atomic<uint32_t> m_gifEventNext{0};
    std::atomic<uint64_t> m_gifCopyCount{0};
    std::atomic<uint64_t> m_gsWriteCount{0};
    std::atomic<uint64_t> m_vifWriteCount{0};
    std::atomic<uint64_t> m_vu0CodeGeneration{0};
    std::atomic<uint64_t> m_vu1CodeGeneration{0};
    // I/O registers
    std::unordered_map<uint32_t, uint32_t> m_ioRegisters;
    std::atomic<uint32_t> m_intcStat{0u};

    // Registers
    GSRegisters gs_regs;
    uint8_t *m_gsVRAM;
    VIFRegisters vif0_regs;
    VIFRegisters vif1_regs;
    DMARegisters dma_regs[10]; // 10 DMA channels

    // TLB entries
    struct TLBEntry
    {
        uint32_t vpn;
        uint32_t pfn;
        uint32_t mask;
        bool valid;
    };

    std::vector<TLBEntry> m_tlbEntries;

    GifPacketCallback m_gifPacketCallback;
    struct IpuInputDma
    {
        bool active = false;
        bool chain = false;
        bool endAfterData = false;
        uint32_t tadr = 0;
        uint32_t madr = 0;
        uint32_t qwc = 0;
        uint32_t pollsWithoutPull = 0;
    };
    bool m_ipuInputEnabled = false;
    IpuInputDma m_ipuInputDma;
    GifArbiter *m_gifArbiter = nullptr;
    Vu1MscalCallback m_vu1MscalCallback;
    Vu0MscalCallback m_vu0MscalCallback;
    Vu1MscntCallback m_vu1MscntCallback;

    uint8_t *m_vu0Code = nullptr;
    uint8_t *m_vu0Data = nullptr;
    uint8_t *m_vu1Code = nullptr;
    bool m_vu1ProgramSkipped = false; // the native port draws the running microprogram (ps2x::native::setVu1Gate)
    uint8_t *m_vu1Data = nullptr;
    bool m_path3Masked = false;
    uint32_t m_vif1PendingPath2ImageQwc = 0u;
    bool m_vif1PendingPath2DirectHl = false;
    // The DIRECT that left the IMAGE pending ran out of buffer: its data continues raw at the next VIF1 data.
    bool m_vif1PendingPath2ImageRaw = false;
    std::vector<std::vector<uint8_t>> m_path3MaskedFifo;

    struct DmaSegment
    {
        uint32_t offset = 0;
        uint32_t bytes = 0;
        uint32_t madr = 0;
        uint32_t tadr = 0;
        uint32_t asr0 = 0;
        uint32_t asr1 = 0;
        uint32_t chcr = 0;
        bool last = false;
    };
    struct PendingTransfer
    {
        bool fromScratchpad = false;
        uint32_t srcAddr = 0;
        uint32_t qwc = 0;
        std::vector<uint8_t> chainData;
        std::vector<DmaSegment> segments;
    };
    struct DmaSignalStall
    {
        uint32_t channelBase = 0;
        bool last = false;
        uint64_t generation = 0;
    };
    struct Vif1DirectContinuation
    {
        uint32_t qwLeft = 0;
        bool directHl = false;
        GifSignalScan scan;
    };
    bool m_signalStallEnabled = true;
    bool m_signalPassed = false;
    uint64_t m_signalStallGeneration = 0;
    std::vector<DmaSignalStall> m_signalStalls;
    GifSignalScan m_path3Scan;
    Vif1DirectContinuation m_vif1Direct;
    const std::vector<DmaSegment> *m_vif1Segments = nullptr;
    bool m_vif1StoppedAtSignal = false;
    bool signalStallApplies(uint32_t channelBase) const;
    bool signalStallBlocked(uint32_t channelBase) const;
    uint32_t findSignalStall(GifSignalScan &scan, const uint8_t *data, uint32_t size, uint32_t channelBase,
                             const std::vector<DmaSegment> &segments, uint32_t segmentOffset);
    void stallDmaChannel(uint32_t channelBase, const std::vector<DmaSegment> &segments, uint32_t offset);
    void resumeDmaChannel(const DmaSignalStall &stall);
    std::vector<PendingTransfer> m_pendingGifTransfers;
    std::vector<PendingTransfer> m_pendingVif0Transfers;
    std::vector<PendingTransfer> m_pendingVif1Transfers;
    std::mutex m_completedDmacMutex;
    std::vector<uint32_t> m_completedDmacCauses;

    struct Vif1Job
    {
        enum class Kind : uint8_t
        {
            Vif1,        // VIF1 DMA data
            Path3,       // a GIF packet on PATH3
            PrivWrite32, // GS privileged register writes
            PrivWrite64,
            Callback,
        };
        Kind kind = Kind::Vif1;
        std::vector<uint8_t> data;
        uint32_t vu0Fbrst = 0;
        uint64_t queueCycle = 0;
        uint32_t address = 0;
        uint64_t value = 0;
        std::function<void()> callback;
    };
    void queueVif1Job(std::vector<uint8_t> data);
    void queueVif1Job(Vif1Job job);
    void waitForVif1Until(uint64_t queued) const;
    bool m_gsLag = false;
    mutable std::atomic<uint64_t> m_vif1WaitNs{0};
    uint64_t m_vif1FrameMark = 0;
    uint64_t m_vif1FramesMarked = 0;
    PresentedFramesProvider m_presentedFramesProvider;
    void vif1WorkerMain();
    void stopVif1Worker();
    bool m_vif1Threaded = false;
    std::thread m_vif1Thread;
    mutable std::mutex m_vif1Mutex;
    std::condition_variable m_vif1WorkCv;
    mutable std::condition_variable m_vif1IdleCv;
    std::deque<Vif1Job> m_vif1Jobs;
    std::atomic<uint64_t> m_vif1Queued{0};
    std::atomic<uint64_t> m_vif1Done{0};
    std::atomic<uint64_t> m_vif1Horizon{~0ull};
    std::atomic<uint64_t> m_vif1WorkerCycle{0};
    uint64_t m_vif1Slack = 0;
    mutable std::atomic<bool> m_vif1ClockWaiter{false};
    bool m_vif1Stop = false;
    uint32_t m_vif1JobVu0Fbrst = 0;
    Vu0FbrstProvider m_vu0FbrstProvider;

    struct CodeRegion
    {
        uint32_t start;
        uint32_t end;
        std::vector<bool> modified; // Bitmap of modified 4-byte blocks
    };
    std::vector<CodeRegion> m_codeRegions;

    bool isAddressInRegion(uint32_t address, const CodeRegion &region);
    void markModified(uint32_t address, uint32_t size);
    void markVU0CodeModified() { m_vu0CodeGeneration.fetch_add(1, std::memory_order_relaxed); }
    void markVU1CodeModified() { m_vu1CodeGeneration.fetch_add(1, std::memory_order_relaxed); }
    bool isScratchpad(uint32_t address) const;
    uint8_t *mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit);
    const uint8_t *mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const;
    struct EeTimer
    {
        uint32_t count = 0;
        uint32_t mode = 0;
        uint32_t compare = 0;
        uint32_t hold = 0;
        uint64_t clockRemainder = 0;
        uint32_t lastReadCount = 0xFFFFFFFFu;
    };

    std::array<EeTimer, 4> m_eeTimers{};
    std::array<uint32_t, 4> m_timerOverflowBacklog{};
    std::array<uint32_t, 4> m_timerCompareBacklog{};
    uint32_t m_undeliveredTimerInterrupts = 0;
    uint64_t m_pendingTimerCycles = 0;
    uint64_t m_timerEventCycles = 0;
    bool m_gifFqcSet = false;
    uint32_t applyEeTimerCycles(uint64_t eeCycles) noexcept;
    [[nodiscard]] uint64_t eeTimerEventCycles() const noexcept;
    std::atomic<uint64_t> m_timerSpinCycles{0u};
    std::vector<uint8_t> readDmaSourceChain(uint32_t channelBase, uint32_t chcr, std::vector<DmaSegment> *segments = nullptr,
                                            uint32_t pendingQwc = 0, bool pendingEnds = false);
    bool tryProcessScratchpadDma(uint32_t channelBase, uint32_t chcr);
    LocalToHostReader m_localToHostReader;
    std::vector<uint8_t> m_vif1DownloadFifo;
    size_t m_vif1DownloadRead = 0;
    uint32_t vif1DownloadQuadwords();
    uint32_t readVif1Download(uint8_t *dst, uint32_t bytes);
    void downloadVif1Dma(uint32_t channelBase);
    void completeDmacChannel(uint32_t channelBase, uint32_t cause);
    void queueCompletedDmacCause(uint32_t cause);
};

#endif // PS2_MEMORY_H
