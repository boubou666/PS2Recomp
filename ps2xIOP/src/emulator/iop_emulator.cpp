#include <cstdlib>
#include <cstdio>
#include "iop_emulator.h"
#include "ps2x/iop/iop_native.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "core/iop_sio2.h"
#include "core/iop_spu2.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"

#include <algorithm>
#include <unordered_map>
#include <cstddef>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <optional>
#include <span>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>


namespace ps2x::iop::native
{
    namespace
    {
        struct RegisteredModule
        {
            std::string name;
            uint64_t fileHash = 0;
            uint32_t base = 0, size = 0;
            const Entry *entries = nullptr;
            size_t count = 0;
        };

        std::vector<RegisteredModule> &registeredModules()
        {
            static std::vector<RegisteredModule> modules;
            return modules;
        }
    }

    Registration::Registration(const char *name, uint64_t fileHash, uint32_t base, uint32_t size, const Entry *entries, size_t count)
    {
        registeredModules().push_back(RegisteredModule{name, fileHash, base, size, entries, count});
    }
}

static_assert(offsetof(ps2x::iop::detail::IopCpuState, gpr) == offsetof(ps2x::iop::native::NativeCpu, gpr) &&
                  offsetof(ps2x::iop::detail::IopCpuState, hi) == offsetof(ps2x::iop::native::NativeCpu, hi) &&
                  offsetof(ps2x::iop::detail::IopCpuState, lo) == offsetof(ps2x::iop::native::NativeCpu, lo) &&
                  offsetof(ps2x::iop::detail::IopCpuState, pc) == offsetof(ps2x::iop::native::NativeCpu, pc) &&
                  offsetof(ps2x::iop::detail::IopCpuState, cop0) == offsetof(ps2x::iop::native::NativeCpu, cop0),
              "NativeCpu must lay out as IopCpuState starts");

namespace ps2x::iop::detail
{
    namespace
    {
        // PS2X_IOP_TRACE_PC=0x62914,0x6291c: log a0-a3 and ra each time the IOP reaches one of these addresses.
        const std::unordered_set<uint32_t> *tracedPcs()
        {
            static const std::unordered_set<uint32_t> *pcs = []() -> const std::unordered_set<uint32_t> *
            {
                const char *spec = std::getenv("PS2X_IOP_TRACE_PC");
                if (!spec || !*spec)
                    return nullptr;
                auto *set = new std::unordered_set<uint32_t>();
                std::string text(spec);
                for (size_t start = 0; start < text.size();)
                {
                    size_t end = text.find(',', start);
                    if (end == std::string::npos)
                        end = text.size();
                    set->insert(static_cast<uint32_t>(std::strtoul(text.substr(start, end - start).c_str(), nullptr, 0)));
                    start = end + 1;
                }
                return set;
            }();
            return pcs;
        }

        // PS2X_IOP_COUNT_RANGE=0x56b00-0x63000: count executed instructions per pc in that range and write
        // "pc count" lines to PS2X_IOP_COUNT_OUT (default iop_pc_counts.txt) every 2^24 counted instructions.
        struct PcCounter
        {
            uint32_t low = 0;
            uint32_t high = 0;
            std::vector<uint64_t> counts;
            uint64_t total = 0;

            void write() const
            {
                const char *out = std::getenv("PS2X_IOP_COUNT_OUT");
                if (FILE *f = std::fopen(out && *out ? out : "iop_pc_counts.txt", "w"))
                {
                    for (size_t i = 0; i < counts.size(); ++i)
                        if (counts[i] != 0u)
                            std::fprintf(f, "0x%x %llu\n", low + static_cast<uint32_t>(i) * 4u,
                                         static_cast<unsigned long long>(counts[i]));
                    std::fclose(f);
                }
            }
        };

        PcCounter *pcCounter()
        {
            static PcCounter *counter = []() -> PcCounter *
            {
                const char *spec = std::getenv("PS2X_IOP_COUNT_RANGE");
                if (!spec || !*spec)
                    return nullptr;
                char *end = nullptr;
                auto *result = new PcCounter();
                result->low = static_cast<uint32_t>(std::strtoul(spec, &end, 0));
                result->high = static_cast<uint32_t>(std::strtoul(end && *end == '-' ? end + 1 : "0", nullptr, 0));
                if (result->high <= result->low)
                    return nullptr;
                result->counts.assign((result->high - result->low) / 4u, 0u);
                return result;
            }();
            return counter;
        }

        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;
        constexpr uint32_t kMaxSifCommandPacket = 112u;
        constexpr uint64_t kSifDmaCompletionCycles = 128u;
        constexpr uint32_t kSifCommandHandlerBudget = 100000u;

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        // EE -> IOP SIF command waiting for its IOP handler (sceSifAddCmdHandler).
        struct PendingSifCommand
        {
            IopRpcBridge::CmdHandler handler;
            std::array<uint8_t, kMaxSifCommandPacket> packet{};
            uint32_t size = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports),
              sio2(memory),
              spu2(memory)
        {
            memory.setSio2(&sio2);
            memory.setSpu2(&spu2);
            spu2.setClock(&totalCycles);
            spu2.setInterruptSink([this](int irq)
                                  { pendingDmaInterrupts[irq] = totalCycles; });
            if (spu2.wallClock())
                host.setAudioSource([this](int16_t *frames, size_t count) { spu2.pull(frames, count); });
            spu2.setOutputSink([this](const int16_t *frames, size_t count)
                               { host.submitAudio(frames, count); });
            for (int port = 0; port < 2; ++port)
                sio2.setPadStateReader(port, [this, port](uint8_t *state, size_t size)
                                       { return host.readPadState(port, state, size); });
            sio2.setPadPresence([this](int port) { return host.padConnected(port); });
            reset();
        }

        void reset()
        {
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            vblank.reset();
            ioman.reset();
            pendingDmaInterrupts.clear();
            heldInterrupts.clear();
            pendingGuestCallbacks.clear();
            handedOffServers.clear();
            reservedCallStacks.clear();
            pendingSifCommands.clear();
            sifCommandBuffer = 0u;
            sio2.reset();
            spu2.reset();
            dmaRequests = {};
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            paidCycles = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            callDepth = 0u;
            maskedThreadId = 0;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        // PS2X_IOP_TRACE_DMA=1: DMA / SIO2 interrupt requests and their dispatch.
        const bool traceDma = []
        {
            const char *value = std::getenv("PS2X_IOP_TRACE_DMA");
            return value && *value == '1';
        }();

        void schedulePendingDma()
        {
            if (const auto dma = memory.takeDmaStart())
            {
                if (traceDma)
                    std::fprintf(stderr, "[iop:dma] irq %d requested at %llu (+%llu)%s, pc 0x%x\n", dma->irq,
                                 static_cast<unsigned long long>(totalCycles), static_cast<unsigned long long>(dma->delayCycles),
                                 pendingDmaInterrupts.count(dma->irq) ? " (one already pending)" : "",
                                 activeCpu ? activeCpu->pc : 0u);
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
            }
        }

        ~Impl()
        {
            // The host's audio callback must not call into a SPU2 that is gone.
            if (spu2.wallClock())
                host.setAudioSource({});
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    pendingGuestCallbacks.emplace(
                        totalCycles + kCdvdCompletionCycles,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            // EnableIntr (6) on an interrupt that came while it was disabled: it is taken now (next instruction).
            if (iequals(call.library, "intrman") && call.ordinal == 6u && heldInterrupts.erase(static_cast<int>(a0)) != 0u)
                pendingDmaInterrupts[static_cast<int>(a0)] = totalCycles;
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                const uint32_t callback = cpu.gpr[6];
                const uint32_t argument = cpu.gpr[7];
                if (!rpc.dispatchSifManImport(call.ordinal, cpu))
                    return ImportDisposition::Missing;
                // sceSifSetDmaIntr: the transfer completes at once; its callback runs like the
                // DMA completion interrupt (Dbc_Manager releases its DMA semaphore there).
                if (call.ordinal == 32u && cpu.gpr[2] != 0u && callback != 0u)
                    pendingGuestCallbacks.emplace(totalCycles + kSifDmaCompletionCycles,
                                                  ScheduledGuestCallback{callback, cpu.gpr[28], argument});
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "dmacman"))
            {
                dispatchDmacImport(call.ordinal, cpu);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        bool step(CpuState &cpu)
        {
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;

            if (PcCounter *counter = pcCounter(); counter && cpu.pc >= counter->low && cpu.pc < counter->high)
            {
                ++counter->counts[(cpu.pc - counter->low) / 4u];
                if ((++counter->total & 0xFFFFFFu) == 0u)
                    counter->write();
            }
            if (const auto *traced = tracedPcs(); traced && traced->count(cpu.pc) != 0u)
            {
                // With the context: running thread, cop0 status (interrupts enabled, in a handler), I_CTRL, and the
                // depth of inline calls (RPC servers run from the EE's side), to tell who touches what when.
                const IopThread *running = kernel.currentThread();
                std::fprintf(stderr, "[iop-pc] pc=0x%x v1=0x%x v0=0x%x a0=0x%x a1=0x%x a2=0x%x a3=0x%x ra=0x%x thread=%d status=0x%x ictrl=%u calls=%u cycle=%llu\n",
                             cpu.pc, cpu.gpr[3], cpu.gpr[2], cpu.gpr[4], cpu.gpr[5], cpu.gpr[6], cpu.gpr[7], cpu.gpr[31],
                             running ? running->id : -1, cpu.cop0[12], memory.interruptControl(), callDepth,
                             static_cast<unsigned long long>(totalCycles));
            }
            if (nativeBitScan != 0u && physicalAddress(cpu.pc) == nativeBitScan && !cpu.branchPending)
            {
                runNativeBitScan(cpu);
                return true;
            }
            if (!cpu.branchPending)
                if (const native::Function function = nativeAt(physicalAddress(cpu.pc)))
                {
                    runNative(cpu, function);
                    return !cpu.stopped;
                }

            if (const auto import = imports.decode(cpu.pc))
            {
                const ImportDisposition disposition = dispatchImport(*import, cpu);
                ++totalInstructions;
                ++totalCycles;
                if (disposition == ImportDisposition::JumpToGuest)
                    return true;
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
                return !cpu.stopped;
            }

            const bool running = cpuCore.executeInstruction(cpu);
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        // Recompiled modules (ps2x/iop/iop_native.h; PS2X_IOP_NATIVE=1, read as each module loads, so a game override
        // can set it): the native entries of the loaded modules whose file hash and base match a registration, by
        // physical address, and a bit per word of RAM to check them.
        static bool nativeModules()
        {
            const char *value = std::getenv("PS2X_IOP_NATIVE");
            return value && *value == '1';
        }
        std::unordered_map<uint32_t, native::Function> nativeEntries;
        std::vector<uint64_t> nativeBits = std::vector<uint64_t>(kRamSize / 4u / 64u, 0u);
        uint64_t sliceEnd = ~0ull; // runCpu's: native code stops at its loop heads once there

        native::Function nativeAt(uint32_t physical) const
        {
            if (physical >= kRamSize || !((nativeBits[physical >> 8] >> ((physical >> 2) & 63u)) & 1u))
                return nullptr;
            const auto it = nativeEntries.find(physical);
            return it == nativeEntries.end() ? nullptr : it->second;
        }

        static uint64_t fileHash(std::span<const uint8_t> file)
        {
            uint64_t hash = 1469598103934665603ull;
            for (const uint8_t byte : file)
                hash = (hash ^ byte) * 1099511628211ull;
            return hash;
        }

        void activateNative(const Module &module, std::span<const uint8_t> file)
        {
            if (!nativeModules())
                return;
            const uint64_t hash = fileHash(file);
            for (const native::RegisteredModule &registered : native::registeredModules())
            {
                if (registered.fileHash != hash || registered.base != module.base)
                    continue;
                for (size_t i = 0; i < registered.count; ++i)
                {
                    const uint32_t physical = physicalAddress(registered.entries[i].pc);
                    nativeEntries[physical] = registered.entries[i].function;
                    nativeBits[physical >> 8] |= 1ull << ((physical >> 2) & 63u);
                }
                std::fprintf(stderr, "[IOP] native: %s (%zu entries)\n", module.name.c_str(), registered.count);
                return;
            }
        }

        void deactivateNative(const Module &module)
        {
            for (auto it = nativeEntries.begin(); it != nativeEntries.end();)
            {
                if (it->first >= physicalAddress(module.base) && it->first < physicalAddress(module.base) + module.size)
                {
                    nativeBits[it->first >> 8] &= ~(1ull << ((it->first >> 2) & 63u));
                    it = nativeEntries.erase(it);
                }
                else
                    ++it;
            }
        }

        // The guest as native code sees it: reads of RAM directly, the rest through the IOP; the IOP's clock kept up
        // before any access that is not RAM (the SPU2 and timers read it), and calls run natively where they can.
        class NativeEnv final : public native::Env
        {
        public:
            NativeEnv(Impl &impl, CpuState &cpu)
                : native::Env(*reinterpret_cast<native::NativeCpu *>(&cpu)), m_impl(impl), m_cpu(cpu)
            {
                ram = const_cast<uint8_t *>(impl.memory.ram().data());
                ramSize = static_cast<uint32_t>(impl.memory.ram().size());
            }

            void sync()
            {
                const uint64_t delta = executed - m_synced;
                m_impl.totalInstructions += delta;
                m_impl.totalCycles += delta;
                m_synced = executed;
            }

            bool call(uint32_t target, uint32_t ret) override
            {
                m_cpu.pc = target;
                for (;;)
                {
                    // A blocking import leaves the thread yielded with pc at its return address: it waits there (the
                    // caller stops at that resume point), it has not returned yet.
                    if (m_cpu.yielded || m_cpu.stopped)
                        return false;
                    if (m_cpu.pc == ret)
                        return true;
                    if (expired())
                        return false;
                    const uint32_t physical = physicalAddress(m_cpu.pc);
                    if (const native::Function function = m_impl.nativeAt(physical))
                    {
                        function(*this);
                        continue;
                    }
                    if (const auto import = m_impl.imports.decode(m_cpu.pc))
                    {
                        sync();
                        const ImportDisposition disposition = m_impl.dispatchImport(*import, m_cpu);
                        tick(1);
                        if (disposition != ImportDisposition::JumpToGuest)
                            m_cpu.pc = m_cpu.gpr[31];
                        continue;
                    }
                    return false; // interpreted code: the IOP goes on with it, and back into native code after
                }
            }

        private:
            uint8_t read8(uint32_t address) override { sync(); return m_impl.memory.read8(address); }
            uint16_t read16(uint32_t address) override { sync(); return m_impl.memory.read16(address); }
            uint32_t read32(uint32_t address) override { sync(); return m_impl.memory.read32(address); }
            void write8(uint32_t address, uint8_t value) override { sync(); m_impl.memory.write8(address, value); m_impl.schedulePendingDma(); }
            void write16(uint32_t address, uint16_t value) override { sync(); m_impl.memory.write16(address, value); m_impl.schedulePendingDma(); }
            void write32(uint32_t address, uint32_t value) override { sync(); m_impl.memory.write32(address, value); m_impl.schedulePendingDma(); }

            Impl &m_impl;
            CpuState &m_cpu;
            uint64_t m_synced = 0;
        };

        void runNative(CpuState &cpu, native::Function function)
        {
            if (cpu.pendingLoad)
            {
                if (cpu.pendingLoadReg != 0u)
                    cpu.gpr[cpu.pendingLoadReg] = cpu.pendingLoadValue;
                cpu.pendingLoad = false;
            }
            NativeEnv env(*this, cpu);
            env.budget = sliceEnd > totalInstructions ? sliceEnd - totalInstructions : 1u;
            const uint32_t entered = cpu.pc;
            function(env);
            env.sync();
            if (env.executed == 0u && cpu.pc == entered)
            {
                // No progress (not an entry after all): interpret the instruction instead.
                cpuCore.executeInstruction(cpu);
                ++totalInstructions;
                ++totalCycles;
            }
        }

        // 989snd's voice search: the first voice from a1 (of 48, 24 per SPU2 core) set in the masks at a0 + 0x18,
        // or -1. A fifth of TMB's IOP instructions in the match go through it (it divides by 24 at every voice), so
        // when a module holds it, it runs natively: same registers, cycles and instruction count as the guest code.
        static constexpr std::array<uint32_t, 28> kBitScanCode = {
            0x28a20030u, 0x10400017u, 0x3c062aaau, 0x34c6aaabu, 0x24070001u, 0x00a60018u, 0x000517c3u,
            0x00004010u, 0x00081883u, 0x00621823u, 0x00031040u, 0x00431021u, 0x000210c0u, 0x00a21023u,
            0x00031880u, 0x00831821u, 0x8c630018u, 0x00471004u, 0x00621824u, 0x14600007u, 0x00a01021u,
            0x24a50001u, 0x28a20030u, 0x1440ffeeu, 0x00a60018u, 0x03e00008u, 0x2402ffffu, 0x03e00008u};
        uint32_t nativeBitScan = 0u; // physical address of the function, 0 if no module holds it

        void findNativeFunctions(uint32_t base, uint32_t size)
        {
            for (uint32_t address = base; address + kBitScanCode.size() * 4u <= base + size; address += 4u)
            {
                bool match = true;
                for (size_t i = 0; i < kBitScanCode.size() && match; ++i)
                    match = memory.read32(address + static_cast<uint32_t>(i) * 4u) == kBitScanCode[i];
                if (match)
                {
                    nativeBitScan = physicalAddress(address);
                    return;
                }
            }
        }

        void runNativeBitScan(CpuState &cpu)
        {
            if (cpu.pendingLoad)
            {
                if (cpu.pendingLoadReg != 0u)
                    cpu.gpr[cpu.pendingLoadReg] = cpu.pendingLoadValue;
                cpu.pendingLoad = false;
            }
            // PS2X_IOP_NATIVE_CHECK=1: also interpret the guest code on a copy and compare.
            static const bool check = []
            {
                const char *value = std::getenv("PS2X_IOP_NATIVE_CHECK");
                return value && *value == '1';
            }();
            CpuState reference = cpu;
            uint64_t referenceInstructions = 0u;
            if (check)
            {
                const uint32_t returnAddress = cpu.gpr[31];
                while (reference.pc != returnAddress && referenceInstructions < 10000u && !reference.stopped)
                {
                    (void)cpuCore.executeInstruction(reference);
                    ++referenceInstructions;
                }
            }
            const uint64_t instructionsBefore = totalInstructions;
            int32_t voice = static_cast<int32_t>(cpu.gpr[5]);
            uint64_t instructions = 3u; // slti, beq, lui
            cpu.gpr[6] = 0x2AAA0000u;
            uint32_t result = 0xFFFFFFFFu;
            if (voice < 48)
            {
                cpu.gpr[6] = 0x2AAAAAABu;
                cpu.gpr[7] = 1u;
                instructions += 3u; // ori, addiu, mult
                for (;;)
                {
                    const int64_t product = static_cast<int64_t>(voice) * 0x2AAAAAABll;
                    cpu.hi = static_cast<uint32_t>(static_cast<uint64_t>(product) >> 32);
                    cpu.lo = static_cast<uint32_t>(product);
                    const int32_t quotient = (static_cast<int32_t>(cpu.hi) >> 2) - (voice >> 31);
                    const int32_t remainder = voice - quotient * 24;
                    const uint32_t mask = 1u << (static_cast<uint32_t>(remainder) & 31u);
                    const uint32_t word = memory.read32(cpu.gpr[4] + static_cast<uint32_t>(quotient) * 4u + 0x18u);
                    cpu.gpr[8] = cpu.hi;
                    cpu.gpr[3] = word & mask;
                    instructions += 15u;
                    if (cpu.gpr[3] != 0u)
                    {
                        result = static_cast<uint32_t>(voice);
                        cpu.gpr[2] = result;
                        break;
                    }
                    ++voice;
                    instructions += 4u; // addiu, slti, bne, mult
                    if (voice >= 48)
                    {
                        const int64_t last = static_cast<int64_t>(voice) * 0x2AAAAAABll; // the delay slot's mult
                        cpu.hi = static_cast<uint32_t>(static_cast<uint64_t>(last) >> 32);
                        cpu.lo = static_cast<uint32_t>(last);
                        break;
                    }
                }
                cpu.gpr[5] = static_cast<uint32_t>(voice);
            }
            cpu.gpr[2] = result;
            instructions += 2u; // jr ra and its delay slot
            cpu.pc = cpu.gpr[31];
            cpu.branchPending = false;
            totalInstructions += instructions;
            totalCycles += instructions;
            schedulePendingDma();
            if (check)
            {
                static uint64_t checks = 0u, mismatches = 0u;
                ++checks;
                const bool same = reference.gpr == cpu.gpr && reference.hi == cpu.hi && reference.lo == cpu.lo &&
                                  reference.pc == cpu.pc && referenceInstructions == totalInstructions - instructionsBefore;
                if (!same && mismatches++ < 10u)
                    std::fprintf(stderr, "[iop:native] voice search differs (a1 0x%x): v0 0x%x/0x%x, %llu/%llu instructions\n",
                                 reference.gpr[5], cpu.gpr[2], reference.gpr[2],
                                 static_cast<unsigned long long>(totalInstructions - instructionsBefore),
                                 static_cast<unsigned long long>(referenceInstructions));
                if ((checks & 0xFFFFu) == 0u)
                    std::fprintf(stderr, "[iop:native] voice search: %llu checked, %llu differ\n",
                                 static_cast<unsigned long long>(checks), static_cast<unsigned long long>(mismatches));
            }
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            const uint64_t previousSliceEnd = sliceEnd;
            sliceEnd = start + instructionBudget;
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (profiling && (totalInstructions & 63u) == 0u)
                    profileSample(cpu.pc);
                if (!step(cpu))
                    break;
                if (totalCycles >= spu2.nextEventCycle())
                    spu2.advanceTo(totalCycles);
                if (memory.interruptControl() == 0u)
                    continue;
                if (!servicingDmaInterrupts && !pendingDmaInterrupts.empty())
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty())
                    servicePendingGuestCallbacks();
            }
            activeCpu = previous;
            sliceEnd = previousSliceEnd;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        // PS2X_IOP_PROFILE=1: one PC sample every 64 instructions; every 10 s of IOP clock, the busiest code (by
        // 64-byte block, with its module) and threads (-1: inline calls such as RPC servers run from the EE).
        const bool profiling = []
        {
            const char *value = std::getenv("PS2X_IOP_PROFILE");
            return value && *value == '1';
        }();
        std::map<uint32_t, uint64_t> profileBlocks;
        std::map<int, uint64_t> profileThreads;
        uint64_t profileSamples = 0;
        uint64_t profileNextReport = 0;

        void profileSample(uint32_t pc)
        {
            ++profileBlocks[pc & ~63u];
            const IopThread *thread = kernel.currentThread();
            ++profileThreads[callDepth != 0u || !thread ? -1 : thread->id];
            ++profileSamples;
            if (totalCycles < profileNextReport)
                return;
            profileNextReport = totalCycles + 36'864'000ull * 10ull;
            std::vector<std::pair<uint64_t, uint32_t>> top;
            for (const auto &[block, count] : profileBlocks)
                top.emplace_back(count, block);
            std::sort(top.rbegin(), top.rend());
            std::fprintf(stderr, "[iop:profile] %llu samples (x64 instructions); threads:", static_cast<unsigned long long>(profileSamples));
            for (const auto &[id, count] : profileThreads)
                std::fprintf(stderr, " %d:%.1f%%", id, 100.0 * count / profileSamples);
            std::fputc('\n', stderr);
            for (size_t i = 0; i < top.size() && i < 25u; ++i)
            {
                const uint32_t physical = top[i].second & 0x1FFFFFu;
                const Module *owner = nullptr;
                for (const auto &[id, module] : modules)
                    if (physical >= module.base && physical < module.base + module.size)
                        owner = &module;
                std::fprintf(stderr, "[iop:profile]   %5.1f%% 0x%06x %s+0x%x\n", 100.0 * top[i].first / profileSamples,
                             top[i].second, owner ? owner->name.c_str() : "?", owner ? physical - owner->base : physical);
            }
            profileBlocks.clear();
            profileThreads.clear();
            profileSamples = 0;
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (const uint32_t slot = callStackSlot(depth); slot < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - slot * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            const uint32_t executed = runCpu(cpu, budget);
            if (!cpu.stopped)
            {
                // The function did not return: it blocked (an inline call cannot wait for other
                // threads) or ran out of budget. Its caller receives whatever v0 holds.
                static const bool trace = []
                {
                    const char *value = std::getenv("PS2X_IOP_TRACE_RPC");
                    return value && *value == '1';
                }();
                if (trace)
                    std::fprintf(stderr, "[iop:call] 0x%x did not return: %s after %u instructions, pc=0x%x ra=0x%x v0=0x%x\n",
                                 address, cpu.yielded ? "blocked" : "budget exhausted", executed, cpu.pc, cpu.gpr[31],
                                 cpu.gpr[2]);
            }
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        [[nodiscard]] bool threadRunnable(const IopThread &thread) const noexcept
        {
            return thread.state == IopThreadState::Ready || thread.state == IopThreadState::Running ||
                   (thread.state == IopThreadState::Delay && thread.wakeCycle <= totalCycles);
        }

        // Run the other threads (and the interrupt/timer/SIF services) until `owner` is runnable
        // again or `deadline` passes. Never schedules `owner` itself: its real CPU context sits in
        // sceSifRpcLoop while the RPC function runs on a separate call frame.
        // Pending events, then one slice of the next ready thread (or time up to the next event if none is): what
        // the other threads get while an inline call keeps running.
        void runOthersSlice(uint64_t deadline)
        {
            IopThread *next = resumeMaskedThread();
            uint64_t nextDelayWake = UINT64_MAX;
            if (!next)
            {
                spu2.advanceTo(totalCycles);
                servicePendingDmaInterrupts();
                servicePendingSifCommands();
                servicePendingGuestCallbacks();
                timrman.serviceDue(totalCycles, *this);
                vblank.serviceDue(totalCycles, *this);
                next = kernel.beginNextReady(totalCycles, &nextDelayWake);
            }
            if (!next)
            {
                uint64_t nextWake = std::min(deadline, nextDelayWake);
                for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                    nextWake = std::min(nextWake, completionCycle);
                if (!pendingGuestCallbacks.empty())
                    nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                nextWake = timrman.nextEventCycle(nextWake);
                nextWake = vblank.nextEventCycle(nextWake);
                nextWake = std::min(nextWake, spu2.nextEventCycle());
                // Only so far: the polling thread may be waiting for time alone.
                totalCycles = std::max(totalCycles + 1u, std::min(nextWake, totalCycles + kDefaultSlice));
                return;
            }
            const uint64_t before = totalCycles;
            runCpu(next->cpu, kDefaultSlice);
            endTimeslice(*next);
            if (totalCycles == before)
                ++totalCycles;
        }

        // The Ready threads an inline call as `owner` must wait for (executeGuestFunctionAsThread): a thread that ended
        // its slice with interrupts disabled, then those at owner's priority or above, until they block. Bounded:
        // a thread that never blocks would starve the server on the IOP too, but must not stall the EE.
        void runThreadsAhead(const IopThread &owner, uint64_t deadline)
        {
            constexpr uint32_t kMaxSlicesAhead = 4096u; // ~1M IOP cycles
            for (uint32_t slices = 0; slices < kMaxSlicesAhead && totalCycles < deadline; ++slices)
            {
                IopThread *ahead = resumeMaskedThread();
                if (!ahead)
                {
                    ahead = kernel.readyAtOrAbove(owner.priority, owner.id);
                    if (!ahead)
                        return;
                    static uint32_t reported = 0u;
                    if (slices == 0u && reported++ < 5u)
                        std::fprintf(stderr, "[iop:call] thread %d (priority %u) runs before an inline call as thread %d (priority %u)\n",
                                     ahead->id, ahead->priority, owner.id, owner.priority);
                    kernel.beginThread(*ahead);
                }
                const uint64_t before = totalCycles;
                runCpu(ahead->cpu, kDefaultSlice);
                endTimeslice(*ahead);
                if (totalCycles == before)
                    ++totalCycles;
            }
        }

        [[nodiscard]] bool runOthersUntilRunnable(IopThread &owner, uint64_t deadline)
        {
            while (!threadRunnable(owner))
            {
                if (totalCycles >= deadline)
                    return false;
                IopThread *next = resumeMaskedThread();
                uint64_t nextDelayWake = UINT64_MAX;
                if (!next)
                {
                    spu2.advanceTo(totalCycles);
                    servicePendingDmaInterrupts();
                    servicePendingSifCommands();
                    servicePendingGuestCallbacks();
                    timrman.serviceDue(totalCycles, *this);
                    vblank.serviceDue(totalCycles, *this);
                    if (threadRunnable(owner))
                        break;
                    next = kernel.beginNextReady(totalCycles, &nextDelayWake);
                }
                if (!next)
                {
                    uint64_t nextWake = std::min(deadline, nextDelayWake);
                    for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                        nextWake = std::min(nextWake, completionCycle);
                    if (!pendingGuestCallbacks.empty())
                        nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                    nextWake = timrman.nextEventCycle(nextWake);
                    nextWake = vblank.nextEventCycle(nextWake);
                    nextWake = std::min(nextWake, spu2.nextEventCycle());
                    totalCycles = std::max(totalCycles + 1u, nextWake);
                    continue;
                }
                const uint64_t before = totalCycles;
                runCpu(next->cpu, kDefaultSlice);
                endTimeslice(*next);
                if (totalCycles == before)
                    ++totalCycles;
            }
            return true;
        }

        // Instructions an RPC server runs before the other threads get a slice (PS2X_IOP_CALL_BUDGET, for tests).
        const uint32_t serverCallBudget = []
        {
            const char *value = std::getenv("PS2X_IOP_CALL_BUDGET");
            return value && *value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : kMaxCallInstructions;
        }();
        // A thread that ends its slice with interrupts disabled keeps the CPU: no interrupt can be taken
        // to wake another thread before it re-enables them.
        IopThread *resumeMaskedThread()
        {
            const int id = maskedThreadId;
            maskedThreadId = 0;
            if (id == 0 || memory.interruptControl() != 0u)
                return nullptr;
            IopThread *thread = kernel.thread(id);
            if (thread == nullptr || thread->state != IopThreadState::Ready)
                return nullptr;
            kernel.beginThread(*thread);
            return thread;
        }

        void endTimeslice(IopThread &thread)
        {
            const int id = thread.id;
            if (thread.cpu.stopped && thread.cpu.pc == kCallReturnSentinel)
            {
                if (auto it = handedOffServers.find(id); it != handedOffServers.end())
                {
                    // The server handed off to this thread (executeGuestFunctionAsThreadDeferrable) returned: the thread
                    // goes back to its sceSifRpcLoop wait, then the deferred call completes with its v0.
                    HandedOffServer server = std::move(it->second);
                    handedOffServers.erase(it);
                    const uint32_t v0 = thread.cpu.gpr[2];
                    thread.cpu = server.rpcLoopCpu;
                    thread.state = server.rpcLoopState;
                    reservedCallStacks.erase(server.stackSlot);
                    kernel.endTimeslice(thread, kThreadReturnSentinel);
                    maskedThreadId = 0;
                    if (server.onReturn)
                        server.onReturn(v0);
                    return;
                }
            }
            kernel.endTimeslice(thread, kThreadReturnSentinel);
            const IopThread *after = kernel.thread(id);
            maskedThreadId = memory.interruptControl() == 0u && after != nullptr && after->state == IopThreadState::Ready ? id : 0;
        }

        uint32_t executeGuestFunctionAsThread(uint32_t address,
                                              uint32_t a0,
                                              uint32_t a1,
                                              uint32_t a2,
                                              uint32_t a3,
                                              uint32_t gp,
                                              int threadId) override
        {
            bool deferred = false;
            return executeGuestFunctionAsThreadDeferrable(address, a0, a1, a2, a3, gp, threadId, nullptr, deferred);
        }

        uint32_t executeGuestFunctionAsThreadDeferrable(uint32_t address,
                                                        uint32_t a0,
                                                        uint32_t a1,
                                                        uint32_t a2,
                                                        uint32_t a3,
                                                        uint32_t gp,
                                                        int threadId,
                                                        std::function<void(uint32_t)> onReturn,
                                                        bool &deferred) override
        {
            deferred = false;
            IopThread *owner = kernel.thread(threadId);
            if (owner == nullptr || owner == kernel.currentThread())
                return callFunction(address, a0, a1, a2, a3, gp);
            if (onReturn && handedOffServers.count(threadId) != 0u)
            {
                // Its thread still runs a server handed off earlier (on the IOP the request would wait in the queue):
                // this one runs inline as before rather than taking over the thread's context.
                std::fprintf(stderr, "[iop:call] 0x%x as thread %d: the thread still serves a handed-off call, running inline\n",
                             address, threadId);
                onReturn = nullptr;
            }

            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };
            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            const uint32_t slot = callStackSlot(depth);
            cpu.gpr[29] = (slot < kCallStackCapacity ? kCallStackLimit - slot * kCallStackSize : kCallStackBase) - 32u;
            cpu.gpr[31] = kCallReturnSentinel;

            // Bound the wait in IOP time so a server that never gets woken cannot hang the EE.
            constexpr uint64_t kMaxBlockedCycles = 36'864'000ull * 20ull; // 20 s of IOP clock
            const uint64_t deadline = totalCycles + kMaxBlockedCycles;
            // On the IOP the server's thread only gets the CPU once the Ready threads that outrank it or tie with it
            // have blocked: no thread preempts one of equal or higher priority. The inline call used to run at once,
            // in the middle of whatever such a thread was doing when its slice ended: 989snd's voice thread was
            // unlinking a voice when its command server, called inline, appended it again; the active-voice list
            // looped on itself and the game froze in snd_SendIOPCommandNoWait.
            runThreadsAhead(*owner, deadline);
            IopThread *previous = kernel.currentThread();
            const IopThreadState ownerState = owner->state;
            uint32_t blocks = 0u;
            bool timedOut = false;
            kernel.setCurrentThread(owner);
            owner->state = IopThreadState::Running;
            uint32_t preemptions = 0u;
            // A server that keeps running for hundreds of slices is stuck: where its pc spends the time, and the
            // return addresses on its stack (the call chain), once, to see what it waits for.
            std::map<uint32_t, uint32_t> stuckPcs;
            const auto moduleOf = [&](uint32_t pc) -> std::pair<const Module *, uint32_t>
            {
                const uint32_t physical = pc & 0x1FFFFFu;
                for (const auto &[id, module] : modules)
                    if (physical >= module.base && physical < module.base + module.size)
                        return {&module, physical - module.base};
                return {nullptr, physical};
            };
            for (;;)
            {
                runCpu(cpu, preemptions == 0u ? serverCallBudget : kDefaultSlice);
                if (cpu.stopped)
                    break;
                if (preemptions >= 200u && preemptions < 1200u)
                    ++stuckPcs[cpu.pc];
                if (preemptions == 1200u)
                {
                    std::fprintf(stderr, "[iop:call] 0x%x as thread %d stuck (a0 0x%x a1 0x%x a2 0x%x a3 0x%x): pc 0x%x ra 0x%x sp 0x%x\n",
                                 address, threadId, a0, a1, a2, a3, cpu.pc, cpu.gpr[31], cpu.gpr[29]);
                    std::vector<std::pair<uint32_t, uint32_t>> hot(stuckPcs.begin(), stuckPcs.end());
                    std::sort(hot.begin(), hot.end(), [](const auto &l, const auto &r) { return l.second > r.second; });
                    for (size_t i = 0; i < hot.size() && i < 12u; ++i)
                    {
                        const auto [module, offset] = moduleOf(hot[i].first);
                        std::fprintf(stderr, "[iop:call]   pc 0x%x (%s+0x%x) at the end of %u of 1000 slices\n", hot[i].first,
                                     module ? module->name.c_str() : "?", offset, hot[i].second);
                    }
                    std::fprintf(stderr, "[iop:call]   code addresses on its stack:");
                    for (uint32_t at = cpu.gpr[29]; at < cpu.gpr[29] + 0x400u; at += 4u)
                    {
                        const uint32_t word = memory.read32(at);
                        const auto [module, offset] = moduleOf(word);
                        if (module && (word & 3u) == 0u)
                            std::fprintf(stderr, " [sp+0x%x] %s+0x%x", at - cpu.gpr[29], module->name.c_str(), offset);
                    }
                    std::fputc('\n', stderr);
                    std::fprintf(stderr, "[iop:call]   registers:");
                    for (int r = 1; r < 32; ++r)
                        std::fprintf(stderr, " r%d=0x%x", r, cpu.gpr[r]);
                    std::fputc('\n', stderr);
                }
                if (!cpu.yielded)
                {
                    // A whole budget without blocking: the server polls for something another thread or an event
                    // provides (989snd's command server, now and then while loading). Abandoning it there left
                    // its reply unwritten and the game waiting for it forever; give the others a slice and go on.
                    if (preemptions++ == 0u)
                    {
                        const uint32_t physical = cpu.pc & 0x1FFFFFu;
                        const Module *where = nullptr;
                        for (const auto &[id, module] : modules)
                            if (physical >= module.base && physical < module.base + module.size)
                                where = &module;
                        std::fprintf(stderr, "[iop:call] 0x%x as thread %d still running after %u instructions, at "
                                             "0x%x (%s+0x%x): letting the other threads run\n",
                                     address, threadId, serverCallBudget, cpu.pc, where ? where->name.c_str() : "?",
                                     where ? physical - where->base : physical);
                        std::fprintf(stderr, "[iop:call]   cycle %llu, servicing DMA interrupts %d, I_STAT 0x%x I_MASK 0x%x "
                                             "I_CTRL 0x%x, pending DMA interrupts:",
                                     static_cast<unsigned long long>(totalCycles), servicingDmaInterrupts ? 1 : 0,
                                     memory.interruptStatus(), memory.interruptMask(), memory.interruptControl());
                        for (const auto &[irq, completion] : pendingDmaInterrupts)
                            std::fprintf(stderr, " irq %d at %llu", irq, static_cast<unsigned long long>(completion));
                        std::fputc('\n', stderr);
                        spu2.printDmaHistory();
                    }
                    if (totalCycles >= deadline)
                        break;
                    owner->state = ownerState;
                    runOthersSlice(deadline);
                    kernel.setCurrentThread(owner);
                    owner->state = IopThreadState::Running;
                    continue;
                }
                cpu.yielded = false;
                ++blocks;
                if (onReturn && slot < kCallStackCapacity)
                {
                    // Hand off: the server goes on as its thread from where it blocked, in IOP time, with the state its
                    // wait set; the thread's sceSifRpcLoop context is kept to go back to when the server returns
                    // (endTimeslice). Its call stack stays reserved meanwhile.
                    handedOffServers.emplace(threadId, HandedOffServer{owner->cpu, ownerState, slot, std::move(onReturn)});
                    reservedCallStacks.insert(slot);
                    owner->cpu = cpu;
                    kernel.setCurrentThread(previous);
                    deferred = true;
                    return 0u;
                }
                if (!runOthersUntilRunnable(*owner, deadline))
                {
                    timedOut = true;
                    break;
                }
                kernel.setCurrentThread(owner);
                owner->state = IopThreadState::Running;
            }
            owner->state = ownerState; // back to its sceSifRpcLoop wait
            kernel.setCurrentThread(previous);
            lastThreadCallReturned = cpu.stopped;

            static const bool trace = []
            {
                const char *value = std::getenv("PS2X_IOP_TRACE_RPC");
                return value && *value == '1';
            }();
            if (preemptions != 0u)
                std::fprintf(stderr, "[iop:call] 0x%x as thread %d: %s after %u slice(s) for the others\n", address,
                             threadId, cpu.stopped ? "returned" : "gave up", preemptions);
            if (trace && (blocks != 0u || !cpu.stopped))
                std::fprintf(stderr, "[iop:call] 0x%x as thread %d: %s after %u block(s), v0=0x%x\n", address, threadId,
                             cpu.stopped ? "returned" : (timedOut ? "gave up waiting" : "budget exhausted"), blocks,
                             cpu.gpr[2]);
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            if (servicingDmaInterrupts || pendingDmaInterrupts.empty())
                return;

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            try
            {
                for (const int irq : completed)
                {
                    // Disabled (libsd disables a DMA interrupt while it sets the next transfer up): it stays pending
                    // until EnableIntr, as on the hardware. Dropping it lost 989snd's transfer callback now and then,
                    // and its command server waited for it forever (the game hung in snd_SendIOPCommandNoWait).
                    if (intrman.heldBack(irq))
                    {
                        static uint32_t reported = 0u;
                        if (reported++ < 5u)
                            std::fprintf(stderr, "[iop] irq %d came while disabled: held until EnableIntr\n", irq);
                        heldInterrupts.insert(irq);
                        continue;
                    }
                    if (traceDma)
                        std::fprintf(stderr, "[iop:dma] irq %d dispatched at %llu\n", irq, static_cast<unsigned long long>(totalCycles));
                    (void)intrman.dispatchInterrupt(irq, *this);
                }
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void servicePendingGuestCallbacks()
        {
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty())
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        // dmacman: only the SIO2 channels move data (dmac_request stores the transfer,
        // dmac_transfer runs it); the other calls succeed without effect.
        void dispatchDmacImport(uint16_t ordinal, CpuState &cpu)
        {
            const uint32_t channel = cpu.gpr[4];
            switch (ordinal)
            {
            case 28: // dmac_request(channel, address, words per block, blocks, direction)
                if (channel < dmaRequests.size())
                {
                    const uint32_t blocks = std::max<uint32_t>(cpu.gpr[7], 1u);
                    dmaRequests[channel] = {cpu.gpr[5], cpu.gpr[6] * blocks * 4u};
                }
                cpu.gpr[2] = 1u;
                return;
            case 32: // dmac_transfer(channel)
                if (channel == IopSio2::DmaInChannel)
                    sio2.dmaToDevice(dmaRequests[channel].address, dmaRequests[channel].bytes);
                else if (channel == IopSio2::DmaOutChannel)
                    sio2.armDmaFromDevice(dmaRequests[channel].address, dmaRequests[channel].bytes);
                cpu.gpr[2] = 0u;
                return;
            default:
                cpu.gpr[2] = 0u;
                return;
            }
        }

        bool queueSifCommand(uint32_t cid, const void *packet, uint32_t size)
        {
            if (!packet || size < 16u || size > kMaxSifCommandPacket)
                return false;
            // SET_SREG is served by the IOP kernel's sifcmd itself: header, then the register and its value.
            if (cid == IopRpcBridge::kSifCmdSetSreg && size >= 24u)
            {
                uint32_t reg = 0u;
                uint32_t value = 0u;
                std::memcpy(&reg, static_cast<const uint8_t *>(packet) + 16u, sizeof(reg));
                std::memcpy(&value, static_cast<const uint8_t *>(packet) + 20u, sizeof(value));
                rpc.setSreg(reg, value);
                return true;
            }
            const auto handler = rpc.cmdHandler(cid);
            if (!handler)
                return false;
            PendingSifCommand command{*handler, {}, size};
            std::memcpy(command.packet.data(), packet, size);
            pendingSifCommands.push_back(command);
            return true;
        }

        // Runs the handlers like the SIF command interrupt does: in order, outside any thread,
        // with the packet copied to an IOP receive buffer.
        void servicePendingSifCommands()
        {
            if (servicingSifCommands || pendingSifCommands.empty())
                return;
            if (sifCommandBuffer == 0u)
                sifCommandBuffer = memory.allocate(kMaxSifCommandPacket, 16u);
            if (sifCommandBuffer == 0u)
            {
                log(LogLevel::Error, "[IOP] no memory for the SIF command receive buffer");
                pendingSifCommands.clear();
                return;
            }
            servicingSifCommands = true;
            try
            {
                while (!pendingSifCommands.empty())
                {
                    const PendingSifCommand command = pendingSifCommands.front();
                    pendingSifCommands.pop_front();
                    if (memory.writeRam(sifCommandBuffer, command.packet.data(), command.size))
                    {
                        (void)callFunction(command.handler.function, sifCommandBuffer, command.handler.data, 0u, 0u,
                                           command.handler.gp, kSifCommandHandlerBudget);
                    }
                }
            }
            catch (...)
            {
                servicingSifCommands = false;
                throw;
            }
            servicingSifCommands = false;
        }

        void runCycles(uint64_t cycles) noexcept
        {
            try
            {
                paidCycles += cycles;
                if (totalCycles > paidCycles + kMaxLeadCycles)
                    paidCycles = totalCycles - kMaxLeadCycles;
                const uint64_t target = paidCycles;
                while (totalCycles < target)
                {
                    IopThread *next = resumeMaskedThread();
                    uint64_t nextDelayWake = UINT64_MAX;
                    if (!next)
                    {
                        spu2.advanceTo(totalCycles);
                        servicePendingDmaInterrupts();
                        servicePendingSifCommands();
                        servicePendingGuestCallbacks();
                        timrman.serviceDue(totalCycles, *this);
                        vblank.serviceDue(totalCycles, *this);
                        next = kernel.beginNextReady(totalCycles, &nextDelayWake);
                    }
                    if (!next)
                    {
                        uint64_t nextWake = std::min(target, nextDelayWake);
                        for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                            nextWake = std::min(nextWake, completionCycle);
                        if (!pendingGuestCallbacks.empty())
                            nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                        nextWake = timrman.nextEventCycle(nextWake);
                        nextWake = vblank.nextEventCycle(nextWake);
                        nextWake = std::min(nextWake, spu2.nextEventCycle());
                        totalCycles = std::max(totalCycles + 1u, std::min(target, nextWake));
                        continue;
                    }
                    const uint64_t before = totalCycles;
                    runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
                    endTimeslice(*next);
                    if (totalCycles == before)
                        ++totalCycles;
                }
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        // PS2X_IOP_DUMP_MODULES=<folder>: each module as loaded, relocated at its base and before its start routine
        // runs, as <name>_<base>.bin, with <name>_<base>.json (base, size, entry, gp, the file's FNV-1a 64 hash): the
        // input of tools/iop_recomp.py, which recompiles IRX code to native C++.
        void dumpModule(const Module &module, std::span<const uint8_t> file) const
        {
            static const char *s_dir = std::getenv("PS2X_IOP_DUMP_MODULES");
            if (!s_dir || !*s_dir || module.size == 0u)
                return;
            uint64_t hash = 1469598103934665603ull;
            for (const uint8_t byte : file)
                hash = (hash ^ byte) * 1099511628211ull;
            std::vector<uint8_t> bytes(module.size);
            if (!memory.readRam(module.base, bytes.data(), bytes.size()))
                return;
            char stem[512];
            std::snprintf(stem, sizeof(stem), "%s/%s_%06x", s_dir, module.name.c_str(), module.base);
            if (FILE *out = std::fopen((std::string(stem) + ".bin").c_str(), "wb"))
            {
                std::fwrite(bytes.data(), 1, bytes.size(), out);
                std::fclose(out);
            }
            if (FILE *out = std::fopen((std::string(stem) + ".json").c_str(), "w"))
            {
                std::fprintf(out, "{\"name\": \"%s\", \"base\": %u, \"size\": %u, \"entry\": %u, \"gp\": %u, \"hash\": \"%016llx\"}\n",
                             module.name.c_str(), module.base, module.size, module.entry, module.gp,
                             static_cast<unsigned long long>(hash));
                std::fclose(out);
            }
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;
            if (nativeBitScan == 0u)
                findNativeFunctions(module.base, module.size);
            dumpModule(module, image);
            activateNative(module, image);

            // Like loadcore: start(argc, argv) with argv[0] the module path, then one entry per
            // NUL-separated argument. The block stays allocated, as resident modules may keep argv strings.
            std::vector<std::string> argv{module.path};
            if (arguments && argumentSize)
            {
                const char *text = static_cast<const char *>(arguments);
                for (uint32_t start = 0u; start < argumentSize;)
                {
                    uint32_t end = start;
                    while (end < argumentSize && text[end] != '\0')
                        ++end;
                    if (end > start)
                        argv.emplace_back(text + start, end - start);
                    start = end + 1u;
                }
            }
            const uint32_t pointerBytes = static_cast<uint32_t>(argv.size() + 1u) * 4u;
            uint32_t stringBytes = 0u;
            for (const std::string &arg : argv)
                stringBytes += static_cast<uint32_t>(arg.size()) + 1u;
            const uint32_t argvBlock = allocate(pointerBytes + stringBytes, 16u);
            if (argvBlock)
            {
                uint32_t cursor = argvBlock + pointerBytes;
                for (size_t index = 0; index < argv.size(); ++index)
                {
                    write32(argvBlock + static_cast<uint32_t>(index) * 4u, cursor);
                    writeRam(cursor, argv[index].c_str(), argv[index].size() + 1u);
                    cursor += static_cast<uint32_t>(argv[index].size()) + 1u;
                }
                write32(argvBlock + static_cast<uint32_t>(argv.size()) * 4u, 0u);
            }
            const uint32_t argc = argvBlock ? static_cast<uint32_t>(argv.size()) : 0u;
            // Start routines run in MODLOAD's thread on the IOP and may block on what they started (LIBSD waiting
            // for its first SPU2 transfers); priority 1 keeps them running before any other thread, as before.
            constexpr uint32_t kNoResidentEnd = 1u;
            (void)kernel.hostThread(IopKernel::kModuleLoaderThreadId, 1u);
            lastThreadCallReturned = true;
            uint32_t startResult = executeGuestFunctionAsThread(module.entry, argc, argvBlock, 0u, 0u, module.gp,
                                                                IopKernel::kModuleLoaderThreadId);
            if (!lastThreadCallReturned)
            {
                std::fprintf(stderr, "[IOP] %s: start routine abandoned before it returned, not resident\n",
                             module.name.c_str());
                startResult = kNoResidentEnd;
            }
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromGuest(host, guestAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBufferFromIop(uint32_t iopAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromIop(memory, iopAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "iop-buffer@0x" << std::hex << iopAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            if (nativeBitScan >= physicalAddress(it->second.base) &&
                nativeBitScan < physicalAddress(it->second.base) + it->second.size)
                nativeBitScan = 0u;
            deactivateNative(it->second);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::set<int> heldInterrupts; // came while disabled, delivered at EnableIntr
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        IopSio2 sio2;
        IopSpu2 spu2;
        struct DmaRequest
        {
            uint32_t address = 0u;
            uint32_t bytes = 0u;
        };
        std::array<DmaRequest, 16> dmaRequests{};
        std::deque<PendingSifCommand> pendingSifCommands;
        uint32_t sifCommandBuffer = 0u;
        bool servicingSifCommands = false;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t totalCycles = 0;
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        uint64_t paidCycles = 0;
        static constexpr uint64_t kMaxLeadCycles = 36'864'000ull / 25u;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        uint32_t callDepth = 0u;
        bool lastThreadCallReturned = true;

        // EE -> IOP servers handed off to their queue thread (executeGuestFunctionAsThreadDeferrable), by thread id.
        struct HandedOffServer
        {
            CpuState rpcLoopCpu;               // the thread's sceSifRpcLoop context, restored when the server returns
            IopThreadState rpcLoopState = IopThreadState::Sleep;
            uint32_t stackSlot = 0u;           // the call stack the server runs on, kept until it returns
            std::function<void(uint32_t)> onReturn;
        };
        std::map<int, HandedOffServer> handedOffServers;
        std::set<uint32_t> reservedCallStacks;

        // The call stack of an inline call at `depth`: the depth-th slot not kept by a handed-off server (slot ==
        // depth when none is). kCallStackCapacity: none left.
        uint32_t callStackSlot(uint32_t depth) const
        {
            uint32_t free = 0u;
            for (uint32_t slot = 0u; slot < kCallStackCapacity; ++slot)
            {
                if (reservedCallStacks.count(slot) != 0u)
                    continue;
                if (free++ == depth)
                    return slot;
            }
            return kCallStackCapacity;
        }
        int maskedThreadId = 0;
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopEmulator::~IopEmulator() = default;

    void IopEmulator::reset()
    {
        m_impl->reset();
    }

    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBufferFromIop(uint32_t iopAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBufferFromIop(iopAddress, arguments, argumentSize);
    }

    bool IopEmulator::sendSifCommand(uint32_t cid, const void *packet, uint32_t size)
    {
        return m_impl->queueSifCommand(cid, packet, size);
    }

    void IopEmulator::completeEeRpc(uint64_t token, const void *data, size_t size)
    {
        // The end function runs after the copy, at the IOP's next scheduling point, like a SIF DMA completion.
        if (const auto end = m_impl->rpc.completeEeRpc(token, data, size))
            m_impl->pendingGuestCallbacks.emplace(m_impl->totalCycles,
                                                  Impl::ScheduledGuestCallback{end->function, end->gp, end->argument});
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        // The EE accounts a few dozen cycles per checkpoint: running the IOP scheduler that
        // often cost a third of the EE thread. Batch 256 IOP cycles (about 7 us) per run.
        constexpr uint64_t kRunBatchEeCycles = 8u * 256u;
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        if (total < kRunBatchEeCycles)
        {
            m_impl->eeCycleCarry = total;
            return;
        }
        m_impl->eeCycleCarry = total % 8u;
        m_impl->runCycles(total / 8u);
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request)
    {
        return m_impl->rpc.handleRpc(request, *m_impl);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    std::vector<std::string> IopEmulator::describeThreads() const
    {
        return m_impl->kernel.describeThreads();
    }

    std::vector<std::string> IopEmulator::describeSpu2() const
    {
        return m_impl->spu2.describe();
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}
