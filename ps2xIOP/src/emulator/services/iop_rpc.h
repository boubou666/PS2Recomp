#pragma once

#include "ps2x/iop/iop_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopKernel;
    class IopMemory;

    class IopGuestExecutor
    {
    public:
        virtual ~IopGuestExecutor() = default;

        [[nodiscard]] virtual uint32_t executeGuestFunction(uint32_t address,
                                                            uint32_t a0,
                                                            uint32_t a1,
                                                            uint32_t a2,
                                                            uint32_t a3,
                                                            uint32_t gp) = 0;
        [[nodiscard]] virtual uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                                      uint32_t a0,
                                                                      uint32_t a1,
                                                                      uint32_t a2,
                                                                      uint32_t a3,
                                                                      uint32_t gp,
                                                                      uint32_t instructionBudget)
        {
            return executeGuestFunction(address, a0, a1, a2, a3, gp);
        }
        // Runs `address` as thread `threadId` (the RPC server thread of a sceSifRpcLoop): the code
        // sees that thread as current and may block on it (SleepThread, WaitSema...); other
        // threads run until it is woken. Falls back to executeGuestFunction for unknown threads.
        [[nodiscard]] virtual uint32_t executeGuestFunctionAsThread(uint32_t address,
                                                                    uint32_t a0,
                                                                    uint32_t a1,
                                                                    uint32_t a2,
                                                                    uint32_t a3,
                                                                    uint32_t gp,
                                                                    int threadId)
        {
            (void)threadId;
            return executeGuestFunction(address, a0, a1, a2, a3, gp);
        }
    };

    class IopRpcBridge
    {
    public:
        IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept;

        void reset();
        [[nodiscard]] bool dispatchSifManImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] bool dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request, IopGuestExecutor &executor);
        void onSifTransfer(const SifTransfer &transfer);
        void removeServersInRange(uint32_t base, uint32_t size);

        [[nodiscard]] bool hasServer(uint32_t sid) const noexcept;
        [[nodiscard]] size_t serverCount() const noexcept { return m_servers.size(); }

        // IOP-side SIF command handler installed by sceSifAddCmdHandler: called as
        // function(packet, data) when the EE sends that command id.
        struct CmdHandler
        {
            uint32_t function = 0;
            uint32_t data = 0;
            uint32_t gp = 0;
        };
        [[nodiscard]] std::optional<CmdHandler> cmdHandler(uint32_t cid) const;

        // The IOP's 32 SIF software registers: sceSifGetSreg/sceSifSetSreg, and the EE's SET_SREG system command
        // (0x80000001), which EA's AUDIOPF.IRX waits on to learn its EE buffer addresses.
        static constexpr uint32_t kSifCmdSetSreg = 0x80000001u;
        void setSreg(uint32_t reg, uint32_t value) noexcept;

        // IOP -> EE RPC: the reply to a call this bridge handed to the host (sceSifCallRpc on an EE server). Copies it
        // to the IRX's receive buffer and wakes a waiting caller; returns the IRX's end function, to run after the copy.
        struct EndFunction
        {
            uint32_t function = 0;
            uint32_t gp = 0;
            uint32_t argument = 0;
        };
        [[nodiscard]] std::optional<EndFunction> completeEeRpc(uint64_t token, const void *data, size_t size);
        [[nodiscard]] size_t pendingEeRpcCount() const noexcept { return m_eeCalls.size(); }

    private:
        struct RpcServer
        {
            uint32_t sid = 0;
            uint32_t serverData = 0;
            uint32_t function = 0;
            uint32_t gp = 0;
            uint32_t buffer = 0;
            uint32_t callback = 0;
            uint32_t callbackBuffer = 0;
            uint32_t queue = 0;
        };

        IopHost &m_host;
        IopMemory &m_memory;
        IopKernel &m_kernel;
        std::unordered_map<uint32_t, RpcServer> m_servers;
        std::unordered_map<uint32_t, CmdHandler> m_cmdHandlers;
        std::unordered_map<uint32_t, int> m_queueThreads; // sceSifSetRpcQueue: queue -> server thread

        // IOP -> EE RPC: clients bound to an EE sid, and the calls handed to the host until the EE replies.
        struct EeCall
        {
            int thread = 0; // the IOP thread asleep in sceSifCallRpc; 0 for SIF_RPC_M_NOWAIT
            uint32_t receive = 0;
            uint32_t receiveSize = 0;
            uint32_t endFunction = 0;
            uint32_t endParam = 0;
            uint32_t gp = 0;
        };
        std::unordered_map<uint32_t, uint32_t> m_eeClients; // client address -> EE sid
        std::unordered_map<uint64_t, EeCall> m_eeCalls;     // token -> call
        // The wake cycle of a caller waiting for the EE reply: a delay that never expires.
        static constexpr uint64_t kEeRpcWaitCycle = ~uint64_t{0};
        uint64_t m_nextEeCallToken = 1u; // never reset: a reply from before an IOP reset must not match a new call
        bool bindEeRpc(IopCpuState &cpu);
        bool callEeRpc(IopCpuState &cpu);
        uint32_t m_nextDmaId = 1u;
        bool m_sifInitialized = false;
        std::array<uint32_t, 32> m_sregs{};
    };
}
