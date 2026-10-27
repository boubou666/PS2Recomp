#include "iop_rpc.h"

#include "../core/iop_cpu.h"
#include "../core/iop_kernel.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace ps2x::iop::detail
{
    IopRpcBridge::IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept
        : m_host(host), m_memory(memory), m_kernel(kernel)
    {
    }

    void IopRpcBridge::reset()
    {
        m_servers.clear();
        m_cmdHandlers.clear();
        m_queueThreads.clear();
        m_eeClients.clear();
        m_eeCalls.clear();
        m_nextEeCallToken = 1u;
        m_nextDmaId = 1u;
        m_sifInitialized = false;
        m_sregs.fill(0u);
    }

    void IopRpcBridge::setSreg(uint32_t reg, uint32_t value) noexcept
    {
        if (reg < m_sregs.size())
            m_sregs[reg] = value;
    }

    bool IopRpcBridge::dispatchSifManImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 4: // sceSifDma2Init
        case 5: // sceSifInit
            m_sifInitialized = true;
            setV0(0u);
            return true;
        case 7:  // sceSifSetDma(descriptors, count)
        case 32: // sceSifSetDmaIntr(descriptors, count, callback, argument): the emulator calls
                 // the completion callback (see IopEmulator's sifman dispatch)
        {
            constexpr uint32_t kDescriptorSize = 16u;
            constexpr uint32_t kMaxDescriptors = 32u;
            const uint32_t descriptorAddress = cpu.gpr[4];
            const uint32_t descriptorCount = cpu.gpr[5];
            if (descriptorAddress == 0u || descriptorCount == 0u || descriptorCount > kMaxDescriptors)
            {
                setV0(0u);
                return true;
            }

            struct PendingTransfer
            {
                uint32_t source = 0u;
                uint32_t destination = 0u;
                uint32_t size = 0u;
            };

            std::array<uint32_t, kMaxDescriptors * 4u> descriptorWords{};
            const size_t descriptorBytes = static_cast<size_t>(descriptorCount) * kDescriptorSize;
            if (!m_memory.readRam(descriptorAddress, descriptorWords.data(), descriptorBytes))
            {
                setV0(0u);
                return true;
            }

            std::array<PendingTransfer, kMaxDescriptors> pending{};
            uint32_t pendingCount = 0u;
            uint32_t largestTransfer = 0u;
            for (uint32_t i = 0u; i < descriptorCount; ++i)
            {
                const uint32_t source = descriptorWords[i * 4u + 0u];
                const uint32_t destination = descriptorWords[i * 4u + 1u];
                const int32_t signedSize = static_cast<int32_t>(descriptorWords[i * 4u + 2u]);
                if (signedSize <= 0)
                    continue;

                const uint32_t size = static_cast<uint32_t>(signedSize);
                if (!m_memory.isRamRange(source, size))
                {
                    char message[128];
                    std::snprintf(message, sizeof(message), "[IOP] sceSifSetDma: bad IOP source 0x%x (%u bytes)", source, size);
                    m_host.log(LogLevel::Warning, message);
                    setV0(0u);
                    return true;
                }
                pending[pendingCount++] = {source, destination, size};
                largestTransfer = std::max(largestTransfer, size);
            }

            // IOP-side sceSifSetDma sends IOP RAM to the EE. Validate all EE
            // destinations before committing any write so a bad chain cannot
            // partially update guest memory, but maybe we could skip this check if we trust the EE-side SIF driver to validate the chain ?!
            // TODO check later
            std::vector<uint8_t> scratch(largestTransfer);
            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_host.readGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    char message[128];
                    std::snprintf(message, sizeof(message), "[IOP] sceSifSetDma: bad EE destination 0x%x (source 0x%x, %u bytes)",
                                  transfer.destination, transfer.source, transfer.size);
                    m_host.log(LogLevel::Warning, message);
                    setV0(0u);
                    return true;
                }
            }

            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_memory.readRam(transfer.source, scratch.data(), transfer.size) || !m_host.writeGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            static uint32_t transferLogCount = 0;
            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                if (transferLogCount < 64u || (transferLogCount % 512u) == 0u)
                {
                    char message[160];
                    std::snprintf(message, sizeof(message),
                                  "[IOP sif dma] #%u iop 0x%x -> ee 0x%x (%u bytes) ra=0x%x",
                                  transferLogCount, pending[i].source, pending[i].destination, pending[i].size,
                                  cpu.gpr[31]);
                    m_host.log(LogLevel::Info, message);
                }
                ++transferLogCount;
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            {
                m_nextDmaId = 1u;
            }
            setV0(dmaId);
            return true;
        }
        case 8: // sceSifDmaStat
            setV0(0xFFFFFFFFu);
            return true;
        case 29: // sceSifCheckInit
            setV0(m_sifInitialized ? 1u : 0u);
            return true;
        default:
            setV0(0u);
            return true;
        }
    }

    // IOP -> EE RPC. The client is the sifrpc t_SifRpcClientData, the same layout as on the EE: buf +0x14, cbuf +0x18,
    // server +0x24 (EE addresses). A bind that finds no EE server leaves server 0, the SDK's signal to retry.
    bool IopRpcBridge::bindEeRpc(IopCpuState &cpu)
    {
        constexpr uint32_t kClientBuffer = 0x14u, kClientCallbackBuffer = 0x18u, kClientServer = 0x24u;
        const uint32_t client = cpu.gpr[4];
        const uint32_t sid = cpu.gpr[5];
        cpu.gpr[2] = 0u;
        IopHost::EeRpcServer server;
        if (client == 0u || !m_memory.isRamRange(client, 0x28u) || !m_host.findEeRpcServer(sid, server))
        {
            static uint32_t logged = 0;
            if (logged++ < 8u)
            {
                char message[128];
                std::snprintf(message, sizeof(message), "[IOP sifcmd] sceSifBindRpc sid=0x%x: no EE server yet (client=0x%x)",
                              sid, client);
                m_host.log(LogLevel::Info, message);
            }
            return true;
        }
        m_memory.write32(client + kClientBuffer, server.buffer);
        m_memory.write32(client + kClientCallbackBuffer, server.callbackBuffer);
        m_memory.write32(client + kClientServer, server.serverData);
        m_eeClients[client] = sid;
        char message[128];
        std::snprintf(message, sizeof(message), "[IOP sifcmd] sceSifBindRpc sid=0x%x -> EE server 0x%x (client=0x%x)", sid,
                      server.serverData, client);
        m_host.log(LogLevel::Info, message);
        return true;
    }

    // The call goes to the host (the EE server thread serves it); the IRX caller sleeps until the reply unless the call
    // is SIF_RPC_M_NOWAIT. The receive size is rounded up to whole quadwords, as the SIF DMA moves them.
    bool IopRpcBridge::callEeRpc(IopCpuState &cpu)
    {
        constexpr uint32_t kRpcNowait = 0x1u;
        const uint32_t client = cpu.gpr[4];
        const uint32_t function = cpu.gpr[5];
        const uint32_t mode = cpu.gpr[6];
        const uint32_t send = cpu.gpr[7];
        const uint32_t sp = cpu.gpr[29];
        const uint32_t sendSize = m_memory.read32(sp + 16u);
        const uint32_t receive = m_memory.read32(sp + 20u);
        const uint32_t receiveSize = m_memory.read32(sp + 24u);
        const uint32_t endFunction = m_memory.read32(sp + 28u);
        const uint32_t endParam = m_memory.read32(sp + 32u);
        const auto bound = m_eeClients.find(client);
        if (bound == m_eeClients.end())
        {
            static uint32_t logged = 0;
            if (logged++ < 8u)
            {
                char message[128];
                std::snprintf(message, sizeof(message), "[IOP sifcmd] sceSifCallRpc on an unbound client 0x%x (fno=%u)",
                              client, function);
                m_host.log(LogLevel::Warning, message);
            }
            cpu.gpr[2] = static_cast<uint32_t>(-1);
            return true;
        }

        IopHost::EeRpcCall call;
        call.token = m_nextEeCallToken++;
        call.sid = bound->second;
        call.function = function;
        if (send != 0u && sendSize != 0u && m_memory.isRamRange(send, sendSize))
        {
            call.send.resize(sendSize);
            (void)m_memory.readRam(send, call.send.data(), sendSize);
        }
        const uint32_t transfer = receive != 0u ? (receiveSize + 15u) & ~15u : 0u;
        call.receiveSize = transfer;

        IopThread *const caller = m_kernel.currentThread();
        const bool wait = (mode & kRpcNowait) == 0u && caller != nullptr;
        EeCall pending;
        pending.thread = wait ? caller->id : 0;
        pending.receive = receive;
        pending.receiveSize = transfer;
        pending.endFunction = endFunction;
        pending.endParam = endParam;
        pending.gp = cpu.gpr[28];
        const uint64_t token = call.token;
        // Recorded first: a host that replies at once completes the call before callEeRpc returns.
        m_eeCalls[token] = pending;
        if (!m_host.callEeRpc(std::move(call)))
        {
            m_eeCalls.erase(token);
            cpu.gpr[2] = static_cast<uint32_t>(-1);
            return true;
        }
        cpu.gpr[2] = 0u;
        if (wait && m_eeCalls.count(token) != 0u)
            m_kernel.sleepCurrent(cpu);
        return true;
    }

    std::optional<IopRpcBridge::EndFunction> IopRpcBridge::completeEeRpc(uint64_t token, const void *data, size_t size)
    {
        const auto it = m_eeCalls.find(token);
        if (it == m_eeCalls.end())
            return std::nullopt;
        const EeCall call = it->second;
        m_eeCalls.erase(it);
        if (call.receive != 0u && call.receiveSize != 0u && m_memory.isRamRange(call.receive, call.receiveSize))
        {
            std::vector<uint8_t> words(call.receiveSize, 0u);
            if (data != nullptr)
                std::memcpy(words.data(), data, std::min<size_t>(size, words.size()));
            (void)m_memory.writeRam(call.receive, words.data(), words.size());
        }
        if (call.thread != 0)
        {
            if (IopThread *thread = m_kernel.thread(call.thread); thread && thread->state == IopThreadState::Sleep)
            {
                thread->cpu.gpr[2] = 0u;
                thread->state = IopThreadState::Ready;
            }
        }
        if (call.endFunction == 0u)
            return std::nullopt;
        return EndFunction{call.endFunction, call.gp, call.endParam};
    }

    std::optional<IopRpcBridge::CmdHandler> IopRpcBridge::cmdHandler(uint32_t cid) const
    {
        const auto it = m_cmdHandlers.find(cid);
        if (it == m_cmdHandlers.end() || it->second.function == 0u)
            return std::nullopt;
        return it->second;
    }

    bool IopRpcBridge::dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 10: // sceSifAddCmdHandler(cid, function, data)
        {
            m_cmdHandlers[cpu.gpr[4]] = CmdHandler{cpu.gpr[5], cpu.gpr[6], cpu.gpr[28]};
            char message[96];
            std::snprintf(message, sizeof(message), "[IOP] SIF command handler cid=0x%x function=0x%x",
                          cpu.gpr[4], cpu.gpr[5]);
            m_host.log(LogLevel::Info, message);
            setV0(0);
            return true;
        }
        case 11: // sceSifRemoveCmdHandler(cid)
            m_cmdHandlers.erase(cpu.gpr[4]);
            setV0(0);
            return true;
        case 6: // sceSifGetSreg(reg)
            setV0(cpu.gpr[4] < m_sregs.size() ? m_sregs[cpu.gpr[4]] : 0u);
            return true;
        case 7: // sceSifSetSreg(reg, value): returns the previous value
        {
            const uint32_t reg = cpu.gpr[4];
            const uint32_t previous = reg < m_sregs.size() ? m_sregs[reg] : 0u;
            setSreg(reg, cpu.gpr[5]);
            setV0(previous);
            return true;
        }
        case 4: // InitCmd
        case 5:
        case 8:
        case 9:
        case 14: // InitRpc
            setV0(0);
            return true;
        case 15: // sceSifBindRpc(client, sid, mode): an EE server
            return bindEeRpc(cpu);
        case 16: // sceSifCallRpc(client, fno, mode, send, ssize, recv, rsize, end, endparam)
            return callEeRpc(cpu);
        case 12: // sceSifSendCmd
        case 13: // isceSifSendCmd
        {
            constexpr uint32_t kHeaderSize = 16u;
            constexpr uint32_t kMaxPacketSize = 112u;
            const uint32_t commandId = cpu.gpr[4];
            const uint32_t packetAddress = cpu.gpr[5];
            const uint32_t packetSize = cpu.gpr[6];
            const uint32_t extraSource = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            const uint32_t extraDestination = m_memory.read32(stackPointer + 16u);
            const int32_t signedExtraSize = static_cast<int32_t>(m_memory.read32(stackPointer + 20u));

            if (packetAddress == 0u || packetSize < kHeaderSize || packetSize > kMaxPacketSize ||
                !m_memory.isRamRange(packetAddress, packetSize))
            {
                setV0(0u);
                return true;
            }

            std::array<uint8_t, kMaxPacketSize> packet{};
            if (!m_memory.readRam(packetAddress, packet.data(), packetSize))
            {
                setV0(0u);
                return true;
            }

            uint32_t extraSize = 0u;
            if (signedExtraSize > 0)
            {
                extraSize = static_cast<uint32_t>(signedExtraSize);
                if (extraSource == 0u || extraDestination == 0u ||
                    !m_memory.isRamRange(extraSource, extraSize) ||
                    !m_host.writeGuest(extraDestination, m_memory.ram().data() + IopMemory::physicalAddress(extraSource), extraSize))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t sizeWord = packetSize | (extraSize << 8u);
            std::memcpy(packet.data() + 0u, &sizeWord, sizeof(sizeWord));
            std::memcpy(packet.data() + 4u, &extraDestination, sizeof(extraDestination));
            std::memcpy(packet.data() + 8u, &commandId, sizeof(commandId));

            if (!m_host.sendSifCommand(commandId, packet.data(), packetSize))
            {
                // A command without an EE handler is still a completed DMA on  real hardware. Only malformed packets fail above.
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
                m_nextDmaId = 1u;
            setV0(dmaId);
            return true;
        }
        case 17: // sceSifRegisterRpc
        {
            RpcServer server;
            server.serverData = cpu.gpr[4];
            server.sid = cpu.gpr[5];
            server.function = cpu.gpr[6];
            server.gp = cpu.gpr[28];
            server.buffer = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            server.callback = m_memory.read32(stackPointer + 16u);
            server.callbackBuffer = m_memory.read32(stackPointer + 20u);
            server.queue = m_memory.read32(stackPointer + 24u);
            m_servers[server.sid] = server;
            if (server.serverData != 0u)
            {
                m_memory.write32(server.serverData + 0x20u, server.sid);
                m_memory.write32(server.serverData + 0x28u, server.function);
                m_memory.write32(server.serverData + 0x2Cu, server.buffer);
            }
            setV0(server.serverData);
            return true;
        }
        case 18:
            setV0(0);
            return true;
        case 19: // SetRpcQueue(queue, thread)
            m_queueThreads[cpu.gpr[4]] = static_cast<int>(cpu.gpr[5]);
            setV0(cpu.gpr[4]);
            return true;
        case 20:
        case 21:
            setV0(0);
            return true;
        case 22: // RpcLoop
            m_kernel.sleepCurrent(cpu);
            setV0(0);
            return true;
        case 23: // sceSifGetOtherData(receiveData, eeSource, iopDestination, size, mode)
        {
            const uint32_t size = cpu.gpr[7];
            bool copied = size == 0u;
            if (!copied && size <= IopMemory::RamSize)
            {
                std::vector<uint8_t> data(size);
                copied = m_host.readGuest(cpu.gpr[5], data.data(), data.size()) &&
                         m_memory.writeRam(cpu.gpr[6], data.data(), data.size());
            }
            setV0(copied ? 0u : static_cast<uint32_t>(-1));
            return true;
        }
        case 24: // RemoveRpc
        {
            const uint32_t serverData = cpu.gpr[4];
            for (auto server = m_servers.begin(); server != m_servers.end(); ++server)
            {
                if (server->second.serverData == serverData)
                {
                    m_servers.erase(server);
                    break;
                }
            }
            setV0(0);
            return true;
        }
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
            setV0(0);
            return true;
        default:
            return false;
        }
    }

    RpcResult IopRpcBridge::handleRpc(const RpcRequest &request, IopGuestExecutor &executor)
    {
        RpcResult result{};
        const auto serverIt = m_servers.find(request.sid);
        if (serverIt == m_servers.end() || serverIt->second.function == 0u)
            return result;

        RpcServer &server = serverIt->second;
        if (request.send.size != 0u && server.buffer != 0u)
        {
            const uint32_t copySize = std::min<uint32_t>(request.send.size, IopMemory::RamSize - std::min(server.buffer, IopMemory::RamSize));
            if (copySize != 0u)
            {
                if (request.sendData)
                    (void)m_memory.writeRam(server.buffer, request.sendData, copySize);
                else
                {
                    std::vector<uint8_t> payload(copySize);
                    if (m_host.readGuest(request.send.address, payload.data(), payload.size()))
                        (void)m_memory.writeRam(server.buffer, payload.data(), payload.size());
                }
            }
        }

        static const bool traceRpc = []
        {
            const char *value = std::getenv("PS2X_IOP_TRACE_RPC");
            return value && *value == '1';
        }();
        if (traceRpc)
            std::fprintf(stderr, "[iop:rpc] sid=0x%x fno=%u mode=%u send=%u recv=%u -> server 0x%x (inline)\n", request.sid,
                         request.function, request.mode, request.send.size, request.receive.size, server.function);
        // Run the server function as the thread that serves its queue, like sceSifRpcLoop does:
        // 989snd's bank load records GetThreadId() and sleeps until its loader thread wakes it.
        const auto queueThread = m_queueThreads.find(server.queue);
        uint32_t returnPointer = queueThread != m_queueThreads.end()
                                     ? executor.executeGuestFunctionAsThread(server.function,
                                                                             request.function,
                                                                             server.buffer,
                                                                             request.send.size,
                                                                             0u,
                                                                             server.gp,
                                                                             queueThread->second)
                                     : executor.executeGuestFunction(server.function,
                                                                     request.function,
                                                                     server.buffer,
                                                                     request.send.size,
                                                                     0u,
                                                                     server.gp);
        if (traceRpc)
            std::fprintf(stderr, "[iop:rpc] sid=0x%x fno=%u returned 0x%x\n", request.sid, request.function, returnPointer);
        if (returnPointer == 0u)
            returnPointer = server.buffer;
        if (request.receive.address != 0u && request.receive.size != 0u && returnPointer != 0u)
        {
            const uint32_t physical = IopMemory::physicalAddress(returnPointer);
            if (physical < IopMemory::RamSize)
            {
                const uint32_t copySize = std::min<uint32_t>(request.receive.size, IopMemory::RamSize - physical);
                (void)m_host.writeGuest(request.receive.address, m_memory.ram().data() + physical, copySize);
                if (copySize < request.receive.size)
                    (void)m_host.zeroGuest(request.receive.address + copySize, request.receive.size - copySize);
            }
        }

        result.handled = true;
        result.resultAddress = request.receive.address;
        result.serverDispatchPolicy = ServerDispatchPolicy::Suppress;
        result.signalNowaitCompletion = true;
        result.signalCompletion = true;
        return result;
    }

    void IopRpcBridge::onSifTransfer(const SifTransfer &transfer)
    {
        // The EE SIF transport owns the actual directional memory movement.
        // Services still receive both phases through IopSubsystem, but mirroring
        // IOP bytes through an equal-numbered EE address would alias two distinct
        // PS2 address spaces and can overwrite live game data.
        (void)transfer;
    }

    void IopRpcBridge::removeServersInRange(uint32_t base, uint32_t size)
    {
        for (auto server = m_servers.begin(); server != m_servers.end();)
        {
            const uint32_t function = IopMemory::physicalAddress(server->second.function);
            if (function >= base && function < base + size)
                server = m_servers.erase(server);
            else
                ++server;
        }
        // SIF command handlers of the unloaded module go with its servers.
        for (auto handler = m_cmdHandlers.begin(); handler != m_cmdHandlers.end();)
        {
            const uint32_t function = IopMemory::physicalAddress(handler->second.function);
            if (function >= base && function < base + size)
                handler = m_cmdHandlers.erase(handler);
            else
                ++handler;
        }
    }

    bool IopRpcBridge::hasServer(uint32_t sid) const noexcept
    {
        const auto server = m_servers.find(sid);
        return server != m_servers.end() && server->second.function != 0u;
    }
}
