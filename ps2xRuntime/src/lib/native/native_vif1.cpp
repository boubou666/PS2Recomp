#include "runtime/native/native_vif1.h"

#include <atomic>

namespace ps2x::native
{
    namespace
    {
        std::atomic<Vif1Observer> g_observer{nullptr};
        std::atomic<Vu1Gate> g_gate{nullptr};
        std::atomic<DirectGate> g_directGate{nullptr};
        std::atomic<const VIFRegisters *> g_vif1Registers{nullptr};
        std::atomic<Vif1ClockAdvance> g_clockAdvance{nullptr};
        std::atomic<void *> g_clockUser{nullptr};
    }

    void setVif1Observer(Vif1Observer observer)
    {
        g_observer.store(observer, std::memory_order_release);
    }

    bool vif1Observed()
    {
        return g_observer.load(std::memory_order_relaxed) != nullptr;
    }

    void observeVif1(const uint8_t *stream, size_t bytes)
    {
        if (Vif1Observer observer = g_observer.load(std::memory_order_acquire); observer && bytes != 0u)
            observer(stream, bytes);
    }

    void setVu1Gate(Vu1Gate gate)
    {
        g_gate.store(gate, std::memory_order_release);
    }

    bool vu1Skipped(const uint8_t *vu1Code, uint32_t startPc)
    {
        const Vu1Gate gate = g_gate.load(std::memory_order_acquire);
        return gate && vu1Code && gate(vu1Code, startPc);
    }

    void setVif1Registers(const VIFRegisters *registers) { g_vif1Registers.store(registers, std::memory_order_release); }

    const VIFRegisters *vif1Registers() { return g_vif1Registers.load(std::memory_order_acquire); }

    void setVif1ClockAdvance(Vif1ClockAdvance advance, void *user)
    {
        g_clockUser.store(user, std::memory_order_release);
        g_clockAdvance.store(advance, std::memory_order_release);
    }

    void advanceVif1Clock(uint64_t cycles)
    {
        if (Vif1ClockAdvance advance = g_clockAdvance.load(std::memory_order_acquire); advance && cycles != 0u)
            advance(g_clockUser.load(std::memory_order_acquire), cycles);
    }

    void setDirectGate(DirectGate gate)
    {
        g_directGate.store(gate, std::memory_order_release);
    }

    bool directSkipped(uint32_t qwords)
    {
        const DirectGate gate = g_directGate.load(std::memory_order_acquire);
        return gate && gate(qwords);
    }
}
