#include "runtime/native/native_vu1_sandbox.h"

#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_vu1.h"

namespace ps2x::native
{
    Vu1Sandbox::Vu1Sandbox() : m_vu(std::make_unique<VU1Interpreter>(VU1Interpreter::Unit::VU1)), m_gs(std::make_unique<GS>())
    {
        m_vu->reset();
        m_vu->setPacketsToObserverOnly(true);
        m_vu->setPacketObserver([this](const uint8_t *packet, uint32_t size) { m_gif.packet(packet, size); });
    }

    Vu1Sandbox::~Vu1Sandbox() = default;

    uint64_t Vu1Sandbox::cycles() const { return m_vu->state().cycles; }

    VU1Interpreter &Vu1Sandbox::vu() { return *m_vu; }

    void Vu1Sandbox::codeChanged() { m_vu->invalidateDecodedCode(); }

    // As the runtime runs VU1 for VIF1 (to the program's end: the budget is only a safety limit).
    void Vu1Sandbox::call(uint8_t *code, uint32_t codeSize, uint8_t *data, uint32_t dataSize, uint32_t startPc, uint32_t top,
                          uint32_t itop)
    {
        m_vu->execute(code, codeSize, data, dataSize, *m_gs, nullptr, startPc, top, itop, 1u << 24);
        m_gif.flush();
    }

    void Vu1Sandbox::resume(uint8_t *code, uint32_t codeSize, uint8_t *data, uint32_t dataSize, uint32_t top, uint32_t itop)
    {
        m_vu->resume(code, codeSize, data, dataSize, *m_gs, nullptr, top, itop, 1u << 24);
        m_gif.flush();
    }
}
