#ifndef PS2X_NATIVE_VU0_HLE_H
#define PS2X_NATIVE_VU0_HLE_H

// A port's native versions of its game's VU0 microprograms (VCALLMS / VCALLMSR): asked first at each call, with the
// EE context (its VU0 registers: vu0_vf, vi, vu0_acc, vu0_q, vu0_i, the flags) and VU0's code and data memory; true,
// and the program is done, its results in the context and data memory as the program leaves them; false, and VU0's
// interpreter runs it.
//
// PS2X_VU0_HLE_CHECK=1 (a diagnostic): each program the port runs natively runs on the interpreter too, from the same
// state; registers and data memory are compared and the differences reported per program (the interpreter's results
// kept).

#include <atomic>
#include <cstdint>

struct R5900Context;

namespace ps2x::native
{
    using Vu0Hle = bool (*)(R5900Context *ctx, const uint8_t *vu0Code, uint8_t *vu0Data, uint32_t startPc);

    inline std::atomic<Vu0Hle> g_vu0Hle{nullptr};
    inline void setVu0Hle(Vu0Hle hle) { g_vu0Hle.store(hle, std::memory_order_release); }
    inline Vu0Hle vu0Hle() { return g_vu0Hle.load(std::memory_order_acquire); }
}

#endif
