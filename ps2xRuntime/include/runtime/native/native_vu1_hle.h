#ifndef PS2X_NATIVE_VU1_HLE_H
#define PS2X_NATIVE_VU1_HLE_H

// High-level emulation of a game's VU1 microprograms: at each MSCAL the port's handler may do what the microprogram
// would have done (read VU1 data memory and registers, send the PATH1 packets it would have sent with XGKICK) in
// native code, and VU1 does not run. A handler returns false for what it does not do yet, and VU1 runs as usual.

#include "runtime/ps2_vu1.h"

#include <atomic>
#include <cstdint>

namespace ps2x::native
{
    struct Vu1HleRun
    {
        const uint8_t *code = nullptr; // VU1 code memory as VIF1 left it (16 KiB)
        uint8_t *data = nullptr;       // VU1 data memory (16 KiB)
        VU1State *state = nullptr;     // registers as the previous programs left them
        uint32_t startPc = 0;
        uint32_t top = 0;
        uint32_t itop = 0;
        // Sends a GIF packet on PATH1, as XGKICK would, in order with the rest of the GIF stream.
        void (*kick)(void *user, const uint8_t *packet, uint32_t bytes) = nullptr;
        void *user = nullptr;
    };

    using Vu1Hle = bool (*)(Vu1HleRun &run);

    inline std::atomic<Vu1Hle> g_vu1Hle{nullptr};
    inline void setVu1Hle(Vu1Hle handler) { g_vu1Hle.store(handler, std::memory_order_release); }
    inline Vu1Hle vu1Hle() { return g_vu1Hle.load(std::memory_order_acquire); }
}

#endif
