#ifndef PS2X_NATIVE_VU1_SANDBOX_H
#define PS2X_NATIVE_VU1_SANDBOX_H

// The microprograms an adapter does not translate, run on the PC as VU1 would run them and drawn natively: a VU1 of
// its own runs a program on the adapter's copy of VU1's code and data memory (filled from the VIF1 stream, with
// ps2x::native::vifUnpack), and what the program sends the GS (its XGKICKs) goes to a GifToNative instead. The real
// VU1 skips the program (the adapter's Vu1Gate says so), so the frame's every VU1 draw is native, in stream order.
//
// On the thread the adapter draws from (the game's). Registers carry over from one run to the next, as on VU1.

#include "runtime/native/native_gif.h"

#include <cstdint>
#include <memory>

class GS;
class VU1Interpreter;

namespace ps2x::native
{
    class Vu1Sandbox
    {
    public:
        Vu1Sandbox();
        ~Vu1Sandbox();

        // The code memory changed (an MPG): what was decoded from it is dropped.
        void codeChanged();
        // MSCAL: the program at startPc, with VIF1's TOP and ITOP; MSCNT: where the last one ended.
        void call(uint8_t *code, uint32_t codeSize, uint8_t *data, uint32_t dataSize, uint32_t startPc, uint32_t top, uint32_t itop);
        void resume(uint8_t *code, uint32_t codeSize, uint8_t *data, uint32_t dataSize, uint32_t top, uint32_t itop);
        // The GS registers the stream sets around the programs (DIRECT): the programs' packets draw with them.
        GifToNative &gif() { return m_gif; }

    private:
        std::unique_ptr<VU1Interpreter> m_vu;
        std::unique_ptr<GS> m_gs; // never fed: the interpreter's packets go to m_gif
        GifToNative m_gif;
    };
}

#endif
