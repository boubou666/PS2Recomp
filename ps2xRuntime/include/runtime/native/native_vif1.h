#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

struct VIFRegisters;

namespace ps2x::native
{
    // The VIF1 stream a game sends, handed to its native adapter as the game sends it (on the EE thread, so EE memory
    // still holds what the stream refers to): a chain flattened the way VIF1 receives it (each tag's two VIF codes,
    // then its data), or a normal transfer's quadwords. The adapter reads the game's own 3D data from it (meshes,
    // matrices, lights) wherever in the game's code the send happens.
    // onVif1Thread: called on the VIF1 thread instead, with each transfer as VIF1 gets it, in pieces: up to each MSCAL /
    // MSCNT before VIF1 asks the gate about it, then the rest after the transfer (for an adapter that reads only the
    // stream, not the EE memory it refers to: its programs then run beside the EE, between VIF1's own commands).
    using Vif1Observer = void (*)(const uint8_t *stream, size_t bytes);

    void setVif1Observer(Vif1Observer observer, bool onVif1Thread = false);
    bool vif1Observed();
    bool vif1ObservedOnVif1Thread();
    void observeVif1(const uint8_t *stream, size_t bytes);

    // Runs a job on the VIF1 thread after the transfers sent so far (at once without a VIF1 thread, or on it).
    using Vif1Runner = void (*)(void *user, std::function<void()> job);

    void setVif1Runner(Vif1Runner runner, void *user);
    void runOnVif1Thread(std::function<void()> job);

    // A microprogram the native port draws itself: VU1 does not run it, nor the MSCNTs that continue it. Asked on the
    // VIF1 thread at each MSCAL / MSCALF with VU1's code memory as VIF1 left it and the start address.
    using Vu1Gate = bool (*)(const uint8_t *vu1Code, uint32_t startPc);

    void setVu1Gate(Vu1Gate gate);
    bool vu1Skipped(const uint8_t *vu1Code, uint32_t startPc);

    // A DIRECT packet the native port draws itself (a game's particles sent as GS packets in its VIF1 stream): the GS
    // does not get it. Asked on the VIF1 thread for each whole DIRECT (not one finishing an image transfer) with its
    // size in quadwords, as the VIFcode gives it; a skipped one counts as a skipped program (the GS stream's groups).
    using DirectGate = bool (*)(uint32_t qwords);

    void setDirectGate(DirectGate gate);
    bool directSkipped(uint32_t qwords);

    // On the VIF1 thread, for a program its gate skipped: the VU1 cycles the program took elsewhere, which move the VIF1
    // thread's clock on as VU1 running it would have (with the GS lag, the EE waits for that clock).
    using Vif1ClockAdvance = void (*)(void *user, uint64_t cycles);

    void setVif1ClockAdvance(Vif1ClockAdvance advance, void *user);
    void advanceVif1Clock(uint64_t cycles);

    // VIF1's own registers (the VIF1 thread's, read without a lock): STMASK / STROW / STCOL / STMOD a game set in a
    // transfer the adapter did not see (at boot) are there. Null before the runtime registers them.
    void setVif1Registers(const VIFRegisters *registers);
    const VIFRegisters *vif1Registers();

    // One UNPACK as VIF1 does it (STCYCL, STMOD, STMASK, STROW / STCOL in regs, TOPS for the FLG bit), into VU data
    // memory of vuDataSize bytes: an adapter that keeps its own copy of VU1 memory (to run microprograms itself,
    // ps2x::native::Vu1Sandbox) fills it the same way. cmd is the UNPACK code; returns the payload bytes it used.
    uint32_t vifUnpack(VIFRegisters &regs, uint8_t *vuData, uint32_t vuDataSize, uint32_t cmd, const uint8_t *payload,
                       uint32_t available);
}
