#pragma once
// Lifted microcode: native code standing in for a loop of a known microprogram, written on the game's side from
// what the loop computes rather than instruction by instruction (several iterations at once, no per-pair timing).
// VU1Interpreter::registerLiftedEntry() puts it in place of the compiled code at the loop's first pair while the
// compiled fragment it was written for is in code memory (PS2X_VU_LIFTED=0 leaves it out); ps2_vu1_replay checks it
// against recorded runs. The entry works on the state the compiled code would: the registers, flags and clip
// register as the pairs before it left them, then moves the clock by the cycles the iterations take.
#include "runtime/ps2_vu1_inline.h"

struct Vu1Lifted
{
    using Vu = VU1Interpreter;

    static VU1State &state(Vu &vu) { return vu.m_state; }
    static uint64_t cycle(const Vu &vu) { return vu.m_cycle; }

    // Whether an entry may run ahead `cycles` cycles: no time slice (whose end shows the registers as the pipeline
    // would), no branch or halt pending, the budget not reached.
    static bool canRunAhead(const Vu &vu, const Vu::RunContext &context, uint64_t cycles)
    {
        return !vu.m_logInFlight && !vu.m_state.branchPending && !vu.m_state.haltAfterDelaySlot && !vu.m_state.ebit &&
               vu.m_cycle + cycles < context.budgetEnd &&
               !vu.m_state.dBitEnabled && !vu.m_state.tBitEnabled;
    }

    // The queued flag writes and Q/P results land now, as if their latency had passed (the clock does not move):
    // m_state then holds every result the pairs before the entry produced.
    static void landQueued(Vu &vu)
    {
        const uint64_t now = vu.m_cycle;
        vu.m_cycle = UINT64_MAX;
        vu.commitLandedFlags();
        vu.commitScalarPipelines();
        vu.m_cycle = now;
        vu.foldDeadSticky();
    }

    // The clip flag register, with the copy the next CLIP shifts (call after landQueued).
    static void setClip(Vu &vu, uint32_t clip)
    {
        vu.m_state.clip = clip & 0xFFFFFFu;
        vu.m_workingClip = vu.m_state.clip;
    }

    // The clock to `cycle`: PATH1 transfers progress (reading VU memory as it is now), ready results land.
    static void advanceTo(Vu &vu, uint64_t cycle) { vu.advanceTo(cycle); }

    static float normalizeOperand(const Vu &vu, float value) { return vu.normalizeOperand(value); }
    static float normalizeResult(const Vu &vu, float value)
    {
        uint32_t ignored = 0u;
        return vu.normalizeResult(value, ignored);
    }
};
