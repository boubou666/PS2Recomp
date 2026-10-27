// The out-of-line half of ps2_recomp_abi.h: the runtime calls of recompiled code, forwarded to PS2Runtime
// (ps2rCheckpointDue is in ps2_runtime.cpp, next to the member it implements).

#include "ps2_recomp_abi.h"
#include "ps2_runtime.h"

// Generated code is compiled against R5900Context from ps2_recomp_abi.h; the build recompiles it when that header
// changes, but an object built elsewhere would not be. A layout change must update this size (and regenerate).
static_assert(sizeof(R5900Context) == 1648, "R5900Context layout changed: update the size and rebuild all generated code");

uint8_t ps2rLoad8(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load8(rdram, ctx, vaddr); }
uint16_t ps2rLoad16(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load16(rdram, ctx, vaddr); }
uint32_t ps2rLoad32(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load32(rdram, ctx, vaddr); }
uint64_t ps2rLoad64(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load64(rdram, ctx, vaddr); }
__m128i ps2rLoad128(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load128(rdram, ctx, vaddr); }

void ps2rStore8(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value) { runtime->Store8(rdram, ctx, vaddr, value); }
void ps2rStore16(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value) { runtime->Store16(rdram, ctx, vaddr, value); }
void ps2rStore32(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value) { runtime->Store32(rdram, ctx, vaddr, value); }
void ps2rStore64(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value) { runtime->Store64(rdram, ctx, vaddr, value); }
void ps2rStore128(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value) { runtime->Store128(rdram, ctx, vaddr, value); }

bool ps2rDispatchGuestBranch(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t targetPc, uint32_t sourcePc,
                             uint32_t fallthroughPc, Ps2GuestBranchKind kind, const char *debugName)
{
    return runtime->dispatchGuestBranch(rdram, ctx, targetPc, sourcePc, fallthroughPc, kind, debugName);
}

void ps2rSignalException(PS2Runtime *runtime, R5900Context *ctx, PS2Exception exception) { runtime->SignalException(ctx, exception); }
void ps2rExecuteVU0Microprogram(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t address) { runtime->executeVU0Microprogram(rdram, ctx, address); }
void ps2rVu0StartMicroProgram(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t address) { runtime->vu0StartMicroProgram(rdram, ctx, address); }
uint8_t *ps2rVu0Data(PS2Runtime *runtime) { return runtime->memory().getVU0Data(); }

void ps2rHandleSyscall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleSyscall(rdram, ctx); }
void ps2rHandleSyscall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId) { runtime->handleSyscall(rdram, ctx, encodedSyscallId); }
void ps2rHandleBreak(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleBreak(rdram, ctx); }
void ps2rHandleTrap(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTrap(rdram, ctx); }
void ps2rHandleTLBR(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBR(rdram, ctx); }
void ps2rHandleTLBWI(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBWI(rdram, ctx); }
void ps2rHandleTLBWR(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBWR(rdram, ctx); }
void ps2rHandleTLBP(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBP(rdram, ctx); }
void ps2rClearLLBit(PS2Runtime *runtime, R5900Context *ctx) { runtime->clearLLBit(ctx); }

void ps2rKickGifDmaChainFromMMIO(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t dPcrValue,
                                 uint32_t dStatValue, uint32_t tadr, uint32_t chcr)
{
    runtime->kickGifDmaChainFromMMIO(rdram, ctx, dPcrValue, dStatValue, tadr, chcr);
}
