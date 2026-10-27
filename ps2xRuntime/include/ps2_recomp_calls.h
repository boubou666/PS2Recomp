#ifndef PS2_RECOMP_CALLS_H
#define PS2_RECOMP_CALLS_H

// The runtime's syscall and SDK stub functions as recompiled code calls them: declarations only, over
// ps2_recomp_abi.h (ps2_syscalls.h and ps2_stubs.h declare the same functions for the runtime itself).

#include <cstdint>

#include "ps2_call_list.h"
#include "ps2_recomp_abi.h"

namespace ps2_syscalls
{
#define PS2_DECLARE_SYSCALL(name) void name(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    PS2_SYSCALL_LIST(PS2_DECLARE_SYSCALL)
#undef PS2_DECLARE_SYSCALL

    void iDeleteSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void EnableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void DisableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void EnableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void DisableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void TODO(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t encodedSyscallId);
}

namespace ps2_stubs
{
#define PS2_DECLARE_STUB(name) void name(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    PS2_STUB_LIST(PS2_DECLARE_STUB)
#undef PS2_DECLARE_STUB

    void TODO(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void TODO_NAMED(const char *name, uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
}

#endif // PS2_RECOMP_CALLS_H
