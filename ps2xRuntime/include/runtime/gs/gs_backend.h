#pragma once

#include "runtime/gs/gs_types.h"
#include "runtime/native/native_gs_hook.h"

#include <cstdint>
#include <vector>

class GSRasterBackend
{
public:
    virtual ~GSRasterBackend() = default;

    virtual void Initialize(uint8_t *vram, uint32_t vramSize) = 0;
    virtual void Reset() = 0;

    virtual void Submit(const GSPrimitiveBatch &batch) = 0;
    virtual void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) = 0;

    virtual void BeginTransfer(const GSTransferCommand &command) = 0;
    virtual void UploadImage(const uint8_t *data, uint32_t sizeBytes) = 0;

    virtual void Flush() = 0;
    virtual void TextureFlush() = 0;
    virtual void Sync(GSSyncReason reason) = 0;
    virtual PresentationFrame Present(const GSPresentationRequest &request) = 0;

    virtual bool ClearFramebuffer(const GSContext &context, uint32_t rgba) = 0;
    virtual uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) = 0;

    virtual uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const = 0;
    virtual void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) = 0;
    virtual void SnapshotVram(std::vector<uint8_t> &out) const = 0;
    virtual GSTransferSnapshot GetTransferSnapshot() const = 0;
    // Present can only run on the thread that draws (a GPU context there): presentation latches go through the
    // GS queue.
    virtual bool PresentsOnDrawThread() const { return false; }
    // A native renderer's draws at this point of the stream (GS::queueNativeDraw): the hook with this context's
    // render target. Renderers that keep GS memory themselves have none to give: a null target, with the context's
    // offset and scissor still (the native renderer clips its own frame to them).
    // The context's drawing registers for a native draw marker (GsDrawTarget's raw registers).
    static void fillNativeRegisters(ps2x::native::GsDrawTarget &target, const GSContext &context)
    {
        const GSTex0Reg &t = context.tex0;
        target.tex0 = static_cast<uint64_t>(t.tbp0 & 0x3FFFu) | (static_cast<uint64_t>(t.tbw & 0x3Fu) << 14) |
                      (static_cast<uint64_t>(t.psm & 0x3Fu) << 20) | (static_cast<uint64_t>(t.tw & 0xFu) << 26) |
                      (static_cast<uint64_t>(t.th & 0xFu) << 30) | (static_cast<uint64_t>(t.tcc & 1u) << 34) |
                      (static_cast<uint64_t>(t.tfx & 3u) << 35) | (static_cast<uint64_t>(t.cbp & 0x3FFFu) << 37) |
                      (static_cast<uint64_t>(t.cpsm & 0xFu) << 51) | (static_cast<uint64_t>(t.csm & 1u) << 55) |
                      (static_cast<uint64_t>(t.csa & 0x1Fu) << 56) | (static_cast<uint64_t>(t.cld & 7u) << 61);
        target.texa = context.texaRaw;
        target.alpha = context.alpha;
        target.test = context.test;
        target.clamp = context.clamp;
        target.fogcol = context.fogcol;
        target.zmask = context.zbuf.zmask;
    }

    virtual void NativeDraw(uint64_t group, const GSContext &context)
    {
        if (const ps2x::native::GsDrawHook hook = ps2x::native::gsDrawHook())
        {
            ps2x::native::GsDrawTarget target;
            fillNativeRegisters(target, context);
            target.xOffset = context.xyoffset.ofx;
            target.yOffset = context.xyoffset.ofy;
            target.scissorX0 = context.scissor.x0;
            target.scissorY0 = context.scissor.y0;
            target.scissorX1 = context.scissor.x1;
            target.scissorY1 = context.scissor.y1;
            hook(group, target);
        }
    }
};
