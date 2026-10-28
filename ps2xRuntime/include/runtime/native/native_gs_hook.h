#ifndef PS2X_NATIVE_GS_HOOK_H
#define PS2X_NATIVE_GS_HOOK_H

// A native renderer drawing inside the GS stream: where the VIF1 interpreter skips a VU1 program that renderer draws
// itself (its PATH1 output never reaches the GS), GS::queueNativeDraw puts a marker in the GS queue, in PATH1 order.
// On the GS thread the marker calls the hook with the render target the GS output would have gone to, so the native
// draws sort with the GS draws around them (particles in front of a mesh, a HUD over it).

#include <atomic>
#include <cstdint>

namespace ps2x::native
{
    struct GsDrawTarget
    {
        // GL objects on the GS thread's context, current while the hook runs. A null framebuffer means the GS
        // renderer in use is not the native one (CPU, paraLLEl-GS): nothing to draw into.
        uint32_t framebuffer = 0, color = 0, depth = 0;
        uint32_t width = 0, height = 0;
        // Depth is GS Z times zScale (1/2^32 Z32, 1/2^24 Z24, 1/2^16 Z16), larger nearer, GEQUAL; colour keeps the
        // GS alpha as stored (0x80 = 1.0).
        double zScale = 0.0;
        // The context's XYOFFSET (12.4 fixed point, as written) and SCISSOR (pixels, inclusive).
        uint32_t xOffset = 0, yOffset = 0;
        uint32_t scissorX0 = 0, scissorY0 = 0, scissorX1 = 0, scissorY1 = 0;
        // The context's drawing registers as written at the marker (raw GS layouts), for draws whose material is the
        // GS state where they sit in the stream (Material::fromGsContext): TEX0, TEXA, ALPHA, TEST, CLAMP, FOGCOL,
        // and the Z buffer's write mask.
        uint64_t tex0 = 0, texa = 0, alpha = 0, test = 0, clamp = 0;
        uint32_t fogcol = 0;
        bool zmask = false;
    };

    using GsDrawHook = void (*)(uint64_t group, const GsDrawTarget &target);

    inline std::atomic<GsDrawHook> g_gsDrawHook{nullptr};
    inline void setGsDrawHook(GsDrawHook hook) { g_gsDrawHook.store(hook, std::memory_order_release); }
    inline GsDrawHook gsDrawHook() { return g_gsDrawHook.load(std::memory_order_acquire); }
}

#endif
