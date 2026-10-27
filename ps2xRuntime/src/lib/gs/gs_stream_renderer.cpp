#include "runtime/gs/gs_stream_renderer.h"

#include <cstdlib>

bool GSStreamRenderer::available(GSStreamRendererKind kind)
{
    switch (kind)
    {
    case GSStreamRendererKind::Parallel:
#if defined(PS2X_HAVE_PARALLEL_GS)
        return true;
#else
        return false;
#endif
    }
    return false;
}

const char *GSStreamRenderer::name(GSStreamRendererKind kind)
{
    switch (kind)
    {
    case GSStreamRendererKind::Parallel:
        return "parallel";
    }
    return "?";
}

bool GSStreamRenderer::parse(std::string_view text, GSStreamRendererKind &kind)
{
    for (const GSStreamRendererKind candidate : {GSStreamRendererKind::Parallel})
        if (text == name(candidate))
        {
            kind = candidate;
            return true;
        }
    return false;
}

bool GSStreamRenderer::requested(GSStreamRendererKind &kind)
{
    const char *renderer = std::getenv("PS2X_GS_RENDERER");
    GSStreamRendererKind parsed{};
    if (!renderer || !parse(renderer, parsed) || !available(parsed))
        return false;
    kind = parsed;
    return true;
}

std::unique_ptr<GSStreamRenderer> GSStreamRenderer::create(GSStreamRendererKind kind, uint32_t vramSize,
                                                           const Callbacks &callbacks)
{
    switch (kind)
    {
    case GSStreamRendererKind::Parallel:
#if defined(PS2X_HAVE_PARALLEL_GS)
        return createParallelGsRenderer(vramSize, callbacks);
#else
        (void)vramSize;
        (void)callbacks;
        return nullptr;
#endif
    }
    return nullptr;
}
