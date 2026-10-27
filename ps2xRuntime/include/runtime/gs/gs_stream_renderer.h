#pragma once

// A GS renderer behind the GS frontend that takes the GIF stream as it comes (raw packets, register writes) and
// hands back the displayed picture at each presentation latch, instead of the frontend's decoded primitives:
// paraLLEl-GS today (PS2X_GS_RENDERER=parallel). The frontend's GS thread creates it and makes every call from
// that thread.
#include "runtime/gs/gs_types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

enum class GSStreamRendererKind : uint8_t
{
    Parallel,
};

class GSStreamRenderer
{
public:
    struct Callbacks
    {
        std::function<void(uint64_t)> signal; // SIGNAL register written (its value)
        std::function<void()> finish;          // FINISH
        std::function<void(uint64_t)> label;   // LABEL
    };

    // Null when the build does not have that renderer or it cannot start (no Vulkan device with the features it
    // needs).
    static std::unique_ptr<GSStreamRenderer> create(GSStreamRendererKind kind, uint32_t vramSize,
                                                    const Callbacks &callbacks);
    static bool available(GSStreamRendererKind kind);
    static const char *name(GSStreamRendererKind kind);
    static bool parse(std::string_view name, GSStreamRendererKind &kind);
    // PS2X_GS_RENDERER names a stream renderer this build has.
    static bool requested(GSStreamRendererKind &kind);

    virtual ~GSStreamRenderer() = default;

    virtual void gifTransfer(const uint8_t *data, uint32_t sizeBytes) = 0;
    virtual void writeRegister(uint8_t address, uint64_t value) = 0;
    // Submits the drawing queued so far to the GPU.
    virtual void flush() = 0;
    // Scans out the display circuits the request's privileged registers select (waits for the GPU); RGBA rows
    // of the picture's width.
    virtual PresentationFrame present(const GSPresentationRequest &request) = 0;
    // Local to host transfer data (TRXDIR 1), in 16-byte words.
    virtual void readTransferFifo(uint8_t *dst, uint32_t qwords) = 0;
    virtual void readVram(std::vector<uint8_t> &out) = 0;
    virtual void reset() = 0;
};

std::unique_ptr<GSStreamRenderer> createParallelGsRenderer(uint32_t vramSize, const GSStreamRenderer::Callbacks &callbacks);
