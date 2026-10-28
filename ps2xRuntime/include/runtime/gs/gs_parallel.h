#pragma once

// paraLLEl-GS (Vulkan compute GS emulation) behind the GS frontend: PS2X_GS_RENDERER=parallel. The frontend's GS
// thread hands it the GIF packets as they come and asks it for the displayed picture at each presentation latch;
// VRAM lives on the GPU. Every call must come from the thread that created it.
#include "runtime/gs/gs_types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class GSParallelRenderer
{
public:
    struct Callbacks
    {
        std::function<void(uint64_t)> signal; // SIGNAL register written (its value)
        std::function<void()> finish;          // FINISH
        std::function<void(uint64_t)> label;   // LABEL
    };

    // Null when the build has no paraLLEl-GS or no Vulkan device with the features it needs.
    static std::unique_ptr<GSParallelRenderer> create(uint32_t vramSize, const Callbacks &callbacks);
    // PS2X_GS_RENDERER=parallel and the build has paraLLEl-GS.
    static bool requested();

    virtual ~GSParallelRenderer() = default;

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
    // A drawing context's XYOFFSET and SCISSOR as the packets so far set them (context 0 or 1).
    virtual void contextRegisters(int context, uint64_t &xyoffset, uint64_t &scissor) const = 0;
    virtual void reset() = 0;
};
