#pragma once

// A GS renderer that takes the GIF stream as it comes (raw packets, register writes, native uploads) and hands
// back the displayed picture at each presentation latch. paraLLEl-GS (PS2X_GS_RENDERER=parallel) runs behind the
// GS frontend this way, on its GS thread; the CPU renderer (a synchronous GS with the CPU backend) implements it
// too, for the tools and tests that drive several renderers with one stream (ps2_gs_replay --compare). Every
// call must come from the thread that created it.
#include "runtime/gs/gs_types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

struct GSStateImage;

enum class GSStreamRendererKind : uint8_t
{
    Cpu,
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
    // Whether the frontend hands its stream to that renderer (the CPU renderer is the frontend's own path).
    static bool hostedByFrontend(GSStreamRendererKind kind);
    // PS2X_GS_RENDERER names a renderer this build has that the frontend hands its stream to.
    static bool requested(GSStreamRendererKind &kind);

    virtual ~GSStreamRenderer() = default;

    virtual void gifTransfer(const uint8_t *data, uint32_t sizeBytes) = 0;
    // A packet the EE side validated as PACKED only (GS::processNativePackedGIFPacket).
    virtual void nativePackedGifTransfer(const uint8_t *data, uint32_t sizeBytes) { gifTransfer(data, sizeBytes); }
    // Host to local transfer: BITBLTBUF, TRXPOS, TRXREG and TRXDIR, then the image data (GS::uploadImageNative).
    virtual void uploadImage(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, uint64_t trxdir, const uint8_t *data,
                             uint32_t sizeBytes) = 0;
    virtual void writeRegister(uint8_t address, uint64_t value) = 0;
    // Submits the drawing queued so far to the GPU.
    virtual void flush() = 0;
    // Scans out the display circuits the request's privileged registers select (waits for the GPU); RGBA rows
    // of the picture's width.
    virtual PresentationFrame present(const GSPresentationRequest &request) = 0;
    // Local to host transfer data (TRXDIR 1), in 16-byte words.
    virtual void readTransferFifo(uint8_t *dst, uint32_t qwords) = 0;
    virtual void readVram(std::vector<uint8_t> &out) = 0;
    // Replaces all of VRAM (sizeBytes is the renderer's VRAM size).
    virtual void writeVram(const uint8_t *data, uint32_t sizeBytes) = 0;
    // One pixel in the PSM's own format (GS::ReadVram / WriteVram).
    virtual uint32_t readVramPixel(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) = 0;
    virtual void writeVramPixel(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) = 0;
    // Drawing state between packets, with no image transfer in progress. A renderer that cannot see part of it
    // leaves the matching GSStateImage flag clear.
    virtual void saveState(GSStateImage &image) = 0;
    virtual void loadState(const GSStateImage &image) = 0;
    virtual void reset() = 0;
};

std::unique_ptr<GSStreamRenderer> createCpuGsRenderer(uint32_t vramSize, const GSStreamRenderer::Callbacks &callbacks);
std::unique_ptr<GSStreamRenderer> createParallelGsRenderer(uint32_t vramSize, const GSStreamRenderer::Callbacks &callbacks);
