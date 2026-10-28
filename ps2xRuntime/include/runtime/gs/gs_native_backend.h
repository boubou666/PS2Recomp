#pragma once

#include "runtime/gs/gs_backend.h"

#include <cstdint>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

class GSCpuBackend;
struct GLFWwindow;

// Native renderer (PS2X_GS_RENDERER=native): translates the primitives the GS frontend decodes into OpenGL draws
// into GPU render targets, instead of reproducing the GS. See docs/native.md, layer 3.
//
// It runs on the GS thread, on an OpenGL context shared with the window's (createSharedContext, called on the main
// thread after the window exists). GS local memory stays on the CPU for what is not translated yet: transfers,
// CLUT loads, readbacks and VRAM access go to an internal CPU backend used as memory only (it never draws).
class GSNativeBackend final : public GSRasterBackend
{
public:
    // Creates a hidden window whose context shares objects with the current (window) context. Main thread only.
    static GLFWwindow *createSharedContext();

    explicit GSNativeBackend(GLFWwindow *sharedContext);
    ~GSNativeBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;

    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    struct RenderTarget
    {
        uint32_t fbp = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t framebuffer = 0;
        uint32_t color = 0;
        uint32_t depth = 0;
    };

    struct Vertex
    {
        float x, y, z;
        uint8_t r, g, b, a;
    };

    // Makes our context current on the drawing (GS) thread the first time it draws (create = true); other calls
    // only use it when they run on that thread. The main thread presents the very first frame and must keep raylib's.
    bool ensureContext(bool create = false);
    RenderTarget &renderTarget(const GSFrameReg &frame);
    void appendPrimitive(const GSPrimitiveBatch &batch);
    void flushBatch();

    GLFWwindow *m_sharedContext = nullptr;
    bool m_contextReady = false;
    bool m_contextFailed = false;
    // Framebuffers and vertex arrays do not cross contexts: GL work happens only on the thread that made ours current.
    std::thread::id m_glThread{};
    std::thread::id m_submitThread{}; // the thread that draws: the only one that may bind our context
    std::unique_ptr<GSCpuBackend> m_memory;

    uint32_t m_program = 0;
    uint32_t m_vertexArray = 0;
    uint32_t m_vertexBuffer = 0;
    int m_targetSizeLocation = -1;

    std::unordered_map<uint32_t, RenderTarget> m_targets;
    // The batch being collected: draws that share a render target and the state that changes GL.
    std::vector<Vertex> m_batch;
    uint32_t m_batchTarget = ~0u;
    uint64_t m_batchStateKey = ~0ull;
    GSDrawState m_batchState{};
};
