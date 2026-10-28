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
// thread after the window exists). Textures come from the shared ps2x::native::TextureCache (the GS front end feeds
// it every upload while native rendering is on), or from a render target when a draw samples one (feedback effects). GS local memory also stays on
// the CPU for what is not translated yet: transfers, CLUT loads, readbacks and VRAM access go to an internal CPU
// backend used as memory only (it never draws).
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
        uint32_t fbw = 0;
        uint8_t psm = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t framebuffer = 0;
        uint32_t color = 0;
        uint32_t depth = 0;
        uint32_t usedHeight = 0; // rows drawn so far (the lowest vertex, within the scissor), what a resolve reads back
        bool dirty = false;      // drawn since its pixels were last written to GS memory (resolveTarget)
    };

    struct Vertex
    {
        float x, y, z;
        uint8_t r, g, b, a;
        float s, t, q; // texel coordinates (UV), or S and T in texels with their Q (divided per pixel, as the GS)
        float fog;
    };

    struct UploadedTexture
    {
        uint32_t texture = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t version = 0;
    };

    // Makes our context current on the drawing (GS) thread the first time it draws (create = true); other calls
    // only use it when they run on that thread. The main thread presents the very first frame and must keep raylib's.
    bool ensureContext(bool create = false);
    RenderTarget &renderTarget(const GSFrameReg &frame);
    RenderTarget *renderTargetAtBlock(uint32_t block);
    // A texture read the GPU cannot serve from a render target (another format, an offset base) sees GS memory:
    // first write back there the drawn pixels of every render target it overlaps.
    void resolveTargetsUnder(const GSTex0Reg &tex0);
    void resolveTarget(RenderTarget &target);
    // An image transfer into GS memory overwrites what the render targets under it hold: those are dropped (the
    // texture cache has the uploaded image; a later draw starts the target again).
    void dropTargetsUnder(const GSTransferCommand &command);
    // A local-to-local transfer whose source is a render target, done on the GPU; false when it is not one.
    bool copyBetweenTargets(const GSTransferCommand &command);
    void appendPrimitive(const GSPrimitiveBatch &batch);
    void flushBatch();
    // Binds the batch's texture to unit 0 and gives its size in texels; false when it has none to sample.
    bool bindTexture(const GSDrawState &state, const RenderTarget &target, float &width, float &height);
    void applyBlend(const GSDrawState &state);
    // What applyBlend asks of the shader: 0 its colour, 1 the blend factor as colour, 2 the colour times (1 + factor).
    int m_blendOut = 0;
    // One PCRTC read circuit's picture (RGBA8, alpha as the GS stores it), from its render target or else GS memory.
    bool readCircuit(uint64_t dispfb, uint64_t display, bool fieldMode, std::vector<uint8_t> &pixels, uint32_t &width,
                     uint32_t &height, uint32_t &fbp);

    GLFWwindow *m_sharedContext = nullptr;
    bool m_contextReady = false;
    bool m_contextFailed = false;
    // Framebuffers and vertex arrays do not cross contexts: GL work happens only on the thread that made ours current.
    std::thread::id m_glThread{};
    std::thread::id m_submitThread{}; // the thread that draws: the only one that may bind our context
    std::unique_ptr<GSCpuBackend> m_memory;
    uint8_t *m_vram = nullptr; // GS local memory (the CPU copy), for what is displayed without being drawn

    uint32_t m_program = 0;
    uint32_t m_vertexArray = 0;
    uint32_t m_vertexBuffer = 0;
    uint32_t m_lastSourceFramebuffer = 0; // the render target the last textured batch sampled (diagnostics)
    uint32_t m_feedbackTexture = 0; // copy of the render target a draw both samples and draws into
    uint32_t m_feedbackWidth = 0;
    uint32_t m_feedbackHeight = 0;
    struct Uniforms
    {
        int targetSize = -1, texSize = -1, textured = -1, fst = -1, tfx = -1, tcc = -1, fog = -1, fogColor = -1;
        int alphaTest = -1, alphaRef = -1, sampler = -1, blendOut = -1, blendFactor = -1, blendFix = -1;
    } m_uniforms;

    std::unordered_map<uint32_t, RenderTarget> m_targets;
    struct ReadbackBuffers
    {
        uint32_t pbo[2] = {0, 0};
        size_t bytes[2] = {0, 0};
        bool ready[2] = {false, false};
        int next = 0;
    };
    std::unordered_map<uint32_t, ReadbackBuffers> m_readbacks; // per displayed buffer (readCircuit)
    std::unordered_map<uint64_t, UploadedTexture> m_uploaded;

    // The batch being collected: draws whose state draws the same way (sameDrawState).
    std::vector<Vertex> m_batch;
    float m_batchMaxY = 0.0f; // the batch's lowest row, for its render target's usedHeight
    GSDrawState m_batchState{};
};
