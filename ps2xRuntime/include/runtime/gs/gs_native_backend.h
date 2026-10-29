#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/native/native_texture_cache.h"

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

    // Diagnostics (ps2_gs_replay --dump): every drawn render target's pixels written to GS memory (m_vram).
    void resolveAllTargets();

    // For a native renderer drawing into the same frame (a mesh renderer's 3D between GS draws): the render target of
    // a FRAME register, with the batch collected so far drawn first. GL objects on the GS thread's context. Depth is
    // GS Z times zScale (1/2^32 Z32, 1/2^24 Z24, 1/2^16 Z16), larger nearer, cleared to 0; colour keeps the GS alpha
    // as stored (0x80 = 1.0). Null framebuffer when the context is not available.
    struct FrameTarget
    {
        uint32_t framebuffer = 0, color = 0, depth = 0;
        uint32_t width = 0, height = 0; // in GS pixels; the GL textures are scale times that
        uint32_t scale = 1;
        double zScale = 0.0;
    };
    // rows: how far down the native draw may write (its scissor), what the target then covers in GS memory.
    FrameTarget frameTarget(const GSFrameReg &frame, const GSZbufReg &zbuf, uint32_t rows);
    void NativeDraw(uint64_t group, const GSContext &context) override;
    bool PresentsOnDrawThread() const override { return true; }

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
        uint32_t zbp = ~0u;      // the ZBUF its depth buffer was last written as (pages), for depth read as a texture
        uint8_t zpsm = 0;
        uint64_t lastDraw = 0;     // m_drawSerial of its last batch
        // Per GS memory page it covers (fbp + i, m_pageVersion's numbering): the version its texture holds.
        std::vector<uint64_t> pages;
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
    // Coherence between render targets and GS memory, page by page (8 KiB: 64 x 32 pixels in CT32, 64 x 64 in
    // CT16). Every write to a page gives it a new version (m_pageVersion): a draw into a render target over it, a
    // transfer, a CPU write. GS memory (m_vram and its GPU copy, kept in step by m_dirtyBlocks / m_gpuNewer) holds
    // version m_memoryVersion of each page, each render target the versions in its `pages`. Before GS memory is read
    // (a texture in another format than the target's, a transfer, a CLUT, the display of a buffer no target drew),
    // the target holding a page's newest version writes it there (syncMemoryPages); before a target is drawn into or
    // sampled, its pages that are not the newest are reloaded from GS memory (syncTargetPages). Targets sharing
    // memory in different formats or widths (VP2's CT16 / CT32 views of one buffer and the blur levels inside its
    // pages, SCTPT2's 4-bit font in a frame buffer's alpha bits) then see each other's pixels as the GS would.
    std::vector<uint64_t> m_pageVersion = std::vector<uint64_t>(512u, 1u);
    std::vector<uint64_t> m_memoryVersion = std::vector<uint64_t>(512u, 1u);
    uint64_t m_version = 1;
    uint32_t targetPageCount(const RenderTarget &target) const;
    bool holdsNewest(const RenderTarget &target, uint32_t page) const;
    // The target's pages a rectangle of its pixels touches (x1, y1 exclusive).
    void targetPagesIn(const RenderTarget &target, int x0, int y0, int x1, int y1, std::vector<uint32_t> &pages) const;
    void syncMemoryPages(uint32_t firstPage, uint32_t endPage);
    void syncTargetPages(RenderTarget &target, const std::vector<uint32_t> &pages);
    void wroteTargetPages(RenderTarget &target, const std::vector<uint32_t> &pages);
    void wroteMemoryPages(uint32_t firstPage, uint32_t endPage);
    // A target's pages [first, end) written into GS memory: on the GPU copy, or through the CPU (diagnostics).
    void resolveTargetPages(RenderTarget &target, uint32_t firstPage, uint32_t endPage);
    bool resolveTargetPagesGpu(RenderTarget &target, uint32_t firstPage, uint32_t endPage);
    // The target's pages drawn again from what GS memory holds, in its format (a sprite per page sampling memory).
    void reloadTargetPages(RenderTarget &target, const std::vector<uint32_t> &pages);
    // A texture sampled straight from a render target (its base, format layout and width); null when the read goes
    // through GS memory.
    RenderTarget *directSource(const GSDrawState &state);
    std::vector<uint32_t> m_scratchPages;
    // A local-to-local transfer whose source is a render target, done on the GPU; false when it is not one.
    bool copyBetweenTargets(const GSTransferCommand &command);
    void appendPrimitive(const GSPrimitiveBatch &batch);
    void flushBatch();
    // Binds the batch's texture to unit 0 and gives its size in texels; false when it has none to sample.
    bool bindTexture(const GSDrawState &state, const RenderTarget &target, float &width, float &height);
    uint64_t m_drawSerial = 0;
    // A Z-format texture over the Z buffer a render target drew (depth of field, fog from depth): that depth as the
    // GS would read it as texels, into m_depthTexture. False when no render target wrote that Z buffer.
    bool bindDepthAsTexture(const GSDrawState &state, float &width, float &height);
    uint32_t m_depthTexture = 0;
    // The GPU decoding path (default; PS2X_GS_NATIVE_CPUDECODE=1 uses the texture cache's images): GS memory on the
    // GPU, brought up to date page by page from m_vram, and the formats' swizzle tables.
    bool bindDecodedTexture(const GSDrawState &state);
    uint32_t m_vramTexture = 0;
    uint32_t m_inverseTexture = 0;
    uint32_t m_vramFramebuffer = 0;
    uint32_t m_vramOld = 0, m_vramOldFramebuffer = 0; // CT24 resolves: GS memory's words before them
    uint32_t m_resolveProgram = 0;
    // A draw whose ZBUF is another render target's colour tests against that colour read as Z, in m_borrowedDepth,
    // attached to the draw's framebuffer for that draw. False when no colour target is there.
    bool depthFromColorTarget(const RenderTarget &target, const GSZbufReg &zbuf);
    // The other target whose depth attachment holds that Z buffer (it last drew Z there), the same size as this one.
    RenderTarget *depthOwner(const RenderTarget &target, const GSZbufReg &zbuf);
    uint32_t m_depthFromColorProgram = 0;
    uint32_t m_borrowedDepth = 0;
    uint32_t m_borrowedDepthWidth = 0;
    uint32_t m_borrowedDepthHeight = 0;
    void ensureVram();
    void syncVramToGpu();
    // Pages the GPU copy has newer than m_vram (resolved there): read back before the CPU reads or partly rewrites
    // GS memory (transfers, CLUT, local-to-host, presentation from memory).
    std::vector<uint8_t> m_gpuNewer = std::vector<uint8_t>(512u, 0u);
    bool m_anyGpuNewer = false;
    void pullGpuPages();
    void pullGpuPages(uint32_t firstPage, uint32_t endPage); // only GS memory pages in [first, end)
    uint32_t m_readFirstPage = 0, m_readEndPage = 0xFFFFFFFFu; // the current local-to-host transfer's source pages
    uint32_t m_swizzleTexture = 0;
    // 256-byte blocks of GS memory (m_vram, written in stream order by the transfers, CLUT writes and CPU resolves)
    // the GPU copy has not received yet.
    std::vector<uint8_t> m_dirtyBlocks = std::vector<uint8_t>(16384u, 1u);
    bool m_anyDirty = true;
    GSTransferCommand m_transfer{}; // the current transfer, whose data UploadImage brings
    uint32_t m_transferDirect = 3u;
    void markDirty(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t w, uint32_t h);
    uint32_t m_destTexture = 0; // copy of the render target for DATE
    uint32_t m_destWidth = 0;
    uint32_t m_destHeight = 0;
    void applyBlend(const GSDrawState &state);
    // What applyBlend asks of the shader: 0 its colour, 1 the blend factor as colour, 2 the colour times (1 + factor).
    int m_blendOut = 0;
    // One PCRTC read circuit's picture (RGBA8, alpha as the GS stores it), from its render target or else GS memory.
    // The presentation on the GPU (default; PS2X_GS_NATIVE_CPUPRESENT=1 reads the circuits back and merges them on the
    // CPU): the two read circuits merged as PMODE says into a texture the window draws (PresentationFrame::glTexture),
    // three in turn, the window drawing the one composed a presentation earlier while the next is composed.
    struct PresentSlot
    {
        uint32_t texture = 0, framebuffer = 0, width = 0, height = 0;
        void *fence = nullptr; // GLsync: the composition done
    };
    PresentSlot m_presentSlots[3];
    int m_presentNext = 0;
    int m_presentComposed = -1;
    uint32_t m_presentProgram = 0;
    uint32_t m_circuitStaging[3] = {}; // a circuit read from GS memory (no render target there), per circuit
    // A read circuit's picture on the GPU: a render target's texture, or GS memory read on the CPU into a staging one.
    struct CircuitSource
    {
        uint32_t texture = 0;
        int x = 0, y = 0, width = 0, height = 0; // GS pixels
        int scale = 1;                            // texels per GS pixel
        uint32_t fbp = 0;
        bool opaque = false; // CT24: no alpha (0x80)
    };
    bool circuitSource(int circuit, uint64_t dispfb, uint64_t display, bool fieldMode, CircuitSource &out);
    // The internal scale (PS2X_GS_SCALE=<n>, 1 to 4, default 1): render targets are n times the GS's size each way,
    // draws land on them with their viewport and scissor scaled, and whatever goes back to GS memory or the CPU
    // (resolves, read backs, a Z buffer read as a texture) is read from a 1x copy (readFramebufferAt1x).
    uint32_t m_scale = 1;
    uint32_t m_downFramebuffer = 0, m_downColor = 0, m_downDepth = 0;
    // A framebuffer to read a target's rectangle from at the GS's size: the target's own at scale 1, else a nearest
    // copy of the rectangle into m_downFramebuffer (colour, or depth). Leaves the draw framebuffer as it was.
    uint32_t readFramebufferAt1x(const RenderTarget &target, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                 bool depth = false);
    bool presentOnGpu(const GSPresentationRequest &request, PresentationFrame &frame);
    bool readCircuit(int circuit, uint64_t dispfb, uint64_t display, bool fieldMode, std::vector<uint8_t> &pixels,
                     uint32_t &width, uint32_t &height, uint32_t &fbp);

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
        int alphaTest = -1, alphaRef = -1, alphaFail = -1, sampler = -1, blendOut = -1, blendFactor = -1, blendFix = -1;
        int dest = -1, date = -1, colorKeep = -1;
        int vram = -1, swizzle = -1, decode = -1, texFmt = -1, texSel = -1, texKind = -1, clutFmt = -1, cbp = -1;
        int texa = -1, wrapU = -1, wrapV = -1, linear = -1, targetWrap = -1, sampleShift = -1;
    } m_uniforms;

    std::unordered_map<uint32_t, RenderTarget> m_targets;
    struct ReadbackBuffers
    {
        uint32_t pbo[2] = {0, 0};
        size_t bytes[2] = {0, 0};
        bool ready[2] = {false, false};
        int next = 0;
    };
    std::unordered_map<uint32_t, ReadbackBuffers> m_readbacks; // per read circuit, 1 or 2 (readCircuit)
    std::unordered_map<uint64_t, UploadedTexture> m_uploaded;
    struct LastLookup
    {
        uint64_t tex0 = 0, texa = 0, uploads = ~0ull;
        std::shared_ptr<const ps2x::native::TextureCache::Texture> texture;
    };
    std::unordered_map<uint64_t, LastLookup> m_lastLookups; // texture cache lookups since the last upload

    // The batch being collected: draws whose state draws the same way (sameDrawState).
    std::vector<Vertex> m_batch;
    bool m_fillingFromMemory = false; // reloadTargetPages' draw: textures from GS memory, never a target
    float m_batchMaxY = 0.0f; // the batch's lowest row, for its render target's usedHeight
    GSDrawState m_batchState{};
};
