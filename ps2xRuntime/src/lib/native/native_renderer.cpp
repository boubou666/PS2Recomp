#include "runtime/native/native_renderer.h"

#include "raylib.h"
#include "rlgl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace ps2x::native
{
    namespace
    {
        // The object's matrix takes a vertex to PS2 screen space (x, y in GS window coordinates, z GS depth, w); the
        // PS2 divides by w. Here the same point goes to the window: NDC from the screen rectangle, depth from the Z
        // range (larger is nearer on the GS), and w kept so texture coordinates interpolate in perspective and the
        // GPU clips what lies behind the camera (w <= 0: where the PS2 relies on the microprogram's clipping).
        const char *kVertexShader = R"(#version 330
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUv;
layout(location = 2) in vec4 aCol;
layout(location = 3) in float aFog;
uniform vec4 uM[4];
uniform vec4 uScreen;
uniform float uDepthMax;
out vec2 vUv;
out vec4 vCol;
out float vFog;
void main()
{
    vec4 c = uM[0] * aPos.x + uM[1] * aPos.y + uM[2] * aPos.z + uM[3];
    float w = c.w;
    vec2 s = c.xy / c.w;
    vec2 ndc = vec2((s.x - uScreen.x) / uScreen.z * 2.0 - 1.0, 1.0 - (s.y - uScreen.y) / uScreen.w * 2.0);
    float z = clamp(c.z / c.w / uDepthMax, 0.0, 1.0);
    gl_Position = vec4(ndc * w, (1.0 - 2.0 * z) * w, w);
    vUv = aUv;
    vCol = aCol;
    vFog = aFog;
}
)";

        // GS modulate: texel * vertex colour, where a vertex colour of 0x80 is 1.0, then fog toward the fog colour (RGB
        // only, before blending). The render target holds
        // premultiplied colour and coverage (opaque draws cover fully), so present() composites it over the window.
        const char *kFragmentShader = R"(#version 330
in vec2 vUv;
in vec4 vCol;
in float vFog;
uniform sampler2D uTex;
uniform int uTextured;
uniform int uAlphaTest;
uniform float uAlphaRef;
uniform int uOpaque;
uniform int uFog;
uniform vec3 uFogColour;
out vec4 fragColor;
void main()
{
    vec4 colour = vCol * (255.0 / 128.0);
    vec4 texel = uTextured != 0 ? texture(uTex, vUv) : vec4(1.0);
    fragColor = clamp(texel * colour, 0.0, 1.0);
    if (uAlphaTest != 0 && fragColor.a <= uAlphaRef)
        discard;
    if (uFog != 0)
        fragColor.rgb = mix(uFogColour, fragColor.rgb, clamp(vFog, 0.0, 1.0));
    if (uOpaque != 0)
        fragColor.a = 1.0;
}
)";

        constexpr int kGlUnsignedByte = 0x1401;
        constexpr int kGlFloat = 0x1406;
        constexpr int kGlSrcAlpha = 0x0302;
        constexpr int kGlOneMinusSrcAlpha = 0x0303;
        constexpr int kGlOne = 1;
        constexpr int kGlFuncAdd = 0x8006;
    }

    struct Renderer::Gpu
    {
        unsigned int shader = 0;
        int locM = -1, locScreen = -1, locDepthMax = -1, locTex = -1, locTextured = -1, locAlphaTest = -1, locAlphaRef = -1, locOpaque = -1, locFog = -1,
            locFogColour = -1;
        unsigned int vao = 0, vbo = 0;
        size_t vboBytes = 0;
        RenderTexture2D target{};
        int targetWidth = 0, targetHeight = 0;
        struct GpuTexture
        {
            unsigned int id = 0;
            std::shared_ptr<const TextureCache::Texture> source; // held, so its address is not reused while compared
        };
        std::unordered_map<uint64_t, GpuTexture> textures;

        bool init()
        {
            shader = rlLoadShaderCode(kVertexShader, kFragmentShader);
            if (shader == 0)
                return false;
            locM = rlGetLocationUniform(shader, "uM");
            locScreen = rlGetLocationUniform(shader, "uScreen");
            locDepthMax = rlGetLocationUniform(shader, "uDepthMax");
            locTex = rlGetLocationUniform(shader, "uTex");
            locTextured = rlGetLocationUniform(shader, "uTextured");
            locAlphaTest = rlGetLocationUniform(shader, "uAlphaTest");
            locAlphaRef = rlGetLocationUniform(shader, "uAlphaRef");
            locOpaque = rlGetLocationUniform(shader, "uOpaque");
            locFog = rlGetLocationUniform(shader, "uFog");
            locFogColour = rlGetLocationUniform(shader, "uFogColour");
            vao = rlLoadVertexArray();
            return true;
        }

        unsigned int texture(uint64_t key, const std::shared_ptr<const TextureCache::Texture> &source)
        {
            GpuTexture &slot = textures[key];
            if (slot.source != source || slot.id == 0)
            {
                if (slot.id != 0)
                    rlUnloadTexture(slot.id);
                slot.id = rlLoadTexture(source->rgba.data(), static_cast<int>(source->width), static_cast<int>(source->height),
                                        RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1);
                rlTextureParameters(slot.id, RL_TEXTURE_MAG_FILTER, RL_TEXTURE_FILTER_LINEAR);
                rlTextureParameters(slot.id, RL_TEXTURE_MIN_FILTER, RL_TEXTURE_FILTER_LINEAR);
                rlTextureParameters(slot.id, RL_TEXTURE_WRAP_S, RL_TEXTURE_WRAP_REPEAT);
                rlTextureParameters(slot.id, RL_TEXTURE_WRAP_T, RL_TEXTURE_WRAP_REPEAT);
                slot.source = source;
            }
            return slot.id;
        }
    };

    Renderer::Renderer() = default;
    Renderer::~Renderer() = default;

    Renderer &Renderer::instance()
    {
        static Renderer renderer;
        return renderer;
    }

    void Renderer::beginFrame(const Screen &screen)
    {
        m_building = Frame{};
        m_building.screen = screen;
    }

    void Renderer::draw(const Material &material, const float matrix[16], const Vertex *vertices, uint32_t count)
    {
        if (count == 0u)
            return;
        Batch batch;
        batch.material = material;
        std::memcpy(batch.matrix, matrix, sizeof(batch.matrix));
        if (material.textured && !material.fromGsContext)
            batch.texture = ps2x::native::textures().texture(material.tex0, material.texa);
        batch.first = static_cast<uint32_t>(m_building.vertices.size());
        batch.count = count;
        batch.key = m_key;
        batch.group = m_streamDraws ? m_group : 0u;
        m_building.lastGroup = std::max(m_building.lastGroup, batch.group);
        // Once the GS front end draws the GS-context batches itself (drawn into the GS frame), present() never draws
        // the frame: such a batch only goes to the stream.
        const bool frameCopy = !(m_streamDraws && material.fromGsContext && m_drewIntoGs.load(std::memory_order_relaxed));
        if (frameCopy)
            m_building.vertices.insert(m_building.vertices.end(), vertices, vertices + count);
        if (m_streamDraws)
        {
            StreamBatch entry;
            entry.group = m_group;
            entry.gsZScale = m_building.screen.gsZScale;
            entry.batch = batch;
            entry.vertices.assign(vertices, vertices + count);
            std::lock_guard<std::mutex> lock(m_streamMutex);
            m_stream.push_back(std::move(entry));
            if (m_stream.size() > 65536u) // nobody drains them (no GS markers): keep the latest frames' worth
                m_stream.pop_front();
        }
        if (frameCopy)
            m_building.batches.push_back(batch);
    }

    void Renderer::endFrame()
    {
        auto frame = std::make_shared<Frame>(std::move(m_building));
        m_building = Frame{};
        m_key = 0;
        frame->time = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        for (uint32_t i = 0; i < frame->batches.size(); ++i)
            if (frame->batches[i].key != 0u)
                frame->keys.emplace(frame->batches[i].key, i);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_previous = std::move(m_published);
        m_published = std::move(frame);
    }

    bool Renderer::hasDraws()
    {
        if (m_drewIntoGs.load(std::memory_order_relaxed))
            return false;
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_published && !m_published->batches.empty();
    }

    // native_renderer_gs.cpp (raw GL): the bound framebuffer's depth, width x height floats, bottom-up rows, from the
    // call before this one (false when there is none yet).
    bool readFramebufferDepth(int width, int height, std::vector<float> &out);

    namespace
    {
        double secondsNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    }

    void Renderer::wantDepth() { m_depthWantedAt.store(secondsNow(), std::memory_order_relaxed); }

    bool Renderer::depthAt(float gsX, float gsY, uint32_t &gsZ)
    {
        std::lock_guard<std::mutex> lock(m_depthMutex);
        if (m_depth.empty() || m_depthScreen.width <= 0.0f || m_depthScreen.height <= 0.0f)
            return false;
        const int x = static_cast<int>((gsX - m_depthScreen.originX) / m_depthScreen.width * static_cast<float>(m_depthWidth));
        const int y = static_cast<int>((gsY - m_depthScreen.originY) / m_depthScreen.height * static_cast<float>(m_depthHeight));
        if (x < 0 || y < 0 || x >= m_depthWidth || y >= m_depthHeight)
            return false;
        // present()'s depth: window depth 1 - (z / w) / depthMax (cleared to 1, far), GS Z = z / w * gsZScale.
        const float d = m_depth[static_cast<size_t>(m_depthHeight - 1 - y) * m_depthWidth + x];
        const double z = (1.0 - static_cast<double>(d)) * m_depthScreen.depthMax * m_depthScreen.gsZScale;
        gsZ = static_cast<uint32_t>(std::clamp(z, 0.0, 4294967295.0));
        return true;
    }

    bool Renderer::present(float x, float y, float width, float height)
    {
        std::shared_ptr<Frame> published, before;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            published = m_published;
            before = m_previous;
        }
        // The game publishes a frame before the GS reaches its groups' markers (their clips), often the next one too:
        // the newest of the two the GS has reached the last group of is shown (the latest one after 100 ms of
        // neither, markers that never come).
        {
            const uint64_t marked = m_markedGroup.load(std::memory_order_acquire);
            const auto reached = [marked](const std::shared_ptr<Frame> &f) { return f && (f->lastGroup == 0u || marked >= f->lastGroup); };
            std::shared_ptr<Frame> next = reached(published) ? published : reached(before) ? before : nullptr;
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (!next && published && (!m_shown || now - m_shown->time > 0.1))
                next = published;
            if (next && next != m_shown && (!m_shown || next->time > m_shown->time))
            {
                m_shownPrevious = std::move(m_shown);
                m_shown = std::move(next);
            }
        }
        const std::shared_ptr<Frame> frame = m_shown, previous = m_shownPrevious;
        if (!frame || !m_enabled || m_drewIntoGs.load(std::memory_order_relaxed))
            return false; // or the GS frame has the native draws already (drawn into it in the GS stream)
        if (!m_gpu)
        {
            m_gpu = std::make_unique<Gpu>();
            if (!m_gpu->init())
            {
                m_enabled = false;
                return false;
            }
        }
        Gpu &gpu = *m_gpu;
        const int w = std::max(1, static_cast<int>(width)), h = std::max(1, static_cast<int>(height));
        if (gpu.targetWidth != w || gpu.targetHeight != h)
        {
            if (gpu.targetWidth != 0)
                UnloadRenderTexture(gpu.target);
            gpu.target = LoadRenderTexture(w, h);
            gpu.targetWidth = w;
            gpu.targetHeight = h;
        }

        const size_t bytes = frame->vertices.size() * sizeof(Vertex);
        rlEnableVertexArray(gpu.vao);
        if (bytes > gpu.vboBytes)
        {
            if (gpu.vbo != 0)
                rlUnloadVertexBuffer(gpu.vbo);
            gpu.vboBytes = std::max(bytes, gpu.vboBytes * 2u);
            gpu.vbo = rlLoadVertexBuffer(nullptr, static_cast<int>(gpu.vboBytes), true);
            rlSetVertexAttribute(0, 3, kGlFloat, false, sizeof(Vertex), 0);
            rlSetVertexAttribute(1, 2, kGlFloat, false, sizeof(Vertex), 12);
            rlSetVertexAttribute(2, 4, kGlUnsignedByte, true, sizeof(Vertex), 20);
            rlSetVertexAttribute(3, 1, kGlFloat, false, sizeof(Vertex), 24);
            rlEnableVertexAttribute(0);
            rlEnableVertexAttribute(1);
            rlEnableVertexAttribute(2);
            rlEnableVertexAttribute(3);
        }
        if (bytes != 0u)
            rlUpdateVertexBuffer(gpu.vbo, frame->vertices.data(), static_cast<int>(bytes), 0);
        rlDisableVertexArray();

        BeginTextureMode(gpu.target);
        ClearBackground(BLANK);
        rlDrawRenderBatchActive();
        const Screen &screen = frame->screen;
        const float screenRect[4] = {screen.originX, screen.originY, screen.width, screen.height};
        const int slot = 0;
        rlDisableBackfaceCulling();
        rlEnableDepthTest();
        // PS2X_NATIVE_INTERPOLATE=1: matrices between the previous game frame and this one (setKey).
        static const bool s_interpolate = []
        {
            const char *value = std::getenv("PS2X_NATIVE_INTERPOLATE");
            return value && *value == '1';
        }();
        float alpha = 1.0f;
        static uint64_t s_matched = 0; // batches drawn interpolated since the last stats line
        if (s_interpolate && previous && frame->time > previous->time)
        {
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            alpha = static_cast<float>(std::clamp((now - frame->time) / (frame->time - previous->time), 0.0, 1.0));
        }
        // Each batch's clip (its group's GS scissor) in the target's pixels, bottom-up; x1 <= x0: none.
        struct Rect
        {
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        };
        std::vector<Rect> clips(frame->batches.size());
        {
            std::lock_guard<std::mutex> lock(m_streamMutex);
            for (size_t i = 0; i < frame->batches.size(); ++i)
            {
                const uint64_t group = frame->batches[i].group;
                const Clip &clip = m_clips[group % kClips];
                if (group == 0u || clip.group != group)
                    continue;
                const float sx = static_cast<float>(w) / screen.width, sy = static_cast<float>(h) / screen.height;
                // PS2X_NATIVE_LOG_CLIPS=1: now and then, a batch's clip.
                static const bool s_logClips = std::getenv("PS2X_NATIVE_LOG_CLIPS") != nullptr;
                static uint32_t s_clipLogs = 0;
                if (s_logClips && ++s_clipLogs % 5000u == 1u)
                    std::fprintf(stderr, "[native] clip group %llu: %.1f %.1f - %.1f %.1f (screen %.1f %.1f %.1fx%.1f)\n",
                                 static_cast<unsigned long long>(group), clip.x0, clip.y0, clip.x1, clip.y1, screen.originX,
                                 screen.originY, screen.width, screen.height);
                Rect &rect = clips[i];
                rect.x0 = std::clamp(static_cast<int>(std::lround((clip.x0 - screen.originX) * sx)), 0, w);
                rect.x1 = std::clamp(static_cast<int>(std::lround((clip.x1 - screen.originX) * sx)), 0, w);
                rect.y0 = std::clamp(static_cast<int>(std::lround(h - (clip.y1 - screen.originY) * sy)), 0, h);
                rect.y1 = std::clamp(static_cast<int>(std::lround(h - (clip.y0 - screen.originY) * sy)), 0, h);
                if (rect.x0 == 0 && rect.y0 == 0 && rect.x1 == w && rect.y1 == h)
                    rect = Rect{}; // the whole target: no clip
                else if (rect.x1 <= rect.x0 || rect.y1 <= rect.y0)
                    rect = Rect{0, 0, 0, -1}; // nothing shows
            }
        }
        bool scissored = false;
        for (size_t index = 0; index < frame->batches.size(); ++index)
        {
            const Batch &batch = frame->batches[index];
            const Material &m = batch.material;
            const Rect &clip = clips[index];
            if (clip.y1 < 0)
                continue;
            float matrix[16];
            std::memcpy(matrix, batch.matrix, sizeof(matrix));
            if (s_interpolate && previous && batch.key != 0u)
                if (const auto it = previous->keys.find(batch.key); it != previous->keys.end())
                {
                    ++s_matched;
                    const float *from = previous->batches[it->second].matrix;
                    for (int i = 0; i < 16; ++i)
                        matrix[i] = from[i] + (batch.matrix[i] - from[i]) * alpha;
                }
            // Blend state first: rlSetBlendMode flushes raylib's own batch, which unbinds the shader and the vertex
            // array (even with nothing to draw), so they are bound after it.
            if (m.blend == Material::Blend::Opaque)
                rlDisableColorBlend();
            else
            {
                rlEnableColorBlend();
                rlSetBlendFactorsSeparate(kGlSrcAlpha, m.blend == Material::Blend::Additive ? kGlOne : kGlOneMinusSrcAlpha,
                                          kGlOne, kGlOneMinusSrcAlpha, kGlFuncAdd, kGlFuncAdd);
                rlSetBlendMode(RL_BLEND_CUSTOM_SEPARATE);
            }
            if (m.depthWrite)
                rlEnableDepthMask();
            else
                rlDisableDepthMask();
            rlEnableShader(gpu.shader);
            rlEnableVertexArray(gpu.vao);
            rlSetUniform(gpu.locScreen, screenRect, RL_SHADER_UNIFORM_VEC4, 1);
            rlSetUniform(gpu.locDepthMax, &screen.depthMax, RL_SHADER_UNIFORM_FLOAT, 1);
            rlSetUniform(gpu.locTex, &slot, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(gpu.locM, matrix, RL_SHADER_UNIFORM_VEC4, 4);
            std::shared_ptr<const TextureCache::Texture> texture = batch.texture;
            if (m.textured && !texture)
                texture = ps2x::native::textures().texture(m.tex0, m.texa);
            // PS2X_NATIVE_DUMP_TEXTURE=<tex0, hex>: that texture as the cache decoded it, once, as texture_<tex0>.png
            // (in PS2X_NATIVE_CAPTURE_DIR, or the current folder).
            static const uint64_t s_dumpTex0 = []
            {
                const char *value = std::getenv("PS2X_NATIVE_DUMP_TEXTURE");
                return value && *value ? std::strtoull(value, nullptr, 16) : 0ull;
            }();
            static bool s_dumped = false;
            if (s_dumpTex0 != 0u && !s_dumped && texture && m.tex0 == s_dumpTex0)
            {
                s_dumped = true;
                Image image{const_cast<uint32_t *>(texture->rgba.data()), static_cast<int>(texture->width),
                            static_cast<int>(texture->height), 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
                const char *dir = std::getenv("PS2X_NATIVE_CAPTURE_DIR");
                char path[1024];
                std::snprintf(path, sizeof(path), "%s/texture_%016llx.png", dir && *dir ? dir : ".", static_cast<unsigned long long>(s_dumpTex0));
                ExportImage(image, path);
            }
            const int textured = m.textured && texture ? 1 : 0, alphaTest = m.alphaTest ? 1 : 0;
            const int opaque = m.blend == Material::Blend::Opaque ? 1 : 0;
            rlSetUniform(gpu.locOpaque, &opaque, RL_SHADER_UNIFORM_INT, 1);
            const int fog = m.fog ? 1 : 0;
            const float fogColour[3] = {static_cast<float>(m.fogColour & 0xffu) / 255.0f, static_cast<float>((m.fogColour >> 8) & 0xffu) / 255.0f,
                                        static_cast<float>((m.fogColour >> 16) & 0xffu) / 255.0f};
            rlSetUniform(gpu.locFog, &fog, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(gpu.locFogColour, fogColour, RL_SHADER_UNIFORM_VEC3, 1);
            const float alphaRef = m.alphaRef / 128.0f;
            rlSetUniform(gpu.locTextured, &textured, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(gpu.locAlphaTest, &alphaTest, RL_SHADER_UNIFORM_INT, 1);
            rlSetUniform(gpu.locAlphaRef, &alphaRef, RL_SHADER_UNIFORM_FLOAT, 1);
            if (textured)
            {
                const unsigned int id = gpu.texture(m.tex0 ^ (m.texa * 0x9e3779b97f4a7c15ull), texture);
                // Wrap per draw (rlTextureParameters binds and unbinds the texture: before enabling it).
                rlTextureParameters(id, RL_TEXTURE_WRAP_S, m.clampU ? RL_TEXTURE_WRAP_CLAMP : RL_TEXTURE_WRAP_REPEAT);
                rlTextureParameters(id, RL_TEXTURE_WRAP_T, m.clampV ? RL_TEXTURE_WRAP_CLAMP : RL_TEXTURE_WRAP_REPEAT);
                rlActiveTextureSlot(0);
                rlEnableTexture(id);
            }
            if (clip.x1 > clip.x0)
            {
                rlEnableScissorTest();
                rlScissor(clip.x0, clip.y0, clip.x1 - clip.x0, clip.y1 - clip.y0);
                scissored = true;
            }
            else if (scissored)
            {
                rlDisableScissorTest();
                scissored = false;
            }
            rlDrawVertexArray(static_cast<int>(batch.first), static_cast<int>(batch.count));
        }
        rlDisableScissorTest();
        if (secondsNow() - m_depthWantedAt.load(std::memory_order_relaxed) < 1.0)
        {
            std::vector<float> depth;
            if (readFramebufferDepth(w, h, depth))
            {
                std::lock_guard<std::mutex> lock(m_depthMutex);
                m_depth.swap(depth);
                m_depthWidth = w;
                m_depthHeight = h;
                m_depthScreen = screen;
            }
        }
        rlDisableVertexArray();
        rlDisableShader();
        rlDisableTexture();
        rlEnableDepthMask();
        rlDisableDepthTest();
        rlEnableColorBlend();
        rlSetBlendMode(RL_BLEND_ALPHA);
        EndTextureMode();

        static const bool s_stats = []
        {
            const char *value = std::getenv("PS2X_NATIVE_STATS");
            return value && *value == '1';
        }();
        static uint32_t s_presents = 0;
        if (s_stats && ++s_presents % 250u == 0u)
        {
            std::fprintf(stderr, "[native] present %u: %zu batches, %zu vertices, shader %u (default %u), target %dx%d, alpha %.2f, interpolated batches %llu\n",
                         s_presents, frame->batches.size(), frame->vertices.size(), gpu.shader, rlGetShaderIdDefault(), w, h, alpha,
                         static_cast<unsigned long long>(s_matched));
            s_matched = 0;
        }

        // PS2X_NATIVE_CAPTURE_DIR=<dir>: the native frame as native_<n>.png every PS2X_NATIVE_CAPTURE_INTERVAL seconds
        // (default 5), at the window's resolution, to look at it without the GS frame; with gs_<n>.png, the GS frame of
        // the same moment (a reference to compare to, when the game still runs on VU1 what the native port draws).
        static const char *s_captureDir = std::getenv("PS2X_NATIVE_CAPTURE_DIR");
        if (s_captureDir && *s_captureDir)
        {
            static const double s_interval = []
            {
                const char *value = std::getenv("PS2X_NATIVE_CAPTURE_INTERVAL");
                return value && *value ? std::atof(value) : 5.0;
            }();
            static double s_next = 0.0;
            static int s_count = 0;
            const double now = GetTime();
            if (now >= s_next)
            {
                s_next = now + s_interval;
                Image image = LoadImageFromTexture(gpu.target.texture);
                ImageFlipVertical(&image);
                const std::string path = std::string(s_captureDir) + "/native_" + std::to_string(++s_count) + ".png";
                ExportImage(image, path.c_str());
                UnloadImage(image);
                if (m_gsFrame[0] != 0)
                {
                    const Texture2D gs{static_cast<unsigned int>(m_gsFrame[0]), m_gsFrame[1], m_gsFrame[2], 1, m_gsFrame[3]};
                    Image gsImage = LoadImageFromTexture(gs);
                    const std::string gsPath = std::string(s_captureDir) + "/gs_" + std::to_string(s_count) + ".png";
                    ExportImage(gsImage, gsPath.c_str());
                    UnloadImage(gsImage);
                }
            }
        }

        // Render textures are stored bottom-up.
        BeginBlendMode(BLEND_ALPHA_PREMULTIPLY);
        DrawTexturePro(gpu.target.texture, Rectangle{0, 0, static_cast<float>(w), -static_cast<float>(h)},
                       Rectangle{x, y, width, height}, Vector2{0, 0}, 0.0f, WHITE);
        EndBlendMode();
        return true;
    }
}
