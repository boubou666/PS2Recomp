// The mesh renderer drawing inside the GS stream (runtime/native/native_gs_hook.h): where VU1 skips a program the
// native port draws, the GS thread calls gsDraw with the render target that program's output would have gone to,
// and the batches of that program's group are drawn into it there, so they sort with the GS draws around them.
// Runs on the GS thread with the GS renderer's GL context current: raw GL (rlgl's state belongs to the window's
// thread), its own vertex array (not shared between contexts) and its own copies of the textures.

#include "runtime/native/native_renderer.h"
#include "runtime/native/native_gs_hook.h"

#include "external/glad.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace ps2x::native
{
    namespace
    {
        // The GS renderer's conventions: x, y in the target's pixels (window coordinates minus XYOFFSET) to NDC
        // without a flip, depth = GS Z times the Z buffer's scale, larger nearer (GEQUAL).
        const char *kVertexShader = R"(#version 330
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUv;
layout(location = 2) in vec4 aCol;
layout(location = 3) in float aFog;
uniform vec4 uM[4];
uniform vec2 uOffset;
uniform vec2 uTargetSize;
uniform float uGsZScale;     // GS Z per unit of the matrix's z / w
uniform float uTargetZScale; // the target's depth per GS Z (1/2^24 for Z24)
out vec2 vUv;
out vec4 vCol;
out float vFog;
void main()
{
    vec4 c = uM[0] * aPos.x + uM[1] * aPos.y + uM[2] * aPos.z + uM[3];
    vec2 s = c.xy / c.w - uOffset;
    vec2 ndc = s / uTargetSize * 2.0 - 1.0;
    // GS Z as XYZF2 holds it: 24 bits, so a value past 2^24 wraps (a program's FTOI4 into XYZF2 keeps the low 24
    // bits; Twisted Metal: Black's HUD drums sit behind their bezel that way). Kept in the target's range by
    // GL_DEPTH_CLAMP rather than clipped.
    float gz = c.z / c.w * uGsZScale;
    gz -= 16777216.0 * floor(gz / 16777216.0);
    gl_Position = vec4(ndc * c.w, (2.0 * gz * uTargetZScale - 1.0) * c.w, c.w);
    vUv = aUv;
    vCol = aCol;
    vFog = aFog;
}
)";

        // As the present pass, but every draw stores its source alpha as the GS does (0x80 = 1.0, byte 128); a blend
        // takes As at its own scale from the second output.
        const char *kFragmentShader = R"(#version 330
in vec2 vUv;
in vec4 vCol;
in float vFog;
uniform sampler2D uTex;
uniform int uTextured;
uniform int uAlphaTest;
uniform float uAlphaRef;
uniform int uFog;
uniform vec3 uFogColour;
uniform int uDebug;
uniform int uTfx;
uniform int uTcc;
layout(location = 0, index = 0) out vec4 fragColor;
layout(location = 0, index = 1) out vec4 fragFactor;
void main()
{
    vec4 colour = vCol * (255.0 / 128.0);
    vec4 c = colour;
    if (uTextured != 0)
    {
        vec4 texel = texture(uTex, vUv);
        texel.a *= 255.0 / 128.0; // the GS's alpha (TextureCache: 0x80 is 1.0)
        float highlight = uTfx >= 2 ? vCol.a : 0.0;
        c.rgb = uTfx == 1 ? texel.rgb : texel.rgb * colour.rgb + highlight;
        if (uTcc != 0)
            c.a = uTfx == 0 ? texel.a * colour.a : uTfx == 2 ? texel.a + colour.a : texel.a;
    }
    c = clamp(c, vec4(0.0), vec4(1.0, 1.0, 1.0, 255.0 / 128.0));
    if (uAlphaTest != 0 && c.a <= uAlphaRef)
        discard;
    if (uFog != 0)
        c.rgb = mix(uFogColour, c.rgb, clamp(vFog, 0.0, 1.0));
    fragFactor = vec4(min(c.a, 1.0));
    c.a *= 128.0 / 255.0;
    if (uDebug != 0)
        c = vec4(1.0, 0.0, 1.0, 128.0 / 255.0);
    fragColor = c;
}
)";

        GLuint compile(GLenum type, const char *source)
        {
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            GLint ok = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (!ok)
            {
                char log[1024] = {};
                glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
                std::fprintf(stderr, "[native] GS pass shader: %s\n", log);
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        }
    }

    struct Renderer::GsPass
    {
        GLuint program = 0, vao = 0, vbo = 0;
        GLint locM = -1, locOffset = -1, locTargetSize = -1, locGsZScale = -1, locTargetZScale = -1, locTex = -1, locTextured = -1;
        GLint locTfx = -1, locTcc = -1;
        GLint locAlphaTest = -1, locAlphaRef = -1, locFog = -1, locFogColour = -1;
        bool failed = false;
        struct GpuTexture
        {
            GLuint id = 0;
            std::shared_ptr<const TextureCache::Texture> source;
        };
        std::unordered_map<uint64_t, GpuTexture> textures;

        bool init()
        {
            const GLuint vs = compile(GL_VERTEX_SHADER, kVertexShader), fs = compile(GL_FRAGMENT_SHADER, kFragmentShader);
            if (vs == 0 || fs == 0)
                return false;
            program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, fs);
            glLinkProgram(program);
            glDeleteShader(vs);
            glDeleteShader(fs);
            GLint ok = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &ok);
            if (!ok)
                return false;
            locM = glGetUniformLocation(program, "uM");
            locOffset = glGetUniformLocation(program, "uOffset");
            locTargetSize = glGetUniformLocation(program, "uTargetSize");
            locGsZScale = glGetUniformLocation(program, "uGsZScale");
            locTargetZScale = glGetUniformLocation(program, "uTargetZScale");
            locTex = glGetUniformLocation(program, "uTex");
            locTextured = glGetUniformLocation(program, "uTextured");
            locTfx = glGetUniformLocation(program, "uTfx");
            locTcc = glGetUniformLocation(program, "uTcc");
            locAlphaTest = glGetUniformLocation(program, "uAlphaTest");
            locAlphaRef = glGetUniformLocation(program, "uAlphaRef");
            locFog = glGetUniformLocation(program, "uFog");
            locFogColour = glGetUniformLocation(program, "uFogColour");
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(0));
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(12));
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), reinterpret_cast<const void *>(20));
            glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(24));
            for (GLuint i = 0; i < 4; ++i)
                glEnableVertexAttribArray(i);
            glBindVertexArray(0);
            return true;
        }

        GLuint texture(uint64_t key, const std::shared_ptr<const TextureCache::Texture> &source)
        {
            GpuTexture &slot = textures[key];
            if (slot.source != source || slot.id == 0)
            {
                if (slot.id == 0)
                    glGenTextures(1, &slot.id);
                // The GS renderer's unpack state stays its own (a pixel buffer bound for its transfers would make
                // this read from it): plain client memory, tightly packed, then back as it was.
                GLint unpackBuffer = 0, rowLength = 0, skipRows = 0, skipPixels = 0, alignment = 4;
                glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpackBuffer);
                glGetIntegerv(GL_UNPACK_ROW_LENGTH, &rowLength);
                glGetIntegerv(GL_UNPACK_SKIP_ROWS, &skipRows);
                glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &skipPixels);
                glGetIntegerv(GL_UNPACK_ALIGNMENT, &alignment);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
                glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                glBindTexture(GL_TEXTURE_2D, slot.id);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(source->width), static_cast<GLsizei>(source->height), 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, source->rgba.data());
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(unpackBuffer));
                glPixelStorei(GL_UNPACK_ROW_LENGTH, rowLength);
                glPixelStorei(GL_UNPACK_SKIP_ROWS, skipRows);
                glPixelStorei(GL_UNPACK_SKIP_PIXELS, skipPixels);
                glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
                slot.source = source;
            }
            return slot.id;
        }
    };

    bool readFramebufferDepth(int width, int height, std::vector<float> &out)
    {
        // Asynchronous: this frame's depth goes into one pixel buffer while the other, read the frame before, is
        // copied out (a synchronous read made the present thread wait for the GPU every frame).
        static GLuint s_buffers[2] = {};
        static int s_width = 0, s_height = 0, s_next = 0;
        static bool s_filled[2] = {};
        const GLsizeiptr bytes = static_cast<GLsizeiptr>(width) * height * 4;
        GLint packBuffer = 0, alignment = 4;
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuffer);
        glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
        if (s_buffers[0] == 0)
            glGenBuffers(2, s_buffers);
        if (width != s_width || height != s_height)
        {
            for (GLuint buffer : s_buffers)
            {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer);
                glBufferData(GL_PIXEL_PACK_BUFFER, bytes, nullptr, GL_STREAM_READ);
            }
            s_width = width, s_height = height;
            s_filled[0] = s_filled[1] = false;
        }
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, s_buffers[s_next]);
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        s_filled[s_next] = true;
        const int previous = 1 - s_next;
        s_next = previous;
        bool copied = false;
        if (s_filled[previous])
        {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, s_buffers[previous]);
            if (const void *data = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT))
            {
                out.resize(static_cast<size_t>(width) * height);
                std::memcpy(out.data(), data, static_cast<size_t>(bytes));
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                copied = true;
            }
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(packBuffer));
        glPixelStorei(GL_PACK_ALIGNMENT, alignment);
        return copied;
    }

    namespace
    {
        // A fromGsContext material completed with the GS registers at its group's marker.
        Material materialFromGs(Material m, const GsDrawTarget &target)
        {
            m.tex0 = target.tex0;
            m.texa = target.texa;
            m.blend = m.alphaBlend ? blendOf(target.alpha) : Material::Blend::Opaque;
            const bool ate = target.test & 1u;
            const uint32_t atst = (target.test >> 1) & 7u, aref = (target.test >> 4) & 0xFFu;
            m.alphaTest = ate && (atst == 5u || atst == 6u); // GEQUAL / GREATER (the others: drawn as always)
            m.alphaRef = static_cast<uint8_t>(atst == 5u && aref > 0u ? aref - 1u : aref);
            const bool zte = (target.test >> 16) & 1u;
            const uint32_t ztst = (target.test >> 17) & 3u;
            m.depthTest = zte && ztst != 1u;
            m.depthWrite = !target.zmask;
            m.clampU = (target.clamp & 3u) == 1u;
            m.clampV = ((target.clamp >> 2) & 3u) == 1u;
            m.fogColour = target.fogcol;
            return m;
        }
    }

    void Renderer::gsDraw(uint64_t group, const GsDrawTarget &target)
    {
        instance().drawGroups(group, target);
    }

    void Renderer::setStreamDraws(bool on)
    {
        m_streamDraws = on;
        setGsDrawHook(on ? &Renderer::gsDraw : nullptr);
        setUploadsInStreamOrder(on);
    }

    bool Renderer::takeGsContextBatches(uint64_t group, StreamVisitor visit, void *user)
    {
        std::vector<StreamBatch> taken;
        bool left = false;
        {
            std::lock_guard<std::mutex> lock(m_streamMutex);
            for (auto it = m_stream.begin(); it != m_stream.end() && it->group <= group;)
            {
                if (it->batch.material.fromGsContext)
                {
                    taken.push_back(std::move(*it));
                    it = m_stream.erase(it);
                }
                else
                {
                    left = true;
                    ++it;
                }
            }
        }
        if (!taken.empty()) // the GS frame has them: present() must not draw the frame over it again
            m_drewIntoGs.store(true, std::memory_order_relaxed);
        for (const StreamBatch &entry : taken)
            visit(user, entry.batch.material, entry.batch.matrix, entry.gsZScale, entry.vertices.data(),
                  static_cast<uint32_t>(entry.vertices.size()));
        return left;
    }

    bool Renderer::hasGroup(uint64_t group)
    {
        std::lock_guard<std::mutex> lock(m_streamMutex);
        const auto it = std::lower_bound(m_stream.begin(), m_stream.end(), group,
                                         [](const StreamBatch &entry, uint64_t g) { return entry.group < g; });
        return it != m_stream.end() && it->group == group;
    }

    void Renderer::drawGroups(uint64_t group, const GsDrawTarget &target)
    {
        // PS2X_NATIVE_GS_STATS=1: every 2 s, the markers drawn, their batches and vertices, the time spent drawing
        // them on the GS thread and the textures the cache decoded meanwhile.
        static const bool s_stats = [] { const char *v = std::getenv("PS2X_NATIVE_GS_STATS"); return v && *v == '1'; }();
        struct Stats
        {
            uint64_t markers = 0, batches = 0, vertices = 0, decodes = 0;
            double seconds = 0.0;
            std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
        };
        static Stats s_window;
        const auto statsStart = std::chrono::steady_clock::now();
        struct StatsScope
        {
            Stats &w;
            std::chrono::steady_clock::time_point start;
            ~StatsScope()
            {
                if (!s_stats)
                    return;
                const auto now = std::chrono::steady_clock::now();
                w.seconds += std::chrono::duration<double>(now - start).count();
                ++w.markers;
                if (now - w.since >= std::chrono::seconds(2))
                {
                    const uint64_t decodes = ps2x::native::textures().decodes();
                    std::fprintf(stderr, "[native-gs] 2 s: %llu markers, %llu batches, %llu vertices, %.1f ms drawing, %llu texture decodes\n",
                                 static_cast<unsigned long long>(w.markers), static_cast<unsigned long long>(w.batches),
                                 static_cast<unsigned long long>(w.vertices), w.seconds * 1000.0,
                                 static_cast<unsigned long long>(decodes - w.decodes));
                    const uint64_t keep = decodes;
                    w = Stats{};
                    w.decodes = keep;
                }
            }
        } statsScope{s_window, statsStart};
        std::vector<StreamBatch> batches;
        {
            std::lock_guard<std::mutex> lock(m_streamMutex);
            while (!m_stream.empty() && m_stream.front().group <= group)
            {
                batches.push_back(std::move(m_stream.front()));
                m_stream.pop_front();
            }
        }
        // PS2X_NATIVE_GS_LOG=1: the first markers, their batches and the target they draw into.
        static const bool s_log = [] { const char *v = std::getenv("PS2X_NATIVE_GS_LOG"); return v && *v == '1'; }();
        static int s_logged = 0;
        if (s_log && s_logged < 40)
        {
            ++s_logged;
            std::fprintf(stderr, "[native-gs] group %llu: %zu batch(es), target fb %u %ux%u off %u,%u scissor %u,%u-%u,%u tex0 %#llx\n",
                         static_cast<unsigned long long>(group), batches.size(), target.framebuffer, target.width, target.height,
                         target.xOffset, target.yOffset, target.scissorX0, target.scissorY0, target.scissorX1, target.scissorY1,
                         static_cast<unsigned long long>(target.tex0));
        }
        if (target.framebuffer == 0u)
        {
            // A GS renderer without GL targets (CPU, paraLLEl): present() draws these groups, clipped as the GS has
            // them here (the groups before this marker had none of their own: the same state).
            const float ox = static_cast<float>(target.xOffset) / 16.0f, oy = static_cast<float>(target.yOffset) / 16.0f;
            const uint64_t from = batches.empty() ? group : std::min(batches.front().group, group);
            std::lock_guard<std::mutex> lock(m_streamMutex);
            for (uint64_t g = std::max(from, group > kClips ? group - kClips + 1u : 1u); g <= group; ++g)
            {
                Clip &clip = m_clips[g % kClips];
                clip.group = g;
                clip.x0 = ox + static_cast<float>(target.scissorX0);
                clip.y0 = oy + static_cast<float>(target.scissorY0);
                clip.x1 = ox + static_cast<float>(target.scissorX1) + 1.0f;
                clip.y1 = oy + static_cast<float>(target.scissorY1) + 1.0f;
            }
            m_markedGroup.store(group, std::memory_order_release);
            return;
        }
        if (batches.empty())
            return; // nothing for this group
        if (!m_gsPass)
        {
            m_gsPass = new GsPass();
            m_gsPass->failed = !m_gsPass->init();
        }
        GsPass &pass = *m_gsPass;
        if (pass.failed)
            return;
        m_drewIntoGs.store(true, std::memory_order_relaxed);

        std::vector<Vertex> vertices;
        for (const StreamBatch &entry : batches)
            vertices.insert(vertices.end(), entry.vertices.begin(), entry.vertices.end());
        s_window.batches += batches.size();
        s_window.vertices += vertices.size();
        glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
        const uint32_t scale = std::max(target.scale, 1u);
        glViewport(0, 0, static_cast<GLsizei>(target.width * scale), static_cast<GLsizei>(target.height * scale));
        glEnable(GL_SCISSOR_TEST);
        const uint32_t x0 = std::min(target.scissorX0, target.width), y0 = std::min(target.scissorY0, target.height);
        const uint32_t x1 = std::min(target.scissorX1 + 1u, target.width), y1 = std::min(target.scissorY1 + 1u, target.height);
        glScissor(static_cast<GLint>(x0 * scale), static_cast<GLint>(y0 * scale),
                  static_cast<GLsizei>((x1 > x0 ? x1 - x0 : 0u) * scale), static_cast<GLsizei>((y1 > y0 ? y1 - y0 : 0u) * scale));
        glDisable(GL_CULL_FACE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glUseProgram(pass.program);
        glBindVertexArray(pass.vao);
        glBindBuffer(GL_ARRAY_BUFFER, pass.vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STREAM_DRAW);
        const float offset[2] = {static_cast<float>(target.xOffset) / 16.0f, static_cast<float>(target.yOffset) / 16.0f};
        const float size[2] = {static_cast<float>(target.width), static_cast<float>(target.height)};
        glUniform2fv(pass.locOffset, 1, offset);
        glUniform2fv(pass.locTargetSize, 1, size);
        glUniform1i(pass.locTex, 0);
        // PS2X_NATIVE_GS_DEBUG=1: every native fragment magenta (where the draws land in the GS frame).
        static const int s_debug = []
        {
            const char *value = std::getenv("PS2X_NATIVE_GS_DEBUG");
            return value && *value == '1' ? 1 : 0;
        }();
        static const GLint s_locDebug = glGetUniformLocation(pass.program, "uDebug"); // one program per GS thread
        glUniform1i(s_locDebug, s_debug);
        glActiveTexture(GL_TEXTURE0);
        // Our textures' own filtering, not a sampler object the GS renderer left on unit 0.
        GLint sampler = 0;
        glGetIntegerv(GL_SAMPLER_BINDING, &sampler);
        glBindSampler(0, 0);

        GLint first = 0;
        for (const StreamBatch &entry : batches)
        {
            const Material m = entry.batch.material.fromGsContext ? materialFromGs(entry.batch.material, target) : entry.batch.material;
            const GLsizei count = static_cast<GLsizei>(entry.vertices.size());
            // PS2X_NATIVE_STATE_DIFF=1: a port's own material against the GS's state at the batch's marker (what
            // materialFromGs would give), field by field; each distinct difference logged once, with a count every
            // 20000 batches (a port finding the state its translation gets wrong).
            static const bool s_stateDiff = std::getenv("PS2X_NATIVE_STATE_DIFF") != nullptr;
            if (s_stateDiff && !entry.batch.material.fromGsContext)
            {
                const Material g = materialFromGs(entry.batch.material, target);
                char what[256];
                const int n = std::snprintf(what, sizeof(what), "%s%s%s%s%s%s%s",
                                            m.blend != g.blend ? " blend" : "", m.alphaTest != g.alphaTest || (m.alphaTest && m.alphaRef != g.alphaRef) ? " alphaTest" : "",
                                            m.depthTest != g.depthTest ? " depthTest" : "", m.depthWrite != g.depthWrite ? " depthWrite" : "",
                                            m.clampU != g.clampU || m.clampV != g.clampV ? " clamp" : "", m.texa != g.texa ? " texa" : "",
                                            m.fog && m.fogColour != g.fogColour ? " fogColour" : "");
                if (n > 0 && what[0])
                {
                    static std::unordered_map<std::string, uint64_t> s_seen;
                    char key[512];
                    std::snprintf(key, sizeof(key), "%s | mine blend %d at %d/%u dt %d dw %d clamp %d%d texa %llx | GS blend %d at %d/%u dt %d dw %d clamp %d%d texa %llx (alpha %llx test %llx clamp %llx)",
                                  what, static_cast<int>(m.blend), m.alphaTest ? 1 : 0, m.alphaRef, m.depthTest ? 1 : 0, m.depthWrite ? 1 : 0,
                                  m.clampU ? 1 : 0, m.clampV ? 1 : 0, static_cast<unsigned long long>(m.texa), static_cast<int>(g.blend),
                                  g.alphaTest ? 1 : 0, g.alphaRef, g.depthTest ? 1 : 0, g.depthWrite ? 1 : 0, g.clampU ? 1 : 0, g.clampV ? 1 : 0,
                                  static_cast<unsigned long long>(g.texa), static_cast<unsigned long long>(target.alpha),
                                  static_cast<unsigned long long>(target.test), static_cast<unsigned long long>(target.clamp));
                    if (s_seen[key]++ == 0u)
                        std::fprintf(stderr, "[native:statediff] new:%s\n", key);
                    static uint64_t s_batches = 0;
                    if (++s_batches % 20000u == 0u)
                        for (const auto &[k, c] : s_seen)
                            std::fprintf(stderr, "[native:statediff] %llu x%s\n", static_cast<unsigned long long>(c), k.c_str());
                }
            }
            glUniform1f(pass.locGsZScale, entry.gsZScale);
            glUniform1f(pass.locTargetZScale, static_cast<float>(target.zScale));
            glUniform4fv(pass.locM, 4, entry.batch.matrix);
            std::shared_ptr<const TextureCache::Texture> texture = entry.batch.texture;
            if (m.textured && !texture)
                texture = ps2x::native::textures().texture(m.tex0, m.texa, m.region);
            const bool textured = m.textured && texture;
            // PS2X_NATIVE_TEXSTATS=1: each texture a port's own draw uses, once: TEX0's format, CLUT and size, and the
            // share of fully transparent texels; over half transparent with TCC set, written out as tex_<tex0>.ppm (RGB)
            // and .pgm (alpha) in the working directory, to see what was decoded.
            static const bool s_texStats = std::getenv("PS2X_NATIVE_TEXSTATS") != nullptr;
            if (s_texStats && textured && !entry.batch.material.fromGsContext)
            {
                static std::unordered_map<uint64_t, bool> s_seenTex;
                if (!s_seenTex[texture->key])
                {
                    s_seenTex[texture->key] = true;
                    size_t clear = 0;
                    for (uint32_t c : texture->rgba)
                        clear += (c >> 24) == 0u ? 1u : 0u;
                    const double share = texture->rgba.empty() ? 0.0 : static_cast<double>(clear) / static_cast<double>(texture->rgba.size());
                    const uint64_t t0 = m.tex0;
                    const bool tcc = (t0 >> 34) & 1u;
                    std::fprintf(stderr, "[native:tex] tex0 %016llx psm %#x tbp %llu cbp %llu cpsm %llu csa %llu tcc %d texa %llx %ux%u: %.0f%% clear\n",
                                 static_cast<unsigned long long>(t0), static_cast<unsigned>((t0 >> 20) & 0x3f),
                                 static_cast<unsigned long long>(t0 & 0x3fff), static_cast<unsigned long long>((t0 >> 37) & 0x3fff),
                                 static_cast<unsigned long long>((t0 >> 51) & 0xf), static_cast<unsigned long long>((t0 >> 56) & 0x1f), tcc ? 1 : 0,
                                 static_cast<unsigned long long>(m.texa), texture->width, texture->height, share * 100.0);
                    if (share > 0.5)
                    {
                        std::string pages;
                        for (const auto &[page, generation] : texture->pages)
                            pages += " " + std::to_string(page) + "@" + std::to_string(generation);
                        std::fprintf(stderr, "[native:tex]   pages (page@generation, 0 never uploaded):%s\n", pages.c_str());
                    }
                    if (tcc && share > 0.5)
                    {
                        char name[96];
                        std::snprintf(name, sizeof(name), "tex_%016llx", static_cast<unsigned long long>(t0));
                        const std::string base(name);
                        if (FILE *f = std::fopen((base + ".ppm").c_str(), "wb"))
                        {
                            std::fprintf(f, "P6\n%u %u\n255\n", texture->width, texture->height);
                            for (uint32_t c : texture->rgba)
                            {
                                const unsigned char rgb[3] = {static_cast<unsigned char>(c), static_cast<unsigned char>(c >> 8), static_cast<unsigned char>(c >> 16)};
                                std::fwrite(rgb, 1, 3, f);
                            }
                            std::fclose(f);
                        }
                        if (FILE *f = std::fopen((base + ".pgm").c_str(), "wb"))
                        {
                            std::fprintf(f, "P5\n%u %u\n255\n", texture->width, texture->height);
                            for (uint32_t c : texture->rgba)
                            {
                                const unsigned char a = static_cast<unsigned char>(c >> 24);
                                std::fwrite(&a, 1, 1, f);
                            }
                            std::fclose(f);
                        }
                    }
                }
            }
            if (textured)
            {
                glBindTexture(GL_TEXTURE_2D, pass.texture(entry.batch.ownTexture ? (texture->version | 1ull << 63)
                                                                                 : m.tex0 ^ (m.texa * 0x9e3779b97f4a7c15ull) ^ (m.region * 0xc2b2ae3d27d4eb4full),
                                                          texture));
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, m.clampU ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, m.clampV ? GL_CLAMP_TO_EDGE : GL_REPEAT);
            }
            glUniform1i(pass.locTextured, textured ? 1 : 0);
            glUniform1i(pass.locTfx, static_cast<GLint>((m.tex0 >> 35) & 3u));
            glUniform1i(pass.locTcc, static_cast<GLint>((m.tex0 >> 34) & 1u));
            glUniform1i(pass.locAlphaTest, m.alphaTest ? 1 : 0);
            glUniform1f(pass.locAlphaRef, m.alphaRef / 128.0f);
            glUniform1i(pass.locFog, m.fog ? 1 : 0);
            const float fogColour[3] = {static_cast<float>(m.fogColour & 0xffu) / 255.0f, static_cast<float>((m.fogColour >> 8) & 0xffu) / 255.0f,
                                        static_cast<float>((m.fogColour >> 16) & 0xffu) / 255.0f};
            glUniform3fv(pass.locFogColour, 1, fogColour);
            if (m.blend == Material::Blend::Opaque)
                glDisable(GL_BLEND);
            else
            {
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_ADD);
                if (m.blend == Material::Blend::AdditiveByDestination)
                    glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
                else
                    glBlendFuncSeparate(GL_SRC1_ALPHA, m.blend == Material::Blend::Additive ? GL_ONE : GL_ONE_MINUS_SRC1_ALPHA, GL_ONE, GL_ZERO);
            }
            if (m.depthTest)
            {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_GEQUAL);
            }
            else
                glDisable(GL_DEPTH_TEST);
            glDepthMask(m.depthWrite ? GL_TRUE : GL_FALSE);
            glEnable(GL_DEPTH_CLAMP); // depth beyond the target's range clamped, not clipped (see the vertex shader)
            glDrawArrays(GL_TRIANGLES, first, count);
            if (m.blend == Material::Blend::AdditiveByDestination) // DST_ALPHA is Ad / 255, half the GS's Ad / 128
            {
                glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ONE, GL_ZERO);
                glDrawArrays(GL_TRIANGLES, first, count);
            }
            first += count;
        }
        glDisable(GL_DEPTH_CLAMP);
        glBindSampler(0, static_cast<GLuint>(sampler));
        glBindVertexArray(0);
        glUseProgram(0);
        glDepthMask(GL_TRUE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
    }
}
