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
uniform float uDepthScale;
out vec2 vUv;
out vec4 vCol;
out float vFog;
void main()
{
    vec4 c = uM[0] * aPos.x + uM[1] * aPos.y + uM[2] * aPos.z + uM[3];
    vec2 s = c.xy / c.w - uOffset;
    vec2 ndc = s / uTargetSize * 2.0 - 1.0;
    float z = clamp(c.z / c.w * uDepthScale, 0.0, 1.0);
    gl_Position = vec4(ndc * c.w, (z * 2.0 - 1.0) * c.w, c.w);
    vUv = aUv;
    vCol = aCol;
    vFog = aFog;
}
)";

        // As the present pass, but the alpha an opaque draw stores is the GS's (0x80 = 1.0, byte 128); blended
        // draws keep the target's alpha (their blend factors).
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
uniform int uDebug;
uniform int uAlphaAsColour;
uniform ivec2 uWrap;      // CLAMP WMS / WMT: 0 repeat, 1 clamp, 2 region clamp, 3 region repeat
uniform ivec4 uRegion;    // MINU, MAXU, MINV, MAXV
uniform vec2 uTexSize;
uniform int uShadowCount; // a shadow volume's faces: +1 seen from the front, -1 from the back (a float count image)
uniform int uFba;
out vec4 fragColor;
float wrap(float t, int mode, int lo, int hi, float size)
{
    if (mode == 2)
        return clamp(t * size, float(lo) + 0.5, float(hi) + 0.5) / size;
    if (mode == 3)
    {
        float f = t * size;
        int i = int(floor(f));
        return (float((i & lo) | hi) + fract(f)) / size;
    }
    return t;
}
void main()
{
    if (uShadowCount != 0)
    {
        fragColor = vec4(gl_FrontFacing ? 1.0 : -1.0, 0.0, 0.0, 0.0);
        return;
    }
    vec4 colour = vCol * (255.0 / 128.0);
    vec2 uv = vec2(wrap(vUv.x, uWrap.x, uRegion.x, uRegion.y, uTexSize.x), wrap(vUv.y, uWrap.y, uRegion.z, uRegion.w, uTexSize.y));
    vec4 texel = uTextured != 0 ? texture(uTex, uv) : vec4(1.0);
    vec4 c = clamp(texel * colour, 0.0, 1.0);
    if (uAlphaTest != 0 && c.a <= uAlphaRef)
        discard;
    if (uFog != 0)
        c.rgb = mix(uFogColour, c.rgb, clamp(vFog, 0.0, 1.0));
    if (uOpaque != 0)
        c.a *= 128.0 / 255.0;
    if (uFba != 0) // the stored alpha byte with its high bit set
    {
        float b = floor(c.a * 255.0 + 0.5);
        c.a = (b < 128.0 ? b + 128.0 : b) / 255.0;
    }
    if (uAlphaAsColour != 0)
        c.rgb = vec3(c.a);
    if (uDebug != 0)
        c = vec4(1.0, 0.0, 1.0, 128.0 / 255.0);
    fragColor = c;
}
)";

        // A shadow volume's darkening: one triangle over the target, black at uAlpha where the count is not 0.
        const char *kShadowCompositeVertex = R"(#version 330
void main()
{
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
        const char *kShadowCompositeFragment = R"(#version 330
uniform sampler2D uCount;
uniform float uAlpha;
out vec4 fragColor;
void main()
{
    if (abs(texelFetch(uCount, ivec2(gl_FragCoord.xy), 0).r) < 0.5)
        discard;
    fragColor = vec4(0.0, 0.0, 0.0, uAlpha);
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
        GLint locM = -1, locOffset = -1, locTargetSize = -1, locDepthScale = -1, locTex = -1, locTextured = -1;
        GLint locAlphaTest = -1, locAlphaRef = -1, locOpaque = -1, locFog = -1, locFogColour = -1, locAlphaAsColour = -1;
        GLint locWrap = -1, locRegion = -1, locTexSize = -1, locFba = -1;
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
            locDepthScale = glGetUniformLocation(program, "uDepthScale");
            locTex = glGetUniformLocation(program, "uTex");
            locTextured = glGetUniformLocation(program, "uTextured");
            locAlphaTest = glGetUniformLocation(program, "uAlphaTest");
            locAlphaRef = glGetUniformLocation(program, "uAlphaRef");
            locOpaque = glGetUniformLocation(program, "uOpaque");
            locAlphaAsColour = glGetUniformLocation(program, "uAlphaAsColour");
            locWrap = glGetUniformLocation(program, "uWrap");
            locRegion = glGetUniformLocation(program, "uRegion");
            locTexSize = glGetUniformLocation(program, "uTexSize");
            locFba = glGetUniformLocation(program, "uFba");
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
            locShadowCount = glGetUniformLocation(program, "uShadowCount");
            const GLuint cvs = compile(GL_VERTEX_SHADER, kShadowCompositeVertex), cfs = compile(GL_FRAGMENT_SHADER, kShadowCompositeFragment);
            if (cvs == 0 || cfs == 0)
                return false;
            shadowProgram = glCreateProgram();
            glAttachShader(shadowProgram, cvs);
            glAttachShader(shadowProgram, cfs);
            glLinkProgram(shadowProgram);
            glDeleteShader(cvs);
            glDeleteShader(cfs);
            glGetProgramiv(shadowProgram, GL_LINK_STATUS, &ok);
            if (!ok)
                return false;
            locShadowTex = glGetUniformLocation(shadowProgram, "uCount");
            locShadowAlpha = glGetUniformLocation(shadowProgram, "uAlpha");
            glGenVertexArrays(1, &shadowVao);
            return true;
        }

        // Shadow volumes: a float count image the size of the target, with the target's depth attached (tested, not
        // written), to count each volume's faces where it is in front of the scene.
        GLint locShadowCount = -1, locShadowTex = -1, locShadowAlpha = -1;
        GLuint shadowProgram = 0, shadowVao = 0, shadowFbo = 0, shadowCount = 0;
        GLsizei shadowWidth = 0, shadowHeight = 0;

        bool shadowTarget(GLsizei width, GLsizei height, GLuint depth)
        {
            if (shadowFbo == 0)
                glGenFramebuffers(1, &shadowFbo);
            if (shadowCount == 0 || shadowWidth != width || shadowHeight != height)
            {
                if (shadowCount == 0)
                    glGenTextures(1, &shadowCount);
                glBindTexture(GL_TEXTURE_2D, shadowCount);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                shadowWidth = width;
                shadowHeight = height;
            }
            glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, shadowCount, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
            return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
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
            m.blend = m.alphaBlend ? blendFromAlpha(target.alpha) : Material::Blend::Opaque;
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
            m.clamp = target.clamp;
            m.fogColour = target.fogcol;
            return m;
        }
    }

    Material::Blend blendFromAlpha(uint64_t alpha)
    {
        // A, B, D: 0 Cs, 1 Cd, 2 zero; C: 0 As, 1 Ad, 2 FIX.
        const uint32_t a = alpha & 3u, b = (alpha >> 2) & 3u, d = (alpha >> 6) & 3u;
        if (a == b)
            return Material::Blend::Opaque; // 0 * C + D: D, Cs as it is for the D = Cs games use
        if (a == 0u && b == 1u && d == 1u)
            return Material::Blend::Alpha;
        if (a == 0u && b == 2u && d == 1u)
            return Material::Blend::Additive;
        if (a == 2u && b == 0u && d == 1u)
            return Material::Blend::Subtract;
        if (a == 2u && b == 1u && d == 1u)
            return Material::Blend::Darken;
        if (a == 1u && b == 2u && d == 1u)
            return Material::Blend::Brighten;
        return Material::Blend::Alpha;
    }

    void Renderer::gsDraw(uint64_t group, const GsDrawTarget &target)
    {
        instance().drawGroups(group, target);
    }

    void Renderer::setStreamDraws(bool on)
    {
        m_streamDraws = on;
        setGsDrawHook(on ? &Renderer::gsDraw : nullptr);
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
            Material m = entry.batch.material.fromGsContext ? materialFromGs(entry.batch.material, target) : entry.batch.material;
            if (m.texaFromGs)
                m.texa = target.texa;
            const GLsizei count = static_cast<GLsizei>(entry.vertices.size());
            glUniform1f(pass.locDepthScale, static_cast<float>(entry.gsZScale * target.zScale));
            glUniform4fv(pass.locM, 4, entry.batch.matrix);
            if (m.blend == Material::Blend::ShadowVolume)
            {
                // Its faces counted where the volume is in front of the scene (the target's depth, tested, not
                // written), then the target darkened once where the count is not 0.
                const GLsizei width = static_cast<GLsizei>(target.width * scale), height = static_cast<GLsizei>(target.height * scale);
                if (count > 0 && pass.shadowTarget(width, height, target.depth))
                {
                    GLfloat clear[4];
                    glGetFloatv(GL_COLOR_CLEAR_VALUE, clear);
                    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                    glClear(GL_COLOR_BUFFER_BIT);
                    glClearColor(clear[0], clear[1], clear[2], clear[3]);
                    glUniform1i(pass.locShadowCount, 1);
                    glEnable(GL_DEPTH_TEST);
                    glDepthFunc(GL_GEQUAL);
                    glDepthMask(GL_FALSE);
                    glEnable(GL_BLEND);
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_ONE, GL_ONE);
                    glDrawArrays(GL_TRIANGLES, first, count);
                    glUniform1i(pass.locShadowCount, 0);
                    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
                    glDisable(GL_DEPTH_TEST);
                    glBlendFuncSeparate(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
                    glUseProgram(pass.shadowProgram);
                    glBindVertexArray(pass.shadowVao);
                    glBindTexture(GL_TEXTURE_2D, pass.shadowCount);
                    glUniform1i(pass.locShadowTex, 0);
                    glUniform1f(pass.locShadowAlpha, std::min(1.0f, static_cast<float>(entry.vertices[0].rgba >> 24) / 128.0f));
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                    glUseProgram(pass.program);
                    glBindVertexArray(pass.vao);
                    glBindBuffer(GL_ARRAY_BUFFER, pass.vbo);
                }
                else
                    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
                first += count;
                continue;
            }
            std::shared_ptr<const TextureCache::Texture> texture = entry.batch.texture;
            if (m.textured && (!texture || m.texaFromGs))
                texture = ps2x::native::textures().texture(m.tex0, m.texa);
            const bool textured = m.textured && texture;
            // PS2X_NATIVE_GS_LOG=<n>: the first n batches drawn: material, texture, target (PS2X_NATIVE_GS_LOG_EVERY=<k>:
            // one batch in k).
            static int s_logLeft = [] { const char *v = std::getenv("PS2X_NATIVE_GS_LOG"); return v ? std::atoi(v) : 0; }();
            static const uint64_t s_logEvery = [] { const char *v = std::getenv("PS2X_NATIVE_GS_LOG_EVERY"); return v ? std::max<uint64_t>(1u, std::strtoull(v, nullptr, 0)) : 1u; }();
            static uint64_t s_logSeen = 0;
            if (s_logLeft > 0 && (s_logSeen++ % s_logEvery) == 0u)
            {
                --s_logLeft;
                std::fprintf(stderr,
                             "[native-gs] batch group %llu: %d vertices, blend %d tex0 %#llx textured %d/%d (%ux%u) texa %#llx alphaTest %d ref %u "
                             "depth %d/%d clamp %#llx fog %d; target fb %u %ux%u scale %u zScale %g offset %u,%u scissor %u,%u-%u,%u; m3 %g %g %g %g\n",
                             static_cast<unsigned long long>(entry.group), count, static_cast<int>(m.blend), static_cast<unsigned long long>(m.tex0),
                             m.textured ? 1 : 0, textured ? 1 : 0, textured ? texture->width : 0u, textured ? texture->height : 0u,
                             static_cast<unsigned long long>(m.texa), m.alphaTest ? 1 : 0, m.alphaRef, m.depthTest ? 1 : 0, m.depthWrite ? 1 : 0,
                             static_cast<unsigned long long>(m.clamp), m.fog ? 1 : 0, target.framebuffer, target.width, target.height, target.scale,
                             target.zScale, target.xOffset, target.yOffset, target.scissorX0, target.scissorY0, target.scissorX1, target.scissorY1,
                             entry.batch.matrix[12], entry.batch.matrix[13], entry.batch.matrix[14], entry.batch.matrix[15]);
            }
            if (textured)
            {
                glBindTexture(GL_TEXTURE_2D, pass.texture(m.tex0 ^ (m.texa * 0x9e3779b97f4a7c15ull), texture));
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, m.clampU ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, m.clampV ? GL_CLAMP_TO_EDGE : GL_REPEAT);
            }
            // The region modes in the shader (a sub-rectangle of the texture, clamped or repeated); the plain modes
            // stay the sampler's.
            const int wms = static_cast<int>(m.clamp & 3u), wmt = static_cast<int>((m.clamp >> 2) & 3u);
            glUniform2i(pass.locWrap, textured && wms >= 2 ? wms : 0, textured && wmt >= 2 ? wmt : 0);
            glUniform4i(pass.locRegion, static_cast<int>((m.clamp >> 4) & 0x3FFu), static_cast<int>((m.clamp >> 14) & 0x3FFu),
                        static_cast<int>((m.clamp >> 24) & 0x3FFu), static_cast<int>((m.clamp >> 34) & 0x3FFu));
            glUniform2f(pass.locTexSize, textured ? static_cast<float>(texture->width) : 1.0f,
                        textured ? static_cast<float>(texture->height) : 1.0f);
            glUniform1i(pass.locTextured, textured ? 1 : 0);
            glUniform1i(pass.locAlphaTest, m.alphaTest ? 1 : 0);
            glUniform1f(pass.locAlphaRef, m.alphaRef / 128.0f);
            glUniform1i(pass.locOpaque, m.blend == Material::Blend::Opaque ? 1 : 0);
            glUniform1i(pass.locFog, m.fog ? 1 : 0);
            const uint32_t fog = m.fogColourFromGs ? target.fogcol : m.fogColour;
            const float fogColour[3] = {static_cast<float>(fog & 0xffu) / 255.0f, static_cast<float>((fog >> 8) & 0xffu) / 255.0f,
                                        static_cast<float>((fog >> 16) & 0xffu) / 255.0f};
            glUniform3fv(pass.locFogColour, 1, fogColour);
            glUniform1i(pass.locAlphaAsColour, m.blend == Material::Blend::Brighten ? 1 : 0);
            glUniform1i(pass.locFba, m.fba ? 1 : 0);
            switch (m.blend)
            {
            case Material::Blend::Opaque:
                glDisable(GL_BLEND);
                break;
            case Material::Blend::Subtract: // Cd - Cs * As
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
                glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
                break;
            case Material::Blend::Darken: // Cd * (1 - As)
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_ADD);
                glBlendFuncSeparate(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
                break;
            case Material::Blend::Brighten: // Cd + Cd * As: the shader outputs As as the colour, times Cd
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_ADD);
                glBlendFuncSeparate(GL_DST_COLOR, GL_ONE, GL_ZERO, GL_ONE);
                break;
            default:
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_ADD);
                glBlendFuncSeparate(GL_SRC_ALPHA, m.blend == Material::Blend::Additive ? GL_ONE : GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
                break;
            }
            if (m.depthTest)
            {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_GEQUAL);
            }
            else
                glDisable(GL_DEPTH_TEST);
            glDepthMask(m.depthWrite ? GL_TRUE : GL_FALSE);
            glDrawArrays(GL_TRIANGLES, first, count);
            first += count;
        }
        glBindSampler(0, static_cast<GLuint>(sampler));
        glBindVertexArray(0);
        glUseProgram(0);
        glDepthMask(GL_TRUE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
    }
}
