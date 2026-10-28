// The mesh renderer drawing inside the GS stream (runtime/native/native_gs_hook.h): where VU1 skips a program the
// native port draws, the GS thread calls gsDraw with the render target that program's output would have gone to,
// and the batches of that program's group are drawn into it there, so they sort with the GS draws around them.
// Runs on the GS thread with the GS renderer's GL context current: raw GL (rlgl's state belongs to the window's
// thread), its own vertex array (not shared between contexts) and its own copies of the textures.

#include "runtime/native/native_renderer.h"
#include "runtime/native/native_gs_hook.h"

#include "external/glad.h"

#include <algorithm>
#include <cstdio>
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
out vec4 fragColor;
void main()
{
    vec4 colour = vCol * (255.0 / 128.0);
    vec4 texel = uTextured != 0 ? texture(uTex, vUv) : vec4(1.0);
    vec4 c = clamp(texel * colour, 0.0, 1.0);
    if (uAlphaTest != 0 && c.a <= uAlphaRef)
        discard;
    if (uFog != 0)
        c.rgb = mix(uFogColour, c.rgb, clamp(vFog, 0.0, 1.0));
    if (uOpaque != 0)
        c.a *= 128.0 / 255.0;
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
        GLint locM = -1, locOffset = -1, locTargetSize = -1, locDepthScale = -1, locTex = -1, locTextured = -1;
        GLint locAlphaTest = -1, locAlphaRef = -1, locOpaque = -1, locFog = -1, locFogColour = -1;
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

    void Renderer::gsDraw(uint64_t group, const GsDrawTarget &target)
    {
        instance().drawGroups(group, target);
    }

    void Renderer::setStreamDraws(bool on)
    {
        m_streamDraws = on;
        setGsDrawHook(on ? &Renderer::gsDraw : nullptr);
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
        std::vector<StreamBatch> batches;
        {
            std::lock_guard<std::mutex> lock(m_streamMutex);
            while (!m_stream.empty() && m_stream.front().group <= group)
            {
                batches.push_back(std::move(m_stream.front()));
                m_stream.pop_front();
            }
        }
        if (batches.empty() || target.framebuffer == 0u)
            return; // nothing for this group, or a GS renderer without GL targets (CPU, paraLLEl): present() draws
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
        glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
        glViewport(0, 0, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height));
        glEnable(GL_SCISSOR_TEST);
        const uint32_t x0 = std::min(target.scissorX0, target.width), y0 = std::min(target.scissorY0, target.height);
        const uint32_t x1 = std::min(target.scissorX1 + 1u, target.width), y1 = std::min(target.scissorY1 + 1u, target.height);
        glScissor(static_cast<GLint>(x0), static_cast<GLint>(y0), static_cast<GLsizei>(x1 > x0 ? x1 - x0 : 0u),
                  static_cast<GLsizei>(y1 > y0 ? y1 - y0 : 0u));
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
        glActiveTexture(GL_TEXTURE0);
        // Our textures' own filtering, not a sampler object the GS renderer left on unit 0.
        GLint sampler = 0;
        glGetIntegerv(GL_SAMPLER_BINDING, &sampler);
        glBindSampler(0, 0);

        GLint first = 0;
        for (const StreamBatch &entry : batches)
        {
            const Material &m = entry.batch.material;
            const GLsizei count = static_cast<GLsizei>(entry.vertices.size());
            glUniform1f(pass.locDepthScale, static_cast<float>(entry.gsZScale * target.zScale));
            glUniform4fv(pass.locM, 4, entry.batch.matrix);
            std::shared_ptr<const TextureCache::Texture> texture = entry.batch.texture;
            if (m.textured && !texture)
                texture = ps2x::native::textures().texture(m.tex0, m.texa);
            const bool textured = m.textured && texture;
            if (textured)
                glBindTexture(GL_TEXTURE_2D, pass.texture(m.tex0 ^ (m.texa * 0x9e3779b97f4a7c15ull), texture));
            glUniform1i(pass.locTextured, textured ? 1 : 0);
            glUniform1i(pass.locAlphaTest, m.alphaTest ? 1 : 0);
            glUniform1f(pass.locAlphaRef, m.alphaRef / 128.0f);
            glUniform1i(pass.locOpaque, m.blend == Material::Blend::Opaque ? 1 : 0);
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
                glBlendFuncSeparate(GL_SRC_ALPHA, m.blend == Material::Blend::Additive ? GL_ONE : GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
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
