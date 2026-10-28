#include "runtime/gs/gs_native_backend.h"

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/native/native_texture_cache.h"

// raylib's OpenGL 3.3 loader: its function pointers are loaded by raylib for the window context and stay valid on a
// context that shares with it.
#include "external/glad.h"
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
    constexpr uint32_t kMaxTargetWidth = 2048u;
    constexpr uint32_t kTargetHeight = 1024u;

    constexpr const char *kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec3 aTex;
layout(location = 3) in float aFog;
uniform vec2 uTargetSize;
out vec4 vColor;
noperspective out vec3 vTex;
out float vFog;
void main()
{
    // GS pixel coordinates: row 0 at the bottom of the GL texture, so a read back starts at GS row 0.
    gl_Position = vec4(aPosition.xy / uTargetSize * 2.0 - 1.0, aPosition.z * 2.0 - 1.0, 1.0);
    vColor = aColor;
    vTex = aTex;
    vFog = aFog;
}
)";

    // GS colour arithmetic with 0x80 as 1.0. The render target keeps alpha as the GS stores it (0x80 for 1.0), so a
    // texture read back from a render target doubles it (uTexAlphaScale 2), one from the texture cache does not.
    constexpr const char *kFragmentShader = R"(#version 330 core
in vec4 vColor;
noperspective in vec3 vTex;
in float vFog;
layout(location = 0, index = 0) out vec4 outColor;
layout(location = 0, index = 1) out vec4 outFactor;
uniform sampler2D uSampler;
uniform vec2 uTexSize;
uniform float uTexAlphaScale;
uniform int uTextured;
uniform int uTfx;
uniform int uTcc;
uniform int uFog;
uniform vec3 uFogColor;
uniform int uAlphaTest;
uniform float uAlphaRef;
void main()
{
    vec4 vc = vColor * (255.0 / 128.0);
    vec4 c = vc;
    if (uTextured != 0)
    {
        vec4 t = texture(uSampler, (vTex.xy / vTex.z) / uTexSize);
        t.a *= uTexAlphaScale;
        if (uTfx == 0)
            c = vec4(t.rgb * vc.rgb, uTcc != 0 ? t.a * vc.a : vc.a);
        else if (uTfx == 1)
            c = vec4(t.rgb, uTcc != 0 ? t.a : vc.a);
        else if (uTfx == 2)
            c = vec4(t.rgb * vc.rgb + vc.a, uTcc != 0 ? t.a + vc.a : vc.a);
        else
            c = vec4(t.rgb * vc.rgb + vc.a, uTcc != 0 ? t.a : vc.a);
    }
    if (uFog != 0)
        c.rgb = mix(uFogColor, c.rgb, vFog);
    float a = floor(clamp(c.a, 0.0, 2.0) * 128.0 + 0.5);
    bool pass = true;
    if (uAlphaTest == 0) pass = false;
    else if (uAlphaTest == 2) pass = a < uAlphaRef;
    else if (uAlphaTest == 3) pass = a <= uAlphaRef;
    else if (uAlphaTest == 4) pass = a == uAlphaRef;
    else if (uAlphaTest == 5) pass = a >= uAlphaRef;
    else if (uAlphaTest == 6) pass = a > uAlphaRef;
    else if (uAlphaTest == 7) pass = a != uAlphaRef;
    if (!pass)
        discard;
    outColor = vec4(clamp(c.rgb, 0.0, 1.0), clamp(c.a, 0.0, 2.0) * (128.0 / 255.0));
    outFactor = vec4(clamp(c.a, 0.0, 1.0));
}
)";

    GLuint compileShader(GLenum type, const char *source)
    {
        const GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint ok = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (ok != GL_TRUE)
        {
            char log[1024] = {};
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            std::fprintf(stderr, "[gs:native] shader compile failed: %s\n", log);
        }
        return shader;
    }

    // Z as a 0..1 depth for the ZBUF format's range: Z32, Z24, Z16/Z16S.
    double depthScale(uint8_t zpsm)
    {
        switch (zpsm & 0x0Fu)
        {
        case 0x0u:
            return 1.0 / 4294967296.0;
        case 0x1u:
            return 1.0 / 16777216.0;
        default:
            return 1.0 / 65536.0;
        }
    }

    uint32_t bitsPerPixel(uint32_t psm)
    {
        switch (psm)
        {
        case 0x00: return 32; // CT32
        case 0x01: return 24; // CT24
        case 0x02:
        case 0x0A: return 16; // CT16, CT16S
        case 0x13:
        case 0x1B: return 8; // T8, T8H
        case 0x14:
        case 0x24:
        case 0x2C: return 4; // T4, T4HL, T4HH
        default: return 32;
        }
    }

    bool colorFormat(uint32_t psm)
    {
        return psm == 0x00u || psm == 0x01u || psm == 0x02u || psm == 0x0Au;
    }

    uint64_t rawTex0(const GSTex0Reg &tex0)
    {
        return static_cast<uint64_t>(tex0.tbp0) | static_cast<uint64_t>(tex0.tbw) << 14 |
               static_cast<uint64_t>(tex0.psm) << 20 | static_cast<uint64_t>(tex0.tw) << 26 |
               static_cast<uint64_t>(tex0.th) << 30 | static_cast<uint64_t>(tex0.tcc) << 34 |
               static_cast<uint64_t>(tex0.tfx) << 35 | static_cast<uint64_t>(tex0.cbp) << 37 |
               static_cast<uint64_t>(tex0.cpsm) << 51 | static_cast<uint64_t>(tex0.csm) << 55 |
               static_cast<uint64_t>(tex0.csa) << 56;
    }

    uint64_t rawTexa(const GSTexaReg &texa)
    {
        return static_cast<uint64_t>(texa.ta0) | static_cast<uint64_t>(texa.aem ? 1u : 0u) << 15 |
               static_cast<uint64_t>(texa.ta1) << 32;
    }

    // Everything in a draw's state that changes how it is drawn, compared to decide when a batch ends.
    bool sameDrawState(const GSDrawState &a, const GSDrawState &b)
    {
        const GSContext &x = a.context;
        const GSContext &y = b.context;
        if (x.frame.fbp != y.frame.fbp || x.frame.psm != y.frame.psm || x.frame.fbmsk != y.frame.fbmsk ||
            x.zbuf.zmask != y.zbuf.zmask || x.test != y.test ||
            std::memcmp(&x.scissor, &y.scissor, sizeof(GSScissorReg)) != 0)
            return false;
        if (a.prim.tme != b.prim.tme || a.prim.fge != b.prim.fge || a.prim.abe != b.prim.abe || a.prim.fst != b.prim.fst)
            return false;
        if (a.prim.abe && x.alpha != y.alpha)
            return false;
        if (a.prim.fge && (a.fogR != b.fogR || a.fogG != b.fogG || a.fogB != b.fogB))
            return false;
        if (a.prim.tme && (rawTex0(x.tex0) != rawTex0(y.tex0) || x.clamp != y.clamp || x.tex1 != y.tex1 ||
                           rawTexa(a.texa) != rawTexa(b.texa)))
            return false;
        return true;
    }

    struct DisplayRead
    {
        uint32_t fbp = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    DisplayRead decodeDisplay(uint64_t dispfb, uint64_t display)
    {
        DisplayRead read;
        read.fbp = static_cast<uint32_t>(dispfb & 0x1FFu);
        read.x = static_cast<uint32_t>((dispfb >> 32) & 0x7FFu);
        read.y = static_cast<uint32_t>((dispfb >> 43) & 0x7FFu);
        const uint32_t magh = static_cast<uint32_t>((display >> 23) & 0xFu) + 1u;
        read.width = (static_cast<uint32_t>((display >> 32) & 0xFFFu) + 1u) / magh;
        read.height = static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u;
        return read;
    }
}

GLFWwindow *GSNativeBackend::createSharedContext()
{
    GLFWwindow *window = glfwGetCurrentContext();
    if (!window)
        return nullptr;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *shared = glfwCreateWindow(16, 16, "gs-native", nullptr, window);
    glfwMakeContextCurrent(window);
    if (!shared)
        std::fprintf(stderr, "[gs:native] could not create the shared OpenGL context\n");
    return shared;
}

GSNativeBackend::GSNativeBackend(GLFWwindow *sharedContext)
    : m_sharedContext(sharedContext), m_memory(std::make_unique<GSCpuBackend>()),
      m_textures(std::make_unique<ps2x::native::TextureCache>())
{
}

GSNativeBackend::~GSNativeBackend() = default;

bool GSNativeBackend::ensureContext(bool create)
{
    if (m_contextReady)
        return std::this_thread::get_id() == m_glThread;
    if (!create || std::this_thread::get_id() != m_submitThread)
        return false;
    if (m_contextFailed || !m_sharedContext)
    {
        m_contextFailed = true;
        return false;
    }
    glfwMakeContextCurrent(m_sharedContext);
    m_glThread = std::this_thread::get_id();

    const GLuint vertex = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    m_program = glCreateProgram();
    glAttachShader(m_program, vertex);
    glAttachShader(m_program, fragment);
    glLinkProgram(m_program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint linked = GL_FALSE;
    glGetProgramiv(m_program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE)
    {
        char log[1024] = {};
        glGetProgramInfoLog(m_program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[gs:native] shader link failed: %s\n", log);
        m_contextFailed = true;
        return false;
    }
    m_uniforms.targetSize = glGetUniformLocation(m_program, "uTargetSize");
    m_uniforms.texSize = glGetUniformLocation(m_program, "uTexSize");
    m_uniforms.textured = glGetUniformLocation(m_program, "uTextured");
    m_uniforms.tfx = glGetUniformLocation(m_program, "uTfx");
    m_uniforms.tcc = glGetUniformLocation(m_program, "uTcc");
    m_uniforms.fog = glGetUniformLocation(m_program, "uFog");
    m_uniforms.fogColor = glGetUniformLocation(m_program, "uFogColor");
    m_uniforms.alphaTest = glGetUniformLocation(m_program, "uAlphaTest");
    m_uniforms.alphaRef = glGetUniformLocation(m_program, "uAlphaRef");
    m_uniforms.sampler = glGetUniformLocation(m_program, "uSampler");
    m_uniforms.fst = glGetUniformLocation(m_program, "uTexAlphaScale");

    glGenVertexArrays(1, &m_vertexArray);
    glGenBuffers(1, &m_vertexBuffer);
    glBindVertexArray(m_vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, r)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, s)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, fog)));
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);

    std::fprintf(stderr, "[gs:native] OpenGL renderer: %s\n", reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    m_contextReady = true;
    return true;
}

GSNativeBackend::RenderTarget &GSNativeBackend::renderTarget(const GSFrameReg &frame)
{
    auto it = m_targets.find(frame.fbp);
    if (it != m_targets.end())
        return it->second;

    RenderTarget target;
    target.fbp = frame.fbp;
    target.width = std::clamp<uint32_t>(frame.fbw * 64u, 64u, kMaxTargetWidth);
    target.height = kTargetHeight;
    glGenTextures(1, &target.color);
    glBindTexture(GL_TEXTURE_2D, target.color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target.width, target.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &target.depth);
    glBindTexture(GL_TEXTURE_2D, target.depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, target.width, target.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 nullptr);
    glGenFramebuffers(1, &target.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target.color, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, target.depth, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(0.0);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    return m_targets.emplace(frame.fbp, target).first->second;
}

GSNativeBackend::RenderTarget *GSNativeBackend::renderTargetAtBlock(uint32_t block)
{
    // FRAME.FBP counts 2048-word pages, TEX0.TBP0 256-byte blocks: 32 blocks a page.
    if (block % 32u != 0u)
        return nullptr;
    auto it = m_targets.find(block / 32u);
    return it == m_targets.end() ? nullptr : &it->second;
}

void GSNativeBackend::appendPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    if (m_batch.empty() || !sameDrawState(state, m_batchState))
    {
        flushBatch();
        m_batchState = state;
    }

    const GSContext &context = state.context;
    const float ofx = static_cast<float>(context.xyoffset.ofx) / 16.0f;
    const float ofy = static_cast<float>(context.xyoffset.ofy) / 16.0f;
    const double zScale = depthScale(context.zbuf.psm);
    const float texWidth = static_cast<float>(1u << std::min<uint32_t>(context.tex0.tw, 10u));
    const float texHeight = static_cast<float>(1u << std::min<uint32_t>(context.tex0.th, 10u));
    const GSVertex &last = batch.vertices[batch.vertexCount - 1u];
    auto vertex = [&](const GSVertex &v, const GSVertex &color, float x, float y, float s, float t, float q) {
        Vertex out;
        out.x = x - ofx;
        out.y = y - ofy;
        out.z = static_cast<float>(std::clamp(v.z * zScale, 0.0, 1.0));
        out.r = color.r;
        out.g = color.g;
        out.b = color.b;
        out.a = color.a;
        out.s = s;
        out.t = t;
        out.q = q;
        out.fog = static_cast<float>(v.fog) / 255.0f;
        m_batch.push_back(out);
    };
    // Texel coordinates of a vertex: UV are texels (12.4 fixed point), S and T scale to the texture's size.
    auto texel = [&](const GSVertex &v, float &s, float &t, float &q) {
        if (state.prim.fst)
        {
            s = static_cast<float>(v.u) / 16.0f;
            t = static_cast<float>(v.v) / 16.0f;
            q = 1.0f;
        }
        else
        {
            s = v.s * texWidth;
            t = v.t * texHeight;
            q = v.q != 0.0f ? v.q : 1.0f;
        }
    };
    auto point = [&](const GSVertex &v) {
        float s, t, q;
        texel(v, s, t, q);
        vertex(v, state.prim.iip ? v : last, v.x, v.y, s, t, q);
    };

    switch (state.prim.type)
    {
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        if (batch.vertexCount >= 3u)
            for (uint32_t i = 0; i < 3u; ++i)
                point(batch.vertices[i]);
        break;
    case GS_PRIM_SPRITE:
        if (batch.vertexCount >= 2u)
        {
            // Two corners: the texel coordinates stretch across the rectangle; colour, Z and fog from the second.
            const GSVertex &a = batch.vertices[0];
            const GSVertex &b = batch.vertices[1];
            float s0, t0, q0, s1, t1, q1;
            texel(a, s0, t0, q0);
            texel(b, s1, t1, q1);
            if (!state.prim.fst)
            {
                s0 /= q0, t0 /= q0, s1 /= q1, t1 /= q1;
            }
            vertex(b, b, a.x, a.y, s0, t0, 1.0f);
            vertex(b, b, b.x, a.y, s1, t0, 1.0f);
            vertex(b, b, a.x, b.y, s0, t1, 1.0f);
            vertex(b, b, b.x, a.y, s1, t0, 1.0f);
            vertex(b, b, b.x, b.y, s1, t1, 1.0f);
            vertex(b, b, a.x, b.y, s0, t1, 1.0f);
        }
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        float s, t, q;
        texel(v, s, t, q);
        vertex(v, v, v.x, v.y, s, t, q);
        vertex(v, v, v.x + 1.0f, v.y, s, t, q);
        vertex(v, v, v.x, v.y + 1.0f, s, t, q);
        vertex(v, v, v.x + 1.0f, v.y, s, t, q);
        vertex(v, v, v.x + 1.0f, v.y + 1.0f, s, t, q);
        vertex(v, v, v.x, v.y + 1.0f, s, t, q);
        break;
    }
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        if (batch.vertexCount >= 2u)
        {
            // A one-pixel-wide quad along the major axis.
            const GSVertex &a = batch.vertices[0];
            const GSVertex &b = batch.vertices[1];
            const GSVertex &ca = state.prim.iip ? a : b;
            float sa, ta, qa, sb, tb, qb;
            texel(a, sa, ta, qa);
            texel(b, sb, tb, qb);
            const bool horizontal = std::abs(b.x - a.x) >= std::abs(b.y - a.y);
            const float dx = horizontal ? 0.0f : 0.5f;
            const float dy = horizontal ? 0.5f : 0.0f;
            vertex(a, ca, a.x - dx, a.y - dy, sa, ta, qa);
            vertex(b, b, b.x - dx, b.y - dy, sb, tb, qb);
            vertex(a, ca, a.x + dx, a.y + dy, sa, ta, qa);
            vertex(b, b, b.x - dx, b.y - dy, sb, tb, qb);
            vertex(b, b, b.x + dx, b.y + dy, sb, tb, qb);
            vertex(a, ca, a.x + dx, a.y + dy, sa, ta, qa);
        }
        break;
    default:
        break;
    }
}

bool GSNativeBackend::bindTexture(const GSDrawState &state, const RenderTarget &target, float &width, float &height)
{
    const GSContext &context = state.context;
    GLuint texture = 0;
    bool fromTarget = false;
    if (colorFormat(context.tex0.psm))
    {
        if (RenderTarget *source = renderTargetAtBlock(context.tex0.tbp0))
        {
            // A render target sampled as a texture (copies, blurs, depth of field). When the draw also writes it,
            // sample a copy of it as it was before the draw.
            fromTarget = true;
            width = static_cast<float>(source->width);
            height = static_cast<float>(source->height);
            texture = source->color;
            if (source->fbp == target.fbp)
            {
                if (!m_feedbackTexture || m_feedbackWidth != source->width || m_feedbackHeight != source->height)
                {
                    if (!m_feedbackTexture)
                        glGenTextures(1, &m_feedbackTexture);
                    glBindTexture(GL_TEXTURE_2D, m_feedbackTexture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, source->width, source->height, 0, GL_RGBA,
                                 GL_UNSIGNED_BYTE, nullptr);
                    m_feedbackWidth = source->width;
                    m_feedbackHeight = source->height;
                }
                glBindTexture(GL_TEXTURE_2D, m_feedbackTexture);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, source->framebuffer);
                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, source->width, source->height);
                glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
                texture = m_feedbackTexture;
            }
        }
    }
    if (!fromTarget)
    {
        const uint64_t tex0 = rawTex0(context.tex0);
        const uint64_t texa = rawTexa(state.texa);
        const ps2x::native::TextureCache::Texture &decoded = m_textures->texture(tex0, texa);
        if (decoded.width == 0u || decoded.height == 0u || decoded.rgba.empty())
            return false;
        uint64_t version = 1469598103934665603ull;
        for (const auto &[page, generation] : decoded.pages)
            version = (version ^ (static_cast<uint64_t>(page) << 32 | generation)) * 1099511628211ull;
        UploadedTexture &uploaded = m_uploaded[decoded.key ^ (texa << 1)];
        if (!uploaded.texture)
            glGenTextures(1, &uploaded.texture);
        glBindTexture(GL_TEXTURE_2D, uploaded.texture);
        if (uploaded.version != version || uploaded.width != decoded.width || uploaded.height != decoded.height)
        {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, decoded.width, decoded.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         decoded.rgba.data());
            uploaded.version = version;
            uploaded.width = decoded.width;
            uploaded.height = decoded.height;
        }
        texture = uploaded.texture;
        width = static_cast<float>(decoded.width);
        height = static_cast<float>(decoded.height);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    // CLAMP: repeat (0, region repeat 3) or clamp (1, region clamp 2); TEX1.MMAG picks bilinear.
    const GLint wrapS = (context.clamp & 0x3u) == 1u || (context.clamp & 0x3u) == 2u ? GL_CLAMP_TO_EDGE : GL_REPEAT;
    const GLint wrapT = (context.clamp >> 2 & 0x3u) == 1u || (context.clamp >> 2 & 0x3u) == 2u ? GL_CLAMP_TO_EDGE
                                                                                              : GL_REPEAT;
    const GLint filter = (context.tex1 >> 5 & 0x1u) != 0u ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glUniform1f(m_uniforms.fst, fromTarget ? 2.0f : 1.0f);
    return true;
}

void GSNativeBackend::applyBlend(const GSDrawState &state)
{
    if (!state.prim.abe)
    {
        glDisable(GL_BLEND);
        return;
    }
    // GS blending: (A - B) * C >> 7 + D, A/B/D from Cs, Cd, 0 and C from As, Ad, FIX. Mapped onto fixed-function
    // blending, with As as the shader's second output (dual-source) and FIX as the constant alpha.
    const uint64_t alpha = state.context.alpha;
    const uint32_t a = alpha & 0x3u, b = alpha >> 2 & 0x3u, c = alpha >> 4 & 0x3u, d = alpha >> 6 & 0x3u;
    const GLenum factor = c == 0u ? GL_SRC1_ALPHA : c == 1u ? GL_DST_ALPHA : GL_CONSTANT_ALPHA;
    const GLenum inverse = c == 0u ? GL_ONE_MINUS_SRC1_ALPHA : c == 1u ? GL_ONE_MINUS_DST_ALPHA
                                                                       : GL_ONE_MINUS_CONSTANT_ALPHA;
    if (c == 2u)
        glBlendColor(0.0f, 0.0f, 0.0f, std::min(1.0f, static_cast<float>(alpha >> 32 & 0xFFu) / 128.0f));
    enum { Cs = 0, Cd = 1, Zero = 2 };
    GLenum equation = GL_FUNC_ADD, source = GL_ONE, destination = GL_ZERO;
    bool mapped = true;
    if (a == b)
    {
        source = d == Cs ? GL_ONE : GL_ZERO;
        destination = d == Cd ? GL_ONE : GL_ZERO;
    }
    else if (a == Cs && b == Cd && d == Cd)
        source = factor, destination = inverse;
    else if (a == Cs && b == Zero && d == Cd)
        source = factor, destination = GL_ONE;
    else if (a == Cs && b == Zero && d == Zero)
        source = factor, destination = GL_ZERO;
    else if (a == Zero && b == Cs && d == Cd)
        equation = GL_FUNC_REVERSE_SUBTRACT, source = factor, destination = GL_ONE;
    else if (a == Cd && b == Cs && d == Cs)
        source = inverse, destination = factor;
    else if (a == Cd && b == Zero && d == Cs)
        source = GL_ONE, destination = factor;
    else if (a == Zero && b == Cd && d == Cs)
        equation = GL_FUNC_SUBTRACT, source = GL_ONE, destination = factor;
    else if (a == Cd && b == Zero && d == Zero)
        source = GL_ZERO, destination = factor;
    else
        mapped = false;
    if (!mapped)
    {
        glDisable(GL_BLEND);
        return;
    }
    glEnable(GL_BLEND);
    glBlendEquationSeparate(equation, GL_FUNC_ADD);
    glBlendFuncSeparate(source, destination, GL_ONE, GL_ZERO);
}

void GSNativeBackend::flushBatch()
{
    if (m_batch.empty())
        return;
    if (!ensureContext(true))
    {
        if (m_contextFailed)
            m_batch.clear();
        return;
    }
    const GSDrawState &state = m_batchState;
    const GSContext &context = state.context;
    RenderTarget &target = renderTarget(context.frame);
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glViewport(0, 0, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height));
    glEnable(GL_SCISSOR_TEST);
    glScissor(context.scissor.x0, context.scissor.y0, context.scissor.x1 - context.scissor.x0 + 1,
              context.scissor.y1 - context.scissor.y0 + 1);

    const bool depthTest = (context.test >> 16 & 0x1u) != 0u;
    if (depthTest)
    {
        static constexpr GLenum kDepthFunc[4] = {GL_NEVER, GL_ALWAYS, GL_GEQUAL, GL_GREATER};
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(kDepthFunc[context.test >> 17 & 0x3u]);
        glDepthMask(context.zbuf.zmask ? GL_FALSE : GL_TRUE);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }
    // FBMSK bits set keep the frame buffer's bits; whole bytes map to channel masks. CT24 has no alpha to write.
    const uint32_t fbmsk = context.frame.fbmsk;
    const bool keepAlpha = (fbmsk >> 24) == 0xFFu || context.frame.psm == 0x01u;
    glColorMask((fbmsk & 0xFFu) != 0xFFu, (fbmsk >> 8 & 0xFFu) != 0xFFu, (fbmsk >> 16 & 0xFFu) != 0xFFu, !keepAlpha);
    glDisable(GL_CULL_FACE);
    applyBlend(state);

    glUseProgram(m_program);
    glUniform2f(m_uniforms.targetSize, static_cast<float>(target.width), static_cast<float>(target.height));
    float texWidth = 1.0f, texHeight = 1.0f;
    const bool textured = state.prim.tme && bindTexture(state, target, texWidth, texHeight);
    glUniform1i(m_uniforms.textured, textured ? 1 : 0);
    glUniform2f(m_uniforms.texSize, texWidth, texHeight);
    glUniform1i(m_uniforms.sampler, 0);
    glUniform1i(m_uniforms.tfx, context.tex0.tfx);
    glUniform1i(m_uniforms.tcc, context.tex0.tcc);
    glUniform1i(m_uniforms.fog, state.prim.fge ? 1 : 0);
    glUniform3f(m_uniforms.fogColor, state.fogR / 255.0f, state.fogG / 255.0f, state.fogB / 255.0f);
    // TEST: ATE, ATST, AREF. A failed test keeps the pixel unwritten (AFAIL other than KEEP is not translated yet).
    const bool alphaTest = (context.test & 0x1u) != 0u;
    glUniform1i(m_uniforms.alphaTest, alphaTest ? static_cast<int>(context.test >> 1 & 0x7u) : 1);
    glUniform1f(m_uniforms.alphaRef, static_cast<float>(context.test >> 4 & 0xFFu));

    glBindVertexArray(m_vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(m_batch.size() * sizeof(Vertex)), m_batch.data(),
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_batch.size()));
    glBindVertexArray(0);
    m_batch.clear();
}

void GSNativeBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    m_memory->Initialize(vram, vramSize);
}

void GSNativeBackend::Reset()
{
    m_batch.clear();
    m_memory->Reset();
}

void GSNativeBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (m_submitThread == std::thread::id{})
        m_submitThread = std::this_thread::get_id();
    appendPrimitive(batch);
}

void GSNativeBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    m_memory->LoadClut(tex0, texclut);
}

void GSNativeBackend::BeginTransfer(const GSTransferCommand &command)
{
    flushBatch();
    m_memory->BeginTransfer(command);
    m_transfer = command;
    m_transferData.clear();
    m_transferBytes = 0;
    if (command.direction == 0u)
    {
        // Host to local: collect the image, the texture cache takes it whole.
        const uint64_t pixels = static_cast<uint64_t>(command.trxreg.rrw) * command.trxreg.rrh;
        m_transferBytes = static_cast<size_t>((pixels * bitsPerPixel(command.bitbltbuf.dpsm) + 7u) / 8u);
        m_transferData.reserve(m_transferBytes);
    }
}

void GSNativeBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    m_memory->UploadImage(data, sizeBytes);
    if (m_transferBytes == 0u || m_transferData.size() >= m_transferBytes)
        return;
    const size_t take = std::min<size_t>(sizeBytes, m_transferBytes - m_transferData.size());
    m_transferData.insert(m_transferData.end(), data, data + take);
    if (m_transferData.size() < m_transferBytes)
        return;
    const GSBitBltBuf &b = m_transfer.bitbltbuf;
    const uint64_t bitbltbuf = static_cast<uint64_t>(b.sbp) | static_cast<uint64_t>(b.sbw) << 16 |
                               static_cast<uint64_t>(b.spsm) << 24 | static_cast<uint64_t>(b.dbp) << 32 |
                               static_cast<uint64_t>(b.dbw) << 48 | static_cast<uint64_t>(b.dpsm) << 56;
    const GSTrxPos &p = m_transfer.trxpos;
    const uint64_t trxpos = static_cast<uint64_t>(p.ssax) | static_cast<uint64_t>(p.ssay) << 16 |
                            static_cast<uint64_t>(p.dsax) << 32 | static_cast<uint64_t>(p.dsay) << 48 |
                            static_cast<uint64_t>(p.dir) << 59;
    const uint64_t trxreg = static_cast<uint64_t>(m_transfer.trxreg.rrw) | static_cast<uint64_t>(m_transfer.trxreg.rrh) << 32;
    m_textures->upload(bitbltbuf, trxpos, trxreg, m_transferData.data(), m_transferData.size());
    m_transferBytes = 0;
}

void GSNativeBackend::Flush()
{
    flushBatch();
    m_memory->Flush();
}

void GSNativeBackend::TextureFlush()
{
    m_memory->TextureFlush();
}

void GSNativeBackend::Sync(GSSyncReason reason)
{
    flushBatch();
    m_memory->Sync(reason);
}

PresentationFrame GSNativeBackend::Present(const GSPresentationRequest &request)
{
    flushBatch();
    PresentationFrame frame;
    if (!ensureContext())
        return frame;
    const bool crt1 = (request.pmode & 0x1u) != 0u;
    const DisplayRead read = crt1 || (request.pmode & 0x2u) == 0u ? decodeDisplay(request.dispfb1, request.display1)
                                                                    : decodeDisplay(request.dispfb2, request.display2);
    auto it = m_targets.find(read.fbp);
    if (it == m_targets.end())
        return frame;
    const RenderTarget &target = it->second;
    const uint32_t width = std::min(read.width, target.width - std::min(read.x, target.width));
    const uint32_t height = std::min(read.height, target.height - std::min(read.y, target.height));
    if (width == 0u || height == 0u)
        return frame;
    frame.width = width;
    frame.height = height;
    frame.displayFbp = read.fbp;
    frame.sourceFbp = read.fbp;
    frame.pixels.resize(static_cast<size_t>(width) * height * 4u);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, target.framebuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(static_cast<GLint>(read.x), static_cast<GLint>(read.y), static_cast<GLsizei>(width),
                 static_cast<GLsizei>(height), GL_RGBA, GL_UNSIGNED_BYTE, frame.pixels.data());
    for (size_t i = 3; i < frame.pixels.size(); i += 4)
        frame.pixels[i] = 0xFFu;
    return frame;
}

bool GSNativeBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    flushBatch();
    if (!ensureContext())
        return false;
    RenderTarget &target = renderTarget(context.frame);
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearColor(static_cast<float>(rgba & 0xFFu) / 255.0f, static_cast<float>(rgba >> 8 & 0xFFu) / 255.0f,
                 static_cast<float>(rgba >> 16 & 0xFFu) / 255.0f, static_cast<float>(rgba >> 24 & 0xFFu) / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    return true;
}

uint32_t GSNativeBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    return m_memory->ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSNativeBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    return m_memory->ReadVram(psm, base, bw, x, y);
}

void GSNativeBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    m_memory->WriteVram(psm, base, bw, x, y, value);
}

void GSNativeBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    m_memory->SnapshotVram(out);
}

GSTransferSnapshot GSNativeBackend::GetTransferSnapshot() const
{
    return m_memory->GetTransferSnapshot();
}
