#include "runtime/gs/gs_native_backend.h"

#include "runtime/gs/gs_cpu_backend.h"

// raylib's OpenGL 3.3 loader: its function pointers are loaded by raylib for the window context and stay valid on a
// context that shares with it.
#include "external/glad.h"
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace
{
    constexpr uint32_t kMaxTargetWidth = 2048u;
    constexpr uint32_t kTargetHeight = 1024u;

    constexpr const char *kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;
uniform vec2 uTargetSize;
out vec4 vColor;
void main()
{
    // GS pixel coordinates: row 0 at the bottom of the GL texture, so a read back starts at GS row 0.
    gl_Position = vec4(aPosition.xy / uTargetSize * 2.0 - 1.0, aPosition.z * 2.0 - 1.0, 1.0);
    vColor = aColor;
}
)";

    constexpr const char *kFragmentShader = R"(#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main()
{
    fragColor = vColor;
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
    : m_sharedContext(sharedContext), m_memory(std::make_unique<GSCpuBackend>())
{
}

GSNativeBackend::~GSNativeBackend() = default;

bool GSNativeBackend::ensureContext()
{
    if (m_contextReady)
        return std::this_thread::get_id() == m_glThread;
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
        std::fprintf(stderr, "[gs:native] shader link failed\n");
        m_contextFailed = true;
        return false;
    }
    m_targetSizeLocation = glGetUniformLocation(m_program, "uTargetSize");

    glGenVertexArrays(1, &m_vertexArray);
    glGenBuffers(1, &m_vertexBuffer);
    glBindVertexArray(m_vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, r)));
    glEnableVertexAttribArray(1);
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
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(0.0);
    glDisable(GL_SCISSOR_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    return m_targets.emplace(frame.fbp, target).first->second;
}

void GSNativeBackend::appendPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSContext &context = state.context;
    // What changes GL state: the render target, the depth test and the depth write mask.
    const uint64_t stateKey = (context.test >> 16 & 0x7u) | (context.zbuf.zmask ? 0x8u : 0u);
    if (context.frame.fbp != m_batchTarget || stateKey != m_batchStateKey ||
        std::memcmp(&context.scissor, &m_batchState.context.scissor, sizeof(GSScissorReg)) != 0)
    {
        flushBatch();
        m_batchTarget = context.frame.fbp;
        m_batchStateKey = stateKey;
        m_batchState = state;
    }

    const float ofx = static_cast<float>(context.xyoffset.ofx) / 16.0f;
    const float ofy = static_cast<float>(context.xyoffset.ofy) / 16.0f;
    const double zScale = depthScale(context.zbuf.psm);
    const GSVertex &last = batch.vertices[batch.vertexCount - 1u];
    auto vertex = [&](const GSVertex &v, float x, float y) {
        const GSVertex &color = state.prim.iip ? v : last;
        Vertex out;
        out.x = x - ofx;
        out.y = y - ofy;
        out.z = static_cast<float>(std::clamp(v.z * zScale, 0.0, 1.0));
        out.r = color.r;
        out.g = color.g;
        out.b = color.b;
        out.a = color.a;
        m_batch.push_back(out);
    };
    auto quad = [&](const GSVertex &corner, float x0, float y0, float x1, float y1) {
        GSVertex flat = corner;
        vertex(flat, x0, y0);
        vertex(flat, x1, y0);
        vertex(flat, x0, y1);
        vertex(flat, x1, y0);
        vertex(flat, x1, y1);
        vertex(flat, x0, y1);
    };

    switch (state.prim.type)
    {
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        if (batch.vertexCount >= 3u)
            for (uint32_t i = 0; i < 3u; ++i)
                vertex(batch.vertices[i], batch.vertices[i].x, batch.vertices[i].y);
        break;
    case GS_PRIM_SPRITE:
        if (batch.vertexCount >= 2u)
        {
            const GSVertex &a = batch.vertices[0];
            const GSVertex &b = batch.vertices[1];
            quad(b, std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y));
        }
        break;
    case GS_PRIM_POINT:
        quad(batch.vertices[0], batch.vertices[0].x, batch.vertices[0].y, batch.vertices[0].x + 1.0f,
             batch.vertices[0].y + 1.0f);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        if (batch.vertexCount >= 2u)
        {
            // A one-pixel-wide quad along the major axis.
            const GSVertex &a = batch.vertices[0];
            const GSVertex &b = batch.vertices[1];
            const bool horizontal = std::abs(b.x - a.x) >= std::abs(b.y - a.y);
            const float dx = horizontal ? 0.0f : 0.5f;
            const float dy = horizontal ? 0.5f : 0.0f;
            vertex(a, a.x - dx, a.y - dy);
            vertex(b, b.x - dx, b.y - dy);
            vertex(a, a.x + dx, a.y + dy);
            vertex(b, b.x - dx, b.y - dy);
            vertex(b, b.x + dx, b.y + dy);
            vertex(a, a.x + dx, a.y + dy);
        }
        break;
    default:
        break;
    }
}

void GSNativeBackend::flushBatch()
{
    if (m_batch.empty())
        return;
    if (!ensureContext())
    {
        if (m_contextFailed)
            m_batch.clear();
        return;
    }
    const GSContext &context = m_batchState.context;
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
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    glUseProgram(m_program);
    glUniform2f(m_targetSizeLocation, static_cast<float>(target.width), static_cast<float>(target.height));
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
}

void GSNativeBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    m_memory->UploadImage(data, sizeBytes);
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
