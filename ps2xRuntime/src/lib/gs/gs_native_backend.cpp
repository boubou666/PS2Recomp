#include "runtime/gs/gs_native_backend.h"

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "runtime/gs/gs_texture_page_cache.h"
#include "runtime/native/native_texture_cache.h"

// raylib's OpenGL 3.3 loader: its function pointers are loaded by raylib for the window context and stay valid on a
// context that shares with it.
#include "external/glad.h"
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
noperspective out float vDepth;
out float vFog;
void main()
{
    // GS pixel coordinates: row 0 at the bottom of the GL texture, so a read back starts at GS row 0.
    gl_Position = vec4(aPosition.xy / uTargetSize * 2.0 - 1.0, aPosition.z * 2.0 - 1.0, 1.0);
    vColor = aColor;
    vTex = aTex;
    vDepth = aPosition.z;
    vFog = aFog;
}
)";

    // GS colour arithmetic with 0x80 as 1.0. The render target keeps alpha as the GS stores it (0x80 for 1.0), so a
    // texture read back from a render target doubles it (uTexAlphaScale 2), one from the texture cache does not.
    constexpr const char *kFragmentShader = R"(#version 330 core
in vec4 vColor;
noperspective in vec3 vTex;
noperspective in float vDepth;
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
uniform int uBlendOut;
uniform sampler2D uDest; // a copy of the render target, for the destination alpha test
uniform int uDate;       // 0 off, 1 pass where the stored alpha's top bit is 0, 2 where it is 1
uniform int uBlendFactor;
uniform float uBlendFix;

// Textures read straight from a GPU copy of GS memory (uVram: 1024 x 1024 32-bit words), through per-format page
// swizzle tables (uSwizzle: a page's pixel offsets, one table per layout at its row): no CPU decode.
uniform usampler2D uVram;
uniform usampler2D uSwizzle;
uniform int uDecode;
uniform ivec4 uTexFmt;  // swizzle row, page width, page height, bits a pixel takes in memory
uniform ivec4 uTexSel;  // bit offset in the word (T8H, T4HL, T4HH), bits kept, TBP0, TBW
uniform ivec4 uTexKind; // 0 CT32 / 1 CT24 / 2 CT16 / 3 indexed, CT16 palette, CSA, palette entries
uniform ivec4 uClutFmt; // the palette's layout, as uTexFmt
uniform int uCbp;
uniform vec3 uTexa;     // TA0, TA1, AEM
uniform ivec4 uWrapU;   // CLAMP mode, 2^TW - 1, MINU, MAXU
uniform ivec4 uWrapV;
uniform int uLinear;
uniform int uTargetWrap; // a render target sampled directly: apply uWrapU / uWrapV to its texel coordinates

uint gsRead(ivec4 fmt, int bitOffset, int maskBits, int bp, int bw, ivec2 p)
{
    int pw = fmt.y, ph = fmt.z;
    int perRow = max(1, (bw * 64) / pw);
    int page = bp / 32 + (p.y / ph) * perRow + p.x / pw;
    int ppp = pw * ph;
    uint offset = texelFetch(uSwizzle, ivec2(p.x % pw, fmt.x + p.y % ph), 0).r;
    uint pixel = uint(page * ppp + (bp % 32) * (ppp / 32)) + offset;
    uint bits = pixel * uint(fmt.w) + uint(bitOffset);
    uint word = (bits >> 5u) & 0xFFFFFu;
    uint value = texelFetch(uVram, ivec2(int(word & 1023u), int(word >> 10u)), 0).r >> (bits & 31u);
    return maskBits >= 32 ? value : value & ((1u << uint(maskBits)) - 1u);
}

vec4 gsColor(uint v, int kind)
{
    if (kind == 2)
    {
        vec3 rgb = vec3(float(v & 31u), float((v >> 5u) & 31u), float((v >> 10u) & 31u)) * (8.0 / 255.0);
        float a = (v & 0x8000u) != 0u ? uTexa.y : ((uTexa.z != 0.0 && (v & 0x7FFFu) == 0u) ? 0.0 : uTexa.x);
        return vec4(rgb, a / 128.0);
    }
    vec3 rgb = vec3(float(v & 255u), float((v >> 8u) & 255u), float((v >> 16u) & 255u)) / 255.0;
    if (kind == 1)
        return vec4(rgb, (uTexa.z != 0.0 && (v & 0xFFFFFFu) == 0u) ? 0.0 : uTexa.x / 128.0);
    return vec4(rgb, float(v >> 24u) / 128.0);
}

int gsWrap(int c, ivec4 w)
{
    if (w.x == 0) return c & w.y;
    if (w.x == 1) return clamp(c, 0, w.y);
    if (w.x == 2) return clamp(c, w.z, w.w);
    return (c & w.z) | w.w;
}

vec4 gsTexel(ivec2 p)
{
    p = ivec2(gsWrap(p.x, uWrapU), gsWrap(p.y, uWrapV));
    uint v = gsRead(uTexFmt, uTexSel.x, uTexSel.y, uTexSel.z, uTexSel.w, p);
    if (uTexKind.x != 3)
        return gsColor(v, uTexKind.x);
    // CSM1 palette: entry i at (x, y) of a 16-wide image, a 16-entry palette CSA * 16 entries in.
    uint index = uTexKind.w == 16 ? (uint(uTexKind.z) * 16u + v) & 255u : v;
    ivec2 q = ivec2(int((index & 7u) | ((index & 16u) >> 1u)), int(((index >> 5u) << 1u) | ((index >> 3u) & 1u)));
    uint entry = gsRead(uClutFmt, 0, uClutFmt.w, uCbp, 1, q);
    return gsColor(entry, uTexKind.y != 0 ? 2 : 0);
}

vec4 gsSample(vec2 t)
{
    if (uLinear == 0)
        return gsTexel(ivec2(floor(t)));
    vec2 s = t - 0.5;
    ivec2 p = ivec2(floor(s));
    vec2 f = s - floor(s);
    return mix(mix(gsTexel(p), gsTexel(p + ivec2(1, 0)), f.x), mix(gsTexel(p + ivec2(0, 1)), gsTexel(p + ivec2(1, 1)), f.x),
               f.y);
}

void main()
{
    // The depth itself, not gl_Position.z: z * 2 - 1 next to -1 has float steps of ~1e-7, so the small Z a game
    // tests with GREATER (Jade Cocoon 2: a movie sprite at Z 1 over a clear at Z 0, Z32) fell on the clear's depth.
    gl_FragDepth = vDepth;
    vec4 vc = vColor * (255.0 / 128.0);
    vec4 c = vc;
    if (uTextured != 0)
    {
        vec2 tc = vTex.xy / vTex.z;
        if (uDecode == 0 && uTargetWrap != 0)
        {
            // A render target sampled directly: the GS's CLAMP modes over 2^TW x 2^TH texels (or the region), not
            // the GL texture's whole size, so filtering never reaches rows the game did not draw.
            if (uWrapU.x == 0) tc.x = mod(tc.x, float(uWrapU.y + 1));
            else if (uWrapU.x == 1) tc.x = clamp(tc.x, 0.5, float(uWrapU.y) + 0.5);
            else if (uWrapU.x == 2) tc.x = clamp(tc.x, float(uWrapU.z) + 0.5, float(uWrapU.w) + 0.5);
            if (uWrapV.x == 0) tc.y = mod(tc.y, float(uWrapV.y + 1));
            else if (uWrapV.x == 1) tc.y = clamp(tc.y, 0.5, float(uWrapV.y) + 0.5);
            else if (uWrapV.x == 2) tc.y = clamp(tc.y, float(uWrapV.z) + 0.5, float(uWrapV.w) + 0.5);
        }
        vec4 t = uDecode != 0 ? gsSample(tc) : texture(uSampler, tc / uTexSize);
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
    if (uDate != 0)
    {
        bool set = texelFetch(uDest, ivec2(gl_FragCoord.xy), 0).a * 255.0 >= 127.5;
        if (set != (uDate == 2))
            discard;
    }
    // The GS blend's C: the source alpha or FIX (0x80 is 1.0). Blends that fixed-function GL cannot express take
    // part of the work here: the factor as colour (Cd * (1 + C) through DST_COLOR, ONE), or Cs * (1 + C).
    float factor = uBlendFactor == 2 ? uBlendFix : clamp(c.a, 0.0, 2.0);
    vec3 rgb = c.rgb;
    if (uBlendOut == 1)
        rgb = vec3(factor);
    else if (uBlendOut == 2)
        rgb = c.rgb * (1.0 + factor);
    outColor = vec4(clamp(rgb, 0.0, 1.0), clamp(c.a, 0.0, 2.0) * (128.0 / 255.0));
    outFactor = vec4(clamp(factor, 0.0, 1.0));
}
)";

    // Writing a render target into the GPU copy of GS memory: one fragment per 32-bit word of the target's pages,
    // finding the pixel(s) stored there through the inverse swizzle table, so no read back to the CPU.
    constexpr const char *kResolveVertexShader = R"(#version 330 core
void main()
{
    gl_Position = vec4((gl_VertexID & 1) != 0 ? 1.0 : -1.0, (gl_VertexID & 2) != 0 ? 1.0 : -1.0, 0.0, 1.0);
}
)";

    constexpr const char *kResolveFragmentShader = R"(#version 330 core
layout(location = 0) out uint outWord;
uniform sampler2D uTarget;   // the render target's colour, GL row 0 = GS row 0, alpha as the GS stores it
uniform usampler2D uInverse; // page pixel offset -> x | y << 16
uniform int uFbp;
uniform int uFbw;
uniform int uKind;           // 0 CT32, 1 CT16, 2 CT16S
uniform int uUsedHeight;     // rows drawn: words of pixels below them keep what GS memory holds

uvec4 bytes(ivec2 p)
{
    return uvec4(round(texelFetch(uTarget, p, 0) * 255.0));
}

uint pack16(uvec4 c)
{
    return (c.r >> 3u) | ((c.g >> 3u) << 5u) | ((c.b >> 3u) << 10u) | (c.a >= 128u ? 0x8000u : 0u);
}

void main()
{
    ivec2 f = ivec2(gl_FragCoord.xy);
    int word = f.y * 1024 + f.x;
    int rel = word / 2048 - uFbp;
    int inPage = word % 2048;
    int prow = rel / uFbw, pcol = rel % uFbw;
    if (uKind == 0)
    {
        uint e = texelFetch(uInverse, ivec2(inPage % 128, inPage / 128), 0).r;
        ivec2 p = ivec2(pcol * 64 + int(e & 0xFFFFu), prow * 32 + int(e >> 16u));
        if (p.y >= uUsedHeight)
            discard;
        uvec4 c = bytes(p);
        outWord = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
        return;
    }
    int row = uKind == 1 ? 16 : 48;
    int i0 = inPage * 2, i1 = inPage * 2 + 1;
    uint e0 = texelFetch(uInverse, ivec2(i0 % 128, row + i0 / 128), 0).r;
    uint e1 = texelFetch(uInverse, ivec2(i1 % 128, row + i1 / 128), 0).r;
    ivec2 p0 = ivec2(pcol * 64 + int(e0 & 0xFFFFu), prow * 64 + int(e0 >> 16u));
    ivec2 p1 = ivec2(pcol * 64 + int(e1 & 0xFFFFu), prow * 64 + int(e1 >> 16u));
    if (p0.y >= uUsedHeight && p1.y >= uUsedHeight)
        discard;
    outWord = pack16(bytes(p0)) | (pack16(bytes(p1)) << 16u);
}
)";

    // GL state a helper changes while flushBatch has the draw set up (a resolve or the GPU copy's set-up from inside
    // bindTexture): saved on construction, put back on destruction.
    struct GlStateRestore
    {
        GLint drawFramebuffer = 0, readFramebuffer = 0, program = 0, vertexArray = 0, activeTexture = 0;
        GLint viewport[4] = {}, scissorBox[4] = {};
        GLboolean scissor = GL_FALSE, depth = GL_FALSE, blend = GL_FALSE, colorMask[4] = {};
        GlStateRestore()
        {
            glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFramebuffer);
            glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vertexArray);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
            glGetIntegerv(GL_VIEWPORT, viewport);
            glGetIntegerv(GL_SCISSOR_BOX, scissorBox);
            scissor = glIsEnabled(GL_SCISSOR_TEST);
            depth = glIsEnabled(GL_DEPTH_TEST);
            blend = glIsEnabled(GL_BLEND);
            glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
        }
        ~GlStateRestore()
        {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(drawFramebuffer));
            glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(readFramebuffer));
            glUseProgram(static_cast<GLuint>(program));
            glBindVertexArray(static_cast<GLuint>(vertexArray));
            glActiveTexture(static_cast<GLenum>(activeTexture));
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            glScissor(scissorBox[0], scissorBox[1], scissorBox[2], scissorBox[3]);
            (scissor ? glEnable : glDisable)(GL_SCISSOR_TEST);
            (depth ? glEnable : glDisable)(GL_DEPTH_TEST);
            (blend ? glEnable : glDisable)(GL_BLEND);
            glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        }
    };

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

    // The decoding shader's view of a pixel format: its page swizzle table (row in the table texture, page size,
    // bits a pixel takes in memory) and what it keeps of the word.
    struct GpuLayout
    {
        int row, pageWidth, pageHeight, bits;
    };
    constexpr GpuLayout kLayoutC32{0, 64, 32, 32}, kLayoutC16{32, 64, 64, 16}, kLayoutC16S{96, 64, 64, 16},
        kLayoutP8{160, 128, 64, 8}, kLayoutP4{224, 128, 128, 4}, kLayoutZ32{352, 64, 32, 32}, kLayoutZ16{384, 64, 64, 16},
        kLayoutZ16S{448, 64, 64, 16};
    constexpr uint32_t kSwizzleWidth = 128u, kSwizzleHeight = 512u;
    // The other way, for writing render targets into GS memory: page pixel offset -> x | y << 16, for the colour
    // layouts a render target can have (rows: CT32 0-15, CT16 16-47, CT16S 48-79).
    constexpr uint32_t kInverseHeight = 80u;
    constexpr int kInverseRowC32 = 0, kInverseRowC16 = 16, kInverseRowC16S = 48;

    // Each layout's page: the pixel offset (in its own pixels) where page pixel (x, y) is stored, found by writing
    // every pixel's index through GSMem and reading back where it landed (a few bits at a time for 8- and 4-bit).
    std::vector<uint32_t> buildSwizzleTables(std::vector<uint32_t> &inverse)
    {
        GSMem::InitLookupTables();
        std::vector<uint32_t> table(kSwizzleWidth * kSwizzleHeight, 0u);
        inverse.assign(kSwizzleWidth * kInverseHeight, 0u);
        std::vector<uint8_t> scratch(4u * 1024u * 1024u, 0u);
        using Write = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
        const std::pair<Write, GpuLayout> layouts[] = {
            {GSMem::WriteCT32, kLayoutC32}, {GSMem::WriteCT16, kLayoutC16}, {GSMem::WriteCT16S, kLayoutC16S},
            {GSMem::WriteP8, kLayoutP8},    {GSMem::WriteP4, kLayoutP4},    {GSMem::WriteZ32, kLayoutZ32},
            {GSMem::WriteZ16, kLayoutZ16},  {GSMem::WriteZ16S, kLayoutZ16S}};
        for (const auto &[write, layout] : layouts)
        {
            const uint32_t pw = layout.pageWidth, ph = layout.pageHeight, bits = layout.bits;
            const uint32_t pixels = pw * ph, bw = pw / 64u;
            const uint32_t mask = bits >= 32u ? ~0u : (1u << bits) - 1u;
            std::vector<uint32_t> owner(pixels, 0u); // pixel address in the page -> index y * pw + x stored there
            for (uint32_t shift = 0; shift < 16u; shift += bits)
            {
                std::fill(scratch.begin(), scratch.begin() + 8192, uint8_t{0});
                for (uint32_t y = 0; y < ph; ++y)
                    for (uint32_t x = 0; x < pw; ++x)
                        write(scratch.data(), 0u, bw, x, y, ((y * pw + x) >> shift) & mask);
                for (uint32_t address = 0; address < pixels; ++address)
                {
                    const uint32_t bit = address * bits;
                    uint32_t word = 0;
                    std::memcpy(&word, scratch.data() + (bit >> 5) * 4u, 4u);
                    owner[address] |= ((word >> (bit & 31u)) & mask) << shift;
                }
                if (bits >= 16u)
                    break;
            }
            const int inverseRow = layout.row == kLayoutC32.row    ? kInverseRowC32
                                   : layout.row == kLayoutC16.row  ? kInverseRowC16
                                   : layout.row == kLayoutC16S.row ? kInverseRowC16S
                                                                   : -1;
            for (uint32_t address = 0; address < pixels; ++address)
            {
                const uint32_t index = owner[address];
                table[(layout.row + index / pw) * kSwizzleWidth + index % pw] = address;
                if (inverseRow >= 0)
                    inverse[static_cast<size_t>(inverseRow) * kSwizzleWidth + address] = (index % pw) | (index / pw) << 16;
            }
        }
        return table;
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
    : m_sharedContext(sharedContext), m_memory(std::make_unique<GSCpuBackend>())
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
    m_uniforms.blendOut = glGetUniformLocation(m_program, "uBlendOut");
    m_uniforms.blendFactor = glGetUniformLocation(m_program, "uBlendFactor");
    m_uniforms.blendFix = glGetUniformLocation(m_program, "uBlendFix");
    m_uniforms.fst = glGetUniformLocation(m_program, "uTexAlphaScale");
    m_uniforms.dest = glGetUniformLocation(m_program, "uDest");
    m_uniforms.date = glGetUniformLocation(m_program, "uDate");
    m_uniforms.vram = glGetUniformLocation(m_program, "uVram");
    m_uniforms.swizzle = glGetUniformLocation(m_program, "uSwizzle");
    m_uniforms.decode = glGetUniformLocation(m_program, "uDecode");
    m_uniforms.texFmt = glGetUniformLocation(m_program, "uTexFmt");
    m_uniforms.texSel = glGetUniformLocation(m_program, "uTexSel");
    m_uniforms.texKind = glGetUniformLocation(m_program, "uTexKind");
    m_uniforms.clutFmt = glGetUniformLocation(m_program, "uClutFmt");
    m_uniforms.cbp = glGetUniformLocation(m_program, "uCbp");
    m_uniforms.texa = glGetUniformLocation(m_program, "uTexa");
    m_uniforms.wrapU = glGetUniformLocation(m_program, "uWrapU");
    m_uniforms.wrapV = glGetUniformLocation(m_program, "uWrapV");
    m_uniforms.linear = glGetUniformLocation(m_program, "uLinear");
    m_uniforms.targetWrap = glGetUniformLocation(m_program, "uTargetWrap");

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
    {
        RenderTarget &existing = it->second;
        const bool sameLayout = existing.psm == frame.psm || ((existing.psm | frame.psm) & ~0x1u) == 0u;
        if (sameLayout && existing.fbw == frame.fbw)
            return existing;
        // The same memory drawn in another format or width (a CT32 buffer reused as CT16): its pixels go to GS
        // memory first, where the new target's readers will find them, and the target starts over.
        if (static const bool log = std::getenv("PS2X_GS_NATIVE_LOG") != nullptr; log)
        {
            static uint32_t s_logged = 0;
            if (s_logged++ < 100u)
                std::fprintf(stderr, "[gs:native] target fbp=%u redrawn as psm=0x%x fbw=%u (was psm=0x%x fbw=%u)\n",
                             frame.fbp, frame.psm, frame.fbw, existing.psm, existing.fbw);
        }
        if (existing.dirty && existing.usedHeight != 0u)
            resolveTarget(existing);
        glDeleteFramebuffers(1, &existing.framebuffer);
        glDeleteTextures(1, &existing.color);
        glDeleteTextures(1, &existing.depth);
        m_targets.erase(it);
    }

    RenderTarget target;
    target.fbp = frame.fbp;
    target.fbw = frame.fbw;
    target.psm = frame.psm;
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
        m_batchMaxY = std::max(m_batchMaxY, out.y);
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

void GSNativeBackend::ensureVram()
{
    if (m_vramTexture)
        return;
    const GlStateRestore restore;
    glGenTextures(1, &m_vramTexture);
    glBindTexture(GL_TEXTURE_2D, m_vramTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 1024, 1024, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    std::vector<uint32_t> inverse;
    const std::vector<uint32_t> swizzle = buildSwizzleTables(inverse);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGenTextures(1, &m_swizzleTexture);
    glBindTexture(GL_TEXTURE_2D, m_swizzleTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, kSwizzleWidth, kSwizzleHeight, 0, GL_RED_INTEGER, GL_UNSIGNED_INT,
                 swizzle.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &m_inverseTexture);
    glBindTexture(GL_TEXTURE_2D, m_inverseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, kSwizzleWidth, kInverseHeight, 0, GL_RED_INTEGER, GL_UNSIGNED_INT,
                 inverse.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &m_vramFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_vramFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_vramTexture, 0);
    const GLuint vertex = compileShader(GL_VERTEX_SHADER, kResolveVertexShader);
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, kResolveFragmentShader);
    m_resolveProgram = glCreateProgram();
    glAttachShader(m_resolveProgram, vertex);
    glAttachShader(m_resolveProgram, fragment);
    glLinkProgram(m_resolveProgram);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint linked = GL_FALSE;
    glGetProgramiv(m_resolveProgram, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE)
    {
        std::fprintf(stderr, "[gs:native] resolve program link failed: resolves read back to the CPU\n");
        glDeleteProgram(m_resolveProgram);
        m_resolveProgram = 0;
    }
}

void GSNativeBackend::syncVramToGpu()
{
    // The blocks transfers changed since the last draw: 64 words each, 16 to a row of the texture, sent as runs.
    if (!m_anyDirty || !m_vram)
        return;
    m_anyDirty = false;
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_vramTexture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    const uint32_t blocks = static_cast<uint32_t>(m_dirtyBlocks.size());
    for (uint32_t block = 0; block < blocks;)
    {
        if (!m_dirtyBlocks[block])
        {
            ++block;
            continue;
        }
        uint32_t end = block;
        while (end < blocks && m_dirtyBlocks[end] && (end == block || end % 16u != 0u))
            m_dirtyBlocks[end++] = 0u;
        glTexSubImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>((block % 16u) * 64u), static_cast<GLint>(block / 16u),
                        static_cast<GLsizei>((end - block) * 64u), 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        m_vram + static_cast<size_t>(block) * 256u);
        block = end;
    }
    glActiveTexture(GL_TEXTURE0);
}

void GSNativeBackend::pullGpuPages()
{
    if (!m_anyGpuNewer || !m_vram || !m_vramFramebuffer || !ensureContext())
        return;
    const GlStateRestore restore;
    m_anyGpuNewer = false;
    syncVramToGpu();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_vramFramebuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    for (uint32_t page = 0; page < m_gpuNewer.size(); ++page)
        if (m_gpuNewer[page])
        {
            m_gpuNewer[page] = 0u;
            glReadPixels(0, static_cast<GLint>(page * 2u), 1024, 2, GL_RED_INTEGER, GL_UNSIGNED_INT,
                         m_vram + static_cast<size_t>(page) * 8192u);
        }
}

bool GSNativeBackend::resolveTargetGpu(RenderTarget &target)
{
    const int kind = target.psm == 0x00u ? 0 : target.psm == 0x02u ? 1 : target.psm == 0x0Au ? 2 : -1;
    // A CT32 target whose alpha was never drawn (FBMSK kept it): GS memory's top bytes must stay, the CPU path does.
    if (kind < 0 || !m_vram || (kind == 0 && !target.alphaWritten))
        return false;
    ensureVram();
    if (!m_resolveProgram)
        return false;
    const GlStateRestore restore;
    // Transfers before this resolve land first, then the target's pixels over them, all on the GPU copy.
    syncVramToGpu();
    const uint32_t pageRows = kind == 0 ? 32u : 64u;
    const uint32_t first = target.fbp;
    const uint32_t end = std::min<uint32_t>(
        512u, target.fbp + std::max<uint32_t>(target.fbw, 1u) * ((target.usedHeight + pageRows - 1u) / pageRows));
    if (end <= first)
        return true;
    glBindFramebuffer(GL_FRAMEBUFFER, m_vramFramebuffer);
    glViewport(0, 0, 1024, 1024);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, static_cast<GLint>(first * 2u), 1024, static_cast<GLsizei>((end - first) * 2u));
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glUseProgram(m_resolveProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, target.color);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, m_inverseTexture);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(m_resolveProgram, "uTarget"), 0);
    glUniform1i(glGetUniformLocation(m_resolveProgram, "uInverse"), 4);
    glUniform1i(glGetUniformLocation(m_resolveProgram, "uFbp"), static_cast<GLint>(target.fbp));
    glUniform1i(glGetUniformLocation(m_resolveProgram, "uFbw"), static_cast<GLint>(std::max<uint32_t>(target.fbw, 1u)));
    glUniform1i(glGetUniformLocation(m_resolveProgram, "uKind"), kind);
    glUniform1i(glGetUniformLocation(m_resolveProgram, "uUsedHeight"), static_cast<GLint>(target.usedHeight));
    glBindVertexArray(m_vertexArray);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    for (uint32_t page = first; page < end; ++page)
        m_gpuNewer[page] = 1u;
    m_anyGpuNewer = true;
    return true;
}

void GSNativeBackend::resolveTarget(RenderTarget &target)
{
    const uint32_t width = target.fbw * 64u;
    const uint32_t height = target.usedHeight;
    target.dirty = false;
    // PS2X_GS_NATIVE_NORESOLVE=1: skip resolves (wrong pictures), to time what they cost.
    static const bool noResolve = std::getenv("PS2X_GS_NATIVE_NORESOLVE") != nullptr;
    if (noResolve || width == 0u || height == 0u || width > target.width)
        return;
    static const bool cpuDecode = std::getenv("PS2X_GS_NATIVE_CPUDECODE") != nullptr;
    static const bool cpuResolve = std::getenv("PS2X_GS_NATIVE_CPURESOLVE") != nullptr; // diagnostics
    if (!cpuDecode && !cpuResolve && resolveTargetGpu(target))
        return;
    pullGpuPages();
    // The pixels as a host-to-local CT32 transfer at the frame's base: rows top first (GL row 0 is GS row 0), alpha
    // as the GS stores it.
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4u);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, target.framebuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height), GL_RGBA, GL_UNSIGNED_BYTE,
                 pixels.data());
    uint32_t dpsm = 0x00u;
    if (target.psm == 0x02u || target.psm == 0x0Au)
    {
        // A 16-bit target: RGB5A1 (alpha bit from the GS alpha's top bit, 0x80), written in its own format.
        dpsm = target.psm;
        for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i)
        {
            const uint8_t *p = &pixels[i * 4u];
            const uint16_t v = static_cast<uint16_t>((p[0] >> 3) | (p[1] >> 3) << 5 | (p[2] >> 3) << 10 |
                                                     ((p[3] & 0x80u) ? 0x8000u : 0u));
            std::memcpy(&pixels[i * 2u], &v, 2u);
        }
        pixels.resize(static_cast<size_t>(width) * height * 2u);
    }
    // Into GS memory (where draws in another format and the GPU copy of it read), and into the texture cache for the
    // CPU decoding path.
    if (m_vram)
    {
        const uint32_t bp = target.fbp * 32u, bw = std::max<uint32_t>(target.fbw, 1u);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
            {
                const size_t i = static_cast<size_t>(y) * width + x;
                if (dpsm == 0x00u)
                {
                    uint32_t v = 0;
                    std::memcpy(&v, &pixels[i * 4u], 4u);
                    // CT24 leaves the top byte alone (where a game keeps T8H / T4HL / T4HH textures, the GS too).
                    (target.psm == 0x01u || !target.alphaWritten ? GSMem::WriteCT24 : GSMem::WriteCT32)(m_vram, bp, bw, x,
                                                                                                         y, v);
                }
                else
                {
                    uint16_t v = 0;
                    std::memcpy(&v, &pixels[i * 2u], 2u);
                    (dpsm == 0x0Au ? GSMem::WriteCT16S : GSMem::WriteCT16)(m_vram, bp, bw, x, y, v);
                }
            }
        markDirty(dpsm, bp, bw, 0u, 0u, width, height);
    }
    const uint64_t bitbltbuf = static_cast<uint64_t>(target.fbp * 32u) << 32 | static_cast<uint64_t>(target.fbw) << 48 |
                               static_cast<uint64_t>(dpsm) << 56;
    const uint64_t trxreg = static_cast<uint64_t>(width) | static_cast<uint64_t>(height) << 32;
    ps2x::native::textures().upload(bitbltbuf, 0u, trxreg, pixels.data(), pixels.size());
}

GSNativeBackend::FrameTarget GSNativeBackend::frameTarget(const GSFrameReg &frame, const GSZbufReg &zbuf, uint32_t rows)
{
    FrameTarget out;
    // A game whose first draws are all native submits no GS primitive before this: the GS thread is the submit thread.
    if (m_submitThread == std::thread::id{})
        m_submitThread = std::this_thread::get_id();
    flushBatch();
    if (!ensureContext(true))
        return out;
    RenderTarget &target = renderTarget(frame);
    target.dirty = true;
    // Down to the scissor's last row, as a GS draw counts the rows it reaches: the whole target would also cover the
    // GS memory below the picture, where a game uploads its textures, and each upload there dropped the target with
    // the native draws in it (TMB's 3D frame came out black).
    target.usedHeight = std::min(target.height, std::max(target.usedHeight, rows));
    target.alphaWritten = true;
    target.zbp = zbuf.zbp;
    target.zpsm = zbuf.psm;
    out.framebuffer = target.framebuffer;
    out.color = target.color;
    out.depth = target.depth;
    out.width = target.width;
    out.height = target.height;
    out.zScale = depthScale(zbuf.psm);
    return out;
}

void GSNativeBackend::NativeDraw(uint64_t group, const GSContext &context)
{
    const ps2x::native::GsDrawHook hook = ps2x::native::gsDrawHook();
    if (!hook)
        return;
    const FrameTarget frame = frameTarget(context.frame, context.zbuf, context.scissor.y1 + 1u);
    ps2x::native::GsDrawTarget target;
    fillNativeRegisters(target, context);
    target.framebuffer = frame.framebuffer;
    target.color = frame.color;
    target.depth = frame.depth;
    target.width = frame.width;
    target.height = frame.height;
    target.zScale = frame.zScale;
    target.xOffset = context.xyoffset.ofx;
    target.yOffset = context.xyoffset.ofy;
    target.scissorX0 = context.scissor.x0;
    target.scissorY0 = context.scissor.y0;
    target.scissorX1 = context.scissor.x1;
    target.scissorY1 = context.scissor.y1;
    hook(group, target);
}

void GSNativeBackend::resolveAllTargets()
{
    flushBatch();
    if (!ensureContext())
        return;
    // PS2X_GS_NATIVE_VRAMCHECK=1: the GPU copy against GS memory, block by block, for pages not resolved on the GPU.
    if (static const bool check = std::getenv("PS2X_GS_NATIVE_VRAMCHECK") != nullptr; check && m_vram && m_vramFramebuffer)
    {
        syncVramToGpu();
        std::vector<uint32_t> gpu(1024u * 1024u);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, m_vramFramebuffer);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadPixels(0, 0, 1024, 1024, GL_RED_INTEGER, GL_UNSIGNED_INT, gpu.data());
        uint32_t differing = 0;
        for (uint32_t block = 0; block < 16384u; ++block)
        {
            if (m_gpuNewer[block / 32u])
                continue;
            if (std::memcmp(gpu.data() + block * 64u, m_vram + static_cast<size_t>(block) * 256u, 256u) != 0 &&
                differing++ < 40u)
            {
                uint32_t cpu = 0;
                std::memcpy(&cpu, m_vram + static_cast<size_t>(block) * 256u, 4u);
                std::fprintf(stderr, "[gs:native] vram check: block %u (page %u) differs: gpu %08x cpu %08x dirty %u\n",
                             block, block / 32u, gpu[block * 64u], cpu, m_dirtyBlocks[block]);
            }
        }
        std::fprintf(stderr, "[gs:native] vram check: %u blocks differ\n", differing);
    }
    for (auto &[fbp, target] : m_targets)
        if (target.dirty && target.usedHeight != 0u)
            resolveTarget(target);
    pullGpuPages();
}

void GSNativeBackend::resolveTargetsUnder(const GSTex0Reg &tex0)
{
    // The texture's blocks: TBW * 64 texels by its height, at its format's size (a conservative extent).
    static constexpr uint32_t kBits[64] = {32, 24, 16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0, 0, 0,
                                           0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4};
    const uint32_t bits = kBits[tex0.psm & 0x3Fu] ? kBits[tex0.psm & 0x3Fu] : 32u;
    const uint64_t texRows = 1ull << std::min<uint32_t>(tex0.th, 10u);
    const uint64_t texBlocks = (std::max<uint32_t>(tex0.tbw, 1u) * 64ull * texRows * bits / 8u + 255u) / 256u;
    for (auto &[fbp, target] : m_targets)
    {
        if (!target.dirty || target.usedHeight == 0u)
            continue;
        const uint64_t base = static_cast<uint64_t>(fbp) * 32u;
        // Pages of 32 blocks: 64 x 32 pixels in CT32/CT24, 64 x 64 in CT16.
        const uint32_t pageRows = (target.psm == 0x02u || target.psm == 0x0Au) ? 64u : 32u;
        const uint64_t blocks = static_cast<uint64_t>(target.fbw) * ((target.usedHeight + pageRows - 1u) / pageRows) * 32u;
        if (tex0.tbp0 < base + blocks && base < tex0.tbp0 + texBlocks)
            resolveTarget(target);
    }
}

void GSNativeBackend::dropTargetsUnder(const GSTransferCommand &command)
{
    // The transfer's pages: rows TRXPOS.DSAY to + TRXREG.RRH at the destination format's page size.
    const uint32_t psm = command.bitbltbuf.dpsm;
    const bool fourBit = psm == 0x14u || psm == 0x24u || psm == 0x2Cu;
    const bool eightBit = psm == 0x13u || psm == 0x1Bu;
    const bool sixteen = psm == 0x02u || psm == 0x0Au;
    const uint32_t pageWidth = (fourBit || eightBit) && psm != 0x1Bu && psm != 0x24u && psm != 0x2Cu ? 128u : 64u;
    const uint32_t pageHeight = psm == 0x14u ? 128u : (psm == 0x13u || sixteen) ? 64u : 32u;
    const uint32_t width = std::max<uint32_t>(command.bitbltbuf.dbw, 1u) * 64u;
    const uint32_t pagesPerRow = std::max<uint32_t>(width / pageWidth, 1u);
    const uint32_t y0 = command.trxpos.dsay, y1 = y0 + std::max<uint32_t>(command.trxreg.rrh, 1u);
    const uint64_t first = command.bitbltbuf.dbp / 32u + static_cast<uint64_t>(y0 / pageHeight) * pagesPerRow;
    const uint64_t last = command.bitbltbuf.dbp / 32u + static_cast<uint64_t>((y1 + pageHeight - 1u) / pageHeight) * pagesPerRow;
    for (auto it = m_targets.begin(); it != m_targets.end();)
    {
        const RenderTarget &target = it->second;
        const uint64_t begin = target.fbp;
        const uint64_t end = begin + static_cast<uint64_t>(target.fbw) * std::max<uint32_t>((target.usedHeight + 31u) / 32u, 1u);
        if (target.usedHeight != 0u && first < end && begin < last)
        {
            if (ensureContext())
            {
                glDeleteFramebuffers(1, &target.framebuffer);
                glDeleteTextures(1, &target.color);
                glDeleteTextures(1, &target.depth);
            }
            it = m_targets.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool GSNativeBackend::bindDecodedTexture(const GSDrawState &state)
{
    const GSContext &context = state.context;
    const GSTex0Reg &tex0 = context.tex0;
    GpuLayout layout{};
    int bitOffset = 0, bitsKept = 0, kind = 0, entries = 0;
    switch (tex0.psm)
    {
    case 0x00u: layout = kLayoutC32, bitsKept = 32, kind = 0; break;
    case 0x01u: layout = kLayoutC32, bitsKept = 24, kind = 1; break;
    case 0x02u: layout = kLayoutC16, bitsKept = 16, kind = 2; break;
    case 0x0Au: layout = kLayoutC16S, bitsKept = 16, kind = 2; break;
    case 0x13u: layout = kLayoutP8, bitsKept = 8, kind = 3, entries = 256; break;
    case 0x14u: layout = kLayoutP4, bitsKept = 4, kind = 3, entries = 16; break;
    case 0x1Bu: layout = kLayoutC32, bitOffset = 24, bitsKept = 8, kind = 3, entries = 256; break;
    case 0x24u: layout = kLayoutC32, bitOffset = 24, bitsKept = 4, kind = 3, entries = 16; break;
    case 0x2Cu: layout = kLayoutC32, bitOffset = 28, bitsKept = 4, kind = 3, entries = 16; break;
    case 0x30u: layout = kLayoutZ32, bitsKept = 32, kind = 0; break;
    case 0x31u: layout = kLayoutZ32, bitsKept = 24, kind = 1; break;
    case 0x32u: layout = kLayoutZ16, bitsKept = 16, kind = 2; break;
    case 0x3Au: layout = kLayoutZ16S, bitsKept = 16, kind = 2; break;
    default: return false;
    }
    ensureVram();
    syncVramToGpu();
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_swizzleTexture);
    glActiveTexture(GL_TEXTURE0);

    const GpuLayout clut = tex0.cpsm == 0x02u ? kLayoutC16 : tex0.cpsm == 0x0Au ? kLayoutC16S : kLayoutC32;
    const uint64_t clamp = context.clamp;
    const int maxU = (1 << std::min<uint32_t>(tex0.tw, 10u)) - 1, maxV = (1 << std::min<uint32_t>(tex0.th, 10u)) - 1;
    glUniform1i(m_uniforms.vram, 2);
    glUniform1i(m_uniforms.swizzle, 3);
    glUniform1i(m_uniforms.decode, 1);
    glUniform4i(m_uniforms.texFmt, layout.row, layout.pageWidth, layout.pageHeight, layout.bits);
    glUniform4i(m_uniforms.texSel, bitOffset, bitsKept, static_cast<int>(tex0.tbp0), static_cast<int>(tex0.tbw));
    glUniform4i(m_uniforms.texKind, kind, clut.bits == 16 ? 1 : 0, static_cast<int>(tex0.csa), entries);
    glUniform4i(m_uniforms.clutFmt, clut.row, clut.pageWidth, clut.pageHeight, clut.bits);
    glUniform1i(m_uniforms.cbp, static_cast<int>(tex0.cbp));
    glUniform3f(m_uniforms.texa, static_cast<float>(state.texa.ta0), static_cast<float>(state.texa.ta1),
                state.texa.aem ? 1.0f : 0.0f);
    glUniform4i(m_uniforms.wrapU, static_cast<int>(clamp & 0x3u), maxU, static_cast<int>(clamp >> 4 & 0x3FFu),
                static_cast<int>(clamp >> 14 & 0x3FFu));
    glUniform4i(m_uniforms.wrapV, static_cast<int>(clamp >> 2 & 0x3u), maxV, static_cast<int>(clamp >> 24 & 0x3FFu),
                static_cast<int>(clamp >> 34 & 0x3FFu));
    glUniform1i(m_uniforms.linear, (context.tex1 >> 5 & 0x1u) != 0u ? 1 : 0);
    glUniform1f(m_uniforms.fst, 1.0f);
    return true;
}

bool GSNativeBackend::bindDepthAsTexture(const GSDrawState &state, float &width, float &height)
{
    const GSTex0Reg &tex0 = state.context.tex0;
    RenderTarget *source = nullptr;
    for (auto &[fbp, target] : m_targets)
        if (target.zbp != ~0u && target.zbp * 32u == tex0.tbp0 && target.usedHeight != 0u)
            source = &target;
    if (!source)
        return false;
    const uint32_t w = source->width, h = source->usedHeight;
    std::vector<float> depth(static_cast<size_t>(w) * h);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, source->framebuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h), GL_DEPTH_COMPONENT, GL_FLOAT, depth.data());
    // Back to GS Z in the Z buffer's range, then laid out as the texture format reads it: Z32 as CT32, Z24 as CT24
    // (alpha from TEXA), Z16 as CT16.
    const double range = 1.0 / depthScale(source->zpsm);
    const uint8_t ta0 = static_cast<uint8_t>(state.texa.ta0), ta1 = static_cast<uint8_t>(state.texa.ta1);
    std::vector<uint8_t> texels(depth.size() * 4u);
    for (size_t i = 0; i < depth.size(); ++i)
    {
        const uint32_t z = static_cast<uint32_t>(std::min(static_cast<double>(depth[i]) * range, range - 1.0));
        uint8_t *t = &texels[i * 4u];
        switch (tex0.psm)
        {
        case 0x30u:
            t[0] = z & 0xFFu, t[1] = z >> 8 & 0xFFu, t[2] = z >> 16 & 0xFFu, t[3] = z >> 24 & 0xFFu;
            break;
        case 0x31u:
            t[0] = z & 0xFFu, t[1] = z >> 8 & 0xFFu, t[2] = z >> 16 & 0xFFu;
            t[3] = state.texa.aem && (z & 0xFFFFFFu) == 0u ? 0u : ta0;
            break;
        default:
            t[0] = static_cast<uint8_t>((z & 0x1Fu) << 3), t[1] = static_cast<uint8_t>((z >> 5 & 0x1Fu) << 3);
            t[2] = static_cast<uint8_t>((z >> 10 & 0x1Fu) << 3);
            t[3] = (z & 0x8000u) ? ta1 : (state.texa.aem && (z & 0x7FFFu) == 0u ? 0u : ta0);
            break;
        }
    }
    if (!m_depthTexture)
        glGenTextures(1, &m_depthTexture);
    glBindTexture(GL_TEXTURE_2D, m_depthTexture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(w), static_cast<GLsizei>(h), 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, texels.data());
    width = static_cast<float>(w);
    height = static_cast<float>(h);
    return true;
}

bool GSNativeBackend::bindTexture(const GSDrawState &state, const RenderTarget &target, float &width, float &height)
{
    const GSContext &context = state.context;
    GLuint texture = 0;
    bool fromTarget = false;
    // PS2X_GS_NATIVE_LOG: each distinct texture read that overlaps a render target's pages, as (render target, texture
    // format and base, offset in pages): what still falls back to the texture cache, which has no drawn pixels.
    if (static const bool logOverlaps = std::getenv("PS2X_GS_NATIVE_LOG") != nullptr; logOverlaps)
    {
        static std::unordered_map<uint64_t, uint32_t> s_seen;
        for (const auto &[fbp, rt] : m_targets)
        {
            const uint32_t base = fbp * 32u;
            const uint32_t blocks = (rt.fbw * 64u * rt.height * 4u) / 256u;
            if (context.tex0.tbp0 < base || context.tex0.tbp0 >= base + blocks)
                continue;
            const uint64_t key = static_cast<uint64_t>(fbp) << 40 | static_cast<uint64_t>(context.tex0.psm) << 32 |
                                 static_cast<uint64_t>(context.tex0.tbp0 - base) << 8 | context.tex0.tbw;
            if (s_seen.size() < 300u && s_seen.emplace(key, 0u).second)
                std::fprintf(stderr,
                             "[gs:native] read over target fbp=%u fbw=%u psm=0x%x: tex psm=0x%x tbp0=+%u blocks tbw=%u "
                             "tw=%u th=%u into fbp=%u\n",
                             fbp, rt.fbw, rt.psm, context.tex0.psm, context.tex0.tbp0 - base, context.tex0.tbw,
                             context.tex0.tw, context.tex0.th, target.fbp);
        }
    }
    if ((context.tex0.psm & 0x30u) == 0x30u && bindDepthAsTexture(state, width, height))
    {
        fromTarget = true;
        texture = m_depthTexture;
    }
    else if (colorFormat(context.tex0.psm))
    {
        // Only a read in the target's own layout samples it directly (CT24 reads CT32 pixels as they are); another
        // format (a CT32 buffer read as CT16 to split its channels) sees GS memory, through resolveTargetsUnder.
        RenderTarget *source = renderTargetAtBlock(context.tex0.tbp0);
        const auto sameLayout = [](uint32_t a, uint32_t b) { return a == b || ((a | b) & ~0x1u) == 0u; };
        if (source && !sameLayout(source->psm, context.tex0.psm))
            source = nullptr;
        if (source)
        {
            // A render target sampled as a texture (copies, blurs, depth of field). When the draw also writes it,
            // sample a copy of it as it was before the draw.
            static const bool logSamples = std::getenv("PS2X_GS_NATIVE_LOG") != nullptr;
            static uint32_t s_logged = 0;
            if (logSamples && s_logged < 200u)
            {
                ++s_logged;
                std::fprintf(stderr,
                             "[gs:native] target as texture: fbp=%u fbw=%u psm=0x%x -> tex psm=0x%x tbw=%u tw=%u th=%u "
                             "fst=%d into fbp=%u (self=%d)\n",
                             source->fbp, source->fbw, source->psm, context.tex0.psm, context.tex0.tbw,
                             context.tex0.tw, context.tex0.th, state.prim.fst ? 1 : 0, target.fbp,
                             source->fbp == target.fbp ? 1 : 0);
            }
            fromTarget = true;
            width = static_cast<float>(source->width);
            height = static_cast<float>(source->height);
            texture = source->color;
            m_lastSourceFramebuffer = source->framebuffer;
            if (source->fbp == target.fbp && !std::getenv("PS2X_GS_NATIVE_NOFEEDBACKCOPY"))
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
    glUniform1i(m_uniforms.decode, 0);
    if (!fromTarget)
    {
        resolveTargetsUnder(context.tex0);
        if (static const bool cpuDecode = std::getenv("PS2X_GS_NATIVE_CPUDECODE") != nullptr;
            !cpuDecode && bindDecodedTexture(state))
        {
            width = height = 1.0f;
            return true;
        }
        const uint64_t tex0 = rawTex0(context.tex0);
        const uint64_t texa = rawTexa(state.texa);
        // The cache revalidates a texture's pages on every lookup; while nothing was uploaded since this TEX0/TEXA was
        // last looked up, its texture is still the same one.
        ps2x::native::TextureCache &cache = ps2x::native::textures();
        LastLookup &last = m_lastLookups[tex0 ^ (texa * 0x9E3779B97F4A7C15ull)];
        if (!last.texture || last.tex0 != tex0 || last.texa != texa || last.uploads != cache.uploads())
        {
            last.texture = cache.texture(tex0, texa);
            last.tex0 = tex0;
            last.texa = texa;
            last.uploads = cache.uploads();
        }
        const std::shared_ptr<const ps2x::native::TextureCache::Texture> decoded = last.texture;
        if (!decoded || decoded->width == 0u || decoded->height == 0u || decoded->rgba.empty())
            return false;
        const uint64_t version = decoded->version;
        if (static const bool logTextures = std::getenv("PS2X_GS_NATIVE_LOG") != nullptr; logTextures)
        {
            static std::unordered_map<uint64_t, uint64_t> s_seen;
            if (s_seen.size() < 200u && s_seen.emplace(tex0 ^ (version << 1), 0u).second)
            {
                size_t nonZero = 0, opaque = 0;
                for (const uint32_t texel : decoded->rgba)
                {
                    nonZero += (texel & 0x00FFFFFFu) != 0u;
                    opaque += (texel >> 24) != 0u;
                }
                std::fprintf(stderr, "[gs:native] texture psm=0x%x tbp=%u cbp=%u %ux%u: %zu non-black, %zu with alpha\n",
                             context.tex0.psm, context.tex0.tbp0, context.tex0.cbp, decoded->width, decoded->height,
                             nonZero, opaque);
            }
        }
        UploadedTexture &uploaded = m_uploaded[decoded->key ^ (texa << 1)];
        if (!uploaded.texture)
            glGenTextures(1, &uploaded.texture);
        glBindTexture(GL_TEXTURE_2D, uploaded.texture);
        if (uploaded.version != version || uploaded.width != decoded->width || uploaded.height != decoded->height)
        {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, decoded->width, decoded->height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         decoded->rgba.data());
            uploaded.version = version;
            uploaded.width = decoded->width;
            uploaded.height = decoded->height;
        }
        texture = uploaded.texture;
        width = static_cast<float>(decoded->width);
        height = static_cast<float>(decoded->height);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    // CLAMP: repeat (0, region repeat 3) or clamp (1, region clamp 2); TEX1.MMAG picks bilinear.
    const GLint wrapS = (context.clamp & 0x3u) == 1u || (context.clamp & 0x3u) == 2u ? GL_CLAMP_TO_EDGE : GL_REPEAT;
    const GLint wrapT = (context.clamp >> 2 & 0x3u) == 1u || (context.clamp >> 2 & 0x3u) == 2u ? GL_CLAMP_TO_EDGE
                                                                                              : GL_REPEAT;
    const GLint filter = (context.tex1 >> 5 & 0x1u) != 0u ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fromTarget ? GL_CLAMP_TO_EDGE : wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fromTarget ? GL_CLAMP_TO_EDGE : wrapT);
    glUniform1i(m_uniforms.targetWrap, fromTarget ? 1 : 0);
    if (fromTarget)
    {
        const uint64_t clamp = context.clamp;
        const int maxU = (1 << std::min<uint32_t>(context.tex0.tw, 10u)) - 1;
        const int maxV = (1 << std::min<uint32_t>(context.tex0.th, 10u)) - 1;
        glUniform4i(m_uniforms.wrapU, static_cast<int>(clamp & 0x3u), maxU, static_cast<int>(clamp >> 4 & 0x3FFu),
                    static_cast<int>(clamp >> 14 & 0x3FFu));
        glUniform4i(m_uniforms.wrapV, static_cast<int>(clamp >> 2 & 0x3u), maxV, static_cast<int>(clamp >> 24 & 0x3FFu),
                    static_cast<int>(clamp >> 34 & 0x3FFu));
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glUniform1f(m_uniforms.fst, fromTarget ? 2.0f : 1.0f);
    return true;
}

void GSNativeBackend::applyBlend(const GSDrawState &state)
{
    m_blendOut = 0;
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
    else if (a == Cd && b == Zero && d == Cd && c != 1u)
        m_blendOut = 1, source = GL_DST_COLOR, destination = GL_ONE; // Cd * C + Cd: brightening
    else if (a == Cs && b == Zero && d == Cs)
        m_blendOut = 2, source = GL_ONE, destination = GL_ZERO; // Cs * (1 + C)
    else if (a == Cs && b == Cd && d == Zero)
        equation = GL_FUNC_SUBTRACT, source = factor, destination = factor;
    else if (a == Cd && b == Cs && d == Zero)
        equation = GL_FUNC_REVERSE_SUBTRACT, source = factor, destination = factor;
    else if (a == Zero && b == Cs && d == Cs)
        source = inverse, destination = GL_ZERO;
    else if (a == Zero && b == Cd && d == Cd)
        source = GL_ZERO, destination = inverse;
    else if (a == Zero && d == Zero)
        source = GL_ZERO, destination = GL_ZERO; // a negative result clamps to black
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
    target.dirty = true;
    // Rows reached: the lowest vertex, clipped by the scissor (a full-screen scissor says nothing about the buffer).
    const uint32_t lowest = std::min<uint32_t>(context.scissor.y1 + 1u, static_cast<uint32_t>(std::max(0.0f, std::ceil(m_batchMaxY))));
    target.usedHeight = std::min(target.height, std::max(target.usedHeight, lowest));
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glViewport(0, 0, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height));
    glEnable(GL_SCISSOR_TEST);
    glScissor(context.scissor.x0, context.scissor.y0, context.scissor.x1 - context.scissor.x0 + 1,
              context.scissor.y1 - context.scissor.y0 + 1);

    // PS2X_GS_NATIVE_OFF=depth|alpha|blend|color (any of them): diagnostics that turn one stage off.
    static const char *s_off = std::getenv("PS2X_GS_NATIVE_OFF");
    const bool offDepth = s_off && std::strstr(s_off, "depth"), offAlpha = s_off && std::strstr(s_off, "alpha");
    const bool offBlend = s_off && std::strstr(s_off, "blend"), offColor = s_off && std::strstr(s_off, "color");
    const bool depthTest = !offDepth && (context.test >> 16 & 0x1u) != 0u;
    if (depthTest)
    {
        static constexpr GLenum kDepthFunc[4] = {GL_NEVER, GL_ALWAYS, GL_GEQUAL, GL_GREATER};
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(kDepthFunc[context.test >> 17 & 0x3u]);
        glDepthMask(context.zbuf.zmask ? GL_FALSE : GL_TRUE);
        if (!context.zbuf.zmask)
        {
            target.zbp = context.zbuf.zbp;
            target.zpsm = context.zbuf.psm;
        }
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }
    // FBMSK bits set keep the frame buffer's bits; whole bytes map to channel masks. CT24 has no alpha to write.
    const uint32_t fbmsk = context.frame.fbmsk;
    const bool keepAlpha = (fbmsk >> 24) == 0xFFu || context.frame.psm == 0x01u;
    glColorMask((fbmsk & 0xFFu) != 0xFFu, (fbmsk >> 8 & 0xFFu) != 0xFFu, (fbmsk >> 16 & 0xFFu) != 0xFFu, !keepAlpha);
    target.alphaWritten = target.alphaWritten || !keepAlpha;
    glDisable(GL_CULL_FACE);
    if (offBlend)
        glDisable(GL_BLEND);
    else
        applyBlend(state);
    if (offColor)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glUseProgram(m_program);
    glUniform1i(m_uniforms.blendOut, state.prim.abe ? m_blendOut : 0);
    glUniform1i(m_uniforms.blendFactor, static_cast<int>(context.alpha >> 4 & 0x3u));
    glUniform1f(m_uniforms.blendFix, static_cast<float>(context.alpha >> 32 & 0xFFu) / 128.0f);
    glUniform2f(m_uniforms.targetSize, static_cast<float>(target.width), static_cast<float>(target.height));
    float texWidth = 1.0f, texHeight = 1.0f;
    const bool textured = state.prim.tme && bindTexture(state, target, texWidth, texHeight);
    // PS2X_GS_NATIVE_DRAWLOG=<first batch>,<count>: what each batch draws, to compare a broken screen with the GS's.
    if (static const char *drawLog = std::getenv("PS2X_GS_NATIVE_DRAWLOG"); drawLog)
    {
        static uint64_t s_batches = 0;
        static const uint64_t s_first = std::strtoull(drawLog, nullptr, 10);
        static const uint64_t s_count = [] {
            const char *comma = std::strchr(std::getenv("PS2X_GS_NATIVE_DRAWLOG"), ',');
            return comma ? std::strtoull(comma + 1, nullptr, 10) : 400ull;
        }();
        const uint64_t index = s_batches++;
        if (index >= s_first && index < s_first + s_count)
            std::fprintf(stderr,
                         "[gs:draw] #%llu fbp=%u psm=0x%x fbmsk=0x%x prim=%d tme=%d textured=%d tex0 psm=0x%x tbp=%u tbw=%u "
                         "tw=%u th=%u tfx=%u tcc=%u cbp=%u abe=%d alpha=0x%llx test=0x%llx fst=%d verts=%zu "
                         "v0=(%.1f,%.1f z=%.6f rgba=%u,%u,%u,%u st=%.1f,%.1f,%.2f) zbuf=%u/0x%x/%d ctxt=%d iip=%d "
                         "scissor=%u-%u,%u-%u\n",
                         static_cast<unsigned long long>(index), context.frame.fbp, context.frame.psm,
                         context.frame.fbmsk, static_cast<int>(state.prim.type), state.prim.tme ? 1 : 0,
                         textured ? 1 : 0, context.tex0.psm, context.tex0.tbp0, context.tex0.tbw, context.tex0.tw,
                         context.tex0.th, context.tex0.tfx, context.tex0.tcc, context.tex0.cbp, state.prim.abe ? 1 : 0,
                         static_cast<unsigned long long>(context.alpha), static_cast<unsigned long long>(context.test),
                         state.prim.fst ? 1 : 0, m_batch.size(), m_batch[0].x, m_batch[0].y, m_batch[0].z, m_batch[0].r,
                         m_batch[0].g, m_batch[0].b, m_batch[0].a, m_batch[0].s, m_batch[0].t, m_batch[0].q,
                         context.zbuf.zbp, context.zbuf.psm, context.zbuf.zmask ? 1 : 0, state.prim.ctxt ? 1 : 0,
                         state.prim.iip ? 1 : 0, context.scissor.x0, context.scissor.x1, context.scissor.y0, context.scissor.y1);
    }
    glUniform1i(m_uniforms.textured, textured ? 1 : 0);
    glUniform2f(m_uniforms.texSize, texWidth, texHeight);
    glUniform1i(m_uniforms.sampler, 0);
    glUniform1i(m_uniforms.tfx, context.tex0.tfx);
    glUniform1i(m_uniforms.tcc, context.tex0.tcc);
    glUniform1i(m_uniforms.fog, state.prim.fge ? 1 : 0);
    glUniform3f(m_uniforms.fogColor, state.fogR / 255.0f, state.fogG / 255.0f, state.fogB / 255.0f);
    // TEST: ATE, ATST, AREF. A failed test keeps the pixel unwritten (AFAIL other than KEEP is not translated yet).
    bool alphaTest = !offAlpha && (context.test & 0x1u) != 0u;
    // ATST NEVER with AFAIL other than KEEP: every pixel fails, and AFAIL says what is written anyway. That is a
    // write mask (FB_ONLY colour without Z, ZB_ONLY Z alone, RGB_ONLY colour without alpha or Z), not a test.
    if (alphaTest && (context.test >> 1 & 0x7u) == 0u && (context.test >> 12 & 0x3u) != 0u)
    {
        alphaTest = false;
        switch (context.test >> 12 & 0x3u)
        {
        case 1u:
            glDepthMask(GL_FALSE);
            break;
        case 2u:
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            break;
        default:
            glDepthMask(GL_FALSE);
            glColorMask((fbmsk & 0xFFu) != 0xFFu, (fbmsk >> 8 & 0xFFu) != 0xFFu, (fbmsk >> 16 & 0xFFu) != 0xFFu,
                        GL_FALSE);
            break;
        }
    }
    glUniform1i(m_uniforms.alphaTest, alphaTest ? static_cast<int>(context.test >> 1 & 0x7u) : 1);
    glUniform1f(m_uniforms.alphaRef, static_cast<float>(context.test >> 4 & 0xFFu));
    // DATE: the destination alpha's top bit (DATM picks which value passes), read from a copy of the target.
    const bool date = (context.test >> 14 & 0x1u) != 0u;
    glUniform1i(m_uniforms.date, date ? ((context.test >> 15 & 0x1u) ? 2 : 1) : 0);
    glUniform1i(m_uniforms.dest, 1);
    if (date)
    {
        if (!m_destTexture || m_destWidth != target.width || m_destHeight != target.height)
        {
            if (!m_destTexture)
                glGenTextures(1, &m_destTexture);
            glBindTexture(GL_TEXTURE_2D, m_destTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target.width, target.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            m_destWidth = target.width;
            m_destHeight = target.height;
        }
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_destTexture);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, target.framebuffer);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, target.width, target.height);
        glActiveTexture(GL_TEXTURE0);
    }

    glBindVertexArray(m_vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(m_batch.size() * sizeof(Vertex)), m_batch.data(),
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_batch.size()));
    // PS2X_GS_NATIVE_PROBE=<fbp>: after each batch into that render target, its pixel (320, 128) and the batch.
    if (static const char *probe = std::getenv("PS2X_GS_NATIVE_PROBE");
        probe && std::strtoul(probe, nullptr, 10) == target.fbp)
    {
        static uint32_t s_probes = 0;
        if (s_probes++ < 60u)
        {
            uint8_t pixel[4] = {};
            glBindFramebuffer(GL_READ_FRAMEBUFFER, target.framebuffer);
            glReadPixels(320, 128, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
            std::fprintf(stderr, "[gs:probe] fbp=%u after %zu verts tme=%d textured=%d tbp=%u: (320,128)=%u,%u,%u,%u "
                                 "fbo=%x\n",
                         target.fbp, m_batch.size(), state.prim.tme ? 1 : 0, textured ? 1 : 0, context.tex0.tbp0,
                         pixel[0], pixel[1], pixel[2], pixel[3], glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER));
            if (m_lastSourceFramebuffer)
            {
                uint8_t texel[4] = {};
                glBindFramebuffer(GL_READ_FRAMEBUFFER, m_lastSourceFramebuffer);
                glReadPixels(320, 224, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, texel);
                std::fprintf(stderr, "[gs:probe]   source (320,224)=%u,%u,%u,%u\n", texel[0], texel[1], texel[2],
                             texel[3]);
                m_lastSourceFramebuffer = 0;
            }
            float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, s0 = 1e9f, s1 = -1e9f, t0 = 1e9f, t1 = -1e9f, z0 = 1e9f;
            for (const Vertex &v : m_batch)
            {
                x0 = std::min(x0, v.x), x1 = std::max(x1, v.x), y0 = std::min(y0, v.y), y1 = std::max(y1, v.y);
                s0 = std::min(s0, v.s), s1 = std::max(s1, v.s), t0 = std::min(t0, v.t), t1 = std::max(t1, v.t);
                z0 = std::min(z0, v.z);
            }
            std::fprintf(stderr, "[gs:probe]   extent x %.1f..%.1f y %.1f..%.1f s %.1f..%.1f t %.1f..%.1f z %.6f "
                                 "tex %.0fx%.0f\n",
                         x0, x1, y0, y1, s0, s1, t0, t1, z0, texWidth, texHeight);
            if (m_batch.size() <= 6u)
                for (const Vertex &v : m_batch)
                    std::fprintf(stderr, "[gs:probe]   vertex (%.1f,%.1f) z=%.6f rgba=%u,%u,%u,%u\n", v.x, v.y, v.z, v.r,
                                 v.g, v.b, v.a);
        }
    }
    glBindVertexArray(0);
    m_batch.clear();
    m_batchMaxY = 0.0f;
}

void GSNativeBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    m_vram = vram;
    // New GS memory (or the same, reloaded): the GPU copy takes all of it again.
    std::fill(m_dirtyBlocks.begin(), m_dirtyBlocks.end(), uint8_t{1});
    m_anyDirty = true;
    std::fill(m_gpuNewer.begin(), m_gpuNewer.end(), uint8_t{0});
    m_anyGpuNewer = false;
    m_memory->Initialize(vram, vramSize);
}

void GSNativeBackend::Reset()
{
    m_batch.clear();
    std::fill(m_dirtyBlocks.begin(), m_dirtyBlocks.end(), uint8_t{1});
    m_anyDirty = true;
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

void GSNativeBackend::markDirty(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    // The 256-byte blocks a rectangle of the buffer touches: every 8 pixels of a row lie in one block, and no block
    // is less than 4 rows high; the far edges too.
    const uint32_t x1 = x + std::max<uint32_t>(w, 1u), y1 = y + std::max<uint32_t>(h, 1u);
    for (uint32_t yy = y;; yy = std::min(yy + 4u, y1 - 1u))
    {
        for (uint32_t xx = x;; xx = std::min(xx + 8u, x1 - 1u))
        {
            m_dirtyBlocks[GSMem::BlockOf(psm, bp, std::max<uint32_t>(bw, 1u), xx, yy) % m_dirtyBlocks.size()] = 1u;
            if (xx == x1 - 1u)
                break;
        }
        if (yy == y1 - 1u)
            break;
    }
    m_anyDirty = true;
}

void GSNativeBackend::BeginTransfer(const GSTransferCommand &command)
{
    flushBatch();
    // A transfer that reads GS memory on the CPU side (local to host, local to local) needs the resolved pixels
    // there; one into GS memory only replaces blocks, which reach the GPU copy by themselves.
    if (command.direction == 1u || command.direction == 2u)
        pullGpuPages();
    if (static const bool logTransfers = std::getenv("PS2X_GS_NATIVE_LOG") != nullptr; logTransfers)
    {
        static uint32_t s_logged = 0, s_copies = 0;
        if (command.direction == 0u ? s_logged++ < 100u : s_copies++ < 300u)
            std::fprintf(stderr, "[gs:native] transfer dir=%u dbp=%u dbw=%u dpsm=0x%x dst=%u,%u sbp=%u size=%ux%u\n",
                         command.direction, command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm,
                         command.trxpos.dsax, command.trxpos.dsay, command.bitbltbuf.sbp, command.trxreg.rrw,
                         command.trxreg.rrh);
    }
    m_transfer = command;
    m_transferDirect = command.direction;
    if (command.direction == 0u || command.direction == 2u)
        markDirty(command.bitbltbuf.dpsm, command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.trxpos.dsax,
                  command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh);
    if (command.direction == 2u && copyBetweenTargets(command))
    {
        m_memory->BeginTransfer(command);
        return;
    }
    if (command.direction == 0u || command.direction == 2u)
        dropTargetsUnder(command);
    m_memory->BeginTransfer(command);
}

bool GSNativeBackend::copyBetweenTargets(const GSTransferCommand &command)
{
    // A local-to-local copy out of a render target (a frame copied to the displayed buffer, a region saved for
    // later): the same copy on the GPU, into the destination render target (made if need be).
    const GSBitBltBuf &b = command.bitbltbuf;
    const bool colour = [](uint8_t psm) { return psm == 0x00u || psm == 0x01u; }(b.spsm);
    if (!colour || b.spsm != b.dpsm || b.sbp % 32u != 0u || b.dbp % 32u != 0u || !ensureContext())
        return false;
    auto source = m_targets.find(b.sbp / 32u);
    if (source == m_targets.end() || source->second.fbw != b.sbw)
        return false;
    GSFrameReg frame{};
    frame.fbp = b.dbp / 32u;
    frame.fbw = b.dbw;
    frame.psm = b.dpsm;
    RenderTarget &destination = renderTarget(frame);
    const RenderTarget &from = m_targets.find(b.sbp / 32u)->second; // renderTarget may have rehashed the map
    const GLint sx = command.trxpos.ssax, sy = command.trxpos.ssay;
    const GLint dx = command.trxpos.dsax, dy = command.trxpos.dsay;
    const GLint w = command.trxreg.rrw, h = command.trxreg.rrh;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, from.framebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, destination.framebuffer);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glBlitFramebuffer(sx, sy, sx + w, sy + h, dx, dy, dx + w, dy + h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    destination.dirty = true;
    destination.usedHeight = std::min(destination.height,
                                      std::max(destination.usedHeight, static_cast<uint32_t>(dy + h)));
    return true;
}

void GSNativeBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    m_memory->UploadImage(data, sizeBytes);
    // The transfer's data may come in pieces, with draws (and a send of the dirty blocks) in between.
    if (m_transferDirect == 0u)
        markDirty(m_transfer.bitbltbuf.dpsm, m_transfer.bitbltbuf.dbp, m_transfer.bitbltbuf.dbw, m_transfer.trxpos.dsax,
                  m_transfer.trxpos.dsay, m_transfer.trxreg.rrw, m_transfer.trxreg.rrh);
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

bool GSNativeBackend::readCircuit(int circuit, uint64_t dispfb, uint64_t display, bool fieldMode,
                                  std::vector<uint8_t> &pixels, uint32_t &width, uint32_t &height, uint32_t &fbp)
{
    const DisplayRead read = decodeDisplay(dispfb, display);
    fbp = read.fbp;
    const uint32_t lines = fieldMode ? (read.height + 1u) / 2u : read.height;
    auto it = m_targets.find(read.fbp);
    if (it != m_targets.end())
    {
        const RenderTarget &target = it->second;
        width = std::min(read.width, target.width - std::min(read.x, target.width));
        height = std::min(lines, target.height - std::min(read.y, target.height));
        if (width == 0u || height == 0u)
            return false;
        const size_t bytes = static_cast<size_t>(width) * height * 4u;
        pixels.resize(bytes);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, target.framebuffer);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        // Asynchronous read back through two pixel buffers per read circuit: this frame's read is queued and the
        // previous presentation's is copied out, so the GS thread does not wait for the GPU to finish the frame (one
        // presentation of latency). Per circuit, not per displayed buffer: a game alternating two buffers (one per
        // field) would otherwise get every buffer's picture of two presentations ago, every other one repeated.
        // A size change, or the first frame, reads synchronously.
        ReadbackBuffers &buffers = m_readbacks[static_cast<uint32_t>(circuit)];
        if (!buffers.pbo[0])
            glGenBuffers(2, buffers.pbo);
        const int current = buffers.next;
        const int previous = current ^ 1;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, buffers.pbo[current]);
        if (buffers.bytes[current] != bytes)
        {
            glBufferData(GL_PIXEL_PACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_STREAM_READ);
            buffers.bytes[current] = bytes;
        }
        glReadPixels(static_cast<GLint>(read.x), static_cast<GLint>(read.y), static_cast<GLsizei>(width),
                     static_cast<GLsizei>(height), GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        // PS2X_GS_NATIVE_SYNCREAD=1: show this presentation's own picture (waits for the GPU), for comparison.
        static const bool syncRead = std::getenv("PS2X_GS_NATIVE_SYNCREAD") != nullptr;
        const int source = !syncRead && buffers.ready[previous] && buffers.bytes[previous] == bytes ? previous : current;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, buffers.pbo[source]);
        if (const void *mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr>(bytes), GL_MAP_READ_BIT))
        {
            std::memcpy(pixels.data(), mapped, bytes);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        buffers.ready[current] = true;
        buffers.next = previous;
        return true;
    }
    // Not drawn by us (an image the game uploaded straight into the displayed buffer): read GS memory, CT32 / CT24.
    if (!m_vram)
        return false;
    pullGpuPages();
    const uint32_t fbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
    const uint32_t psm = static_cast<uint32_t>((dispfb >> 15) & 0x1Fu);
    width = std::min<uint32_t>(read.width, 2048u);
    height = std::min<uint32_t>(lines, 1024u);
    if (width == 0u || height == 0u || fbw == 0u || (psm != 0x00u && psm != 0x01u))
        return false;
    pixels.resize(static_cast<size_t>(width) * height * 4u);
    thread_local GSMem::TexturePageCache cache;
    cache.Invalidate();
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
        {
            uint32_t c = GSMem::ReadTexture(cache, m_vram, psm, read.fbp * 32u, fbw, read.x + x, read.y + y);
            if (psm == 0x01u)
                c |= 0x80000000u;
            std::memcpy(&pixels[(static_cast<size_t>(y) * width + x) * 4u], &c, 4u);
        }
    return true;
}

PresentationFrame GSNativeBackend::Present(const GSPresentationRequest &request)
{
    flushBatch();
    PresentationFrame frame;
    if (!ensureContext())
        return frame;
    // Interlaced field mode (SMODE2 INT and FFMD): the game draws one field of half the lines; show each line twice.
    const bool fieldMode = (request.smode2 & 0x3u) == 0x3u;
    // PMODE: EN1, EN2, MMOD (blend with ALP rather than circuit 1's alpha), SLBG (blend over BGCOLOR rather than
    // circuit 2), ALP. Circuit 1 is blended over circuit 2 (or the background colour).
    const uint64_t pmode = request.pmode;
    const bool en1 = (pmode & 0x1u) != 0u, en2 = (pmode & 0x2u) != 0u;
    const bool mmod = (pmode >> 5 & 0x1u) != 0u, slbg = (pmode >> 7 & 0x1u) != 0u;
    const float alp = static_cast<float>(pmode >> 8 & 0xFFu) / 255.0f;
    std::vector<uint8_t> top, bottom;
    uint32_t topWidth = 0, topHeight = 0, topFbp = 0, bottomWidth = 0, bottomHeight = 0, bottomFbp = 0;
    const bool haveTop = en1 && readCircuit(1, request.dispfb1, request.display1, fieldMode, top, topWidth, topHeight, topFbp);
    const bool haveBottom = en2 && !slbg &&
                            readCircuit(2, request.dispfb2, request.display2, fieldMode, bottom, bottomWidth, bottomHeight,
                                        bottomFbp);
    if (static const bool logPresents = std::getenv("PS2X_GS_NATIVE_LOG") != nullptr; logPresents)
    {
        static uint32_t s_presents = 0;
        if ((s_presents++ % 50u) == 0u)
        {
            std::string targets;
            for (const auto &[fbp, rt] : m_targets)
                targets += std::string(" ") + std::to_string(fbp) + '/' + std::to_string(rt.fbw) + (rt.dirty ? "*" : "");
            std::fprintf(stderr,
                         "[gs:native] present #%u pmode=0x%llx dispfb1=0x%llx display1=0x%llx dispfb2=0x%llx "
                         "display2=0x%llx smode2=0x%llx top=%d(fbp %u) bottom=%d(fbp %u) targets:%s\n",
                         s_presents, static_cast<unsigned long long>(pmode),
                         static_cast<unsigned long long>(request.dispfb1), static_cast<unsigned long long>(request.display1),
                         static_cast<unsigned long long>(request.dispfb2), static_cast<unsigned long long>(request.display2),
                         static_cast<unsigned long long>(request.smode2), haveTop ? 1 : 0, topFbp, haveBottom ? 1 : 0,
                         bottomFbp, targets.c_str());
        }
    }
    if (!haveTop && !haveBottom)
        return frame;
    const uint32_t width = std::max(haveTop ? topWidth : 0u, haveBottom ? bottomWidth : 0u);
    const uint32_t height = std::max(haveTop ? topHeight : 0u, haveBottom ? bottomHeight : 0u);
    frame.width = width;
    frame.height = height;
    frame.displayFbp = haveTop ? topFbp : bottomFbp;
    frame.sourceFbp = frame.displayFbp;
    frame.pixels.assign(static_cast<size_t>(width) * height * 4u, 0u);
    const uint8_t bg[3] = {static_cast<uint8_t>(request.bgcolor & 0xFFu), static_cast<uint8_t>(request.bgcolor >> 8 & 0xFFu),
                           static_cast<uint8_t>(request.bgcolor >> 16 & 0xFFu)};
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
        {
            uint8_t *out = &frame.pixels[(static_cast<size_t>(y) * width + x) * 4u];
            uint8_t under[3] = {bg[0], bg[1], bg[2]};
            if (haveBottom && x < bottomWidth && y < bottomHeight)
                std::memcpy(under, &bottom[(static_cast<size_t>(y) * bottomWidth + x) * 4u], 3u);
            else if (!slbg && !haveBottom)
                under[0] = under[1] = under[2] = 0u;
            if (haveTop && x < topWidth && y < topHeight)
            {
                const uint8_t *over = &top[(static_cast<size_t>(y) * topWidth + x) * 4u];
                // Circuit 1's alpha counts 0x80 as 1.0. With only circuit 1 shown it simply replaces the background.
                const float a = !(en2 || slbg) ? 1.0f : mmod ? alp : std::min(1.0f, over[3] / 128.0f);
                for (int c = 0; c < 3; ++c)
                    out[c] = static_cast<uint8_t>(over[c] * a + under[c] * (1.0f - a) + 0.5f);
            }
            else
            {
                std::memcpy(out, under, 3u);
            }
            out[3] = 0xFFu;
        }
    if (fieldMode)
    {
        std::vector<uint8_t> doubled(frame.pixels.size() * 2u);
        const size_t row = static_cast<size_t>(width) * 4u;
        for (uint32_t y = 0; y < height; ++y)
        {
            std::memcpy(&doubled[(2u * y) * row], &frame.pixels[y * row], row);
            std::memcpy(&doubled[(2u * y + 1u) * row], &frame.pixels[y * row], row);
        }
        frame.pixels.swap(doubled);
        frame.height = height * 2u;
    }
    // The host reads presented frames 640 pixels per row (GS::copyLatchedHostPresentationFrame, as the CPU and
    // paraLLEl-GS backends hand them): a narrower picture (The Third Age's is 512 wide) is laid out on that stride,
    // or its copy runs past the end and the window shows the magenta blank.
    constexpr uint32_t kHostFrameWidth = 640u;
    if (frame.width != kHostFrameWidth)
    {
        const uint32_t shown = std::min(frame.width, kHostFrameWidth);
        std::vector<uint8_t> strided(static_cast<size_t>(kHostFrameWidth) * frame.height * 4u, 0u);
        for (uint32_t y = 0; y < frame.height; ++y)
            std::memcpy(&strided[static_cast<size_t>(y) * kHostFrameWidth * 4u],
                        &frame.pixels[static_cast<size_t>(y) * frame.width * 4u], static_cast<size_t>(shown) * 4u);
        frame.pixels.swap(strided);
        frame.width = shown;
    }
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
    target.dirty = true;
    target.alphaWritten = true;
    return true;
}

uint32_t GSNativeBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    pullGpuPages();
    return m_memory->ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSNativeBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    return m_memory->ReadVram(psm, base, bw, x, y);
}

void GSNativeBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    pullGpuPages();
    markDirty(psm, base, bw, x, y, 1u, 1u);
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
