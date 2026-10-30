#include "MiniTest.h"
#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_registers.h"
#include "runtime/native/native_renderer.h"

#include <cstdint>
#include <vector>

using ps2x::native::GsMaterial;
using ps2x::native::GsVertex;
using ps2x::native::Renderer;

namespace
{
    struct Taken
    {
        GsMaterial material;
        std::vector<GsVertex> triangles;
    };
    std::vector<Taken> g_taken;

    void take(void *, const GsMaterial &material, const GsVertex *triangles, uint32_t count)
    {
        g_taken.push_back({material, std::vector<GsVertex>(triangles, triangles + count)});
    }

    class RecordingBackend final : public GSRasterBackend
    {
    public:
        std::vector<GSPrimitiveBatch> batches;
        std::vector<GSTex0Reg> cluts;

        void Initialize(uint8_t *, uint32_t) override {}
        void Reset() override {}
        void Submit(const GSPrimitiveBatch &batch) override { batches.push_back(batch); }
        void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &) override { cluts.push_back(tex0); }
        void BeginTransfer(const GSTransferCommand &) override {}
        void UploadImage(const uint8_t *, uint32_t) override {}
        void Flush() override {}
        void TextureFlush() override {}
        void Sync(GSSyncReason) override {}
        PresentationFrame Present(const GSPresentationRequest &) override { return {}; }
        bool ClearFramebuffer(const GSContext &, uint32_t) override { return false; }
        uint32_t ConsumeLocalToHostBytes(uint8_t *, uint32_t) override { return 0; }
        uint32_t ReadVram(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) const override { return 0; }
        void WriteVram(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) override {}
        void SnapshotVram(std::vector<uint8_t> &) const override {}
        GSTransferSnapshot GetTransferSnapshot() const override { return {}; }
    };

    GsVertex vertex(float x, float y, uint32_t z, uint32_t rgba)
    {
        GsVertex v;
        v.x = x;
        v.y = y;
        v.z = z;
        v.s = 0.25f;
        v.t = 0.5f;
        v.q = 2.0f;
        v.rgba = rgba;
        v.fog = 0x40;
        return v;
    }

    GsMaterial material()
    {
        GsMaterial m;
        m.prim = ps2x::gs::decodePrim(0x5c);
        m.tex0 = ps2x::gs::decodeTex0(0x20064d66a93131ebull);
        m.tex1 = 0x61;
        m.clamp = 0x7f0007ff;
        m.alpha = 0x44;
        m.test = 0x50000;
        return m;
    }
}

void register_native_gs_draw_tests()
{
    MiniTest::Case("NativeGsDraw", [](TestCase &tc)
    {
        tc.Run("TEX0 and PRIM decode the fields the GS reads", [](TestCase &t)
        {
            const GSTex0Reg tex0 = ps2x::gs::decodeTex0(0x20064d66a93131ebull);
            t.Equals(tex0.tbp0, 0x31ebu, "TBP0");
            t.Equals(static_cast<uint32_t>(tex0.tbw), 4u, "TBW");
            t.Equals(static_cast<uint32_t>(tex0.psm), 0x13u, "PSM T8");
            t.Equals(static_cast<uint32_t>(tex0.tw), 10u, "TW");
            t.Equals(static_cast<uint32_t>(tex0.th), 10u, "TH");
            t.Equals(tex0.cbp, 0x326bu, "CBP");
            t.Equals(static_cast<uint32_t>(tex0.cld), 1u, "CLD");
            const GSPrimReg prim = ps2x::gs::decodePrim(0x27d);
            t.IsTrue(prim.type == GS_PRIM_TRIFAN && prim.iip && prim.tme && prim.fge && prim.abe && prim.ctxt, "a fan in context 2");
            t.IsTrue(ps2x::gs::linearFilter(0x61) && !ps2x::gs::linearFilter(0x0), "MMAG / MMIN");
        });
        tc.Run("drawGs batches wait for their group's marker, in order", [](TestCase &t)
        {
            Renderer &renderer = Renderer::instance();
            renderer.setStreamDraws(true);
            renderer.takeGsBatches(~0ull, &take, nullptr);
            g_taken.clear();
            const GsVertex tri[3] = {vertex(1, 2, 3, 0x80402010u), vertex(4, 5, 6, 0x80u), vertex(7, 8, 9, 0u)};
            GsMaterial first = material(), second = material();
            second.alpha = 0x48;
            renderer.beginGroup(5);
            renderer.drawGs(first, tri, 3);
            renderer.beginGroup(6);
            renderer.drawGs(second, tri, 3);
            t.IsTrue(renderer.hasGroup(5) && renderer.hasGroup(6) && !renderer.hasGroup(7), "groups with batches");
            renderer.takeGsBatches(5, &take, nullptr);
            t.Equals(g_taken.size(), size_t{1}, "only group 5 at its marker");
            renderer.takeGsBatches(6, &take, nullptr);
            t.Equals(g_taken.size(), size_t{2}, "then group 6");
            t.Equals(g_taken.size() == 2 ? g_taken[1].material.alpha : 0ull, 0x48ull, "its own material");
            t.Equals(g_taken.empty() ? 0u : g_taken[0].triangles[0].z, 3u, "exact Z");
            renderer.setStreamDraws(false);
        });
        tc.Run("SubmitNative draws with the material's registers over the target's", [](TestCase &t)
        {
            RecordingBackend backend;
            GSDrawState target{};
            target.context.frame.fbp = 0x40;
            target.context.zbuf.zbp = 0x96;
            target.context.tex0 = ps2x::gs::decodeTex0(0);
            target.context.alpha = 0xa;
            target.context.xyoffset.ofx = 0x6c00;
            target.texa.ta0 = 0x80;
            GsMaterial m = material();
            m.loadClut = true;
            const GsVertex tris[6] = {vertex(1, 2, 0xFFFFFFu, 0x80402010u), vertex(4, 5, 6, 0x80u), vertex(7, 8, 9, 0u),
                                      vertex(1, 2, 3, 0u),                vertex(4, 5, 6, 0u),   vertex(7, 8, 9, 0u)};
            backend.SubmitNative(target, m, tris, 6);
            t.Equals(backend.batches.size(), size_t{2}, "a batch per triangle");
            t.Equals(backend.cluts.size(), size_t{1}, "the CLUT loaded once");
            if (backend.batches.empty())
                return;
            const GSPrimitiveBatch &b = backend.batches[0];
            t.Equals(b.state.context.frame.fbp, 0x40u, "the target's frame");
            t.Equals(b.state.context.zbuf.zbp, 0x96u, "the target's Z buffer");
            t.Equals(static_cast<uint32_t>(b.state.context.xyoffset.ofx), 0x6c00u, "the target's offset");
            t.Equals(static_cast<uint32_t>(b.state.texa.ta0), 0x80u, "the GS's TEXA");
            t.Equals(b.state.context.tex0.tbp0, 0x31ebu, "the material's TEX0");
            t.Equals(b.state.context.alpha, 0x44ull, "the material's ALPHA");
            t.Equals(b.state.context.test, 0x50000ull, "the material's TEST");
            t.Equals(static_cast<uint32_t>(b.state.textureWidth), 1024u, "texture width from TW");
            t.IsTrue(b.state.linearFilter, "TEX1's filter");
            t.IsTrue(b.state.prim.type == GS_PRIM_TRISTRIP && b.state.prim.tme && b.state.prim.abe, "the material's PRIM");
            t.Equals(b.vertices[0].z, 16777215.0, "Z");
            t.IsTrue(b.vertices[0].r == 0x10 && b.vertices[0].g == 0x20 && b.vertices[0].b == 0x40 && b.vertices[0].a == 0x80, "RGBA");
            t.IsTrue(b.vertices[0].q == 2.0f && b.vertices[0].s == 0.25f && b.vertices[0].fog == 0x40, "S, T, Q and fog");
            backend.cluts.clear();
            m.loadClut = false;
            backend.SubmitNative(target, m, tris, 3);
            t.Equals(backend.cluts.size(), size_t{0}, "no CLUT load without a TEX0 write");
        });
    });
}
