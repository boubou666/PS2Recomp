#ifndef PS2X_NATIVE_GIF_H
#define PS2X_NATIVE_GIF_H

// GIF packets drawn by the native renderer: what a microprogram sends the GS (its XGKICK output, from a
// ps2x::native::Vu1Sandbox) read the way the GS reads it, its primitives submitted to ps2x::native::Renderer in screen
// space (the GS's own coordinates: x, y the primitive coordinates over 16, z the GS Z over the frame's gsZScale). For
// the microprograms an adapter does not translate into its game's own data, so that the whole frame is native.
//
// It keeps the GS registers drawing depends on (PRIM, RGBAQ, ST, UV, FOG, TEX0, TEXA, ALPHA, TEST, ZBUF, FOGCOL,
// PRMODE): registers(...) gives it the ones set elsewhere (a DIRECT packet's A+D writes) without drawing anything.
// Points are drawn one pixel wide, lines as one pixel wide quads; textures with perspective (ST / Q) are interpolated
// in screen space.

#include "runtime/native/native_renderer.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ps2x::native
{
    class GifToNative
    {
    public:
        // A GIF packet (PACKED, REGLIST and IMAGE tags, up to its EOP or its end): its primitives are drawn.
        void packet(const uint8_t *data, size_t bytes);
        // A packet whose primitives the GS draws itself (PATH2 / PATH3): only its register writes are kept.
        void registers(const uint8_t *data, size_t bytes);
        // Submits the batch being built (a draw call per material change otherwise).
        void flush();
        // What the next packets come from (a microprogram's address, for PS2X_NATIVE_LOG_GIF=2's statistics).
        void setSource(uint32_t source) { m_source = source; }
        // Draws for a GS renderer that keeps its own textures and targets (the native GL one): primitives become
        // fromGsContext batches, drawn by the GS front end with its state at their group's marker, each carrying the
        // register writes its packet made since the batch before (Material::gsWrites). Textures, CLUTs, render
        // targets and blending are then the GS renderer's, as for the GS's own draws.
        void setGsWrites(bool on) { m_gsWrites = on; }
        // Triangles kept here instead of drawn (with each batch's material, one per triangle), for a port checking its
        // own translation of a microprogram against the program run here (null: drawn, the default).
        void setCapture(std::vector<Vertex> *vertices, std::vector<Material> *materials)
        {
            m_captureVertices = vertices;
            m_captureMaterials = materials;
        }

    private:
        struct Kick
        {
            float x = 0, y = 0, z = 0, u = 0, v = 0, fog = 1.0f, q = 1.0f;
            uint32_t rgba = 0;
        };
        void parse(const uint8_t *data, size_t bytes, bool draw);
        void write(uint32_t reg, uint64_t value, bool draw);
        void vertex(uint64_t xyz, bool withFog, bool kick, bool draw);
        void emit(const Kick *kicks, int count);
        Material material() const;

        uint64_t m_prim = 0, m_prmode = 0, m_prmodecont = 1;
        uint64_t m_rgbaq = 0x3f80000000000000ull; // Q = 1
        float m_s = 0, m_t = 0;
        uint64_t m_uv = 0, m_fog = 0;
        uint64_t m_tex0[2] = {}, m_alpha[2] = {}, m_test[2] = {}, m_zbuf[2] = {}, m_clamp[2] = {};
        uint64_t m_texa = 0, m_fogcol = 0;
        Kick m_queue[3];
        int m_queued = 0, m_strip = 0; // vertices waiting, and kicks since the primitive started (strips, fans)

        Material m_batchMaterial;
        bool m_batchOpen = false;
        std::vector<Vertex> m_triangles;
        uint32_t m_source = 0;
        bool m_gsWrites = false;
        std::vector<Vertex> *m_captureVertices = nullptr;
        std::vector<Material> *m_captureMaterials = nullptr;
        std::vector<std::pair<uint8_t, uint64_t>> m_pendingWrites; // setGsWrites: the packets' writes not yet drawn with
    };
}

#endif
