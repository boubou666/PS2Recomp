#ifndef PS2X_NATIVE_RENDERER_H
#define PS2X_NATIVE_RENDERER_H

// Native PC rendering, shared by the ports: a port's adapter reads its game's own 3D data (meshes, matrices,
// materials) and submits it here instead of the PS2 VU1 / GS path; the runtime's present loop draws it with the PC
// GPU at the window's resolution, over the GS frame while a game is only partly native.
//
// Game thread: beginFrame(), draw(...) per mesh, endFrame() publishes the frame. Present thread: present() draws
// the latest published frame. Textures come from the game's own GS uploads (textures(), a TextureCache, fed by the
// runtime's GS front end), decoded when the frame is drawn, by which time the frame's own uploads have arrived.

#include "runtime/native/native_texture_cache.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ps2x::native
{
    struct Vertex
    {
        float x = 0, y = 0, z = 0; // the object's space: the draw's matrix takes it to PS2 screen space
        float u = 0, v = 0;        // texture coordinates, 0..1 across the texture (the GS's S / Q, T / Q)
        uint32_t rgba = 0;         // lit colour, R in the low byte; 0x80 per channel is 1.0, as the GS modulates
    };

    // How a draw blends and tests, from the GS registers a game sets (ALPHA, TEST) reduced to what a PC GPU does.
    struct Material
    {
        uint64_t tex0 = 0; // the texture (TextureCache::texture); 0 with textured = false
        uint64_t texa = 0;
        bool textured = true;
        enum class Blend : uint8_t
        {
            Opaque,   // no blending
            Alpha,    // (Cs - Cd) * As + Cd
            Additive, // Cs * As + Cd
        } blend = Blend::Alpha;
        uint8_t alphaRef = 0;   // texels at or below are discarded when alphaTest
        bool alphaTest = false;
        bool depthTest = true;  // GS depth: larger is nearer (GEQUAL)
        bool depthWrite = true;
    };

    // PS2 screen space as the game draws to it: the screen point at the window's top-left corner and the size
    // that fills the window (the GS XYOFFSET and the frame size; a 640 x 448 / 512 or 640 x 256 picture).
    struct Screen
    {
        float originX = 2048.0f - 320.0f, originY = 2048.0f - 256.0f;
        float width = 640.0f, height = 512.0f;
        float depthMax = 16777215.0f; // the Z buffer's range (24-bit)
    };

    class Renderer
    {
    public:
        static Renderer &instance();

        // Game thread.
        void beginFrame(const Screen &screen);
        // Triangles (three vertices each) with an object-to-screen matrix: column c is matrix[4c..4c+3], screen
        // point = M * (x, y, z, 1), then divided by w (as a VU's projection).
        void draw(const Material &material, const float matrix[16], const Vertex *vertices, uint32_t count);
        void endFrame();
        TextureCache &textures() { return ps2x::native::textures(); }

        // Present thread: draws the latest frame into the given window rectangle; false when there is none yet.
        bool present(float x, float y, float width, float height);
        bool enabled() const { return m_enabled; }
        void setEnabled(bool on)
        {
            m_enabled = on;
            if (on)
                setUploadsWanted(true);
        }

        struct Batch
        {
            Material material;
            float matrix[16];
            uint32_t first = 0, count = 0;
        };
        struct Frame
        {
            Screen screen;
            std::vector<Vertex> vertices;
            std::vector<Batch> batches;
        };

    private:
        Renderer();
        ~Renderer();
        struct Gpu;

        Frame m_building;
        std::mutex m_mutex;
        std::shared_ptr<Frame> m_published;
        std::unique_ptr<Gpu> m_gpu; // present thread only
        bool m_enabled = false;
    };
}

#endif
