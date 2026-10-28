#ifndef PS2X_NATIVE_RENDERER_H
#define PS2X_NATIVE_RENDERER_H

// Native PC rendering, shared by the ports: a port's adapter reads its game's own 3D data (meshes, matrices,
// materials) and submits it here instead of the PS2 VU1 / GS path; the runtime's present loop draws it with the PC
// GPU at the window's resolution, over the GS frame while a game is only partly native.
//
// Game thread: beginFrame(), draw(...) per mesh, endFrame() publishes the frame. Present thread: present() draws
// the latest published frame. Textures come from the game's own GS uploads (textures(), a TextureCache, fed by the
// runtime's GS front end), decoded when the frame is drawn, by which time the frame's own uploads have arrived.

#include "runtime/native/native_gs_hook.h"
#include "runtime/native/native_texture_cache.h"

#include <atomic>
#include <cstdint>
#include <deque>
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
        float fog = 1.0f;          // the GS's F / 255: 1 keeps the colour, 0 is all fog colour (Material::fog)
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
        bool fog = false;           // PRIM.FGE: the colour goes toward fogColour by 1 - Vertex::fog, after texturing
        uint32_t fogColour = 0;     // FOGCOL, R in the low byte
    };

    // PS2 screen space as the game draws to it: the screen point at the window's top-left corner and the size
    // that fills the window (the GS XYOFFSET and the frame size; a 640 x 448 / 512 or 640 x 256 picture).
    struct Screen
    {
        float originX = 2048.0f - 320.0f, originY = 2048.0f - 256.0f;
        float width = 640.0f, height = 512.0f;
        float depthMax = 16777215.0f; // the Z buffer's range (24-bit)
        float gsZScale = 1.0f;        // GS Z per unit of the matrices' z / w (a microprogram writing FTOI4 z: 16)
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
        // Drawing inside the GS stream (runtime/native/native_gs_hook.h). The native port numbers the VU1 programs it
        // draws instead of VU1, in stream order (1, 2, ...: the VIF1 interpreter numbers the programs it skips the
        // same way), and draws a program's batches after beginGroup(its number). With a GS renderer that keeps GL
        // render targets, each group is drawn into its target at its point of the GS stream, so it sorts with the GS
        // draws around it, and present() leaves the frame to the GS; otherwise present() draws as before.
        void setStreamDraws(bool on);
        void beginGroup(uint64_t group) { m_group = group; }
        // A group has batches waiting (its GS marker is worth queueing): the VIF1 interpreter reaches a program after
        // the native port has drawn it (at the transfer's start), so a group without batches by then has none.
        bool hasGroup(uint64_t group);

        // Present thread: draws the latest frame into the given window rectangle; false when there is none yet.
        bool present(float x, float y, float width, float height);
        // The GS frame shown with the native one (a GL texture): PS2X_NATIVE_CAPTURE_DIR saves it with each capture.
        void setGsFrame(unsigned int textureId, int width, int height, int format)
        {
            m_gsFrame[0] = static_cast<int>(textureId);
            m_gsFrame[1] = width;
            m_gsFrame[2] = height;
            m_gsFrame[3] = format;
        }
        bool enabled() const { return m_enabled; }
        // The latest published frame has something to draw (not a menu or a movie the native port does not draw yet),
        // and present() draws it (the GS frame does not already have it).
        bool hasDraws();
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
            // Taken from the texture cache when the batch is drawn (the game's thread, in stream order): games reuse
            // texture memory before the frame is presented.
            std::shared_ptr<const TextureCache::Texture> texture;
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
        struct GsPass; // GS thread only (native_renderer_gs.cpp)
        struct StreamBatch
        {
            uint64_t group = 0;
            float gsZScale = 1.0f;
            Batch batch;
            std::vector<Vertex> vertices;
        };
        static void gsDraw(uint64_t group, const GsDrawTarget &target);
        void drawGroups(uint64_t group, const GsDrawTarget &target);

        Frame m_building;
        std::mutex m_mutex;
        std::shared_ptr<Frame> m_published;
        int m_gsFrame[4] = {}; // id, width, height, pixel format
        std::unique_ptr<Gpu> m_gpu; // present thread only
        bool m_enabled = false;
        bool m_streamDraws = false;
        uint64_t m_group = 0;
        std::mutex m_streamMutex;
        std::deque<StreamBatch> m_stream; // batches waiting for their group's point of the GS stream
        GsPass *m_gsPass = nullptr;       // its GL objects live in the GS thread's context: kept to the end
        std::atomic<bool> m_drewIntoGs{false};
    };
}

#endif
