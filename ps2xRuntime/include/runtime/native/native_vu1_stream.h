#ifndef PS2X_NATIVE_VU1_STREAM_H
#define PS2X_NATIVE_VU1_STREAM_H

// A whole native frame for a port that has not translated its microprograms yet: every VU1 program the game runs is
// run on the adapter's own VU1 (Vu1Sandbox) and drawn natively, the real VU1 skipping it. The adapter reads every VIF1
// stream as the game sends it (setVif1Observer), keeps its copy of VU1 memory the way the stream fills it (UNPACK as
// VIF1 does it, MPG, the double buffer, the VIF registers a game set before the streams it sees), and decides for each
// MSCAL, in stream order, what the VU1 gate then tells VIF1 (the two agree program for program: the GS stream's groups
// count the skipped ones on both sides). DIRECT packets that only draw can be drawn natively too.
//
// A port gives it the screen its microprograms draw to, when the native path is on (in game, not in its menus), and
// calls frameEnd() where its frame ends. A program the port translates itself goes to its draw hook instead: the
// adapter keeps the memory, the group and the decision, the hook draws from the adapter's copy of VU1 memory.
//
// PS2X_NATIVE_STREAM_LOG=1: per 250 frames, how many programs and DIRECTs were drawn natively, by start address.

#include "runtime/native/native_renderer.h"
#include "runtime/ps2_memory.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ps2x::native
{
    class Vu1Sandbox;

    class Vu1StreamAdapter
    {
    public:
        // Where a program runs, decided at its MSCAL; its MSCNTs follow it.
        enum class Choice
        {
            Vu1,     // left to VU1 (and the GS)
            Sandbox, // the adapter's VU1, drawn natively
            Port,    // the port draws it itself (Config::draw), from the adapter's copy of VU1 memory
        };

        struct Config
        {
            Screen screen;
            // Whether the frame being sent is drawn natively at all (null: always); asked at each MSCAL and DIRECT.
            // Off, everything is left to VU1 and the GS (a port's menus).
            std::function<bool()> active;
            // Where a program runs (null: the sandbox). Called at its MSCAL while active.
            std::function<Choice(Vu1StreamAdapter &adapter, uint32_t startPc)> choose;
            // A Port program: at its MSCAL (resumed = false) and at each MSCNT continuing it (resumed = true), after
            // its group began.
            std::function<void(Vu1StreamAdapter &adapter, uint32_t startPc, bool resumed)> draw;
            // A DIRECT packet the port draws itself (true: drawn, the GS does not get it); asked before the adapter's
            // own rule (directs) while active, with the packet's quadwords.
            std::function<bool(Vu1StreamAdapter &adapter, const uint32_t *packet, size_t qwords)> drawDirect;
            bool directs = true; // DIRECT packets that only draw (no image transfer) drawn natively
            // The sandbox's and the DIRECTs' draws go to the GS front end with their packets' register writes
            // (GifToNative::setGsWrites): for the native GL GS renderer, whose textures and render targets they then
            // use (a game drawing into its own targets, or texturing from them).
            bool gsWrites = false;
        };

        static Vu1StreamAdapter &instance();

        // Turns the native path on: the renderer, the VIF1 observer and the VU1 / DIRECT gates.
        void install(Config config);
        bool installed() const { return m_installed; }
        // The end of the game's frame: publishes what was drawn (nothing in a frame without the native path, so the
        // last native one does not stay over a menu) and starts the next frame.
        void frameEnd();

        // For the port's hooks: VU1 memory as the stream filled it, the running program's TOP and ITOP.
        const uint8_t *code() const { return m_code; }
        const uint32_t *data() const { return m_data; }
        uint32_t top() const { return m_top; }
        uint32_t itop() const { return m_itops; }

        // VIF1's stream, from the observer.
        void observe(const uint8_t *stream, size_t bytes);

    private:
        Vu1StreamAdapter();
        ~Vu1StreamAdapter();
        void run(const uint32_t *words, size_t count);
        void program(uint32_t cmd, uint32_t pc);
        void direct(const uint32_t *words, size_t qwords, size_t size);
        void publish();

        Config m_config;
        bool m_installed = false;
        Vu1Sandbox *m_sandbox = nullptr;

        uint8_t m_code[16 * 1024] = {};
        uint32_t m_data[1024 * 4] = {};
        uint32_t m_cl = 1, m_wl = 1, m_base = 0, m_offset = 0, m_tops = 0, m_top = 0, m_itops = 0;
        VIFRegisters m_vif{}; // STMOD, STMASK, STROW / STCOL (and STCYCL, TOPS) for vifUnpack
        bool m_sawMask = false, m_sawMode = false, m_sawRow = false, m_sawCol = false;

        enum class Running
        {
            None,       // on VU1 (or nothing yet): its MSCNTs too
            Sandbox,    // the adapter's VU1
            Port,       // the port's draw hook
        } m_running = Running::None;
        uint32_t m_runningPc = 0;
        uint64_t m_group = 0;
        // A GIF IMAGE a DIRECT left unfinished: its quadwords continue in the next DIRECTs (or right after, without a
        // VIFcode), as the VIF1 interpreter takes them, so neither is read as GIF tags nor asked about.
        uint32_t m_pendingImageQwc = 0;
        bool m_pendingImageRaw = false;
    };
}

#endif
