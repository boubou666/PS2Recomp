// Replays a GS command stream recorded with PS2X_GS_RECORD (GSR2, or the older GSR1) and prints the time per pass
// and a hash of the final VRAM, to time a renderer and check that a change draws the same pixels.
//   ps2_gs_replay <trace> [repeat] [out.rgba] [--list] [--until N] [--dump NAME=FBP,FBW,PSM,WIDTH,HEIGHT ...]
//                 [--convert OUT] [--renderer cpu|parallel] [--compare cpu|parallel] [--every N]
// PS2X_GS_ASYNC_RASTER=1 (and PS2X_GS_RASTER_THREADS=<n>) replays the CPU renderer on the rasterization workers.
//
//   --renderer  who replays (default cpu): any GSStreamRenderer the build has
//   --compare   replay on --renderer and on this one side by side and compare their VRAM byte for byte every
//               --every N commands (default 64); on a difference, replay again comparing after every command of
//               the failing interval, report the first command that differs, where, and with --dump write the
//               buffers of both renderers as NAME.<renderer>.ppm
//
// To inspect a multi-pass effect one pass at a time:
//   --list     print every command: index, kind, size, and the first GIF tag or the upload destination
//   --until N  replay commands 0..N-1 only
//   --dump     after the replay, write a buffer as NAME.ppm: FBP in 2048-word pages, FBW in 64-pixel units, PSM as
//              in the GS registers (0 CT32, 1 CT24, 2 CT16, 10 CT16S, 19 T8, 20 T4, 48-58 Z formats, as grey)
//   --convert  write the trace as GSR2 to OUT (a GSR1 trace's drawing state goes through the CPU path) and exit
// PS2X_GS_REPLAY_COMMANDS=<n> replays only the first n commands.
// PS2X_GS_REPLAY_CHECK=<n> replays on the workers and on the caller side by side, compares their VRAM every n commands
// and reports the first comparison that differs.
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_registers.h"
#include "runtime/gs/gs_stream_renderer.h"
#include "runtime/gs/gs_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ps2_runtime expects the generated runner to provide its function table; this tool runs no guest code.
struct R5900Context;
class PS2Runtime;
extern const uint32_t g_ps2RecompiledFunctionTableBase = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = 0u;
void (*g_ps2RecompiledFunctionTable[1])(uint8_t *, R5900Context *, PS2Runtime *) = {nullptr};

namespace
{
    struct DumpRequest
    {
        std::string name;
        uint32_t fbp = 0;
        uint32_t fbw = 0;
        uint32_t psm = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    bool parseDump(const char *text, DumpRequest &out)
    {
        const char *eq = std::strchr(text, '=');
        if (!eq)
            return false;
        out.name.assign(text, eq);
        unsigned fbp = 0, fbw = 0, width = 0, height = 0;
        int psm = 0;
        if (std::sscanf(eq + 1, "%u,%u,%i,%u,%u", &fbp, &fbw, &psm, &width, &height) != 5)
            return false;
        out.fbp = fbp;
        out.fbw = fbw;
        out.psm = static_cast<uint32_t>(psm);
        out.width = width;
        out.height = height;
        return !out.name.empty() && width != 0u && height != 0u;
    }

    void toRgb(uint32_t psm, uint32_t value, uint8_t rgb[3])
    {
        switch (psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_CT24:
            rgb[0] = static_cast<uint8_t>(value);
            rgb[1] = static_cast<uint8_t>(value >> 8);
            rgb[2] = static_cast<uint8_t>(value >> 16);
            return;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            rgb[0] = static_cast<uint8_t>((value & 0x1Fu) << 3);
            rgb[1] = static_cast<uint8_t>(((value >> 5) & 0x1Fu) << 3);
            rgb[2] = static_cast<uint8_t>(((value >> 10) & 0x1Fu) << 3);
            return;
        case GS_PSM_Z32:
        case GS_PSM_Z24:
            rgb[0] = rgb[1] = rgb[2] = static_cast<uint8_t>(value >> 16);
            return;
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            rgb[0] = rgb[1] = rgb[2] = static_cast<uint8_t>(value >> 8);
            return;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            rgb[0] = rgb[1] = rgb[2] = static_cast<uint8_t>(value << 4);
            return;
        default: // 8-bit indices
            rgb[0] = rgb[1] = rgb[2] = static_cast<uint8_t>(value);
            return;
        }
    }

    void writeDump(std::vector<uint8_t> &vram, const DumpRequest &dump, const std::string &suffix = {})
    {
        const std::string path = dump.name + suffix + ".ppm";
        FILE *file = std::fopen(path.c_str(), "wb");
        if (!file)
        {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            return;
        }
        std::fprintf(file, "P6\n%u %u\n255\n", dump.width, dump.height);
        std::vector<uint8_t> row(static_cast<size_t>(dump.width) * 3u);
        for (uint32_t y = 0; y < dump.height; ++y)
        {
            for (uint32_t x = 0; x < dump.width; ++x)
                toRgb(dump.psm,
                      GSCpuBackend::ReadVramPixel(vram.data(), dump.psm, dump.fbp * 32u, std::max<uint32_t>(dump.fbw, 1u), x, y),
                      &row[x * 3u]);
            std::fwrite(row.data(), 1, row.size(), file);
        }
        std::fclose(file);
        std::printf("wrote %s\n", path.c_str());
    }

    void listCommand(size_t index, const GsTraceCommand &command)
    {
        static const char *const kNames[] = {"gif", "native-gif", "upload", "latch", "register"};
        std::printf("%6zu %-10s %8zu", index, command.kind < 5u ? kNames[command.kind] : "?", command.bytes.size());
        if ((command.kind == GsTraceCommand::GifPacket || command.kind == GsTraceCommand::NativePackedGifPacket) &&
            command.bytes.size() >= 16u)
        {
            uint64_t lo = 0;
            uint64_t hi = 0;
            std::memcpy(&lo, command.bytes.data(), sizeof(lo));
            std::memcpy(&hi, command.bytes.data() + 8, sizeof(hi));
            std::printf("  tag nloop=%u eop=%u flg=%u nreg=%u regs=%016llx", static_cast<unsigned>(lo & 0x7FFFu),
                        static_cast<unsigned>((lo >> 15) & 1u), static_cast<unsigned>((lo >> 58) & 3u),
                        static_cast<unsigned>((lo >> 60) & 0xFu), static_cast<unsigned long long>(hi));
        }
        else if (command.kind == GsTraceCommand::NativeImageUpload)
        {
            const uint64_t bitbltbuf = command.registers[0];
            std::printf("  dbp=%u dbw=%u dpsm=0x%x", static_cast<unsigned>((bitbltbuf >> 32) & 0x3FFFu),
                        static_cast<unsigned>((bitbltbuf >> 48) & 0x3Fu), static_cast<unsigned>((bitbltbuf >> 56) & 0x3Fu));
        }
        else if (command.kind == GsTraceCommand::Register)
            std::printf("  reg=0x%02x value=%016llx", static_cast<unsigned>(command.registers[0] & 0xFFu),
                        static_cast<unsigned long long>(command.registers[1]));
        else if (command.kind == GsTraceCommand::PresentationLatch &&
                 command.bytes.size() >= GsTraceCommand::kPresentationLatchQwords * 8u)
        {
            uint64_t display[GsTraceCommand::kPresentationLatchQwords];
            std::memcpy(display, command.bytes.data(), sizeof(display));
            std::printf("  dispfb1=%016llx dispfb2=%016llx", static_cast<unsigned long long>(display[3]),
                        static_cast<unsigned long long>(display[5]));
        }
        std::printf("\n");
    }

    void replayCommand(GS &gs, const GsTraceCommand &command)
    {
        const uint8_t *data = command.bytes.data();
        const uint32_t size = static_cast<uint32_t>(command.bytes.size());
        switch (command.kind)
        {
        case GsTraceCommand::GifPacket:
            gs.processGIFPacket(data, size);
            break;
        case GsTraceCommand::NativePackedGifPacket:
            gs.processNativePackedGIFPacket(data, size);
            break;
        case GsTraceCommand::NativeImageUpload:
            gs.uploadImageNative(command.registers[0], command.registers[1], command.registers[2], command.registers[3],
                                 data, size);
            break;
        case GsTraceCommand::Register:
            gs.writeRegister(static_cast<uint8_t>(command.registers[0]), command.registers[1]);
            break;
        }
    }

    void replayCommand(GSStreamRenderer &renderer, const GsTraceCommand &command)
    {
        const uint8_t *data = command.bytes.data();
        const uint32_t size = static_cast<uint32_t>(command.bytes.size());
        switch (command.kind)
        {
        case GsTraceCommand::GifPacket:
            renderer.gifTransfer(data, size);
            break;
        case GsTraceCommand::NativePackedGifPacket:
            renderer.nativePackedGifTransfer(data, size);
            break;
        case GsTraceCommand::NativeImageUpload:
            renderer.uploadImage(command.registers[0], command.registers[1], command.registers[2], command.registers[3],
                                 data, size);
            break;
        case GsTraceCommand::Register:
            renderer.writeRegister(static_cast<uint8_t>(command.registers[0]), command.registers[1]);
            break;
        }
    }

    std::unique_ptr<GSStreamRenderer> startRenderer(GSStreamRendererKind kind, const GsTraceHeader &header,
                                                    const GSStateImage &image)
    {
        std::unique_ptr<GSStreamRenderer> renderer =
            GSStreamRenderer::create(kind, static_cast<uint32_t>(header.vram.size()), {});
        if (!renderer)
        {
            std::fprintf(stderr, "renderer %s unavailable\n", GSStreamRenderer::name(kind));
            return nullptr;
        }
        renderer->writeVram(header.vram.data(), static_cast<uint32_t>(header.vram.size()));
        renderer->loadState(image);
        return renderer;
    }

    void readVram(GSStreamRenderer &renderer, std::vector<uint8_t> &out)
    {
        renderer.flush();
        renderer.readVram(out);
    }

    uint64_t vramHash(const std::vector<uint8_t> &vram)
    {
        uint64_t hash = 0xCBF29CE484222325ull;
        for (const uint8_t byte : vram)
            hash = (hash ^ byte) * 0x100000001B3ull;
        return hash;
    }

    // Where a VRAM page falls in a buffer: its first pixel and size, if it is inside the buffer's rows.
    bool pageInBuffer(uint32_t page, uint32_t basePage, uint32_t widthUnits, uint32_t psm, uint32_t &x, uint32_t &y,
                      uint32_t &pageWidth, uint32_t &pageHeight)
    {
        switch (psm)
        {
        case GS_PSM_CT32: case GS_PSM_CT24: case GS_PSM_Z32: case GS_PSM_Z24:
            pageWidth = 64u; pageHeight = 32u; break;
        case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S:
            pageWidth = 64u; pageHeight = 64u; break;
        case GS_PSM_T8:
            pageWidth = 128u; pageHeight = 64u; break;
        case GS_PSM_T4:
            pageWidth = 128u; pageHeight = 128u; break;
        default: // T8H, T4HL, T4HH share CT32's layout
            pageWidth = 64u; pageHeight = 32u; break;
        }
        const uint32_t pagesPerRow = widthUnits * 64u / pageWidth;
        if (page < basePage || pagesPerRow == 0u)
            return false;
        x = (page - basePage) % pagesPerRow * pageWidth;
        y = (page - basePage) / pagesPerRow * pageHeight;
        return y < 2048u;
    }

    void describeDifference(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, const GSStateImage &state)
    {
        size_t words = 0, first = 0;
        std::vector<uint32_t> pages;
        for (size_t offset = 0; offset + 4 <= std::min(a.size(), b.size()); offset += 4)
        {
            if (std::memcmp(a.data() + offset, b.data() + offset, 4) == 0)
                continue;
            if (words++ == 0)
                first = offset;
            const uint32_t page = static_cast<uint32_t>(offset / 8192u);
            if (pages.empty() || pages.back() != page)
                pages.push_back(page);
        }
        uint32_t valueA = 0, valueB = 0;
        std::memcpy(&valueA, a.data() + first, 4);
        std::memcpy(&valueB, b.data() + first, 4);
        std::printf("  %zu words differ in %zu pages; first at byte 0x%zx (page %zu, block %zu): %08x vs %08x\n", words,
                    pages.size(), first, first / 8192u, first % 8192u / 256u, valueA, valueB);
        std::printf("  pages:");
        for (size_t i = 0; i < pages.size() && i < 16u; ++i)
            std::printf(" %u", pages[i]);
        std::printf(pages.size() > 16u ? " ...\n" : "\n");

        struct Buffer
        {
            const char *name;
            uint32_t basePage, widthUnits, psm;
        };
        std::vector<Buffer> buffers;
        for (uint8_t i = 0; i < 2u; ++i)
        {
            const GSFrameReg frame = GSReg::decodeFrame(state.reg(GS_REG_FRAME_1 + i));
            const GSZbufReg zbuf = GSReg::decodeZbuf(state.reg(GS_REG_ZBUF_1 + i));
            const GSTex0Reg tex0 = GSReg::decodeTex0(state.reg(GS_REG_TEX0_1 + i));
            buffers.push_back({i ? "FRAME_2" : "FRAME_1", frame.fbp, frame.fbw, frame.psm});
            buffers.push_back({i ? "ZBUF_2" : "ZBUF_1", zbuf.zbp, frame.fbw, zbuf.psm});
            buffers.push_back({i ? "TEX0_2" : "TEX0_1", tex0.tbp0 / 32u, tex0.tbw, tex0.psm});
        }
        const uint32_t page = static_cast<uint32_t>(first / 8192u);
        for (const Buffer &buffer : buffers)
        {
            uint32_t x = 0, y = 0, width = 0, height = 0;
            if (pageInBuffer(page, buffer.basePage, buffer.widthUnits, buffer.psm, x, y, width, height))
                std::printf("  in %s (page %u, width %u, psm 0x%x): pixels x %u-%u, y %u-%u\n", buffer.name,
                            buffer.basePage, buffer.widthUnits, buffer.psm, x, x + width - 1u, y, y + height - 1u);
        }
    }

    // Replays both renderers in step; compares after every command index in `checks` (ascending). Returns the
    // first index whose comparison differs, or SIZE_MAX.
    size_t replayCompared(GSStreamRendererKind kindA, GSStreamRendererKind kindB, const GsTraceHeader &header,
                          const GSStateImage &image, const std::vector<GsTraceCommand> &commands, size_t count,
                          size_t every, size_t from, std::vector<uint8_t> &vramA, std::vector<uint8_t> &vramB,
                          size_t &comparisons, double &secondsA, double &secondsB)
    {
        std::unique_ptr<GSStreamRenderer> a = startRenderer(kindA, header, image);
        std::unique_ptr<GSStreamRenderer> b = startRenderer(kindB, header, image);
        if (!a || !b)
            return SIZE_MAX - 1u;
        comparisons = 0;
        secondsA = secondsB = 0.0;
        for (size_t index = 0; index < count; ++index)
        {
            auto start = std::chrono::steady_clock::now();
            replayCommand(*a, commands[index]);
            auto middle = std::chrono::steady_clock::now();
            replayCommand(*b, commands[index]);
            secondsA += std::chrono::duration<double>(middle - start).count();
            secondsB += std::chrono::duration<double>(std::chrono::steady_clock::now() - middle).count();
            if (index < from || ((index + 1u) % every != 0u && index + 1u != count))
                continue;
            readVram(*a, vramA);
            readVram(*b, vramB);
            ++comparisons;
            if (vramA != vramB)
                return index;
        }
        return SIZE_MAX;
    }

    int compareRenderers(GSStreamRendererKind kindA, GSStreamRendererKind kindB, const GsTraceHeader &header,
                         const GSStateImage &image, const std::vector<GsTraceCommand> &commands, size_t count,
                         size_t every, const std::vector<DumpRequest> &dumps)
    {
        const char *nameA = GSStreamRenderer::name(kindA);
        const char *nameB = GSStreamRenderer::name(kindB);
        std::vector<uint8_t> vramA, vramB;
        size_t comparisons = 0;
        double secondsA = 0.0, secondsB = 0.0;
        size_t failed = replayCompared(kindA, kindB, header, image, commands, count, every, 0u, vramA, vramB,
                                       comparisons, secondsA, secondsB);
        if (failed == SIZE_MAX - 1u)
            return 2;
        if (failed == SIZE_MAX)
        {
            std::printf("%s and %s match over %zu commands (%zu comparisons, every %zu) vram-hash=%016llx\n", nameA,
                        nameB, count, comparisons, every, static_cast<unsigned long long>(vramHash(vramA)));
            std::printf("replay time: %s %.1f ms, %s %.1f ms\n", nameA, secondsA * 1000.0, nameB, secondsB * 1000.0);
            return 0;
        }
        if (every > 1u)
        {
            const size_t from = failed + 1u > every ? failed + 1u - every : 0u;
            failed = replayCompared(kindA, kindB, header, image, commands, failed + 1u, 1u, from, vramA, vramB,
                                    comparisons, secondsA, secondsB);
        }
        std::printf("%s and %s differ after command %zu:\n", nameA, nameB, failed);
        listCommand(failed, commands[failed]);
        GSStateImage state;
        if (std::unique_ptr<GSStreamRenderer> reference = startRenderer(GSStreamRendererKind::Cpu, header, image))
        {
            for (size_t index = 0; index <= failed; ++index)
                replayCommand(*reference, commands[index]);
            reference->saveState(state);
        }
        describeDifference(vramA, vramB, state);
        for (const DumpRequest &dump : dumps)
        {
            writeDump(vramA, dump, std::string(".") + nameA);
            writeDump(vramB, dump, std::string(".") + nameB);
        }
        return 1;
    }

    bool loadTraceState(const GsTraceHeader &header, GSStateImage &image)
    {
        if (header.version != 1u)
        {
            image = header.image;
            return true;
        }
        std::vector<uint8_t> vram(header.vram.size());
        GS gs;
        gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);
        if (!gs.loadDrawingState(header.legacyState))
            return false;
        gs.saveState(image);
        return true;
    }

    int checkWorkersAgainstCaller(const GsTraceHeader &header, const GSStateImage &image, const std::vector<GsTraceCommand> &commands, size_t interval)
    {
        GSCpuBackend::SetAsyncRasterization(true);
        std::vector<uint8_t> asyncVram(header.vram), serialVram(header.vram);
        GS async, serial;
        async.init(asyncVram.data(), static_cast<uint32_t>(asyncVram.size()), nullptr);
        serial.init(serialVram.data(), static_cast<uint32_t>(serialVram.size()), nullptr);
        if (auto *cpu = dynamic_cast<GSCpuBackend *>(serial.rasterBackend()))
            cpu->DisallowAsyncRasterization();
        async.loadState(image);
        serial.loadState(image);
        std::vector<uint8_t> asyncSnapshot, serialSnapshot;
        for (size_t index = 0; index < commands.size(); ++index)
        {
            replayCommand(async, commands[index]);
            replayCommand(serial, commands[index]);
            if ((index + 1u) % interval != 0u && index + 1u != commands.size())
                continue;
            async.rasterBackend()->SnapshotVram(asyncSnapshot);
            serial.rasterBackend()->SnapshotVram(serialSnapshot);
            if (asyncSnapshot == serialSnapshot)
                continue;
            size_t words = 0, first = 0;
            for (size_t offset = 0; offset + 4 <= serialSnapshot.size(); offset += 4)
                if (std::memcmp(asyncSnapshot.data() + offset, serialSnapshot.data() + offset, 4) != 0 && words++ == 0)
                    first = offset;
            std::printf("command %zu: %zu words differ, first at byte 0x%zx (page %zu)\n", index, words, first,
                        first / 8192u);
            return 1;
        }
        std::printf("workers match the caller over %zu commands\n", commands.size());
        return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <trace> [repeat] [vram-out] [--list] [--until N] "
                             "[--dump NAME=FBP,FBW,PSM,WIDTH,HEIGHT ...] [--convert OUT] [--renderer KIND] "
                             "[--compare KIND] [--every N]\n", argv[0]);
        return 2;
    }
    bool list = false;
    size_t until = SIZE_MAX;
    std::vector<DumpRequest> dumps;
    const char *convertTo = nullptr;
    GSStreamRendererKind rendererKind = GSStreamRendererKind::Cpu;
    GSStreamRendererKind compareKind = GSStreamRendererKind::Cpu;
    bool compare = false;
    size_t every = 64u;
    std::vector<const char *> positional;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--list")
            list = true;
        else if (arg == "--until" && i + 1 < argc)
            until = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--convert" && i + 1 < argc)
            convertTo = argv[++i];
        else if ((arg == "--renderer" || arg == "--compare") && i + 1 < argc)
        {
            GSStreamRendererKind &kind = arg == "--renderer" ? rendererKind : compareKind;
            if (!GSStreamRenderer::parse(argv[++i], kind) || !GSStreamRenderer::available(kind))
            {
                std::fprintf(stderr, "unknown or unavailable renderer %s\n", argv[i]);
                return 2;
            }
            compare = compare || arg == "--compare";
        }
        else if (arg == "--every" && i + 1 < argc)
            every = std::max<size_t>(1u, std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--dump" && i + 1 < argc)
        {
            DumpRequest dump;
            if (!parseDump(argv[++i], dump))
            {
                std::fprintf(stderr, "bad --dump %s (NAME=FBP,FBW,PSM,WIDTH,HEIGHT)\n", argv[i]);
                return 2;
            }
            dumps.push_back(dump);
        }
        else
            positional.push_back(argv[i]);
    }

    FILE *in = std::fopen(argv[1], "rb");
    GsTraceHeader header;
    if (!in || !readGsTraceHeader(in, header))
    {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::vector<GsTraceCommand> commands;
    uint64_t bytes = 0;
    for (GsTraceCommand command; readGsTraceCommand(in, command);)
    {
        bytes += command.bytes.size();
        commands.push_back(std::move(command));
    }
    std::fclose(in);
    GSStateImage image;
    if (!loadTraceState(header, image))
    {
        std::fprintf(stderr, "bad drawing state\n");
        return 1;
    }
    if (convertTo)
    {
        FILE *out = std::fopen(convertTo, "wb");
        if (!out)
        {
            std::fprintf(stderr, "cannot write %s\n", convertTo);
            return 2;
        }
        writeGsTraceHeader(out, header.vram, image);
        for (const GsTraceCommand &command : commands)
            writeGsTraceCommand(out, command.kind, command.registers, command.bytes.data(),
                                static_cast<uint32_t>(command.bytes.size()));
        const bool written = std::fclose(out) == 0;
        std::printf("wrote %s (GSR2, %zu commands)\n", convertTo, commands.size());
        return written ? 0 : 1;
    }
    if (list)
    {
        for (size_t i = 0; i < commands.size(); ++i)
            listCommand(i, commands[i]);
        if (dumps.empty() && until == SIZE_MAX)
            return 0;
    }
    if (const char *limit = std::getenv("PS2X_GS_REPLAY_COMMANDS"))
        commands.resize(std::min<size_t>(commands.size(), std::strtoull(limit, nullptr, 0)));
    if (std::getenv("PS2X_GS_REPLAY_CHECK"))
        return checkWorkersAgainstCaller(header, image, commands, std::max<size_t>(1u, std::strtoull(std::getenv("PS2X_GS_REPLAY_CHECK"), nullptr, 0)));
    const size_t replayed = std::min(until, commands.size());
    if (std::getenv("PS2X_GS_ASYNC_RASTER"))
        GSCpuBackend::SetAsyncRasterization(true);
    if (compare)
        return compareRenderers(rendererKind, compareKind, header, image, commands, replayed, every, dumps);
    const int repeat = !positional.empty() ? std::max(1, std::atoi(positional[0])) : 1;

    std::vector<uint8_t> final;
    double best = 0.0, total = 0.0;
    for (int pass = 0; pass < repeat; ++pass)
    {
        std::unique_ptr<GSStreamRenderer> renderer = startRenderer(rendererKind, header, image);
        if (!renderer)
            return 2;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < replayed; ++i)
            replayCommand(*renderer, commands[i]);
        readVram(*renderer, final); // waits for the workers or the GPU
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        total += seconds;
        best = pass == 0 ? seconds : std::min(best, seconds);
        if (pass + 1 == repeat)
            for (const DumpRequest &dump : dumps)
                writeDump(final, dump);
    }

    std::printf("commands=%zu (replayed %zu) bytes=%llu vram-hash=%016llx\n", commands.size(), replayed,
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(vramHash(final)));
    std::printf("time per pass: best %.1f ms, mean %.1f ms\n", best * 1000.0, total * 1000.0 / repeat);
    if (positional.size() > 1)
        if (FILE *out = std::fopen(positional[1], "wb"))
        {
            std::fwrite(final.data(), 1, final.size(), out);
            std::fclose(out);
        }
    return 0;
}
