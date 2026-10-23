// Replays a GS command stream recorded with PS2X_GS_RECORD and prints the time per pass and a hash of the final
// VRAM, to time the rasterizer and check that a change draws the same pixels.
//   ps2_gs_replay <trace> [repeat] [out.rgba] [--list] [--until N] [--dump NAME=FBP,FBW,PSM,WIDTH,HEIGHT ...]
// PS2X_GS_ASYNC_RASTER=1 (and PS2X_GS_RASTER_THREADS=<n>) replays on the rasterization workers.
//
// To inspect a multi-pass effect one pass at a time:
//   --list     print every command: index, kind, size, and the first GIF tag or the upload destination
//   --until N  replay commands 0..N-1 only
//   --dump     after the replay, write a buffer as NAME.ppm: FBP in 2048-word pages, FBW in 64-pixel units, PSM as
//              in the GS registers (0 CT32, 1 CT24, 2 CT16, 10 CT16S, 19 T8, 20 T4, 48-58 Z formats, as grey)
// PS2X_GS_REPLAY_COMMANDS=<n> replays only the first n commands.
// PS2X_GS_REPLAY_CHECK=<n> replays on the workers and on the caller side by side, compares their VRAM every n commands
// and reports the first comparison that differs.
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
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

    void writeDump(GS &gs, const DumpRequest &dump)
    {
        const std::string path = dump.name + ".ppm";
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
                toRgb(dump.psm, gs.ReadVram(dump.psm, dump.fbp * 32u, std::max<uint32_t>(dump.fbw, 1u), x, y), &row[x * 3u]);
            std::fwrite(row.data(), 1, row.size(), file);
        }
        std::fclose(file);
        std::printf("wrote %s\n", path.c_str());
    }

    void listCommand(size_t index, const GsTraceCommand &command)
    {
        static const char *const kNames[] = {"gif", "native-gif", "upload"};
        std::printf("%6zu %-10s %8zu", index, command.kind < 3u ? kNames[command.kind] : "?", command.bytes.size());
        if (command.kind != GsTraceCommand::NativeImageUpload && command.bytes.size() >= 16u)
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
        }
    }

    int checkWorkersAgainstCaller(const GsTraceHeader &header, const std::vector<GsTraceCommand> &commands, size_t interval)
    {
        GSCpuBackend::SetAsyncRasterization(true);
        std::vector<uint8_t> asyncVram(header.vram), serialVram(header.vram);
        GS async, serial;
        async.init(asyncVram.data(), static_cast<uint32_t>(asyncVram.size()), nullptr);
        serial.init(serialVram.data(), static_cast<uint32_t>(serialVram.size()), nullptr);
        if (auto *cpu = dynamic_cast<GSCpuBackend *>(serial.rasterBackend()))
            cpu->DisallowAsyncRasterization();
        if (!async.loadDrawingState(header.state) || !serial.loadDrawingState(header.state))
        {
            std::fprintf(stderr, "bad drawing state\n");
            return 1;
        }
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
                             "[--dump NAME=FBP,FBW,PSM,WIDTH,HEIGHT ...]\n", argv[0]);
        return 2;
    }
    bool list = false;
    size_t until = SIZE_MAX;
    std::vector<DumpRequest> dumps;
    std::vector<const char *> positional;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--list")
            list = true;
        else if (arg == "--until" && i + 1 < argc)
            until = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
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
        return checkWorkersAgainstCaller(header, commands, std::max<size_t>(1u, std::strtoull(std::getenv("PS2X_GS_REPLAY_CHECK"), nullptr, 0)));
    const size_t replayed = std::min(until, commands.size());
    const int repeat = !positional.empty() ? std::max(1, std::atoi(positional[0])) : 1;
    if (std::getenv("PS2X_GS_ASYNC_RASTER"))
        GSCpuBackend::SetAsyncRasterization(true);

    std::vector<uint8_t> vram(header.vram.size());
    std::vector<uint8_t> final;
    double best = 0.0, total = 0.0;
    for (int pass = 0; pass < repeat; ++pass)
    {
        std::memcpy(vram.data(), header.vram.data(), vram.size());
        GS gs;
        gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);
        if (!gs.loadDrawingState(header.state))
        {
            std::fprintf(stderr, "bad drawing state\n");
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < replayed; ++i)
            replayCommand(gs, commands[i]);
        gs.rasterBackend()->SnapshotVram(final); // waits for the workers
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        total += seconds;
        best = pass == 0 ? seconds : std::min(best, seconds);
        if (pass + 1 == repeat)
            for (const DumpRequest &dump : dumps)
                writeDump(gs, dump);
    }

    uint64_t hash = 0xCBF29CE484222325ull;
    for (const uint8_t byte : final)
        hash = (hash ^ byte) * 0x100000001B3ull;
    std::printf("commands=%zu (replayed %zu) bytes=%llu vram-hash=%016llx\n", commands.size(), replayed,
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(hash));
    std::printf("time per pass: best %.1f ms, mean %.1f ms\n", best * 1000.0, total * 1000.0 / repeat);
    if (positional.size() > 1)
        if (FILE *out = std::fopen(positional[1], "wb"))
        {
            std::fwrite(final.data(), 1, final.size(), out);
            std::fclose(out);
        }
    return 0;
}
