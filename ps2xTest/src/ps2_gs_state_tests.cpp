#include "MiniTest.h"
#include "gs/gs_gif_decoder.h"
#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_registers.h"
#include "runtime/gs/gs_state_image.h"
#include "runtime/gs/gs_stream_renderer.h"
#include "runtime/gs/gs_trace.h"
#include "runtime/ps2_memory.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
    void setTestEnv(const char *name, const char *value)
    {
#if defined(_WIN32)
        _putenv_s(name, value ? value : "");
#else
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
#endif
    }

    struct Rng
    {
        uint64_t state;
        uint64_t next()
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        }
        uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
    };

    // Random GIF packets. Renderable ones keep primitives, textures and transfers small enough to rasterize.
    struct PacketGenerator
    {
        Rng rng;
        bool renderable;

        uint64_t value(uint8_t address)
        {
            uint64_t v = rng.next();
            const uint64_t xyMask = renderable ? 0x3FFu : 0x7FFFu;
            switch (address)
            {
            case GS_REG_XYZF2:
            case GS_REG_XYZF3:
            case GS_REG_XYZ2:
            case GS_REG_XYZ3:
                return (v & ~0xFFFFFFFFull) | (v & xyMask) | (((v >> 16) & xyMask) << 16);
            case GS_REG_XYOFFSET_1:
            case GS_REG_XYOFFSET_2:
                return renderable ? 0u : v;
            case GS_REG_TEX0_1:
            case GS_REG_TEX0_2:
                return renderable ? (v & ~(0xFFull << 26)) | ((v >> 8 & 0x7u) << 26) | ((v >> 12 & 0x7u) << 30) : v;
            case GS_REG_TRXREG:
                return renderable ? (1u + (v & 0xFu)) | ((1u + ((v >> 8) & 0xFu)) << 32) : v;
            case GS_REG_TRXPOS:
                return renderable ? v & 0x003F003F003F003Full : v;
            case GS_REG_FRAME_1:
            case GS_REG_FRAME_2:
                return renderable ? (v & ~(0x3Full << 16)) | ((1u + (v >> 40) % 10u) << 16) : v;
            default:
                return v;
            }
        }

        uint8_t adAddress()
        {
            static constexpr uint8_t kAddresses[] = {
                GS_REG_PRIM, GS_REG_RGBAQ, GS_REG_ST, GS_REG_UV, GS_REG_XYZF2, GS_REG_XYZ2, GS_REG_XYZF3, GS_REG_XYZ3,
                GS_REG_FOG, GS_REG_TEX0_1, GS_REG_TEX0_2, GS_REG_TEX2_1, GS_REG_TEX2_2, GS_REG_TEX1_1, GS_REG_CLAMP_1,
                GS_REG_XYOFFSET_1, GS_REG_PRMODECONT, GS_REG_PRMODE, GS_REG_TEXCLUT, GS_REG_SCANMSK, GS_REG_MIPTBP1_1,
                GS_REG_TEXA, GS_REG_FOGCOL, GS_REG_TEXFLUSH, GS_REG_SCISSOR_1, GS_REG_ALPHA_1, GS_REG_DIMX, GS_REG_DTHE,
                GS_REG_COLCLAMP, GS_REG_TEST_1, GS_REG_PABE, GS_REG_FBA_1, GS_REG_FRAME_1, GS_REG_FRAME_2,
                GS_REG_ZBUF_1, GS_REG_BITBLTBUF, GS_REG_TRXPOS, GS_REG_TRXREG, GS_REG_TRXDIR, GS_REG_HWREG};
            return kAddresses[rng.below(sizeof(kAddresses))];
        }

        std::vector<uint8_t> packet()
        {
            std::vector<uint64_t> q;
            const uint32_t tags = 1u + rng.below(3u);
            for (uint32_t t = 0; t < tags; ++t)
            {
                const uint32_t kind = rng.below(20u);
                const uint32_t flg = kind < 12u ? GIF_FMT_PACKED : kind < 17u ? GIF_FMT_REGLIST : GIF_FMT_IMAGE;
                const uint32_t nloop = 1u + rng.below(flg == GIF_FMT_IMAGE ? 8u : 6u);
                const uint32_t nreg = 1u + rng.below(4u);
                static constexpr uint8_t kPacked[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0A,
                                                      0x0C, 0x0D, 0x0E, 0x0E, 0x0E, 0x0E, 0x0F};
                static constexpr uint8_t kReglist[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x08, 0x0A, 0x0C, 0x0D};
                uint64_t regs = 0;
                uint8_t descriptors[4]{};
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    descriptors[r] = flg == GIF_FMT_PACKED ? kPacked[rng.below(sizeof(kPacked))]
                                                           : kReglist[rng.below(sizeof(kReglist))];
                    regs |= uint64_t(descriptors[r]) << (4u * r);
                }
                const bool pre = rng.below(10u) < 3u;
                uint64_t tagLo = nloop | (uint64_t(t + 1u == tags) << 15) | (uint64_t(flg) << 58) | (uint64_t(nreg) << 60);
                if (pre)
                    tagLo |= (1ull << 46) | ((value(GS_REG_PRIM) & 0x7FFu) << 47);
                q.push_back(tagLo);
                q.push_back(regs);
                if (flg == GIF_FMT_IMAGE)
                {
                    for (uint32_t i = 0; i < nloop * 2u; ++i)
                        q.push_back(rng.next());
                    continue;
                }
                std::vector<uint64_t> words;
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        const uint8_t d = descriptors[r];
                        if (flg == GIF_FMT_REGLIST)
                        {
                            words.push_back(value(d));
                            continue;
                        }
                        uint64_t lo = rng.next(), hi = rng.next();
                        if (d == 0x0E)
                        {
                            const uint8_t address = adAddress();
                            lo = value(address);
                            hi = address;
                        }
                        else if (d == 0x04 || d == 0x05 || d == 0x0C || d == 0x0D)
                        {
                            const uint64_t xyz = value(GS_REG_XYZ2);
                            lo = (xyz & 0xFFFFu) | (((xyz >> 16) & 0xFFFFu) << 32);
                            hi = (rng.next() & ~(1ull << 47)) | (uint64_t(rng.below(5u) == 0u) << 47);
                        }
                        else if (d == 0x06)
                            lo = value(GS_REG_TEX0_1);
                        words.push_back(lo);
                        words.push_back(hi);
                    }
                if (words.size() & 1u)
                    words.push_back(0u);
                q.insert(q.end(), words.begin(), words.end());
            }
            std::vector<uint8_t> bytes(q.size() * 8u);
            std::memcpy(bytes.data(), q.data(), bytes.size());
            return bytes;
        }
    };

    std::vector<uint8_t> serialized(const GSStateImage &image) { return serializeGsStateImage(image); }

    GSVertex vertexFromRaw(const GSStateImage::Vertex &in)
    {
        GSVertex v;
        v.x = static_cast<float>(in.xyz & 0xFFFF) / 16.0f;
        v.y = static_cast<float>((in.xyz >> 16) & 0xFFFF) / 16.0f;
        v.z = static_cast<double>(in.xyz >> 32);
        v.r = static_cast<uint8_t>(in.rgbaq);
        v.g = static_cast<uint8_t>(in.rgbaq >> 8);
        v.b = static_cast<uint8_t>(in.rgbaq >> 16);
        v.a = static_cast<uint8_t>(in.rgbaq >> 24);
        uint32_t bits = static_cast<uint32_t>(in.rgbaq >> 32);
        std::memcpy(&v.q, &bits, 4);
        if (v.q == 0.0f)
            v.q = 1.0f;
        bits = static_cast<uint32_t>(in.st);
        std::memcpy(&v.s, &bits, 4);
        bits = static_cast<uint32_t>(in.st >> 32);
        std::memcpy(&v.t, &bits, 4);
        v.u = static_cast<uint16_t>(in.uv & 0x3FFFu);
        v.v = static_cast<uint16_t>((in.uv >> 16) & 0x3FFFu);
        v.fog = in.fog;
        return v;
    }

    // What a renderer is told, as comparable words: the frontend's backend calls and the decoder's sink events
    // land in the same log.
    struct EventLog
    {
        std::vector<uint64_t> words;

        static uint64_t floatBits(float value)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, 4);
            return bits;
        }

        void primitive(const GSPrimitiveBatch &batch)
        {
            words.push_back(1u);
            words.push_back(batch.vertexCount);
            for (uint32_t i = 0; i < batch.vertexCount; ++i)
            {
                const GSVertex &v = batch.vertices[i];
                words.insert(words.end(), {uint64_t(v.x * 16.0f), uint64_t(v.y * 16.0f), uint64_t(v.z), v.r, v.g, v.b,
                                           v.a, floatBits(v.q), floatBits(v.s), floatBits(v.t), v.u, v.v, v.fog});
            }
            const GSDrawState &s = batch.state;
            const GSContext &c = s.context;
            words.insert(words.end(),
                         {GSReg::encodePrim(s.prim), GSReg::encodeFrame(c.frame), GSReg::encodeZbuf(c.zbuf),
                          GSReg::encodeTex0(c.tex0), GSReg::encodeXyoffset(c.xyoffset), GSReg::encodeScissor(c.scissor),
                          c.tex1, c.miptbp1, c.miptbp2, c.clamp, c.alpha, c.test, c.fba, GSReg::encodeTexa(s.texa),
                          GSReg::encodeTexclut(s.texclut), s.pabe, s.scanmsk, s.dimx, s.dthe, s.colclamp, s.fogR,
                          s.fogG, s.fogB});
        }

        void clutLoad(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
        {
            words.insert(words.end(), {2u, GSReg::encodeTex0(tex0), GSReg::encodeTexclut(texclut)});
        }

        void transfer(const GSTransferCommand &command)
        {
            words.insert(words.end(), {3u, GSReg::encodeBitbltbuf(command.bitbltbuf), GSReg::encodeTrxpos(command.trxpos),
                                       GSReg::encodeTrxreg(command.trxreg), command.direction});
        }

        void image(const uint8_t *data, uint32_t size)
        {
            words.push_back(4u);
            words.push_back(size);
            for (uint32_t i = 0; i < size; ++i)
                words.push_back(data[i]);
        }
    };

    class RecordingBackend final : public GSRasterBackend
    {
    public:
        explicit RecordingBackend(EventLog &log) : m_log(log) {}
        void Initialize(uint8_t *, uint32_t) override {}
        void Reset() override {}
        void Submit(const GSPrimitiveBatch &batch) override { m_log.primitive(batch); }
        void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override { m_log.clutLoad(tex0, texclut); }
        void BeginTransfer(const GSTransferCommand &command) override { m_log.transfer(command); }
        void UploadImage(const uint8_t *data, uint32_t sizeBytes) override { m_log.image(data, sizeBytes); }
        void Flush() override {}
        void TextureFlush() override { m_log.words.push_back(5u); }
        void Sync(GSSyncReason) override {}
        PresentationFrame Present(const GSPresentationRequest &) override { return {}; }
        bool ClearFramebuffer(const GSContext &, uint32_t) override { return false; }
        uint32_t ConsumeLocalToHostBytes(uint8_t *, uint32_t) override { return 0u; }
        uint32_t ReadVram(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) const override { return 0u; }
        void WriteVram(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) override {}
        void SnapshotVram(std::vector<uint8_t> &out) const override { out.clear(); }
        GSTransferSnapshot GetTransferSnapshot() const override { return {}; }

    private:
        EventLog &m_log;
    };

    class LoggingSink final : public GSGifDecoder::Sink
    {
    public:
        explicit LoggingSink(EventLog &log) : m_log(log) {}

        void primitive(const GSGifDecoder &decoder, const GSGifDecoder::Primitive &primitive) override
        {
            const GSStateImage &s = decoder.state();
            GSPrimitiveBatch batch;
            batch.vertexCount = static_cast<uint8_t>(primitive.vertexCount);
            for (uint32_t i = 0; i < primitive.vertexCount; ++i)
                batch.vertices[i] = vertexFromRaw(primitive.vertices[i]);
            batch.state.prim = GSReg::decodePrim(primitive.prim);
            const uint8_t c = batch.state.prim.ctxt ? 1u : 0u;
            GSContext &ctx = batch.state.context;
            ctx.frame = GSReg::decodeFrame(s.reg(GS_REG_FRAME_1 + c));
            ctx.zbuf = GSReg::decodeZbuf(s.reg(GS_REG_ZBUF_1 + c));
            ctx.tex0 = GSReg::decodeTex0(s.reg(GS_REG_TEX0_1 + c));
            ctx.xyoffset = GSReg::decodeXyoffset(s.reg(GS_REG_XYOFFSET_1 + c));
            ctx.scissor = GSReg::decodeScissor(s.reg(GS_REG_SCISSOR_1 + c));
            ctx.tex1 = s.reg(GS_REG_TEX1_1 + c);
            ctx.miptbp1 = s.reg(GS_REG_MIPTBP1_1 + c);
            ctx.miptbp2 = s.reg(GS_REG_MIPTBP2_1 + c);
            ctx.clamp = s.reg(GS_REG_CLAMP_1 + c);
            ctx.alpha = s.reg(GS_REG_ALPHA_1 + c);
            ctx.test = s.reg(GS_REG_TEST_1 + c);
            ctx.fba = s.reg(GS_REG_FBA_1 + c);
            batch.state.texa = GSReg::decodeTexa(s.reg(GS_REG_TEXA));
            batch.state.texclut = GSReg::decodeTexclut(s.reg(GS_REG_TEXCLUT));
            batch.state.pabe = (s.reg(GS_REG_PABE) & 1u) != 0u;
            batch.state.scanmsk = s.reg(GS_REG_SCANMSK);
            batch.state.dimx = s.reg(GS_REG_DIMX);
            batch.state.dthe = s.reg(GS_REG_DTHE);
            batch.state.colclamp = s.reg(GS_REG_COLCLAMP);
            batch.state.fogR = static_cast<uint8_t>(s.reg(GS_REG_FOGCOL));
            batch.state.fogG = static_cast<uint8_t>(s.reg(GS_REG_FOGCOL) >> 8);
            batch.state.fogB = static_cast<uint8_t>(s.reg(GS_REG_FOGCOL) >> 16);
            m_log.primitive(batch);
        }

        void registerWritten(const GSGifDecoder &decoder, uint8_t address, uint64_t value) override
        {
            const GSStateImage &s = decoder.state();
            switch (address)
            {
            case GS_REG_TEX0_1:
            case GS_REG_TEX0_2:
            case GS_REG_TEX2_1:
            case GS_REG_TEX2_2:
            {
                const uint8_t c = (address == GS_REG_TEX0_2 || address == GS_REG_TEX2_2) ? 1u : 0u;
                m_log.clutLoad(GSReg::decodeTex0(s.reg(GS_REG_TEX0_1 + c)), GSReg::decodeTexclut(s.reg(GS_REG_TEXCLUT)));
                break;
            }
            case GS_REG_TRXDIR:
            {
                GSTransferCommand command;
                command.bitbltbuf = GSReg::decodeBitbltbuf(s.reg(GS_REG_BITBLTBUF));
                command.trxpos = GSReg::decodeTrxpos(s.reg(GS_REG_TRXPOS));
                command.trxreg = GSReg::decodeTrxreg(s.reg(GS_REG_TRXREG));
                command.direction = static_cast<uint32_t>(value & 3u);
                m_log.transfer(command);
                break;
            }
            case GS_REG_TEXFLUSH:
                m_log.words.push_back(5u);
                break;
            default:
                break;
            }
        }

        void imageData(const GSGifDecoder &, const uint8_t *data, uint32_t size) override { m_log.image(data, size); }

    private:
        EventLog &m_log;
    };

    struct CpuGs
    {
        std::vector<uint8_t> vram = std::vector<uint8_t>(PS2_GS_VRAM_SIZE, 0u);
        GS gs;
        CpuGs() { gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr); }
        std::vector<uint8_t> snapshot()
        {
            std::vector<uint8_t> out;
            gs.rasterBackend()->SnapshotVram(out);
            return out;
        }
    };
}

void register_ps2_gs_state_tests()
{
    MiniTest::Case("PS2GSState", [](TestCase &tc)
    {
        tc.Run("register encoders invert the frontend's decoders", [](TestCase &t)
        {
            Rng rng{0x1234567u};
            for (int i = 0; i < 2000; ++i)
            {
                const uint64_t v = rng.next();
                t.Equals(GSReg::encodePrim(GSReg::decodePrim(GSReg::encodePrim(GSReg::decodePrim(v)))),
                         GSReg::encodePrim(GSReg::decodePrim(v)), "PRIM");
                t.Equals(GSReg::encodeTex0(GSReg::decodeTex0(v)), v, "TEX0 keeps every bit");
                const GSFrameReg frame = GSReg::decodeFrame(v);
                t.Equals(GSReg::encodeFrame(GSReg::decodeFrame(GSReg::encodeFrame(frame))), GSReg::encodeFrame(frame), "FRAME");
                const GSZbufReg zbuf = GSReg::decodeZbuf(v);
                const GSZbufReg again = GSReg::decodeZbuf(GSReg::encodeZbuf(zbuf));
                t.IsTrue(again.zbp == zbuf.zbp && again.psm == zbuf.psm && again.zmask == zbuf.zmask, "ZBUF");
                const GSTrxPos pos = GSReg::decodeTrxpos(v);
                const GSTrxPos posAgain = GSReg::decodeTrxpos(GSReg::encodeTrxpos(pos));
                t.IsTrue(pos.ssax == posAgain.ssax && pos.ssay == posAgain.ssay && pos.dsax == posAgain.dsax &&
                             pos.dsay == posAgain.dsay && pos.dir == posAgain.dir,
                         "TRXPOS");
                t.Equals(GSReg::encodeBitbltbuf(GSReg::decodeBitbltbuf(GSReg::encodeBitbltbuf(GSReg::decodeBitbltbuf(v)))),
                         GSReg::encodeBitbltbuf(GSReg::decodeBitbltbuf(v)), "BITBLTBUF");
            }
        });

        tc.Run("a state image serializes and reads back unchanged", [](TestCase &t)
        {
            Rng rng{0xC0FFEEu};
            GSStateImage image;
            image.flags = GSStateImage::HasVertexQueue | GSStateImage::HasClut;
            for (const uint8_t address : GSStateImage::kRegisters)
                image.setReg(address, rng.next());
            image.vertexCount = 2u;
            for (GSStateImage::Vertex &vertex : image.vertexQueue)
                vertex = {rng.next(), rng.next(), rng.next(), rng.next(), static_cast<uint8_t>(rng.next())};
            for (uint16_t &entry : image.clut)
                entry = static_cast<uint16_t>(rng.next());
            image.clutCbp = {static_cast<uint32_t>(rng.next()), static_cast<uint32_t>(rng.next())};
            const std::vector<uint8_t> bytes = serializeGsStateImage(image);
            GSStateImage read;
            size_t consumed = 0;
            t.IsTrue(deserializeGsStateImage(bytes.data(), bytes.size(), read, &consumed), "reads back");
            t.Equals(consumed, bytes.size(), "consumes every byte");
            t.IsTrue(serializeGsStateImage(read) == bytes, "same image");
            t.IsTrue(read.registers == image.registers && read.clut == image.clut, "same registers and CLUT");
            t.IsFalse(deserializeGsStateImage(bytes.data(), bytes.size() - 1u, read), "a truncated image is refused");
        });

        tc.Run("GSR2 trace header and commands read back as written", [](TestCase &t)
        {
            // std::tmpfile writes to the drive root on Windows; a file next to the tests works everywhere.
            const char *path = "ps2x_gs_state_test.gsr2";
            FILE *file = std::fopen(path, "w+b");
            t.IsNotNull(file, "temporary file");
            if (!file)
                return;
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE);
            for (size_t i = 0; i < vram.size(); ++i)
                vram[i] = static_cast<uint8_t>(i * 7u);
            GSStateImage image;
            image.flags = GSStateImage::HasVertexQueue;
            image.setReg(GS_REG_FRAME_1, 0x0123456789ull);
            writeGsTraceHeader(file, vram, image);
            const uint64_t registers[4] = {GS_REG_TEXFLUSH, 0u, 0u, 0u};
            const uint8_t payload[3] = {1u, 2u, 3u};
            writeGsTraceCommand(file, GsTraceCommand::Register, registers, nullptr, 0u);
            writeGsTraceCommand(file, GsTraceCommand::GifPacket, registers, payload, sizeof(payload));
            std::rewind(file);
            GsTraceHeader header;
            t.IsTrue(readGsTraceHeader(file, header), "header");
            t.Equals(header.version, 2u, "GSR2");
            t.IsTrue(header.vram == vram, "VRAM");
            t.Equals(header.image.reg(GS_REG_FRAME_1), 0x0123456789ull, "state image");
            GsTraceCommand command;
            t.IsTrue(readGsTraceCommand(file, command) && command.kind == GsTraceCommand::Register &&
                         command.registers[0] == GS_REG_TEXFLUSH && command.bytes.empty(),
                     "register command");
            t.IsTrue(readGsTraceCommand(file, command) && command.kind == GsTraceCommand::GifPacket &&
                         command.bytes == std::vector<uint8_t>(payload, payload + 3),
                     "packet command");
            t.IsFalse(readGsTraceCommand(file, command), "end of trace");
            std::fclose(file);
            std::remove(path);
        });

        tc.Run("drawing continues the same after the state goes through an image", [](TestCase &t)
        {
            for (uint64_t seed = 1; seed <= 6; ++seed)
            {
                PacketGenerator generator{{seed * 0x9E3779B97F4A7C15ull}, true};
                CpuGs a;
                for (int i = 0; i < 60; ++i)
                {
                    const std::vector<uint8_t> packet = generator.packet();
                    a.gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                }
                // Between packets with no transfer in progress, as a recording starts.
                for (int guard = 0; guard < 200; ++guard)
                {
                    const GSTransferSnapshot transfer = a.gs.rasterBackend()->GetTransferSnapshot();
                    if (transfer.direction == 3u || transfer.totalPixels == 0u)
                        break;
                    const std::vector<uint8_t> packet = generator.packet();
                    a.gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                }
                GSStateImage image;
                a.gs.saveState(image);
                t.IsTrue((image.flags & GSStateImage::HasClut) != 0u, "the CPU path saves its CLUT");
                const std::vector<uint8_t> bytes = serializeGsStateImage(image);
                GSStateImage read;
                t.IsTrue(deserializeGsStateImage(bytes.data(), bytes.size(), read), "image reads back");

                CpuGs b;
                b.vram = a.snapshot();
                b.gs.init(b.vram.data(), static_cast<uint32_t>(b.vram.size()), nullptr);
                b.gs.loadState(read);
                GSStateImage reloaded;
                b.gs.saveState(reloaded);
                t.IsTrue(serialized(reloaded) == bytes, "loading then saving gives the same image");

                for (int i = 0; i < 60; ++i)
                {
                    const std::vector<uint8_t> packet = generator.packet();
                    a.gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                    b.gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                }
                t.IsTrue(a.snapshot() == b.snapshot(), "seed " + std::to_string(seed) + ": same VRAM after the image");
            }
        });

        tc.Run("a CLUT image loads back through a CSM1 CT32 palette", [](TestCase &t)
        {
            Rng rng{0xABCDEFu};
            std::array<uint16_t, 512> clut{};
            for (uint16_t &entry : clut)
                entry = static_cast<uint16_t>(rng.next());
            CpuGs cpu;
            for (uint32_t y = 0; y < 16u; ++y)
                for (uint32_t x = 0; x < 16u; ++x)
                    cpu.gs.WriteVram(GS_PSM_CT32, 0u, 1u, x, y, clutPalettePixel(clut, x, y));
            cpu.gs.writeRegister(GS_REG_TEX0_1, (uint64_t(GS_PSM_T8) << 20) | (uint64_t(GS_PSM_CT32) << 51) | (1ull << 61));
            GSStateImage image;
            cpu.gs.saveState(image);
            t.IsTrue(image.clut == clut, "the whole CLUT buffer is the image's");
        });

        tc.Run("the CPU stream renderer draws what the GS draws", [](TestCase &t)
        {
            std::unique_ptr<GSStreamRenderer> renderer =
                GSStreamRenderer::create(GSStreamRendererKind::Cpu, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), {});
            t.IsNotNull(renderer.get(), "the CPU renderer is always there");
            if (!renderer)
                return;
            CpuGs reference;
            std::vector<uint8_t> start(PS2_GS_VRAM_SIZE);
            Rng rng{77u};
            for (uint8_t &byte : start)
                byte = static_cast<uint8_t>(rng.next());
            renderer->writeVram(start.data(), static_cast<uint32_t>(start.size()));
            std::memcpy(reference.vram.data(), start.data(), start.size());
            reference.gs.init(reference.vram.data(), static_cast<uint32_t>(reference.vram.size()), nullptr);

            PacketGenerator generator{{0x5151u}, true};
            for (int i = 0; i < 120; ++i)
            {
                const std::vector<uint8_t> packet = generator.packet();
                renderer->gifTransfer(packet.data(), static_cast<uint32_t>(packet.size()));
                reference.gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
            }
            const uint8_t image[64] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            const uint64_t bitbltbuf = (uint64_t(0x1000u) << 32) | (1ull << 48);
            const uint64_t trxreg = 4u | (4ull << 32);
            renderer->uploadImage(bitbltbuf, 0u, trxreg, 0u, image, sizeof(image));
            reference.gs.uploadImageNative(bitbltbuf, 0u, trxreg, 0u, image, sizeof(image));
            std::vector<uint8_t> vram;
            renderer->readVram(vram);
            t.IsTrue(vram == reference.snapshot(), "same VRAM");

            renderer->writeVramPixel(GS_PSM_CT16, 0x2000u, 2u, 5u, 7u, 0xBEEFu);
            t.Equals(renderer->readVramPixel(GS_PSM_CT16, 0x2000u, 2u, 5u, 7u), 0xBEEFu, "pixel written and read back");
            GSStateImage a, b;
            renderer->saveState(a);
            reference.gs.saveState(b);
            t.IsTrue(serialized(a) == serialized(b), "same state");
        });

        tc.Run("the shared GIF decoder tells a renderer what the frontend tells its backend", [](TestCase &t)
        {
            for (uint64_t seed = 1; seed <= 24; ++seed)
            {
                EventLog frontendLog, decoderLog;
                CpuGs frontend;
                frontend.gs.setRasterBackend(std::make_unique<RecordingBackend>(frontendLog));
                LoggingSink sink(decoderLog);
                GSGifDecoder decoder(sink);
                PacketGenerator generator{{seed * 0xD1B54A32D192ED03ull}, false};
                for (int i = 0; i < 150; ++i)
                {
                    const std::vector<uint8_t> packet = generator.packet();
                    frontend.gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                    decoder.processPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                }
                t.IsTrue(frontendLog.words.size() > 1000u, "the stream draws, loads CLUTs and transfers");
                t.IsTrue(frontendLog.words == decoderLog.words, "seed " + std::to_string(seed) + ": same events");

                GSStateImage fromFrontend, fromDecoder;
                frontend.gs.saveState(fromFrontend);
                CpuGs viaDecoder;
                viaDecoder.gs.setRasterBackend(std::make_unique<RecordingBackend>(decoderLog));
                viaDecoder.gs.loadState(decoder.state());
                viaDecoder.gs.saveState(fromDecoder);
                t.IsTrue(serialized(fromFrontend) == serialized(fromDecoder),
                         "seed " + std::to_string(seed) + ": same state");
            }
        });

        tc.Run("only renderers the frontend hosts are requested", [](TestCase &t)
        {
            GSStreamRendererKind kind{};
            t.IsTrue(GSStreamRenderer::parse("cpu", kind) && kind == GSStreamRendererKind::Cpu, "cpu");
            t.IsTrue(GSStreamRenderer::parse("parallel", kind) && kind == GSStreamRendererKind::Parallel, "parallel");
            t.IsFalse(GSStreamRenderer::parse("gpu", kind), "no gpu renderer yet");
            setTestEnv("PS2X_GS_RENDERER", "cpu");
            t.IsFalse(GSStreamRenderer::requested(kind), "PS2X_GS_RENDERER=cpu keeps the frontend's own path");
            setTestEnv("PS2X_GS_RENDERER", nullptr);
            CpuGs cpu;
            t.IsFalse(cpu.gs.requestStreamRenderer(GSStreamRendererKind::Cpu), "the CPU renderer is not handed the stream");
        });
    });
}
