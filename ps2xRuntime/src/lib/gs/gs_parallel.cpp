// paraLLEl-GS as a GS stream renderer (gs_stream_renderer.h). Compiled only with PS2X_ENABLE_PARALLEL_GS.
#include "runtime/gs/gs_stream_renderer.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_state_image.h"
#include "runtime/gs/ps2_gs_memory.h"

#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"
#include "thread_id.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace Vulkan;
using namespace ParallelGS;

namespace
{
    // Frames are handed to the host 640 pixels per row (GS::copyLatchedHostPresentationFrame).
    constexpr uint32_t kHostFrameWidth = 640u;
    constexpr uint32_t kHostFrameHeight = 512u;
    // GIF path the packets go through: each packet is complete (the frontend's contract), so one path is enough.
    constexpr uint32_t kGifPath = 3u;

    class Signals final : public SignalInterface
    {
    public:
        explicit Signals(const GSStreamRenderer::Callbacks &callbacks) : m_callbacks(callbacks) {}

        bool on_signal(uint64_t payload) override
        {
            if (m_callbacks.signal)
                m_callbacks.signal(payload);
            return false;
        }
        bool on_finish(uint64_t) override
        {
            if (m_callbacks.finish)
                m_callbacks.finish();
            // The GPU can start on what is queued: games FINISH once a frame's drawing is sent.
            return true;
        }
        bool on_label(uint64_t payload) override
        {
            if (m_callbacks.label)
                m_callbacks.label(payload);
            return false;
        }

    private:
        GSStreamRenderer::Callbacks m_callbacks;
    };

    class ParallelRenderer final : public GSStreamRenderer
    {
    public:
        ParallelRenderer(uint32_t vramSize, const Callbacks &callbacks) : m_vramSize(vramSize), m_signals(callbacks) {}

        ~ParallelRenderer() override
        {
            if (m_initialized)
                m_device.wait_idle();
        }

        bool init()
        {
            // Granite indexes per-thread state with this; the renderer lives on the GS thread.
            Util::register_thread_index(0);
            if (!Context::init_loader(nullptr))
            {
                std::fprintf(stderr, "[gs-parallel] no Vulkan loader\n");
                return false;
            }
            m_context.set_num_thread_indices(1);
            if (!m_context.init_instance_and_device(nullptr, 0, nullptr, 0,
                                                    CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT |
                                                        CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT |
                                                        CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT))
            {
                std::fprintf(stderr, "[gs-parallel] no Vulkan device\n");
                return false;
            }
            m_device.set_context(m_context);
            m_device.init_frame_contexts(4);

            GSOptions options{};
            options.vram_size = m_vramSize;
            if (!m_interface.init(&m_device, options))
            {
                std::fprintf(stderr, "[gs-parallel] renderer init failed (missing Vulkan features?)\n");
                return false;
            }
            m_interface.set_signal_interface(&m_signals);
            m_initialized = true;
            std::fprintf(stderr, "[gs-parallel] drawing on %s\n", m_device.get_gpu_properties().deviceName);
            return true;
        }

        void gifTransfer(const uint8_t *data, uint32_t sizeBytes) override
        {
            sizeBytes &= ~15u;
            if (!data || sizeBytes == 0u)
                return;
            m_interface.gif_transfer(kGifPath, data, sizeBytes);
            // A packet that ends early leaves no tag pending: the next one starts with its own (as in
            // GS::processGIFPacket).
            GIFPath &path = m_interface.get_gif_path(kGifPath);
            path.loop = path.tag.NLOOP;
            path.reg = 0;
        }

        void writeRegister(uint8_t address, uint64_t value) override
        {
            m_interface.write_register(static_cast<RegisterAddr>(address), value);
        }

        void uploadImage(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, uint64_t trxdir, const uint8_t *data,
                         uint32_t sizeBytes) override
        {
            if (!data || sizeBytes == 0u)
                return;
            const uint64_t registers[4] = {bitbltbuf, trxpos, trxreg, trxdir};
            for (uint32_t i = 0; i < 4u; ++i)
                writeRegister(static_cast<uint8_t>(GS_REG_BITBLTBUF + i), registers[i]);
            // The image data behind an IMAGE tag, padded to whole quadwords.
            const uint32_t qwords = (sizeBytes + 15u) / 16u;
            m_packet.assign(16u + qwords * 16u, 0u);
            const uint64_t tag = qwords | (1ull << 15) | (static_cast<uint64_t>(GIF_FMT_IMAGE) << 58);
            std::memcpy(m_packet.data(), &tag, sizeof(tag));
            std::memcpy(m_packet.data() + 16u, data, sizeBytes);
            gifTransfer(m_packet.data(), static_cast<uint32_t>(m_packet.size()));
        }

        void flush() override { m_interface.flush(); }

        PresentationFrame present(const GSPresentationRequest &request) override
        {
            PrivRegisterState &priv = m_interface.get_priv_register_state();
            std::memcpy(&priv.pmode, &request.pmode, sizeof(uint64_t));
            std::memcpy(&priv.smode2, &request.smode2, sizeof(uint64_t));
            std::memcpy(&priv.dispfb1, &request.dispfb1, sizeof(uint64_t));
            std::memcpy(&priv.display1, &request.display1, sizeof(uint64_t));
            std::memcpy(&priv.dispfb2, &request.dispfb2, sizeof(uint64_t));
            std::memcpy(&priv.display2, &request.display2, sizeof(uint64_t));
            std::memcpy(&priv.bgcolor, &request.bgcolor, sizeof(uint64_t));
            if (request.smode1 != 0u)
                std::memcpy(&priv.smode1, &request.smode1, sizeof(uint64_t));
            else
            {
                // SetGsCrt did not say: PAL when a circuit shows more lines than NTSC has.
                const auto lines = [](uint64_t display) { return static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u; };
                priv.smode1.CMOD = std::max(lines(request.display1), lines(request.display2)) > 480u
                                       ? SMODE1Bits::CMOD_PAL
                                       : SMODE1Bits::CMOD_NTSC;
                priv.smode1.LC = SMODE1Bits::LC_ANALOG;
            }

            m_interface.flush();
            VSyncInfo info{};
            info.phase = static_cast<uint32_t>(request.vsyncTick & 1u);
            info.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            info.dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT;
            info.dst_access = VK_ACCESS_2_TRANSFER_READ_BIT;
            info.force_progressive = true;
            info.anti_blur = true;
            info.adapt_to_internal_horizontal_resolution = true;
            info.raw_circuit_scanout = true;
            const ScanoutResult scanout = m_interface.vsync(info);

            PresentationFrame frame{};
            // PS2X_GS_PARALLEL_LOG=1: the display registers and the picture size whenever they change.
            static const bool s_log = std::getenv("PS2X_GS_PARALLEL_LOG") != nullptr;
            if (s_log)
            {
                static uint64_t s_last[8]{};
                const uint64_t now[8] = {request.pmode, request.smode1, request.smode2, request.dispfb1, request.display1,
                                         request.dispfb2, request.display2,
                                         scanout.image ? (uint64_t(scanout.image->get_width()) << 32 | scanout.image->get_height()) : 0u};
                if (std::memcmp(now, s_last, sizeof(now)) != 0)
                {
                    std::memcpy(s_last, now, sizeof(now));
                    std::fprintf(stderr,
                                 "[gs-parallel] pmode=%llx smode1=%llx smode2=%llx dispfb1=%llx display1=%llx dispfb2=%llx "
                                 "display2=%llx -> %ux%u (internal %ux%u mode %ux%u)\n",
                                 (unsigned long long)now[0], (unsigned long long)now[1], (unsigned long long)now[2],
                                 (unsigned long long)now[3], (unsigned long long)now[4], (unsigned long long)now[5],
                                 (unsigned long long)now[6], unsigned(now[7] >> 32), unsigned(now[7] & 0xFFFFFFFFu),
                                 scanout.internal_width, scanout.internal_height, scanout.mode_width, scanout.mode_height);
                }
            }
            if (!scanout.image)
                return frame;
            const uint32_t width = scanout.image->get_width();
            const uint32_t height = scanout.image->get_height();
            const VkDeviceSize bytes = VkDeviceSize(width) * height * 4u;
            if (!m_readback || m_readback->get_create_info().size < bytes)
            {
                BufferCreateInfo info{};
                info.size = bytes;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                info.domain = BufferDomain::CachedHost;
                m_readback = m_device.create_buffer(info);
            }
            CommandBufferHandle cmd = m_device.request_command_buffer();
            cmd->copy_image_to_buffer(*m_readback, *scanout.image, 0, {}, {width, height, 1}, 0, 0,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
            cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_READ_BIT);
            Fence fence;
            m_device.submit(cmd, &fence);
            fence->wait();

            const auto *pixels = static_cast<const uint8_t *>(m_device.map_host_buffer(*m_readback, MEMORY_ACCESS_READ_BIT));
            frame.width = std::min(width, kHostFrameWidth);
            frame.height = std::min(height, kHostFrameHeight);
            frame.pixels.resize(size_t(kHostFrameWidth) * kHostFrameHeight * 4u);
            for (uint32_t y = 0; y < frame.height; ++y)
            {
                const uint8_t *src = pixels + size_t(y) * width * 4u;
                uint8_t *dst = frame.pixels.data() + size_t(y) * kHostFrameWidth * 4u;
                if (width <= kHostFrameWidth)
                    std::memcpy(dst, src, size_t(width) * 4u);
                else // wider than the host frame: nearest sample
                    for (uint32_t x = 0; x < kHostFrameWidth; ++x)
                        std::memcpy(dst + x * 4u, src + size_t(x) * width / kHostFrameWidth * 4u, 4u);
                for (uint32_t x = 0; x < frame.width; ++x)
                    dst[x * 4u + 3u] = 0xFFu;
            }
            m_device.unmap_host_buffer(*m_readback, MEMORY_ACCESS_READ_BIT);
            return frame;
        }

        void readTransferFifo(uint8_t *dst, uint32_t qwords) override { m_interface.read_transfer_fifo(dst, qwords); }

        void readVram(std::vector<uint8_t> &out) override
        {
            const auto *vram = static_cast<const uint8_t *>(m_interface.map_vram_read(0, m_vramSize));
            out.assign(vram, vram + m_vramSize);
        }

        void writeVram(const uint8_t *data, uint32_t sizeBytes) override
        {
            const uint32_t bytes = std::min(sizeBytes, m_vramSize);
            if (!data || bytes == 0u)
                return;
            std::memcpy(m_interface.map_vram_write(0, bytes), data, bytes);
            m_interface.end_vram_write(0, bytes);
        }

        uint32_t readVramPixel(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) override
        {
            // Read-only: the pixel readers take a mutable pointer but do not write through it.
            auto *vram = const_cast<uint8_t *>(static_cast<const uint8_t *>(m_interface.map_vram_read(0, m_vramSize)));
            return GSCpuBackend::ReadVramPixel(vram, psm, base, bw, x, y);
        }

        void writeVramPixel(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            const size_t offset = (size_t(GSMem::PageOf(psm, base, bw, x, y)) * kPageBytes) % m_vramSize;
            uint8_t *page = static_cast<uint8_t *>(m_interface.map_vram_write(offset, kPageBytes));
            // The pixel writers address the whole of VRAM; only this page is mapped for writing.
            GSCpuBackend::WriteVramPixel(page - offset, psm, base, bw, x, y, value);
            m_interface.end_vram_write(offset, kPageBytes);
        }

        // paraLLEl keeps the vertex queue and the CLUT buffer to itself: the image has neither (flags clear).
        // PRMODE is folded into PRIM as paraLLEl holds it, which a renderer loading the image resolves the same way.
        void saveState(GSStateImage &image) override
        {
            const RegisterState &r = m_interface.get_register_state();
            image = GSStateImage{};
            for (uint8_t i = 0; i < 2u; ++i)
            {
                const ContextState &c = r.ctx[i];
                image.setReg(GS_REG_TEX0_1 + i, c.tex0.bits);
                image.setReg(GS_REG_CLAMP_1 + i, c.clamp.bits);
                image.setReg(GS_REG_TEX1_1 + i, c.tex1.bits);
                image.setReg(GS_REG_XYOFFSET_1 + i, c.xyoffset.bits);
                image.setReg(GS_REG_MIPTBP1_1 + i, c.miptbl_1_3.bits);
                image.setReg(GS_REG_MIPTBP2_1 + i, c.miptbl_4_6.bits);
                image.setReg(GS_REG_SCISSOR_1 + i, c.scissor.bits);
                image.setReg(GS_REG_ALPHA_1 + i, c.alpha.bits);
                image.setReg(GS_REG_TEST_1 + i, c.test.bits);
                image.setReg(GS_REG_FBA_1 + i, c.fba.bits);
                image.setReg(GS_REG_FRAME_1 + i, c.frame.bits);
                image.setReg(GS_REG_ZBUF_1 + i, c.zbuf.bits);
            }
            image.setReg(GS_REG_PRIM, r.prim.bits);
            image.setReg(GS_REG_PRMODE, r.prim.bits);
            image.setReg(GS_REG_PRMODECONT, r.prmodecont.bits);
            image.setReg(GS_REG_RGBAQ, r.rgbaq.bits);
            image.setReg(GS_REG_ST, r.st.bits);
            image.setReg(GS_REG_UV, r.uv.bits);
            image.setReg(GS_REG_FOG, r.fog.bits);
            image.setReg(GS_REG_TEXCLUT, r.texclut.bits);
            image.setReg(GS_REG_TEXA, r.texa.bits);
            image.setReg(GS_REG_FOGCOL, r.fogcol.bits);
            image.setReg(GS_REG_DIMX, r.dimx.bits);
            image.setReg(GS_REG_DTHE, r.dthe.bits);
            image.setReg(GS_REG_COLCLAMP, r.colclamp.bits);
            image.setReg(GS_REG_PABE, r.pabe.bits);
            image.setReg(GS_REG_SCANMSK, r.scanmsk.bits);
            image.setReg(GS_REG_BITBLTBUF, r.bitbltbuf.bits);
            image.setReg(GS_REG_TRXPOS, r.trxpos.bits);
            image.setReg(GS_REG_TRXREG, r.trxreg.bits);
            image.setReg(GS_REG_TRXDIR, r.trxdir.bits);
            image.clutCbp = {r.cached_cbp[0], r.cached_cbp[1]};
        }

        void loadState(const GSStateImage &image) override
        {
            const bool prmodecont = (image.reg(GS_REG_PRMODECONT) & 1u) != 0u;
            const uint64_t prim = prmodecont ? image.reg(GS_REG_PRIM)
                                             : (image.reg(GS_REG_PRMODE) & ~0x7ull) | (image.reg(GS_REG_PRIM) & 0x7u);
            writeRegister(GS_REG_PRMODECONT, 1u);
            writeRegister(GS_REG_PRIM, prim);
            writeRegister(GS_REG_PRMODECONT, image.reg(GS_REG_PRMODECONT));

            if ((image.flags & GSStateImage::HasClut) != 0u)
                loadClut(image.clut);

            // TEX0 without its CLUT load (the buffer was just restored); TEX0 regenerates MIPTBP1 when TEX1.MTBA
            // is set, so MIPTBP comes after it.
            static constexpr uint8_t kContextRegisters[] = {GS_REG_TEX1_1, GS_REG_TEX0_1, GS_REG_CLAMP_1,
                                                            GS_REG_XYOFFSET_1, GS_REG_SCISSOR_1, GS_REG_ALPHA_1,
                                                            GS_REG_TEST_1, GS_REG_FBA_1, GS_REG_FRAME_1,
                                                            GS_REG_ZBUF_1, GS_REG_MIPTBP1_1, GS_REG_MIPTBP2_1};
            for (uint8_t i = 0; i < 2u; ++i)
                for (const uint8_t address : kContextRegisters)
                {
                    uint64_t value = image.reg(address + i);
                    if (address == GS_REG_TEX0_1)
                        value &= ~(7ull << 61);
                    writeRegister(static_cast<uint8_t>(address + i), value);
                }
            for (const uint8_t address : {GS_REG_TEXCLUT, GS_REG_TEXA, GS_REG_FOGCOL, GS_REG_DIMX, GS_REG_DTHE,
                                          GS_REG_COLCLAMP, GS_REG_PABE, GS_REG_SCANMSK, GS_REG_BITBLTBUF,
                                          GS_REG_TRXPOS, GS_REG_TRXREG})
                writeRegister(address, image.reg(address));

            if ((image.flags & GSStateImage::HasVertexQueue) != 0u)
                for (uint32_t i = 0; i < std::min(image.vertexCount, GSStateImage::kMaxQueuedVertices); ++i)
                {
                    const GSStateImage::Vertex &vertex = image.vertexQueue[i];
                    writeRegister(GS_REG_RGBAQ, vertex.rgbaq);
                    writeRegister(GS_REG_ST, vertex.st);
                    writeRegister(GS_REG_UV, vertex.uv);
                    writeRegister(GS_REG_FOG, uint64_t(vertex.fog) << 56);
                    writeRegister(GS_REG_XYZ3, vertex.xyz);
                }
            for (const uint8_t address : {GS_REG_RGBAQ, GS_REG_ST, GS_REG_UV, GS_REG_FOG})
                writeRegister(address, image.reg(address));

            RegisterState &r = m_interface.get_register_state();
            r.trxdir.bits = image.reg(GS_REG_TRXDIR);
            const uint32_t qBits = static_cast<uint32_t>(image.reg(GS_REG_RGBAQ) >> 32);
            std::memcpy(&r.internal_q, &qBits, sizeof(qBits));
            if ((image.flags & GSStateImage::HasClut) != 0u)
            {
                r.cached_cbp[0] = image.clutCbp[0];
                r.cached_cbp[1] = image.clutCbp[1];
            }
        }

        void reset() override { m_interface.reset_context_state(); }

    private:
        static constexpr size_t kPageBytes = 8192u;

        // Fills paraLLEl's CLUT buffer by having it load the image as a CSM1 CT32 palette from page 0, then puts
        // page 0 back.
        void loadClut(const std::array<uint16_t, 512> &clut)
        {
            std::vector<uint8_t> saved(kPageBytes);
            uint8_t *page = static_cast<uint8_t *>(m_interface.map_vram_write(0, kPageBytes));
            std::memcpy(saved.data(), page, kPageBytes);
            for (uint32_t y = 0; y < 16u; ++y)
                for (uint32_t x = 0; x < 16u; ++x)
                    GSCpuBackend::WriteVramPixel(page, GS_PSM_CT32, 0u, 1u, x, y, clutPalettePixel(clut, x, y));
            m_interface.end_vram_write(0, kPageBytes);
            const uint64_t tex0 = (uint64_t(GS_PSM_T8) << 20) | (uint64_t(GS_PSM_CT32) << 51) | (1ull << 61);
            writeRegister(GS_REG_TEX0_1, tex0);
            std::memcpy(m_interface.map_vram_write(0, kPageBytes), saved.data(), kPageBytes);
            m_interface.end_vram_write(0, kPageBytes);
        }

        std::vector<uint8_t> m_packet;
        uint32_t m_vramSize;
        Signals m_signals;
        Context m_context;
        Device m_device;
        GSInterface m_interface;
        BufferHandle m_readback;
        bool m_initialized = false;
    };
}

std::unique_ptr<GSStreamRenderer> createParallelGsRenderer(uint32_t vramSize, const GSStreamRenderer::Callbacks &callbacks)
{
    auto renderer = std::make_unique<ParallelRenderer>(vramSize, callbacks);
    if (!renderer->init())
        return nullptr;
    return renderer;
}
