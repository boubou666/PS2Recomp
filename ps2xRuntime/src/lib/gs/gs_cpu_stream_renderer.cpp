// The CPU renderer as a GS stream renderer: a synchronous GS (the frontend's GIF decoder and the CPU backend) over
// its own VRAM. It reports no SIGNAL / FINISH / LABEL callbacks.
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_state_image.h"
#include "runtime/gs/gs_stream_renderer.h"

#include <algorithm>
#include <cstring>

namespace
{
    class CpuRenderer final : public GSStreamRenderer
    {
    public:
        explicit CpuRenderer(uint32_t vramSize) : m_vram(vramSize, 0u)
        {
            m_gs.init(m_vram.data(), vramSize, nullptr);
        }

        void gifTransfer(const uint8_t *data, uint32_t sizeBytes) override { m_gs.processGIFPacket(data, sizeBytes); }

        void nativePackedGifTransfer(const uint8_t *data, uint32_t sizeBytes) override
        {
            m_gs.processNativePackedGIFPacket(data, sizeBytes);
        }

        void uploadImage(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, uint64_t trxdir, const uint8_t *data,
                         uint32_t sizeBytes) override
        {
            m_gs.uploadImageNative(bitbltbuf, trxpos, trxreg, trxdir, data, sizeBytes);
        }

        void writeRegister(uint8_t address, uint64_t value) override { m_gs.writeRegister(address, value); }

        void flush() override { backend().Flush(); }

        PresentationFrame present(const GSPresentationRequest &display) override
        {
            GSPresentationRequest request = display;
            request.contextFrames[0] = m_gs.getContextFrame(0);
            request.contextFrames[1] = m_gs.getContextFrame(1);
            request.hasPreferredSource = m_gs.getPreferredDisplaySource(request.preferredSource, request.preferredDestFbp);
            backend().Flush();
            backend().Sync(GSSyncReason::Presentation);
            return backend().Present(request);
        }

        void readTransferFifo(uint8_t *dst, uint32_t qwords) override { m_gs.consumeLocalToHostBytes(dst, qwords * 16u); }

        void readVram(std::vector<uint8_t> &out) override { backend().SnapshotVram(out); }

        void writeVram(const uint8_t *data, uint32_t sizeBytes) override
        {
            if (!data)
                return;
            // The backend's texture caches follow its own writes only: start it over on the new contents, keeping
            // the CLUT buffer.
            const std::vector<uint8_t> clut = backend().SaveClutState();
            std::memcpy(m_vram.data(), data, std::min<size_t>(sizeBytes, m_vram.size()));
            backend().Initialize(m_vram.data(), static_cast<uint32_t>(m_vram.size()));
            backend().LoadClutState(clut.data(), clut.size());
        }

        uint32_t readVramPixel(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) override
        {
            return m_gs.ReadVram(psm, base, bw, x, y);
        }

        void writeVramPixel(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            m_gs.WriteVram(psm, base, bw, x, y, value);
        }

        void saveState(GSStateImage &image) override { m_gs.saveState(image); }
        void loadState(const GSStateImage &image) override { m_gs.loadState(image); }
        void reset() override { m_gs.reset(); }

    private:
        GSCpuBackend &backend() { return static_cast<GSCpuBackend &>(*m_gs.rasterBackend()); }

        std::vector<uint8_t> m_vram;
        GS m_gs;
    };
}

std::unique_ptr<GSStreamRenderer> createCpuGsRenderer(uint32_t vramSize, const GSStreamRenderer::Callbacks &)
{
    return std::make_unique<CpuRenderer>(vramSize);
}
