#ifndef PS2X_NATIVE_TEXTURE_CACHE_H
#define PS2X_NATIVE_TEXTURE_CACHE_H

// Native PC rendering, shared by the ports: the game's textures as RGBA8 images for a PC GPU, without the GS.
// The game uploads its textures and palettes with GS image transfers; this keeps a copy of what it uploaded, in the
// GS memory layout (GSMem), and decodes a texture when a draw selects it with TEX0 (T4, T8, T8H, T4HL, T4HH with a
// CT32 / CT16 palette in CSM1 order, CT32, CT24, CT16). A decoded texture stays cached until an upload touches one
// of its pages or its palette's.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ps2x::native
{
    class TextureCache
    {
    public:
        static constexpr uint32_t kMemoryBytes = 4u * 1024u * 1024u;
        static constexpr uint32_t kPageBytes = 8192u;

        struct Texture
        {
            uint32_t width = 0;
            uint32_t height = 0;
            std::vector<uint32_t> rgba; // R in the low byte; alpha scaled from the GS's 0x80 = opaque to 255
            uint64_t key = 0;
            uint64_t version = 0; // new for every decode: a renderer re-sends the image to its GPU when it changes
            std::vector<std::pair<uint32_t, uint32_t>> pages; // (page, generation) the texture was decoded from
        };

        TextureCache();

        // A host-to-local image transfer as the game sent it: BITBLTBUF, TRXPOS and TRXREG as written to the GS, the
        // pixels packed in the destination format (rows of TRXREG width, 4-bit pixels low nibble first).
        void upload(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, const uint8_t *data, size_t bytes);

        // The image transfers inside GIF packets as a game sends them (PACKED A+D writes of BITBLTBUF, TRXPOS,
        // TRXREG, TRXDIR, then IMAGE data, possibly split over packets): fed to upload().
        void scanGifPacket(const uint8_t *packet, size_t bytes);

        // The texture a draw selects: TEX0 (the CLD bits are ignored), and TEXA for the alpha of CT24 / CT16 texels.
        // Shared, never modified once decoded: uploads may arrive from the GS thread while another thread draws.
        std::shared_ptr<const Texture> texture(uint64_t tex0, uint64_t texa = 0);

        const uint8_t *memory() const { return m_memory.data(); }
        uint64_t uploads() const { return m_uploads; }

        // For a GPU copy of the memory: copy(page, bytes) for every 8 KiB page whose contents changed since
        // `generations` was last brought up to date (every page the first time), under the cache's lock.
        template <typename Copy>
        void changedPages(std::vector<uint32_t> &generations, Copy &&copy) const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (generations.size() != m_pageGeneration.size())
                generations.assign(m_pageGeneration.size(), ~0u);
            for (size_t page = 0; page < m_pageGeneration.size(); ++page)
                if (generations[page] != m_pageGeneration[page])
                {
                    generations[page] = m_pageGeneration[page];
                    copy(static_cast<uint32_t>(page), m_memory.data() + page * kPageBytes);
                }
        }

    private:
        uint32_t decodeTexel(uint32_t psm, uint32_t tbp, uint32_t tbw, uint32_t x, uint32_t y, uint64_t tex0,
                             uint64_t texa, const uint32_t *palette) const;
        // The pages of a rectangle: bumped (an upload) or recorded with their generations (a decoded texture).
        void forEachPage(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                         std::vector<std::pair<uint32_t, uint32_t>> *record);
        bool stillValid(const Texture &texture) const;

        void uploadLocked(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, const uint8_t *data, size_t bytes);

        mutable std::mutex m_mutex;
        std::vector<uint8_t> m_memory;
        // scanGifPacket's transfer state: registers, and the image data of a transfer split over packets
        uint64_t m_bitbltbuf = 0, m_trxpos = 0, m_trxreg = 0;
        std::vector<uint8_t> m_pending;
        size_t m_pendingWanted = 0;
        std::vector<uint32_t> m_pageGeneration;
        std::unordered_map<uint64_t, std::shared_ptr<const Texture>> m_textures;
        uint64_t m_uploads = 0;
        uint64_t m_decodes = 0;
    };

    // The one texture cache every native consumer shares (the mesh API, the GS translating backend), fed by the
    // runtime's GS front end with every upload while uploadsWanted() (any consumer turns it on).
    TextureCache &textures();
    bool uploadsWanted();
    void setUploadsWanted(bool wanted);

    // Index of palette entry `index` in a CSM1 palette stored as a 16-pixel-wide CT32 / CT16 image: (x, y).
    inline void csm1Position(uint32_t index, uint32_t &x, uint32_t &y)
    {
        x = (index & 7u) | ((index & 0x10u) >> 1);
        y = ((index >> 5) << 1) | ((index >> 3) & 1u);
    }
}

#endif
