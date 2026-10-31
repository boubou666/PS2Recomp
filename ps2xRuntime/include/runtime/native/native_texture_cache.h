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
        uint64_t decodes() const { return m_decodes; }

        // Store mode (a port that takes its textures from the game's own data, no GS memory): each upload is kept as
        // the game sent it (destination base, width and format, rectangle, pixels); texture() lays out the uploads
        // touching its pages and its palette's, oldest first, in a scratch image by the GS's layout formulas, and reads
        // it back in its format (an asset sent as CT32 and read as T4, a palette sent in parts). Nothing of the scratch
        // is kept; an upload identical to one already kept changes nothing. textureMisses() counts the textures no
        // upload covers.
        void setStoreMode(bool on);
        bool storeMode() const { return m_storeMode; }
        uint64_t textureMisses() const { return m_storeMisses; }
        // TEXA and FOGCOL as the game last wrote them in the packets scanGifPacket saw (A+D), for draws whose
        // material takes them from the game's data.
        uint64_t gameTexa() const { return m_gameTexa; }
        uint32_t gameFogColour() const { return m_gameFogColour; }

    private:
        struct StoredUpload
        {
            uint32_t dbp = 0, psm = 0, dx = 0, dy = 0, w = 0, h = 0, dbw = 0;
            uint32_t begin = 0, end = 0; // the blocks of the pages it touches
            uint64_t seq = 0; // when the game last sent it (an identical upload again counts)
            uint64_t id = 0;  // its pixels: new for every upload that changes them
            std::vector<uint16_t> blocks; // the blocks it writes
            std::vector<uint8_t> data;
        };
        std::shared_ptr<const Texture> textureFromStore(uint64_t tex0, uint64_t texa);
        std::shared_ptr<const Texture> memoryTexture(uint64_t tex0, uint64_t texa);
        static bool compareStore();
        void compareWithMemory(uint64_t tex0, uint64_t texa, const Texture *stored);
        int m_lastSource = 0; // the last store decode: 0 nothing, 1 one upload, 2 several, 3 cached
        void storeUpload(uint32_t dbp, uint32_t dbw, uint32_t dpsm, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h,
                         const uint8_t *data, size_t bytes);
        bool m_storeMode = false;
        std::vector<StoredUpload> m_stored; // oldest first
        struct StoredTexture
        {
            std::shared_ptr<const Texture> texture;
            uint64_t signature = 0; // the uploads it was decoded from
        };
        std::unordered_map<uint64_t, StoredTexture> m_storeTextures;
        uint64_t m_storeMisses = 0;
        uint64_t m_uploadSeq = 0, m_contentSerial = 0;
        std::vector<uint64_t> m_blockSeq = std::vector<uint64_t>(kMemoryBytes / 256u, 0u); // the last upload filling each block whole
        uint64_t m_gameTexa = 0;
        uint32_t m_gameFogColour = 0;

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
