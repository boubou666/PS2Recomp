// SPU2 output: the IOP emulator mixes 48 kHz stereo in emulated time; a raylib stream plays it in
// real time. A time stretcher sits between the two, so that an emulation running slower or faster
// than real time slows down or speeds up the sound (same pitch) instead of cutting it.
// PS2X_SPU2_WAV=<path> also records the output to a WAV file. PS2X_SPU2_STATS=1 prints, every second, the frames
// the emulation produced and the device took, the stretcher's buffer fill and rate, underruns and dropped frames.

#include "runtime/ps2_audio.h"
#include "ps2_audio_stretch.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace
{
    constexpr uint32_t kSampleRate = 48000u;
    constexpr unsigned int kStreamBufferFrames = 1024u;

    void writeWavHeader(std::ofstream &file, uint32_t dataBytes)
    {
        const auto u32 = [&](uint32_t value)
        { file.write(reinterpret_cast<const char *>(&value), 4); };
        const auto u16 = [&](uint16_t value)
        { file.write(reinterpret_cast<const char *>(&value), 2); };
        file.write("RIFF", 4);
        u32(36u + dataBytes);
        file.write("WAVEfmt ", 8);
        u32(16u);
        u16(1u);
        u16(2u);
        u32(kSampleRate);
        u32(kSampleRate * 4u);
        u16(4u);
        u16(16u);
        file.write("data", 4);
        u32(dataBytes);
    }
}

struct PS2AudioBackend::Spu2Stream
{
#if !defined(PLATFORM_VITA)
    AudioStream stream{};
#endif
    bool playing = false;
    std::mutex mutex;
    AudioTimeStretcher stretcher;
    std::ofstream wav;
    uint32_t wavBytes = 0;
    const bool stats = []
    {
        const char *value = std::getenv("PS2X_SPU2_STATS");
        return value && *value == '1';
    }();
    uint64_t pushedFrames = 0;
    uint64_t pulledFrames = 0;
    std::chrono::steady_clock::time_point nextStats = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    PS2AudioBackend::Spu2Source source; // the SPU2 on the wall clock (setSpu2Source)

    void reportStats()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now < nextStats)
            return;
        nextStats = now + std::chrono::seconds(1);
        std::fprintf(stderr, "[spu2:stats] produced %llu taken %llu fill %zu rate %.2f underruns %llu dropped %llu\n",
                     static_cast<unsigned long long>(pushedFrames), static_cast<unsigned long long>(pulledFrames),
                     stretcher.fill(), stretcher.rate(), static_cast<unsigned long long>(stretcher.underruns()),
                     static_cast<unsigned long long>(stretcher.dropped()));
        pushedFrames = pulledFrames = 0;
    }

    void push(const int16_t *frames, size_t count)
    {
        std::lock_guard<std::mutex> lock(mutex);
        stretcher.push(frames, count);
        if (stats)
        {
            pushedFrames += count;
            reportStats();
        }
        if (wav.is_open())
        {
            wav.write(reinterpret_cast<const char *>(frames), static_cast<std::streamsize>(count * 4u));
            wavBytes += static_cast<uint32_t>(count * 4u);
        }
    }

    void pull(int16_t *out, unsigned int frames)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (source)
        {
            source(out, frames);
            if (wav.is_open())
            {
                wav.write(reinterpret_cast<const char *>(out), static_cast<std::streamsize>(frames * 4u));
                wavBytes += static_cast<uint32_t>(frames * 4u);
            }
        }
        else
            stretcher.pull(out, frames);
        pulledFrames += frames;
    }

    void closeWav()
    {
        if (!wav.is_open())
            return;
        wav.seekp(0);
        writeWavHeader(wav, wavBytes);
        wav.close();
    }
};

namespace
{
    std::atomic<PS2AudioBackend::Spu2Stream *> g_spu2Stream{nullptr};

#if !defined(PLATFORM_VITA)
    void spu2StreamCallback(void *buffer, unsigned int frames)
    {
        if (PS2AudioBackend::Spu2Stream *stream = g_spu2Stream.load())
            stream->pull(static_cast<int16_t *>(buffer), frames);
        else
            std::memset(buffer, 0, static_cast<size_t>(frames) * 4u);
    }
#endif
}

void PS2AudioBackend::Spu2StreamDeleter::operator()(Spu2Stream *stream) const
{
    delete stream;
}

void PS2AudioBackend::pushSpu2Frames(const int16_t *frames, size_t count)
{
    if (!frames || count == 0u)
        return;
    if (!m_spu2)
    {
        m_spu2.reset(new Spu2Stream());
        if (const char *path = std::getenv("PS2X_SPU2_WAV"); path && *path)
        {
            m_spu2->wav.open(path, std::ios::binary | std::ios::trunc);
            if (m_spu2->wav.is_open())
                writeWavHeader(m_spu2->wav, 0u);
        }
#if !defined(PLATFORM_VITA)
        if (m_audioReady && IsAudioDeviceReady())
        {
            SetAudioStreamBufferSizeDefault(static_cast<int>(kStreamBufferFrames));
            m_spu2->stream = LoadAudioStream(kSampleRate, 16u, 2u);
            SetAudioStreamBufferSizeDefault(0);
            if (IsAudioStreamValid(m_spu2->stream))
            {
                g_spu2Stream.store(m_spu2.get());
                SetAudioStreamCallback(m_spu2->stream, spu2StreamCallback);
                PlayAudioStream(m_spu2->stream);
                m_spu2->playing = true;
            }
        }
#endif
    }
    {
        std::lock_guard<std::mutex> lock(m_spu2MixerMutex);
        if (m_spu2Mixer)
        {
            m_spu2MixBuffer.assign(frames, frames + count * 2u);
            m_spu2Mixer(m_spu2MixBuffer.data(), count);
            frames = m_spu2MixBuffer.data();
        }
    }
    m_spu2->push(frames, count);
}

void PS2AudioBackend::setSpu2Source(Spu2Source source)
{
    if (source && !m_spu2)
    {
        const int16_t silence[2] = {0, 0};
        pushSpu2Frames(silence, 1u); // opens the stream (and PS2X_SPU2_WAV) as the first pushed frames would
    }
    if (m_spu2)
    {
        std::lock_guard<std::mutex> lock(m_spu2->mutex);
        m_spu2->source = std::move(source);
    }
}

void PS2AudioBackend::setSpu2Mixer(Spu2Mixer mixer)
{
    std::lock_guard<std::mutex> lock(m_spu2MixerMutex);
    m_spu2Mixer = std::move(mixer);
}

void PS2AudioBackend::closeSpu2Stream()
{
    if (!m_spu2)
        return;
#if !defined(PLATFORM_VITA)
    if (m_spu2->playing && IsAudioDeviceReady())
    {
        StopAudioStream(m_spu2->stream);
        g_spu2Stream.store(nullptr);
        UnloadAudioStream(m_spu2->stream);
    }
#endif
    g_spu2Stream.store(nullptr);
    m_spu2->playing = false;
    m_spu2->closeWav();
    m_spu2.reset();
}
