#pragma once

#include "ps2_stubs.h"

namespace ps2_stubs
{
    void resetMpegStubState();
    void enqueueMpegDecodedFrameForTesting(uint32_t mpegAddr);
    void notifyMpegCdStreamStart(PS2Runtime *runtime = nullptr);
    void notifyMpegCdStreamDataProduced(uint32_t byteCount, bool endOfStream);
    void notifyMpegCdStreamEof(PS2Runtime *runtime = nullptr);
    // End an HLE playback the game abandoned before its stream ended. libmpeg games abort by
    // feeding a sequence_end_code to the IPU (e.g. the ezmpeg sample's videoDecFlush), which
    // never reaches the host decoder: game overrides call this at that point instead. Drops
    // the queued pictures, makes sceMpegIsEnd/sceMpegIsRefBuffEmpty report the end, and wakes
    // a thread blocked in sceMpegGetPicture.
    void abortMpegPlayback(PS2Runtime *runtime, uint32_t mpegAddr);
    // The game fed all of a movie's data: decode what is buffered, then let sceMpegIsEnd report the
    // end once the queued pictures are drained. libmpeg players end a stream by appending a
    // sequence_end_code to the IPU ring (the ezmpeg sample's videoDecFlush) and may never read
    // the PSS's trailing program end code; game overrides call this at that point.
    void endMpegPlaybackInput(PS2Runtime *runtime, uint32_t mpegAddr);
    void sceMpegFlush(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegAddBs(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegAddCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegAddStrCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegClearRefBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegCreate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDelete(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDemuxPss(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDemuxPssRing(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispCenterOffX(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispCenterOffY(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispHeight(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispWidth(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetDecodeMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetPicture(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetPictureRAW8(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetPictureRAW8xy(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegIsEnd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegIsRefBuffEmpty(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegReset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegResetDefaultPtsGap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegSetDecodeMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegSetDefaultPtsGap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegSetImageBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
}
