# Native mode: running a recompiled game without simulating the PS2

Status: layer 1 (frame loop) in progress on `native/frame-loop`. Discussed in f2f as `[NATIVE-n]` topics, decisions
on c2c.

## Goal

The static recompiler already turns the game's EE code into native code. Native mode takes the rest of the way:
nothing around that code simulates PS2 hardware or its timing any more. The game's code (EE, VU microprograms, IOP
modules) runs natively; the data it produces (GS command streams, sound samples, movies) goes through native
translators; time comes from the host.

Native mode is a separate, opt-in mode: a game turns each layer on from its game override, and the default path keeps
the hardware model and its checks unchanged. Early native layers may regress a game; they are fixed as they go.

## Layers

| # | Layer | What goes away | State |
| --- | --- | --- | --- |
| 1 | Frame loop on real time | vblank pacing, EE time held to the vblank grid | first version (below) |
| 2 | Audio on wall time | IOP time paid by EE cycles, SPU2 output following emulation speed | proposed |
| 3 | Translating renderer | GS emulation: GS state and primitives become GPU draws with GPU render targets | proposed |
| 4 | Hardware waits | modelled latencies of GS/VIF/DMA status, readbacks served from host state | proposed |
| 5 | Native VU output | VU1 microprograms feed the GPU directly instead of the simulated VIF/GIF path | later |

Long-term goals that shape the layers: a frame cap chosen by the player (30 to 60, or unlimited), higher internal
resolution, post-process passes on the final frame, and texture replacement (modding).

## Layer 1: frame loop on real time

A game declares where its frame ends and how its engine steps time; the runtime paces frames on the host and gives
the game the real time each frame took.

- `FramePacer` (`runtime/frame_pacer.h`, `PS2Runtime::framePacer()`): `frameEnd()` holds the frame to the cap
  (`PS2X_FRAME_CAP`, default 60, 0 = unlimited, `setCap()` at run time) and returns the frame's real duration in
  seconds, clamped to 0.1 s.
- `EeScheduler::setNativeFrames(true)`: vblanks no longer latch the picture (the game's frame end does, through
  `presentGuestFrame()`), and due events no longer wait for their host deadline: EE time is kept on real time by the
  game's frame end instead. The vblank timer keeps running for what else counts on it (vblank handlers, IOP vblank
  waits).
- The host window follows the cap (`SetTargetFPS`), and uploads a picture per frame the game ends.

What the game override does at its frame end (first user: VP2S, `port/vp2s_overrides.cpp`, `VP2S_NATIVE=1`):

1. replace the game's vblank wait with a function that calls `framePacer().frameEnd()`, charges the EE the real time
   the frame did not account (`accountCycles`), calls `presentGuestFrame()` and returns at once;
2. write the real step into the engine's time-step variables (VP2: its field loop's per-frame step at `0x35DED8`,
   1/60 s units, and `0x35DEE8`, seconds) and make the loop count one tick per frame.

VP2S first field scene, cap 60: ~44 distinct frames/s, 48 presented/s, host never idle (the EE and the VIF1 worker
are now the limit, which layers 3 and 5 address).
