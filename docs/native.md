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

## Layer 3: translating renderer (design, not started)

Translate what the game asks the GS to draw into GPU work, instead of reproducing the GS. Generic: every PS2 game
talks to the GS through the same GIF/register interface, so nothing here is game-specific.

- **Placement:** written from scratch, no paraLLEl-GS code. It implements the GS frontend's `GSRasterBackend`
  interface, the one the CPU rasterizer implements: the frontend keeps decoding GIF and tracking GS state, and hands
  the backend decoded primitives (`Submit(GSPrimitiveBatch)`: vertices plus the full drawing state), CLUT loads,
  transfers, presentation, readbacks and VRAM access. Selected with `PS2X_GS_RENDERER=native`. Once it covers the
  games, paraLLEl-GS (and its LGPL dependency) leaves the project; the CPU GS stays as the validation reference.
- **API:** OpenGL 4.5 core on a context created on the GS thread and shared with raylib's window context (GLFW),
  so the frame the renderer produces is a GL texture the window draws directly: no CPU copy, and post-process
  passes are plain GLSL. (Vulkan was the alternative; it buys nothing here and needs interop to reach raylib.)
- **State:** comes decoded with each primitive (`GSDrawState`); the backend only notices changes to batch draws.
- **Geometry:** vertex kicks assembled into points/lines/triangles/strips/fans/sprites, batched while the state
  does not change; positions from the 12.4 fixed-point XY with the XYOFFSET, Z as 24/32-bit depth.
- **Render targets:** a cache keyed by FBP/FBW/PSM (and ZBP/PSM for depth) of GPU textures at an internal scale
  (1x first, then the player's choice). Overlaps and format reinterpretations resolve by copy passes on the GPU,
  page by page: each 8 KiB page of GS memory has a version bumped by every write (a draw, a transfer); GS memory and
  each target know the version they hold. A read of GS memory (another format, a transfer, a CLUT) first gets the
  pages a target holds newer written there; a target drawn into or sampled first reloads its older pages from it.
- **Textures:** a CPU shadow of GS local memory takes the uploads (swizzle via `ps2_gs_memory`); textures are
  decoded from it per TBP/TBW/PSM/CLUT and cached on the GPU, invalidated by uploads that overlap them. A texture
  whose TBP is a live render target samples that GPU texture directly (feedback effects: blur, bloom, DOF).
- **Shading:** one shader family with specialisation by state: TFX (modulate/decal/highlight/highlight2), TCC,
  texture function, alpha test and its fail modes, fog, FBA, dithering. GS blending (A-B)*C+D maps to fixed-function
  blending for the common cases; the rest uses a framebuffer copy and a shader blend. With COLCLAMP off, a blend that
  adds or subtracts two colours runs on a float copy of the target and is written back modulo 256, as the GS keeps
  the low 8 bits (Soulcalibur III counts its shadow volumes that way).
- **Z from a colour target:** a draw whose ZBUF is another target's colour tests against that colour read as Z, in
  the Z format's block order (Z16S blocks are the CT16S ones XOR 24), so a mask copied through a Z-format texture
  read comes back in place.
- **Transfers:** host-to-local writes the shadow and invalidates; local-to-local copies run on the GPU;
  local-to-host (readbacks, e.g. VP2's per-frame StoreImage) reads the render target back (one sync per readback,
  later async with a frame of latency where the game allows).
- **Presentation:** the PCRTC merge (DISPFB1/2, DISPLAY1/2, PMODE, BGCOLOR) as a GPU pass from the render targets
  into the final frame texture, then the post-process chain, then the window.
- **Validation:** GS recordings (`PS2X_GS_RECORD`, `ps2_gs_replay`) replayed through both the CPU reference and the
  native renderer; images compared, not VRAM bytes (the native renderer is allowed to differ by rounding, filtering
  and resolution).

Order: state tracker, geometry and untextured draws into one render target at 1x; textures and CLUTs; blending
and tests; render target cache and feedback textures; transfers and readbacks; PCRTC presentation; scale factor;
post-process hook.

## Microprograms an adapter does not translate (native/renderer)

A game's adapter turns its main microprograms into its own data (meshes, matrices, lights) and draws them with
`ps2x::native::Renderer`; VU1 skips them (`setVu1Gate`). The others (particles, billboards, effects) can still be
native: `ps2x::native::Vu1Sandbox` (`runtime/native/native_vu1_sandbox.h`) runs them on a VU1 of the adapter's own,
on its copy of VU1 code and data memory (filled from the VIF1 stream with `ps2x::native::vifUnpack`, VIF1's own
UNPACK), on the game's thread in stream order, and `GifToNative` (`runtime/native/native_gif.h`) draws what they send
the GS: it keeps the GS registers drawing depends on and submits the primitives in screen space. The adapter's gate
then skips every program, so the whole VU1 part of the frame is native. Without it, with renderers that keep GS memory
themselves (paraLLEl, CPU), what VU1 still draws is under the native frame: covered by it, or showing it through.

- An MSCAL the adapter does not draw: `call(code, data, pc, TOP, ITOP)` under its own group (`beginGroup`); its
  MSCNTs: `resume(...)`; an MPG: `codeChanged()`; a DIRECT packet: `gif().registers(...)` (its register writes, the
  GS draws the packet).
- `VU1Interpreter::setPacketsToObserverOnly` sends the sandbox's XGKICK packets to its observer only.
- Native draws are clipped to the GS scissor (and XYOFFSET) where their group's marker reaches the GS, on every GS
  renderer (paraLLEl-GS gives its own registers): a rear-view camera's strip, split screens.
- TMB: `TMB_NATIVE_SANDBOX=0` keeps them on VU1.

## IOP modules recompiled (native/renderer)

The IOP's own code (IRX modules: sound drivers, streaming, the game's helpers) runs as native C++ instead of being
interpreted, the way the EE's code is recompiled.

- `PS2X_IOP_DUMP_MODULES=<dir>`: each module as the IOP loaded it, relocated at its base and before its start
  routine, as `<name>_<base>.bin` plus `.json` (base, size, entry, gp, the file's FNV-1a 64 hash).
- `tools/iop_recomp.py <module>.json --out <dir>` writes `<module>_iop.cpp`: functions from the entry, the export
  tables and the JAL targets; branch delay slots inline; resume points at the entry, after each call and at loop
  heads. SYSCALL, BREAK (gcc's divide-by-zero trap), COP0 writes, RFE and unknown words hand back to the
  interpreter at that instruction; a load whose register the next instruction reads (an R3000 load delay) leaves
  its function interpreted. `--list` shows functions, exports, imports and what stays interpreted.
- Compile the `.cpp` into the game (it registers itself: `ps2x/iop/iop_native.h`); `PS2X_IOP_NATIVE=1` runs it
  for a loaded module with the same file hash at the same base, everything else interpreted as before.
- Native code reads RAM directly and goes through the IOP for writes and hardware, keeping the IOP clock up to date
  (each instruction counts one, as interpreted, so timers, the SPU2 and the scheduler see the same time). Calls run
  native functions and import stubs (the existing HLE dispatch); a blocking import or the end of the time slice
  stops at a resume point and the IOP resumes the thread there later.
- TMB: 989snd (282 functions) and libsd (54) all native; match at 49.8 fps, sound identical to the interpreter
  where the runs are the same.

## Sound on the wall clock (native/renderer)

`PS2X_SPU2_WALLCLOCK=1`: the SPU2 mixes when the host's audio callback asks for frames, at 48 kHz of real time, as
the console's SPU2 runs on its own clock whatever the CPUs do. Voices, envelopes, the AutoDMA input and reverb
advance by the frames played, not by IOP cycles or the game's frame rate; transfers that are not AutoDMA still end
on the IOP clock (`PS2X_SPU2_DMA_CYCLES_PER_2BYTES`), and the interrupts the mixing raises reach the IOP at its next
advance (within 1 ms of IOP time). Register and DMA accesses and the mixing share a lock. Without the switch the
SPU2 mixes a sample per 768 IOP cycles and the host stretches the output to its rate, as before.
