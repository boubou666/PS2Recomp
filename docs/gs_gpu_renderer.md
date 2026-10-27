# GPU GS renderer: design and plan

Status: phases 0 (groundwork) and 1 (CPU reference fixes) done, see 7.1 and 7.2; phase 2 next. Branch
`gs/gpu-renderer`.

Goal: a GPU implementation of the PS2 Graphics Synthesizer, written entirely in this repository, that replaces
paraLLEl-GS (LGPL-3.0+, Arntzen Software) once it matches it on every port we ship, and that matches the CPU
rasterizer (`gs_cpu_backend.cpp`) byte for byte in VRAM.

## 1. Summary of the decisions

| Question | Decision |
| --- | --- |
| Approach | Compute-shader rasterizer over an exact copy of GS local memory (4 MiB, swizzled, in a GPU storage buffer). No render-target or texture cache that owns pixels, no hardware blending, no game hacks. |
| API | Vulkan 1.2 through `volk` (runtime loader) and Khronos `Vulkan-Headers`, with our own thin wrapper (no Granite). GLSL compiled to SPIR-V offline, the SPIR-V committed as a generated header. |
| Placement | Lives on the GS thread behind the same interface the paraLLEl adapter uses today (raw GIF packets in, `PresentationFrame` out), selected by `PS2X_GS_RENDERER=gpu`. |
| Reference | The CPU backend in synchronous mode. Where it is known to differ from the GS (section 6.2), the CPU backend is fixed first, then the GPU follows. |
| Proof of parity | `ps2_gs_replay --renderer gpu --compare cpu`: same trace through both, VRAM compared byte for byte at a fixed command interval, first divergence reported down to the pixel. Plus differential fuzzing and per-feature unit tests, run on lavapipe (Mesa CPU Vulkan) in CI. |

## 2. What exists today

- `gs_frontend.cpp` / `gs_frontend.h` (`class GS`): owns the GS thread and the packet queue (`QueuedCommand`:
  `GifPacket`, `NativePackedGifPacket`, `NativeImageUpload`, `PresentationLatch`, `Register`, `Call`). At queue
  time, on the EE side, `postQueuedCpuEvents` scans each packet for SIGNAL / FINISH / LABEL and posts them to CSR
  with the `GsDrawEstimate` delay, so the EE never waits for drawing to see them. FINISH also calls
  `waitForGsFrames`, which lets the EE run at most one frame ahead of the GS thread
  (`m_finishesQueued - m_finishesDrawn <= 1`). `m_finishesDrawn` is advanced by the renderer when it reaches the
  FINISH in the stream.
- With the CPU backend the frontend decodes GIF itself (`processGIFPacket`, `writeRegisterPacked`,
  `vertexKick`) and hands the backend one `GSPrimitiveBatch` per primitive (three vertices plus a full copy of
  the drawing state), plus `LoadClut`, `BeginTransfer` / `UploadImage`, `TextureFlush`, `Present`.
- `gs_cpu_backend.cpp` (4.7 kLOC): all PSMs through `ps2_gs_memory` swizzle tables, CLUT with CSM1/CSM2 and a
  serial-based palette cache, decoded-texture cache, alpha / destination alpha / depth tests, blending with PABE,
  FBA, FBMSK, fog, region clamp/repeat, bilinear, sprites / triangles / lines, local-to-local and local-to-host
  transfers, CRTC merge, interlace. Optional asynchronous raster workers that own VRAM blocks and keep per-word
  write order, with barriers driven by block ranges. It is the accuracy reference.
- `gs_parallel.cpp` / `gs_parallel.h` (`GSParallelRenderer`): created on the GS thread when
  `PS2X_GS_RENDERER=parallel` or a game calls `GS::requestParallelRenderer()`. The frontend bypasses its own
  decoder (`processWithParallel`) and passes raw GIF packets (`gifTransfer`), queued register writes
  (`writeRegister`), native uploads repacked as an IMAGE packet, presentation latches (`present` with the
  privileged registers captured at queue time, returning a 640x512 RGBA host frame), local-to-host reads
  (`readTransferFifo`) and VRAM snapshots (`readVram`). It flushes to the GPU when the queue drains, at FINISH
  and at presentation.
- Presentation: `GS::copyLatchedHostPresentationFrame` gives the main thread the latched RGBA frame,
  `ps2_runtime.cpp` uploads it with raylib `UpdateTexture` and draws it with `DrawTexturePro` (OpenGL context
  owned by the main thread).
- Tooling: `PS2X_GS_RECORD` / `_AT` / `_SECONDS` write `GSR1` traces (VRAM + `saveDrawingState` + raw queued
  commands); `ps2_gs_replay` replays them (`--list`, `--until N`, `--dump NAME=FBP,FBW,PSM,W,H`, VRAM hash,
  `PS2X_GS_REPLAY_CHECK` workers-vs-caller comparison). `PS2X_GS_SHADOW_CHECK` runs a second synchronous GS and
  compares VRAM in game. `PS2X_GS_DUMP*` dumps primitives and textures. `ps2xTest/src/ps2_gs_tests.cpp` has
  ~60 GS cases (transfers, swizzle, CLUT, tests, presentation, async ordering).

Gaps of the paraLLEl integration as it was before phase 0, which the new renderer must close, not copy (phase 0
closed the first, the second for `ReadVram` / `WriteVram`, and the third):

- Recording was disabled when paraLLEl is active (`!m_parallel` in `workerLoop`), so no trace could be taken
  from the fast path. Now a stream renderer records too: VRAM from `readVram`, state from its `saveState`,
  starting at the first presentation latch after `PS2X_GS_RECORD_AT` (it cannot tell whether an image transfer is
  in progress, and games have none across a latch).
- `GS::ReadVram` / `WriteVram` / `clearFramebufferContext` went to the CPU backend (or were skipped) when
  paraLLEl owns VRAM, so any HLE path using them saw stale memory. `ReadVram` / `WriteVram` now go to the stream
  renderer on the GS thread (`readVramPixel` / `writeVramPixel` through `runOnWorker`). `clearFramebufferContext`
  stays skipped with a stream renderer: its only caller (`sceGsSwapDBuffDc`) sends the clear packet right after,
  which draws the same clear, and clearing from the host would add a GPU round trip per frame to SC3 for nothing.
  Phase 6 revisits it if another caller appears.
- The trace's drawing state was the frontend's and the CPU backend's private serialization, so a trace could
  not seed a renderer that decodes GIF itself. `GSR2` traces carry a `GSStateImage`.

## 3. Approach: compute over exact VRAM

### 3.1 Options

(a) Compute rasterizer. GS local memory is one GPU buffer with the exact byte layout the CPU backend uses.
Every write (pixels, Z, uploads, copies, CLUT) lands in it at its swizzled address; every read (textures, CLUT,
display, readbacks) comes from it. Rasterization, the pixel pipeline and blending run in compute shaders.

(b) Hardware-pipeline renderer (PCSX2 HW style). Render targets and depth buffers are GPU images keyed by
(base pointer, width, format); textures are decoded into a cache; blending maps to fixed-function blend
where possible, with framebuffer fetch / barriers / multi-pass emulation where not; VRAM coherency is
reconstructed by tracking which cached object covers which pages and converting between them.

### 3.2 Recommendation: (a)

1. Accuracy-first policy. (a) has one source of truth with the GS's own layout, so "the GPU did what the GS
   does" is checkable by comparing 4 MiB of bytes with the CPU reference. (b) never has an exact VRAM image: its
   state is spread over render targets, depth images and a texture cache, and its correctness is a property of
   heuristics (target lookup, format conversion, invalidation), which is the opposite of a byte-exact contract.
2. SC3's workload is what (b) is worst at. Bloom and feedback passes sample the frame buffer that is being drawn
   (render target as texture, often with a different PSM or a half-size BW), the shadow buffer is a depth or
   color target reused as a texture, and many small strips change state constantly. In (b) each of these is a
   cache lookup, a possible format conversion, a possible copy of the target and a pipeline change; in (a) a
   texture fetch from a page written earlier is just a read from the same buffer after a barrier.
3. Format aliasing is free in (a): CT32 written then sampled as T8/T4/T8H/T4HL/T4HH, Z buffers read as color,
   24-bit targets whose top byte is used as an 8-bit texture, CLUT data rendered by primitives. Today's tests
   already pin several of these ("CT32 upload aliases cleanly into later PSMT8 sampling", "asynchronous
   rasterization loads a CLUT drawn by queued primitives"). In (b) each needs a dedicated conversion path.
4. GS blending (`((A - B) * C >> 7) + D` with source, destination or zero for A/B/D, source alpha, destination
   alpha or FIX for C, COLCLAMP clamp or wrap, PABE, FBA, FBMSK, DATE, alpha test fail modes that write only
   some channels, dithering, 16-bit packing) is not expressible in fixed-function blending. (b) needs
   framebuffer fetch or per-primitive barriers for the general case, which is (a) with extra steps.
5. Synchronisation and readbacks are simple in (a): local-to-host and `readVram` copy bytes out of one buffer;
   in (b) they require resolving every cached target that overlaps the region first.
6. Performance is not a reason for (b) at native resolution. The PS2 draws at most ~640x512, a few hundred
   thousand pixels per pass; a mid-range GPU fills that in compute with room to spare. paraLLEl-GS (approach
   (a)) already holds SC3 fights at 60 fps on our dev machines, so the approach has been shown to meet the
   target; the remaining risk is our implementation, not the approach.
7. Upscaling is the main thing (b) gives that (a) does not. It is out of scope; (a) can later add
   supersampling the way paraLLEl does, without changing VRAM semantics.

Cost of (a): per-pixel work is in shaders we write (no ROPs, no early-Z hardware), primitive order must be kept
explicitly (tile binning, section 5.5), and hazards between "this batch writes page P" and "a later primitive
samples page P" must be tracked on the CPU (section 5.4).

## 4. Graphics API: Vulkan

### 4.1 Options

| | Vulkan 1.2 (volk) | OpenGL 4.3/4.5 compute | Others |
| --- | --- | --- | --- |
| New dependencies | `volk` (MIT, one .c) + `Vulkan-Headers` (Apache-2.0), fetched like `fmt` / `elfio`. No link-time dependency: `vulkan-1.dll` / `libvulkan.so.1` is loaded at run time. | None (raylib's context) | D3D12: Windows only. WebGPU (Dawn/wgpu): heavy build, Rust or Chromium toolchain. CUDA/OpenCL: vendor or driver lottery, no presentation story. |
| Threading | The GS thread owns its own instance/device/queue, independent of raylib. | A GL context is current on one thread. The GS thread would need a second context shared with raylib's, created through raylib's private GLFW (`rglfw.c`), with WGL sharing on Windows. Fragile and outside raylib's API. | |
| MSVC / MinGW | volk and the headers are plain C/C++; paraLLEl's Vulkan already builds on both. | Fine | |
| Sync model | Explicit barriers, timeline semaphores (a monotonically increasing value per submission maps directly onto page tracking and host-read waits). | `glMemoryBarrier` + `glFenceSync`; implicit driver hazards, hard to reason about for 4 MiB of aliased storage. | |
| Shader features | Subgroup ops, 8/16-bit storage, `shaderInt64`, `NoContraction` honoured, specialization constants, SPIR-V compiled once offline. | Subgroups only through extensions (missing on several Windows drivers), GLSL compiled by each driver at run time with its own bugs. | |
| Compute quality on Windows | Good on NVIDIA, AMD, Intel. | Historically weak compute and memory-barrier bugs on AMD and Intel Windows GL drivers. | |
| Debugging | Validation layers (incl. sync validation and GPU-assisted validation), RenderDoc, Nsight, Radeon GPU Profiler, timestamp queries. | RenderDoc (GL compute support is weaker), `KHR_debug`. | |
| CI | lavapipe (Mesa's CPU Vulkan driver) runs the same shaders on a GPU-less Linux runner. | llvmpipe GL 4.5 also exists. | |
| Platforms | Linux, Windows; Android possible; Apple via MoltenVK later. | Linux, Windows; Apple stuck at GL 4.1 (no compute). | |

### 4.2 Recommendation: Vulkan 1.2

The decisive reasons are threading (the renderer lives on the GS thread; GL would force a shared context hack
under raylib), explicit synchronisation for a design whose correctness is about ordering writes and reads to
one buffer, and shader determinism (offline SPIR-V with `NoContraction`, no per-driver GLSL compiler). The
dependency cost is two small fetched projects and no new runtime requirement beyond what paraLLEl already
demands of users, and it drops Granite and paraLLEl's patch step when paraLLEl is retired.

Required device features (checked at creation, CPU fallback otherwise, as paraLLEl does today): Vulkan 1.2,
timeline semaphores, `storageBuffer8BitAccess`, `storageBuffer16BitAccess`, `shaderInt16`, subgroup basic +
ballot + arithmetic in compute. `shaderInt64` is optional (used only if section 6.3 needs it). No descriptor
buffers, no mesh shaders, no ray tracing.

Shaders: GLSL 4.60 under `ps2xRuntime/src/lib/gs/gpu/shaders/`, compiled with `glslangValidator` (or `glslc`)
by a CMake target that is only enabled when the compiler is found (`PS2X_GS_GPU_COMPILE_SHADERS`). Its output,
a header of `uint32_t` arrays, is committed, so ordinary MSVC / MinGW / Linux builds need no shader toolchain. A
CI job regenerates it and fails on a diff.

Presentation stays as today: the renderer composes the displayed picture on the GPU, copies the 640x512 RGBA
frame into a host-visible buffer and returns it as a `PresentationFrame`; `ps2_runtime.cpp` is unchanged. The
copy is 1.3 MiB per frame (well under a millisecond). Zero-copy interop with raylib's GL texture
(`GL_EXT_memory_object` + `VK_KHR_external_memory`) is an optional later step, not part of parity.

## 5. Architecture

### 5.1 Plugging into the frontend

- Mechanical first: rename `GSParallelRenderer` to an interface `GSStreamRenderer` (raw GIF in, frames out,
  owned by the GS thread) in `gs_stream_renderer.h`; `gs_parallel.cpp` becomes one implementation, the new
  `GSGpuRenderer` another. `processWithParallel`, `m_parallel*`, `requestParallelRenderer` become
  `processWithStreamRenderer`, `m_streamRenderer`, `requestStreamRenderer(kind)` with
  `requestParallelRenderer()` kept as a thin wrapper until the games' overrides are updated.
- `PS2X_GS_RENDERER=gpu` selects ours, `parallel` keeps paraLLEl, anything else or unset keeps the CPU backend.
  A game override may request `gpu` the same way SC3 requests `parallel` today.
- Interface additions (also implemented by the paraLLEl adapter where it can):
  - `uploadImage(bitbltbuf, trxpos, trxreg, trxdir, bytes)`: native uploads without repacking into an IMAGE
    packet (the paraLLEl adapter now does the repacking the frontend used to), and `nativePackedGifTransfer` for
    the frontend's validated PACKED packets (a plain `gifTransfer` unless the renderer has a fast path).
  - `loadState(const GSStateImage &)` / `saveState(GSStateImage &)`: a renderer-neutral image of the GS (raw
    64-bit register file for both contexts and the general registers, PRMODE/PRMODECONT, the vertex queue,
    the 1 KiB CLUT buffer and its CBP cache), used by traces and by `setRasterBackend`-style renderer switches.
    An in-progress transfer is not part of it: images are taken between packets with no transfer in progress
    (the recorder's rule), which keeps every renderer's loader simple. A renderer that cannot see a part leaves
    its flag clear: paraLLEl keeps its vertex queue and CLUT buffer private, so its images have neither (the CBP
    cache is there), and PRIM holds PRMODE's attributes when PRMODECONT.AC is 0, as paraLLEl folds them. Loading
    into paraLLEl writes the registers in an order that keeps their side effects out (TEX0 without CLD, MIPTBP
    after TEX0, TRXDIR set without starting a transfer), rebuilds the queue with XYZ3 writes, and refills the
    CLUT buffer by having paraLLEl load the image's CLUT as a CSM1 CT32 palette written to page 0, then puts
    the page back.
  - `writeVram` (all of VRAM, to seed a replay), `readVramPixel` / `writeVramPixel`: so `GS::ReadVram` and
    `WriteVram` work on whichever renderer owns VRAM (through `runOnWorker`). `clearFrame` is not added in
    phase 0 (see the gaps in section 2).
- The CPU renderer is a `GSStreamRenderer` too (kind `cpu`: a synchronous `GS` with the CPU backend over its own
  VRAM), so tools and tests drive every renderer the same way. The frontend never hands its stream to it: in
  game the CPU path stays the frontend's own decoder and backend, untouched, and `PS2X_GS_RENDERER=cpu` keeps
  meaning that.
- Unchanged: SIGNAL / FINISH / LABEL are still posted EE-side at queue time by `postQueuedCpuEvents`; the
  renderer only reports FINISH in stream order through the `finish` callback that advances `m_finishesDrawn`,
  so the one-frame lag limit and the CSR timing model stay exactly as with paraLLEl. SIGNAL never forces a GPU
  submission (SC3's ~30 handshakes per frame cost nothing on the GPU side).
- Submission points: queue drained (as `PS2X_GS_PARALLEL_FLUSH`, renamed `PS2X_GS_STREAM_FLUSH`), FINISH,
  presentation latch, any host read (local-to-host, `readVram`, `ReadVram`), batch overflow.

### 5.2 Module layout

```
ps2xRuntime/include/runtime/gs/gs_stream_renderer.h   interface shared by paraLLEl and ours
ps2xRuntime/include/runtime/gs/gs_state_image.h       renderer-neutral GS state (traces, switches)
ps2xRuntime/src/lib/gs/gs_gif_decoder.{h,cpp}         GIF tags, PACKED/REGLIST/IMAGE, A+D, register file,
                                                      vertex queue and kicks (no rendering)
ps2xRuntime/src/lib/gs/gpu/
  gpu_renderer.{h,cpp}        GSStreamRenderer implementation, owns everything below
  vk/vk_context.{h,cpp}       volk init, instance/device/queue selection, feature checks, debug utils
  vk/vk_buffers.{h,cpp}       device buffers, host-visible rings, staging, readback buffers
  vk/vk_pipelines.{h,cpp}     compute pipelines from the embedded SPIR-V, descriptor layout, push constants
  vk/vk_submit.{h,cpp}        command buffer pool, timeline semaphore, submission and host waits
  page_tracker.{h,cpp}        per page / per block read and write state, hazard queries, host-read timelines
  draw_batcher.{h,cpp}        primitive records, state table, CLUT snapshots, batch boundaries
  transfers.{h,cpp}           host-to-local, local-to-local, local-to-host scheduling
  presenter.{h,cpp}           CRTC merge dispatch and frame readback
  shaders/
    gs_common.glsl            VRAM addressing for every PSM (block/column/page swizzle), pixel pack/unpack
    gs_texture.glsl           texel fetch, CLUT lookup, TEXA, wrap modes, bilinear, LOD
    gs_pixel.glsl             fog, alpha test, DATE, depth test, blend, dither, COLCLAMP, FBA, masks, packing
    upload.comp               host-to-local
    copy.comp                 local-to-local
    readback.comp             local-to-host packing
    clut_load.comp            CLUT buffer loads (CSM1/CSM2, CSA, CT32/CT16 palettes)
    setup.comp                primitive setup (edge equations, attribute gradients, bounding boxes)
    binning.comp              tile lists per batch
    raster.comp               per-tile rasterizer + pixel pipeline
    present.comp              DISPFB/DISPLAY read, PMODE merge, interlace
  shaders_spirv.h             generated, committed
ps2xRuntime/tools/ps2_gs_replay.cpp                   gains --renderer / --compare (section 6)
ps2xTest/src/ps2_gs_gpu_tests.cpp                     differential tests and fuzzing
```

`gs_gif_decoder` is new shared code: the GPU renderer needs a GIF decoder, and having a second private one
next to the frontend's would let them drift. The frontend keeps its decoder at first; a differential test
feeds the same packets to both and compares the primitives and transfers they produce. Phase 0 built it as
`GSGifDecoder`: its state is a `GSStateImage` (raw register values, TEX2 folded into TEX0), and a sink receives
primitives (raw vertices, PRIM as drawn), every A+D / REGLIST register write after it took effect, and image
data. The differential test (`PS2GSState` in `ps2xTest/src/ps2_gs_state_tests.cpp`) runs seeded random streams
through both and compares every backend call (primitives with their full draw state, CLUT loads, transfer
starts, image data, TEXFLUSH) and the final state. Moving the CPU path
onto the shared decoder is a later, separate cleanup, not on the critical path.

### 5.3 VRAM model

- One 4 MiB device-local storage buffer, byte-identical to the CPU backend's `m_vram` (same swizzle tables,
  ported to GLSL from `ps2_gs_memory` and `ps2_gs_psm*.h`, which are ours). Accessed as `uint32` words, with
  8/16-bit views where a PSM writes sub-words (CT16, T8, T4, the H formats, Z16, CT24/Z24 keeping the top
  byte).
- GPU memory is authoritative. There is no persistent host mirror; host reads go through readback (section
  5.7). `GS::m_localMemoryStorage` is filled from a readback when the renderer is switched or torn down, as
  `setRasterBackend` already does.
- The page tracker (CPU side, GS thread) keeps, for each of the 512 pages of 8 KiB, 32-bit block masks
  (32 blocks of 256 bytes): blocks written by the open batch as color, as Z, by copies or uploads; blocks
  sampled or CLUT-loaded by the open batch; and the timeline value of the last submission that wrote each page.
  CT24 / Z24 writes and the T8H / T4HL / T4HH reads are tracked as "low 24 bits" and "top byte" so they do not
  conflict with each other, as the CPU backend's `m_topByteWritesSinceBarrier` already does.
- Hazards decide batch boundaries:
  - a primitive whose texture or CLUT source blocks were written earlier in the open batch closes the batch
    (render-to-texture, bloom, feedback), unless the overlap is only "different byte lanes";
  - an upload, copy or CLUT load touching blocks the open batch reads or writes closes it, or is ordered into
    the batch as a command when the batch format allows it (section 5.5);
  - a FRAME / ZBUF change does not close a batch by itself; the batch keeps a short list of targets.
  Closing a batch records a pipeline barrier, not a submission.

### 5.4 Primitive path

1. GIF decode (CPU, GS thread): `gs_gif_decoder` applies registers and produces vertex kicks. Each drawing kick
   appends a primitive record: up to three vertices in raw GS formats (XY 12.4 fixed with XYOFFSET applied,
   Z 32-bit, RGBAQ with Q as the raw float bits, ST floats, UV 14.4 fixed, fog byte), a state index and a
   CLUT snapshot index. Raw values, no float conversion on the CPU: the GPU then sees exactly the bits the GS
   saw.
2. State table: the drawing state (FRAME, ZBUF, TEST, ALPHA, FBA, TEX0/1/2, CLAMP, MIPTBP, SCISSOR, TEXA,
   FOGCOL, DTHE/DIMX, COLCLAMP, PABE, SCANMSK, PRIM bits) is hashed per primitive and deduplicated into a
   per-batch table, so strips that change nothing reuse one entry. Specialization stays dynamic (uniform per
   primitive, branches are coherent within a tile); a small number of specialized raster pipelines (e.g.
   "untextured opaque", "textured no blend", "general") can come in phase 6 if profiling asks for it.
3. Setup on the GPU (`setup.comp`, one thread per primitive): culling (degenerate, outside scissor), bounding
   box in pixels, edge equations and attribute gradients in fixed point following the GS rules (12.4
   positions, top-left fill, sprite and line rules), the same as the CPU reference after section 6.2. Setup on
   the GPU keeps the GS thread light for SC3's many small strips; the CPU only appends records.
4. CLUT: the 1 KiB CLUT buffer is GS state that changes between primitives (games reload it before almost every
   primitive). Each CLUT load executes in stream order: `clut_load.comp` reads VRAM into a new snapshot slot of a
   per-batch CLUT array (CLD conditions and CBP0/CBP1 caching evaluated on the CPU, like the CPU backend), and
   later primitives reference that slot. A load whose source blocks the open batch writes closes the batch
   first. Repeated identical loads (same TEX0 CLUT fields, no write since) reuse the previous slot, as the CPU
   backend's `m_lastClutLoadValid` does.

### 5.5 Batching, binning, rasterization

- A batch is an ordered list of primitives (and in-batch CLUT snapshots) with no hazard between them, capped
  (e.g. 16k primitives or a buffer size) to bound memory.
- Binning (`binning.comp`): the screen is cut into 16x16 pixel tiles in each target's coordinate space; for each
  tile, a bitmask over the batch's primitives in submission order (one bit per primitive, built with subgroup
  ballots from bounding boxes and, for large triangles, an exact edge test against the tile).
- Raster (`raster.comp`): one workgroup per non-empty tile, one invocation per pixel (256). The workgroup walks
  the tile's primitive bits in order; each invocation evaluates coverage, interpolates, samples, runs the pixel
  pipeline and keeps its color and Z in registers, reading VRAM for the destination only when the tile is first
  touched for that target and writing back when the target changes or the batch ends. Each pixel is owned by
  exactly one invocation, so per-pixel primitive order is exactly the GS's, with no atomics.
- Several targets in one batch (e.g. drawing to a shadow buffer then to the frame) are handled by binning per
  target and running the target groups in order; aliasing between two targets of one batch closes the batch.
- Before binning exists (phase 2), the raster shader runs one dispatch per primitive over its bounding box:
  slow but trivially ordered, and the reference for the binned path.

### 5.6 Pixel pipeline (shader side)

In order, per covered pixel, mirroring `GSCpuBackend::WritePixel` after the section 6.2 fixes:

- SCANMSK (skip even or odd lines), scissor.
- Color: flat (IIP=0, last vertex) or Gouraud; Z interpolation with the GS precision; fog coefficient.
- Texture (TME): STQ (perspective, `S/Q`, `T/Q`) or UV (FST); LOD from Q or fixed (LCM, L, K), MXL and MIPTBP1/2
  levels; MMAG / MMIN nearest or bilinear with the GS's weight precision; CLAMP REPEAT / CLAMP / REGION_CLAMP /
  REGION_REPEAT; all PSMs (CT32, CT24, CT16, CT16S, T8, T4, T8H, T4HL, T4HH, Z32, Z24, Z16, Z16S); CLUT CT32 /
  CT16 with CSA; TEXA (TA0, TA1, AEM) for 16/24-bit sources; TCC; TFX MODULATE / DECAL / HIGHLIGHT / HIGHLIGHT2.
- Fog (FGE) toward FOGCOL.
- Alpha test (8 ATST methods) with AFAIL KEEP / FB_ONLY / ZB_ONLY / RGB_ONLY (RGB_ONLY on CT32 keeps
  destination alpha, on 16-bit formats behaves as FB_ONLY).
- Destination alpha test (DATE, DATM) on CT32 / CT16 / CT16S.
- Depth test (ZTE, ZTST) against Z32 / Z24 / Z16 / Z16S, Z write unless ZMSK or AFAIL forbids it, with the
  Z saturated to the format's range first (section 6.2).
- Blend (ABE, PABE, ALPHA A/B/C/D, FIX) with the 24-bit destination alpha of 0x80, then dither (DTHE, DIMX 4x4
  matrix, 16-bit frames only) and COLCLAMP (clamp or wrap to 8 bits), as section 6.2 specifies.
- FBA, FBMSK, pack to CT32 / CT24 / CT16 / CT16S, write.
- AA1 is ignored (as the CPU backend does); PRMODE / PRMODECONT are resolved by the decoder before the record.

### 5.7 Transfers

- Host-to-local (`BITBLTBUF` / `TRXPOS` / `TRXREG` / `TRXDIR=0`, IMAGE data or HWREG, native uploads): bytes are
  copied into a host-visible ring buffer; `upload.comp` writes them with the destination PSM's swizzle,
  including CT24 (3 bytes per pixel), T4 nibbles, the H formats' byte lanes and transfers split across several
  IMAGE packets (the pixel cursor is kept on the CPU, each chunk is a dispatch with its start pixel). Upload is
  ordered into the stream as a batch command, closing the batch only on a real hazard.
- Local-to-local (`TRXDIR=2`): `copy.comp`, PSM to PSM. When the source and destination rectangles overlap, the
  copy runs row by row in the order the GS walks them (DIR bits), which is what the CPU reference does; a test
  pins the overlapping case.
- Local-to-host (`TRXDIR=1`): the batch is submitted, `readback.comp` packs the rectangle into a readback
  buffer, the GS thread waits for that submission's timeline value and serves `readTransferFifo` (and the
  frontend's `consumeLocalToHostBytes`) from it. EE readbacks stall the GS thread by design: they are rare and
  the EE is waiting anyway.
- `readVram` / state snapshots: a full-buffer copy into a readback buffer, waited for.

### 5.8 Presentation

`present.comp` reproduces `GSCpuBackend::PresentFromLocalMemory`: DISPFB1/2 (FBP, FBW, PSM CT32 / CT24 / CT16 /
CT16S, DBX/DBY), DISPLAY1/2 (DX/DY, MAGH/MAGV, DW/DH), PMODE (EN1/EN2, MMOD, AMOD, SLBG, ALP), BGCOLOR,
SMODE2 field / frame interlace and odd/even field from `vsyncTick`, into the 640x512 host frame with the same
width and height rules. The CPU path also has non-GS heuristics (`preferredSource` from display-copy tracking,
falling back to a non-black context frame when DISPFB points at page 0); they are ported as-is for parity and
listed for review in section 9.4, since paraLLEl shows scanning out DISPFB alone works for SC3.

The result is copied into one of two host-visible frame buffers. The latch waits for the previous frame's copy,
not the current one, so presentation does not stall the GS thread on the GPU (one frame of latency, as
`waitForGsFrames` already allows); `PS2X_GS_GPU_PRESENT_SYNC=1` restores the wait for diagnosis.

### 5.9 Diagnostics

- `PS2X_GS_GPU_VALIDATE=1`: validation layers with synchronization validation.
- `PS2X_GS_GPU_LOG=1`: batch boundaries with their flush reason (texture hazard, copy hazard, CLUT hazard, host
  read, overflow, FINISH, present), per frame counts, GPU timestamps per pass.
- Debug-utils labels per batch and per pass for RenderDoc.
- `PS2X_GS_RECORD` works with the GPU renderer (VRAM from a readback, state from `saveState`), so traces can be
  taken at full speed.

## 6. Verification

### 6.1 Rule

The CPU backend, run synchronously (no async raster workers), is the reference. The GPU renderer must leave
VRAM byte-identical to it after every replayed command interval, and present byte-identical frames. A tolerance
is allowed only where the GS itself is undefined, each one written down in this document with the test that
shows it, and applied only to the bytes concerned (never "within N per channel" globally).

### 6.2 Fix the reference first

Reading `gs_cpu_backend.cpp` showed places where it was not yet the GS. Each was fixed in the CPU backend first, as
its own layer with a unit test (phase 1, section 7.2), and only then is implemented on the GPU, so the two never
agree on a known bug. What the CPU backend does now, and the GPU must do byte for byte:

- Rasterization (the written spec). A vertex's window position is its 12.4 XYZ value minus XYOFFSET, kept in
  1/16 pixel units (subpixels). Pixel (x, y) is sampled at subpixel (16x, 16y): whole pixel positions, not centres
  at +0.5.
  - Triangles: edge functions in 64-bit integers over subpixels, oriented so the inside is positive. A sample on an
    edge is covered only by a top edge (horizontal, inside below) or a left edge (inside to the right); a
    zero-area triangle draws nothing. Each row's covered span is solved exactly from the three edges.
  - Attributes of triangles are planes in fixed point anchored at the last vertex: gradient per subpixel =
    round-half-up(sum of (a_i - a_2) * edge coefficient * 2^F / twice the area), value at a sample =
    (a_2 << F) + gx * (X - X2) + gy * (Y - Y2) evaluated modulo 2^64 (only the sum at a covered sample has to fit).
    Colours, fog and UV: F = 16, value truncated, clamped to the attribute's range. Z: F = 12, value rounded half
    up, clamped to 0..2^32-1 then to the Z format (below). Flat shading takes the last vertex's colour. S, T and Q
    are floats, as the GS interpolates them: gradients computed in double and rounded to float once, value =
    (base + gx * dX) + gy * dY with each operation rounded to float (no fused multiply-add).
  - Sprites: covered pixels are [ceil(min / 16), ceil(max / 16)) on each axis, whichever vertex comes first, so a
    zero-width sprite draws nothing. Colour, Z and fog are v1's. UV run from v0's value at v0's position to v1's at
    v1's along each axis, as 16-fraction-bit fixed point (truncated); S and T the same way as floats; Q is v1's.
  - Lines: one pixel per whole step of the major axis (x when |dx| >= |dy|) in [ceil(min / 16), ceil(max / 16)),
    the minor coordinate interpolated in 16 fraction bits and rounded half up; attributes interpolated along the
    major axis the same way. Lines stay untextured, as before (a known gap, section 7.2).
  - Points: the position rounded half up, ((X + 8) >> 4, (Y + 8) >> 4), as paraLLEl-GS's notes report from
    TJnotJT's hardware tests.
- Fog: C = (F * C + (256 - F) * FOGCOL) >> 8, one lerp truncated. It keeps F = 255, FOGCOL = 128, C = 128 at 128
  (paraLLEl's reported hardware observation, which the manual's 255 - F form misses) and gives FOGCOL exactly at
  F = 0. Whether the GS rounds this lerp (paraLLEl adds one half) or truncates it (PCSX2's software and hardware
  renderers) has no hardware result yet; the truncated form is kept until one exists.
- Blend, dither, COLCLAMP: ((A - B) * C >> 7) + D unclamped, then the dither, then COLCLAMP.CLAMP = 1 clamps to
  0..255 and CLAMP = 0 keeps the low 8 bits (also for unblended pixels when dithering moved them). Dithering adds
  DIMX[y & 3][x & 3] (3-bit two's complement, x and y in frame pixels) to R, G and B, only when DTHE = 1 and FRAME
  is PSMCT16 or PSMCT16S, as PCSX2 does in both its renderers: the dither exists to hide the 16-bit pack's lost
  bits. paraLLEl also dithers 32-bit frames, which changes their low bits; SC3's traces never dither a 32-bit
  frame, so they cannot tell the two apart, and a hardware result should settle it.
- Z range: Z is saturated to the Z buffer format's maximum (0xFFFFFF for Z24, 0xFFFF for Z16 and Z16S) before the
  depth test and the write, as PCSX2's software renderer and paraLLEl do; it was compared at 32 bits and stored
  truncated.
- Mipmapping and the LOD. LOD = (log2(1 / |Q|) << L) + K in sixteenths of a level, K alone when LCM = 1 or FST = 1,
  log2 |Q| taken as (exponent - 127) + (top four mantissa bits) / 16. LOD <= 0 filters with MMAG, LOD > 0 with
  MMIN (nearest for 0, 2, 3; bilinear for 1, 4, 5). With MXL > 0 and MMIN 2..5, level n (n <= MXL, at most 6) reads
  TBPn / TBWn from MIPTBP1 (1..3) and MIPTBP2 (4..6), is 2^max(TW - n, 0) by 2^max(TH - n, 0), with UV and the CLAMP
  region shifted right by n; MMIN 2 and 4 take round-half-up(LOD) (clamped to MXL), 3 and 5 blend levels floor(LOD)
  and floor(LOD) + 1 by the LOD's sixteenths, c0 + ((c1 - c0) * fraction >> 4). MTBA (automatic MIPTBP1 on TEX0
  writes) is not implemented: no trace uses it.

### 6.3 Floating point

STQ division, bilinear weights and LOD are the only float paths. Vulkan does not require correctly rounded
division (OpFDiv may be 2.5 ULP) and may fuse multiply-adds unless told not to, so naive GLSL can differ from
the CPU in the last bit and change a texel choice. Mitigations, in order of preference: compute in the GS's
own fixed-point formats where the GS does (UV are 14.4, bilinear weights are 4-bit); for S/Q and T/Q use an
integer or correctly rounded reciprocal (Newton step with an exactness check) shared verbatim by the CPU
reference; mark the rest `precise` (SPIR-V `NoContraction`) and disable FMA contraction in the CPU build for
those functions. The differential fuzzer (6.5) exercises these paths specifically.

### 6.4 Trace replay

`ps2_gs_replay` gains:

- `--renderer cpu|gpu|parallel` to choose who replays (default `cpu`). Every renderer is replayed through
  `GSStreamRenderer`, the CPU one included (a synchronous GS with the CPU backend), so the tool has one replay
  path.
- `--compare cpu` (with `--renderer gpu`): replays on both, compares VRAM every N commands (`--every N`, default
  64; `--every 1` during bring-up), and on the first difference replays again comparing after every command of the
  failing interval (the bisect step is automatic), then prints the command (as `--list` does), how many words and
  pages differ, the first differing byte with both values, and which page-sized pixel rectangle of FRAME / ZBUF /
  TEX0 of either context (as the CPU reference holds them after that command) it falls in; `--dump` writes the
  buffer from both renderers as `NAME.<renderer>.ppm`. Phase 0 has this skeleton; decoding the exact pixel inside
  the page and printing the draw state come with phase 2.
- `--present` compares the presented frames at every latch.
- Timing per pass for both renderers (GPU time from timestamp queries).

Traces become renderer-neutral (`GSR2`: VRAM + `GSStateImage` instead of the frontend's private state), with a
converter for existing `GSR1` traces (load into the CPU path, save as `GSR2`): `ps2_gs_replay <in> --convert
<out.gsr2>`. A `GSR1` trace is also replayed directly, through the same conversion in memory. The command records
keep the `GSR1` layout (kind, four registers, payload) and gain two kinds: presentation latches (kind 3, the nine
display registers and the vsync tick as payload, for the `--present` comparison) and queued register writes
(kind 4, what `GS::writeRegister` queues for a stream renderer). `GSStateImage` serializes as a version, flags
(vertex queue present, CLUT buffer present), (address, value) pairs for every register it holds, the vertex queue
as the registers that made each vertex, then the CLUT buffer and CBP0/CBP1; all little endian.

Corpus: traces live outside the repository, under `~/Games/PS2/gs-traces/<game>/<scene>.gsr2`, with a
`manifest.toml` per game (scene, how to reach it, recording command, expected CPU VRAM hash). Recording:

```
PS2X_GS_RECORD=~/Games/PS2/gs-traces/sc3/battle_stage1.gsr2 PS2X_GS_RECORD_AT=<seconds> \
PS2X_GS_RECORD_SECONDS=2 PS2X_PAD_SCRIPT=<script reaching the scene> <port binary>
```

SC3 scenes to record first: title / menu, character select, a fight on a stage with bloom, a stage with
shadow buffers, Tales of Souls text, a stage transition (copy + clear frames). Then two scenes per other port
(TMB menus with both circuits, VP2 overlay screens, Drakengard boot frame copies, TLOTR, FFOP, SCTPT2, JC2).
`PS2X_PAD_SCRIPT` makes reaching the scene repeatable; the trace itself is deterministic regardless of game
timing.

Intentional hash changes. A trace's `cpu_vram_hash` is what the CPU reference must produce at the tip of the
branch; any other value is a regression until explained. A commit that changes CPU output on purpose is a
`fix(gs)` (or `feat(gs)`) layer whose message names the behaviour it changes; the session that lands it replays
the whole corpus after that layer alone, and for each scene whose hash moved it writes the new `cpu_vram_hash` and
appends the old one to the scene's history with the commit subject:

```toml
[[scene.history]]
cpu_vram_hash = "<hash before the layer>"
changed_by = "<commit subject of the layer that changed it>"
```

A layer that should not change output (refactors, tests, docs) must leave every hash as it is. The section of
this document for the phase lists the per-layer hash table and the `--compare` results, so a reviewer can check
that each fix moved only the scenes that use its feature.

### 6.5 Unit and differential tests

- The existing `ps2_gs_tests.cpp` cases that observe VRAM or presentation run against each available renderer
  (a fixture builds a GS with the CPU backend, then with the GPU renderer, and skips GPU cases when no device
  is present).
- New per-feature tests in `ps2_gs_gpu_tests.cpp`, each a short packet sequence compared CPU vs GPU: every PSM
  upload / copy / readback; every primitive type; each ATST / AFAIL / DATE / ZTST; every blend selector
  combination with and without PABE, FBA, COLCLAMP, DTHE; every TFX x TCC; every wrap mode; CLUT CSM1 / CSM2 /
  CSA / CLD cases; feedback (sampling the target being drawn); overlapping local-to-local copies.
- Differential fuzzer: seeded random packet streams (random state per primitive from a weighted generator,
  random small primitives in a 256x256 window, random uploads and CLUT loads), compared after every primitive;
  any failing seed is minimized and added as a regression case. Runs 1-2 minutes in CI on lavapipe and longer
  locally.

### 6.6 In game

- Frame dumps (`PS2X_FRAME_DUMP`) of the same `PS2X_PAD_SCRIPT` run with `cpu` and `gpu` for visual review;
  game timing is not deterministic, so these are a sanity check, traces are the proof.
- paraLLEl stays buildable until phase 8 as a second opinion when CPU and GPU agree but a game looks wrong.
- Performance: `PS2X_SHOW_FPS`, `GS::takeLagStats` (EE time spent waiting for the GS thread), replay timings.
  All builds and measurements go through `benchlock`.

## 7. Phased plan

Each phase is a run of layered commits on `gs/gpu-renderer` (mechanical, then behaviour, then tests and docs),
green at every commit. A phase that would push the branch past 149 changed files becomes its own stacked branch
(`gs/gpu-renderer-<phase>`), cut along the module seams of section 5.2. Effort is in focused working days for
one developer.

| # | Milestone | Acceptance test | Effort |
| --- | --- | --- | --- |
| 0 | Groundwork. `GSStreamRenderer` interface (mechanical rename of the paraLLEl adapter), `GSStateImage`, `GSR2` traces with the GSR1 converter, `ps2_gs_replay --renderer/--compare` skeleton, shared `gs_gif_decoder` with a differential test against the frontend's decoder. | Existing tests green; paraLLEl unchanged in SC3; a GSR1 trace converts and replays to the same hash. | 4-6 d |
| 1 | CPU reference fixes of section 6.2 (dither, COLCLAMP wrap, mipmapping, fixed-point raster spec, fog, Z range), one layer each. | New CPU unit tests per item; SC3 and TMB traces re-hashed and frames reviewed. | 6-9 d |
| 2 | Vulkan bootstrap and VRAM: volk, device selection and feature check with CPU fallback, SPIR-V pipeline and committed header, lavapipe CI job; VRAM buffer; host-to-local (all PSMs, split packets, HWREG), local-to-local, local-to-host, `readVram`, `ReadVram` / `WriteVram`. | All transfer and swizzle tests in `ps2_gs_tests.cpp` pass on the GPU byte-exact; upload-only trace segments compare equal. | 8-10 d |
| 3 | Untextured drawing: points, lines, triangles, strips, fans, sprites; flat / Gouraud; XYOFFSET, scissor, SCANMSK; Z test and write (all Z formats); FBMSK, ZMSK; one dispatch per primitive. | Fuzzer (untextured subset) 10^5 primitives equal; existing primitive tests pass on GPU. | 8-10 d |
| 4 | Textures: all PSMs, CLUT loads and snapshots (CSM1 / CSM2, CSA, CLD), TEXA, TFX / TCC, all wrap modes, FST and STQ, bilinear, mipmapping, TEXFLUSH; feedback hazards in the page tracker. | Texture tests and fuzzer (textured subset) equal; "asynchronous rasterization ..." ordering cases pass on GPU. | 12-15 d |
| 5 | Pixel pipeline completeness: alpha test and AFAIL, DATE / DATM, blending with PABE / FIX, FBA, COLCLAMP, dither, fog, 16- and 24-bit targets. | Full fuzzer equal; first SC3 and TMB traces equal end to end (slow is fine). | 6-8 d |
| 6 | Presentation and wiring: `present.comp`, double-buffered readback, `PS2X_GS_RENDERER=gpu`, game override request, recording under the GPU renderer, `clearFramebufferContext`. | Presentation tests pass on GPU; SC3 boots to a fight with `gpu`, frames match `cpu` by eye; `--present` compare equal on traces. | 4-6 d |
| 7 | Performance: binning + per-tile raster, state dedupe, batch hazard tracking at block granularity, in-batch uploads / CLUT loads, specialized pipelines if profiling asks, submission pacing. | Byte-exact unchanged on the whole corpus; SC3 battle trace GPU time per frame below 8 ms on the reference machine; GS thread CPU time below 4 ms per frame. | 12-16 d |
| 8 | SC3 parity: full SC3 corpus equal; fights at 60 fps with `gpu` under the same settings that give 60 with `parallel`; lag stats no worse than paraLLEl. | Corpus green; 5-minute scripted fight at 60 fps (PS2X_SHOW_FPS log); no visual regressions in the scene list. | 5-10 d |
| 9 | Other ports and platforms: corpus for TMB, VP2, Drakengard, TLOTR, FFOP, SCTPT2, JC2; Windows MSVC and MinGW builds and runs; `gpu` becomes the default renderer where the CPU path was used for speed. | Every port's corpus green; each port playable with `gpu` on Linux and Windows. | 10-15 d |
| 10 | Retire paraLLEl (separate branch): remove the FetchContent of paraLLEl-GS and Granite, `patch_parallel_gs.cmake`, `gs_parallel.cpp`; `PS2X_GS_RENDERER=parallel` maps to `gpu` with a warning. | Clean build without Granite on all three toolchains; all tests and corpus green. | 1-2 d |

Total: roughly 75 to 105 working days. Phases 1 and 2 can run in parallel (different files); everything after 2
is sequential.

### 7.1 Phase 0 results

Layers on `gs/gpu-renderer`: the `GSStreamRenderer` rename, register decoders and encoders
(`gs_registers.h`), `GSStateImage` with `GS::saveState` / `loadState`, `GSR2` traces and `--convert`, per-PSM
pixel helpers, the interface additions with the paraLLEl and CPU implementations, recording and
`ReadVram` / `WriteVram` under a stream renderer, `ps2_gs_replay --renderer / --compare / --every`,
`GSGifDecoder`, one fix (below) and the `PS2GSState` tests.

Acceptance:

- Existing tests green at every layer: `ps2x_tests` (564 before, 572 with `PS2GSState`) and the 45 `gs_cache`
  cases. The five SC3 `GSR1` traces in `~/Games/PS2/SoulcaliburIII/runs` give the same CPU VRAM hash before
  phase 0, after each layer, replayed directly and after conversion to `GSR2` (`~/Games/PS2/gs-traces/sc3/`,
  hashes in its `manifest.toml`).
- paraLLEl unchanged in SC3: the phase 0 layers on top of `sc3/bringup` boot to the attract-mode fight on
  paraLLEl the same way as `sc3/bringup` itself (same log apart from timing noise, frames correct). A 2-second
  trace recorded under paraLLEl (`parallel_route.gsr2`, starting at a latch) replays on both renderers.
- `--compare`: `cpu` against `cpu` matches on every trace. `parallel` against `cpu` differs early on every trace,
  as expected before phase 1: on `fight`, `gallery` and `parallel_route` the first difference is the CT16 clear at
  page 140 drawn with DTHE=1 (paraLLEl dithers, the CPU backend does not: section 6.2); on `t` it is command 31,
  a 64x32 area of FRAME_1 where paraLLEl writes 0x80000000 and the CPU 0x7f131313, not triaged yet.

Deviations from the plan above, each written into its section: the CPU renderer implements `GSStreamRenderer`
(5.1); the state image has no in-progress transfer and a paraLLEl image has no vertex queue or CLUT buffer
(5.1); recording under a stream renderer starts at a latch, and `clearFramebufferContext` stays skipped with a
stream renderer (section 2); `--every` defaults to 64 and the bisect is automatic (6.4). One CPU path change was
needed for images to round trip: after a reset ZBUF now decodes as a written zero (PSMZ32) instead of PSM 0,
which no write can produce; no trace or test changed.

### 7.2 Phase 1 results

The `t` discrepancy of phase 0 (command 31, a 64x32 area of FRAME_1 where paraLLEl wrote 0x80000000 and the CPU
0x7f131313) is a CPU bug, and paraLLEl is right. Command 31 is one 238 KB packet: a texture upload, then 48
Gouraud, alpha-blended triangle strips (ALPHA = Cd + (Cs - Cd) * As) drawing a glow outline of thin slivers, about
4 pixels wide and 90 tall, from colour 0xc3 at one side to 0 at the other. Split into one command per GIF tag, the
first difference is the first strip's first triangle: at (223, 175) the CPU covers the pixel and paraLLEl does
not, at (224, 175) the CPU writes 0x4b and paraLLEl 0x32. The CPU sampled coverage and attributes at the pixel
centre plus one half (x + 0.5, with a 1e-4 tolerance on the edges); the GS samples at whole pixel positions, which
is also the rule the CPU's own sprite path already followed (a sprite covers x0 <= x < x1 at whole x) and the one
both PCSX2 and paraLLEl implement. At y = 175 the triangle spans x = 223.02 to 224.81: only pixel 224 has its
sample inside, and its colour, a quarter of the way from the 0 edge, is about 0x33 before blending (paraLLEl:
0x32). Sampling half a pixel to the right, the CPU lit column 223 down the whole sliver and every pixel came out
half a pixel brighter; 48 strips of such slivers left 3954 words different. With the fixed-point rasterizer of
section 6.2, command 31 matches paraLLEl word for word, and so does the whole `t` trace at its end.

Layers on `gs/gpu-renderer`, each with the trace hashes it produced (CPU, synchronous replay):

| Layer | t | fight | attack | gallery | free | parallel_route |
| --- | --- | --- | --- | --- | --- | --- |
| phase 0 | 9f57179650881733 | d24e92a9bc2a7981 | 13bd6761ae261dba | 537460974613ade1 | 09b626f92089d47e | c5e927fa19e7dfca |
| sampler refactor | = | = | = | = | = | = |
| fog | = | = | 035a351706548124 | = | = | = |
| COLCLAMP wrap | = | = | = | = | e235c6aae93e8dee | = |
| DTHE dither | 9e7143ca8f9a29f8 | 0fc7a14ecb4f044f | 0cc7bb788ed6edd9 | 6ea52a97d18cfde1 | 860b4c4fdfb6fa6a | 68d6a358027dea95 |
| Z range | = | = | = | = | = | = |
| fixed-point raster | efdde1828bcc158e | 1785ba1af3c1f13a | bd4efda565124016 | ebc5d38aafe7150a | 2e11ae1dac1e6273 | e91ef9bdb13337e6 |
| mipmapping | = | = | = | = | 833fdafa00cde365 | = |
| perf (plane stepping) | = | = | = | = | = | = |
| tests | = | = | = | = | = | = |

Each moved only where its feature is used: fog only `attack` (the one scene with a non-zero FOGCOL; the others fog
toward black, where both formulas agree), COLCLAMP only `free` (the one scene that writes COLCLAMP = 0), dithering
every scene (each clears a CT16 buffer with DTHE = 1), Z saturation none (no scene sends a Z past its format),
mipmapping only `free` (the one scene with MXL > 0). At the tip, the asynchronous raster workers match the
synchronous replay on all six traces (`PS2X_GS_REPLAY_CHECK=128`).

`ps2_gs_replay --renderer parallel --compare cpu` (every 64 commands), first difference and words differing at the
end of the trace (paraLLEl's final VRAM against the CPU's):

| Layer | t | fight | attack | gallery | free | parallel_route |
| --- | --- | --- | --- | --- | --- | --- |
| phase 0 | 31 / 36142 | 0 / 48579 | 0 / 357193 | 0 / 216416 | 0 / 588649 | 1 / 405467 |
| fog | 31 / 36142 | 0 / 48579 | 0 / 315383 | 0 / 216416 | 0 / 588649 | 1 / 405467 |
| COLCLAMP wrap | 31 / 36142 | 0 / 48579 | 0 / 315383 | 0 / 216416 | 0 / 574185 | 1 / 405467 |
| DTHE dither | 31 / 528 | 0 / 15335 | 0 / 280547 | 1 / 180576 | 0 / 558656 | 605 / 402222 |
| Z range | 31 / 528 | 0 / 15335 | 0 / 280547 | 1 / 180576 | 0 / 558656 | 605 / 402222 |
| fixed-point raster | 32 / 0 | none / 0 | 33 / 182146 | 15 / 124786 | 22 / 210057 | 605 / 218547 |
| mipmapping | 32 / 0 | none / 0 | 33 / 182146 | 15 / 124786 | 22 / 178209 | 605 / 218547 |

`fight` now matches paraLLEl at every comparison and `t` at its end (the transient difference at command 32 is
overwritten later). The differences left are small: off by one in a channel on textured, bilinear-filtered
primitives (`attack` command 33, `gallery` command 15, `free` command 22) and an alpha difference (0x10 against 0x16) in
`parallel_route` command 605. They point at the precision of the bilinear weights (the CPU uses 8 fraction bits; the GS's
precision is unknown), the fog rounding (section 6.2) and the S, T, Q interpolation, the candidates for the next
reference fixes; none is in the phase 1 list. Lines are still untextured in the CPU backend, also a gap to close
before the GPU copies it.

Tests: `ps2xTest/src/ps2_gs_reference_tests.cpp` (`PS2GSReference`) pins each fix against values computed by hand
from the rules above: fog (F = 255 over FOGCOL = 128, F = 128, F = 0, F = 1), COLCLAMP clamp and wrap of 300 and
-100, the dither of every pixel of an 8x8 CT16 sprite, no dither on CT32 or with DTHE = 0, dither then clamp or
wrap at both ends, Z24 and Z16 saturation in the write and in GREATER / GEQUAL tests, the top-left rule on a quad
split along its diagonal (each pixel exactly once, right and bottom edges out), whole-pixel coverage and truncated
Gouraud values, sprite coverage and texture coordinates (bilinear halfway at pixel 1), point and line rounding,
MMAG / MMIN selection by the LOD's sign, nearest and blended mip levels from MIPTBP1 and the LOD from Q. Existing
tests that drew zero-size sprites to reach one pixel now draw 1x1 sprites (a zero-size sprite draws nothing on the
GS), the fan quad test reads its frame through the CT32 swizzle (it read linear memory), and the STQ and bilinear
triangle tests sample pixels that still separate their cases at whole-pixel sample points; the fog test's F = 0
case now expects FOGCOL exactly, as its own message said.

Performance (`ps2_gs_replay <trace> 3`, synchronous CPU replay, best of 3, under `benchlock -m bench`), phase 0
against the tip of phase 1:

| Trace | phase 0 | phase 1 | change |
| --- | --- | --- | --- |
| t | 919 ms | 937 ms | +2.0% |
| fight | 3732 ms | 3862 ms | +3.5% |
| gallery | 3013 ms | 3163 ms | +5.0% |
| free | 6355 ms | 6697 ms | +5.4% |
| attack | 9735 ms | 10020 ms | +2.9% |
| fight, `PS2X_GS_ASYNC_RASTER=1` | 964 ms | 1010 ms | +4.8% |

Dithering costs nothing measurable (the dither layer alone: -1.7% to +1.9% across the traces, within noise; the
no-dither path is one branch per block of 8 pixels). The fixed-point rasterizer accounts for most of the rest, and
it is work the GS does, not overhead: counting bilinear samples on `fight`, the CPU now blends 91.9 M four-tap
samples against 83.2 M before (`gallery`: 62.8 M against 57.1 M), because SC3's full-screen copy passes sample
whole-pixel positions that fall between texels, where the old half-pixel centres fell on texel centres and took
the one-tap shortcut; restoring the old sprite code alone brings the time back, and the extra blends are exactly
the ones that make `fight` match paraLLEl. Mipmapping costs `free` about 10% on its own (its mipmapped primitives
take the per-pixel LOD path, without the decoded-texture fast path for levels above 0); traces without mipmaps are
unaffected. The perf layer (32-bit plane stepping, no divisions for constant attributes) changes no output.

Not done in phase 1: the plan's TMB traces (the corpus has none yet, so only SC3 was re-hashed), a review of
frames by eye (the paraLLEl comparison stands in for it: `fight` and `t` now match it exactly), MTBA, textured
lines, and the candidates above (bilinear weight precision, fog rounding, S/T/Q precision), which need hardware
results or a better paraLLEl-independent argument before they change the reference.

## 8. Things deliberately left out

- Upscaling, supersampling, texture filtering beyond the GS's own, widescreen hacks.
- Per-game hacks of any kind (CRC-style fixes). A game that renders wrong is a GS bug to fix in both renderers.
- macOS (no Vulkan without MoltenVK; can be tried later, nothing in the design blocks it).
- Moving the CPU backend onto the shared GIF decoder (useful cleanup, not needed for parity).

## 9. Risks and open questions

1. Bit-exactness across GPU vendors. Float division and FMA contraction (6.3), shared-memory or subgroup
   behaviour differences, and driver bugs can each break byte equality on one vendor only. Mitigation: fixed
   point wherever the GS is fixed point, `precise` elsewhere, the fuzzer run on NVIDIA, AMD, Intel and lavapipe
   before each phase is called done.
2. The reference is not the GS. Parity with the CPU backend only proves agreement. Section 6.2 lists the known
   gaps; more will appear. Where hardware behaviour is uncertain, PS2 hardware test results (or the GS manual)
   decide, and the decision is written next to the test. Open: whether we can run small GS test programs on a
   real PS2 to settle such cases.
3. Performance of strict-order compute on SC3. Many tiny strips mean many primitives per tile and per batch;
   bloom and feedback passes mean frequent hazard boundaries and barriers. paraLLEl shows it can be done, but our
   first binned version may miss 60 fps. Mitigations: block-granular hazards (not page), in-batch CLUT and
   upload commands, state dedupe, specialized pipelines, measuring flush reasons from day one.
4. The CPU presentation heuristics (`preferredSource`, non-black fallback) are not GS behaviour. Porting them
   keeps parity with CPU but carries their failure modes to the GPU. Open: drop them in both once the GPU
   renderer makes them unnecessary for speed-related artifacts (SC3's "copy of a cleared buffer" was a VU1 speed
   symptom, not a display one).
5. GS thread load. Decoding, state hashing and record building for SC3's packet volume must fit in a few ms per
   frame; the frontend currently also scans every packet at queue time. If needed, setup moves fully to the GPU
   (already planned) and the decoder gets the same fast paths as the frontend (native packed packets).
6. Undefined or unclear GS behaviour that tests must pin: overlapping local-to-local copies, sampling a texel
   written by the same primitive, CLUT loads during a transfer, AA1, interlaced field selection at odd
   `vsyncTick`, fog rounding (section 6.2). Phase 1 pinned Z overflow for Z24 / Z16 (saturates) and dithering with
   COLCLAMP=0 (dither, then wrap), both with PCSX2 and paraLLEl agreeing.
7. Device loss or missing features at run time. Startup failure falls back to the CPU backend (as with paraLLEl
   today). Device loss mid-game is not recoverable without a VRAM copy; first version reports and exits.
8. Shader toolchain drift between the committed SPIR-V and the GLSL. Mitigated by the CI regeneration check.
9. Clean-room discipline. paraLLEl-GS (LGPL) and PCSX2 (GPL) are read for ideas only (tile binning, page
   tracking, hazard-driven batching are general techniques); no code, shader or table is copied. The swizzle
   tables and GS rules come from our own `ps2_gs_memory` and the GS manual. Reviewers should reject any change
   that looks transliterated.
10. Open: tile size (16x16 vs 32x32 with 4 pixels per invocation), batch cap, whether `shaderInt64` is worth
    requiring, whether to expose the renderer choice in the debug UI, and whether `gpu` should become the default
    for all ports at phase 9 or per port.
