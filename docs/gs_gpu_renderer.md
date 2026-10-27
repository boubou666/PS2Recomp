# GPU GS renderer: design and plan

Status: design, no code yet. Branch `gs/gpu-renderer`.

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

Gaps of today's paraLLEl integration that the new renderer must close, not copy:

- Recording is disabled when paraLLEl is active (`!m_parallel` in `workerLoop`), so no trace can be taken from
  the fast path.
- `GS::ReadVram` / `WriteVram` / `clearFramebufferContext` still go to the CPU backend (or are skipped) when
  paraLLEl owns VRAM, so any HLE path using them sees stale memory.
- The trace's drawing state is the frontend's and the CPU backend's private serialization, so a trace cannot
  seed a renderer that decodes GIF itself.

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
    packet.
  - `loadState(const GSStateImage &)` / `saveState(GSStateImage &)`: a renderer-neutral image of the GS (raw
    64-bit register file for both contexts and the general registers, PRMODE/PRMODECONT, the vertex queue,
    the 1 KiB CLUT buffer and its CBP cache, an in-progress transfer), used by traces and by
    `setRasterBackend`-style renderer switches.
  - `readVramPixel` / `writeVramPixel` / `clearFrame`: so `GS::ReadVram`, `WriteVram` and
    `clearFramebufferContext` work on whichever renderer owns VRAM (through `runOnWorker`).
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
feeds the same packets to both and compares the primitives and transfers they produce. Moving the CPU path
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
  format's range handling pinned by a test (open question 9.6).
- Blend (ABE, PABE, ALPHA A/B/C/D, FIX) with the 24-bit destination alpha of 0x80, then dither (DTHE, DIMX 4x4
  matrix) and COLCLAMP (clamp or wrap to 8 bits).
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

Reading `gs_cpu_backend.cpp` shows places where it is not yet the GS. Each is fixed in the CPU backend first, as
its own layer with a unit test, and only then implemented on the GPU, so the two never agree on a known bug:

- DTHE / DIMX dithering is not applied.
- COLCLAMP is not read: blending always clamps, the GS wraps when CLAMP=0.
- Mipmapping is absent (TEX1 only selects bilinear; MXL, LCM, L, K, MIPTBP1/2 unused).
- Positions are floats and Z a double; the GS is 12.4 fixed point with integer edge functions. The raster rules
  get a written spec (fill convention, sprite and line rules, Z and attribute interpolation precision) and the
  CPU moves to it.
- Fog uses two `>> 8` products; the GS formula and rounding need checking against hardware results.
- Z range behaviour for Z24 / Z16 (clamp or truncate) needs a pinning test.

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

- `--renderer cpu|gpu|parallel` to choose who replays (default `cpu`).
- `--compare cpu` (with `--renderer gpu`): replays on both, compares VRAM every N commands (`--every N`, default
  1 during bring-up, 64 for the corpus), and on the first difference prints the command index, its GIF tag and
  draw state, the differing blocks, and for each the PSM-decoded (x, y) under the active FRAME / ZBUF / TEX0, then
  writes both buffers with `--dump`. A bisect mode re-runs with `--every 1` inside the failing interval.
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
   written by the same primitive, CLUT loads during a transfer, Z overflow for Z24 / Z16, AA1, dithering with
   COLCLAMP=0, interlaced field selection at odd `vsyncTick`.
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
