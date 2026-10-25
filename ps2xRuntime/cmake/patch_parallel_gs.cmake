# Run on the fetched paraLLEl-GS sources (cmake -DSOURCE_DIR=... -DPATCH_DIR=... -P): its shaders come precompiled in
# gs/shaders/slangmosh.hpp, so a fixed shader replaces the embedded one there. Idempotent.
#  - blit_circuit: the PCRTC merges circuit 1 over circuit 2 with circuit 1's pixel alpha, 0x80 being 1.0 (as
#    PCSX2 does); unpatched, sprites drawn with alpha 0x80 come out half transparent (TMB's level loading screen).
set(header "${SOURCE_DIR}/gs/shaders/slangmosh.hpp")
file(READ "${header}" text)
string(FIND "${text}" "ps2x_blit_circuit_spirv" patched)
if(NOT patched EQUAL -1)
    return()
endif()

file(READ "${PATCH_DIR}/parallel_gs_blit_circuit.inc" blit)
string(REGEX REPLACE "this->blit_circuit = device\\.request_shader\\(spirv_bank \\+ [0-9]+, [0-9]+, &layout\\);"
       "this->blit_circuit = device.request_shader(ps2x_blit_circuit_spirv, sizeof(ps2x_blit_circuit_spirv), &layout);"
       replaced "${text}")
if(replaced STREQUAL text)
    message(FATAL_ERROR "patch_parallel_gs: blit_circuit not found in ${header}")
endif()
string(REPLACE "static const uint32_t spirv_bank[] =" "${blit}static const uint32_t spirv_bank[] =" replaced "${replaced}")
file(WRITE "${header}" "${replaced}")
