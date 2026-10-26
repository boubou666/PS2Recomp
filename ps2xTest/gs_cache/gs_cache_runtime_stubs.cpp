// The GS sources under test call into the runtime on SIGNAL/FINISH edges (ps2_runtime.cpp); these suites link the
// GS alone.
void ps2NotifyGsCsrEdge() {}
#include "runtime/ps2_memory.h"

uint64_t ps2CurrentEeCycle() { return 0u; }
bool ps2ScheduleGsInterruptCheck(uint64_t) { return false; }
bool ps2OnVif1WorkerClock() { return false; }
uint32_t ps2GsPostEvent(GSRegisters &, GsCpuEvent, uint32_t, uint32_t, uint64_t, uint64_t) { return 0u; }
