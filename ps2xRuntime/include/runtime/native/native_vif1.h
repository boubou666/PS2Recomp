#pragma once

#include <cstddef>
#include <cstdint>

namespace ps2x::native
{
    // The VIF1 stream a game sends, handed to its native adapter as the game sends it (on the EE thread, so EE memory
    // still holds what the stream refers to): a chain flattened the way VIF1 receives it (each tag's two VIF codes,
    // then its data), or a normal transfer's quadwords. The adapter reads the game's own 3D data from it (meshes,
    // matrices, lights) wherever in the game's code the send happens.
    using Vif1Observer = void (*)(const uint8_t *stream, size_t bytes);

    void setVif1Observer(Vif1Observer observer);
    bool vif1Observed();
    void observeVif1(const uint8_t *stream, size_t bytes);
}
