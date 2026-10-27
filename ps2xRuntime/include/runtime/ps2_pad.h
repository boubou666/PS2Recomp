#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);
    // Port 0 always (gamepad 0, else the keyboard); port 1 while a second gamepad is plugged in, or with
    // PS2X_PAD2_SCRIPT (the same format as PS2X_PAD_SCRIPT, for player 2).
    bool connected(int port) const;
};

// Buttons a PS2X_PAD_SCRIPT string holds down at a time (seconds since process start), as a mask of
// pad button bits (1 = pressed).
uint16_t ps2PadScriptButtonsAt(const char *script, double seconds);
// Buttons that PS2X_PAD_INPUT lines, read at addedAt, hold down at a time (same clock).
uint16_t ps2PadLiveInputButtonsAt(const char *lines, double addedAt, double seconds);

#endif
