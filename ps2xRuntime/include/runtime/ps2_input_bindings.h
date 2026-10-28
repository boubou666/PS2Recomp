#pragma once

// What drives each PS2 pad button and stick direction on the PC: keys, gamepad buttons and axes, mouse buttons, the
// wheel and mouse movement, any number per action, per player. Loaded from input.ini next to the game's ELF (or
// PS2X_INPUT_CONFIG), edited in the Controls window (F2) and saved back there. Without a file the defaults are the
// runtime's former fixed mapping: keyboard and the first gamepad for player 1, the second gamepad for player 2.
//
// The window's thread samples the mouse and the wheel once a frame (frameUpdate); the pad reads (padState) come from
// the game's thread.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ps2_input
{
    // The PS2 side: the 16 buttons, then the two sticks' four directions.
    enum Action : int
    {
        Up, Down, Left, Right, Cross, Circle, Square, Triangle, L1, R1, L2, R2, L3, R3, Start, Select,
        LeftStickLeft, LeftStickRight, LeftStickUp, LeftStickDown,
        RightStickLeft, RightStickRight, RightStickUp, RightStickDown,
        ActionCount
    };

    // The PC side.
    enum class Source : uint8_t
    {
        Key,           // code: raylib KeyboardKey
        GamepadButton, // code: raylib GamepadButton
        GamepadAxis,   // code: raylib GamepadAxis, direction: +1 / -1
        MouseButton,   // code: raylib MouseButton
        MouseWheel,    // direction: +1 up / -1 down
        MouseMove,     // code: 0 x, 1 y; direction: +1 / -1 (right / down)
    };

    struct Binding
    {
        Source source = Source::Key;
        int code = 0;
        int direction = 1;
        bool operator==(const Binding &other) const
        {
            return source == other.source && code == other.code && direction == other.direction;
        }
    };

    struct Profile
    {
        std::vector<Binding> actions[ActionCount];
        int gamepad = 0; // which gamepad's buttons and axes this player reads
    };

    struct MouseSettings
    {
        float sensitivity = 1.0f; // stick deflection per (pixels per frame / 10)
        bool invertY = false;
        bool capture = false;     // hide and hold the cursor while playing (released while F1 / F2 windows show)
    };

    struct Config
    {
        Profile players[2];
        MouseSettings mouse;
    };

    // The current configuration (loaded on first use). Changes apply at once (hold configMutex() while changing it,
    // the game's thread reads it); save() writes them.
    Config &config();
    std::mutex &configMutex();
    void resetToDefaults();
    bool save();
    std::string configPath();

    // The PS2 pad for a player: buttons as the PS2 sends them (a bit cleared per pressed button) and the sticks, 0..255
    // with 128 the centre (rx, ry, lx, ly).
    struct PadState
    {
        uint16_t buttons = 0xFFFFu;
        uint8_t rx = 128, ry = 128, lx = 128, ly = 128;
    };
    PadState padState(int player);

    // Window thread, once a frame: mouse movement and wheel, cursor capture. setUiOpen: a runtime window has the mouse
    // (the cursor is free and the mouse drives no binding meanwhile).
    void frameUpdate();
    void setUiOpen(bool open);

    // Names (input.ini, the Controls window).
    const char *actionName(int action);   // "cross", "left_stick_up", ...
    const char *actionLabel(int action);  // "Cross", "Left stick up", ...
    std::string bindingName(const Binding &binding); // "key:SPACE", "pad:RIGHT_FACE_DOWN", "mouse:LEFT", ...
    bool parseBinding(const std::string &text, Binding &binding);

    // The Controls window's capture: the first input pressed since the last call, if any.
    bool pollAnyInput(int gamepad, Binding &binding);
}
