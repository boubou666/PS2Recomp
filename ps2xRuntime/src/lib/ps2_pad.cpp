#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

uint64_t ps2CurrentEeCycle(); // ps2_runtime.cpp

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    // PS2X_PAD_SCRIPT="25:SQUARE,30:START,60.5:CROSS/300" presses buttons at the given times
    // (seconds since process start) for 500 ms or the given milliseconds, without a keyboard or
    // window focus.
    struct ScriptedPress
    {
        double time = 0.0;
        double hold = 0.5;
        uint16_t mask = 0;
    };

    const auto g_processStart = std::chrono::steady_clock::now();

    // The pad scripts' clock, in seconds: wall time since the process started, or with PS2X_PAD_CLOCK=ee the EE's
    // own emulated time (cycles / 294.912 MHz), which a replay follows far more closely than wall time when loads
    // or the frame rate differ from the recorded session. Recording (PS2X_PAD_RECORD) uses the same clock.
    double padClockSeconds()
    {
        static const bool ee = [] { const char *v = std::getenv("PS2X_PAD_CLOCK"); return v && std::strcmp(v, "ee") == 0; }();
        if (ee)
            return static_cast<double>(ps2CurrentEeCycle()) / 294912000.0;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_processStart).count();
    }

    uint16_t scriptButton(const std::string &name)
    {
        static const struct { const char *name; uint16_t mask; } kButtons[] = {
            {"LEFT", PAD_LEFT}, {"DOWN", PAD_DOWN}, {"RIGHT", PAD_RIGHT}, {"UP", PAD_UP}, {"START", PAD_START},
            {"R3", PAD_R3}, {"L3", PAD_L3}, {"SELECT", PAD_SELECT}, {"SQUARE", PAD_SQUARE}, {"CROSS", PAD_CROSS},
            {"CIRCLE", PAD_CIRCLE}, {"TRIANGLE", PAD_TRIANGLE}, {"R1", PAD_R1}, {"L1", PAD_L1}, {"R2", PAD_R2},
            {"L2", PAD_L2}};
        for (const auto &button : kButtons)
            if (name == button.name)
                return button.mask;
        return 0;
    }

    std::vector<ScriptedPress> parsePadScript(const char *text)
    {
        std::vector<ScriptedPress> presses;
        if (!text)
            return presses;
        std::string entries(text);
        size_t start = 0;
        while (start < entries.size())
        {
            size_t end = entries.find(',', start);
            if (end == std::string::npos)
                end = entries.size();
            const std::string entry = entries.substr(start, end - start);
            const size_t colon = entry.find(':');
            if (colon != std::string::npos)
            {
                std::string button = entry.substr(colon + 1);
                double hold = 0.5;
                if (const size_t slash = button.find('/'); slash != std::string::npos)
                {
                    hold = std::atof(button.substr(slash + 1).c_str()) / 1000.0;
                    button.resize(slash);
                }
                presses.push_back({std::atof(entry.substr(0, colon).c_str()), hold, scriptButton(button)});
            }
            start = end + 1;
        }
        return presses;
    }

    uint16_t buttonsDownAt(const std::vector<ScriptedPress> &script, double seconds)
    {
        uint16_t down = 0;
        for (const ScriptedPress &press : script)
            if (seconds >= press.time && seconds < press.time + press.hold)
                down |= press.mask;
        return down;
    }

    // PS2X_PAD_INPUT=<file>: each line appended while the game runs presses its words in turn, 0.35 s apart,
    // as PS2X_PAD_SCRIPT entries without a time ("CROSS", "circle/100", "SQUARE+TRIANGLE" presses both).
    void appendLivePresses(const std::string &added, double now, std::vector<ScriptedPress> &presses)
    {
        double at = now;
        for (const ScriptedPress &press : presses)
            at = std::max(at, press.time + 0.35);
        size_t start = 0;
        while (start < added.size())
        {
            size_t end = added.find_first_of(" ,\n", start);
            if (end == std::string::npos)
                end = added.size();
            std::string word = added.substr(start, end - start);
            start = end + 1;
            double hold = 0.25;
            if (const size_t slash = word.find('/'); slash != std::string::npos)
            {
                hold = std::atof(word.substr(slash + 1).c_str()) / 1000.0;
                word.resize(slash);
            }
            uint16_t mask = 0;
            size_t partStart = 0;
            while (partStart < word.size())
            {
                size_t plus = word.find('+', partStart);
                if (plus == std::string::npos)
                    plus = word.size();
                std::string part = word.substr(partStart, plus - partStart);
                for (char &c : part)
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                mask |= scriptButton(part);
                partStart = plus + 1;
            }
            if (mask == 0)
                continue;
            presses.push_back({at, hold, mask});
            at += 0.35;
        }
    }

    void appendLiveInput(const char *path, double now, std::vector<ScriptedPress> &presses)
    {
        static double nextPoll = 0.0;
        static long offset = 0;
        if (now < nextPoll)
            return;
        nextPoll = now + 0.1;
        FILE *file = std::fopen(path, "rb");
        if (!file)
            return;
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        if (size < offset)
            offset = 0;
        std::string added(static_cast<size_t>(size - offset), '\0');
        std::fseek(file, offset, SEEK_SET);
        added.resize(std::fread(added.data(), 1, added.size(), file));
        std::fclose(file);
        const size_t lastLine = added.rfind('\n');
        if (lastLine == std::string::npos)
            return;
        offset += static_cast<long>(lastLine + 1);
        added.resize(lastLine);
        appendLivePresses(added, now, presses);
    }

    // PS2X_PAD_SCRIPT_FILE=<file>: a pad script read from a file (entries separated by commas or new lines),
    // for scripts too long for the environment, such as a PS2X_PAD_RECORD recording.
    std::string padScriptFromFile()
    {
        std::string text;
        if (const char *path = std::getenv("PS2X_PAD_SCRIPT_FILE"))
        {
            if (FILE *file = std::fopen(path, "rb"))
            {
                char chunk[4096];
                for (size_t got; (got = std::fread(chunk, 1, sizeof(chunk), file)) != 0;)
                    text.append(chunk, got);
                std::fclose(file);
            }
            std::replace(text.begin(), text.end(), '\n', ',');
            std::replace(text.begin(), text.end(), '\r', ',');
        }
        if (const char *inline_ = std::getenv("PS2X_PAD_SCRIPT"))
            text = text.empty() ? std::string(inline_) : text + "," + inline_;
        return text;
    }

    // PS2X_PAD_RECORD=<file>: the buttons pressed on pad 1 (keyboard or gamepad), one PS2X_PAD_SCRIPT entry
    // per press ("<seconds since start>:<BUTTON>/<milliseconds held>") written when it is released, so a
    // session can be replayed with PS2X_PAD_SCRIPT_FILE. Scripted presses are not recorded.
    void recordPhysicalButtons(uint16_t down)
    {
        static const char *const path = std::getenv("PS2X_PAD_RECORD");
        if (!path)
            return;
        static FILE *file = std::fopen(path, "w");
        static uint16_t previous = 0;
        static double pressedAt[16] = {};
        if (!file || down == previous)
            return;
        static const char *const kNames[16] = {"SELECT", "L3", "R3", "START", "UP", "RIGHT", "DOWN", "LEFT",
                                               "L2", "R2", "L1", "R1", "TRIANGLE", "CIRCLE", "CROSS", "SQUARE"};
        const double now = padClockSeconds();
        for (int bit = 0; bit < 16; ++bit)
        {
            const uint16_t mask = static_cast<uint16_t>(1u << bit);
            if ((down & mask) && !(previous & mask))
                pressedAt[bit] = now;
            else if (!(down & mask) && (previous & mask))
                std::fprintf(file, "%.3f:%s/%d\n", pressedAt[bit], kNames[bit],
                             std::max(1, static_cast<int>((now - pressedAt[bit]) * 1000.0 + 0.5)));
        }
        std::fflush(file);
        previous = down;
    }

    uint16_t scriptedButtonsDown()
    {
        static const std::string scriptText = padScriptFromFile();
        static const std::vector<ScriptedPress> script = parsePadScript(scriptText.empty() ? nullptr : scriptText.c_str());
        static const char *const liveInput = std::getenv("PS2X_PAD_INPUT");
        if (script.empty() && !liveInput)
            return 0;
        const double now = padClockSeconds();
        uint16_t down = buttonsDownAt(script, now);
        if (liveInput)
        {
            static std::vector<ScriptedPress> livePresses;
            appendLiveInput(liveInput, now, livePresses);
            down |= buttonsDownAt(livePresses, now);
        }
        return down;
    }
}

uint16_t ps2PadScriptButtonsAt(const char *script, double seconds)
{
    return buttonsDownAt(parsePadScript(script), seconds);
}

uint16_t ps2PadLiveInputButtonsAt(const char *lines, double addedAt, double seconds)
{
    std::vector<ScriptedPress> presses;
    appendLivePresses(lines, addedAt, presses);
    return buttonsDownAt(presses, seconds);
}

namespace
{
    uint16_t pad2ScriptedButtonsDown()
    {
        static const std::vector<ScriptedPress> script = parsePadScript(std::getenv("PS2X_PAD2_SCRIPT"));
        if (script.empty())
            return 0;
        return buttonsDownAt(script, padClockSeconds());
    }
}

namespace
{
    // The host gamepad for a pad port: the first available of the 4 raylib slots for port 0, the second for port 1
    // (a controller is not always in slot 0: a virtual pad or a DualShock 4 seen through DirectInput may take
    // another). Loads SDL-format mappings from PS2X_GAMEPAD_DB or gamecontrollerdb.txt next to the runner once, for
    // controllers raylib does not map by itself, and logs the controllers whenever they change.
    int gamepadForPort(int port)
    {
        static const bool s_mappings = []
        {
            const char *path = std::getenv("PS2X_GAMEPAD_DB");
            if (char *text = LoadFileText(path ? path : "gamecontrollerdb.txt"))
            {
                const int added = SetGamepadMappings(text);
                UnloadFileText(text);
                std::fprintf(stderr, "[pad] %d gamepad mappings loaded\n", added);
            }
            return true;
        }();
        (void)s_mappings;
        int found[4] = {-1, -1, -1, -1}, count = 0;
        for (int i = 0; i < 4; ++i)
            if (IsGamepadAvailable(i))
                found[count++] = i;
        static int s_lastMask = 0;
        int mask = 0;
        for (int i = 0; i < count; ++i)
            mask |= 1 << found[i];
        if (mask != s_lastMask)
        {
            s_lastMask = mask;
            for (int i = 0; i < count; ++i)
                std::fprintf(stderr, "[pad] gamepad slot %d: %s (pad port %d)\n", found[i], GetGamepadName(found[i]), i);
            if (count == 0)
                std::fprintf(stderr, "[pad] no gamepad: keyboard on pad port 0\n");
        }
        return port < count ? found[port] : -1;
    }
}

bool PSPadBackend::connected(int port) const
{
    if (port == 0)
        return true;
    static const bool pad2Script = std::getenv("PS2X_PAD2_SCRIPT") != nullptr;
    return port == 1 && (pad2Script || gamepadForPort(1) >= 0);
}

bool PSPadBackend::readState(int port, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;
    if (port != 0 && !connected(port))
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint16_t btns = 0xFFFFu;
    // PS2X_PAD_NO_HOST=1: neither keyboard nor gamepads, only the pad scripts (a replay nobody can disturb).
    static const bool s_noHost = [] { const char *v = std::getenv("PS2X_PAD_NO_HOST"); return v && v[0] == '1'; }();
    const int hostPad = s_noHost ? -1 : gamepadForPort(port);
    const bool useGamepad = hostPad >= 0;
    const int kGamepad = useGamepad ? hostPad : 0;
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    if (useGamepad)
    {
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_THUMB))
            clearBit(PAD_L3);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_THUMB))
            clearBit(PAD_R3);

        float lx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    else if (port == 0 && !s_noHost)
    {
        // The movement keys also push the left stick fully, for games that move with it (sticks are read only in
        // analog mode).
        const bool up = IsKeyDown(KEY_UP) || IsKeyDown(KEY_W), down = IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S);
        const bool left = IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A), right = IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D);
        data[6] = static_cast<uint8_t>(left == right ? kPadStickCenter : left ? 0x00 : 0xFF);
        data[7] = static_cast<uint8_t>(up == down ? kPadStickCenter : up ? 0x00 : 0xFF);
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
            clearBit(PAD_UP);
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
            clearBit(PAD_DOWN);
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
            clearBit(PAD_LEFT);
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
            clearBit(PAD_RIGHT);
        if (IsKeyDown(KEY_X) || IsKeyDown(KEY_SPACE))
            clearBit(PAD_CROSS);
        if (IsKeyDown(KEY_C) || IsKeyDown(KEY_ESCAPE))
            clearBit(PAD_CIRCLE);
        if (IsKeyDown(KEY_Z) || IsKeyDown(KEY_KP_0))
            clearBit(PAD_SQUARE);
        if (IsKeyDown(KEY_V) || IsKeyDown(KEY_KP_1))
            clearBit(PAD_TRIANGLE);
        if (IsKeyDown(KEY_Q))
            clearBit(PAD_L1);
        if (IsKeyDown(KEY_E))
            clearBit(PAD_R1);
        if (IsKeyDown(KEY_LEFT_SHIFT))
            clearBit(PAD_L2);
        if (IsKeyDown(KEY_RIGHT_SHIFT))
            clearBit(PAD_R2);
        if (IsKeyDown(KEY_ENTER))
            clearBit(PAD_START);
        if (IsKeyDown(KEY_TAB))
            clearBit(PAD_SELECT);
        if (IsKeyDown(KEY_F))
            clearBit(PAD_L3);
        if (IsKeyDown(KEY_G))
            clearBit(PAD_R3);
        // Right stick on I/J/K/L (US layout positions).
        const bool rUp = IsKeyDown(KEY_I), rDown = IsKeyDown(KEY_K), rLeft = IsKeyDown(KEY_J), rRight = IsKeyDown(KEY_L);
        data[4] = static_cast<uint8_t>(rLeft == rRight ? kPadStickCenter : rLeft ? 0x00 : 0xFF);
        data[5] = static_cast<uint8_t>(rUp == rDown ? kPadStickCenter : rUp ? 0x00 : 0xFF);
    }

    if (port == 0)
        recordPhysicalButtons(static_cast<uint16_t>(~btns));
    const uint16_t scripted = port == 1 ? pad2ScriptedButtonsDown() : scriptedButtonsDown();
    btns &= static_cast<uint16_t>(~scripted);
    // Scripted directions push the left stick too, as the movement keys do: a recorded keyboard session moved
    // with them where the game reads only the stick (VP2's battles).
    if (scripted & (PAD_LEFT | PAD_RIGHT))
        data[6] = (scripted & PAD_LEFT) ? 0x00 : 0xFF;
    if (scripted & (PAD_UP | PAD_DOWN))
        data[7] = (scripted & PAD_UP) ? 0x00 : 0xFF;

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
