#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include "runtime/ps2_input_bindings.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

    uint16_t scriptedButtonsDown()
    {
        static const std::vector<ScriptedPress> script = parsePadScript(std::getenv("PS2X_PAD_SCRIPT"));
        static const char *const liveInput = std::getenv("PS2X_PAD_INPUT");
        if (script.empty() && !liveInput)
            return 0;
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_processStart).count();
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
        return buttonsDownAt(script, std::chrono::duration<double>(std::chrono::steady_clock::now() - g_processStart).count());
    }
}

bool PSPadBackend::connected(int port) const
{
    if (port == 0)
        return true;
    static const bool pad2Script = std::getenv("PS2X_PAD2_SCRIPT") != nullptr;
    return port == 1 && (pad2Script || IsGamepadAvailable(1));
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

    // The player's bindings (runtime/ps2_input_bindings.h: input.ini, the Controls window).
    const ps2_input::PadState state = ps2_input::padState(port == 1 ? 1 : 0);
    uint16_t btns = state.buttons;
    data[4] = state.rx;
    data[5] = state.ry;
    data[6] = state.lx;
    data[7] = state.ly;

    btns &= static_cast<uint16_t>(~(port == 1 ? pad2ScriptedButtonsDown() : scriptedButtonsDown()));

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
