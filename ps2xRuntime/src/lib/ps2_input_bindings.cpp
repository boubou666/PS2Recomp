#include "runtime/ps2_input_bindings.h"

#include "ps2_runtime.h"
#include "raylib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace ps2_input
{
    namespace
    {
        struct Name
        {
            const char *name;
            int code;
        };

        const Name kKeys[] = {
            {"A", KEY_A}, {"B", KEY_B}, {"C", KEY_C}, {"D", KEY_D}, {"E", KEY_E}, {"F", KEY_F}, {"G", KEY_G},
            {"H", KEY_H}, {"I", KEY_I}, {"J", KEY_J}, {"K", KEY_K}, {"L", KEY_L}, {"M", KEY_M}, {"N", KEY_N},
            {"O", KEY_O}, {"P", KEY_P}, {"Q", KEY_Q}, {"R", KEY_R}, {"S", KEY_S}, {"T", KEY_T}, {"U", KEY_U},
            {"V", KEY_V}, {"W", KEY_W}, {"X", KEY_X}, {"Y", KEY_Y}, {"Z", KEY_Z},
            {"0", KEY_ZERO}, {"1", KEY_ONE}, {"2", KEY_TWO}, {"3", KEY_THREE}, {"4", KEY_FOUR}, {"5", KEY_FIVE},
            {"6", KEY_SIX}, {"7", KEY_SEVEN}, {"8", KEY_EIGHT}, {"9", KEY_NINE},
            {"F1", KEY_F1}, {"F2", KEY_F2}, {"F3", KEY_F3}, {"F4", KEY_F4}, {"F5", KEY_F5}, {"F6", KEY_F6},
            {"F7", KEY_F7}, {"F8", KEY_F8}, {"F9", KEY_F9}, {"F10", KEY_F10}, {"F11", KEY_F11}, {"F12", KEY_F12},
            {"SPACE", KEY_SPACE}, {"ESCAPE", KEY_ESCAPE}, {"ENTER", KEY_ENTER}, {"TAB", KEY_TAB},
            {"BACKSPACE", KEY_BACKSPACE}, {"INSERT", KEY_INSERT}, {"DELETE", KEY_DELETE}, {"RIGHT", KEY_RIGHT},
            {"LEFT", KEY_LEFT}, {"DOWN", KEY_DOWN}, {"UP", KEY_UP}, {"PAGE_UP", KEY_PAGE_UP},
            {"PAGE_DOWN", KEY_PAGE_DOWN}, {"HOME", KEY_HOME}, {"END", KEY_END}, {"CAPS_LOCK", KEY_CAPS_LOCK},
            {"LEFT_SHIFT", KEY_LEFT_SHIFT}, {"LEFT_CONTROL", KEY_LEFT_CONTROL}, {"LEFT_ALT", KEY_LEFT_ALT},
            {"RIGHT_SHIFT", KEY_RIGHT_SHIFT}, {"RIGHT_CONTROL", KEY_RIGHT_CONTROL}, {"RIGHT_ALT", KEY_RIGHT_ALT},
            {"KP_0", KEY_KP_0}, {"KP_1", KEY_KP_1}, {"KP_2", KEY_KP_2}, {"KP_3", KEY_KP_3}, {"KP_4", KEY_KP_4},
            {"KP_5", KEY_KP_5}, {"KP_6", KEY_KP_6}, {"KP_7", KEY_KP_7}, {"KP_8", KEY_KP_8}, {"KP_9", KEY_KP_9},
            {"KP_DECIMAL", KEY_KP_DECIMAL}, {"KP_DIVIDE", KEY_KP_DIVIDE}, {"KP_MULTIPLY", KEY_KP_MULTIPLY},
            {"KP_SUBTRACT", KEY_KP_SUBTRACT}, {"KP_ADD", KEY_KP_ADD}, {"KP_ENTER", KEY_KP_ENTER},
            {"APOSTROPHE", KEY_APOSTROPHE}, {"COMMA", KEY_COMMA}, {"MINUS", KEY_MINUS}, {"PERIOD", KEY_PERIOD},
            {"SLASH", KEY_SLASH}, {"SEMICOLON", KEY_SEMICOLON}, {"EQUAL", KEY_EQUAL},
            {"LEFT_BRACKET", KEY_LEFT_BRACKET}, {"BACKSLASH", KEY_BACKSLASH}, {"RIGHT_BRACKET", KEY_RIGHT_BRACKET},
            {"GRAVE", KEY_GRAVE}};

        const Name kPadButtons[] = {
            {"LEFT_FACE_UP", GAMEPAD_BUTTON_LEFT_FACE_UP}, {"LEFT_FACE_RIGHT", GAMEPAD_BUTTON_LEFT_FACE_RIGHT},
            {"LEFT_FACE_DOWN", GAMEPAD_BUTTON_LEFT_FACE_DOWN}, {"LEFT_FACE_LEFT", GAMEPAD_BUTTON_LEFT_FACE_LEFT},
            {"RIGHT_FACE_UP", GAMEPAD_BUTTON_RIGHT_FACE_UP}, {"RIGHT_FACE_RIGHT", GAMEPAD_BUTTON_RIGHT_FACE_RIGHT},
            {"RIGHT_FACE_DOWN", GAMEPAD_BUTTON_RIGHT_FACE_DOWN}, {"RIGHT_FACE_LEFT", GAMEPAD_BUTTON_RIGHT_FACE_LEFT},
            {"LEFT_TRIGGER_1", GAMEPAD_BUTTON_LEFT_TRIGGER_1}, {"LEFT_TRIGGER_2", GAMEPAD_BUTTON_LEFT_TRIGGER_2},
            {"RIGHT_TRIGGER_1", GAMEPAD_BUTTON_RIGHT_TRIGGER_1}, {"RIGHT_TRIGGER_2", GAMEPAD_BUTTON_RIGHT_TRIGGER_2},
            {"MIDDLE_LEFT", GAMEPAD_BUTTON_MIDDLE_LEFT}, {"MIDDLE", GAMEPAD_BUTTON_MIDDLE},
            {"MIDDLE_RIGHT", GAMEPAD_BUTTON_MIDDLE_RIGHT}, {"LEFT_THUMB", GAMEPAD_BUTTON_LEFT_THUMB},
            {"RIGHT_THUMB", GAMEPAD_BUTTON_RIGHT_THUMB}};

        const Name kPadAxes[] = {{"LEFT_X", GAMEPAD_AXIS_LEFT_X},         {"LEFT_Y", GAMEPAD_AXIS_LEFT_Y},
                                 {"RIGHT_X", GAMEPAD_AXIS_RIGHT_X},       {"RIGHT_Y", GAMEPAD_AXIS_RIGHT_Y},
                                 {"LEFT_TRIGGER", GAMEPAD_AXIS_LEFT_TRIGGER}, {"RIGHT_TRIGGER", GAMEPAD_AXIS_RIGHT_TRIGGER}};

        const Name kMouseButtons[] = {{"LEFT", MOUSE_BUTTON_LEFT}, {"RIGHT", MOUSE_BUTTON_RIGHT},
                                      {"MIDDLE", MOUSE_BUTTON_MIDDLE}, {"SIDE", MOUSE_BUTTON_SIDE},
                                      {"EXTRA", MOUSE_BUTTON_EXTRA}, {"FORWARD", MOUSE_BUTTON_FORWARD},
                                      {"BACK", MOUSE_BUTTON_BACK}};

        const char *const kActionNames[ActionCount] = {
            "up", "down", "left", "right", "cross", "circle", "square", "triangle", "l1", "r1", "l2", "r2", "l3",
            "r3", "start", "select", "left_stick_left", "left_stick_right", "left_stick_up", "left_stick_down",
            "right_stick_left", "right_stick_right", "right_stick_up", "right_stick_down"};
        const char *const kActionLabels[ActionCount] = {
            "Up", "Down", "Left", "Right", "Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2", "L3",
            "R3", "Start", "Select", "Left stick left", "Left stick right", "Left stick up", "Left stick down",
            "Right stick left", "Right stick right", "Right stick up", "Right stick down"};
        // PS2 button bits (cleared when pressed), in Action order for the 16 buttons.
        const uint16_t kButtonBits[16] = {0x0010, 0x0040, 0x0080, 0x0020, 0x4000, 0x2000, 0x8000, 0x1000,
                                          0x0400, 0x0800, 0x0100, 0x0200, 0x0002, 0x0004, 0x0008, 0x0001};

        const char *nameOf(const Name *table, size_t count, int code)
        {
            for (size_t i = 0; i < count; ++i)
                if (table[i].code == code)
                    return table[i].name;
            return nullptr;
        }

        bool codeOf(const Name *table, size_t count, const std::string &name, int &code)
        {
            for (size_t i = 0; i < count; ++i)
                if (name == table[i].name)
                {
                    code = table[i].code;
                    return true;
                }
            return false;
        }

        bool isTrigger(int axis) { return axis == GAMEPAD_AXIS_LEFT_TRIGGER || axis == GAMEPAD_AXIS_RIGHT_TRIGGER; }

        std::mutex g_mutex;
        Config g_config;
        bool g_loaded = false;

        // The window thread's mouse sample (stick deflection per axis, -1..1) and the wheel's last notches.
        std::atomic<float> g_mouseX{0.0f}, g_mouseY{0.0f};
        std::atomic<double> g_wheelUpUntil{0.0}, g_wheelDownUntil{0.0};

        double seconds()
        {
            static const auto start = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }

        Binding key(int code) { return {Source::Key, code, 1}; }
        Binding pad(int code) { return {Source::GamepadButton, code, 1}; }
        Binding axis(int code, int direction) { return {Source::GamepadAxis, code, direction}; }

        void defaults(Config &config)
        {
            config = Config{};
            for (int player = 0; player < 2; ++player)
            {
                Profile &p = config.players[player];
                p.gamepad = player;
                const struct { int action; int button; } kPad[] = {
                    {Up, GAMEPAD_BUTTON_LEFT_FACE_UP}, {Down, GAMEPAD_BUTTON_LEFT_FACE_DOWN},
                    {Left, GAMEPAD_BUTTON_LEFT_FACE_LEFT}, {Right, GAMEPAD_BUTTON_LEFT_FACE_RIGHT},
                    {Cross, GAMEPAD_BUTTON_RIGHT_FACE_DOWN}, {Circle, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT},
                    {Square, GAMEPAD_BUTTON_RIGHT_FACE_LEFT}, {Triangle, GAMEPAD_BUTTON_RIGHT_FACE_UP},
                    {L1, GAMEPAD_BUTTON_LEFT_TRIGGER_1}, {R1, GAMEPAD_BUTTON_RIGHT_TRIGGER_1},
                    {L2, GAMEPAD_BUTTON_LEFT_TRIGGER_2}, {R2, GAMEPAD_BUTTON_RIGHT_TRIGGER_2},
                    {L3, GAMEPAD_BUTTON_LEFT_THUMB}, {R3, GAMEPAD_BUTTON_RIGHT_THUMB},
                    {Start, GAMEPAD_BUTTON_MIDDLE_RIGHT}, {Select, GAMEPAD_BUTTON_MIDDLE_LEFT}};
                for (const auto &entry : kPad)
                    p.actions[entry.action].push_back(pad(entry.button));
                p.actions[LeftStickLeft].push_back(axis(GAMEPAD_AXIS_LEFT_X, -1));
                p.actions[LeftStickRight].push_back(axis(GAMEPAD_AXIS_LEFT_X, 1));
                p.actions[LeftStickUp].push_back(axis(GAMEPAD_AXIS_LEFT_Y, -1));
                p.actions[LeftStickDown].push_back(axis(GAMEPAD_AXIS_LEFT_Y, 1));
                p.actions[RightStickLeft].push_back(axis(GAMEPAD_AXIS_RIGHT_X, -1));
                p.actions[RightStickRight].push_back(axis(GAMEPAD_AXIS_RIGHT_X, 1));
                p.actions[RightStickUp].push_back(axis(GAMEPAD_AXIS_RIGHT_Y, -1));
                p.actions[RightStickDown].push_back(axis(GAMEPAD_AXIS_RIGHT_Y, 1));
            }
            // Player 1's keyboard (US key positions), as before.
            Profile &p1 = config.players[0];
            const struct { int action; int keys[2]; } kKeyboard[] = {
                {Up, {KEY_UP, KEY_W}}, {Down, {KEY_DOWN, KEY_S}}, {Left, {KEY_LEFT, KEY_A}}, {Right, {KEY_RIGHT, KEY_D}},
                {Cross, {KEY_X, KEY_SPACE}}, {Circle, {KEY_C, KEY_ESCAPE}}, {Square, {KEY_Z, KEY_KP_0}},
                {Triangle, {KEY_V, KEY_KP_1}}, {L1, {KEY_Q, 0}}, {R1, {KEY_E, 0}}, {L2, {KEY_LEFT_SHIFT, 0}},
                {R2, {KEY_RIGHT_SHIFT, 0}}, {Start, {KEY_ENTER, 0}}, {Select, {KEY_TAB, 0}}};
            for (const auto &entry : kKeyboard)
                for (int code : entry.keys)
                    if (code)
                        p1.actions[entry.action].push_back(key(code));
        }

        std::string trim(const std::string &text)
        {
            const size_t first = text.find_first_not_of(" \t\r");
            if (first == std::string::npos)
                return {};
            return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
        }

        void load()
        {
            defaults(g_config);
            std::ifstream file(configPath());
            if (!file)
                return;
            std::string line;
            int player = -1;
            bool mouse = false;
            bool cleared[2][ActionCount] = {};
            while (std::getline(file, line))
            {
                line = trim(line.substr(0, line.find('#')));
                if (line.empty())
                    continue;
                if (line.front() == '[')
                {
                    const std::string section = line.substr(1, line.find(']') - 1);
                    player = section == "player1" ? 0 : section == "player2" ? 1 : -1;
                    mouse = section == "mouse";
                    continue;
                }
                const size_t equal = line.find('=');
                if (equal == std::string::npos)
                    continue;
                const std::string name = trim(line.substr(0, equal)), value = trim(line.substr(equal + 1));
                if (mouse)
                {
                    if (name == "sensitivity")
                        g_config.mouse.sensitivity = static_cast<float>(std::atof(value.c_str()));
                    else if (name == "invert_y")
                        g_config.mouse.invertY = value == "true" || value == "1";
                    else if (name == "capture")
                        g_config.mouse.capture = value == "true" || value == "1";
                    continue;
                }
                if (player < 0)
                    continue;
                Profile &profile = g_config.players[player];
                if (name == "gamepad")
                {
                    profile.gamepad = std::atoi(value.c_str());
                    continue;
                }
                for (int action = 0; action < ActionCount; ++action)
                {
                    if (name != kActionNames[action])
                        continue;
                    // A line replaces the action's defaults (an empty one unbinds it).
                    if (!cleared[player][action])
                    {
                        profile.actions[action].clear();
                        cleared[player][action] = true;
                    }
                    std::stringstream list(value);
                    std::string item;
                    while (std::getline(list, item, ','))
                    {
                        Binding binding;
                        item = trim(item);
                        if (!item.empty() && parseBinding(item, binding))
                            profile.actions[action].push_back(binding);
                        else if (!item.empty())
                            std::fprintf(stderr, "[input] %s: unknown binding '%s' for %s\n", configPath().c_str(), item.c_str(), name.c_str());
                    }
                }
            }
        }

        void ensureLoaded()
        {
            if (!g_loaded)
            {
                g_loaded = true;
                load();
            }
        }

        // How far a binding is pushed, 0..1.
        float strength(const Binding &binding, int gamepad, double now)
        {
            switch (binding.source)
            {
            case Source::Key:
                return IsKeyDown(binding.code) ? 1.0f : 0.0f;
            case Source::GamepadButton:
                return IsGamepadAvailable(gamepad) && IsGamepadButtonDown(gamepad, binding.code) ? 1.0f : 0.0f;
            case Source::GamepadAxis:
            {
                if (!IsGamepadAvailable(gamepad))
                    return 0.0f;
                float value = GetGamepadAxisMovement(gamepad, binding.code);
                if (isTrigger(binding.code))
                    value = (value + 1.0f) * 0.5f; // -1 at rest, 1 fully pressed
                return std::clamp(value * static_cast<float>(binding.direction), 0.0f, 1.0f);
            }
            case Source::MouseButton:
                return IsMouseButtonDown(binding.code) ? 1.0f : 0.0f;
            case Source::MouseWheel:
                return now < (binding.direction > 0 ? g_wheelUpUntil.load() : g_wheelDownUntil.load()) ? 1.0f : 0.0f;
            case Source::MouseMove:
            {
                const float value = binding.code == 0 ? g_mouseX.load() : g_mouseY.load();
                return std::clamp(value * static_cast<float>(binding.direction), 0.0f, 1.0f);
            }
            }
            return 0.0f;
        }
    }

    std::mutex &configMutex() { return g_mutex; }

    Config &config()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensureLoaded();
        return g_config;
    }

    void resetToDefaults()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_loaded = true;
        defaults(g_config);
    }

    std::string configPath()
    {
        if (const char *path = std::getenv("PS2X_INPUT_CONFIG"); path && *path)
            return path;
        const std::filesystem::path dir = PS2Runtime::getIoPaths().elfDirectory;
        return (dir.empty() ? std::filesystem::path("input.ini") : dir / "input.ini").string();
    }

    bool save()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensureLoaded();
        std::ofstream file(configPath(), std::ios::trunc);
        if (!file)
            return false;
        file << "# PS2 pad bindings (the Controls window, F2, writes this file). One line per PS2 button or stick\n"
                "# direction, any number of bindings separated by commas: key:<KEY>, pad:<BUTTON>, axis:<AXIS>+ or -,\n"
                "# mouse:<BUTTON>, wheel:UP / wheel:DOWN, mousemove:X+ / X- / Y+ / Y-. An empty line unbinds.\n";
        for (int player = 0; player < 2; ++player)
        {
            const Profile &profile = g_config.players[player];
            file << "\n[player" << (player + 1) << "]\ngamepad = " << profile.gamepad << "\n";
            for (int action = 0; action < ActionCount; ++action)
            {
                file << kActionNames[action] << " =";
                for (size_t i = 0; i < profile.actions[action].size(); ++i)
                    file << (i ? ", " : " ") << bindingName(profile.actions[action][i]);
                file << "\n";
            }
        }
        file << "\n[mouse]\nsensitivity = " << g_config.mouse.sensitivity << "\ninvert_y = "
             << (g_config.mouse.invertY ? "true" : "false") << "\ncapture = " << (g_config.mouse.capture ? "true" : "false") << "\n";
        return static_cast<bool>(file);
    }

    PadState padState(int player)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensureLoaded();
        const Profile &profile = g_config.players[player != 0 ? 1 : 0];
        const double now = seconds();
        float value[ActionCount] = {};
        for (int action = 0; action < ActionCount; ++action)
            for (const Binding &binding : profile.actions[action])
                value[action] = std::max(value[action], strength(binding, profile.gamepad, now));
        PadState state;
        for (int button = 0; button < 16; ++button)
            if (value[button] > 0.5f)
                state.buttons &= static_cast<uint16_t>(~kButtonBits[button]);
        auto stick = [&value](int negative, int positive)
        {
            const float v = std::clamp(value[positive] - value[negative], -1.0f, 1.0f);
            return static_cast<uint8_t>(std::lround(127.5f + v * 127.5f));
        };
        state.lx = stick(LeftStickLeft, LeftStickRight);
        state.ly = stick(LeftStickUp, LeftStickDown);
        state.rx = stick(RightStickLeft, RightStickRight);
        state.ry = stick(RightStickUp, RightStickDown);
        // PS2X_INPUT_LOG=1: each change of a player's buttons.
        static const bool log = std::getenv("PS2X_INPUT_LOG") != nullptr;
        static uint16_t s_lastButtons[2] = {0xFFFFu, 0xFFFFu};
        if (log && state.buttons != s_lastButtons[player != 0 ? 1 : 0])
        {
            s_lastButtons[player != 0 ? 1 : 0] = state.buttons;
            std::fprintf(stderr, "[input] t=%.2f player %d buttons %04x\n", now, player != 0 ? 2 : 1, state.buttons);
        }
        return state;
    }

    namespace
    {
        std::atomic<bool> g_uiOpen{false};
    }

    void setUiOpen(bool open) { g_uiOpen.store(open); }

    void frameUpdate()
    {
        const bool uiOpen = g_uiOpen.load();
        MouseSettings mouse;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            ensureLoaded();
            mouse = g_config.mouse;
        }
        // Cursor capture while playing: the motion is then unbounded (no screen edge), and the cursor hidden.
        static bool s_captured = false;
        const bool capture = mouse.capture && !uiOpen && IsWindowFocused();
        if (capture != s_captured)
        {
            s_captured = capture;
            if (capture)
                DisableCursor();
            else
                EnableCursor();
        }
        static const bool log = std::getenv("PS2X_INPUT_LOG") != nullptr;
        if (log)
            for (int button = MOUSE_BUTTON_LEFT; button <= MOUSE_BUTTON_MIDDLE; ++button)
                if (IsMouseButtonPressed(button))
                    std::fprintf(stderr, "[input] t=%.2f mouse button %d pressed (ui %d)\n", seconds(), button, uiOpen ? 1 : 0);
        // Mouse speed (pixels a frame, smoothed over a few frames) as stick deflection.
        static float s_x = 0.0f, s_y = 0.0f;
        const Vector2 delta = uiOpen ? Vector2{0.0f, 0.0f} : GetMouseDelta();
        s_x = s_x * 0.5f + delta.x * 0.5f;
        s_y = s_y * 0.5f + delta.y * 0.5f;
        const float scale = mouse.sensitivity / 10.0f;
        g_mouseX.store(std::clamp(s_x * scale, -1.0f, 1.0f));
        g_mouseY.store(std::clamp(s_y * scale * (mouse.invertY ? -1.0f : 1.0f), -1.0f, 1.0f));
        // A wheel notch holds its binding pressed for a moment (the game polls the pad at its own pace).
        const float wheel = uiOpen ? 0.0f : GetMouseWheelMove();
        if (wheel > 0.0f)
            g_wheelUpUntil.store(seconds() + 0.12);
        else if (wheel < 0.0f)
            g_wheelDownUntil.store(seconds() + 0.12);
    }

    const char *actionName(int action) { return action >= 0 && action < ActionCount ? kActionNames[action] : "?"; }
    const char *actionLabel(int action) { return action >= 0 && action < ActionCount ? kActionLabels[action] : "?"; }

    std::string bindingName(const Binding &binding)
    {
        const char *name = nullptr;
        switch (binding.source)
        {
        case Source::Key:
            name = nameOf(kKeys, std::size(kKeys), binding.code);
            return name ? std::string("key:") + name : "key:" + std::to_string(binding.code);
        case Source::GamepadButton:
            name = nameOf(kPadButtons, std::size(kPadButtons), binding.code);
            return name ? std::string("pad:") + name : "pad:" + std::to_string(binding.code);
        case Source::GamepadAxis:
            name = nameOf(kPadAxes, std::size(kPadAxes), binding.code);
            return std::string("axis:") + (name ? name : std::to_string(binding.code)) + (binding.direction > 0 ? "+" : "-");
        case Source::MouseButton:
            name = nameOf(kMouseButtons, std::size(kMouseButtons), binding.code);
            return name ? std::string("mouse:") + name : "mouse:" + std::to_string(binding.code);
        case Source::MouseWheel:
            return binding.direction > 0 ? "wheel:UP" : "wheel:DOWN";
        case Source::MouseMove:
            return std::string("mousemove:") + (binding.code == 0 ? "X" : "Y") + (binding.direction > 0 ? "+" : "-");
        }
        return "?";
    }

    bool parseBinding(const std::string &text, Binding &binding)
    {
        const size_t colon = text.find(':');
        if (colon == std::string::npos)
            return false;
        const std::string kind = text.substr(0, colon);
        std::string name = text.substr(colon + 1);
        int direction = 1;
        if ((kind == "axis" || kind == "mousemove") && !name.empty() && (name.back() == '+' || name.back() == '-'))
        {
            direction = name.back() == '+' ? 1 : -1;
            name.pop_back();
        }
        auto numberOr = [&name](const Name *table, size_t count, int &code)
        {
            if (codeOf(table, count, name, code))
                return true;
            char *end = nullptr;
            const long value = std::strtol(name.c_str(), &end, 10);
            if (end == name.c_str() || *end != '\0')
                return false;
            code = static_cast<int>(value);
            return true;
        };
        binding.direction = direction;
        if (kind == "key")
        {
            binding.source = Source::Key;
            return numberOr(kKeys, std::size(kKeys), binding.code);
        }
        if (kind == "pad")
        {
            binding.source = Source::GamepadButton;
            return numberOr(kPadButtons, std::size(kPadButtons), binding.code);
        }
        if (kind == "axis")
        {
            binding.source = Source::GamepadAxis;
            return numberOr(kPadAxes, std::size(kPadAxes), binding.code);
        }
        if (kind == "mouse")
        {
            binding.source = Source::MouseButton;
            return numberOr(kMouseButtons, std::size(kMouseButtons), binding.code);
        }
        if (kind == "wheel" && (name == "UP" || name == "DOWN"))
        {
            binding.source = Source::MouseWheel;
            binding.code = 0;
            binding.direction = name == "UP" ? 1 : -1;
            return true;
        }
        if (kind == "mousemove" && (name == "X" || name == "Y"))
        {
            binding.source = Source::MouseMove;
            binding.code = name == "X" ? 0 : 1;
            return true;
        }
        return false;
    }

    bool pollAnyInput(int gamepad, Binding &binding)
    {
        if (const int pressed = GetKeyPressed(); pressed != 0)
        {
            binding = key(pressed);
            return true;
        }
        for (const Name &button : kMouseButtons)
            if (IsMouseButtonPressed(button.code))
            {
                binding = {Source::MouseButton, button.code, 1};
                return true;
            }
        if (const float wheel = GetMouseWheelMove(); wheel != 0.0f)
        {
            binding = {Source::MouseWheel, 0, wheel > 0.0f ? 1 : -1};
            return true;
        }
        const Vector2 delta = GetMouseDelta();
        if (std::fabs(delta.x) > 40.0f || std::fabs(delta.y) > 40.0f)
        {
            const bool x = std::fabs(delta.x) >= std::fabs(delta.y);
            const float v = x ? delta.x : delta.y;
            binding = {Source::MouseMove, x ? 0 : 1, v > 0.0f ? 1 : -1};
            return true;
        }
        if (IsGamepadAvailable(gamepad))
        {
            for (const Name &button : kPadButtons)
                if (IsGamepadButtonPressed(gamepad, button.code))
                {
                    binding = pad(button.code);
                    return true;
                }
            for (const Name &a : kPadAxes)
            {
                float value = GetGamepadAxisMovement(gamepad, a.code);
                if (isTrigger(a.code))
                    value = (value + 1.0f) * 0.5f;
                if (std::fabs(value) > 0.6f)
                {
                    binding = axis(a.code, value > 0.0f ? 1 : -1);
                    return true;
                }
            }
        }
        return false;
    }
}
