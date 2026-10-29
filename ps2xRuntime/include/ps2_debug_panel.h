#ifndef PS2_DEBUG_PANEL_H
#define PS2_DEBUG_PANEL_H

class PS2Runtime;

class PS2DebugPanel
{
public:
    void initialize();
    void shutdown();
    void draw(PS2Runtime &runtime);

    bool isVisible() const { return m_visible; }
    void setVisible(bool visible) { m_visible = visible; }
    void toggleVisible() { m_visible = !m_visible; }

private:
    // The pad bindings window (F2): runtime/ps2_input_bindings.h.
    void drawControlsWindow();

    bool m_initialized = false;
    bool m_visible = true;
    bool m_controlsVisible = false;
    int m_controlsPlayer = 0;
    int m_bindingAction = -1; // the action waiting for an input in the Controls window
    bool m_debugExpanded = false;
    bool m_showRegisters = true;
    unsigned int m_memoryAddress = 0x00100000u;
    unsigned int m_memoryBytes = 0x100u;
};

#endif // PS2_DEBUG_PANEL_H
