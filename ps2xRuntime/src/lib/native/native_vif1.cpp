#include "runtime/native/native_vif1.h"

#include <atomic>

namespace ps2x::native
{
    namespace
    {
        std::atomic<Vif1Observer> g_observer{nullptr};
    }

    void setVif1Observer(Vif1Observer observer)
    {
        g_observer.store(observer, std::memory_order_release);
    }

    bool vif1Observed()
    {
        return g_observer.load(std::memory_order_relaxed) != nullptr;
    }

    void observeVif1(const uint8_t *stream, size_t bytes)
    {
        if (Vif1Observer observer = g_observer.load(std::memory_order_acquire); observer && bytes != 0u)
            observer(stream, bytes);
    }
}
