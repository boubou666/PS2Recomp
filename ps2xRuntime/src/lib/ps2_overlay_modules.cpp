#include "ps2_overlay_modules.h"

#include <atomic>
#include <cstring>
#include <iostream>

namespace ps2_overlays
{
    namespace
    {
        // Function-local statics: generated modules register during static
        // initialization, before any other global of this file is guaranteed to exist.
        std::vector<Module> &registry()
        {
            static std::vector<Module> modules;
            return modules;
        }

        // Bumped when the registry changes, so a cached Module pointer is never used past it.
        std::atomic<uint32_t> &registryGeneration()
        {
            static std::atomic<uint32_t> generation{0};
            return generation;
        }

        ResidentResolver &resolverSlot()
        {
            static ResidentResolver resolver = nullptr;
            return resolver;
        }

        bool tableSlot(const Module &module, uint32_t address, uint32_t &slot)
        {
            if ((address & 3u) != 0u || address < module.tableBase || address >= module.tableEnd)
            {
                return false;
            }
            slot = (address - module.tableBase) >> 2;
            return slot < module.slotCount;
        }
    }

    void registerModule(const Module &module)
    {
        for (const Module &existing : registry())
        {
            if (std::strcmp(existing.name, module.name) == 0)
            {
                std::cerr << "[overlay] duplicate module registration: " << module.name << std::endl;
                return;
            }
        }
        registry().push_back(module);
        registryGeneration().fetch_add(1u, std::memory_order_relaxed);
    }

    void clearModules()
    {
        registry().clear();
        resolverSlot() = nullptr;
        registryGeneration().fetch_add(1u, std::memory_order_relaxed);
    }

    const std::vector<Module> &modules()
    {
        return registry();
    }

    const Module *findModule(const char *name)
    {
        for (const Module &module : registry())
        {
            if (std::strcmp(module.name, name) == 0)
            {
                return &module;
            }
        }
        return nullptr;
    }

    const Module *guardResolver(const uint8_t *rdram, uint32_t address,
                                const std::vector<const Module *> &candidates)
    {
        (void)address;
        if (rdram == nullptr)
        {
            return nullptr;
        }
        for (const Module *module : candidates)
        {
            const uint32_t offset = module->loadAddress & PS2_RAM_MASK;
            if (module->guard == nullptr || module->guardSize == 0u ||
                offset + module->guardSize > PS2_RAM_SIZE)
            {
                continue;
            }
            if (std::memcmp(rdram + offset, module->guard, module->guardSize) == 0)
            {
                return module;
            }
        }
        return nullptr;
    }

    void setResidentResolver(ResidentResolver resolver)
    {
        resolverSlot() = resolver;
    }

    bool coversAddress(uint32_t address)
    {
        for (const Module &module : registry())
        {
            uint32_t slot = 0u;
            if (tableSlot(module, address, slot))
            {
                return true;
            }
        }
        return false;
    }

    const Module *residentModuleFor(const uint8_t *rdram, uint32_t address)
    {
        // Every guest dispatch into an overlay range comes here. The module found last is checked first: its guard
        // still in memory means it is still the resident one (the guards, module headers, differ between modules).
        thread_local const Module *t_lastResident = nullptr;
        thread_local uint32_t t_lastGeneration = 0u;
        thread_local std::vector<const Module *> t_candidates;
        const bool guardResolution = resolverSlot() == nullptr;
        const uint32_t generation = registryGeneration().load(std::memory_order_relaxed);
        if (t_lastGeneration != generation)
        {
            t_lastResident = nullptr;
            t_lastGeneration = generation;
        }
        if (guardResolution && t_lastResident != nullptr && rdram != nullptr)
        {
            uint32_t slot = 0u;
            const Module &last = *t_lastResident;
            const uint32_t offset = last.loadAddress & PS2_RAM_MASK;
            if (tableSlot(last, address, slot) && last.guard != nullptr && last.guardSize != 0u &&
                offset + last.guardSize <= PS2_RAM_SIZE && std::memcmp(rdram + offset, last.guard, last.guardSize) == 0)
            {
                return t_lastResident;
            }
        }
        t_candidates.clear();
        for (const Module &module : registry())
        {
            uint32_t slot = 0u;
            if (tableSlot(module, address, slot))
            {
                t_candidates.push_back(&module);
            }
        }
        if (t_candidates.empty())
        {
            return nullptr;
        }
        const ResidentResolver resolver = guardResolution ? &guardResolver : resolverSlot();
        const Module *module = resolver(rdram, address, t_candidates);
        if (guardResolution && module != nullptr)
        {
            t_lastResident = module;
        }
        return module;
    }

    PS2Runtime::RecompiledFunction *residentEntry(const uint8_t *rdram, uint32_t address)
    {
        const Module *module = residentModuleFor(rdram, address);
        uint32_t slot = 0u;
        if (module == nullptr || !tableSlot(*module, address, slot))
        {
            return nullptr;
        }
        return &module->table[slot];
    }

    bool replaceModuleFunction(const char *moduleName, uint32_t address, PS2Runtime::RecompiledFunction func)
    {
        const Module *module = findModule(moduleName);
        uint32_t slot = 0u;
        if (module == nullptr || !tableSlot(*module, address, slot))
        {
            std::cerr << "[overlay] cannot replace 0x" << std::hex << address << std::dec
                      << " in module " << moduleName << std::endl;
            return false;
        }
        module->table[slot] = func;
        return true;
    }
}
