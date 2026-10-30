// The host file's creation time, for the memory card HLE's directory order (kept apart: it needs <windows.h>).
#include <cstdint>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace ps2_stubs
{
    // The file's creation time in 100 ns units on Windows; 0 elsewhere (no creation time to rely on: name order).
    uint64_t mcHostCreationTime(const std::filesystem::path &path)
    {
#ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA data{};
        if (GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data))
            return (static_cast<uint64_t>(data.ftCreationTime.dwHighDateTime) << 32) | data.ftCreationTime.dwLowDateTime;
#else
        (void)path;
#endif
        return 0u;
    }
}
