#include "iop_sysclib.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        // printf-style formatting of a guest format string. nextWord yields the next 32-bit vararg slot and
        // alignWord skips one slot when the next one is odd (MIPS o32 puts 64-bit arguments on an even slot).
        std::string formatGuest(const IopMemory &memory, const std::string &format,
                                const std::function<uint32_t()> &nextWord, const std::function<void()> &alignWord)
        {
            std::string out;
            for (size_t i = 0; i < format.size(); ++i)
            {
                if (format[i] != '%')
                {
                    out += format[i];
                    continue;
                }
                const size_t start = i++;
                if (i < format.size() && format[i] == '%')
                {
                    out += '%';
                    continue;
                }
                std::string spec = "%";
                while (i < format.size() && std::strchr("-+ #0", format[i]))
                    spec += format[i++];
                if (i < format.size() && format[i] == '*')
                {
                    spec += std::to_string(static_cast<int32_t>(nextWord()));
                    ++i;
                }
                while (i < format.size() && std::isdigit(static_cast<unsigned char>(format[i])))
                    spec += format[i++];
                if (i < format.size() && format[i] == '.')
                {
                    spec += format[i++];
                    if (i < format.size() && format[i] == '*')
                    {
                        spec += std::to_string(static_cast<int32_t>(nextWord()));
                        ++i;
                    }
                    while (i < format.size() && std::isdigit(static_cast<unsigned char>(format[i])))
                        spec += format[i++];
                }
                int longs = 0;
                while (i < format.size() && (format[i] == 'l' || format[i] == 'h' || format[i] == 'z'))
                    longs += format[i++] == 'l' ? 1 : 0;
                if (i >= format.size())
                {
                    out.append(format, start, std::string::npos);
                    break;
                }
                const char conversion = format[i];
                char buffer[512];
                switch (conversion)
                {
                case 'd':
                case 'i':
                case 'u':
                case 'x':
                case 'X':
                case 'o':
                    if (longs >= 2)
                    {
                        alignWord();
                        const uint64_t low = nextWord();
                        const uint64_t value = low | (static_cast<uint64_t>(nextWord()) << 32);
                        std::snprintf(buffer, sizeof(buffer), (spec + "ll" + conversion).c_str(),
                                      static_cast<unsigned long long>(value));
                    }
                    else
                    {
                        const uint32_t value = nextWord();
                        if (conversion == 'd' || conversion == 'i')
                            std::snprintf(buffer, sizeof(buffer), (spec + conversion).c_str(), static_cast<int32_t>(value));
                        else
                            std::snprintf(buffer, sizeof(buffer), (spec + conversion).c_str(), value);
                    }
                    out += buffer;
                    break;
                case 'p':
                    std::snprintf(buffer, sizeof(buffer), "0x%x", nextWord());
                    out += buffer;
                    break;
                case 'c':
                    std::snprintf(buffer, sizeof(buffer), (spec + 'c').c_str(), static_cast<int>(nextWord() & 0xFFu));
                    out += buffer;
                    break;
                case 's':
                {
                    const uint32_t address = nextWord();
                    const std::string text = address ? memory.readString(address, 4096u) : std::string("(null)");
                    std::vector<char> wide(text.size() + 512u);
                    std::snprintf(wide.data(), wide.size(), (spec + 's').c_str(), text.c_str());
                    out += wide.data();
                    break;
                }
                default:
                    out.append(format, start, i - start + 1u); // unknown conversion: keep it as written
                    break;
                }
            }
            return out;
        }
    }

    IopSysclib::IopSysclib(IopMemory &memory) noexcept
        : m_memory(memory)
    {
    }

    bool IopSysclib::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const uint32_t a2 = cpu.gpr[6];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        const auto compare = [&](uint32_t lhs, uint32_t rhs, uint32_t count) -> int32_t
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint8_t left = m_memory.read8(lhs + i);
                const uint8_t right = m_memory.read8(rhs + i);
                if (left != right)
                    return static_cast<int32_t>(left) - static_cast<int32_t>(right);
            }
            return 0;
        };
        const auto copy = [&](uint32_t destination, uint32_t source, uint32_t count)
        {
            for (uint32_t i = 0; i < count; ++i)
                m_memory.write8(destination + i, m_memory.read8(source + i));
        };
        const auto appendString = [&](uint32_t destination, uint32_t source, std::optional<uint32_t> maxAppend = std::nullopt)
        {
            uint32_t destinationOffset = 0;
            while (m_memory.read8(destination + destinationOffset) != 0u && destinationOffset < (1u << 20))
                ++destinationOffset;

            uint32_t sourceOffset = 0;
            while (sourceOffset < (1u << 20) && (!maxAppend || sourceOffset < *maxAppend))
            {
                const uint8_t character = m_memory.read8(source + sourceOffset);
                m_memory.write8(destination + destinationOffset + sourceOffset, character);
                ++sourceOffset;
                if (character == 0u)
                    return;
            }
            m_memory.write8(destination + destinationOffset + sourceOffset, 0u);
        };

        switch (ordinal)
        {
        case 4: // setjmp - enough for callers which only test the initial return.
            setV0(0);
            return true;
        case 5: // longjmp, TODO bc w can do it without the BIOS jmp_buf ABI.
            setV0(a1 == 0u ? 1u : a1);
            return true;
        case 6:
            setV0(static_cast<uint32_t>(std::toupper(static_cast<unsigned char>(a0))));
            return true;
        case 7:
            setV0(static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(a0))));
            return true;
        case 8: // look_ctype_table: _U 0x01, _L 0x02, _N 0x04, _S 0x08, _P 0x10, _C 0x20, _X 0x40, _B 0x80
        {
            const int c = static_cast<int>(a0 & 0xFFu);
            uint32_t flags = 0u;
            if (c < 0x80)
            {
                if (std::isupper(c))
                    flags |= 0x01u;
                if (std::islower(c))
                    flags |= 0x02u;
                if (std::isdigit(c))
                    flags |= 0x04u;
                if (std::isspace(c))
                    flags |= 0x08u;
                if (std::ispunct(c))
                    flags |= 0x10u;
                if (std::iscntrl(c))
                    flags |= 0x20u;
                if (std::isxdigit(c))
                    flags |= 0x40u;
                if (c == ' ')
                    flags |= 0x80u;
            }
            setV0(flags);
            return true;
        }
        case 9: // get_ctype_table: no table in IOP memory
            setV0(0);
            return true;
        case 10: // memchr
            for (uint32_t i = 0; i < a2; ++i)
            {
                if (m_memory.read8(a0 + i) == static_cast<uint8_t>(a1))
                {
                    setV0(a0 + i);
                    return true;
                }
            }
            setV0(0);
            return true;
        case 11:
            setV0(static_cast<uint32_t>(compare(a0, a1, a2)));
            return true;
        case 12:
            copy(a0, a1, a2);
            setV0(a0);
            return true;
        case 13:
        {
            std::vector<uint8_t> temporary(a2);
            for (uint32_t i = 0; i < a2; ++i)
                temporary[i] = m_memory.read8(a1 + i);
            (void)m_memory.writeRam(a0, temporary.data(), temporary.size());
            setV0(a0);
            return true;
        }
        case 14:
            for (uint32_t i = 0; i < a2; ++i)
                m_memory.write8(a0 + i, static_cast<uint8_t>(a1));
            setV0(a0);
            return true;
        case 15: // bcmp
            setV0(static_cast<uint32_t>(compare(a0, a1, a2)));
            return true;
        case 16: // bcopy(src,dst,n)
            copy(a1, a0, a2);
            setV0(0);
            return true;
        case 17:
            for (uint32_t i = 0; i < a1; ++i)
                m_memory.write8(a0 + i, 0u);
            setV0(0);
            return true;
        case 18: // prnt
            setV0(0);
            return true;
        case 19: // sprintf(buffer, format, ...): varargs in a2, a3, then the caller's stack from sp + 16 (MIPS o32)
        case 42: // vsprintf(buffer, format, va_list): varargs read in order from the va_list pointer
        {
            const std::string format = m_memory.readString(a1, 4096u);
            uint32_t slot = 0u;
            uint32_t list = a2;
            const uint32_t stack = cpu.gpr[29];
            const std::function<uint32_t()> nextWord = [&]() -> uint32_t
            {
                if (ordinal == 42u)
                {
                    const uint32_t value = m_memory.read32(list);
                    list += 4u;
                    return value;
                }
                const uint32_t index = slot++;
                if (index == 0u)
                    return cpu.gpr[6];
                if (index == 1u)
                    return cpu.gpr[7];
                return m_memory.read32(stack + 16u + (index - 2u) * 4u);
            };
            const std::function<void()> alignWord = [&]()
            {
                if (ordinal == 42u)
                    list = (list + 7u) & ~7u;
                else if ((slot & 1u) != 0u) // slot 0 is argument 2: even slots are even arguments
                    ++slot;
            };
            const std::string text = formatGuest(m_memory, format, nextWord, alignWord);
            (void)m_memory.writeRam(a0, text.c_str(), text.size() + 1u);
            setV0(static_cast<uint32_t>(text.size()));
            return true;
        }
        case 20:
            appendString(a0, a1);
            setV0(a0);
            return true;
        case 21: // strchr
        case 25: // index
        {
            const uint8_t needle = static_cast<uint8_t>(a1);
            for (uint32_t i = 0; i < (1u << 20); ++i)
            {
                const uint8_t character = m_memory.read8(a0 + i);
                if (character == needle)
                {
                    setV0(a0 + i);
                    return true;
                }
                if (character == 0u)
                    break;
            }
            setV0(0);
            return true;
        }
        case 22: // strcmp
            for (uint32_t i = 0; i < (1u << 20); ++i)
            {
                const uint8_t left = m_memory.read8(a0 + i);
                const uint8_t right = m_memory.read8(a1 + i);
                if (left != right)
                {
                    setV0(static_cast<uint32_t>(static_cast<int32_t>(left) - static_cast<int32_t>(right)));
                    return true;
                }
                if (left == 0u)
                    break;
            }
            setV0(0);
            return true;
        case 23: // strcpy
        {
            uint32_t i = 0;
            for (;; ++i)
            {
                const uint8_t character = m_memory.read8(a1 + i);
                m_memory.write8(a0 + i, character);
                if (character == 0u)
                    break;
            }
            setV0(a0);
            return true;
        }
        case 24: // strcspn
        {
            const std::string reject = m_memory.readString(a1, 4096u);
            uint32_t count = 0;
            for (; count < (1u << 20); ++count)
            {
                const char character = static_cast<char>(m_memory.read8(a0 + count));
                if (character == 0 || reject.find(character) != std::string::npos)
                    break;
            }
            setV0(count);
            return true;
        }
        case 26: // rindex
        case 32: // strrchr
        {
            const uint8_t needle = static_cast<uint8_t>(a1);
            uint32_t found = 0u;
            for (uint32_t i = 0; i < (1u << 20); ++i)
            {
                const uint8_t character = m_memory.read8(a0 + i);
                if (character == needle)
                    found = a0 + i;
                if (character == 0u)
                    break;
            }
            setV0(found);
            return true;
        }
        case 27:
            setV0(static_cast<uint32_t>(m_memory.readString(a0, 1u << 20).size()));
            return true;
        case 28:
            appendString(a0, a1, a2);
            setV0(a0);
            return true;
        case 29: // strncmp
            for (uint32_t i = 0; i < a2; ++i)
            {
                const uint8_t left = m_memory.read8(a0 + i);
                const uint8_t right = m_memory.read8(a1 + i);
                if (left != right)
                {
                    setV0(static_cast<uint32_t>(static_cast<int32_t>(left) - static_cast<int32_t>(right)));
                    return true;
                }
                if (left == 0u)
                    break;
            }
            setV0(0);
            return true;
        case 30: // strncpy
        {
            bool ended = false;
            for (uint32_t i = 0; i < a2; ++i)
            {
                const uint8_t character = ended ? 0u : m_memory.read8(a1 + i);
                if (character == 0u)
                    ended = true;
                m_memory.write8(a0 + i, character);
            }
            setV0(a0);
            return true;
        }
        case 31: // strpbrk
        {
            const std::string accept = m_memory.readString(a1, 4096u);
            for (uint32_t i = 0; i < (1u << 20); ++i)
            {
                const char character = static_cast<char>(m_memory.read8(a0 + i));
                if (character == 0)
                    break;
                if (accept.find(character) != std::string::npos)
                {
                    setV0(a0 + i);
                    return true;
                }
            }
            setV0(0);
            return true;
        }
        case 33: // strspn
        {
            const std::string accept = m_memory.readString(a1, 4096u);
            uint32_t count = 0;
            for (; count < (1u << 20); ++count)
            {
                const char character = static_cast<char>(m_memory.read8(a0 + count));
                if (character == 0 || accept.find(character) == std::string::npos)
                    break;
            }
            setV0(count);
            return true;
        }
        case 34: // strstr
        {
            const std::string needle = m_memory.readString(a1, 4096u);
            if (needle.empty())
            {
                setV0(a0);
                return true;
            }
            const std::string haystack = m_memory.readString(a0, 1u << 20);
            const size_t position = haystack.find(needle);
            setV0(position == std::string::npos
                      ? 0u
                      : a0 + static_cast<uint32_t>(position));
            return true;
        }
        case 35: // strtok state is intentionally not shared across modules yet.
            setV0(0);
            return true;
        case 36:
        case 38: // strtol / strtoul
        {
            const std::string value = m_memory.readString(a0, 4096u);
            char *end = nullptr;
            const int base = static_cast<int>(a2);
            const unsigned long parsed = ordinal == 36
                                             ? static_cast<unsigned long>(std::strtol(value.c_str(), &end, base))
                                             : std::strtoul(value.c_str(), &end, base);
            if (a1 != 0u)
            {
                m_memory.write32(a1, a0 + static_cast<uint32_t>(end - value.c_str()));
            }
            setV0(static_cast<uint32_t>(parsed));
            return true;
        }
        case 37: // atob
            setV0(0);
            return true;
        case 40: // _wmemcopy, count is 32-bit words
            for (uint32_t i = 0; i < a2; ++i)
                m_memory.write32(a0 + i * 4u, m_memory.read32(a1 + i * 4u));
            setV0(a0);
            return true;
        case 41:
            for (uint32_t i = 0; i < a2; ++i)
                m_memory.write32(a0 + i * 4u, a1);
            setV0(a0);
            return true;
        case 43:
            setV0(0);
            return true;
        default:
            return false;
        }
    }
}
