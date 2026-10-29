#include "MiniTest.h"
#include "runtime/native/native_renderer.h"

#include <cstdint>

using ps2x::native::Material;
using ps2x::native::blendOf;

namespace
{
    enum : uint32_t
    {
        Cs = 0,
        Cd = 1,
        Zero = 2,
        As = 0,
        Ad = 1,
        Fix = 2,
    };

    uint64_t alpha(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t fix = 0)
    {
        return a | (b << 2) | (c << 4) | (d << 6) | (uint64_t{fix} << 32);
    }
}

void register_native_blend_tests()
{
    MiniTest::Case("NativeBlend", [](TestCase &tc)
    {
        tc.Run("the GS's ALPHA values map to the renderer's blends", [](TestCase &t)
        {
            t.IsTrue(alpha(Cs, Cd, As, Cd) == 0x44u && blendOf(0x44u) == Material::Blend::Alpha, "0x44: (Cs - Cd) * As + Cd");
            t.IsTrue(alpha(Cs, Zero, As, Cd) == 0x48u && blendOf(0x48u) == Material::Blend::Additive, "0x48: Cs * As + Cd");
            t.IsTrue(alpha(Cs, Zero, Ad, Cd) == 0x58u && blendOf(0x58u) == Material::Blend::AdditiveByDestination,
                     "0x58: Cs * Ad + Cd");
            t.IsTrue(alpha(Zero, Zero, As, Cs) == 0x0au && blendOf(0x0au) == Material::Blend::Opaque, "0x0a: (0 - 0) * As + Cs");
        });

        tc.Run("A == B with D = Cs is opaque whatever C, other D are not", [](TestCase &t)
        {
            t.IsTrue(blendOf(alpha(Cs, Cs, Fix, Cs, 0x40)) == Material::Blend::Opaque, "(Cs - Cs) * FIX + Cs");
            t.IsTrue(blendOf(alpha(Cd, Cd, Ad, Cs)) == Material::Blend::Opaque, "(Cd - Cd) * Ad + Cs");
            t.IsTrue(blendOf(alpha(Zero, Zero, As, Cd)) != Material::Blend::Opaque, "(0 - 0) * As + Cd keeps the target");
        });

        tc.Run("a FIX factor with Cs * C + Cd stays additive", [](TestCase &t)
        {
            t.IsTrue(blendOf(alpha(Cs, Zero, Fix, Cd, 0x80)) == Material::Blend::Additive, "Cs * FIX + Cd");
        });
    });
}
