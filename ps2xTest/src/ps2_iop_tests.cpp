#include "MiniTest.h"
#include "ps2x/iop/ps2_path.h"
#include "../../ps2xIOP/tests/iop_compat_test_support.h"

namespace
{
    iop_test::Irx getOtherDataModule(uint32_t eeSource, uint32_t iopDestination, uint32_t size)
    {
        iop_test::Irx image;
        image.words(0u, {
            0x27BDFFE0u, 0xAFBF001Cu, // save ra
            0x3C040000u,
            0x3C050000u | (eeSource >> 16u), 0x34A50000u | (eeSource & 0xFFFFu),
            0x3C060000u | (iopDestination >> 16u), 0x34C60000u | (iopDestination & 0xFFFFu),
            0x3C070000u | (size >> 16u), 0x34E70000u | (size & 0xFFFFu),
            0xAFA00010u,
            0x0C00401Du, 0u, // jal 0x10074: sceSifGetOtherData
            0x8FBF001Cu, 0x27BD0020u, 0x03E00008u, 0u,
        });
        image.words(0x60u, {0x41E00000u, 0u, 0x0101u, 0x63666973u, 0x0000646Du,
                             0x03E00008u, 0x24000017u, 0u, 0u});
        return image;
    }
}

void register_ps2_iop_tests()
{
    MiniTest::Case("PS2IopSubsystem", [](TestCase &tc)
    {
        tc.Run("PS2 path parsing is shared and normalizes ISO/module names", [](TestCase &t)
        {
            const ps2x::iop::ParsedPs2Path cd = ps2x::iop::parsePs2Path("CDROM0:\\MODULES\\LIBSD.IRX;1");
            t.Equals(cd.device, ps2x::iop::Ps2PathDevice::Cdrom,
                     "device names should be case-insensitive");
            t.Equals(cd.path, std::string("MODULES/LIBSD.IRX"),
                     "separators and ISO version suffixes should normalize once");
            t.Equals(ps2x::iop::ps2PathLeafKey(cd), std::string("libsd"),
                     "module lookup should use a normalized IRX leaf key");

            const ps2x::iop::ParsedPs2Path rom = ps2x::iop::parsePs2Path("rom0:ROMVER");
            t.Equals(rom.device, ps2x::iop::Ps2PathDevice::Rom0,
                     "ROM0 should remain a distinct virtual device");
            t.IsFalse(static_cast<bool>(ps2x::iop::parsePs2Path("unknown0:file.irx")),
                      "unsupported devices must not fall through to cdrom0");
        });

        tc.Run("HLE services activate only after a recognized module load", [](TestCase &t)
        {
            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);
            t.IsFalse(subsystem.canBindRpc(0x80000701u),
                      "LIBSD RPC must not exist before LIBSD is loaded");

            const ps2x::iop::ModuleLoadResult unknown = subsystem.loadModule("rom0:NOT_A_REAL_MODULE");
            t.IsTrue(unknown.handled, "the module manager should return a real load result");
            t.IsTrue(unknown.moduleId < 0, "unknown ROM modules must fail instead of receiving fake IDs");

            const ps2x::iop::ModuleLoadResult loaded = subsystem.loadModule("rom0:LIBSD");
            t.IsTrue(loaded.moduleId > 0, "a registered no-BIOS HLE module should load");
            t.IsTrue(subsystem.canBindRpc(0x80000701u),
                     "loading LIBSD should activate its HLE RPC endpoint");

            ps2x::iop::RpcRequest request{};
            request.sid = 0x80000701u;
            request.function = 3u;
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "the activated LIBSD service should handle its RPC");
            t.Equals(host.audioCalls, 1u, "the RPC should reach the HLE audio contract");

            int32_t stopResult = -1;
            t.IsTrue(subsystem.stopModule(loaded.moduleId, &stopResult),
                     "an HLE module should have a real stoppable lifecycle");
            t.Equals(stopResult, 0, "stopping an HLE module should report success");
            t.IsFalse(subsystem.canBindRpc(0x80000701u),
                      "stopping LIBSD should deactivate its RPC endpoint");
        });

        tc.Run("unknown SID remains unhandled without a loaded IRX", [](TestCase &t)
        {
            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);

            ps2x::iop::RpcRequest request{};
            request.sid = 0xDEADC0DEu;
            request.function = 0x99u;
            const ps2x::iop::RpcResult result = subsystem.handleRpc(request);
            t.IsFalse(result.handled, "an unknown SID should not be claimed by the IOP subsystem");
            t.Equals(result.resultAddress, 0u, "an unknown SID should not return a guest result address");
            t.IsFalse(result.signalNowaitCompletion, "an unknown SID should not signal nowait completion");
            t.Equals(result.callbackPolicy, ps2x::iop::CallbackPolicy::RuntimeDefault,
                     "an unknown SID should preserve runtime callback handling");

        });

        tc.Run("sceSifGetOtherData copies EE memory into IOP RAM", [](TestCase &t)
        {
            constexpr uint32_t kEeSource = 0x800u;
            constexpr uint32_t kIopDestination = 0x10400u;
            constexpr uint8_t kPayload[8] = {0x53u, 0x49u, 0x46u, 0x21u, 0x01u, 0x02u, 0x03u, 0x04u};

            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);
            t.IsTrue(host.writeGuest(kEeSource, kPayload, sizeof(kPayload)), "the EE payload should fit");
            getOtherDataModule(kEeSource, kIopDestination, sizeof(kPayload)).install(host);

            const ps2x::iop::ModuleLoadResult loaded = subsystem.loadModuleBuffer(0x1000u);
            t.IsTrue(loaded.moduleId > 0, "the synthetic IRX should load");
            t.Equals(loaded.startResult, 0, "sceSifGetOtherData should return 0 once the data is copied");

            uint8_t copied[sizeof(kPayload)] = {};
            t.IsTrue(subsystem.readMemory(kIopDestination, copied, sizeof(copied)), "the IOP destination should be readable");
            t.IsTrue(std::memcmp(copied, kPayload, sizeof(kPayload)) == 0, "the IOP buffer should hold the EE payload");
        });

        tc.Run("IOP time spent in a call run from the EE is not run again", [](TestCase &t)
        {
            constexpr uint32_t kLoops = 50000u;
            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);
            iop_test::Irx image;
            image.words(0u, {
                0x3C080000u | (kLoops >> 16u), 0x35080000u | (kLoops & 0xFFFFu), // t0 = kLoops
                0x2508FFFFu, 0x1500FFFEu, 0u,                                     // loop: addiu t0, -1; bnez t0, loop
                0x03E00008u, 0x24020001u,                                         // return 1 (not resident)
            });
            image.install(host);

            const ps2x::iop::ModuleLoadResult loaded = subsystem.loadModuleBuffer(0x1000u);
            t.Equals(loaded.startResult, 1, "the start routine should run to its end");
            const uint64_t afterCall = subsystem.debugSnapshot().emulatorCycles;
            t.IsTrue(afterCall >= kLoops * 2u, "the start routine's loop should take IOP time");

            subsystem.runEeCycles(afterCall / 2u * 8u);
            t.Equals(subsystem.debugSnapshot().emulatorCycles, afterCall,
                     "the IOP should wait while the EE has paid for less time than the call took");
            subsystem.runEeCycles(afterCall * 8u);
            const uint64_t paid = afterCall / 2u + afterCall;
            t.IsTrue(subsystem.debugSnapshot().emulatorCycles >= paid &&
                         subsystem.debugSnapshot().emulatorCycles < paid + afterCall / 4u,
                     "then follow the EE clock, the call's time counted once");
        });

        tc.Run("sceSifGetOtherData fails for an EE range outside guest memory", [](TestCase &t)
        {
            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);
            getOtherDataModule(0x00100000u, 0x10400u, 16u).install(host);

            const ps2x::iop::ModuleLoadResult loaded = subsystem.loadModuleBuffer(0x1000u);
            t.IsTrue(loaded.moduleId > 0, "the synthetic IRX should load");
            t.Equals(loaded.startResult, -1, "a source outside EE memory should fail the transfer");
        });

    });
}
