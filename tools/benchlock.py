#!/usr/bin/env python3
"""Machine-wide lock for PS2Recomp port sessions: benchmarks and compilations.

Modes
  bench  exclusive: game runs whose timing matters (fps, profiles, A/B). Nothing else holds the lock.
  comp   shared: builds, recompiles and correctness game runs. Any number at once, never during a bench.

A waiting bench closes the gate: comps that arrive after it wait behind it, so builds cannot starve a bench.

--host NAME locks a remote build machine instead of this one (separate lock, same announcements). On a remote host
comp is exclusive too: one heavy job per box at a time.

Every hold is a lease with a deadline (bench max 30 min, comp max 90 min). The lock is a kernel flock held by a
process, so it is released when that process exits for any reason: the command finishes, is killed, hits its
deadline, or the session dies. Nothing has to be cleaned up by hand.

  benchlock run  -t TAG -m bench|comp -w WHY -e MIN [-d MIN] [--wait SECS] -- CMD...
        Hold the lock while CMD runs. CMD is killed (whole process group) at the deadline.
  benchlock take -t TAG -m bench|comp -w WHY -e MIN [-d MIN] [--wait SECS]
        Hold the lock from a background holder until `release` or the deadline. Prints the lease id.
  benchlock release -t TAG | --id ID
  benchlock status

-e is the holder's honest estimate of how long the job takes, announced so other sessions know when the box frees
up. -d is the hard deadline (default: twice the estimate, at least 5 min more), capped per mode.

Linux uses flock and a forked holder; Windows (native CPython, e.g. MSYS2 UCRT64 or python.org) uses byte-range
locks, a detached holder and a Job object. Every process sharing one lock must use the same kind of Python: MSYS2's
msys/cygwin Python and native Python do not see each other's locks.

--wait 0 fails at once when busy; the default waits up to 30 min. Exit 75 when the wait runs out.

Every take, release, deadline expiry and dead holder is announced in the c2c f2f lane through the tag's c2c_call.py,
listed in ~/.config/ps2bench/announce.json ({"TAG": "/path/to/c2c_call.py"}). A tag without an announcer cannot take
the lock, and a take that cannot be announced is given back (exit 3).
"""
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

WINDOWS = os.name == "nt"
if WINDOWS:
    import msvcrt
    from ctypes import wintypes
else:
    import fcntl

STATE = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")), "ps2bench")
LOCAL = "local"
LEASES = os.path.join(STATE, "leases")
MAX = {"bench": 30 * 60, "comp": 90 * 60}
EX_TEMPFAIL = 75
EX_ANNOUNCE = 3
# Windows has no shared file lock: comp holds one of these bytes of the lock file, bench and remote hosts all of them.
SLOTS = 64
ANNOUNCERS = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "ps2bench",
                          "announce.json")


def log(msg):
    print(f"[benchlock] {msg}", file=sys.stderr, flush=True)


def announcer(tag):
    try:
        with open(ANNOUNCERS) as f:
            path = json.load(f).get(tag)
    except (OSError, ValueError):
        return None
    return os.path.expanduser(path) if path else None


def announce(tag, text):
    script = announcer(tag)
    if not script:
        log(f"no announcer for tag {tag} in {ANNOUNCERS}")
        return False
    for _ in range(2):
        try:
            out = subprocess.run([sys.executable, "-I", script, "c2c_say", json.dumps({"text": text})],
                                 capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL).stdout
        except subprocess.TimeoutExpired:
            continue
        if "Said to" in out:
            return True
    log(f"could not announce in f2f: {text}")
    return False


def clock(t):
    return time.strftime("%H:%M", time.localtime(t))


def lock_name(lease):
    return f"{lease['mode']} lock" + ("" if host_of(lease) == LOCAL else f" on {host_of(lease)}")


def lease_text(lease):
    return (f"{lock_name(lease)} ({lease['why']}): expect ~{lease['expect']} min, done around "
            f"{clock(lease['start'] + lease['expect'] * 60)}, hard deadline {clock(lease['deadline'])}")


def ended_text(lease, how):
    held = round((time.time() - lease["start"]) / 60, 1)
    return (f"{how} the {lock_name(lease)} ({lease['why']}) after {held} min, expected {lease['expect']}; "
            f"box free from my side.")


if WINDOWS:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def pid_alive(pid):
    if WINDOWS:
        # os.kill(pid, 0) would terminate the process on Windows.
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, not ours
        code = wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Job:
    """The job of a `run`: killed as a whole at the deadline, and with its holder when that dies."""

    def __init__(self, cmd):
        if WINDOWS:
            self.handle = kernel32.CreateJobObjectW(None, None)
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            kernel32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))
            self.proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            kernel32.AssignProcessToJobObject(self.handle, int(self.proc._handle))
        else:
            self.proc = subprocess.Popen(cmd, start_new_session=True, preexec_fn=die_with_parent)

    def signal(self, sig):
        if WINDOWS:
            kernel32.TerminateJobObject(self.handle, 1)
            return
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def kill(self):
        self.signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.signal(getattr(signal, "SIGKILL", signal.SIGTERM))
            self.proc.wait()


def leases():
    out = []
    for name in sorted(os.listdir(LEASES)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(LEASES, name)
        try:
            with open(path) as f:
                lease = json.load(f)
        except (OSError, ValueError):
            continue
        if pid_alive(lease["pid"]):
            out.append(lease)
        else:
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            announce(lease["tag"], ended_text(lease, "benchlock: holder died, freed"))
    return out


def host_of(lease):
    return lease.get("host", LOCAL)


def lock_paths(host):
    if host == LOCAL:
        return os.path.join(STATE, "gate"), os.path.join(STATE, "lock")
    return os.path.join(STATE, f"gate@{host}"), os.path.join(STATE, f"lock@{host}")


def describe(lease, now):
    left = int(lease["deadline"] - now)
    held = int(now - lease["start"])
    where = "" if host_of(lease) == LOCAL else f"@{host_of(lease)}"
    return (f"{lease['mode']:5}{where} {lease['tag']} - {lease['why']} (id {lease['id']}, pid {lease['pid']}, "
            f"held {held // 60}m{held % 60:02d}s, expected {lease['expect']}m, done around "
            f"{clock(lease['start'] + lease['expect'] * 60)}, hard deadline in {left // 60}m{left % 60:02d}s)")


def wait_until(attempt, deadline, what):
    announced = False
    while True:
        if attempt():
            return True
        if time.time() >= deadline:
            return False
        if not announced:
            holders = leases()
            now = time.time()
            log(f"waiting for {what}; held by: " + ("; ".join(describe(l, now) for l in holders) or "(starting)"))
            announced = True
        time.sleep(0.5)


def try_flock(fd, op):
    try:
        fcntl.flock(fd, op | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def try_range(fd, start, length):
    os.lseek(fd, start, os.SEEK_SET)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, length)
        return True
    except OSError:
        return False


class Held:
    def __init__(self, fd, start=0, length=0):
        self.fd, self.start, self.length = fd, start, length

    def release(self):
        if WINDOWS:
            os.lseek(self.fd, self.start, os.SEEK_SET)
            msvcrt.locking(self.fd, msvcrt.LK_UNLCK, self.length)
        else:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def acquire(mode, wait, host):
    """Returns a Held lock, or None when the wait ran out."""
    give_up = time.time() + wait
    gate_path, lock_path = lock_paths(host)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    gate = os.open(gate_path, flags, 0o666)
    lock = os.open(lock_path, flags, 0o666)
    exclusive = mode == "bench" or host != LOCAL
    if not wait_until(lambda: try_range(gate, 0, 1) if WINDOWS else try_flock(gate, fcntl.LOCK_EX), give_up,
                      "the gate"):
        os.close(gate)
        os.close(lock)
        return None
    held = None
    try:
        if WINDOWS:
            slot = []

            def attempt():
                if exclusive:
                    return try_range(lock, 0, SLOTS)
                for i in range(SLOTS):
                    if try_range(lock, i, 1):
                        slot.append(i)
                        return True
                return False

            if wait_until(attempt, give_up, "the lock"):
                held = Held(lock, 0, SLOTS) if exclusive else Held(lock, slot[0], 1)
        elif wait_until(lambda: try_flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH), give_up,
                        "the lock"):
            held = Held(lock)
    finally:
        os.close(gate)
    if held is None:
        os.close(lock)
    return held


def release_requested(path):
    return os.path.exists(path[:-len(".json")] + ".release")


def write_lease(args, deadline):
    lease = {"id": f"{args.tag}-{os.getpid()}", "pid": os.getpid(), "tag": args.tag, "mode": args.mode,
             "why": args.why, "host": args.host, "expect": args.expect, "start": time.time(), "deadline": deadline}
    path = os.path.join(LEASES, f"{os.getpid()}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(lease, f)
    os.replace(path + ".tmp", path)
    return lease, path


def duration(args):
    mins = args.deadline if args.deadline is not None else max(2 * args.expect, args.expect + 5)
    mins = max(mins, args.expect)
    if mins * 60 > MAX[args.mode]:
        log(f"deadline capped at {MAX[args.mode] // 60} min for {args.mode}")
        mins = MAX[args.mode] // 60
    return mins * 60


def die_with_parent():
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def cmd_run(args):
    if not args.cmd:
        sys.exit("benchlock run: no command after --")
    if not announcer(args.tag):
        log(f"no announcer for tag {args.tag} in {ANNOUNCERS}")
        return EX_ANNOUNCE
    lock = acquire(args.mode, args.wait, args.host)
    if lock is None:
        log("gave up waiting")
        return EX_TEMPFAIL
    deadline = time.time() + duration(args)
    lease, path = write_lease(args, deadline)
    if not announce(args.tag, f"took the {lease_text(lease)}."):
        os.unlink(path)
        return EX_ANNOUNCE
    log(f"acquired {describe(lease, time.time())}")
    ended = "released"
    job = Job(args.cmd)

    def forward(sig, _frame):
        job.signal(sig)

    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), forward)
    try:
        while True:
            try:
                return job.proc.wait(timeout=min(1.0, max(0.0, deadline - time.time())))
            except subprocess.TimeoutExpired:
                if release_requested(path):
                    log(f"release requested, killing {args.cmd[0]}")
                    ended = "release requested, killed the job and released"
                    job.kill()
                    return 143
                if time.time() < deadline:
                    continue
                log(f"deadline reached, killing {args.cmd[0]}")
                ended = "deadline reached, killed the job and released"
                job.kill()
                return 124
    finally:
        os.unlink(path)
        lock.release()
        announce(args.tag, ended_text(lease, ended[0].upper() + ended[1:]))
        log(f"released {lease['id']}")


def take_reply(reply):
    if not reply:
        log("holder died before acquiring")
        return 1
    if reply.startswith("timeout"):
        log("gave up waiting")
        return EX_TEMPFAIL
    if reply.startswith("unannounced"):
        log("could not announce in f2f, lock given back")
        return EX_ANNOUNCE
    print(reply)
    return 0


def hold(args, reply_fd):
    """The holder of a `take`: acquires, answers on reply_fd, then holds until release or the deadline."""
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    lock = acquire(args.mode, args.wait, args.host)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    if lock is None:
        os.write(reply_fd, b"timeout")
        return
    deadline = time.time() + duration(args)
    lease, path = write_lease(args, deadline)
    if not announce(args.tag, f"took the {lease_text(lease)}."):
        os.unlink(path)
        os.write(reply_fd, b"unannounced")
        return
    released = False

    def stop(_sig, _frame):
        nonlocal released
        released = True

    signal.signal(signal.SIGTERM, stop)
    for name in ("SIGHUP", "SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)
    os.write(reply_fd, f"{lease['id']} ({lease['mode']}, until {time.strftime('%H:%M:%S', time.localtime(deadline))})".encode())
    os.close(reply_fd)
    while not released and time.time() < deadline:
        released = release_requested(path)
        time.sleep(min(1.0, max(0.0, deadline - time.time())))
    for done in (path, path[:-len(".json")] + ".release"):
        try:
            os.unlink(done)
        except FileNotFoundError:
            pass
    lock.release()
    announce(args.tag, ended_text(lease, "Released" if released else "Lease deadline reached, released"))


def cmd_take(args):
    if not announcer(args.tag):
        log(f"no announcer for tag {args.tag} in {ANNOUNCERS}")
        return EX_ANNOUNCE
    if WINDOWS:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        holder = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_hold", json.dumps(vars(args))],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, creationflags=flags)
        return take_reply(holder.stdout.read().decode())
    read_end, write_end = os.pipe()
    if os.fork():
        os.close(write_end)
        return take_reply(os.read(read_end, 4096).decode())
    os.close(read_end)
    os.setsid()
    hold(args, write_end)
    os._exit(0)


def cmd_release(args):
    targets = [l for l in leases() if (args.id and l["id"] == args.id) or (args.tag and l["tag"] == args.tag)]
    if not targets:
        log("nothing to release")
        return 1
    for lease in targets:
        if WINDOWS:
            open(os.path.join(LEASES, f"{lease['pid']}.release"), "w").close()
        else:
            os.kill(lease["pid"], signal.SIGTERM)
        for _ in range(600):
            if not pid_alive(lease["pid"]):
                break
            time.sleep(0.1)
        log(f"released {lease['id']}")
    return 0


def cmd_status(_args):
    now = time.time()
    held = leases()
    if not held:
        print("free")
    for lease in held:
        print(describe(lease, now))
    return 0


def main():
    os.makedirs(LEASES, exist_ok=True)
    parser = argparse.ArgumentParser(prog="benchlock", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "take"):
        p = sub.add_parser(name)
        p.add_argument("-t", "--tag", required=True)
        p.add_argument("-m", "--mode", required=True, choices=("bench", "comp"))
        p.add_argument("-w", "--why", required=True)
        p.add_argument("-e", "--expect", type=int, required=True, help="expected minutes, announced")
        p.add_argument("-d", "--deadline", type=int, help="hard deadline in minutes")
        p.add_argument("--wait", type=int, default=30 * 60)
        p.add_argument("--host", default=LOCAL, help="remote build machine to lock instead of this one")
        if name == "run":
            p.add_argument("cmd", nargs=argparse.REMAINDER)
    p = sub.add_parser("release")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--tag")
    g.add_argument("--id")
    sub.add_parser("status")
    if len(sys.argv) == 3 and sys.argv[1] == "_hold":
        hold(argparse.Namespace(**json.loads(sys.argv[2])), os.dup(sys.stdout.fileno()))
        return 0
    args = parser.parse_args()
    if getattr(args, "cmd", None) and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return {"run": cmd_run, "take": cmd_take, "release": cmd_release, "status": cmd_status}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
