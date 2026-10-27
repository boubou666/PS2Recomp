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

--wait 0 fails at once when busy; the default waits up to 30 min. Exit 75 when the wait runs out.

Every take, release, deadline expiry and dead holder is announced in the c2c f2f lane through the tag's c2c_call.py,
listed in ~/.config/ps2bench/announce.json ({"TAG": "/path/to/c2c_call.py"}). A tag without an announcer cannot take
the lock, and a take that cannot be announced is given back (exit 3).
"""
import argparse
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

STATE = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")), "ps2bench")
LOCAL = "local"
LEASES = os.path.join(STATE, "leases")
MAX = {"bench": 30 * 60, "comp": 90 * 60}
EX_TEMPFAIL = 75
EX_ANNOUNCE = 3
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


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def leases():
    out = []
    for name in sorted(os.listdir(LEASES)):
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


def flock_wait(fd, op, deadline, what):
    announced = False
    while True:
        try:
            fcntl.flock(fd, op | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            pass
        if time.time() >= deadline:
            return False
        if not announced:
            holders = leases()
            now = time.time()
            log(f"waiting for {what}; held by: " + ("; ".join(describe(l, now) for l in holders) or "(starting)"))
            announced = True
        time.sleep(0.5)


def acquire(mode, wait, host):
    """Returns (lock_fd) holding the lock, or None when the wait ran out."""
    give_up = time.time() + wait
    gate_path, lock_path = lock_paths(host)
    gate = os.open(gate_path, os.O_RDWR | os.O_CREAT, 0o666)
    lock = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    if not flock_wait(gate, fcntl.LOCK_EX, give_up, "the gate"):
        os.close(gate)
        os.close(lock)
        return None
    op = fcntl.LOCK_EX if mode == "bench" or host != LOCAL else fcntl.LOCK_SH
    try:
        if not flock_wait(lock, op, give_up, "the lock"):
            os.close(lock)
            return None
    finally:
        os.close(gate)
    return lock


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
    proc = subprocess.Popen(args.cmd, start_new_session=True, preexec_fn=die_with_parent)

    def forward(sig, _frame):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, forward)
    try:
        while True:
            try:
                return proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                log(f"deadline reached, killing {args.cmd[0]}")
                ended = "deadline reached, killed the job and released"
                forward(signal.SIGTERM, None)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    forward(signal.SIGKILL, None)
                    proc.wait()
                return 124
    finally:
        os.unlink(path)
        fcntl.flock(lock, fcntl.LOCK_UN)
        announce(args.tag, ended_text(lease, ended[0].upper() + ended[1:]))
        log(f"released {lease['id']}")


def cmd_take(args):
    if not announcer(args.tag):
        log(f"no announcer for tag {args.tag} in {ANNOUNCERS}")
        return EX_ANNOUNCE
    read_end, write_end = os.pipe()
    if os.fork():
        os.close(write_end)
        reply = os.read(read_end, 4096).decode()
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
    os.close(read_end)
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    lock = acquire(args.mode, args.wait, args.host)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    if lock is None:
        os.write(write_end, b"timeout")
        os._exit(0)
    deadline = time.time() + duration(args)
    lease, path = write_lease(args, deadline)
    if not announce(args.tag, f"took the {lease_text(lease)}."):
        os.unlink(path)
        os.write(write_end, b"unannounced")
        os._exit(0)
    released = False

    def stop(_sig, _frame):
        nonlocal released
        released = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.write(write_end, f"{lease['id']} ({lease['mode']}, until {time.strftime('%H:%M:%S', time.localtime(deadline))})".encode())
    os.close(write_end)
    while not released and time.time() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.time())))
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    fcntl.flock(lock, fcntl.LOCK_UN)
    announce(args.tag, ended_text(lease, "Released" if released else "Lease deadline reached, released"))
    os._exit(0)


def cmd_release(args):
    targets = [l for l in leases() if (args.id and l["id"] == args.id) or (args.tag and l["tag"] == args.tag)]
    if not targets:
        log("nothing to release")
        return 1
    for lease in targets:
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
    args = parser.parse_args()
    if getattr(args, "cmd", None) and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return {"run": cmd_run, "take": cmd_take, "release": cmd_release, "status": cmd_status}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
