# Shared bench lock, for ports that run games and timed benchmarks on the same machine. Source it from bash (Linux,
# MSYS2, Git Bash); it drives tools/benchlock.py, which holds the lock (see `benchlock.py --help`):
#
#   . PS2Recomp/tools/bench_lock.sh
#   bench_lock_take <port> <minutes> "<what>"   waits while another port holds the lock, then takes it (bench mode)
#   ...run...
#   bench_lock_release
#
# Prefer `python3 PS2Recomp/tools/benchlock.py run ... -- <command>` for new scripts: it also has the shared comp mode
# for builds, and kills the job at its deadline.
#
#   PS2X_BENCH_PYTHON  the Python to run benchlock.py with (default python3, else python). On Windows every port on
#                      the box must use the same kind: native CPython, not MSYS2's msys/cygwin python.
#
# Every take and release is announced in the c2c f2f lane by benchlock.py, through the port's announcer in
# ~/.config/ps2bench/announce.json ({"<port>": "/path/to/c2c_call.py"}). A port without one cannot take the lock
# (exit 3): add yours before the first run.

bench_lock_python()
{
    if [ -n "$PS2X_BENCH_PYTHON" ]; then
        echo "$PS2X_BENCH_PYTHON"
    elif command -v python3 > /dev/null 2>&1; then
        echo python3
    else
        echo python
    fi
}

bench_lock_script() { echo "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchlock.py"; }

bench_lock_take()
{
    BENCH_LOCK_PORT=$1
    "$(bench_lock_python)" "$(bench_lock_script)" take -t "$1" -m bench -e "$2" -w "$3" > /dev/null
}

bench_lock_release()
{
    [ -n "$BENCH_LOCK_PORT" ] || return 0
    "$(bench_lock_python)" "$(bench_lock_script)" release -t "$BENCH_LOCK_PORT"
}
