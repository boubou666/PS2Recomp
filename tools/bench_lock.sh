# Shared bench lock, for ports that run games and timed benchmarks on the same machine: one at a time, so the
# measurements of one do not include the load of another. Source it from bash (Linux, MSYS2, Git Bash):
#
#   . PS2Recomp/tools/bench_lock.sh
#   bench_lock_take <port> <minutes> "<what>"   waits while another port holds the lock, then takes it
#   ...run...
#   bench_lock_release
#
# The lock file holds one line, "<port> until HH:MM (<what>)"; empty means free. A lock past its "until" time is
# stale and taken over (a crashed run must not block the others). It is emptied, not deleted (deleting files at a
# Windows drive root may be refused).
#   PS2X_BENCH_LOCK           the lock file (default D:/ps2recomp_bench.lock)
#   PS2X_BENCH_LOCK_ANNOUNCE  a command run in the background with the message as its last argument on take and
#                             release, e.g. your own client posting to the c2c f2f lane
#
# Kill only your own processes when a run has to stop (by PID, never by image name: the other ports run the same
# runner executable).

bench_lock_file() { echo "${PS2X_BENCH_LOCK:-D:/ps2recomp_bench.lock}"; }

bench_lock_announce()
{
    [ -n "$PS2X_BENCH_LOCK_ANNOUNCE" ] || return 0
    ($PS2X_BENCH_LOCK_ANNOUNCE "$1" > /dev/null 2>&1 &)
}

# Waits until the lock is free: empty, or its "until HH:MM" has passed.
bench_lock_wait()
{
    local file content until now
    file=$(bench_lock_file)
    while :; do
        content=$(tr -d '\r' < "$file" 2> /dev/null)
        [ -z "$(echo "$content" | tr -d '[:space:]')" ] && return 0
        until=$(echo "$content" | grep -o 'until [0-9]*:[0-9]*' | head -1 | cut -d' ' -f2 | tr -d ':')
        now=$(date +%H%M)
        # Past its time (allowing for a lock taken before midnight that ends after it).
        if [ -n "$until" ] && [ "$((10#$now))" -gt "$((10#$until))" ] && [ "$((10#$now - 10#$until))" -lt 1200 ]; then
            return 0
        fi
        sleep 5
    done
}

bench_lock_take()
{
    BENCH_LOCK_PORT=$1
    BENCH_LOCK_MINUTES=$2
    BENCH_LOCK_WHAT=$3
    bench_lock_wait
    BENCH_LOCK_START=$(date +%s)
    local until
    until=$(date -d "+$BENCH_LOCK_MINUTES min" +%H:%M)
    echo "$BENCH_LOCK_PORT until $until ($BENCH_LOCK_WHAT)" > "$(bench_lock_file)"
    bench_lock_announce "$BENCH_LOCK_PORT: took the bench lock ($BENCH_LOCK_WHAT): expect ~$BENCH_LOCK_MINUTES min, done around $until."
}

bench_lock_release()
{
    local file minutes
    file=$(bench_lock_file)
    # Only our own lock: a stale one may have been taken over meanwhile.
    if grep -q "^$BENCH_LOCK_PORT until" "$file" 2> /dev/null; then
        : > "$file"
    fi
    minutes=$(awk -v s="$BENCH_LOCK_START" -v e="$(date +%s)" 'BEGIN { printf "%.1f", (e - s) / 60 }')
    bench_lock_announce "$BENCH_LOCK_PORT: released the bench lock ($BENCH_LOCK_WHAT) after $minutes min, expected $BENCH_LOCK_MINUTES; box free from my side."
}
