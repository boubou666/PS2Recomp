# Shared bench lock, for ports that run games and timed benchmarks on the same machine: one at a time, so the
# measurements of one do not include the load of another. Source it from bash (Linux, MSYS2, Git Bash):
#
#   . PS2Recomp/tools/bench_lock.sh
#   bench_lock_take <port> <minutes> "<what>"   queues behind the other waiting ports, then takes the lock
#   ...run...
#   bench_lock_release
#
# The lock file holds one line, "<port> until HH:MM (<what>)"; empty means free. A lock past its "until" time is
# stale and taken over (a crashed run must not block the others). It is emptied, not deleted (deleting files at a
# Windows drive root may be refused).
#
# Waiters take the lock in arrival order: bench_lock_take adds an entry to the queue directory, named by its
# arrival time, refreshes it while it waits, and takes the lock only when its entry is the oldest one and the lock
# is free; the entry goes once the lock is taken. An entry not refreshed for a minute belongs to a waiter that
# died and is dropped. A port that releases the lock and takes it again queues behind whoever was already waiting.
#   PS2X_BENCH_LOCK           the lock file (default D:/ps2recomp_bench.lock)
#   PS2X_BENCH_QUEUE          the queue directory (default: the lock file's path with .queue.d instead of .lock)
#   PS2X_BENCH_LOCK_ANNOUNCE  a command run in the background with the message as its last argument on take and
#                             release, e.g. your own client posting to the c2c f2f lane
#
# Kill only your own processes when a run has to stop (by PID, never by image name: the other ports run the same
# runner executable).

bench_lock_file() { echo "${PS2X_BENCH_LOCK:-D:/ps2recomp_bench.lock}"; }

bench_lock_queue_dir()
{
    local file
    file=$(bench_lock_file)
    echo "${PS2X_BENCH_QUEUE:-${file%.lock}.queue.d}"
}

bench_lock_announce()
{
    [ -n "${PS2X_BENCH_LOCK_ANNOUNCE:-}" ] || return 0
    (${PS2X_BENCH_LOCK_ANNOUNCE:-} "$1" > /dev/null 2>&1 &)
}

# True when the lock is free: empty, or its "until HH:MM" has passed.
bench_lock_is_free()
{
    local content until now
    content=$(tr -d '\r' 2> /dev/null < "$(bench_lock_file)")
    [ -z "$(echo "$content" | tr -d '[:space:]')" ] && return 0
    until=$(echo "$content" | grep -o 'until [0-9]*:[0-9]*' | head -1 | cut -d' ' -f2 | tr -d ':')
    now=$(date +%H%M)
    # Past its time (allowing for a lock taken before midnight that ends after it).
    [ -n "$until" ] && [ "$((10#$now))" -gt "$((10#$until))" ] && [ "$((10#$now - 10#$until))" -lt 1200 ]
}

# Waits until the lock is free (builds and other work that takes no lock).
bench_lock_wait()
{
    until bench_lock_is_free; do
        sleep 5
    done
}

# Drops the queue entries of waiters that stopped refreshing them, then prints the oldest entry left.
bench_lock_queue_head()
{
    local dir entry now
    dir=$(bench_lock_queue_dir)
    now=$(date +%s)
    for entry in "$dir"/*; do
        [ -f "$entry" ] || continue
        if [ "$((now - $(date -r "$entry" +%s 2> /dev/null || echo "$now")))" -gt 60 ]; then
            rm -f "$entry"
        fi
    done
    ls "$dir" 2> /dev/null | sort | head -1
}

bench_lock_take()
{
    BENCH_LOCK_PORT=$1
    BENCH_LOCK_MINUTES=$2
    BENCH_LOCK_WHAT=$3
    local dir entry until
    dir=$(bench_lock_queue_dir)
    mkdir -p "$dir"
    # Arrival time in nanoseconds, zero-padded so that names sort in arrival order.
    entry="$(printf '%020d' "$(date +%s%N)")-$BENCH_LOCK_PORT-${BASHPID:-$$}"
    : > "$dir/$entry"
    while :; do
        touch "$dir/$entry"
        if [ "$(bench_lock_queue_head)" = "$entry" ] && bench_lock_is_free; then
            break
        fi
        sleep 2
    done
    BENCH_LOCK_START=$(date +%s)
    until=$(date -d "+$BENCH_LOCK_MINUTES min" +%H:%M)
    echo "$BENCH_LOCK_PORT until $until ($BENCH_LOCK_WHAT)" > "$(bench_lock_file)"
    rm -f "$dir/$entry"
    bench_lock_announce "$BENCH_LOCK_PORT: took the bench lock ($BENCH_LOCK_WHAT): expect ~$BENCH_LOCK_MINUTES min, done around $until."
}

bench_lock_release()
{
    local file minutes
    # Nothing taken by this shell (a release without a take, e.g. from a cleanup trap): nothing to release.
    [ -n "${BENCH_LOCK_PORT:-}" ] || return 0
    file=$(bench_lock_file)
    # Only our own lock: a stale one may have been taken over meanwhile.
    if grep -q "^$BENCH_LOCK_PORT until" "$file" 2> /dev/null; then
        : > "$file"
    fi
    minutes=$(awk -v s="$BENCH_LOCK_START" -v e="$(date +%s)" 'BEGIN { printf "%.1f", (e - s) / 60 }')
    bench_lock_announce "$BENCH_LOCK_PORT: released the bench lock ($BENCH_LOCK_WHAT) after $minutes min, expected $BENCH_LOCK_MINUTES; box free from my side."
}
