# Shared bench lock, for ports that run games, builds and timed benchmarks on the same machine. It has two slots:
# a game run takes one, so two games can run at once (native runs leave most of the machine free); a build or a
# timed measurement takes the bench alone (both slots), so its numbers do not include the load of another.
# Source it from bash (Linux, MSYS2, Git Bash):
#
#   . PS2Recomp/tools/bench_lock.sh
#   bench_lock_take <port> <minutes> "<what>" [alone]   queues behind the other waiting ports, then takes a slot
#                                                       (alone: both, once both are free)
#   ...run...
#   bench_lock_release
#   bench_lock_queue                                    the waiters in order, and the open priority requests
#   bench_lock_ask_priority <port> <minutes> "<why>"    (while <port> waits, from another shell) asks the ports
#                                                       ahead of it to let it pass
#   bench_lock_yield <requester> <port>                 <port>, waiting ahead of <requester>, lets it pass
#
# Each slot is a file holding one line, "<port> until HH:MM (<what>)"; empty means free. A slot past its "until"
# time is stale and taken over (a crashed run must not block the others). It is emptied, not deleted (deleting
# files at a Windows drive root may be refused). A script from before the second slot only sees the first one.
#
# Waiters take the lock in arrival order: bench_lock_take adds an entry to the queue directory, named by its
# arrival time, refreshes it while it waits, and takes the lock only when its entry is the oldest one and the lock
# is free; the entry goes once the lock is taken. An entry not refreshed for a minute belongs to a waiter that
# died and is dropped. A port that releases the lock and takes it again queues behind whoever was already waiting.
#
# Priority: arrival order is fair, but a two-minute check can sit behind a 45-minute build, or a run a human waits on
# live behind unattended work. A waiting port can ask to pass (bench_lock_ask_priority, with how long it needs and
# why; the request is posted with PS2X_BENCH_LOCK_ANNOUNCE), and only the ports waiting ahead of it can let it, each
# for itself (bench_lock_yield moves the requester's entry just ahead of the yielder's). Nobody loses their place
# without agreeing, and silence keeps the order: a request lapses after ten minutes. One open request per port, a port
# asks at most once every 30 minutes, and the minutes it gives are its deadline once it has the lock (its slot is
# stale past them, like any slot). Good reasons to yield: a human waiting live beats unattended work; a short job
# beats a long one when the long one loses little; a fix that blocks other ports beats routine measurement. Discuss
# it on f2f.
#   PS2X_BENCH_LOCK           the first slot's file (default D:/ps2recomp_bench.lock)
#   PS2X_BENCH_LOCK2          the second slot's file (default: the first's path with 2.lock instead of .lock)
#   PS2X_BENCH_QUEUE          the queue directory (default: the lock file's path with .queue.d instead of .lock)
#   PS2X_BENCH_LOCK_ANNOUNCE  a command run in the background with the message as its last argument on take and
#                             release, e.g. your own client posting to the c2c f2f lane
#
# Kill only your own processes when a run has to stop (by PID, never by image name: the other ports run the same
# runner executable).

bench_lock_file() { echo "${PS2X_BENCH_LOCK:-D:/ps2recomp_bench.lock}"; }

bench_lock_file2()
{
    local file
    file=$(bench_lock_file)
    echo "${PS2X_BENCH_LOCK2:-${file%.lock}2.lock}"
}

bench_lock_queue_dir()
{
    local file
    file=$(bench_lock_file)
    echo "${PS2X_BENCH_QUEUE:-${file%.lock}.queue.d}"
}

# The priority requests: <port>.ask holds "<minutes>|<why>" (its mtime is the request's time), <port>.last the time of
# the port's last request.
bench_lock_ask_dir()
{
    local file
    file=$(bench_lock_file)
    echo "${file%.lock}.ask.d"
}

bench_lock_announce()
{
    [ -n "${PS2X_BENCH_LOCK_ANNOUNCE:-}" ] || return 0
    (${PS2X_BENCH_LOCK_ANNOUNCE:-} "$1" > /dev/null 2>&1 &)
}

# True when a slot is free (default: the first): empty, or its "until HH:MM" has passed.
bench_lock_slot_is_free()
{
    local content until now
    content=$(tr -d '\r' 2> /dev/null < "${1:-$(bench_lock_file)}")
    [ -z "$(echo "$content" | tr -d '[:space:]')" ] && return 0
    until=$(echo "$content" | grep -o 'until [0-9]*:[0-9]*' | head -1 | cut -d' ' -f2 | tr -d ':')
    now=$(date +%H%M)
    # Past its time (allowing for a lock taken before midnight that ends after it).
    [ -n "$until" ] && [ "$((10#$now))" -gt "$((10#$until))" ] && [ "$((10#$now - 10#$until))" -lt 1200 ]
}

# True when the whole bench is free (both slots).
bench_lock_is_free()
{
    bench_lock_slot_is_free "$(bench_lock_file)" && bench_lock_slot_is_free "$(bench_lock_file2)"
}

# Waits until the whole bench is free (work that takes no lock).
bench_lock_wait()
{
    until bench_lock_is_free; do
        sleep 5
    done
}

# The queue's entries, oldest first: "<arrival, ns, 20 digits>-<port>-<pid>".
bench_lock_entries()
{
    ls "$(bench_lock_queue_dir)" 2> /dev/null | grep -E '^[0-9]{20}-' | sort
}

# Drops the queue entries of waiters that stopped refreshing them and the lapsed priority requests, then prints the
# oldest entry left.
bench_lock_queue_head()
{
    local dir entry now ask
    dir=$(bench_lock_queue_dir)
    now=$(date +%s)
    for entry in $(bench_lock_entries); do
        if [ "$((now - $(date -r "$dir/$entry" +%s 2> /dev/null || echo "$now")))" -gt 60 ]; then
            rm -f "$dir/$entry"
        fi
    done
    for ask in "$(bench_lock_ask_dir)"/*.ask; do
        [ -f "$ask" ] || continue
        [ "$((now - $(date -r "$ask" +%s 2> /dev/null || echo "$now")))" -gt 600 ] && rm -f "$ask"
    done
    bench_lock_entries | head -1
}

bench_lock_entry_port() { echo "$1" | sed -E 's/^[0-9]{20}-(.*)-[0-9]+$/\1/'; }

# The newest queue entry of a port (the one waiting now).
bench_lock_entry_of()
{
    local entry last=
    for entry in $(bench_lock_entries); do
        [ "$(bench_lock_entry_port "$entry")" = "$1" ] && last=$entry
    done
    echo "$last"
}

bench_lock_queue()
{
    local entry ask position=0 port
    for entry in $(bench_lock_entries); do
        position=$((position + 1))
        port=$(bench_lock_entry_port "$entry")
        ask="$(bench_lock_ask_dir)/$port.ask"
        if [ -f "$ask" ]; then
            echo "$position. $port (asks to pass: ~$(cut -d'|' -f1 "$ask") min, $(cut -d'|' -f2- "$ask"))"
        else
            echo "$position. $port"
        fi
    done
    echo "slot 1: $(tr -d '\r' < "$(bench_lock_file)" 2> /dev/null)"
    echo "slot 2: $(tr -d '\r' < "$(bench_lock_file2)" 2> /dev/null)"
}

# Asks the ports waiting ahead of <port> to let it pass (see "Priority" above).
bench_lock_ask_priority()
{
    local port=$1 minutes=$2 why=$3 dir entry ahead= other
    dir=$(bench_lock_ask_dir)
    mkdir -p "$dir"
    entry=$(bench_lock_entry_of "$port")
    [ -n "$entry" ] || { echo "bench_lock_ask_priority: $port is not waiting in the queue" >&2; return 1; }
    [ -f "$dir/$port.ask" ] && { echo "bench_lock_ask_priority: $port already has an open request" >&2; return 1; }
    if [ -f "$dir/$port.last" ] && [ "$(($(date +%s) - $(date -r "$dir/$port.last" +%s)))" -lt 1800 ]; then
        echo "bench_lock_ask_priority: $port asked less than 30 min ago" >&2
        return 1
    fi
    for other in $(bench_lock_entries); do
        [ "$other" = "$entry" ] && break
        ahead="$ahead $(bench_lock_entry_port "$other")"
    done
    [ -n "$ahead" ] || { echo "bench_lock_ask_priority: nobody is waiting ahead of $port" >&2; return 1; }
    echo "$minutes|$why" > "$dir/$port.ask"
    : > "$dir/$port.last"
    bench_lock_announce "[bench] $port asks to pass in the bench queue: ~$minutes min, $why. Waiting ahead:$ahead. Each can let it pass with: bench_lock_yield $port <your port> (silence keeps the order; the request lapses in 10 min)."
}

# <port>, waiting ahead of <requester>, lets it pass: the requester's entry moves just ahead of <port>'s.
bench_lock_yield()
{
    local requester=$1 port=$2 dir mine theirs stamp
    dir=$(bench_lock_queue_dir)
    mine=$(bench_lock_entry_of "$port")
    theirs=$(bench_lock_entry_of "$requester")
    [ -n "$mine" ] || { echo "bench_lock_yield: $port is not waiting in the queue" >&2; return 1; }
    [ -n "$theirs" ] || { echo "bench_lock_yield: $requester is not waiting in the queue" >&2; return 1; }
    [ -f "$(bench_lock_ask_dir)/$requester.ask" ] || { echo "bench_lock_yield: $requester has no open request" >&2; return 1; }
    [[ "$theirs" > "$mine" ]] || { echo "bench_lock_yield: $requester is already ahead of $port" >&2; return 1; }
    stamp=$((10#${mine%%-*} - 1))
    mv "$dir/$theirs" "$dir/$(printf '%020d' "$stamp")-${theirs#*-}"
    bench_lock_announce "[bench] $port lets $requester pass in the bench queue."
}

bench_lock_take()
{
    BENCH_LOCK_PORT=$1
    BENCH_LOCK_MINUTES=$2
    BENCH_LOCK_WHAT=$3
    BENCH_LOCK_FILES=()
    local dir entry until alone=0 first second
    [ "${4:-}" = alone ] && alone=1
    first=$(bench_lock_file)
    second=$(bench_lock_file2)
    dir=$(bench_lock_queue_dir)
    mkdir -p "$dir"
    # Arrival time in nanoseconds, zero-padded so that names sort in arrival order. A yield renames the entry (it moves
    # ahead), so each round finds it again by port and pid, and refreshes it without re-creating the old name.
    local self="$BENCH_LOCK_PORT-${BASHPID:-$$}" found
    entry="$(printf '%020d' "$(date +%s%N)")-$self"
    : > "$dir/$entry"
    while :; do
        found=$(bench_lock_entries | grep -F -- "-$self" | grep -E -- "-$self\$" | head -1)
        if [ -n "$found" ]; then
            entry=$found
        else
            : > "$dir/$entry" # dropped (this shell was suspended for over a minute): queue again
        fi
        touch -c "$dir/$entry"
        if [ "$(bench_lock_queue_head)" = "$entry" ]; then
            if [ "$alone" = 1 ]; then
                bench_lock_slot_is_free "$first" && bench_lock_slot_is_free "$second" && BENCH_LOCK_FILES=("$first" "$second")
            elif bench_lock_slot_is_free "$first"; then
                BENCH_LOCK_FILES=("$first")
            elif bench_lock_slot_is_free "$second"; then
                BENCH_LOCK_FILES=("$second")
            fi
            [ "${#BENCH_LOCK_FILES[@]}" -gt 0 ] && break
        fi
        sleep 2
    done
    BENCH_LOCK_START=$(date +%s)
    # A port that asked to pass holds the lock for the minutes it asked with, at most.
    local asked
    asked=$(cut -d'|' -f1 "$(bench_lock_ask_dir)/$BENCH_LOCK_PORT.ask" 2> /dev/null)
    if [[ "$asked" =~ ^[0-9]+$ ]] && [ "$asked" -lt "$BENCH_LOCK_MINUTES" ]; then
        BENCH_LOCK_MINUTES=$asked
    fi
    until=$(date -d "+$BENCH_LOCK_MINUTES min" +%H:%M)
    local file
    for file in "${BENCH_LOCK_FILES[@]}"; do
        echo "$BENCH_LOCK_PORT until $until ($BENCH_LOCK_WHAT)" > "$file"
    done
    rm -f "$dir/$entry" "$(bench_lock_ask_dir)/$BENCH_LOCK_PORT.ask"
    local how="a slot of the bench"
    [ "$alone" = 1 ] && how="the bench alone"
    bench_lock_announce "$BENCH_LOCK_PORT: took $how ($BENCH_LOCK_WHAT): expect ~$BENCH_LOCK_MINUTES min, done around $until."
}

bench_lock_release()
{
    local file minutes
    # Nothing taken by this shell (a release without a take, e.g. from a cleanup trap): nothing to release.
    [ -n "${BENCH_LOCK_PORT:-}" ] || return 0
    # Only our own slots: a stale one may have been taken over meanwhile.
    for file in "${BENCH_LOCK_FILES[@]:-$(bench_lock_file)}"; do
        if grep -q "^$BENCH_LOCK_PORT until" "$file" 2> /dev/null; then
            : > "$file"
        fi
    done
    minutes=$(awk -v s="$BENCH_LOCK_START" -v e="$(date +%s)" 'BEGIN { printf "%.1f", (e - s) / 60 }')
    bench_lock_announce "$BENCH_LOCK_PORT: released the bench lock ($BENCH_LOCK_WHAT) after $minutes min, expected $BENCH_LOCK_MINUTES; box free from my side."
}
