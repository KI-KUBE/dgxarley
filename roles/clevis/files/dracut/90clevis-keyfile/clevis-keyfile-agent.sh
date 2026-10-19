#!/bin/sh
# Managed by Ansible (clevis role)
#
# systemd password agent for the LUKS keyfile fallback (dracut initrd). It answers
# systemd-cryptsetup's ask-password requests (/run/systemd/ask-password/ask.*, Id=cryptsetup:<dev>)
# with a key from ANY enabled source, in parallel to the stock console prompt (plymouth /
# systemd-tty-ask-password-agent), clevis-dracut (tang) and whatever else answers: the first
# valid answer wins, the ask file then disappears. crypttab is never touched.
#
# Key sources, tried every round in the order embedded -> /boot partition -> USB stick; a failing
# source never blocks the next one:
#   EMBED=1  /etc/clevis-keyfile/fallback.key inside the initrd
#   BOOT=1   BOOT_FILE on the unencrypted partition BOOT_DEV (UUID=/LABEL=/PARTUUID=), mounted read-only
#   USB=1    USB_FILE on the stick with filesystem label USB_LABEL (may be plugged in while the prompt is shown)
# A key is only sent after `cryptsetup open --test-passphrase --key-file <copy> <dev>` accepted it, so a
# stale key/stick never burns one of the prompt's tries. A rejected key is not retried for the same ask
# file until its content changes (stick swapped); after a stale/unreadable source the loop backs off.
#
# Lifetime: runs until killed (the unit conflicts with initrd-switch-root.target). With
# CLEVIS_KEYFILE_IDLE_ROUNDS=N (>0) it exits 0 after N consecutive rounds without any cryptsetup ask
# file once at least one was seen. The key is never logged.
#
# Test seams (all optional; the defaults are the production values):
#   CLEVIS_KEYFILE_CONF           config file                          (/etc/clevis-keyfile/keyfile.conf)
#   CLEVIS_KEYFILE_EMBEDDED_KEY   embedded key                         (/etc/clevis-keyfile/fallback.key)
#   CLEVIS_KEYFILE_ASK_DIR        ask-password directory               (/run/systemd/ask-password)
#   CLEVIS_KEYFILE_REPLY_BIN      systemd-reply-password               (PATH, else /usr/lib/systemd/systemd-reply-password)
#   CLEVIS_KEYFILE_CRYPTSETUP_BIN cryptsetup                           (cryptsetup)
#   CLEVIS_KEYFILE_BOOT_DIR       use this directory as the already mounted boot source (no device, no mount)
#   CLEVIS_KEYFILE_USB_DIR        same for the USB source
#   CLEVIS_KEYFILE_RUN_DIR        scratch directory                    (/run)
#   CLEVIS_KEYFILE_INTERVAL       seconds between rounds               (1)
#   CLEVIS_KEYFILE_BACKOFF        extra seconds after a stale source   (4)
#   CLEVIS_KEYFILE_IDLE_ROUNDS    exit 0 after N idle rounds           (0 = never)
#   CLEVIS_KEYFILE_MAX_SECONDS    hard run-time limit, exit 0          (0 = none)

set -u

CONF=${CLEVIS_KEYFILE_CONF:-/etc/clevis-keyfile/keyfile.conf}
EMBEDDED_KEY=${CLEVIS_KEYFILE_EMBEDDED_KEY:-/etc/clevis-keyfile/fallback.key}
ASK_DIR=${CLEVIS_KEYFILE_ASK_DIR:-/run/systemd/ask-password}
# systemd-reply-password is not in PATH on every distro (Ubuntu ships it only under /usr/lib/systemd)
REPLY_BIN=${CLEVIS_KEYFILE_REPLY_BIN:-$(command -v systemd-reply-password 2>/dev/null || echo /usr/lib/systemd/systemd-reply-password)}
CRYPTSETUP=${CLEVIS_KEYFILE_CRYPTSETUP_BIN:-cryptsetup}
RUN_DIR=${CLEVIS_KEYFILE_RUN_DIR:-/run}
INTERVAL=${CLEVIS_KEYFILE_INTERVAL:-1}
BACKOFF=${CLEVIS_KEYFILE_BACKOFF:-4}
IDLE_ROUNDS=${CLEVIS_KEYFILE_IDLE_ROUNDS:-0}
MAX_SECONDS=${CLEVIS_KEYFILE_MAX_SECONDS:-0}

log() {
    echo "clevis-keyfile-agent: $*" >&2
}

EMBED=0 BOOT=0 USB=0 BOOT_DEV='' BOOT_FILE='' USB_LABEL='' USB_FILE=''
if [ -r "$CONF" ]; then
    # shellcheck disable=SC1090
    . "$CONF"
else
    log "no config $CONF, nothing to do"
    exit 0
fi

umask 077
WORK=$(mktemp -d "$RUN_DIR/clevis-keyfile-agent.XXXXXX") || exit 1
STATE=$WORK/state
mkdir -p "$STATE"
BMNT=$RUN_DIR/clevis-keyfile-boot
UMNT=$RUN_DIR/clevis-keyfile-usb
cleanup() {
    for m in "$BMNT" "$UMNT"; do
        grep -q " $m " /proc/mounts 2>/dev/null && umount "$m" 2>/dev/null
    done
    rm -rf "$WORK"
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

# Mounts read-only $1 (device) on $2 unless already mounted there.
mount_ro() {
    grep -q " $2 " /proc/mounts 2>/dev/null && return 0
    mkdir -p "$2"
    for fstype in vfat exfat ext4; do
        mount -t "$fstype" -o ro "$1" "$2" 2>/dev/null && return 0
    done
    # ext4 with a dirty journal refuses a plain ro mount
    mount -t ext4 -o ro,noload "$1" "$2" 2>/dev/null
}

# "UUID=x" / "LABEL=x" / "PARTUUID=x" -> device node path (stdout).
resolve_dev() {
    case "$1" in
        UUID=*) echo "/dev/disk/by-uuid/${1#UUID=}" ;;
        LABEL=*) echo "/dev/disk/by-label/${1#LABEL=}" ;;
        PARTUUID=*) echo "/dev/disk/by-partuuid/${1#PARTUUID=}" ;;
        *) echo "$1" ;;
    esac
}

# fetch_from <name> <dev|""> <override-dir|""> <mountpoint> <file>
# Copies the key of one source to $WORK/key.<name> (0600). Returns 0 = copied, 1 = source
# absent (nothing to do, no backoff), 2 = source present but unreadable/empty (backoff).
fetch_from() {
    f_name=$1 f_dev=$2 f_dir=$3 f_mnt=$4 f_file=$5
    f_out=$WORK/key.$f_name
    rm -f "$f_out"
    if [ -n "$f_dir" ]; then
        f_src=$f_dir/$f_file
        [ -r "$f_src" ] || return 2
        cat "$f_src" > "$f_out" 2>/dev/null || return 2
    else
        [ -e "$f_dev" ] || return 1
        mount_ro "$f_dev" "$f_mnt" || return 2
        f_src=$f_mnt/$f_file
        if [ -r "$f_src" ]; then
            cat "$f_src" > "$f_out" 2>/dev/null
            f_rc=$?
        else
            f_rc=1
        fi
        umount "$f_mnt" 2>/dev/null
        [ "$f_rc" = 0 ] || return 2
    fi
    [ -s "$f_out" ] || return 2
    return 0
}

# Gathers the candidate keys of all enabled sources into $WORK/key.{embedded,boot,usb}.
# Sets BACKOFF_NEEDED=1 if a present source could not be read.
gather() {
    rm -f "$WORK"/key.embedded "$WORK"/key.boot "$WORK"/key.usb
    if [ "$EMBED" = 1 ] && [ -r "$EMBEDDED_KEY" ]; then
        cat "$EMBEDDED_KEY" > "$WORK/key.embedded" 2>/dev/null
    fi
    if [ "$BOOT" = 1 ]; then
        fetch_from boot "$(resolve_dev "$BOOT_DEV")" "${CLEVIS_KEYFILE_BOOT_DIR:-}" "$BMNT" "$BOOT_FILE"
        [ $? = 2 ] && BACKOFF_NEEDED=1
    fi
    if [ "$USB" = 1 ]; then
        fetch_from usb "$(resolve_dev "LABEL=$USB_LABEL")" "${CLEVIS_KEYFILE_USB_DIR:-}" "$UMNT" "$USB_FILE"
        [ $? = 2 ] && BACKOFF_NEEDED=1
    fi
}

# Milliseconds are not needed: CLOCK_MONOTONIC in microseconds ~ /proc/uptime.
now_usec() {
    read -r up _ < /proc/uptime 2>/dev/null || { echo 0; return; }
    sec=${up%%.*}
    frac=${up#*.}00
    frac=${frac%"${frac#??}"}
    echo $((sec * 1000000 + ${frac#0} * 10000))
}

# handle_ask <ask file>: try all candidate keys against one ask file.
handle_ask() {
    a_file=$1
    a_name=${a_file##*/}
    SOCKET='' ID='' NOTAFTER=0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            Socket=*) SOCKET=${line#Socket=} ;;
            Id=*) ID=${line#Id=} ;;
            NotAfter=*) NOTAFTER=${line#NotAfter=} ;;
        esac
    done < "$a_file" 2>/dev/null
    case "$ID" in
        cryptsetup:*) ;;
        *) return 0 ;;
    esac
    SEEN_ASK=1
    [ -n "$SOCKET" ] || return 0
    [ -e "$STATE/done.$a_name" ] && return 0
    case "$NOTAFTER" in
        '' | *[!0-9]*) NOTAFTER=0 ;;
    esac
    if [ "$NOTAFTER" -gt 0 ] && [ "$NOTAFTER" -lt "$(now_usec)" ]; then
        return 0
    fi
    a_dev=${ID#cryptsetup:}

    for a_src in embedded boot usb; do
        a_key=$WORK/key.$a_src
        [ -s "$a_key" ] || continue
        a_sum=$(cksum < "$a_key")
        grep -qxF "$a_src $a_sum" "$STATE/tried.$a_name" 2>/dev/null && continue
        "$CRYPTSETUP" open --test-passphrase --key-file "$a_key" "$a_dev" > /dev/null 2>&1
        a_rc=$?
        if [ "$a_rc" = 0 ]; then
            if "$REPLY_BIN" 1 "$SOCKET" < "$a_key"; then
                log "answered $a_dev from source '$a_src'"
                : > "$STATE/done.$a_name"
                return 0
            fi
            log "reply for $a_dev from source '$a_src' failed"
            BACKOFF_NEEDED=1
        elif [ "$a_rc" = 2 ]; then
            # wrong passphrase: remember this exact key for this ask file
            echo "$a_src $a_sum" >> "$STATE/tried.$a_name"
            log "source '$a_src' does not unlock $a_dev"
            BACKOFF_NEEDED=1
        else
            # device not (yet) usable: retry next round
            BACKOFF_NEEDED=1
        fi
    done
}

started=$(now_usec)
idle=0
seen_ever=0
while true; do
    BACKOFF_NEEDED=0
    SEEN_ASK=0
    have_ask=0
    for a in "$ASK_DIR"/ask.*; do
        [ -f "$a" ] && { have_ask=1; break; }
    done
    if [ "$have_ask" = 1 ]; then
        gather
        for a in "$ASK_DIR"/ask.*; do
            [ -f "$a" ] || continue
            handle_ask "$a"
        done
    fi
    # forget state of vanished ask files
    for s in "$STATE"/done.* "$STATE"/tried.*; do
        [ -e "$s" ] || continue
        n=${s##*/}
        n=${n#*.}
        [ -f "$ASK_DIR/$n" ] || rm -f "$s"
    done
    if [ "$SEEN_ASK" = 1 ]; then
        seen_ever=1
        idle=0
    elif [ "$seen_ever" = 1 ]; then
        idle=$((idle + 1))
        if [ "$IDLE_ROUNDS" -gt 0 ] && [ "$idle" -ge "$IDLE_ROUNDS" ]; then
            log "no cryptsetup ask file left, exiting"
            exit 0
        fi
    fi
    if [ "$MAX_SECONDS" -gt 0 ] && [ $((($(now_usec) - started) / 1000000)) -ge "$MAX_SECONDS" ]; then
        log "run-time limit reached, exiting"
        exit 0
    fi
    sleep "$INTERVAL"
    [ "$BACKOFF_NEEDED" = 1 ] && sleep "$BACKOFF"
done
