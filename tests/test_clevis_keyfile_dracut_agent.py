"""Tests for roles/clevis/files/dracut/90clevis-keyfile/clevis-keyfile-agent.sh (dracut password agent).

The agent runs unmodified in a temp sandbox: fake ask-password directory, fake systemd-reply-password
(records argv + stdin bytes) and fake cryptsetup (accepts only the known good key). BOOT/USB sources are
simulated through the CLEVIS_KEYFILE_BOOT_DIR / CLEVIS_KEYFILE_USB_DIR test seams instead of real mounts.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

AGENT = (
    Path(__file__).resolve().parent.parent
    / "roles"
    / "clevis"
    / "files"
    / "dracut"
    / "90clevis-keyfile"
    / "clevis-keyfile-agent.sh"
)

GOOD = "G" * 20 + "oodKeyBase64Chars0123456789abcdefghijklmnopqrstuv"[:44]  # 64 chars, no newline
STALE = "S" * 64
STALE2 = "T" * 64
BOOT_REL = "clevis-keyfile/luks-fallback.key"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="sh not installed")


class Sandbox:
    """Temp directory with fake tools; run() starts the agent and collects what it did."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ask = root / "ask"
        self.run_dir = root / "run"
        self.rec = root / "reply.rec"
        self.ask.mkdir()
        self.run_dir.mkdir()
        (root / "good.key").write_bytes(GOOD.encode())
        reply = root / "reply"
        reply.write_text(
            "#!/bin/sh\n"
            f'n=$(ls "{root}"/reply.stdin.* 2>/dev/null | wc -l)\n'
            f'cat > "{root}/reply.stdin.$n"\n'
            f'echo "$@" >> "{self.rec}"\n'
        )
        cs = root / "cryptsetup"
        cs.write_text(
            "#!/bin/sh\n"
            'while [ $# -gt 0 ]; do [ "$1" = --key-file ] && K=$2; shift; done\n'
            f'cmp -s "$K" "{root}/good.key" && exit 0\n'
            "exit 2\n"
        )
        reply.chmod(0o755)
        cs.chmod(0o755)
        self.reply, self.cryptsetup = reply, cs

    def ask_file(self, name: str = "ask.test", id_: str = "cryptsetup:/dev/fake", not_after: str = "0") -> Path:
        p = self.ask / name
        p.write_text(f"[Ask]\nPID=1\nSocket=/tmp/x\nId={id_}\nNotAfter={not_after}\n")
        return p

    def conf(self, **kw: str) -> Path:
        vals = {"BOOT": "0", "USB": "0", "BOOT_DEV": "''", "BOOT_FILE": BOOT_REL, "USB_LABEL": "LUKSKEY"}
        vals |= {"USB_FILE": "luks-fallback.key"} | kw
        p = self.root / "keyfile.conf"
        p.write_text("".join(f"{k}={v}\n" for k, v in vals.items()))
        return p

    def source_dir(self, kind: str, key: str, rel: str) -> Path:
        d = self.root / kind
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(key.encode())
        return d

    def start(self, conf: Path, **env: str) -> subprocess.Popen[str]:
        full = {
            **os.environ,
            "CLEVIS_KEYFILE_CONF": str(conf),
            "CLEVIS_KEYFILE_ASK_DIR": str(self.ask),
            "CLEVIS_KEYFILE_REPLY_BIN": str(self.reply),
            "CLEVIS_KEYFILE_CRYPTSETUP_BIN": str(self.cryptsetup),
            "CLEVIS_KEYFILE_RUN_DIR": str(self.run_dir),
            "CLEVIS_KEYFILE_INTERVAL": "0.1",
            "CLEVIS_KEYFILE_BACKOFF": "0.1",
            "CLEVIS_KEYFILE_MAX_SECONDS": "6",
        }
        full.update(env)
        return subprocess.Popen(["sh", str(AGENT)], env=full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_reply(self, timeout: float = 4.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.rec.exists() and self.rec.read_text().strip():
                return True
            time.sleep(0.05)
        return False

    def replies(self) -> list[bytes]:
        return [p.read_bytes() for p in sorted(self.root.glob("reply.stdin.*"))]


@pytest.fixture
def sb(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def finish(proc: subprocess.Popen[str], settle: float = 0.6) -> tuple[int, str]:
    """Let the agent run a little longer (to catch double replies), stop it, return rc and all output."""
    time.sleep(settle)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=10)
    return proc.returncode, out + err


def test_good_boot_key_replied_once(sb: Sandbox) -> None:
    boot = sb.source_dir("boot", GOOD, BOOT_REL)
    sb.ask_file()
    proc = sb.start(sb.conf(BOOT="1", BOOT_DEV="UUID=x"), CLEVIS_KEYFILE_BOOT_DIR=str(boot))
    assert sb.wait_reply()
    rc, output = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]  # byte-identical, no trailing newline
    assert sb.rec.read_text().split() == ["1", "/tmp/x"]
    assert GOOD not in output
    assert list(sb.run_dir.iterdir()) == []  # scratch dir (key copies) removed


def test_stale_boot_good_usb_key(sb: Sandbox) -> None:
    boot = sb.source_dir("boot", STALE, BOOT_REL)
    usb = sb.source_dir("usb", GOOD, "luks-fallback.key")
    sb.ask_file()
    proc = sb.start(
        sb.conf(BOOT="1", BOOT_DEV="UUID=x", USB="1"),
        CLEVIS_KEYFILE_BOOT_DIR=str(boot),
        CLEVIS_KEYFILE_USB_DIR=str(usb),
    )
    assert sb.wait_reply()
    rc, output = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]
    assert STALE not in output and GOOD not in output
    assert "source 'boot' does not unlock" in output
    assert "answered /dev/fake from source 'usb'" in output


def test_leftover_embed_config_is_ignored(sb: Sandbox) -> None:
    legacy = sb.root / "etc" / "clevis-keyfile" / "fallback.key"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(GOOD.encode())
    sb.ask_file()
    proc = sb.start(sb.conf(EMBED="1"), CLEVIS_KEYFILE_EMBEDDED_KEY=str(legacy), CLEVIS_KEYFILE_MAX_SECONDS="1")
    rc, _ = finish(proc, settle=0.2)
    assert rc == 0
    assert not sb.rec.exists()


def test_good_usb_key(sb: Sandbox) -> None:
    usb = sb.source_dir("usb", GOOD, "luks-fallback.key")
    sb.ask_file()
    proc = sb.start(sb.conf(USB="1"), CLEVIS_KEYFILE_USB_DIR=str(usb))
    assert sb.wait_reply()
    rc, _ = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]


def test_stick_plugged_in_later(sb: Sandbox) -> None:
    sb.ask_file()
    usb = sb.root / "usb"
    proc = sb.start(sb.conf(USB="1"), CLEVIS_KEYFILE_USB_DIR=str(usb))
    time.sleep(0.8)
    assert not sb.rec.exists()
    sb.source_dir("usb", GOOD, "luks-fallback.key")
    assert sb.wait_reply()
    rc, _ = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]


def test_all_stale_no_reply_exits_when_ask_removed(sb: Sandbox) -> None:
    boot = sb.source_dir("boot", STALE, BOOT_REL)
    usb = sb.source_dir("usb", STALE2, "luks-fallback.key")
    ask = sb.ask_file()
    proc = sb.start(
        sb.conf(BOOT="1", BOOT_DEV="UUID=x", USB="1"),
        CLEVIS_KEYFILE_BOOT_DIR=str(boot),
        CLEVIS_KEYFILE_USB_DIR=str(usb),
        CLEVIS_KEYFILE_IDLE_ROUNDS="2",
    )
    time.sleep(1.0)
    assert proc.poll() is None  # never gives up while the ask file exists
    ask.unlink()
    out, err = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert not sb.rec.exists()
    # each stale key is tried once per ask file, not once per round
    assert err.count("source 'boot' does not unlock") == 1
    assert err.count("source 'usb' does not unlock") == 1
    assert STALE not in out + err and STALE2 not in out + err


def test_changed_stick_is_retried(sb: Sandbox) -> None:
    usb = sb.source_dir("usb", STALE, "luks-fallback.key")
    sb.ask_file()
    proc = sb.start(sb.conf(USB="1"), CLEVIS_KEYFILE_USB_DIR=str(usb))
    time.sleep(0.8)
    assert not sb.rec.exists()
    sb.source_dir("usb", GOOD, "luks-fallback.key")
    assert sb.wait_reply()
    rc, _ = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]


def test_non_cryptsetup_ask_ignored(sb: Sandbox) -> None:
    boot = sb.source_dir("boot", GOOD, BOOT_REL)
    sb.ask_file(id_="ask-password:other")
    proc = sb.start(
        sb.conf(BOOT="1", BOOT_DEV="UUID=x"), CLEVIS_KEYFILE_BOOT_DIR=str(boot), CLEVIS_KEYFILE_MAX_SECONDS="1"
    )
    rc, _ = finish(proc, settle=0.2)
    assert rc == 0
    assert not sb.rec.exists()


def test_expired_ask_skipped(sb: Sandbox) -> None:
    boot = sb.source_dir("boot", GOOD, BOOT_REL)
    sb.ask_file(not_after="1")  # monotonic usec 1: long past
    proc = sb.start(
        sb.conf(BOOT="1", BOOT_DEV="UUID=x"), CLEVIS_KEYFILE_BOOT_DIR=str(boot), CLEVIS_KEYFILE_MAX_SECONDS="1"
    )
    rc, _ = finish(proc, settle=0.2)
    assert rc == 0
    assert not sb.rec.exists()


def test_ask_removed_midway_exits_zero(sb: Sandbox) -> None:
    ask = sb.ask_file()
    proc = sb.start(sb.conf(), CLEVIS_KEYFILE_IDLE_ROUNDS="2")  # no source enabled at all
    time.sleep(0.5)
    assert proc.poll() is None
    ask.unlink()
    proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert not sb.rec.exists()


def test_missing_conf_exits_zero(sb: Sandbox) -> None:
    proc = sb.start(sb.root / "does-not-exist.conf")
    proc.communicate(timeout=10)
    assert proc.returncode == 0


def test_multi_host_stick_own_host_first(sb: Sandbox) -> None:
    usb = sb.source_dir("usb", STALE, "luks-keys/aaa.key")
    sb.source_dir("usb", GOOD, "luks-keys/me.key")
    sb.ask_file()
    proc = sb.start(sb.conf(USB="1", USB_FILE="''", USB_DIR="luks-keys", HOST="me"), CLEVIS_KEYFILE_USB_DIR=str(usb))
    assert sb.wait_reply()
    rc, output = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]
    assert "answered /dev/fake from source 'usb:luks-keys/me.key'" in output
    assert "does not unlock" not in output  # own key first, the other host's key is never tried


def test_multi_host_stick_tries_all_keys(sb: Sandbox) -> None:
    usb = sb.source_dir("usb", STALE, "luks-fallback.key")
    sb.source_dir("usb", STALE2, "luks-keys/aaa.key")
    sb.source_dir("usb", STALE, "luks-keys/bbb.key")
    sb.source_dir("usb", GOOD, "luks-keys/ccc.key")
    sb.source_dir("usb", STALE2, "luks-keys/notakey.txt")
    sb.ask_file()
    proc = sb.start(sb.conf(USB="1", USB_DIR="luks-keys", HOST="me"), CLEVIS_KEYFILE_USB_DIR=str(usb))
    assert sb.wait_reply()
    rc, output = finish(proc)
    assert rc == 0
    assert sb.replies() == [GOOD.encode()]
    assert "source 'usb' does not unlock" in output
    assert "source 'usb:luks-keys/aaa.key' does not unlock" in output
    assert "bbb.key" not in output  # same content as the rejected root key: not tried again
    assert "answered /dev/fake from source 'usb:luks-keys/ccc.key'" in output
    assert GOOD not in output and STALE not in output


def test_multi_host_stick_all_stale_tried_once(sb: Sandbox) -> None:
    usb = sb.source_dir("usb", STALE, "luks-keys/aaa.key")
    sb.source_dir("usb", STALE2, "luks-keys/bbb.key")
    ask = sb.ask_file()
    proc = sb.start(
        sb.conf(USB="1", USB_FILE="''", USB_DIR="luks-keys", HOST="me"),
        CLEVIS_KEYFILE_USB_DIR=str(usb),
        CLEVIS_KEYFILE_IDLE_ROUNDS="2",
    )
    time.sleep(1.0)
    ask.unlink()
    out, err = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert not sb.rec.exists()
    assert err.count("does not unlock") == 2
