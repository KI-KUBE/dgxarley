"""Tests for fetch_key in roles/clevis/files/initramfs_local-top_clevis-keyfile (initramfs-tools keyfile fallback).

The functions are cut out of the script and run under sh with a fake cryptsetup (accepts only the good key);
the device path prefixes point into the sandbox and mount_ro is stubbed, the stick is a plain directory.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "roles" / "clevis" / "files" / "initramfs_local-top_clevis-keyfile"

GOOD = "G" * 64
STALE = "S" * 64
STALE2 = "T" * 64

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="sh not installed")


def _functions() -> str:
    src = SCRIPT.read_text()
    body = src[src.index("# try_key <file> <device>") : src.index("keyloop() {")]
    return body.replace("/dev/disk/by-label/", "$FAKEDEV/label/").replace("/dev/disk/by-uuid/", "$FAKEDEV/uuid/")


class Box:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.stick = root / "stick"
        self.stick.mkdir()
        (root / "dev" / "label").mkdir(parents=True)
        (root / "dev" / "uuid").mkdir(parents=True)
        (root / "dev" / "label" / "LUKSKEY").touch()
        (root / "good.key").write_text(GOOD)
        self.log = root / "cs.log"
        cs = root / "bin" / "cryptsetup"
        cs.parent.mkdir()
        cs.write_text(
            "#!/bin/sh\n"
            'while [ $# -gt 0 ]; do [ "$1" = --key-file ] && K=$2; shift; done\n'
            f'echo "$K" >> "{self.log}"\n'
            f'cmp -s "$K" "{root}/good.key" && exit 0\n'
            "exit 2\n"
        )
        cs.chmod(0o755)

    def key(self, rel: str, content: str) -> None:
        p = self.stick / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def run(
        self, rounds: int = 1, usb_file: str = "luks-fallback.key", usb_dir: str = "luks-keys", between: str = ":"
    ) -> list[str]:
        """Calls fetch_key `rounds` times in ONE shell (REJECTED persists), `between` runs between rounds."""
        calls = f"\n{between}\n".join('fetch_key /dev/fake; echo "RES $? ${KEY#$MNT/}"' for _ in range(rounds))
        script = (
            f"FAKEDEV={self.root}/dev\nBOOT=0\nUSB=1\nUSB_LABEL=LUKSKEY\nUSB_FILE='{usb_file}'\n"
            f"USB_DIR='{usb_dir}'\nHOST=me\nMNT={self.stick}\nBMNT={self.root}/boot\nREJECTED=''\n"
            + _functions()
            + "mount_ro() { return 0; }\n"
            + calls
            + "\n"
        )
        env = {**os.environ, "PATH": f"{self.root}/bin:{os.environ['PATH']}"}
        res = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=True)
        return [line[4:] for line in res.stdout.splitlines() if line.startswith("RES ")]

    def tried(self) -> list[str]:
        if not self.log.exists():
            return []
        return [p.removeprefix(f"{self.stick}/") for p in self.log.read_text().split()]


@pytest.fixture
def box(tmp_path: Path) -> Box:
    return Box(tmp_path)


def test_legacy_root_file(box: Box) -> None:
    box.key("luks-fallback.key", GOOD)
    assert box.run() == ["0 luks-fallback.key"]


def test_own_host_key_before_others(box: Box) -> None:
    box.key("luks-keys/aaa.key", STALE)
    box.key("luks-keys/me.key", GOOD)
    assert box.run() == ["0 luks-keys/me.key"]
    assert box.tried() == ["luks-keys/me.key"]


def test_other_host_keys_tried_in_order(box: Box) -> None:
    box.key("luks-fallback.key", STALE)
    box.key("luks-keys/aaa.key", STALE2)
    box.key("luks-keys/bbb.key", GOOD)
    box.key("luks-keys/zzz.txt", GOOD)
    assert box.run() == ["0 luks-keys/bbb.key"]
    assert box.tried() == ["luks-fallback.key", "luks-keys/aaa.key", "luks-keys/bbb.key"]


def test_rejected_keys_not_retried(box: Box) -> None:
    box.key("luks-keys/aaa.key", STALE)
    box.key("luks-keys/bbb.key", STALE2)
    assert box.run(rounds=3) == ["2 ", "1 ", "1 "]  # backoff only after a fresh rejection
    assert box.tried() == ["luks-keys/aaa.key", "luks-keys/bbb.key"]


def test_changed_key_is_retried(box: Box) -> None:
    box.key("luks-keys/aaa.key", STALE)
    (box.root / "good.key").write_text(GOOD + "\n")
    swap = f"printf '%s\\n' {GOOD} > {box.stick}/luks-keys/aaa.key"  # new size, same prompt
    assert box.run(rounds=2, between=swap) == ["2 ", "0 luks-keys/aaa.key"]


def test_empty_stick(box: Box) -> None:
    assert box.run() == ["1 "]
    assert box.tried() == []


def test_without_usb_dir(box: Box) -> None:
    box.key("luks-keys/me.key", GOOD)
    assert box.run(usb_dir="") == ["1 "]
