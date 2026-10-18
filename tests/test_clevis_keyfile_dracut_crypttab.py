"""Tests for roles/clevis/files/clevis_keyfile_crypttab.awk (dracut path of the keyfile fallback)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROLE = Path(__file__).resolve().parent.parent / "roles" / "clevis"
AWK_FILE = ROLE / "files" / "clevis_keyfile_crypttab.awk"

EMBED = "/etc/clevis-keyfile/fallback.key"
USB = "/luks-fallback.key:LABEL=LUKSKEY"
BOOT_UUID = "98b6a6a0-38bd-4c62-8d56-bd8b14c927a6"
BOOT_PREFIX = "/clevis-keyfile/luks-fallback.key:UUID="
BOOT = BOOT_PREFIX + BOOT_UUID

pytestmark = pytest.mark.skipif(shutil.which("awk") is None, reason="awk not installed")


def run_awk(
    crypttab: Path, want: str, mapper: str = "dm_crypt-0", timeout: str = "10s"
) -> subprocess.CompletedProcess[str]:
    """Run the role's awk program like the task does (-v vars, program file, crypttab as input)."""
    return subprocess.run(
        [
            "awk",
            "-v", f"m={mapper}",
            "-v", f"want={want}",
            "-v", f"emb={EMBED}",
            "-v", f"usb={USB}",
            "-v", f"bootv={BOOT}",
            "-v", f"bootp={BOOT_PREFIX}",
            "-v", f"to={timeout}",
            "-f", str(AWK_FILE),
            str(crypttab),
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip


def apply(tmp_path: Path, text: str, want: str, **kw: str) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the awk over text and return the result plus the would-be new crypttab."""
    ct = tmp_path / "crypttab"
    ct.write_text(text)
    res = run_awk(ct, want, **kw)
    return res, res.stdout


def test_task_uses_the_awk_file() -> None:
    """The task must feed the awk file into awk, not carry its own copy of the program."""
    tasks = yaml.safe_load((ROLE / "tasks" / "dracut_crypttab.yml").read_text())
    task = tasks[0]
    assert "clevis_keyfile_crypttab.awk" in task["environment"]["AWKPROG"]
    assert '"$AWKPROG"' in task["shell"]
    assert re.search(r"awk -v m=", task["shell"])
    assert "BOOT_PREFIX" in task["shell"] and "BOOT_VAL" in task["shell"]


def test_plain_none_to_usb_form(tmp_path: Path) -> None:
    res, out = apply(tmp_path, "dm_crypt-0 UUID=abc none luks\n", USB)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=10s\n"


def test_embed_form(tmp_path: Path) -> None:
    res, out = apply(tmp_path, "dm_crypt-0 UUID=abc none luks\n", EMBED)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {EMBED} luks,keyfile-timeout=10s\n"


def test_idempotent_second_run(tmp_path: Path) -> None:
    _, first = apply(tmp_path, "dm_crypt-0 UUID=abc none luks\n", USB)
    res, second = apply(tmp_path, first, USB)
    assert res.returncode == 0
    assert second == first


def test_teardown_from_managed_value(tmp_path: Path) -> None:
    text = f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=10s\n"
    res, out = apply(tmp_path, text, "")
    assert res.returncode == 0
    assert out == "dm_crypt-0 UUID=abc none luks\n"
    assert "keyfile-timeout" not in out


def test_teardown_from_embed_value(tmp_path: Path) -> None:
    res, out = apply(tmp_path, f"dm_crypt-0 UUID=abc {EMBED} luks,keyfile-timeout=10s\n", "")
    assert res.returncode == 0
    assert out == "dm_crypt-0 UUID=abc none luks\n"


def test_foreign_keyfile_refused(tmp_path: Path) -> None:
    text = "dm_crypt-0 UUID=abc /etc/other.key luks\n"
    ct = tmp_path / "crypttab"
    ct.write_text(text)
    res = run_awk(ct, USB)
    assert res.returncode == 3
    assert "unmanaged keyfile" in res.stderr
    assert ct.read_text() == text
    # teardown must not touch it either
    assert run_awk(ct, "").returncode == 3
    assert ct.read_text() == text


def test_missing_mapper(tmp_path: Path) -> None:
    ct = tmp_path / "crypttab"
    ct.write_text("other UUID=x none luks\n")
    res = run_awk(ct, USB)
    assert res.returncode == 4
    assert "no line for mapper" in res.stderr
    assert ct.read_text() == "other UUID=x none luks\n"


def test_comments_blank_lines_and_other_mappers_preserved(tmp_path: Path) -> None:
    text = (
        "# crypttab header\n"
        "\n"
        "   # indented comment dm_crypt-0 UUID=zzz none luks\n"
        "swap\t/dev/sda2\t/dev/urandom\tswap,cipher=aes-xts-plain64\n"
        "dm_crypt-0 UUID=abc none luks\n"
        "dm_crypt-1 UUID=def none luks,discard\n"
        "\n"
    )
    res, out = apply(tmp_path, text, USB)
    assert res.returncode == 0
    lines = text.split("\n")
    out_lines = out.split("\n")
    assert out_lines[:4] == lines[:4]
    assert out_lines[5:] == lines[5:]
    assert out_lines[4] == f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=10s"


def test_tab_separated_fields(tmp_path: Path) -> None:
    res, out = apply(tmp_path, "dm_crypt-0\tUUID=abc\tnone\tluks\n", USB)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=10s\n"


def test_two_field_line(tmp_path: Path) -> None:
    res, out = apply(tmp_path, "dm_crypt-0 UUID=abc\n", USB)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {USB} keyfile-timeout=10s\n"


def test_one_field_line_fails(tmp_path: Path) -> None:
    res, _ = apply(tmp_path, "dm_crypt-0\n", USB)
    assert res.returncode == 3
    assert "no device field" in res.stderr


def test_existing_keyfile_timeout_replaced_not_duplicated(tmp_path: Path) -> None:
    res, out = apply(tmp_path, f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=5s\n", USB)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=10s\n"
    assert out.count("keyfile-timeout") == 1


def test_custom_timeout(tmp_path: Path) -> None:
    res, out = apply(tmp_path, "dm_crypt-0 UUID=abc none luks\n", USB, timeout="30s")
    assert res.returncode == 0
    assert out.rstrip().endswith("luks,keyfile-timeout=30s")


def test_option_order_preserved(tmp_path: Path) -> None:
    res, out = apply(tmp_path, "dm_crypt-0 UUID=abc none discard,luks\n", USB)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {USB} discard,luks,keyfile-timeout=10s\n"
    res2, back = apply(tmp_path, out, "")
    assert res2.returncode == 0
    assert back == "dm_crypt-0 UUID=abc none discard,luks\n"


def test_teardown_empty_options_become_none(tmp_path: Path) -> None:
    res, out = apply(tmp_path, f"dm_crypt-0 UUID=abc {USB} keyfile-timeout=10s\n", "")
    assert res.returncode == 0
    assert out == "dm_crypt-0 UUID=abc none none\n"


DEMO_LINE_IN = "dm_crypt-0 UUID=9edc9306-3675-4490-a5ce-cb6d85c37ed5 none luks\n"
DEMO_LINE_OUT = f"dm_crypt-0 UUID=9edc9306-3675-4490-a5ce-cb6d85c37ed5 {BOOT} luks,keyfile-timeout=10s\n"


def test_boot_file_form_demo_line(tmp_path: Path) -> None:
    res, out = apply(tmp_path, DEMO_LINE_IN, BOOT)
    assert res.returncode == 0
    assert out == DEMO_LINE_OUT
    assert out == (
        "dm_crypt-0 UUID=9edc9306-3675-4490-a5ce-cb6d85c37ed5 "
        "/clevis-keyfile/luks-fallback.key:UUID=" + BOOT_UUID + " luks,keyfile-timeout=10s\n"
    )


def test_boot_file_form_idempotent(tmp_path: Path) -> None:
    res, out = apply(tmp_path, DEMO_LINE_OUT, BOOT)
    assert res.returncode == 0
    assert out == DEMO_LINE_OUT


def test_boot_file_form_teardown(tmp_path: Path) -> None:
    res, out = apply(tmp_path, DEMO_LINE_OUT, "")
    assert res.returncode == 0
    assert out == DEMO_LINE_IN


def test_boot_file_form_changed_uuid_still_managed(tmp_path: Path) -> None:
    old = DEMO_LINE_OUT.replace(BOOT_UUID, "11111111-2222-3333-4444-555555555555")
    res, out = apply(tmp_path, old, BOOT)
    assert res.returncode == 0
    assert out == DEMO_LINE_OUT
    res, out = apply(tmp_path, old, "")
    assert res.returncode == 0
    assert out == DEMO_LINE_IN


def test_switch_between_managed_sources(tmp_path: Path) -> None:
    res, out = apply(tmp_path, f"dm_crypt-0 UUID=abc {USB} luks,keyfile-timeout=10s\n", BOOT)
    assert res.returncode == 0
    assert out == f"dm_crypt-0 UUID=abc {BOOT} luks,keyfile-timeout=10s\n"


def test_similar_foreign_keyfile_still_refused(tmp_path: Path) -> None:
    ct = tmp_path / "crypttab"
    ct.write_text("dm_crypt-0 UUID=abc /other.key:UUID=98b6a6a0 luks\n")
    assert run_awk(ct, BOOT).returncode == 3
