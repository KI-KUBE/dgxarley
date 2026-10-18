"""Static checks for the clevis role's keyfile fallback (syntax, tags, defaults, README, templates)."""

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import jinja2
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROLE = REPO / "roles" / "clevis"
VENV_BIN = REPO / ".venv" / "bin"

NEW_VARS = {
    "clevis_initramfs_generator_override": "",
    "clevis_keyfile_dracut_keyfile_timeout": "10s",
    "clevis_keyfile_auth_keyfile": "",
    "clevis_keyfile_boot_enabled": False,
    "clevis_keyfile_boot_mount": "/boot",
    "clevis_keyfile_boot_file": "clevis-keyfile/luks-fallback.key",
}


def _tool(name: str) -> str | None:
    venv = VENV_BIN / name
    if venv.exists():
        return str(venv)
    return shutil.which(name)


def _playbook(*args: str) -> subprocess.CompletedProcess[str]:
    exe = _tool("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    return subprocess.run([exe, "common.yml", *args], cwd=REPO, capture_output=True, text=True)


def test_syntax_check() -> None:
    res = _playbook("--syntax-check")
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("generator", ["dracut", "initramfs-tools"])
def test_keyfile_tag_lists_both_generator_blocks(generator: str) -> None:
    res = _playbook(
        "--tags", "clevis_keyfile", "--list-tasks", "-e", f"clevis_initramfs_generator_override={generator}"
    )
    assert res.returncode == 0, res.stdout + res.stderr
    # --list-tasks is static (blocks are not filtered by `when`): both generator paths are listed
    assert "deploy clevis-keyfile initramfs hook" in res.stdout
    assert "set crypttab field 3 / keyfile-timeout for the LUKS mapper" in res.stdout
    assert "clevis : detect LUKS mapper name from crypttab" in res.stdout
    assert "copy the key to the boot partition" in res.stdout
    assert "shred the boot key file" in res.stdout
    # the tang configuration must not be part of the keyfile-only tag
    assert "config_clevis" not in res.stdout


def test_defaults_define_new_variables() -> None:
    defaults = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    for name, value in NEW_VARS.items():
        assert name in defaults, name
        assert defaults[name] == value, name
    assert defaults["clevis_keyfile_enabled"] is False
    assert defaults["clevis_keyfile_teardown"] is False


def test_readme_defaults_table_aligned() -> None:
    lines = (ROLE / "README.md").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Variable ") and "Default" in ln)
    table = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        table.append(ln)
    assert len(table) > 5
    # every row has the same total width and its column bars at the same offsets
    bars = [m.start() for m in re.finditer(r"\|", table[0])]
    for ln in table:
        assert [m.start() for m in re.finditer(r"\|", ln)] == bars, ln
    documented = {ln.split("|")[1].strip().strip("`") for ln in table[2:]}
    assert set(NEW_VARS) <= documented


@pytest.mark.parametrize("usb", [True, False])
@pytest.mark.parametrize("embed", [True, False])
def test_dracut_template_renders(usb: bool, embed: bool) -> None:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROLE / "templates"), keep_trailing_newline=True)
    env.filters["bool"] = bool  # Ansible filter
    out = env.get_template("etc_dracut.conf.d_90-clevis-keyfile.conf.j2").render(
        clevis_keyfile_usb_enabled=usb,
        clevis_keyfile_embed_initramfs=embed,
        clevis_keyfile_path="/etc/clevis-keyfile/fallback.key",
    )
    lines = [ln for ln in out.splitlines() if not ln.startswith("#")]
    drivers = [ln for ln in lines if ln.startswith("add_drivers")]
    items = [ln for ln in lines if ln.startswith("install_items")]
    if usb:
        assert drivers == ['add_drivers+=" usb_storage uas sd_mod vfat exfat nls_cp437 nls_iso8859_1 "']
    else:
        assert drivers == []
    if embed:
        assert items == ['install_items+=" /etc/clevis-keyfile/fallback.key "']
    else:
        assert items == []


def test_shell_snippets_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    for script in [
        ROLE / "files" / "initramfs_hooks_clevis-keyfile",
        ROLE / "files" / "initramfs_local-top_clevis-keyfile",
        ROLE / "files" / "initramfs_local-bottom_clevis-keyfile",
    ]:
        shebang = script.read_text().splitlines()[0]
        shell = "sh" if "/bin/sh" in shebang else "bash"
        exe = shutil.which(shell)
        assert exe is not None
        res = subprocess.run([exe, "-n", str(script)], capture_output=True, text=True)
        assert res.returncode == 0, f"{script.name}: {res.stderr}"
        busybox = shutil.which("busybox")
        if busybox:  # initramfs runs busybox ash
            res = subprocess.run([busybox, "sh", "-n", str(script)], capture_output=True, text=True)
            assert res.returncode == 0, f"{script.name} (busybox): {res.stderr}"
    # the embedded task shells need a rendered template; syntax-check the awk instead
    awk = shutil.which("awk")
    if awk:
        res = subprocess.run(
            [awk, "--source", "BEGIN{}", "-f", str(ROLE / "files" / "clevis_keyfile_crypttab.awk"), "/dev/null"],
            capture_output=True,
            text=True,
        )
        assert res.returncode in (0, 4), res.stderr  # 4 = "no mapper line" on empty input, i.e. it parsed


@pytest.mark.parametrize("linter", ["yamllint", "ansible-lint"])
def test_linters_if_present(linter: str) -> None:
    venv = VENV_BIN / linter
    if not venv.exists():
        pytest.skip(f"{linter} not in .venv")
    target = str(ROLE)
    res = subprocess.run([str(venv), target], cwd=REPO, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def _jinja() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROLE / "templates"), keep_trailing_newline=True)
    env.filters["bool"] = bool
    env.filters["quote"] = shlex.quote
    env.filters["regex_replace"] = lambda v, a, b="": re.sub(a, b, v)
    return env


def _keyfile_conf(**kw: object) -> dict[str, str]:
    ctx: dict[str, object] = {
        "clevis_keyfile_path": "/etc/clevis-keyfile/fallback.key",
        "clevis_keyfile_usb_enabled": True,
        "clevis_keyfile_usb_label": "LUKSKEY",
        "clevis_keyfile_usb_file": "luks-fallback.key",
        "clevis_keyfile_embed_initramfs": False,
        "clevis_keyfile_boot_enabled": False,
        "clevis_keyfile_boot_file": "clevis-keyfile/luks-fallback.key",
    }
    ctx.update(kw)
    out = _jinja().get_template("etc_clevis-keyfile_keyfile.conf.j2").render(**ctx)
    return dict(ln.split("=", 1) for ln in out.splitlines() if ln and not ln.startswith("#"))


def test_keyfile_conf_renders_boot() -> None:
    conf = _keyfile_conf(clevis_keyfile_boot_enabled=True, clevis_boot_uuid="98b6a6a0-38bd-4c62-8d56-bd8b14c927a6")
    assert conf["BOOT"] == "1"
    assert conf["BOOT_DEV"] == "UUID=98b6a6a0-38bd-4c62-8d56-bd8b14c927a6"
    assert conf["BOOT_FILE"] == "clevis-keyfile/luks-fallback.key"
    # sourced by sh: must round-trip through the shell
    script = "\n".join(f"{k}={v}" for k, v in conf.items()) + '\necho "$BOOT|$BOOT_DEV|$BOOT_FILE"'
    res = subprocess.run(["sh", "-c", script], capture_output=True, text=True)
    assert res.stdout.strip() == "1|UUID=98b6a6a0-38bd-4c62-8d56-bd8b14c927a6|clevis-keyfile/luks-fallback.key"


def test_keyfile_conf_boot_disabled() -> None:
    conf = _keyfile_conf()
    assert conf["BOOT"] == "0"
    assert conf["BOOT_DEV"] == "''"
    assert conf["USB"] == "1" and conf["EMBED"] == "0"


@pytest.mark.parametrize("fstype", ["ext4", "vfat"])
def test_dracut_template_boot_fstype(fstype: str) -> None:
    out = _jinja().get_template("etc_dracut.conf.d_90-clevis-keyfile.conf.j2").render(
        clevis_keyfile_usb_enabled=False,
        clevis_keyfile_embed_initramfs=False,
        clevis_keyfile_boot_enabled=True,
        clevis_boot_fstype=fstype,
        clevis_keyfile_path="/etc/clevis-keyfile/fallback.key",
    )  # fmt: skip
    drivers = [ln for ln in out.splitlines() if ln.startswith("add_drivers")]
    assert drivers == (['add_drivers+=" vfat nls_cp437 nls_iso8859_1 "'] if fstype == "vfat" else [])


def _find_vars(node: object, key: str) -> dict[str, str] | None:
    if isinstance(node, dict):
        if key in node.get("vars", {}):
            return node["vars"]  # type: ignore[no-any-return]
        for v in node.values():
            found = _find_vars(v, key)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_vars(v, key)
            if found:
                return found
    return None


@pytest.mark.parametrize(
    ("embed", "boot", "usb", "field3", "source"),
    [
        (True, True, True, "/etc/clevis-keyfile/fallback.key", "embedded"),
        (False, True, True, "/clevis-keyfile/luks-fallback.key:UUID=U1", "boot-file"),
        (False, False, True, "/luks-fallback.key:LABEL=LUKSKEY", "usb"),
        (False, False, False, "", "none"),
    ],
)
def test_dracut_source_precedence(embed: bool, boot: bool, usb: bool, field3: str, source: str) -> None:
    tasks = yaml.safe_load((ROLE / "tasks" / "keyfile_fallback.yml").read_text())
    block_vars = _find_vars(tasks, "clevis_dracut_field3")
    assert block_vars is not None
    env = _jinja()
    env.filters["regex_replace"] = lambda v, a, b="": re.sub(a, b, v)
    ctx: dict[str, object] = {
        "clevis_keyfile_embed_initramfs": embed,
        "clevis_keyfile_boot_enabled": boot,
        "clevis_keyfile_usb_enabled": usb,
        "clevis_keyfile_path": "/etc/clevis-keyfile/fallback.key",
        "clevis_keyfile_boot_file": "clevis-keyfile/luks-fallback.key",
        "clevis_keyfile_usb_file": "luks-fallback.key",
        "clevis_keyfile_usb_label": "LUKSKEY",
        "clevis_boot_uuid": "U1",
        "clevis_boot_fstype": "ext4",
    }
    for name in ("clevis_dracut_boot_value", "clevis_dracut_field3", "clevis_dracut_source"):
        ctx[name] = env.from_string(block_vars[name]).render(**ctx).strip()
    assert ctx["clevis_dracut_field3"] == field3
    assert ctx["clevis_dracut_source"] == source
