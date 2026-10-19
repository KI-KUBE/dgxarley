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
DRACUT_MODULE = ROLE / "files" / "dracut" / "90clevis-keyfile"
VENV_BIN = REPO / ".venv" / "bin"

NEW_VARS = {
    "clevis_initramfs_generator_override": "",
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
    assert "deploy the clevis-keyfile dracut module" in res.stdout
    assert "deploy dracut config (add the module)" in res.stdout
    assert "verify keyfile fallback in the dracut initramfs" in res.stdout
    assert "fail on a legacy crypttab edit (never edited silently)" in res.stdout
    assert "remove the clevis-keyfile dracut module and config" in res.stdout
    # the crypttab editor of the old dracut variant is gone
    assert "set crypttab field 3" not in res.stdout
    assert "edit /etc/crypttab line" not in res.stdout
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


SH_FILES = [
    ROLE / "files" / "initramfs_hooks_clevis-keyfile",
    ROLE / "files" / "initramfs_local-top_clevis-keyfile",
    ROLE / "files" / "initramfs_local-bottom_clevis-keyfile",
    DRACUT_MODULE / "clevis-keyfile-agent.sh",
]


@pytest.mark.parametrize("script", SH_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("shell", ["sh", "dash", "busybox"])
def test_shell_snippets_syntax(script: Path, shell: str) -> None:
    exe = shutil.which(shell)
    if exe is None:
        pytest.skip(f"{shell} not available")  # initramfs runs busybox ash / dash
    cmd = [exe, "sh", "-n", str(script)] if shell == "busybox" else [exe, "-n", str(script)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"{script.name} ({shell}): {res.stderr}"
    assert script.read_text().startswith("#!/bin/sh")


def test_module_setup_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = DRACUT_MODULE / "module-setup.sh"
    res = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    text = script.read_text()
    for fn in ("check()", "depends()", "installkernel()", "install()"):
        assert fn in text
    assert "systemd systemd-ask-password crypt" in text
    assert "/usr/libexec/clevis-keyfile-agent" in text
    assert "cryptsetup.target.wants" in text


def test_agent_unit_file() -> None:
    unit = DRACUT_MODULE / "clevis-keyfile-agent.service"
    sections: dict[str, dict[str, str]] = {}
    cur = ""
    for ln in unit.read_text().splitlines():
        if ln.startswith("[") and ln.endswith("]"):
            cur = ln[1:-1]
            sections[cur] = {}
        elif "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            sections[cur][k] = v
    assert sections["Unit"]["DefaultDependencies"] == "no"
    assert sections["Unit"]["Before"] == "cryptsetup.target"
    assert "initrd-switch-root.target" in sections["Unit"]["Conflicts"]
    assert "systemd-udevd.service" in sections["Unit"]["After"]
    assert sections["Service"]["ExecStart"] == "/usr/libexec/clevis-keyfile-agent"
    assert sections["Service"]["Type"] == "simple"
    assert sections["Service"]["KillMode"] == "process"
    assert sections["Install"]["WantedBy"] == "cryptsetup.target"
    analyze = shutil.which("systemd-analyze")
    if analyze:
        # the ExecStart binary does not exist on the test host: only syntax problems count
        res = subprocess.run([analyze, "verify", str(unit)], capture_output=True, text=True)
        assert "Failed to parse" not in res.stderr and "Unknown section" not in res.stderr, res.stderr


def test_dracut_conf_adds_module() -> None:
    text = (ROLE / "files" / "dracut" / "90-clevis-keyfile.conf").read_text()
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert lines == ['add_dracutmodules+=" clevis-keyfile "']


def test_removed_crypttab_variant_is_gone() -> None:
    for gone in (
        ROLE / "tasks" / "dracut_crypttab.yml",
        ROLE / "files" / "clevis_keyfile_crypttab.awk",
        ROLE / "templates" / "etc_dracut.conf.d_90-clevis-keyfile.conf.j2",
    ):
        assert not gone.exists(), gone
    defaults = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    assert "clevis_keyfile_dracut_keyfile_timeout" not in defaults
    readme = (ROLE / "README.md").read_text()
    assert "precedence" not in readme.lower()


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


@pytest.mark.parametrize("embed", [True, False])
@pytest.mark.parametrize("usb", [True, False])
@pytest.mark.parametrize("boot", [True, False])
def test_keyfile_conf_combinations(embed: bool, usb: bool, boot: bool) -> None:
    conf = _keyfile_conf(
        clevis_keyfile_embed_initramfs=embed,
        clevis_keyfile_usb_enabled=usb,
        clevis_keyfile_boot_enabled=boot,
        clevis_boot_uuid="U-1",
    )
    assert (conf["EMBED"], conf["USB"], conf["BOOT"]) == tuple("1" if x else "0" for x in (embed, usb, boot))
    assert conf["BOOT_DEV"] == ("UUID=U-1" if boot else "''")
    # the dracut agent and the initramfs-tools scripts source the file with sh
    script = "\n".join(f"{k}={v}" for k, v in conf.items()) + '\necho "$EMBED$USB$BOOT|$USB_LABEL|$USB_FILE"'
    res = subprocess.run(["sh", "-c", script], capture_output=True, text=True)
    assert res.stdout.strip() == f"{int(embed)}{int(usb)}{int(boot)}|LUKSKEY|luks-fallback.key"
