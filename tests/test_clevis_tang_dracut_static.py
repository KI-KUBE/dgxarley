"""Static checks for the clevis role's tang-on-dracut path (tasks, template, defaults, README)."""

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROLE = REPO / "roles" / "clevis"
VENV_BIN = REPO / ".venv" / "bin"

DRACUT_VARS: dict[str, Any] = {
    "clevis_dracut_extra_modules": ["clevis-pin-tang", "clevis-pin-sss"],
    "clevis_dracut_net_timeout_carrier": 15,
    "clevis_dracut_net_dhcp_retry": 3,
}


def _playbook(*args: str) -> subprocess.CompletedProcess[str]:
    exe = VENV_BIN / "ansible-playbook"
    path = str(exe) if exe.exists() else shutil.which("ansible-playbook")
    if path is None:
        pytest.skip("ansible-playbook not available")
    return subprocess.run([path, "common.yml", *args], cwd=REPO, capture_output=True, text=True)


def _tasks(name: str) -> list[dict[str, Any]]:
    data = yaml.safe_load((ROLE / "tasks" / name).read_text())
    assert isinstance(data, list)
    return data


def test_syntax_check() -> None:
    res = _playbook("--syntax-check")
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("generator", ["dracut", "initramfs-tools"])
def test_clevis_tag_lists_dracut_tang_tasks(generator: str) -> None:
    res = _playbook("--tags", "clevis", "--list-tasks", "-e", f"clevis_initramfs_generator_override={generator}")
    assert res.returncode == 0, res.stdout + res.stderr
    for name in (
        "install clevis-dracut package",
        "deploy dracut config (clevis, network, ip=, NIC driver)",
        "verify clevis, network and NIC driver in the dracut initramfs",
        "set initrd network interface fact",
        "mark initramfs generator as detected",
        "set DEVICE in initramfs.conf",
    ):
        assert name in res.stdout, name


def _walk(tasks: list[dict[str, Any]], gates: tuple[str, ...] = ()) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    out = []
    for t in tasks:
        own = t.get("when", [])
        own = [own] if isinstance(own, str) else list(own)
        g = gates + tuple(str(w) for w in own)
        if "block" in t:
            out += _walk(t["block"], g)
        else:
            out.append((t, g))
    return out


def test_initramfs_tools_tasks_are_gated() -> None:
    marks = ("/etc/initramfs-tools", "update-initramfs", "lsinitramfs")
    seen = 0
    for task, gates in _walk(_tasks("config_clevis.yml")):
        if any(m in yaml.safe_dump(task) for m in marks):
            seen += 1
            assert any("clevis_initramfs_generator == 'initramfs-tools'" in g for g in gates), task["name"]
    assert seen >= 12


def test_dracut_tasks_are_gated_and_skip_initramfs_tools_files() -> None:
    dracut = [(t, g) for t, g in _walk(_tasks("config_clevis.yml")) if "dracut" in yaml.safe_dump(t)]
    assert dracut
    for task, gates in dracut:
        if task["name"].startswith("install clevis-dracut") or "dracut_rebuild" in yaml.safe_dump(task):
            assert any("== 'dracut'" in g for g in gates), task["name"]
        assert not any("initramfs-tools" in g for g in gates if "== 'initramfs-tools'" in g), task["name"]
    # the tang path imports the shared generator detection and the shared rebuild
    text = (ROLE / "tasks" / "config_clevis.yml").read_text()
    assert "import_tasks: detect_generator.yml" in text and "import_tasks: dracut_rebuild.yml" in text
    kf = (ROLE / "tasks" / "keyfile_fallback.yml").read_text()
    assert "import_tasks: detect_generator.yml" in kf
    assert kf.count("import_tasks: dracut_rebuild.yml") == 2
    assert "dracut -f" not in kf
    assert "-n clevis-initramfs" not in text


def _conf(**kw: Any) -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROLE / "templates"), keep_trailing_newline=True)
    env.filters["int"] = int
    ctx: dict[str, Any] = {"hw_iface": "enx6c1ff7bf8053", **DRACUT_VARS}
    ctx["clevis_net_driver"] = type("R", (), {"stdout": "r8152\n"})()
    ctx.update(kw)
    return env.get_template("etc_dracut.conf.d_90-clevis.conf.j2").render(**ctx)


def test_dracut_conf_renders() -> None:
    lines = [ln for ln in _conf().splitlines() if ln and not ln.startswith("#")]
    assert lines == [
        'add_dracutmodules+=" clevis clevis-pin-tang clevis-pin-sss network "',
        'kernel_cmdline+=" rd.neednet=1 ip=enx6c1ff7bf8053:dhcp rd.net.timeout.carrier=15 rd.net.dhcp.retry=3 "',
        'add_drivers+=" r8152 "',
    ]


def test_dracut_conf_optional_parts() -> None:
    out = _conf(
        clevis_dracut_net_timeout_carrier=0,
        clevis_dracut_net_dhcp_retry=0,
        clevis_dracut_extra_modules=[],
        clevis_net_driver=type("R", (), {"stdout": ""})(),
    )
    assert 'add_dracutmodules+=" clevis network "' in out
    assert 'kernel_cmdline+=" rd.neednet=1 ip=enx6c1ff7bf8053:dhcp "' in out
    assert "add_drivers" not in out and "rd.net." not in out


def test_defaults_and_readme() -> None:
    defaults = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    for name, value in DRACUT_VARS.items():
        assert defaults[name] == value, name
    lines = (ROLE / "README.md").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Variable ") and "Default" in ln)
    table = [ln for ln in iter(lines[start:]) if ln.startswith("|")]
    table = table[: next((i for i, ln in enumerate(lines[start:]) if not ln.startswith("|")), len(table))]
    bars = [m.start() for m in re.finditer(r"\|", table[0])]
    for ln in table:
        assert [m.start() for m in re.finditer(r"\|", ln)] == bars, ln
    documented = {ln.split("|")[1].strip().strip("`") for ln in table[2:]}
    assert set(DRACUT_VARS) <= documented
    readme = (ROLE / "README.md").read_text()
    assert "## Tang on dracut" in readme and "not configured by this role yet" not in readme
    for f in ("detect_generator.yml", "dracut_rebuild.yml", "etc_dracut.conf.d_90-clevis.conf.j2"):
        assert f in readme


def test_setup_script_renders_for_both_generators() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROLE / "templates"))
    env.filters["quote"] = shlex.quote
    for gen, marker in (("dracut", "dracut -f"), ("initramfs-tools", "update-initramfs -u -k all")):
        script = env.get_template("setup_clevis_tang.sh.j2").render(
            clevis_luks_device="/dev/sda3",
            clevis_sss_threshold=2,
            hw_iface="eth0",
            clevis_tang_servers=["http://tang.example.com"],
            clevis_initramfs_generator=gen,
        )
        assert marker in script
        assert ("update-initramfs" in script) == (gen == "initramfs-tools")
        res = subprocess.run([bash, "-n"], input=script, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_dracut_rebuild_shell_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    task = _tasks("dracut_rebuild.yml")[0]
    res = subprocess.run([bash, "-n"], input=task["shell"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    verify = next(t for t, _ in _walk(_tasks("config_clevis.yml")) if t["name"].startswith("verify clevis, network"))
    res = subprocess.run([bash, "-n"], input=verify["shell"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    for needle in ("clevis-luks-askpass", "clevis-decrypt-tang", "rd\\.neednet=1"):
        assert needle in verify["shell"]
