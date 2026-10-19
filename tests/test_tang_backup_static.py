"""Static checks for the tang key backup (tags, defaults, gate/cleanup path, script template, units)."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROLE = REPO / "roles" / "tang"
VENV_BIN = REPO / ".venv" / "bin"


def _playbook(*args: str) -> subprocess.CompletedProcess[str]:
    venv = VENV_BIN / "ansible-playbook"
    exe = str(venv) if venv.exists() else shutil.which("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    return subprocess.run([exe, "common.yml", *args], cwd=REPO, capture_output=True, text=True)


def test_syntax_check() -> None:
    res = _playbook("--syntax-check")
    assert res.returncode == 0, res.stdout + res.stderr


def test_tag_lists_tasks() -> None:
    res = _playbook("--tags", "tang_backup", "--list-tasks", "--limit", "k3smaster")
    assert res.returncode == 0, res.stdout + res.stderr
    for task in (
        "Check the rc binary and config exist",
        "Assert rc and its alias config are present",
        "Deploy the tang backup script",
        "Deploy the tang backup service unit",
        "Deploy the tang backup timer unit",
        "Enable and start the tang backup timer",
        "Stop and disable the tang backup timer",
        "Remove the tang backup script and units",
    ):
        assert f"tang : {task}" in res.stdout, task
    assert "Deploy the tang Quadlet unit" not in res.stdout


def test_import_is_tagged() -> None:
    tasks = yaml.safe_load((ROLE / "tasks" / "main.yml").read_text())
    imp = next(t for t in tasks if t.get("import_tasks") == "backup.yml")
    assert imp["tags"] == ["tang", "tang_backup"]


def test_defaults() -> None:
    d = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    assert d["tang_backup_enabled"] is True
    assert d["tang_backup_schedule"] != "*-*-* 02:30:00"
    assert d["tang_backup_script"] == "/usr/local/sbin/tang-backup.sh"
    assert d["tang_backup_rc_bin"] == "/usr/local/bin/rc"
    assert d["tang_backup_rc_config_dir"] == "/root/.config/rc"


def test_gate_and_cleanup_blocks() -> None:
    tasks = yaml.safe_load((ROLE / "tasks" / "backup.yml").read_text())
    install, remove = tasks
    assert install["when"] == "tang_backup_enabled | default(true) | bool"
    assert remove["when"] == "not (tang_backup_enabled | default(true) | bool)"
    text = (ROLE / "tasks" / "backup.yml").read_text()
    assert "ansible_check_mode" in text
    assert "state: absent" in text


def test_handlers() -> None:
    names = [h["name"] for h in yaml.safe_load((ROLE / "handlers" / "main.yml").read_text())]
    assert "reload systemd for tang-backup" in names
    assert "restart tang-backup timer" in names


def test_units() -> None:
    svc = (ROLE / "templates" / "tang-backup.service.j2").read_text()
    assert "Type=oneshot" in svc
    assert "ExecStart={{ tang_backup_script }}" in svc
    timer = (ROLE / "templates" / "tang-backup.timer.j2").read_text()
    assert "Persistent=true" in timer
    assert "OnCalendar={{ tang_backup_schedule }}" in timer
    assert "RandomizedDelaySec" in timer


def test_script_renders_and_parses(tmp_path: Path) -> None:
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.filters["quote"] = lambda s: "'" + str(s).replace("'", "'\"'\"'") + "'"
    src = (ROLE / "templates" / "tang-backup.sh.j2").read_text()
    out = env.from_string(src).render(
        ansible_managed="managed",
        tang_backup_rc_bin="/usr/local/bin/rc",
        tang_backup_rc_config_dir="/root/.config/rc",
        tang_data_dir="/var/lib/tang",
        rustfs_rc_alias="backupstore",
        rustfs_rc_backup_bucket="dgx-backups",
        inventory_hostname="k3smaster",
    )
    script = tmp_path / "tang-backup.sh"
    script.write_text(out)
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "set -euo pipefail" in out
    assert "tang/" in out
    for forbidden in ("rc rm", "--overwrite", "--remove", "cat ", 'echo "${key}'):
        assert forbidden not in out


def test_readme_documents_backup() -> None:
    text = (ROLE / "README.md").read_text()
    assert "## Backup" in text
    assert "tang_backup_enabled" in text
