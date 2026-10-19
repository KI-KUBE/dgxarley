"""Static checks for the k9s install in the k3sserver role (syntax, tags, defaults, arch map, README)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROLE = REPO / "roles" / "k3sserver"
VENV_BIN = REPO / ".venv" / "bin"


def _playbook(*args: str) -> subprocess.CompletedProcess[str]:
    venv = VENV_BIN / "ansible-playbook"
    exe = str(venv) if venv.exists() else shutil.which("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    return subprocess.run([exe, "k3sserver.yml", *args], cwd=REPO, capture_output=True, text=True)


def test_syntax_check() -> None:
    res = _playbook("--syntax-check")
    assert res.returncode == 0, res.stdout + res.stderr


def test_tag_lists_tasks() -> None:
    res = _playbook("--tags", "k9s", "--list-tasks")
    assert res.returncode == 0, res.stdout + res.stderr
    for task in (
        "set k9s arch mapping",
        "check installed k9s version",
        "fetch latest k9s release from GitHub",
        "set target k9s version",
        "create k9s temp dir",
        "download k9s tarball (sha256 checked against the release checksums.sha256)",
        "extract the k9s binary only",
        "install k9s binary",
        "verify k9s version",
    ):
        assert f"k3sserver : {task}" in res.stdout, task


def test_import_is_master_only_and_tagged() -> None:
    tasks = yaml.safe_load((ROLE / "tasks" / "main.yml").read_text())
    imp = next(t for t in tasks if t.get("import_tasks") == "k9s.yml")
    assert imp["tags"] == ["k9s"]
    assert "isk3smaster | default(false) | bool" in imp["when"]
    assert "k9s_enabled | default(true) | bool" in imp["when"]


def test_defaults() -> None:
    d = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    assert d["k9s_enabled"] is True
    assert d["k9s_version"] == "latest"
    assert d["k9s_install_dir"] == "/usr/local/bin"
    assert d["k9s_github_repo"] == "derailed/k9s"


def test_arch_map_and_download() -> None:
    text = (ROLE / "tasks" / "k9s.yml").read_text()
    assert "{'x86_64': 'amd64', 'aarch64': 'arm64'}[ansible_architecture]" in text
    assert "k9s_Linux_{{ _k9s_arch }}.tar.gz" in text
    assert (
        "sha256:https://github.com/{{ k9s_github_repo }}/releases/download/{{ _k9s_target }}/checksums.sha256" in text
    )
    assert "version --short" in text


def test_readme_documents_k9s() -> None:
    assert "--tags k9s" in (REPO / "README.md").read_text()
    readme = (ROLE / "README.md").read_text()
    assert "`k9s.yml`" in readme
    rows = [ln for ln in readme.splitlines() if re.match(r"^\| `k9s_|^\| Variable +\| Default +\| Description", ln)]
    assert len(rows) == 5
    # aligned table: every row (and the separator) has the same width
    start = readme.splitlines().index(rows[0])
    block = readme.splitlines()[start : start + 6]
    assert len({len(ln) for ln in block}) == 1, block
