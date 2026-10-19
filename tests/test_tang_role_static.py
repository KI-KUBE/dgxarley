"""Static checks for the tang role (syntax, tags, play order, Quadlet unit, firewall gate, k3smaster wiring)."""

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
    res = _playbook("--tags", "tang", "--list-tasks", "--limit", "k3smaster")
    assert res.returncode == 0, res.stdout + res.stderr
    for task in (
        "Assert podman is present and new enough for Quadlet",
        "Ensure the tang data directory exists",
        "Deploy the tang Quadlet unit",
        "Ensure tang.service is running",
        "Verify tang serves its advertisement",
    ):
        assert f"tang : {task}" in res.stdout, task


def test_tang_play_runs_before_clevis() -> None:
    plays = yaml.safe_load((REPO / "common.yml").read_text())
    names = [p["name"] for p in plays]
    tang = names.index("Tang NBDE server")
    assert tang < names.index("clevis/tang NBDE automatic LUKS decryption")
    role = plays[tang]["roles"][0]
    assert role["role"] == "tang"
    assert role["when"] == "providestang | default(false) | bool"


def test_defaults() -> None:
    d = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    assert d["tang_port"] == 9090
    assert d["tang_data_dir"] == "/var/lib/tang"
    assert (d["tang_data_uid"], d["tang_data_gid"]) == ("100", "101")
    assert "tang_keys" not in d


def test_quadlet_unit() -> None:
    text = (ROLE / "templates" / "tang.container.j2").read_text()
    assert "Network=host" in text
    assert "PublishPort" not in text
    assert "UserNS" not in text
    assert "Environment=TANGPORT={{ tang_port }}" in text
    assert "Environment=USEHAPROXY=0" in text
    assert "WantedBy=multi-user.target" in text


def test_no_keys_deployed() -> None:
    assert not (ROLE / "files").exists()
    text = (ROLE / "tasks" / "podman_tang.yml").read_text()
    assert ".jwk" not in text


def test_iptables_gate() -> None:
    text = (REPO / "roles" / "common" / "templates" / "iptables.sh.j2").read_text()
    assert "{% if providestang | default(false) | bool %}" in text
    assert "--dport {{ tang_port | default(9090) }} -m set --match-set localnets src -j RETURN" in text


def test_k3smaster_provides_tang() -> None:
    hv = yaml.safe_load((REPO / "host_vars" / "k3smaster" / "main" / "common.yml").read_text())
    assert hv["providestang"] is True
    assert hv["wantspodman"] is True


_OLD = '1: sss \'{"t":2,"pins":{"tang":[{"url":"http://tang.a"},{"url":"http://tang.b:9090"}]}}\''
_NEW = '2: sss \'{"t":1,"pins":{"tang":[{"url":"http://10.0.0.1:9090"}]}}\''


@pytest.mark.parametrize(
    ("lines", "servers", "threshold", "expected"),
    [
        ([_OLD, _NEW], ["http://10.0.0.1:9090"], 1, "match=[2] stale=[1]"),
        ([_OLD], ["http://10.0.0.1:9090"], 1, "match=[] stale=[1]"),
        ([_OLD], ["http://tang.b:9090", "http://tang.a"], 2, "match=[1] stale=[]"),
        ([], ["http://10.0.0.1:9090"], 1, "match=[] stale=[]"),
    ],
)
def test_clevis_binding_classification(
    tmp_path: Path, lines: list[str], servers: list[str], threshold: int, expected: str
) -> None:
    play = [
        {
            "hosts": "localhost",
            "gather_facts": False,
            "tasks": [
                {
                    "include_tasks": str(REPO / "roles" / "clevis" / "tasks" / "parse_bindings.yml"),
                    "vars": {
                        "_clevis_list_lines": lines,
                        "clevis_tang_servers": servers,
                        "clevis_sss_threshold": threshold,
                    },
                },
                {"debug": {"msg": "match={{ _clevis_match_slots }} stale={{ _clevis_stale_slots }}"}},
            ],
        }
    ]
    pb = tmp_path / "parse.yml"
    pb.write_text(yaml.safe_dump(play))
    venv = VENV_BIN / "ansible-playbook"
    exe = str(venv) if venv.exists() else shutil.which("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    res = subprocess.run([exe, "-i", "localhost,", "-c", "local", str(pb)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert expected in res.stdout, res.stdout


def _tang_servers(host: str, tmp_path: Path, extra_inventory: str | None = None) -> list[str]:
    venv = VENV_BIN / "ansible"
    exe = str(venv) if venv.exists() else shutil.which("ansible")
    if exe is None:
        pytest.skip("ansible not available")
    inv = ["-i", "hosts.yml"]
    if extra_inventory is not None:
        extra = tmp_path / "extra.yml"
        extra.write_text(extra_inventory)
        inv += ["-i", str(extra)]
    res = subprocess.run(
        [exe, *inv, host, "-m", "debug", "-a", "var=clevis_tang_servers", "-o"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout.split("=>", 1)[1]
    servers = yaml.safe_load(out)["clevis_tang_servers"]
    assert isinstance(servers, list)
    return [str(s) for s in servers]


_PI = """
k3sserver:
  hosts:
    pi:
      ansible_host: 10.9.9.9
      k3s_node_ip: 10.9.9.9
      providestang: true
"""


def test_tang_servers_derived_from_providestang(tmp_path: Path) -> None:
    only = _tang_servers("spark1", tmp_path)
    assert len(only) == 1 and only[0].startswith("http://") and only[0].endswith(":9090"), only
    master = only[0]
    assert _tang_servers("k3smaster", tmp_path) == []
    assert sorted(_tang_servers("spark1", tmp_path, _PI)) == sorted([master, "http://10.9.9.9:9090"])
    assert _tang_servers("k3smaster", tmp_path, _PI) == ["http://10.9.9.9:9090"]
    assert _tang_servers("pi", tmp_path, _PI) == [master]


def test_threshold_one_and_no_vault_override() -> None:
    main = yaml.safe_load((REPO / "group_vars" / "all" / "main" / "common.yml").read_text())
    assert main["clevis_sss_threshold"] == 1


def test_k3smaster_keeps_clevis_with_boot_keyfile() -> None:
    hosts = yaml.safe_load((REPO / "hosts.yml").read_text())
    assert hosts["k3sserver"]["hosts"]["k3smaster"]["wantsclevis"] is True
    cv = yaml.safe_load((REPO / "host_vars" / "k3smaster" / "main" / "clevis.yml").read_text())
    assert cv["clevis_keyfile_enabled"] is True
    assert cv["clevis_keyfile_boot_enabled"] is True
