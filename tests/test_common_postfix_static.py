"""Static checks for the postfix relay alias (roles/common) and the hub TCP ports of the mikrotik role."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
COMMON = REPO / "roles" / "common"
MIKROTIK = REPO / "roles" / "mikrotik"
VENV_BIN = REPO / ".venv" / "bin"


def _playbook(playbook: str, *args: str) -> subprocess.CompletedProcess[str]:
    venv = VENV_BIN / "ansible-playbook"
    exe = str(venv) if venv.exists() else shutil.which("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    return subprocess.run([exe, playbook, *args], cwd=REPO, capture_output=True, text=True)


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("playbook", ["common.yml", "mikrotik.yml"])
def test_syntax_check(playbook: str) -> None:
    res = _playbook(playbook, "--syntax-check")
    assert res.returncode == 0, res.stdout + res.stderr


def test_postfix_tag_lists_tasks() -> None:
    res = _playbook("common.yml", "--tags", "postfix", "--list-tasks")
    assert res.returncode == 0, res.stdout + res.stderr
    for task in ("ensure postfix relay alias in /etc/hosts", "copy postfix main config", "write /etc/mailname"):
        assert f"common : {task}" in res.stdout, task


def test_common_defaults_and_mirror() -> None:
    for path in (COMMON / "defaults" / "main.yml", REPO / "group_vars" / "all" / "main" / "common.yml"):
        d = _yaml(path)
        assert d["postfix_relay_alias"] == "mailrelay", path
        assert d["postfix_relay_alias_ip"] == "", path
        assert d["postfix_relayhost"] == "[mail.example.com]", path
        assert d["postfix_queue_lifetime"] == "5d", path
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        nets = env.from_string(d["postfix_mynetworks"]).render(
            localnets=[{"cidr": "10.87.0.0/24"}], localnets_k3sserver=[{"cidr": "10.68.0.0/16"}]
        )
        assert nets == "['10.87.0.0/24', '10.68.0.0/16']", path
        assert env.from_string(d["postfix_mynetworks"]).render(localnets=[]) == "[]", path


def test_hosts_alias_task() -> None:
    tasks = _yaml(COMMON / "tasks" / "postfix.yml")
    task = next(t for t in tasks if t["name"] == "ensure postfix relay alias in /etc/hosts")
    li = task["lineinfile"]
    assert li["path"] == "/etc/hosts"
    assert li["line"] == "{{ postfix_relay_alias_ip }} {{ postfix_relay_alias }}"
    assert "present" in li["state"] and "absent" in li["state"]
    assert task["notify"] == "restart postfix"
    rx = re.compile(li["regexp"].replace("{{ postfix_relay_alias }}", "mailrelay"))
    assert rx.match("10.200.1.1 mailrelay")
    assert rx.match("10.87.0.10\tmailrelay extra")
    assert not rx.match("10.87.0.10 mailrelay2")
    assert not rx.match("127.0.1.1 host.example")
    # the alias task precedes main.cf so one restart covers both
    names = [t["name"] for t in tasks]
    assert names.index("ensure postfix relay alias in /etc/hosts") < names.index("copy postfix main config")
    assert names.count("copy postfix main config") == 1


def test_main_cf_template() -> None:
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    env.filters["regex_replace"] = lambda s, pat, rep: re.sub(pat, rep, s)
    out = env.from_string((COMMON / "templates" / "etc_postfix_main.cf.j2").read_text()).render(
        mailname="node.example.invalid",
        postfix_relayhost="[mailrelay]",
        postfix_queue_lifetime="30d",
        postfix_mynetworks=["10.87.0.0/24", "10.68.0.0/16"],
    )
    assert "\nrelayhost = [mailrelay]\n" in out
    assert "\nmaximal_queue_lifetime = 30d\n" in out
    assert "\nbounce_queue_lifetime = 30d\n" in out
    assert "\nsmtp_host_lookup = dns, native\n" in out
    assert "\nmynetworks = 127.0.0.0/8 [::ffff:127.0.0.0]/104 [::1]/128 10.87.0.0/24 10.68.0.0/16\n" in out


def test_mikrotik_defaults() -> None:
    d = _yaml(MIKROTIK / "defaults" / "main.yml")
    assert d["mikrotik_wireguard_hub_tcp_ports"] == [25, 587]
    expr = d["mikrotik_wireguard_hub_address"]
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.filters["regex_replace"] = lambda s, pat, rep: re.sub(pat, rep, s)
    assert env.from_string(expr).render(mikrotik_wireguard_ops_subnet="10.200.7.0/24") == "10.200.7.1"


def test_mikrotik_hub_port_rules() -> None:
    text = (MIKROTIK / "tasks" / "wireguard.yml").read_text()
    tasks = _yaml(MIKROTIK / "tasks" / "wireguard.yml")
    block = next(t for t in tasks if t.get("name") == "Configure hub WireGuard client")["block"]
    by_name = {t["name"]: t for t in block}
    fwd = by_name["Ensure hub port forward rule"]
    nat = by_name["Ensure hub port NAT rule"]
    assert fwd["when"] == "mikrotik_wireguard_hub_tcp_ports | length > 0"
    assert nat["when"] == "mikrotik_wireguard_hub_tcp_ports | length > 0"
    assert by_name["Remove hub port NAT rule"]["when"] == "mikrotik_wireguard_hub_tcp_ports | length == 0"
    assert (
        fwd["vars"]["_hub_fwd_comment"]
        == "{{ mikrotik_wireguard_comment }}-fwd hub tcp {{ mikrotik_wireguard_hub_tcp_ports | join(',') }}"
    )
    assert (
        nat["vars"]["_hub_nat_comment"]
        == "{{ mikrotik_wireguard_comment }} nat tcp {{ mikrotik_wireguard_hub_tcp_ports | join(',') }}"
    )
    assert "chain=forward action=accept" in fwd["community.routeros.command"]["commands"][0]
    assert "chain=srcnat action=masquerade" in nat["community.routeros.command"]["commands"][0]
    for cmd in (fwd, nat):
        c = cmd["community.routeros.command"]["commands"][0]
        assert "dst-address={{ mikrotik_wireguard_hub_address }} protocol=tcp" in c
    # anchored by id before the rule each must precede, never a positional place-before
    fwd_anchor = 'place-before=([/ip/firewall/filter/find comment~"^{{ mikrotik_wireguard_comment }} forward drop"]->0)'
    assert fwd_anchor in fwd["community.routeros.command"]["commands"][0]
    assert (
        'place-before=([/ip/firewall/nat/find comment="{{ mikrotik_wireguard_comment }} no-nat"]->0)'
        in nat["community.routeros.command"]["commands"][0]
    )
    for name in ("Ensure tunnel-to-LAN forward rules", "Ensure tunnel return traffic rule"):
        assert fwd_anchor in by_name[name]["community.routeros.command"]["commands"][0]
    no_nat = by_name["Ensure tunnel traffic is not NATed"]["community.routeros.command"]["commands"][0]
    assert "place-before=([/ip/firewall/nat/find chain=srcnat action=masquerade]->0)" in no_nat
    assert "on-error=" in no_nat  # the only legitimate missing anchor: no masquerade rule at all
    filt = by_name["Ensure WireGuard filter rules"]
    anchors = {i["comment"]: i.get("anchor") for i in filt["loop"]}
    assert anchors["{{ mikrotik_wireguard_comment }} input icmp"] == "{{ mikrotik_wireguard_comment }} input drop"
    assert anchors["{{ mikrotik_wireguard_comment }} input mgmt"] == "{{ mikrotik_wireguard_comment }} input drop"
    assert all(anchors[c] is None for c in anchors if c.endswith(" drop") or " drop " in c)
    assert "{% if item.anchor is defined %}" in filt["community.routeros.command"]["commands"][0]
    # the -fwd rule is re-created after the generic -fwd removal
    names = [t["name"] for t in block]
    assert names.index("Remove tunnel-to-LAN forward rules") < names.index("Ensure hub port forward rule")
    # order assertions: hub NAT above no-nat, hub forward above the drop, hub NAT excluded from the masquerade check
    order = by_name["Assert firewall rule order"]
    assert "reject('match', '^masquerade\\|' ~ mikrotik_wireguard_comment ~ ' nat tcp')" in order["vars"]["_nat_masq"]
    that = "\n".join(order["ansible.builtin.assert"]["that"])
    assert "_fwd.index(_hub_fwd) < _fwd.index('{{ mikrotik_wireguard_comment }} forward drop to tunnel')" in that
    assert "_nat.index(_hub_nat_rule) < _nat.index(_no_nat)" in that
    # teardown removes everything by the comment prefix, which both new comments carry
    assert '/ip/firewall/nat/remove [find comment~"^{{ mikrotik_wireguard_comment }}"]' in text
    assert "hub tcp ports from the LAN" in text


def test_mikrotik_rules_anchored_not_positional() -> None:
    """No numeric place-before anywhere in the role: RouterOS 7 does not treat 0 as the top of the list."""
    for path in sorted((MIKROTIK / "tasks").glob("*.yml")):
        assert not re.search(r"place-before=\d", path.read_text()), path
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    for fname, comment_var, loop_name in (
        ("wireguard.yml", "mikrotik_wireguard_comment", "Ensure WireGuard filter rules"),
        ("uplink.yml", "mikrotik_uplink_comment", "Ensure uplink filter rules"),
    ):
        tasks = _yaml(MIKROTIK / "tasks" / fname)
        flat = [t for t in tasks] + [t for b in tasks if "block" in b for t in b["block"]]
        task = next(t for t in flat if t.get("name") == loop_name)
        cmd = task["community.routeros.command"]["commands"][0]
        # every accept names its drop as anchor, every drop has none
        for item in task["loop"]:
            if "action=drop" in item["rule"]:
                assert "anchor" not in item, item["comment"]
            else:
                assert item["anchor"].endswith(" drop"), item["comment"]
                assert item["anchor"].startswith("{{ " + comment_var + " }}")
        ctx = {
            comment_var: "x-hub",
            "mikrotik_wireguard_interface": "wg",
            "mikrotik_uplink_interface": "e1",
            "mikrotik_wireguard_ops_subnet": "10.0.0.0/24",
            "mikrotik_wireguard_input_tcp_ports": [22],
        }
        rendered = [
            env.from_string(cmd).render(
                item=env.from_string(yaml.safe_dump(i)).render(**ctx)
                and yaml.safe_load(env.from_string(yaml.safe_dump(i)).render(**ctx)),
                **ctx,
            )
            for i in task["loop"]
        ]
        with_anchor = [r for r in rendered if "place-before=" in r]
        assert len(with_anchor) == sum("anchor" in i for i in task["loop"])
        for r in with_anchor:
            assert re.search(r'place-before=\(\[/ip/firewall/filter/find comment="x-hub [a-z ]*drop"\]->0\) \}', r), r
