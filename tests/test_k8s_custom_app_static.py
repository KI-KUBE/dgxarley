"""Static + offline checks for roles/k8s_custom_app and k8s_custom.yml.

The playbook runs against the dummy inventory with a fake kubernetes.core / community.general collection
(tests/fixtures/k8s_custom_app/collections) that records every manifest instead of touching a cluster.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROLE = REPO / "roles" / "k8s_custom_app"
FIXTURES = REPO / "tests" / "fixtures" / "k8s_custom_app"
VENV_BIN = REPO / ".venv" / "bin"


def _ansible_playbook() -> str:
    venv = VENV_BIN / "ansible-playbook"
    exe = str(venv) if venv.exists() else shutil.which("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    return exe


def _run(
    tmp_path: Path, *args: str, apps_dir: Path = FIXTURES / "custom_apps"
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    record = tmp_path / "record.jsonl"
    env = {
        **os.environ,
        "ANSIBLE_COLLECTIONS_PATH": str(FIXTURES / "collections"),
        "K8S_CUSTOM_APP_RECORD": str(record),
        "ANSIBLE_NOCOLOR": "1",
    }
    res = subprocess.run(
        [
            _ansible_playbook(),
            "k8s_custom.yml",
            "-l",
            "k3smaster",
            "-e",
            "kubeconfig_context_name=test",
            "-e",
            f"custom_apps_dir={apps_dir}",
            "-e",
            f"custom_app_test_dump={tmp_path / 'raw.json'}",
            *args,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
    return res, calls


def _objects(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in calls:
        if call["module"] != "k8s":
            continue
        definition = call["args"].get("definition")
        if isinstance(definition, list):
            out.extend(definition)
        elif isinstance(definition, dict):
            out.append(definition)
    return out


def _one(objs: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    matches = [o for o in objs if o.get("kind") == kind and o["metadata"]["name"] == name]
    assert len(matches) == 1, f"{kind}/{name}: {len(matches)} matches"
    return matches[0]


def test_syntax_check() -> None:
    res = subprocess.run(
        [_ansible_playbook(), "k8s_custom.yml", "--syntax-check"], cwd=REPO, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_defaults() -> None:
    d = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())
    assert d["custom_apps_dir"] == "{{ playbook_dir }}/custom_apps"
    assert d["custom_app_teardown"] == ""
    assert d["custom_app_purge"] is False
    spec = d["custom_app_spec_defaults"]
    assert spec["enabled"] is True and spec["teardown"] is False
    assert spec["placement"] == {"node": "", "node_selector": {}, "gpu": 0}
    assert d["custom_app_default_node_selector"] == {"node-role": "k3smaster"}
    assert spec["ingress"]["enabled"] is True and spec["ingress"]["dns"] is True
    assert spec["network_policies"] == {
        "enabled": True,
        "internet": True,
        "egress_namespaces": [],
        "ingress_namespaces": [],
    }


def test_playbook_shape() -> None:
    play = yaml.safe_load((REPO / "k8s_custom.yml").read_text())[0]
    assert play["hosts"] == "k3smaster" and play["connection"] == "local"
    assert play["roles"] == ["k8s_custom_app"]
    pre = play["pre_tasks"][0]["ansible.builtin.import_role"]
    assert pre == {"name": "k8s_infra", "tasks_from": "core/preflight.yml"}


def test_generic_install_renders_all_resources(tmp_path: Path) -> None:
    res, calls = _run(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    objs = _objects(calls)
    ns = {
        c["args"].get("namespace") or o["metadata"].get("namespace")
        for c in calls
        for o in _objects([c])
        if o["kind"] != "Namespace"
    }
    assert ns == {"demo"}, "only the enabled generic app may be applied"

    dep = _one(objs, "Deployment", "demo")
    pod = dep["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"node-role": "k3smaster"}
    assert pod["imagePullSecrets"] == [{"name": "demo-pull"}]
    assert pod["securityContext"] == {"runAsUser": 1000, "runAsGroup": 1000, "runAsNonRoot": True}
    assert [c["name"] for c in pod["initContainers"]] == ["fix-perms"]
    assert pod["initContainers"][0]["command"][-1] == "chown 1000:1000 /data"
    c = pod["containers"][0]
    assert c["image"] == "ghcr.io/example/demo:1.2.3"
    assert [p["containerPort"] for p in c["ports"]] == [8080, 9090]
    assert c["readinessProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}
    assert c["envFrom"] == [{"configMapRef": {"name": "demo-env"}}, {"secretRef": {"name": "demo-env"}}]
    mounts = {m["mountPath"]: m for m in c["volumeMounts"]}
    assert mounts["/etc/demo/config.yml"]["subPath"] == "etc_demo_config.yml"
    assert mounts["/etc/demo/creds.json"]["name"] == "secret-files"
    assert {v["name"]: list(v)[1] for v in pod["volumes"]} == {
        "files": "configMap",
        "secret-files": "secret",
        "data": "hostPath",
        "cache": "persistentVolumeClaim",
        "tmp": "emptyDir",
    }
    assert dep["metadata"]["annotations"]["keel.sh/policy"] == "force"
    assert dep["spec"]["strategy"] == {"type": "Recreate"}

    assert _one(objs, "ConfigMap", "demo-env")["data"] == {"LOG_LEVEL": "info", "WORKERS": "4"}
    files = _one(objs, "ConfigMap", "demo-files")["data"]
    assert files["etc_demo_config.yml"] == "hostname: demo.dgx.example.com\nexposure: lan\nworkers: 4\n"
    assert files["etc_demo_static.txt"] == "static content\n"
    assert _one(objs, "Secret", "demo-files")["stringData"] == {"etc_demo_creds.json": '{"token": "x"}'}
    pull = json.loads(_one(objs, "Secret", "demo-pull")["stringData"][".dockerconfigjson"])
    assert pull["auths"]["ghcr.io"]["username"] == "bot"
    assert _one(objs, "PersistentVolumeClaim", "demo-cache")["spec"]["resources"]["requests"]["storage"] == "5Gi"
    assert [p["port"] for p in _one(objs, "Service", "demo")["spec"]["ports"]] == [8080, 9090]

    egress = _one(objs, "NetworkPolicy", "demo-egress")["spec"]["egress"]
    assert len(egress) == 3 and egress[1]["to"][0]["namespaceSelector"]["matchLabels"] == {
        "kubernetes.io/metadata.name": "litellm"
    }
    ingress = _one(objs, "NetworkPolicy", "demo-ingress")["spec"]["ingress"]
    assert ingress[0]["from"][0]["namespaceSelector"]["matchLabels"] == {"kubernetes.io/metadata.name": "traefik"}

    assert _one(objs, "Certificate", "demo.dgx.example.com")["spec"]["issuerRef"]["name"] == "letsencrypt-production"
    assert _one(objs, "IngressRoute", "demo-https-ingress")["spec"]["tls"] == {"secretName": "demo.dgx.example.com"}
    allow = _one(objs, "Middleware", "demo-ip-allowlist")["spec"]["ipAllowList"]["sourceRange"]
    assert "0.0.0.0/0" not in allow
    assert _one(objs, "ConfigMap", "demo-extra")["data"] == {"k": "v"}
    dns = [c["args"] for c in calls if c["module"] == "nsupdate"]
    assert dns == [{**dns[0], "record": "demo.dgx.example.com.", "state": "present", "value": "192.168.124.100"}]
    assert "demo post hook marker" in res.stdout


def test_custom_tasks_main_replaces_generic_install(tmp_path: Path) -> None:
    res, calls = _run(tmp_path, "--tags", "raw")
    assert res.returncode == 0, res.stdout + res.stderr
    assert calls == [], "the escape hatch app must not emit generic manifests"
    app = json.loads((tmp_path / "raw.json").read_text())
    assert app["name"] == "raw" and app["namespace"] == "raw"
    assert app["placement"] == {"node": "spark1", "node_selector": {}, "gpu": 1}
    assert app["ingress"]["enabled"] is False and app["ingress"]["hostname"] == "raw.dgx.example.com"
    assert app["ingress"]["exposure"] == "lan" and app["ingress"]["dns_backend"] == "primary"
    assert "Create demo namespace" not in res.stdout


def test_teardown_keeps_namespace_with_pvcs(tmp_path: Path) -> None:
    res, calls = _run(tmp_path, "--tags", "demo", "-e", "custom_app_teardown=demo")
    assert res.returncode == 0, res.stdout + res.stderr
    assert not [c for c in calls if c["module"] == "k8s" and c["args"].get("kind") == "Namespace"]
    assert {c["args"]["kind"] for c in calls if c["module"] == "k8s_info"} >= {
        "Deployment",
        "Service",
        "IngressRoute",
        "Secret",
    }
    dns = [c["args"] for c in calls if c["module"] == "nsupdate"]
    assert len(dns) == 1 and dns[0]["state"] == "absent" and dns[0]["record"] == "demo.dgx.example.com."


def test_purge_deletes_namespace(tmp_path: Path) -> None:
    res, calls = _run(tmp_path, "--tags", "demo", "-e", "custom_app_teardown=demo", "-e", "custom_app_purge=true")
    assert res.returncode == 0, res.stdout + res.stderr
    deletes = [c["args"] for c in calls if c["module"] == "k8s" and c["args"].get("state") == "absent"]
    assert deletes == [{**deletes[0], "kind": "Namespace", "name": "demo"}]
    assert not [c for c in calls if c["module"] == "k8s_info"]


def test_missing_spec_file_fails(tmp_path: Path) -> None:
    apps = tmp_path / "apps"
    (apps / "broken").mkdir(parents=True)
    res, _ = _run(tmp_path, apps_dir=apps)
    assert res.returncode != 0
    assert "missing broken.yml" in res.stdout + res.stderr


def test_hostpath_without_pin_fails(tmp_path: Path) -> None:
    apps = tmp_path / "apps"
    (apps / "nopin").mkdir(parents=True)
    (apps / "nopin" / "nopin.yml").write_text(
        yaml.safe_dump(
            {
                "image": "x:1",
                "port": 80,
                "placement": {"node_selector": {}},
                "volumes": [{"name": "d", "mount": "/d", "host_path": "/mnt/d"}],
            }
        )
    )
    res, calls = _run(tmp_path, apps_dir=apps)
    assert res.returncode != 0 and calls == []
    assert "nopin: invalid spec" in res.stdout + res.stderr


def test_docs_mention_playbook() -> None:
    assert "k8s_custom.yml" in (REPO / "README.md").read_text()
