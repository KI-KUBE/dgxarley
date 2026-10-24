"""Static checks of the k3s master HAProxy template (hub_ingress accept-proxy gate)."""

from pathlib import Path
from typing import Any

from tests.test_ops_wireguard_static import _render

REPO = Path(__file__).resolve().parent.parent
HAPROXY_TPL = REPO / "roles" / "k3sserver" / "templates" / "etc_haproxy_haproxy.cfg.j2"
HUB_INGRESS: dict[str, Any] = {
    "enabled": True,
    "bind_address": "10.200.1.10",
    "port": 8443,
    "source_ips": ["10.200.1.1/32", "10.200.1.9/32"],
}


def _render_haproxy(**extra: Any) -> str:
    base: dict[str, Any] = {"haproxy_stats_password": "x", "hub_ingress": {**HUB_INGRESS, "enabled": False}}
    return _render(HAPROXY_TPL, **{**base, **extra})


def test_hub_ingress_frontend_rejects_foreign_tcp_peers_at_connection_stage() -> None:
    out = _render_haproxy(hub_ingress=HUB_INGRESS, isk3smaster=True)
    frontend = out.split("frontend hub_ingress_https", 1)[1].split("backend", 1)[0]
    assert "bind 10.200.1.10:8443 accept-proxy" in frontend
    assert "tcp-request connection reject unless { src 10.200.1.1/32 10.200.1.9/32 }" in frontend
    assert "tcp-request content" not in frontend


def test_hub_ingress_frontend_absent_when_disabled_or_not_master() -> None:
    assert "hub_ingress_https" not in _render_haproxy(isk3smaster=True)
    assert "hub_ingress_https" not in _render_haproxy(hub_ingress=HUB_INGRESS)
