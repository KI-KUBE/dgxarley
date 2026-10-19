# ops_wireguard role

Box-side, OUTBOUND WireGuard tunnel from every node (k3smaster, spark1..N; one key, one tunnel address and one hub peer per node, values from host_vars) to the Ops WireGuard hub. The box dials out; no inbound port is opened. Counterpart of the Mikrotik-terminated variant (`mikrotik_wireguard_*` in `roles/mikrotik`, same variable names). Unrelated to `iswireguardhost` (the lab's INBOUND WireGuard host in `roles/common`).

Gate `ops_wireguard_enabled` (default `false`), opt-in removal `ops_wireguard_teardown`. The play in `common.yml` runs the role only if one of them is true.

```bash
ansible-playbook common.yml --tags ops_wireguard --limit k3smaster --check --diff
ansible-playbook common.yml --tags ops_wireguard,iptables --limit k3smaster   # tunnel + firewall
ansible-playbook common.yml --tags ops_wireguard -e ops_wireguard_teardown=true -e ops_wireguard_enabled=false
```

## What it does

1. Asserts the required variables (plain regex asserts, IPv4 only; `ansible.utils` is not a repo dependency)
2. Installs `wireguard-tools` (+ `wireguard` on Ubuntu), `/etc/wireguard` 0700
3. Key: `ops_wireguard_private_key` set (vault) = written to the key file; empty = `wg genkey` on the host only when the file is missing (never leaves the box). The public key is derived with `wg pubkey` and registered as fact `ops_wireguard_public_key`
4. Renders `/etc/wireguard/<iface>.conf` (0600) WITHOUT the private key: `PostUp = wg set %i private-key <keyfile>` loads it at runtime. No `wg-forward.sh` hooks
5. `systemctl enable --now wg-quick@<iface>`, restart on conf/key change
6. `forward_lan`: `/etc/sysctl.d/90-ops-wireguard.conf` (`net.ipv4.ip_forward=1`); removed again when `forward_lan` is off. The role's own drop-in is used, `roles/common`'s `ip_forward` task (gated on `wantsopenvswitch`) is left unchanged
7. Verification (non-fatal): `wg show`, waits up to `handshake_wait` s for a handshake. When the key was generated here, the public key and the hub command `add-peer <customer> <pubkey> <tunnel-address>[ <lan>]` are printed

## Adopting the bootstrap tunnel

The external initial setup script creates the same tunnel by hand (`/etc/wireguard/<iface>.key`, `<iface>.conf`, `wg-forward.sh` hooks). Run this role with the same `interface` and matching values: the existing key file is reused unchanged (permissions forced to 0600), the template REPLACES the conf (the `PrivateKey` line and the `wg-forward.sh` PostUp/PostDown disappear, the unit is restarted once). its `wg-forward.sh` hook and forward sysctl drop-in stay on disk untouched but are no longer referenced by the conf; remove them manually if wanted.

## Routing modes (`ops_wireguard_route_mode`)

- `full` (default): plain wg-quick, the `AllowedIPs` route (ops subnet) lands in the main table. For the terminating gateway box (`forward_lan`).
- `source`: `Table = <route_table>` in the conf (wg-quick puts the ops route into that table only) plus `PostUp/PostDown = ip -4 rule add|del from <tunnel ip> lookup <table> priority <prio>`. Only packets sourced from the tunnel address use the tunnel. Reason: a node that is ALSO reachable another way (Mikrotik-terminated tunnel into its LAN, or via the k3smaster with `forward_lan`) would otherwise answer ops flows that arrived over the LAN path through its OWN tunnel; the hub drops those (source address not in that peer's AllowedIPs) = asymmetric routing.
- `forward_lan` requires `full` (asserted).

## Firewall semantics

`roles/common` rewrites the firewall (`iptables.sh.j2`), so the tunnel rules live there and are rendered when `ops_wireguard_enabled` is true (byte-identical output when false; the common play does not see this role's defaults, the template carries the same interface default):

- INPUT (chain `HTSTUFFIN`, before the localnets/xosrc rules): `-i <iface> -s <ops_subnet> -j ACCEPT`
- `forward_lan`: chain `HTOPSWG`, jumped to first from `FORWARD`: `-i <iface> -s <ops> -d <lan> -j ACCEPT` per LAN and `-o <iface> -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT`
- `forward_masquerade`: chain `HTOPSWGNAT` in `nat POSTROUTING`: `-s <ops> -d <lan> ! -o <iface> -j MASQUERADE` per LAN

IPv4 only (`ip6tables_setup.sh.j2` is not touched). Rerun `common.yml --tags iptables` after changing these variables.

## Variables

| Variable                            | Default                      | Description                                                                                                              |
| ----------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ops_wireguard_enabled`             | `false`                      | Gate. false = the role does nothing (removes nothing)                                                                    |
| `ops_wireguard_teardown`            | `false`                      | Opt-in removal (`-e ops_wireguard_teardown=true`); exclusive with `enabled`                                              |
| `ops_wireguard_interface`           | `ops-wg`                     | Interface name (max 15 chars); unit `wg-quick@<iface>`                                                                   |
| `ops_wireguard_hub_endpoint`        | `""`                         | Hub `host:port` (required)                                                                                               |
| `ops_wireguard_hub_public_key`      | `""`                         | Hub public key (required)                                                                                                |
| `ops_wireguard_tunnel_address`      | `""`                         | This box's address with prefix, e.g. `10.200.1.2/32` (required)                                                          |
| `ops_wireguard_ops_subnet`          | `""`                         | AllowedIPs of the hub peer = ops subnet, e.g. `10.200.1.0/24` (required)                                                 |
| `ops_wireguard_keepalive`           | `25`                         | `PersistentKeepalive` in seconds                                                                                         |
| `ops_wireguard_route_mode`          | `full`                       | `full` = plain wg-quick (main table); `source` = own table + `ip rule` for the tunnel source address (see Routing modes) |
| `ops_wireguard_route_table`         | `51820`                      | Routing table used by `source` mode                                                                                      |
| `ops_wireguard_rule_priority`       | `1000`                       | `ip rule` priority used by `source` mode                                                                                 |
| `ops_wireguard_forward_lan`         | `false`                      | Let the ops subnet reach `forward_lans` through this box (sysctl + firewall rules)                                       |
| `ops_wireguard_forward_lans`        | `[]`                         | LAN network CIDRs reachable from ops, e.g. `10.87.0.0/24`                                                                |
| `ops_wireguard_forward_masquerade`  | `true`                       | MASQUERADE ops traffic towards those LANs                                                                                |
| `ops_wireguard_private_key`         | `""`                         | Vault only. Empty = key generated on the host; set = written to the key file                                             |
| `ops_wireguard_private_key_file`    | `/etc/wireguard/<iface>.key` | Key file on the host (0600)                                                                                              |
| `ops_wireguard_handshake_wait`      | `30`                         | Seconds to wait for the first handshake (never fatal)                                                                    |
| `ops_wireguard_customer_id`         | `""`                         | Only used in the printed `add-peer` hint                                                                                 |
| `ops_wireguard_teardown_remove_key` | `false`                      | Teardown also deletes the key file (re-enable = new public key)                                                          |

## Teardown

`-e ops_wireguard_teardown=true -e ops_wireguard_enabled=false`: stops/disables `wg-quick@<iface>`, removes the conf, the sysctl drop-in and the `HTOPSWG*` firewall chains; the key only with `ops_wireguard_teardown_remove_key`. Afterwards rerun `common.yml --tags iptables` (drops the INPUT rule) and run `remove-peer <customer>` on the hub. `ip_forward` stays 1 until reboot.

## Examples

k3smaster as `full` gateway with `forward_lan`, sparks `source` (reached via the k3smaster by ops, tunnels of their own as a second path):

(host_vars/k3smaster/main/wireguard.yml)

```yaml
ops_wireguard_enabled: true
ops_wireguard_customer_id: acme
ops_wireguard_hub_endpoint: hub.example.invalid:51820
ops_wireguard_hub_public_key: "<hub public key>"
ops_wireguard_tunnel_address: 10.200.1.2/32
ops_wireguard_ops_subnet: 10.200.1.0/24
ops_wireguard_forward_lan: true
ops_wireguard_forward_lans: ["10.87.0.0/24"]
```

```yaml
# host_vars/spark1/main/wireguard.yml (spark2: tunnel address 10.200.1.4/32)
ops_wireguard_enabled: true
ops_wireguard_customer_id: acme
ops_wireguard_hub_endpoint: hub.example.invalid:51820
ops_wireguard_hub_public_key: "<hub public key>"
ops_wireguard_tunnel_address: 10.200.1.3/32
ops_wireguard_ops_subnet: 10.200.1.0/24
ops_wireguard_route_mode: source
```

Alternative, Mikrotik-terminated LAN (`mikrotik_wireguard_forward_lan` on the switch): k3smaster AND spark1/spark2 all use `route_mode: source` with `forward_lan: false`, each with its own tunnel address.
