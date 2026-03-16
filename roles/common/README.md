# common role

Base system configuration role applied to all managed hosts (except `hetzner` and `disabled` groups). It handles package installation, networking, security hardening, mail relay, firewall management, and scheduled maintenance.

## Usage

```bash
ansible-playbook common.yml

# Run specific subsystems
ansible-playbook common.yml --tags fail2ban
ansible-playbook common.yml --tags iptables
ansible-playbook common.yml --tags netplan
```

## What it configures

### Package management (`installbasicpackages`)
- Removes snapd
- Optionally removes AppArmor on cloud/DNS servers (`deinstall_apparmor: true`)
- Installs essential packages: `net-tools`, `screen`, `vim`, `htop`, `jq`, `lvm2`, `mdadm`, `avahi-daemon`, `rsyslog`, `fail2ban`, `postfix`, `sysstat`, `iptables`, `ipset`, `curl`, `rsync`, etc.
- Installs NFS kernel server when `isnfsmaster: true`
- Runs full upgrade and autoremove

### Network (`netplan`, `avahi`)
- **Netplan**: Configures ethernet and optional WiFi interfaces via templates (conditional on `wantsnetplan: true`). Supports Open vSwitch bridge/VLAN configs when `wantsopenvswitch: true`. When `hw_iface == ovs_bridge_iface`, OVS takes over the NIC and regular netplan is not deployed. When `hw_iface != ovs_bridge_iface`, both regular netplan (for `hw_iface`) and OVS configs are deployed side by side. All VLANs (including `untagged: true` and `native: true`) use netplan fake bridges. For native/untagged VLANs, a systemd drop-in clears and reapplies `vlan_mode`/`tag`/`trunks` around `netplan apply` to keep it idempotent. Installs `systemd-resolved` on Debian 12+. Removes NetworkManager.
- **Verifying OVS native VLAN config:**
  ```bash
  # Show vlan_mode, tag, and trunks for a port
  ovs-vsctl get port <port> vlan_mode tag trunks
  # After configuration: native-untagged / 101 / [101]
  # Before (or unconfigured): [] / 0 / []

  # Full port details
  ovs-vsctl list port <port>
  ```
- **Avahi mDNS**: Enables local service discovery on home lab hosts. Listens on `default_route_iface` (computed: primary VLAN for OVS hosts, `hw_iface` otherwise), plus `hw_iface` if it's a separate NIC. Disabled on cloud/DNS server groups.

### SSH hardening (`sshconfig`)
- Deploys hardened `/etc/ssh/sshd_config` (IPv4 only, root login with pubkey only, challenge-response disabled)
- Adds local RSA and/or Ed25519 public keys to root's `authorized_keys` (whichever exist in `~/.ssh/`)

### Firewall (`iptables`)
- **iptables.sh** (template): IPv4 firewall rules with custom HTSTUFFIN/HTSTUFFOUT chains, local network allowlists, K3s-specific ports
- **ip6tables_setup.sh**: IPv6 firewall with ICMPv6 rules and default DROP policy
- **ipsetgeoblock.sh** / **ipsetgeoallow.sh** (templates): Geographic IP filtering using ipdeny.com zone data
- **blacklist.sh**: Downloads IP blocklists from Emerging Threats, blocklist.de, badips.com
- **blockbadhosts_ipset.sh**: Loads manual blacklist from `/root/badhosts.txt`
- **savetablesandipsets.sh**: Persists iptables/ipset rules to disk (excludes Kubernetes rules)
- **rc.local** (template): Restores saved rules on boot

### Intrusion prevention (`fail2ban`)
- SSH jail and recidive jail (7-day bans) enabled by default
- Email notifications to configurable recipient
- Configurable IP ignore list

### Mail relay (`postfix`)
- Configures Postfix with relay host (default: `mail.example.com`)
- TLS encryption for outgoing mail
- Skipped when `skip_common_postfix` is set

### System monitoring (`sysstat`)
- Enables sysstat collection and summary timers

### Locale and timezone (`locale`)
- Sets locale to `de_DE.UTF-8`
- Sets timezone to `Europe/Berlin`

### MOTD (`fortune`)
- Installs fortune packages for dynamic login messages

### USB autosuspend (`usb_autosuspend`)
- Disables USB autosuspend globally via GRUB kernel parameter (`usbcore.autosuspend=-1`)
- Deploys udev rule to disable autosuspend per-device for Realtek RTL8156B USB Ethernet adapters (known to disconnect with autosuspend enabled)
- Applied to all hosts where GRUB is present

### Kernel module blacklist (`blacklist_simpledrm`)
- Blacklists the `simpledrm` kernel module so the Intel iGPU is assigned to `card0`
- Deploys `/etc/modprobe.d/blacklist-simpledrm.conf` and triggers `update-initramfs -u`
- Conditional on `blacklist_simpledrm: true` host variable

### Podman (`podman`)
- Installs Podman and enables the system-level `podman.socket`
- Enables linger for root
- Adds a daily cron job for `podman system prune`
- Conditional on `wantspodman: true`

### Clevis/Tang NBDE (`clevis`)
- Automatic LUKS disk decryption via clevis/tang (Network-Bound Disk Encryption)
- Conditional on `wantsclevis: true` host variable
- Auto-detects LUKS partition from `/etc/crypttab`
- Binds clevis/tang SSS (Shamir's Secret Sharing) with configurable threshold
- Configures initramfs networking (interface driver, DEVICE, `IP=:::::<device>:dhcp`) for early-boot decryption using the Linux kernel IP autoconfiguration format
- Deploys `setup_clevis_tang.sh` (templatized rebind script) and `tang_check_connection.sh` (verification script) to `/root/`
- `setup_clevis_tang.sh` is a standalone fallback script for rebinding clevis/tang without Ansible (auto-detects UUID and network driver at runtime)
- Initial bind requires LUKS passphrase via `-e 'clevis_luks_passphrase=...'`; subsequent runs are idempotent
- Supports `--check --diff` dry runs (read-only tasks use `check_mode: false`)

### Cron jobs (`crontab`)
- Daily firewall rule refresh (04:00)
- Daily geo-block/allow updates (06:00 / 06:10)
- Daily IP blacklist updates (05:00 / 05:10)
- Daily rule persistence save (12:00)

## Host variables

| Variable | Type | Purpose |
|---|---|---|
| `wantsnetplan` | bool | Enable netplan networking |
| `isnfsmaster` | bool | Install NFS kernel server |
| `hw_iface` | string | Physical hardware NIC |
| `hw_iface2` | string | Secondary hardware NIC (optional) |
| `hw_iface2_address` | string | Secondary interface IP/CIDR (optional) |
| `default_route_iface` | string | Routable interface (computed from OVS/hw_iface, can be overridden) |
| `wantsopenvswitch` | bool | Enable Open vSwitch bridge networking (optional) |
| `ovs_bridge_iface` | string | Physical NIC for OVS bridge port (required if `wantsopenvswitch`) |
| `ovs_bridge_name` | string | OVS bridge name (required if `wantsopenvswitch`) |
| `ovs_vlans` | list(dict) | VLAN definitions; one must have `primary: true`, at most one may have `untagged: true` or `native: true` (required if `wantsopenvswitch`) |
| `wifi_iface` | string | WiFi interface name (optional) |
| `wifi_access_points` | list(dict) | WiFi access points `{ssid, password}` for netplan (optional) |
| `mailname` | string | Hostname for Postfix |
| `fail2ban_recipient` | string | Email for fail2ban alerts |
| `fail2ban_sendermail` | string | Sender email for fail2ban |
| `fail2ban_ignoreips` | list | IPs exempt from fail2ban (optional) |
| `localnets` | list(dict) | Local network CIDRs for iptables |
| `localnets_k3sserver` | list(dict) | K3s-specific local networks (optional) |
| `iswireguardhost` | bool | Host runs WireGuard (optional) |
| `geoblock` | list | Country codes to geo-block (optional) |
| `geoallow` | list | Country codes to geo-allow (optional) |
| `blacklist_simpledrm` | bool | Blacklist simpledrm kernel module for Intel iGPU card0 (optional) |
| `wantspodman` | bool | Install and configure Podman (optional) |
| `wantsclevis` | bool | Enable clevis/tang NBDE auto-decryption (optional) |
| `clevis_luks_device_override` | string | Override auto-detected LUKS device (optional) |

## Defaults

| Variable | Default | Purpose |
|---|---|---|
| `postfix_relayhost` | `[mail.example.com]` | Outgoing mail relay |
| `deinstall_apparmor` | `false` | Remove AppArmor on cloud/DNS servers |
| `smartd_mail_recipients` | `root,admin@example.com` | smartd alert recipients |
| `iptables_xosrc_ips` | `[]` | External trusted source IPs for iptables |
| `iptables_lifeline_ips` | `[]` | Lifeline IPs (always allowed) |
| `iptables_k3s_external_ips` | `[]` | External IPs allowed to reach K3s ports |
| `iptables_static_dns_ips` | `[]` | Static DNS server source IPs for iptables |
| `dyndns_hostname` | `dyn.example.com` | DynDNS hostname for ipset updates |
| `wifi_access_points` | `[]` | WiFi access points for netplan (real values in vault) |
| `clevis_tang_servers` | `["http://tang.example.com", "http://tang2.example.com:9090"]` | Tang server URLs for NBDE (real values in vault) |
| `clevis_sss_threshold` | `2` | SSS threshold (number of tang servers needed) |

## Group-specific behavior

| Group | Behavior |
|---|---|
| `hetzner`, `hetznercloudnew` | Avahi disabled, optional AppArmor removal |
| `dnsserver` | Avahi disabled, DNS-specific iptables rules |
| `k3sserver` | K3s ports (6443, 6720) in firewall, async iptables at boot, `ip_nonlocal_bind=1` sysctl |
| `krennstor101` | SSH config and copyfiles skipped, special cron jobs |

## Tags

`installbasicpackages`, `avahi`, `netplan`, `fortune`, `sshconfig`, `postfix`, `fail2ban`, `iptables`, `crontab`, `sysstat`, `locale`, `sysctl`, `usb_autosuspend`, `podman`, `blacklist_simpledrm`, `upgradestuff`, `clevis`

## Handlers

- `restart sshd` / `restart postfix` / `restart fail2ban` - Service restarts
- `localegen` - Regenerates locale
- `execiptablessh` - Runs `/root/iptables.sh`
- `execrootscripts` - Runs all geo-block/blacklist/persistence scripts
- `sysctl reload` - Reloads sysctl settings (`sysctl -p`)
- `update grub` - Runs `update-grub`
- `reload udev rules` - Reloads udev rules (`udevadm control --reload-rules`)
- `update initramfs` - Rebuilds initramfs (`update-initramfs -u`)
