# clevis role

Configures automatic LUKS disk decryption via clevis/tang (Network-Bound Disk Encryption). Applied per-host via `wantsclevis: true` — no group membership required. The role is invoked from `common.yml` with `when: wantsclevis | default(false)`.

## Usage

```bash
# Initial bind (LUKS passphrase required):
ansible-playbook common.yml --tags clevis \
  -e 'clevis_luks_passphrase=YOUR_LUKS_PASSPHRASE'

# Subsequent runs (idempotent, no passphrase needed):
ansible-playbook common.yml --tags clevis

# Dry run:
ansible-playbook common.yml --tags clevis --check --diff
```

## What it does

1. Installs clevis packages (`clevis`, `clevis-luks`, `clevis-initramfs`, `clevis-systemd`)
2. Auto-detects the LUKS partition from `/etc/crypttab` (supports UUID= format)
3. Checks for existing clevis bindings (skips bind if already present)
4. Verifies tang server connectivity, fetches advertisements, and creates clevis/tang SSS binding (threshold 2)
5. Configures initramfs networking: adds network driver to modules, sets `DEVICE=<hw_iface>` and `IP=:::::<hw_iface>:dhcp` (full kernel IP format — `IP=dhcp` alone is misinterpreted as interface name)
6. Deploys `init-bottom/cleanup-netplan` script to remove stale `/run/netplan/*` before switch_root (prevents OVS interaction issues)
7. Rebuilds initramfs (only when changes occurred)
8. Deploys helper scripts to `/root/`: `setup_clevis_tang.sh` (templatized rebind script) and `tang_check_connection.sh` (verification)
9. Verifies clevis hooks and network driver are present in the built initramfs

The existing LUKS passphrase remains valid as fallback. If tang servers are unreachable at boot, the normal password prompt appears.

## Task Files

### `main.yml`

Single import:
- **`config_clevis.yml`** — all clevis/tang configuration (tag: `clevis`)

### `config_clevis.yml`

The main task file containing all logic. Read-only tasks use `check_mode: false` for `--check --diff` support. The bind step writes a temporary keyfile (`/tmp/.clevis_luks_key`) which is always cleaned up via an `always` block.

## Tags

| Tag | Scope |
|---|---|
| `clevis` | All clevis/tang tasks |

## Host Variables

| Variable | Description |
|---|---|
| `wantsclevis: true` | Enables the clevis role for this host |
| `hw_iface` | Physical hardware NIC (used for initramfs networking) |
| `clevis_luks_device_override` | Override auto-detected LUKS device (optional) |
| `clevis_luks_passphrase` | LUKS passphrase for initial bind (pass via `-e`, required only on first run) |

## Defaults

| Variable | Default | Description |
|---|---|---|
| `clevis_tang_servers` | `["http://tang.example.com", "http://tang2.example.com:9090"]` | Tang server URLs (real values in vault) |
| `clevis_sss_threshold` | `2` | SSS threshold (number of tang servers needed for decryption) |

## Helper Scripts

| Script | Description |
|---|---|
| `setup_clevis_tang.sh` | Standalone rebind script (templatized). Auto-detects UUID and network driver at runtime. Usage: `/root/setup_clevis_tang.sh [PASSPHRASE]` |
| `tang_check_connection.sh` | Verifies tang server connectivity and clevis binding. Usage: `/root/tang_check_connection.sh <luks-device>` |

## Clevis + OVS Interaction

On hosts with both clevis and OVS, initramfs DHCP can leak: `configure_networking()` runs twice in initramfs `init`, and the second call recreates `/run/netplan/*.yaml` after clevis tears down networking, causing unwanted DHCP on the bare interface after switch_root.

The `init-bottom/cleanup-netplan` script removes `/run/netplan/*` before switch_root. It must be `init-bottom` (not `local-bottom`) because `configure_networking()` runs after `local-bottom`. The role also removes any stale `local-bottom/cleanup-netplan` from the wrong location.

## Directory Structure

```
roles/clevis/
├── defaults/
│   └── main.yml                              # Tang servers and SSS threshold
├── tasks/
│   ├── main.yml                              # Imports config_clevis.yml
│   └── config_clevis.yml                     # All clevis/tang configuration
├── templates/
│   └── setup_clevis_tang.sh.j2               # Standalone rebind script
└── files/
    ├── tang_check_connection.sh              # Tang connectivity checker
    └── initramfs_local-bottom_cleanup-netplan # Initramfs cleanup script (deployed to init-bottom/)
```
