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

## Keyfile Fallback (optional, tag `clevis_keyfile`)

An extra LUKS keyslot whose key is offered at boot from a USB stick (by filesystem label) and/or embedded in the initramfs. Gate `clevis_keyfile_enabled` (default `false`), opt-in removal `clevis_keyfile_teardown`.

```bash
# Install (no tang servers needed; auth = clevis_luks_passphrase, else an existing clevis binding):
ansible-playbook common.yml --tags clevis_keyfile -e clevis_keyfile_enabled=true

# Put the key on a stick (partition must be labelled LUKSKEY, e.g. mkfs.vfat -n LUKSKEY /dev/sdb1):
/root/clevis_keyfile_write_usb.sh /dev/sdb1

# Teardown: removes the keyslot (refuses if it is the last one), initramfs scripts, shreds the key:
ansible-playbook common.yml --tags clevis_keyfile -e clevis_keyfile_teardown=true
```

- **crypttab is not modified.** `local-top/clevis-keyfile` works like the clevis initramfs script: a background loop waits for cryptroot's `askpass`, and writes the key into its `passfifo`. Tang, keyfile and typed passphrase race; the first valid answer wins. The stick may be plugged in while the prompt is shown.
- A key is only written after `cryptsetup --test-passphrase` accepted it, so a stale stick never consumes one of cryptroot's tries.
- The key is 64 ASCII chars (base64) without a trailing newline, as required by the passfifo path. Master copy: `/etc/clevis-keyfile/fallback.key` (on the encrypted root).
- `clevis_keyfile_embed_initramfs: true` puts the key into the initramfs, i.e. **unencrypted on /boot** (sets `UMASK=0077` for the initrd). Only for the initial boot: switch it off, or tear down, once tang works. The teardown removes the keyslot, so leftover copies in old initrds or on sticks become useless.
- `setup_clevis_tang.sh` only unbinds clevis slots and leaves the fallback keyslot alone.

## Task Files

### `main.yml`

Single import:
- **`config_clevis.yml`** — all clevis/tang configuration (tag: `clevis`)
- **`keyfile_fallback.yml`**: optional keyfile fallback, install + teardown blocks (tags: `clevis`, `clevis_keyfile`)
- **`detect_luks.yml`**: LUKS device detection, imported by both

### `config_clevis.yml`

The main task file containing all logic. Read-only tasks use `check_mode: false` for `--check --diff` support. The bind step writes a temporary keyfile (`/tmp/.clevis_luks_key`) which is always cleaned up via an `always` block.

## Tags

| Tag | Scope |
|---|---|
| `clevis` | All clevis/tang tasks (incl. keyfile fallback) |
| `clevis_keyfile` | Keyfile fallback only (runs without reachable tang servers) |

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
| `clevis_keyfile_enabled` | `false` | Install the keyfile fallback |
| `clevis_keyfile_teardown` | `false` | Remove keyslot + files (opt-in, `-e`) |
| `clevis_keyfile_path` | `/etc/clevis-keyfile/fallback.key` | Master copy of the key (must stay in `/etc/clevis-keyfile`) |
| `clevis_keyfile_usb_enabled` | `true` | Look for the key on a USB stick |
| `clevis_keyfile_usb_label` | `LUKSKEY` | Filesystem label of the stick (vfat/exfat/ext4) |
| `clevis_keyfile_usb_file` | `luks-fallback.key` | Key file name in the stick's root |
| `clevis_keyfile_embed_initramfs` | `false` | Embed the key into the initramfs (unencrypted /boot, initial boot only) |

## Helper Scripts

| Script | Description |
|---|---|
| `setup_clevis_tang.sh` | Standalone rebind script (templatized). Auto-detects UUID and network driver at runtime. Usage: `/root/setup_clevis_tang.sh [PASSPHRASE]` |
| `tang_check_connection.sh` | Verifies tang server connectivity and clevis binding. Usage: `/root/tang_check_connection.sh <luks-device>` |
| `clevis_keyfile_write_usb.sh` | Copies the fallback key onto a labelled USB partition and verifies it unlocks the LUKS device. Usage: `/root/clevis_keyfile_write_usb.sh /dev/sdX1` |

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
│   ├── config_clevis.yml                     # All clevis/tang configuration
│   ├── detect_luks.yml                       # LUKS device detection
│   └── keyfile_fallback.yml                  # Optional keyfile fallback (install + teardown)
├── templates/
│   ├── setup_clevis_tang.sh.j2               # Standalone rebind script
│   ├── etc_clevis-keyfile_keyfile.conf.j2    # Keyfile fallback config (sourced in initramfs)
│   └── clevis_keyfile_write_usb.sh.j2        # Writes the key onto a USB stick
└── files/
    ├── tang_check_connection.sh              # Tang connectivity checker
    ├── initramfs_local-bottom_cleanup-netplan # Initramfs cleanup script (deployed to init-bottom/)
    ├── initramfs_hooks_clevis-keyfile        # Ships config/key/USB modules into the initramfs
    ├── initramfs_local-top_clevis-keyfile    # Background loop feeding the key into askpass
    └── initramfs_local-bottom_clevis-keyfile # Stops the loop, unmounts the stick
```
