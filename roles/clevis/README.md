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

An extra LUKS keyslot whose key is offered at boot from a USB stick (by filesystem label), embedded in the initramfs and/or read from a plain file on an unencrypted local partition such as `/boot`. Gate `clevis_keyfile_enabled` (default `false`), opt-in removal `clevis_keyfile_teardown`.

```bash
# Install (no tang servers needed; auth = clevis_keyfile_auth_keyfile, else clevis_luks_passphrase, else an existing clevis binding):
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

### dracut / systemd-cryptsetup (e.g. Ubuntu 26.04)

The initramfs generator is auto-detected (fact `clevis_initramfs_generator`): `dracut` if the `dracut` binary exists and the `initramfs-tools` package is **not** installed (a stale `/etc/initramfs-tools/` or an `update-initramfs` compat wrapper does not count), else `initramfs-tools`. Force it with `clevis_initramfs_generator_override`. The keyslot, key, `keyfile.conf`, `clevis_keyfile_write_usb.sh` and the teardown of keyslot + shred are identical on both; the initramfs-tools hooks/scripts are not deployed on dracut.

- Instead of scripts, only the `/etc/crypttab` line of the detected mapper is rewritten (backup once: `/etc/crypttab.clevis-keyfile.bak`; the run fails if field 3 holds anything but `none`/`-` or a value managed here). Stick: `dm_crypt-0 UUID=... /luks-fallback.key:LABEL=LUKSKEY luks,keyfile-timeout=10s` (systemd "keyfile:device" syntax, the key is read from the stick's filesystem root). With `clevis_keyfile_embed_initramfs`: field 3 is `/etc/clevis-keyfile/fallback.key` and `/etc/dracut.conf.d/90-clevis-keyfile.conf` adds it via `install_items`. Existing options (`luks,discard`, ...) are kept; `keyfile-timeout=` is never duplicated.
- `90-clevis-keyfile.conf` also adds `usb_storage uas sd_mod vfat exfat nls_cp437 nls_iso8859_1` via `add_drivers` (hostonly dracut would not ship them). Rebuild: `dracut -f --regenerate-all`; verification reads `etc/crypttab`, the embedded key and `usb-storage` with `lsinitrd` and warns if `/boot/initrd.img-*` is not 0600.
- Fallback: with `keyfile-timeout=` systemd-cryptsetup waits for the key device and falls back to the interactive passphrase prompt if it does not appear or the key is rejected. A missing or stale stick costs only the timeout, never a lockout. There is no tang + keyfile + passphrase race on dracut; clevis/tang on dracut is not covered by this role.
- Teardown: field 3 back to `none`, `keyfile-timeout=` dropped, dracut conf removed, initramfs rebuilt, then keyslot removal and shred.

### Key file on /boot (`clevis_keyfile_boot_enabled`)

A third key source: the key is copied to `<clevis_keyfile_boot_mount>/<clevis_keyfile_boot_file>` (default `/boot/clevis-keyfile/luks-fallback.key`, directory `0700`, file `0600`, root:root). The device is derived at run time with `findmnt -nP -o UUID,SOURCE,FSTYPE <mount>` and referenced as `UUID=<uuid>`; the run fails if the mount does not exist, has no UUID or is not a local block filesystem, and warns on vfat (no POSIX permissions there).

- Why `/boot` (ext4, real `0600`) and not the ESP (vfat, world-readable): same exposure class, but permissions are enforceable.
- Security-equivalent to the embedded key (unencrypted `/boot`), but the initrd is **not** rebuilt when the key changes, because the key is read from the partition at boot. Tear down (or rotate) once tang works.
- dracut: crypttab field 3 becomes `/<boot file>:UUID=<uuid>` (systemd "path:device" syntax) plus `keyfile-timeout=`, e.g. `dm_crypt-0 UUID=<luks> /clevis-keyfile/luks-fallback.key:UUID=<boot-uuid> luks,keyfile-timeout=10s`. Field 3 holds one value only, so with several sources enabled the precedence is **embedded > boot file > usb**; a debug message names the chosen source. The crypttab editor treats any `/<boot file>:UUID=*` as managed, so a changed UUID after a reinstall is still ours. An ext4 `/boot` needs no extra drivers in a hostonly initrd; for vfat `add_drivers` gets `vfat` + nls.
- initramfs-tools: `keyfile.conf` gets `BOOT=1`, `BOOT_DEV=UUID=...`, `BOOT_FILE=...`. `local-top/clevis-keyfile` tries embedded, then boot partition (mounted read-only by `/dev/disk/by-uuid/`), then USB, using the first key that passes `--test-passphrase`; `local-bottom` unmounts it. The hook adds `ext4 vfat nls_*`.
- Teardown shreds and removes the file from the boot partition (plain `rm` where shred is not possible, e.g. vfat) and drops the empty directory.

## Task Files

### `main.yml`

Single import:
- **`config_clevis.yml`** — all clevis/tang configuration (tag: `clevis`)
- **`keyfile_fallback.yml`**: optional keyfile fallback, install + teardown blocks (tags: `clevis`, `clevis_keyfile`); runs without `wantsclevis` (the play in `common.yml` also fires for `clevis_keyfile_enabled`/`clevis_keyfile_teardown`)
- **`dracut_crypttab.yml`**: idempotent crypttab line edit of the dracut path
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

| Variable                                | Default                                                        | Description                                                                                                                  |
|-----------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `clevis_tang_servers`                   | `["http://tang.example.com", "http://tang2.example.com:9090"]` | Tang server URLs (real values in vault)                                                                                      |
| `clevis_sss_threshold`                  | `2`                                                            | SSS threshold (number of tang servers needed for decryption)                                                                 |
| `clevis_keyfile_enabled`                | `false`                                                        | Install the keyfile fallback                                                                                                 |
| `clevis_keyfile_teardown`               | `false`                                                        | Remove keyslot + files (opt-in, `-e`)                                                                                        |
| `clevis_keyfile_path`                   | `/etc/clevis-keyfile/fallback.key`                             | Master copy of the key (must stay in `/etc/clevis-keyfile`)                                                                  |
| `clevis_keyfile_usb_enabled`            | `true`                                                         | Look for the key on a USB stick                                                                                              |
| `clevis_keyfile_usb_label`              | `LUKSKEY`                                                      | Filesystem label of the stick (vfat/exfat/ext4)                                                                              |
| `clevis_keyfile_usb_file`               | `luks-fallback.key`                                            | Key file name in the stick's root                                                                                            |
| `clevis_keyfile_embed_initramfs`        | `false`                                                        | Embed the key into the initramfs (unencrypted /boot, initial boot only). On dracut: `install_items` + explicit crypttab path |
| `clevis_keyfile_boot_enabled`           | `false`                                                        | Third key source: plain key file on an unencrypted local partition (see below)                                               |
| `clevis_keyfile_boot_mount`             | `/boot`                                                        | Mountpoint of that partition; device is derived via `findmnt` and referenced as `UUID=`                                      |
| `clevis_keyfile_boot_file`              | `clevis-keyfile/luks-fallback.key`                             | Key path relative to that filesystem's root (dir `0700`, file `0600`, root:root)                                             |
| `clevis_initramfs_generator_override`   | `""`                                                           | Force `dracut` or `initramfs-tools` (empty = auto-detect)                                                                    |
| `clevis_keyfile_dracut_keyfile_timeout` | `10s`                                                          | dracut: crypttab `keyfile-timeout=` (wait for the stick, then passphrase prompt)                                             |
| `clevis_keyfile_auth_keyfile`           | `""`                                                           | Existing key file for `luksAddKey --key-file` (tried before passphrase / clevis binding)                                     |

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
│   ├── detect_luks.yml                       # LUKS device + mapper detection
│   ├── dracut_crypttab.yml                   # dracut path: crypttab line edit
│   └── keyfile_fallback.yml                  # Optional keyfile fallback (install + teardown)
├── templates/
│   ├── setup_clevis_tang.sh.j2               # Standalone rebind script
│   ├── etc_clevis-keyfile_keyfile.conf.j2    # Keyfile fallback config (sourced in initramfs)
│   ├── etc_dracut.conf.d_90-clevis-keyfile.conf.j2 # dracut: USB drivers / embedded key
│   └── clevis_keyfile_write_usb.sh.j2        # Writes the key onto a USB stick
└── files/
    ├── tang_check_connection.sh              # Tang connectivity checker
    ├── initramfs_local-bottom_cleanup-netplan # Initramfs cleanup script (deployed to init-bottom/)
    ├── clevis_keyfile_crypttab.awk           # dracut: crypttab line rewrite (unit-tested)
    ├── initramfs_hooks_clevis-keyfile        # Ships config/key/USB modules into the initramfs
    ├── initramfs_local-top_clevis-keyfile    # Background loop feeding the key into askpass
    └── initramfs_local-bottom_clevis-keyfile # Stops the loop, unmounts the stick
```
