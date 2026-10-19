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

0. Detects the initramfs generator (`dracut` or `initramfs-tools`, see below); steps 5-9 below describe the initramfs-tools path, dracut hosts: section "Tang on dracut"
1. Installs clevis packages (`clevis`, `clevis-luks`, `clevis-systemd` + `clevis-initramfs` on initramfs-tools, `clevis-dracut` on dracut)
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

An extra LUKS keyslot whose key is offered at boot from a USB stick (by filesystem label) and/or read from a plain file on an unencrypted local partition such as `/boot`. The key is never put into the initrd itself. Gate `clevis_keyfile_enabled` (default `false`), opt-in removal `clevis_keyfile_teardown`.

```bash
# Install (no tang servers needed; auth = clevis_keyfile_auth_keyfile, else clevis_luks_passphrase, else an existing clevis binding):
ansible-playbook common.yml --tags clevis_keyfile -e clevis_keyfile_enabled=true

# Put the key on a stick (partition must be labelled LUKSKEY, e.g. mkfs.vfat -n LUKSKEY /dev/sdb1):
/root/clevis_keyfile_write_usb.sh /dev/sdb1

# Teardown: removes the keyslot (refuses if it is the last one), initramfs scripts, shreds the key:
ansible-playbook common.yml --tags clevis_keyfile -e clevis_keyfile_teardown=true
```

- **Race model (both generators): crypttab is never modified.** All enabled key sources (key file on `/boot`, USB stick) are offered at boot in parallel to the console passphrase prompt (and to tang, see below); the first valid answer wins. The stick may be plugged in while the prompt is shown. initramfs-tools: `local-top/clevis-keyfile` works like the clevis initramfs script, a background loop waits for cryptroot's `askpass` and writes the key into its `passfifo`. dracut: a systemd password agent (below).
- A key is only written after `cryptsetup --test-passphrase` accepted it, so a stale stick never consumes one of cryptroot's tries.
- The key is 64 ASCII chars (base64) without a trailing newline, as required by the passfifo path. Master copy: `/etc/clevis-keyfile/fallback.key` (on the encrypted root).
- **No initrd embedding.** The former option `clevis_keyfile_embed_initramfs` was removed (2026-10); setting it now fails the run. On a host that used it, the next run rebuilds every initrd without the key, drops the old `UMASK=0077` override and fails if any `/boot/initrd.img-*` still carries `etc/clevis-keyfile/fallback.key`. The key is still enrolled, so remove it with the teardown (or rotate) if those initrds were ever copied elsewhere.
- `setup_clevis_tang.sh` only unbinds clevis slots and leaves the fallback keyslot alone.

### dracut / systemd-cryptsetup (e.g. Ubuntu 26.04)

The initramfs generator is auto-detected (fact `clevis_initramfs_generator`): `dracut` if the `dracut` binary exists and the `initramfs-tools` package is **not** installed (a stale `/etc/initramfs-tools/` or an `update-initramfs` compat wrapper does not count), else `initramfs-tools`. Force it with `clevis_initramfs_generator_override`. The keyslot, key, `keyfile.conf`, `clevis_keyfile_write_usb.sh` and the teardown of keyslot + shred are identical on both; the initramfs-tools hooks/scripts are not deployed on dracut.

- **systemd password agent as dracut module.** The role deploys the module `clevis-keyfile` to `/usr/lib/dracut/modules.d/90clevis-keyfile/` (`module-setup.sh`, `clevis-keyfile-agent.sh`, `clevis-keyfile-agent.service`) and `/etc/dracut.conf.d/90-clevis-keyfile.conf` (`add_dracutmodules+=" clevis-keyfile "`). In the initrd, `clevis-keyfile-agent.service` (wanted by `cryptsetup.target`, never a hard dependency) runs `/usr/libexec/clevis-keyfile-agent`. Every second it looks at `/run/systemd/ask-password/ask.*`, and for each `Id=cryptsetup:<device>` it tries the `/boot` key file and then the USB key (in that order, a failing source never blocks the next) with `cryptsetup open --test-passphrase`; the first key that unlocks is sent with `systemd-reply-password 1 <socket>`. The console prompt (plymouth / tty agent) keeps running in parallel and answers the same ask file, so typing a passphrase works at any time. A key rejected for an ask file is not retried until its content changes; after a stale or unreadable source the loop backs off 4 s. The agent keeps running until the unit is stopped before switch-root, it never gives up while ask files exist. The key is never logged.
- The module adds the kernel modules `usb_storage uas sd_mod vfat exfat nls_cp437 nls_iso8859_1 ext4` (hostonly dracut would not ship them), never the key. `check()` includes the module only while `/etc/clevis-keyfile/keyfile.conf` exists. Rebuild: `dracut -f /boot/initrd.img-<v> <v>` for every kernel that still has a `/boot/vmlinuz-<v>` (not `--regenerate-all`, which trips over the depmod leftovers of purged kernels). Verification with `lsinitrd` requires the agent, the unit and `keyfile.conf` in the initrd, fails if any initrd still carries the key (left over from the removed embedding option), and warns if the USB/boot kernel modules are missing (fine only if built in).
- **crypttab stays `none`.** An older version of this role edited the mapper line (`keyfile-timeout=`, `/luks-fallback.key:LABEL=...`). If such an edit is still there, the run **fails** and asks the operator to restore field 3 to `none` and drop `keyfile-timeout=` (backup of the original: `/etc/crypttab.clevis-keyfile.bak`); crypttab is never edited silently.
- **tang on dracut** is provided by the distro's `clevis-dracut` package and configured by this role (next section); it answers the same ask-password requests.
- Teardown: dracut module and conf removed, initramfs rebuilt, then keyslot removal and shred.

### Key file on /boot (`clevis_keyfile_boot_enabled`)

A second key source: the key is copied to `<clevis_keyfile_boot_mount>/<clevis_keyfile_boot_file>` (default `/boot/clevis-keyfile/luks-fallback.key`, directory `0700`, file `0600`, root:root). The device is derived at run time with `findmnt -nP -o UUID,SOURCE,FSTYPE <mount>` and referenced as `UUID=<uuid>`; the run fails if the mount does not exist, has no UUID or is not a local block filesystem, and warns on vfat (no POSIX permissions there).

- Why `/boot` (ext4, real `0600`) and not the ESP (vfat, world-readable): same exposure class, but permissions are enforceable.
- The key lies unencrypted on `/boot`: anyone with the disk can read it. The initrd is **not** rebuilt when the key changes, because the key is read from the partition at boot. Tear down (or rotate) once tang works.
- dracut: the agent resolves `BOOT_DEV` (`UUID=`, also `LABEL=`/`PARTUUID=`) to `/dev/disk/by-*`, mounts it read-only on `/run/clevis-keyfile-boot` (retry with `ro,noload` for a dirty ext4 journal), reads the key, unmounts again.
- Both: `keyfile.conf` gets `BOOT=1`, `BOOT_DEV=UUID=...`, `BOOT_FILE=...`. initramfs-tools `local-top/clevis-keyfile` tries the boot partition (mounted read-only by `/dev/disk/by-uuid/`), then USB, using the first key that passes `--test-passphrase`; `local-bottom` unmounts it. The hook adds `ext4 vfat nls_*`.
- Teardown shreds and removes the file from the boot partition (plain `rm` where shred is not possible, e.g. vfat) and drops the empty directory (`tasks/remove_boot_keyfile.yml`). The same happens on a normal run with `clevis_keyfile_boot_enabled: false`, so switching the option off never leaves the key behind on `/boot`.

## Tang on dracut (systemd-cryptsetup, e.g. Ubuntu 26.04)

With `clevis_initramfs_generator == 'dracut'` (auto-detected, see the keyfile section; shared include `tasks/detect_generator.yml`) `wantsclevis: true` unlocks LUKS via tang at boot. Binding, tang check, binding verification and the helper scripts are identical on both generators; only the initrd part differs. The initramfs-tools tasks are skipped (`initramfs.conf`, `/etc/initramfs-tools/*`, `cleanup-netplan`, dhcpcd hook, `update-initramfs`), `wantsclevis_initramfs: false` skips the initrd steps on both.

- **Packages:** `clevis clevis-luks clevis-systemd clevis-dracut` (not `clevis-initramfs`). `clevis-dracut` depends on `clevis-systemd`, `dracut` and `dracut-network`; `clevis-systemd` on `clevis-luks` and `systemd`. `clevis-dracut` ships only dracut modules (`/usr/lib/dracut/modules.d/60clevis/{module-setup.sh,clevis-hook.sh}`, `60clevis-pin-{null,sss,tang,tpm2}`); the password agent comes from `clevis-systemd`: `/usr/libexec/clevis-luks-askpass` with `clevis-luks-askpass.path` (watches `/run/systemd/ask-password`) and `.service`.
- **Network in the initrd:** `/etc/dracut.conf.d/90-clevis.conf` (template `etc_dracut.conf.d_90-clevis.conf.j2`) holds `add_dracutmodules+=" clevis <clevis_dracut_extra_modules> network "`, `kernel_cmdline+=" rd.neednet=1 ip=<iface>:dhcp rd.net.timeout.carrier=.. rd.net.dhcp.retry=.. "` and `add_drivers+=" <nic driver> "`. `<iface>` is `hw_iface` (fallback `ansible_default_ipv4.interface`), the driver comes from the same detection task as on initramfs-tools. dracut writes `kernel_cmdline` into the initrd (`etc/cmdline.d`), **no GRUB edit** is needed. The `network` meta-module picks the backend (NetworkManager or systemd-networkd, both release their lease themselves, therefore no `dhcp-release` script on dracut). `rd.neednet=1` is what makes the initrd configure the NIC.
- **Race (unit names):** in the initrd `clevis-luks-askpass.path` fires on `/run/systemd/ask-password/ask.*`, `clevis-luks-askpass.service` runs `/usr/libexec/clevis-luks-askpass`, which recovers the key from the tang/sss JWE of the LUKS header and answers the `Id=cryptsetup:<device>` request. In parallel `clevis-keyfile-agent.service` (this role, keyfile fallback) and the console prompt (plymouth / `systemd-tty-ask-password-agent`) answer the same ask file. The first valid answer wins, crypttab is untouched (`none`), unreachable tang just leaves the other sources.
- **Rebuild:** `tasks/dracut_rebuild.yml` (shared with the keyfile fallback): `dracut -f /boot/initrd.img-<v> <v>` for every kernel with a `/boot/vmlinuz-<v>`; runs on first bind or when `90-clevis.conf` changed.
- **Verification (`lsinitrd`, fails on a miss):** `usr/libexec/clevis-luks-askpass`, `clevis-luks-askpass.path/.service`, `clevis-decrypt-tang`, `clevis-decrypt-sss`, a network backend, the NIC driver `.ko`; warns if `rd.neednet=1` is not in `etc/cmdline.d`.
- **Teardown:** the tang path has none on either generator (unbind via `setup_clevis_tang.sh`, which asks before removing bindings). Without `wantsclevis`, remove `/etc/dracut.conf.d/90-clevis.conf` and rebuild by hand. `setup_clevis_tang.sh` is generator-aware: on dracut it re-binds and rebuilds with dracut and leaves `90-clevis.conf` to Ansible.

## Task Files

### `main.yml`

Single import:
- **`config_clevis.yml`** — all clevis/tang configuration (tag: `clevis`)
- **`keyfile_fallback.yml`**: optional keyfile fallback, install + teardown blocks (tags: `clevis`, `clevis_keyfile`); runs without `wantsclevis` (the play in `common.yml` also fires for `clevis_keyfile_enabled`/`clevis_keyfile_teardown`)
- **`detect_luks.yml`**: LUKS device detection, imported by both
- **`remove_boot_keyfile.yml`**: removes the key copy from `/boot`, imported by the keyfile install (option off) and teardown
- **`detect_generator.yml`**: sets `clevis_initramfs_generator` (guarded by the marker fact `clevis_generator_detected`, so it runs once per host), imported by both
- **`dracut_rebuild.yml`**: kernel-exact `dracut -f` loop, imported by the dracut tang path and the keyfile fallback (with a `when:` each)

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

| Variable                              | Default                                                        | Description                                                                              |
|---------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `clevis_tang_servers`                 | `["http://tang.example.com", "http://tang2.example.com:9090"]` | Tang server URLs (real values in vault)                                                  |
| `clevis_sss_threshold`                | `2`                                                            | SSS threshold (number of tang servers needed for decryption)                             |
| `clevis_keyfile_enabled`              | `false`                                                        | Install the keyfile fallback                                                             |
| `clevis_keyfile_teardown`             | `false`                                                        | Remove keyslot + files (opt-in, `-e`)                                                    |
| `clevis_keyfile_path`                 | `/etc/clevis-keyfile/fallback.key`                             | Master copy of the key (must stay in `/etc/clevis-keyfile`)                              |
| `clevis_keyfile_usb_enabled`          | `true`                                                         | Look for the key on a USB stick                                                          |
| `clevis_keyfile_usb_label`            | `LUKSKEY`                                                      | Filesystem label of the stick (vfat/exfat/ext4)                                          |
| `clevis_keyfile_usb_file`             | `luks-fallback.key`                                            | Key file name in the stick's root                                                        |
| `clevis_keyfile_boot_enabled`         | `false`                                                        | Second key source: plain key file on an unencrypted local partition (see below)          |
| `clevis_keyfile_boot_mount`           | `/boot`                                                        | Mountpoint of that partition; device is derived via `findmnt` and referenced as `UUID=`  |
| `clevis_keyfile_boot_file`            | `clevis-keyfile/luks-fallback.key`                             | Key path relative to that filesystem's root (dir `0700`, file `0600`, root:root)         |
| `clevis_initramfs_generator_override` | `""`                                                           | Force `dracut` or `initramfs-tools` (empty = auto-detect)                                |
| `clevis_keyfile_auth_keyfile`         | `""`                                                           | Existing key file for `luksAddKey --key-file` (tried before passphrase / clevis binding) |
| `clevis_dracut_extra_modules`         | `[clevis-pin-tang, clevis-pin-sss]`                            | dracut: modules added next to `clevis` and `network` in `90-clevis.conf`                 |
| `clevis_dracut_net_timeout_carrier`   | `15`                                                           | dracut: `rd.net.timeout.carrier` seconds (0 = omit)                                      |
| `clevis_dracut_net_dhcp_retry`        | `3`                                                            | dracut: `rd.net.dhcp.retry` attempts (0 = omit)                                          |

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
│   ├── detect_generator.yml                  # dracut | initramfs-tools detection (fact)
│   ├── dracut_rebuild.yml                    # kernel-exact dracut -f loop (tang + keyfile)
│   ├── keyfile_fallback.yml                  # Optional keyfile fallback (install + teardown)
│   └── remove_boot_keyfile.yml               # Removes the key copy from /boot
├── templates/
│   ├── setup_clevis_tang.sh.j2               # Standalone rebind script (generator-aware)
│   ├── etc_dracut.conf.d_90-clevis.conf.j2   # dracut: clevis + network + ip= + NIC driver (tang)
│   ├── etc_clevis-keyfile_keyfile.conf.j2    # Keyfile fallback config (sourced in initramfs)
│   └── clevis_keyfile_write_usb.sh.j2        # Writes the key onto a USB stick
└── files/
    ├── tang_check_connection.sh              # Tang connectivity checker
    ├── initramfs_local-bottom_cleanup-netplan # Initramfs cleanup script (deployed to init-bottom/)
    ├── dracut/90-clevis-keyfile.conf         # dracut: add_dracutmodules+=" clevis-keyfile "
    ├── dracut/90clevis-keyfile/              # dracut module (password agent), deployed to /usr/lib/dracut/modules.d/
    │   ├── module-setup.sh
    │   ├── clevis-keyfile-agent.sh           # answers systemd-cryptsetup ask-password requests (unit-tested)
    │   └── clevis-keyfile-agent.service
    ├── initramfs_hooks_clevis-keyfile        # Ships config/key/USB modules into the initramfs
    ├── initramfs_local-top_clevis-keyfile    # Background loop feeding the key into askpass
    └── initramfs_local-bottom_clevis-keyfile # Stops the loop, unmounts the stick
```
