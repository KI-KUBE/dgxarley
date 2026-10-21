# rustfs_server

Native [RustFS](https://github.com/rustfs/rustfs) S3 server as a systemd service (`rustfs.service`),
outside k3s. Standalone (`rustfs_server.yml`) or as the object store under JuiceFS (`storage.yml`
runs it before `juicefs_storage`).

| Piece        | Where                        | What                                                               |
|--------------|------------------------------|--------------------------------------------------------------------|
| **install**  | `rustfs_server_nodes`        | binary, system user, disk checks / fstab, env file, unit           |
| **users**    | `rustfs_server_primary_node` | extra S3 users with bucket-scoped policies (`rustfs_server_users`) |
| **teardown** | former nodes (opt-in)        | stops and removes unit, config, credentials, binary                |

## Single-node vs. distributed object store

`rustfs_server_distributed` (default **false**) selects the RustFS topology:

- **OFF (single-node):** RustFS runs only on `rustfs_server_primary_node`, one local
  volume, no erasure coding.
- **ON (distributed):** RustFS runs on **every** node in `rustfs_server_members`,
  forming one erasure-coded S3 cluster. Object data is striped + made redundant
  across all their USB-SSDs, so no single USB bus carries every write.

`rustfs_server_members` is a **structured, ordered** list — one entry per node,
each listing its USB disk(s) as dicts; heterogeneous paths and disk counts allowed:

```yaml
rustfs_server_members:
  - node: spark1
    disks: [{ path: /mnt/intenso, uuid: "<blkid-UUID>", fstype: ext4 }]
  - node: spark2                                        # two disks on this node
    disks:
      - { path: /mnt/usb-a, uuid: "<uuid>", fstype: ext4 }
      - { path: /mnt/usb-b, uuid: "<uuid>", fstype: ext4 }
  - node: spark3
    disks: [{ path: /mnt/intenso, uuid: "<uuid>", fstype: ext4 }]
  - node: spark4
    disks: [{ path: /mnt/intenso, uuid: "<uuid>", fstype: ext4 }]
```

Per disk, `path` is always required (RustFS data dir = `<path>/rustfs`);
`uuid`/`fstype`/`options` are consulted only when `rustfs_server_manage_fstab` is on.
Order matters (positional erasure membership; `RUSTFS_VOLUMES` is rendered
identically on every node). The **primary** (`rustfs_server_primary_node`) must be a
member; it is the single S3 endpoint clients dial. Redundancy level via `rustfs_server_raid_level`:

| `raid_level` | `EC:M` | usable (n drives) | tolerates         |
|--------------|--------|-------------------|-------------------|
| `raid0`      | `EC:0` | 100 %             | 0 (striping only) |
| `raid5`      | `EC:1` | (n−1)/n           | 1 node/disk       |
| `raid6`      | `EC:2` | (n−2)/n           | 2 nodes/disks     |

**Constraints (verified upstream):** ≥4 total drives; RAID6 needs ≥4 drives.
Distributing RustFS does not distribute JuiceFS metadata (Valkey stays single-node).

**Mounting the disks — `rustfs_server_manage_fstab`** (default **false**):

- **OFF:** you own `/etc/fstab` (mount each USB by hand, e.g.
  `UUID=… /mnt/intenso ext4 defaults,nofail 0 2`); the role only verifies the
  mountpoints and fails if one is missing.
- **ON:** the role writes a **UUID-keyed** fstab entry (via `ansible.posix.mount`,
  `state: mounted`, opts default `defaults,nofail`) and mounts each disk that
  carries a `uuid` before the mountpoint check. UUID is mandatory — USB `/dev/sdX`
  names reorder across boots and a wrong name would drop a disk into the wrong
  erasure slot. Get the UUID with `blkid /dev/sdX1`. Works single-node too (the
  primary is itself a `rustfs_server_members` entry); a disk without a `uuid` is
  skipped. Optional per-disk `dump`/`passno` (default 0) — set `passno` to match a
  pre-existing hand-written line so the first managed run doesn't rewrite it.

**⚠ Migration — enabling distributed is NOT in-place.** Single-node (SNSD) has no
erasure coding and its on-disk layout is incompatible with distributed mode.
Flipping `rustfs_server_distributed` to true on a populated store means the existing
object data is abandoned (for JuiceFS: a fresh `juicefs format`). Move data off first, then wipe the old single-node data dir before deploying
the cluster.

## Extra S3 users (`rustfs_server_users`)

Each entry gets its own buckets (created if missing) and a policy `<name>-policy` limited to them:

```yaml
rustfs_server_users:
  - name: example-access          # = S3 access key
    secret_key: "<>= 8 chars>"
    buckets: [example]
    access: readwrite             # readwrite (s3:*) | readonly (list + get)
```

The role creates missing users, re-applies changed policies and resets the secret when the user no
longer authenticates with the vault value. Buckets in `rustfs_server_protected_buckets` are refused
(JuiceFS sets `[juicefs]`). Removing an entry does **not** delete the user or bucket.

Prerequisite: the `rc` CLI (`/usr/local/bin/rc`), installed by the k3sserver role
(`roles/k3sserver/tasks/rustfs_client.yml`, e.g. `k8s_infra`/`common` runs of that role). The users
tasks and the JuiceFS format step assert its presence and fail otherwise.

## Key variables

Cross-role knobs live in `group_vars/all/main/rustfs_server.yml` (common's iptables template and
`juicefs_storage` read them): `rustfs_server_enabled`, `rustfs_server_primary_node`,
`rustfs_server_distributed`, `rustfs_server_members`, `rustfs_server_manage_fstab`,
`rustfs_server_s3_port`, `rustfs_server_access_key`/`_secret_key`, `rustfs_server_users`. Real values
go in `group_vars/all/vault/rustfs_server.sops.yml`. Everything else is in `defaults/main.yml`.

## Firewall

`common` (iptables) opens `rustfs_server_s3_port` on the primary (and every member in distributed
mode) from `k3snodes` and from `cluster_cidr` (same-node pods arrive with their pod IP). Apply with
`common.yml --tags iptables`.

## Deploy

```bash
ansible-playbook rustfs_server.yml                              # install + users
ansible-playbook rustfs_server.yml --tags rustfs_server_users   # users only
```

Before install the role validates `rustfs_server_members` once (primary is a member, unique node
names, non-empty disk lists with unique paths, exactly one primary disk in single-node mode, and the
raid/drive-count constraints in distributed mode).

## Teardown

`rustfs_server_teardown=true` stops and removes the unit, config, credentials, state dir and binary.
Object data on the disks and the fstab entries are never touched. Install and users tasks do not run
while it is set. Two modes:

| Mode                              | Precondition                             | Effect                                                               |
|-----------------------------------|------------------------------------------|----------------------------------------------------------------------|
| **Former node, cluster stays on** | `rustfs_server_enabled: true`            | hosts still in `rustfs_server_nodes` are skipped                     |
| **Full decommission**             | set `rustfs_server_enabled: false` first | `rustfs_server_nodes` is empty, every host in the limit is torn down |

```bash
# former node while the cluster stays enabled
ansible-playbook rustfs_server.yml -e rustfs_server_teardown=true --limit <former-node>

# full decommission (after setting rustfs_server_enabled: false)
ansible-playbook rustfs_server.yml -e rustfs_server_teardown=true
```

`juicefs_backend_teardown` (Valkey, role `juicefs_storage`) and `rustfs_server_teardown` are
independent flags. Decommissioning a former combined storage node needs both:

```bash
ansible-playbook storage.yml -e juicefs_backend_teardown=true -e rustfs_server_teardown=true --limit <former-node>
```

RustFS has no native `/metrics` endpoint (OTel push only, rustfs#3154).
