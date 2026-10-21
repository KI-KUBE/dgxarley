# juicefs_storage

Distributed JuiceFS storage layer for the DGX cluster, deployed **entirely as
host-level systemd services, outside k3s**. Pools large LLM weights across nodes,
caches hot data per-node on the NVMe, and persists chunks to the RustFS S3 object
store (`rustfs_server` role).

## Why host-level (not a DaemonSet / CSI)

A FUSE mount inside a pod dies with the pod (image update, OOM, k3s upgrade,
eviction) → stale mount, `Transport endpoint is not connected` for every consumer,
plus the `mountPropagation: Bidirectional` risk. A native systemd mount is
isolated from the k3s lifecycle, restarts in seconds, and consumers only need the
safe `HostToContainer` propagation. JuiceFS has **no master**: every mount talks
directly to Valkey + RustFS, so nothing is routed through a coordinator.

## Components

| Component       | Where                 | What                                                                   |
|-----------------|-----------------------|------------------------------------------------------------------------|
| **Valkey**      | metadata node         | metadata engine (apt `valkey-server`, AOF + `noeviction` + auth)       |
| **format**      | metadata node         | one-time idempotent `juicefs format` (guarded by `juicefs status`)     |
| **mount**       | `juicefs_mount_group` | native systemd FUSE mount, `Restart=always`, per-node cache            |
| **meta-backup** | backup node           | root cron job: `juicefs dump` → external backup S3 (`rustfs_rc_alias`) |

## Key variables (see `defaults/main.yml`)

- `juicefs_primary_node` — **the placement knob**: which node runs Valkey + the format.
- Object store: `rustfs_server_*` (see `roles/rustfs_server/README.md`); `juicefs_bucket`
  is the bucket JuiceFS formats into.
- `juicefs_storage_address`: Valkey address each mount resolves per client; the QSFP IP
  of the node (spark → 200GbE) or its k3s VLAN IP (master).
- `juicefs_s3_address`: S3 address fixed into the metadata at `juicefs format` time (like
  the root keys); changing it later needs `juicefs config --bucket/--access-key/--secret-key`.
- `juicefs_cache_dir` / `juicefs_cache_size_mib` — per-node NVMe cache (MiB).
- `juicefs_valkey_password` is a **dummy here** — the real value lives in
  `group_vars/all/vault/juicefs.sops.yml`.

## Deploy

```bash
ansible-playbook storage.yml                      # full
ansible-playbook storage.yml --tags juicefs_mount # mounts only
ansible-playbook rustfs_server.yml                # RustFS standalone
```

Order: `rustfs_server` role first, then binary → backend → format → mount → backup.
The `rc` CLI (rustfs-cli, `roles/k3sserver/tasks/rustfs_client.yml`) must be present on the
RustFS primary for bucket creation and extra users.

`juicefs_backend_teardown` (Valkey on hosts other than `juicefs_primary_node`) and
`rustfs_server_teardown` (RustFS on hosts outside the current RustFS nodes) are independent;
decommissioning a former combined storage node needs both:
`ansible-playbook storage.yml -e juicefs_backend_teardown=true -e rustfs_server_teardown=true --limit <former-node>`.

## Monitoring (gated by `juicefs_enabled` AND `juicefs_metrics_enabled`)

`juicefs_metrics_enabled` (pre-enabled) selects WHETHER the layer is monitored;
every consumer additionally couples to the `juicefs_enabled` master switch, so
nothing scrapes, alerts, or opens ports until the layer is actually deployed.
Deploy day is one flag flip (`juicefs_enabled: true`), then:
`storage.yml → common.yml --tags iptables → k8s_infra.yml --tags prometheus,grafana`:

- **Clients**: `--metrics <k3s_node_ip>:9567` on every FUSE mount unit (native
  `juicefs_*` Prometheus metrics; flipping the gate restarts the mounts once).
- **Valkey**: apt `prometheus-redis-exporter` on the storage node (`:9121`,
  reuses the requirepass; redis_exporter supports Valkey upstream).
- **RustFS**: has **no native `/metrics`** (OTel push only, rustfs#3154) — its
  health is monitored client-side via `juicefs_object_request_errors` (alert
  `JuiceFSObjectErrors`) and the object-request panels.
- **k8s_infra**: scrape jobs `juicefs`/`juicefs-valkey` + `juicefs-alerts`
  rules (`monitoring/prometheus.yml`), Grafana dashboards "JuiceFS Dashboard"
  (grafana.com 20794) and "Valkey (JuiceFS Metadata)" (763) via
  `download-dashboards.sh`; iptables opens both ports from `k3snodes` only.

## Consumer wiring (separate change, not done here)

Pods consume the shared FS by mounting the host path with one-way propagation:

```yaml
volumes:
  - name: jfs
    hostPath: { path: /mnt/jfs, type: Directory }
containers:
  - volumeMounts:
      - name: jfs
        mountPath: /mnt/jfs
        mountPropagation: HostToContainer
    env:
      - { name: HF_HOME, value: /mnt/jfs/hf }
```

After a model download, warm the per-node cache (runs per node):
`juicefs warmup /mnt/jfs/hf/<model-path>`.

## Acceptance

1. `mountpoint -q /mnt/jfs` true on every mount node.
2. A file written on node A under `/mnt/jfs/test` is readable on node B.
3. New objects appear in the RustFS `juicefs` bucket after writes.
4. `juicefs warmup` fills the local cache; a re-read produces no backend traffic.
5. `juicefs dump` runs and the backup lands in the external backup bucket.
