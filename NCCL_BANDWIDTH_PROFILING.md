# NCCL Bandwidth Profiling on DGX Spark (GB10) Clusters

A portable recipe for measuring the **real** NCCL all-reduce bandwidth that a
multi-node SGLang deployment actually gets — including the small-message region
that TP decode depends on. Uses `torch.distributed`, **no MPI / no `mpirun`**.

This is the "how to measure" companion to `nccl-transport-test-plan.md` (which is
the cluster-specific RoCE-vs-socket debugging log).

---

## Why this matters: one cable, two PCIe rails

The DGX Spark / GB10 (and the MSI EdgeXpert, Dell, ASUS Ascent, … re-spins of the
same board) attaches its **ConnectX-7** with only **PCIe Gen5 x4 per link**, so the NIC
runs in multi-host mode over **two x4 links** (~100 Gbit/s each). Every QSFP cage
therefore shows up **twice**, once per link. For the cabled cage:

| Rail | PF netdev       | PCI            | RDMA device    | VFs            | Pod NIC |
|------|-----------------|----------------|----------------|----------------|---------|
| 1    | `enp1s0f0np0`   | `0000:01:00.0` | `rocep1s0f0`   | `enp1s0f0vN`   | `net1`  |
| 2    | `enP2p1s0f0np0` | `0002:01:00.0` | `roceP2p1s0f0` | `enP2p1s0f0vN` | `net2`  |

The second netdev is **not a second port**. It is the other half of the PCIe bandwidth
to the same cable. One netdev / one RDMA device tops out at ~**92–98 Gbit/s** although
the cable links at 200G; only RoCE over **both rails** reaches ~**180–190 Gbit/s**.
Hardware background (both cages share the same two x4 links): `docs/DGX Spark QSFP DeepDive EN.md`.

- **Both rails configured and used** (IP + VFs on both PFs, `NCCL_IB_HCA` lists both): ~**180–190 Gbit/s**.
- **Only rail 1:** capped at ~**92–98 Gbit/s**, roughly **half**. Rail 2 still shows `UP`
  with carrier, but without an IPv4 address it has no RoCEv2 GID, so neither perftest nor
  NCCL can use it.

In this repo the switch is `qsfp_dualrail_enabled` (`group_vars/all/main/qsfp.yml`):
`dgx_prepare` (tags `qsfp`, `qsfp_sriov`) puts the rail-2 IP (third octet +
`qsfp_rail2_octet3_offset`, own /24 per rail) and the same VFs on rail 2, and `k8s_dgx`
attaches the matching rail-2 VF as `net2` (NAD `qsfp-r2-vfN`) and lists both RDMA
devices in `NCCL_IB_HCA`.

The half rate is easy to miss: a single-rail setup passes every individual check (links
`ACTIVE`, jumbo ping clean, perftest a "healthy" ~97 Gbit/s). It only shows when the sum
of both rails is compared with the cable rate, see the host baseline below.

A clean standalone `all_reduce_perf` busbw number (the big, asymptotic value) can
look healthy while the **serving path is at half rate**, because:

1. The benchmark may run with a *tuned* NCCL config that the SGLang pod does not inherit
   (e.g. a different `NCCL_IB_HCA`).
2. The asymptotic busbw saturates at 64 MB+ and **hides the small-message region**.

For **tensor parallelism (TP=N)**, every decoded token does an all-reduce across all
nodes, on the critical path, at **every concurrency level**. Per-GPU compute is tiny
(only the active experts), so the **small-message (KB-range, latency-bound) all-reduce
can dominate per-token decode time**. A halved effective bandwidth there shows up as a
roughly **flat throughput factor across all concurrency levels** — the classic symptom.

> **Key point:** profile the **8 KB – 1 MB** region, not just the headline busbw.
> The default sweep in `nccl_bench.py` starts at 1 MB and would miss exactly this.

---

## The tool: `roles/k8s_dgx/files/nccl_bench.py`

A `torch.distributed` all-reduce sweep (algbw + busbw per size, like
`nccl-tests all_reduce_perf`). Reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT` from the environment and uses `cuda:0` (1 GPU per node).

Sweep range is **env-overridable** (defaults preserve the original 1 MB–1 GB sweep,
so `qsfp_nccl_test.yml` is unchanged):

| Env var                         | Default | Purpose                                                        |
|---------------------------------|---------|----------------------------------------------------------------|
| `NCCL_BENCH_MIN_BYTES`          | `1M`    | sweep floor — **set to `8K` for decode profiling**             |
| `NCCL_BENCH_MAX_BYTES`          | `1G`    | sweep ceiling                                                  |
| `NCCL_BENCH_STEP`               | `2`     | size multiplier per step                                       |
| `NCCL_BENCH_WARMUP`             | `5`     | warmup iters per size                                          |
| `NCCL_BENCH_ITERS`              | `20`    | timed iters per size                                           |
| `NCCL_BENCH_SMALL_REGION_BYTES` | `1M`    | sizes ≤ this count as the "small/decode" region in the summary |

(`*_BYTES` vars accept a `K`/`M`/`G` suffix, e.g. `8K`, `512M`.)

The summary prints a **`RATIO:`** line — small-region peak busbw as a fraction of the
overall peak. Below `0.4×` it flags `LOW: check both rails in NCCL_IB_HCA / channels`.

---

## Method A — in-pod, inherits the real serving config (recommended, portable)

Runs the bench **inside the existing SGLang pods**, so it inherits the exact NCCL
config / interfaces / RoCE path the server actually uses. No MPI, no extra cluster.
Works on any Spark cluster (hand this to anyone running multi-node SGLang).

> ⚠️ **Stop the SGLang server process first** — GPUs are exclusive; running the bench
> alongside a live server contends or OOMs. Do this in a maintenance window. On a
> cluster with a `livenessProbe`/restartPolicy that relaunches the server, prefer
> **Method B** so you never touch the serving pods.

```bash
NS=sglang                              # this repo: {{ sglang_namespace }}
KC="kubectl -n $NS"

# 1. Identify the 4 SGLang pods (head + 3 workers) and the head pod IP.
PODS=( $($KC get pods -l app=sglang -o jsonpath='{.items[*].metadata.name}') )   # adjust selector
HEAD=${PODS[0]}
HEAD_IP=$($KC get pod "$HEAD" -o jsonpath='{.status.podIP}')
PORT=29555                             # any free port for the torch rendezvous

# 2. Copy the bench into each pod.
for p in "${PODS[@]}"; do $KC cp roles/k8s_dgx/files/nccl_bench.py "$p":/tmp/nccl_bench.py; done

# 3. CRITICAL — capture the NCCL_* env the *running* server uses, not the pod-spec env.
#    kubectl exec inherits the image/pod-spec env, NOT vars the entrypoint exported
#    at runtime. Pull the real ones from the live process and re-supply them:
$KC exec "$HEAD" -- sh -c 'cat /proc/$(pgrep -f sglang | head -1)/environ | tr "\0" "\n" | grep ^NCCL_'
#    -> feed any NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_NET_GDR_LEVEL / NCCL_SOCKET_IFNAME
#       below if they are not already in the pod spec. With dual-rail NCCL_IB_HCA must name
#       BOTH VFs (e.g. =rocep1s0f0v0,roceP2p1s0f0v0) and the pod must have net2.

# 4. (stop the SGLang server process in each pod here)

# 5. Launch one rank per pod. Decode-focused range (8K-1G) + NCCL init/net debug.
for i in "${!PODS[@]}"; do
  $KC exec "${PODS[$i]}" -- \
    env MASTER_ADDR="$HEAD_IP" MASTER_PORT="$PORT" WORLD_SIZE=4 RANK="$i" LOCAL_RANK=0 \
        NCCL_BENCH_MIN_BYTES=8K NCCL_BENCH_MAX_BYTES=1G \
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
        python /tmp/nccl_bench.py &
done
wait

# 6. Read rank 0's table + RATIO line:
$KC logs "$HEAD"      # (or wherever rank 0 ran)
```

---

## Method B — dedicated test pods (this repo's framework, least disruptive)

If you have this repo's Ansible, the `nccl_test` framework deploys **separate** 4-pod
test sets (host / socket / roce) on **VF7**, so it runs **alongside** SGLang on VF0
without touching the serving pods:

```bash
ansible-playbook k8s_dgx.yml --tags nccl_test                           # all variants
ansible-playbook k8s_dgx.yml --tags nccl_test -e nccl_test_transport=roce
kubectl -n sglang logs nccl-test-roce-rank0
```

See `roles/k8s_dgx/tasks/nccl_test.yml`. (Note: this path uses `nccl_test.py`, not
`nccl_bench.py`, and its own pod spec — extend there separately if you need the
small-message range in that flow.)

---

## Prerequisite: host baseline over both rails (perftest)

Before any NCCL number, measure the raw RoCE rate on the hosts, **both rails at the same
time** (one `ib_write_bw` per rail, own port each). A single `ib_write_bw` on one device
measures one x4 link and can never show the cable rate.

```bash
# server (spark A)
ib_write_bw -d rocep1s0f0   -F --report_gbits -D 10 -s 65536 -p 18515 &
ib_write_bw -d roceP2p1s0f0 -F --report_gbits -D 10 -s 65536 -p 18516 &
# client (spark B), targets = A's rail-1 and rail-2 host IPs
ib_write_bw -d rocep1s0f0   -F --report_gbits -D 10 -s 65536 -p 18515 <A-rail1-ip> &
ib_write_bw -d roceP2p1s0f0 -F --report_gbits -D 10 -s 65536 -p 18516 <A-rail2-ip> &
wait   # add up "BW average[Gb/sec]" of both clients
```

Reference, two nodes on a direct DAC, MTU 9000, 10 s RC:

| Measurement                 | 64 KiB messages | 1 MiB messages |
|-----------------------------|-----------------|----------------|
| rail 1 alone                | 97.5 Gbit/s     | –              |
| rail 2 alone                | 97.4 Gbit/s     | 97.4 Gbit/s    |
| both rails at once (sum)    | 181.2 Gbit/s    | 182.1 Gbit/s   |

Running both rails at once costs each rail a little (~90.5 instead of ~97.5), the sum is
what counts. If rail 2 has no IPv4 address (`ip -br addr show enP2p1s0f0np0`), 0 VFs
(`/sys/class/net/enP2p1s0f0np0/device/sriov_numvfs`) or `roceP2p1s0f0` is not `ACTIVE`
(`rdma link`), dual-rail is not rolled out: fix that first, NCCL tuning cannot recover it.

---

## Prerequisite: verify the jumbo / MTU path end-to-end

Do this **before** chasing NCCL tuning — a 1500-byte MTU break *anywhere* on the path
forces RoCE to fragment, turning each collective into many small frames. That inflates
the small-message, latency-bound all-reduce and mimics the single-rail penalty (and
shows up on the switch as an average packet size capped near ~1500 bytes). Jumbo must be
intact across **all three** layers, then confirmed end-to-end.

The RoCE path MTU (`active_mtu`) is capped at one of {256, 512, 1024, 2048, **4096**}.
With Ethernet jumbo it negotiates **4096**; on a 1500-MTU path it drops to **1024** —
4× more packets per message, 4× the per-packet overhead.

**1 — Switch (MikroTik):** the L3 MTU *and* the L2MTU must clear a 9000-byte frame
plus RoCE/UDP/IP/Eth headers (≈9100). Read the L2MTU with the generic `interface print`
(`ethernet print` does not show it):

```
/interface print where name~"qsfp"
# active (RUNNING) data ports: ACTUAL-MTU 9000, L2MTU >= ~9100  (this cluster: 9000 / 9500)
# the 1500/1584 SLAVE sub-lanes are inactive breakout lanes -- ignore them
```

**2 — VF inside the pod:** VF MTU is **not** inherited from the PF — netplan must set it
explicitly (see CLAUDE.md). Check the Multus secondary NIC:

```
kubectl -n sglang exec <pod> -c sglang -- ip link show net1
kubectl -n sglang exec <pod> -c sglang -- ip link show net2
# expect: mtu 9000 on both (net1 = rail-1 VF enp1s0f0vN, net2 = rail-2 VF enP2p1s0f0vN)
```

**3 — RoCE active_mtu:** `ibv_devinfo` is usually absent inside the pod, so query it on
the host (the pod's VF0 = devices `rocep1s0f0v0` and `roceP2p1s0f0v0`):

```
for d in rocep1s0f0v0 roceP2p1s0f0v0; do
  ssh root@<spark-ip> "ibv_devinfo -d $d | grep -iE 'active_mtu|state:|link_layer'"
done
# expect for both: active_mtu: 4096 (5) | state: PORT_ACTIVE (4) | link_layer: Ethernet
```

**4 — End-to-end proof** (Spark → switch → Spark), DF bit set so a 9000-byte packet must
pass *without* fragmentation:

```
kubectl -n sglang exec <head-pod> -c sglang -- \
  ping -M do -s 8972 -c 3 <other-spark-qsfp-ip>     # 8972 + 28 = 9000; expect 0% loss
# repeat with the other spark's rail-2 IP (net2 subnet)
```

**Reference — a clean path on this cluster (all four nodes):**

| Layer                                    | Check                  | Expected                    |
|------------------------------------------|------------------------|-----------------------------|
| Switch                                   | `ACTUAL-MTU` / `L2MTU` | 9000 / 9500                 |
| VF (`net1`, `net2`)                      | `ip link` mtu          | 9000                        |
| RoCE (`rocep1s0f0v0`, `roceP2p1s0f0v0`)  | `active_mtu`           | 4096, PORT_ACTIVE, Ethernet |
| End-to-end (both rails)                  | DF ping 9000 B         | 0% loss                     |

All green → MTU is ruled out; a switch-side average packet size near ~1500 B is just RoCE
ACK-mixing + small messages in the latency regime, **not** fragmentation. If `active_mtu`
reads **1024**, or the DF ping returns *"frag needed"* / drops, you have found the break —
fix it at that layer and re-measure before profiling NCCL.

---

## The disambiguator — single-node TP=1 control

The cleanest way to separate **network** from **per-node hardware**: run a model that
fits **one** GB10 at **TP=1**. No inter-node collectives at all.

- TP=1 throughput is **also** low (~same factor) → it's **per-node** (power mode,
  memory clock, thermals — *not* the network). Check `nvidia-smi -q -d POWER,CLOCK`
  under load on all nodes: sustained TGP **and memory clock**, not just SM clock.
- TP=1 throughput is **full speed** → the deficit is in the **multi-node NCCL path** →
  Methods A/B above will show where (which ports/channels, small-message busbw).

---

## Interpreting the results

1. **`NCCL ... INFO` lines** (from `NCCL_DEBUG=INFO`): which HCAs NCCL selected (with
   dual-rail: one `rocep1s0f0vN` **and** one `roceP2p1s0f0vN` device), how many channels
   (`nChannels`), and whether it used `NET/IB` (RoCE) or fell back to `NET/Socket`. Diff
   the in-pod selection against a known-good tuned run.
2. **The `RATIO:` line**: small-region peak busbw vs overall peak.
   - Healthy dual-rail: small busbw a solid fraction of peak.
   - `≈ 0.5×` (or the `LOW` flag): the small/decode region is bandwidth-starved —
     consistent with NCCL running on one rail only (one x4 link, the ~92–98 Gbit/s
     ceiling). Check in this order: rail 2 configured on the hosts (host baseline above),
     `net2` present in the pod, `NCCL_IB_HCA` naming both rails, channel count; then
     re-measure.
3. **Absolute busbw**: ~180–190 Gbit/s peak ≈ both rails in use; ~92–98 Gbit/s peak ≈
   single rail; ~2 GB/s ≈ silent **socket fallback** (see `nccl-transport-test-plan.md`
   for the privileged / IB-char-device / GID prerequisites).
4. **Single-rail control run**: `ansible-playbook k8s_dgx.yml --tags qsfp_nccl
   -e qsfp_nccl_ib_hca==rocep1s0f0v7` pins NCCL to rail 1. If the default dual-rail run
   is not clearly above it, NCCL is not using rail 2.
