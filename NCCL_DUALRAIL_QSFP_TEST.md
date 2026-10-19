# NCCL Dual-Rail QSFP Test (interim report)

Status: **in progress** (2026-10-19). Evaluates the pending dual-rail change
(`qsfp_dualrail_enabled`, rail-2 IPv4 in its own /24, rail-2 VFs + NADs, `NCCL_IB_HCA` with both rails)
before rolling it out.

## Background

The GB10 gives the ConnectX-7 only PCIe Gen5 x4 per link, so every QSFP cage shows up twice:

| Rail | Netdev          | PCI          | RDMA device    |
| ---- | --------------- | ------------ | -------------- |
| 1    | `enp1s0f0np0`   | 0000:01:00.0 | `rocep1s0f0`   |
| 2    | `enP2p1s0f0np0` | 0002:01:00.0 | `roceP2p1s0f0` |

Both PCIe links trained at 32GT/s x4, both RDMA ports `ACTIVE 200 Gb/sec`.
Status quo before the change: only rail 1 has IPv4 (10.10.10.x) and 8 VFs; rail 2 is link-up with
IPv6 link-local only, 0 VFs, no IPv4 RoCEv2 GID (GID index 3 empty).

## Method

Host level only, no pods, no config change:

- Raw RDMA: `ib_write_bw -s 1048576 -q 4 -D 10 --report_gbits`. Rail 1 via GID 3 (IPv4),
  rail 2 via GID 1 (RoCEv2 link-local IPv6), because rail 2 has no IPv4 yet.
- NCCL: `torch.distributed` all-reduce (bf16) from `/root/pytorch-env` (torch 2.10.0+cu130,
  **NCCL 2.28.9, built-in IB transport**, not the HPC-X IBext plugin SGLang uses). Bootstrap over
  rail 1 (`NCCL_SOCKET_IFNAME=enp1s0f0np0`), `NCCL_IB_GID_INDEX=1`, `NCCL_IB_MERGE_NICS=0`,
  GDR 0 (host memory), `[RO]` active. busbw = algbw * 2(n-1)/n.

## Results

### Raw RDMA (spark1 → spark2)

| Config                                   | Gbit/s                  |
| ---------------------------------------- | ----------------------- |
| rail 1 alone                             | 111.8                   |
| rail 2 alone                             | 111.9                   |
| both rails concurrently                  | 97.8 + 97.8 = **195.6** |
| 4-node ring pattern, both rails, 8 flows | 97.5 + 97.5 per node    |

All 12 directed pairs on rail 2 and all tested cross-rail pairs reach 111.85 Gbit/s.
After the runs: no `out_of_sequence`, `packet_seq_err`, retransmits or CNPs; only single-digit
`rx_prio0_buf_discard`. The fabric carries dual-rail cleanly.

### NCCL all-reduce, status quo (rail 1 only)

| Size   | n=4 time | n=4 busbw (Gbit/s) |
| ------ | -------- | ------------------ |
| 32 KB  | ~50 us   | 7.7                |
| 256 KB | 116 us   | 27.1               |
| 1 MB   | 270 us   | 46.7               |
| 8 MB   | 941 us   | 107.0              |
| 64 MB  | 7.4 ms   | 109.2              |
| 256 MB | 29.0 ms  | 111.1              |
| 1 GB   | 115.4 ms | 111.7              |

2 nodes, 1 GB: 111.6 Gbit/s. Rail 2 alone at n=4 gives the same numbers (111.6 @ 1 GB).

### NCCL all-reduce, dual-rail

| Nodes | Result                                                                                                                                                                                                 |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2     | 157 (64 MB) / 164 (256 MB) / **185 Gbit/s (1 GB)**, +66 % vs single rail; one later repeat hung, but it overlapped with a concurrent sshd/fail2ban restart on all sparks (03:18), so it is not counted |
| 3     | **hangs**                                                                                                                                                                                              |
| 4     | **hangs** (reproducible)                                                                                                                                                                               |

Hang pattern at n=3/4: all 16 ring channels connect rail-matched (IB/0↔IB/0, IB/1↔IB/1,
"Connected all rings"), the 32 KB (tree) op completes, then the first ring transfer never finishes.
No NCCL warning, no IB error, even with short `NCCL_IB_TIMEOUT`. Not fixed by any of:
`NCCL_IB_PCI_RELAXED_ORDERING=0`, `NCCL_IB_MERGE_NICS=1`, `NCCL_CROSS_NIC=1`,
`NCCL_MAX_NCHANNELS=8` (gets further, hangs at 8 MB), `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`,
`NCCL_ALGO=Tree`, `NCCL_IB_TIMEOUT=14 NCCL_IB_RETRY_CNT=3`, `NCCL_IB_QPS_PER_CONNECTION=2`.
Not CMA (`cma=128M`, CmaFree 85 MB), no kernel errors.

Small messages (decode-path TP all-reduce): dual-rail does not change latency (32 KB ~50 us either
way). The gain is limited to large transfers (prefill, weight/KV movement).

## Online research

- NVIDIA playbooks ("Connect Multiple DGX Spark through a Switch", NCCL stacked-sparks) give
  **both rails an IPv4 address, each rail in its own subnet**; same subnet on both is called out as
  causing conflicts. The pending change (rail 2 = rail-1 IP with third octet +1, own /24) matches.
- `NCCL_IB_ADDR_FAMILY` defaults to AF_INET since 2.21; the link-local GID used in this test is the
  only deviation from all documented working setups.
- **NVIDIA/nccl#2334**: 4× GB10, MikroTik switch, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`, silent
  collective hang; affected 2.28.9 (torch bundle), 2.30.4, 2.30.7; open. Reporter's untested
  mitigation: `NCCL_IB_QPS_PER_CONNECTION=1` + `NCCL_CROSS_NIC=0`.
  Maintainers point at **NVIDIA/nccl#2393** (acquire fence in `ncclIbIsend`, stale `nreqs` read on
  weakly ordered AArch64 → proxy spins forever), merged to dev 2026-10-01, not in a release yet.
- DGX Spark release notes: nothing on OTA 2607.1 / SoC FW 0x03000008 relating to CX7, PCIe or NCCL.

## Re-run after partial firmware update (2026-10-19 ~03:40)

spark1 + spark3 on EC 0x02000007 / SoC 0x03000008 (OTA2607.1), spark2 + spark4 still on
EC 0x02000006 / SoC 0x03000007.

| Run                                                        | Result                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| n=4 rail 1 only                                            | unchanged: 106.6 (8 MB) / 111.5 Gbit/s (1 GB); 32 KB 64 us, 256 KB 169 us            |
| n=4 dual                                                   | **hangs** after 32 KB                                                                |
| n=4 dual + `NCCL_IB_QPS_PER_CONNECTION=1 NCCL_CROSS_NIC=0` | **hangs** after 32 KB (#2334 mitigation does not help)                               |
| n=2 dual (spark1+spark2)                                   | **hangs** after 32 KB (worked once before the update, so 2 nodes is flaky, not safe) |

## Re-run with all nodes on OTA2607.1 firmware (2026-10-19 ~04:10)

spark1-4 on EC 0x02000007 / SoC 0x03000008.

| Run                                                        | Result                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| n=4 rail 1 only                                            | unchanged: 106.9 (8 MB) / 111.7 Gbit/s (1 GB); 32 KB 61 us, 256 KB 142 us |
| n=4 dual                                                   | **hangs** after 32 KB                                                     |
| n=4 dual + `NCCL_IB_QPS_PER_CONNECTION=1 NCCL_CROSS_NIC=0` | **hangs** after 32 KB                                                     |
| n=2 dual (2 runs)                                          | **hangs** after 32 KB, both runs                                          |

The firmware update neither fixes nor changes the hang; single-rail performance is identical.

## Dual-rail rolled out, production path (2026-10-19 ~04:40)

Applied: `dgx.yml --tags qsfp,qsfp_sriov -l spark1..4` (rail-2 PF 10.10.11.x/24, 8 rail-2 VFs with
10.10.101.x, IPv4 RoCEv2 GID at index 3 on PF + VFs) and `k8s_dgx.yml --tags multus` (16
`qsfp-r2-vfN` NADs in sglang + vllm).

Host-level torch NCCL 2.28.9 with IPv4 GID 3 still **hangs** (n=4 and n=2), so the link-local GID was
not the cause.

Pod test (`k8s_dgx.yml --tags nccl_test -e nccl_test_transport=roce`), image
`xomoxcc/dgx-spark-sglang:0.5.21-sm121`, VF 7 on both rails, `NCCL_IB_GID_INDEX=3`, 4 nodes:

- NCCL **2.30.7+cuda13.2**, `NET/Plugin: Could not find: libnccl-net.so` → NCCL's **built-in
  NET/IB** transport, no HPC-X/IBext plugin.
- Dual: `NET/IB : Using [0]rocep1s0f0v7 [1]roceP2p1s0f0v7`, single (`-e qsfp_dualrail_enabled=false`):
  `[0]rocep1s0f0v7` only.

| Size           | single latency | single BW                   | dual latency | dual BW                            |
| -------------- | -------------- | --------------------------- | ------------ | ---------------------------------- |
| 4 KB           | 54 us          | 0.11 GB/s                   | 55 us        | 0.11 GB/s                          |
| 64 KB          | 69 us          | 1.42 GB/s                   | 73 us        | 1.35 GB/s                          |
| 256 KB         | 212 us         | 1.85 GB/s                   | 220 us       | 1.79 GB/s                          |
| 4 MB           | 849 us         | 7.41 GB/s                   | 784 us       | 8.03 GB/s                          |
| 16 MB          | 2076 us        | 12.12 GB/s                  | 1209 us      | 20.82 GB/s                         |
| 64 MB          | 8165 us        | 12.33 GB/s                  | 4653 us      | 21.63 GB/s                         |
| 256 MB         | 29728 us       | 13.54 GB/s                  | 17342 us     | 23.22 GB/s                         |
| **peak busbw** |                | **13.54 GB/s (108 Gbit/s)** |              | **23.22 GB/s (186 Gbit/s), +71 %** |

Dual-rail repeated 3× in total: 23.22 / 23.19 / 23.17 GB/s, no hang.

Conclusion: in the production path (SGLang image, NCCL 2.30.7, SR-IOV VFs, IPv4 GIDs) dual-rail is
stable at 4 nodes and gives +71 % bus bandwidth from 16 MB upward. Up to 256 KB (decode-path TP
all-reduce) latency is unchanged. Which difference to the hanging host setup matters (NCCL 2.28.9 vs
2.30.7, PF vs VF) is not isolated.

## SGLang GSM8K with dual-rail (2026-10-19 02:41-02:55 UTC)

`ansible-playbook k8s_dgx.yml --tags sglang`, default profile `nvidia/Qwen3.8-Flash-Next-NVFP4`, 4 nodes
TP4/EP4, image 0.5.21-sm121. Head log: NCCL 2.30.7, `NET/IB : Using [0]rocep1s0f0v0 [1]roceP2p1s0f0v0`.
Same method as the 2026-10-12 engine-log rerun in
`TESTLOGS/sglang_nn4_tp4_ep4/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_4n.md`
(`sglang-gsm8k run`, n4/n8/n16/n32 with 32/64/128/256 questions, max_tokens 8192, temperature 0, 20 s
pause; highest `gen throughput` of the head-log `Decode batch` lines per level window, log interval 40).
Baseline = row "adaptive, bfloat16 SSM, max_running 32 (= profile)" of 2026-10-12 (single-rail).

| Level | 2026-10-12 single-rail | 2026-10-19 dual-rail | delta | GSM8K                 |
| ----- | ---------------------: | -------------------: | ----: | --------------------- |
| n4    |                    231 |                  240 |  +4 % | 31/32                 |
| n8    |                    331 |                  340 |  +3 % | 63/64                 |
| n16   |                    476 |                  461 |  -3 % | 126/128               |
| n32   |                    646 |                  623 |  -4 % | 251/256 (2 truncated) |

Head-log POSTs 32/64/128/256 as expected, accept len mean 2.96 (2026-10-12: 2.88), 0 eager decode
batches. All deltas are within the documented day-to-day spread (differences below ~15 % are not a
ranking). Dual-rail gives no measurable decode throughput gain on this workload; GSM8K decode is
bound by small TP all-reduces whose latency dual-rail does not change. A first bench attempt with a
wrong `--url` (all requests 405) was discarded.

## Small-message NCCL tuning (2026-10-19 ~03:05-03:30 UTC)

SGLang scaled to 0; four idle pods (`tail -f /dev/null`, SGLang image 0.5.21-sm121, NCCL 2.30.7, VF 7
on both rails, label `app=nccl-tune`), one `torch.distributed` all-reduce (bf16) per variant via
`kubectl exec`. Base env as SGLang (`NCCL_IB_GID_INDEX=3`, `MERGE_NICS=0`, `NVLS_ENABLE=0`). Per size:
30 fresh correctness checks (exact bf16 sums) + 20 warmup + median of 5 × 100 timed ops (CUDA events).
Latency in us per all-reduce, 4 nodes, dual-rail unless noted; no variant produced a wrong result.

| Variant                            |       16K |       32K |       64K |      128K |      256K |        512K |      1M |
| ---------------------------------- | --------: | --------: | --------: | --------: | --------: | ----------: | ------: |
| default, single-rail               |        48 |        51 |        61 |        87 |       111 |         341 |     900 |
| default                            |     46-52 |     48-50 |     58-61 |     80-86 |   108-111 |         342 | 889-917 |
| `ALGO=Ring`                        |        47 |        50 |        60 |        85 |       108 |         366 |     909 |
| `ALGO=Tree`                        |        64 |        65 |        71 |        87 |       187 |         511 |     827 |
| `PROTO=LL`                         |        47 |        49 |        59 |        80 |       286 |        1187 |    2866 |
| `PROTO=Simple`                     |        54 |        65 |        70 |        95 |       155 |         298 |     721 |
| `PROTO=LL128`                      |        57 |        62 |        70 |        78 |       107 |         346 |    1019 |
| `ALGO=Tree PROTO=LL`               |        47 |        79 |       157 |       585 |       868 |        1267 |    1797 |
| `MIN=MAX_NCHANNELS=1`              |        50 |        57 |        70 |        84 |       106 |         150 |     245 |
| `MIN=MAX_NCHANNELS=2`              |        49 |        51 |        63 |        73 |        94 |         114 |     168 |
| **`MIN=MAX_NCHANNELS=4`**          | **46-48** | **48-50** | **52-60** | **61-71** | **81-83** | **104-109** | **147** |
| `MIN=MAX_NCHANNELS=4`, single-rail |        47 |        50 |        63 |        71 |        84 |         114 |     177 |
| `MIN=MAX_NCHANNELS=8`              |        51 |        49 |        62 |        83 |        83 |         107 |     215 |
| `MIN_NCHANNELS=16`                 |        46 |        48 |        52 |        77 |       107 |         353 |     890 |

Large messages (us; two runs of default to show its spread):

| Variant               |      4M |       16M |        64M |        256M |
| --------------------- | ------: | --------: | ---------: | ----------: |
| default               | 784-796 | 1199-1222 | 7735-12753 | 25301-31280 |
| `MIN=MAX_NCHANNELS=4` |     318 | 1134-1152 |  4387-4414 |       17390 |
| `MIN=MAX_NCHANNELS=6` |     305 |      1135 |       5281 |       18503 |
| `MAX_NCHANNELS=4`     |     319 |      1145 |       4388 |       17390 |
| `MAX_NCHANNELS=8`     |     301 |      1121 |       5468 |       21342 |

- Below 64 KB nothing beats the default (~46-50 us floor, latency-bound).
- The default (16 channels) degrades badly from 256 KB to 1 MB and is noisy on large messages.
  4 channels: 256 KB -25 %, 512 KB -69 %, 1 MB -84 %, 4 MB -60 %, 256 MB ~-30 to -45 %, and stable
  (256 MB = 23.1 GB/s busbw, the dual-rail peak of the `nccl_test` run).
- `ALGO`/`PROTO` overrides bring nothing over the default selection; `LL128` gave no wrong result in
  these runs but also no gain, so it is not worth the ordering risk on AArch64.

### SGLang GSM8K with 4 channels (2026-10-19 03:09-03:21 UTC)

Deployed with `ansible-playbook k8s_dgx.yml --tags sglang -e '{"sglang_nccl_env_extra":
{"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}}'` (no repo change; env verified in the head
container), dual-rail, otherwise identical to the dual-rail run above. Same bench method.

Now the global default: `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` = 4 in `_sglang_nccl_env_roce`
(`roles/k8s_dgx/defaults/main/sglang.yml`); a `--check --diff` run reports no change against the live
ConfigMap.

| Level | 2026-10-12 single-rail | 10-19 dual-rail default | 10-19 dual-rail, 4 channels | vs dual default | GSM8K                 |
| ----- | ---------------------: | ----------------------: | --------------------------: | --------------: | --------------------- |
| n4    |                    231 |                     240 |                     **254** |            +6 % | 31/32                 |
| n8    |                    331 |                     340 |                     **356** |            +5 % | 63/64                 |
| n16   |                    476 |                     461 |                     **504** |            +9 % | 125/128 (1 truncated) |
| n32   |                    646 |                     623 |                     **762** |       **+22 %** | 253/256 (1 truncated) |

POSTs 32/64/128/256 as expected, accept len mean 2.96 (same as default run), 0 eager decode batches.
n32 wall time 4:32 vs 6:05 for the default run. Only the n32 gain is above the ~15 % noise threshold;
n4-n16 move in the same direction but within noise. Consistent with the microbenchmark: the larger the
batch, the larger the all-reduce, the more the 16-channel default loses.

## Next steps

1. ~~Re-run the matrix once spark2 + spark4 are on the new firmware too.~~ Done, no change.
2. ~~Rail-2 IPv4~~ Done (rolled out), did not fix the host-level hang.
3. ~~Redeploy SGLang with dual-rail and measure~~ Done, GSM8K decode throughput unchanged.
   A prefill-heavy load (long prompts) is where a gain would show, if anywhere.
4. Optional: isolate the host-level hang (torch venv with NCCL 2.30.7 via `LD_PRELOAD`, or the #2393
   dev build); NVIDIA/nccl#2334 lists 2.30.7 as affected too, so watch for hangs in production.
