# MikroTik Uplink dst-nat Throughput Test — CRS310-8G+2S+

**Date:** 2026-10-23
**Router:** CRS310-8G+2S+ (ARM, 2 cores), RouterOS 7.22 (stable)
**Path:** workstation in the upstream LAN → router uplink address (VLAN on `ether1`, 1 Gbps) → dst-nat → k3smaster (2.5 Gbps USB NIC)
**Client:** iperf 3.20 (workstation, same L2 segment as the uplink)
**Server:** iperf 3.16 (k3smaster, `iperf3 -s -D`)
**Firewall:** `mikrotik_firewall.fasttrack_enabled: true` (`fasttrack-connection hw-offload=yes` on established/related)

## Goal

Measure the throughput of the `lan_ingress` port forward (`roles/mikrotik/tasks/uplink.yml`), i.e. routed + dst-nat
traffic from the upstream LAN through the CRS310 to the k3smaster.

## Setup (ad hoc, removed after the test)

All forwarded `lan_ingress` ports are taken (80/443 haproxy, 6443 k3s, 222 → sshd), so a temporary forward on 5201
was used. Its comment does not start with `<prefix>-uplink dstnat`, so the stale-forward cleanup ignores it.

```
# router
/ip/firewall/nat/add chain=dstnat action=dst-nat in-interface=<uplink-vlan-iface> dst-address-type=local \
    protocol=tcp dst-port=5201 to-addresses=<k3smaster> comment="tmp iperf"

# k3smaster
iptables -I HTSTUFFIN -p tcp --dport 5201 -j RETURN
iperf3 -s -D

# workstation
iperf3 -c <uplink address> -t 15 [-R] [-P 4]

# router, in parallel to a run
/tool/profile duration=12s
```

The existing `<prefix>-uplink forward dstnat` filter rule (`connection-nat-state=dstnat`) already accepts the
forwarded traffic; no extra filter rule was needed.

## Results

| Direction                   | Streams | Throughput | Retransmits | Router CPU (`total`) |
|-----------------------------|--------:|-----------:|------------:|---------------------:|
| upload (client → k3smaster) |       1 | 927 Mbit/s |         908 |               70–73% |
| download (`-R`)             |       1 | 939 Mbit/s |          39 |               47–49% |
| upload                      |       4 | 938 Mbit/s |      19 293 |                    – |
| download (`-R`)             |       4 | 941 Mbit/s |         470 |                    – |

Profile during the download run (largest entries): `prestera_dx_mac` 15–18%, `kernel` 9–10%, `networking` 5–10%,
`ethernet` 2–6%, `bridging` 2–4%.

## Findings

- dst-nat through the CRS310 reaches TCP line rate of a 1 Gbps link (~940 Mbit/s) in both directions, single stream.
- The limit is the 1 Gbps `ether1` uplink port, not the NAT path; the k3smaster side runs at 2.5 Gbps.
- The router CPU is clearly busy despite `hw-offload=yes` (up to ~73% on upload), so headroom beyond 1 Gbps is
  not established by this test.
- The high retransmit count with 4 upload streams is queue overflow at line rate; it does not reduce throughput.

## Open

- Repeat with a client on a 2.5 Gbps port in the upstream segment to find the actual NAT ceiling of the CRS310.
