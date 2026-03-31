# MikroTik CRS812-8DS-2DQ-2DDQ-RM — Gotchas & Cheatsheet

Switch chip: **Marvell 98DX7335** (Prestera Aldrin3) | RouterOS 7.22.1 | Firmware 7.19.5

## Port Overview

| RouterOS Interface | Type | Lanes | Max Speed | Cable / Connection |
|---|---|---|---|---|
| `ether1` | RJ45 10GbE | — | 10G | Uplink → CRS310-8G-2S |
| `qsfp56-1-1` | QSFP56 | 4 (1-4) | 200G | spark3 (Amphenol NJAAKK-N911 DAC) |
| `qsfp56-2-1` | QSFP56 | 4 (1-4) | 200G | spark4 (prepared) |
| `qsfp56-dd-1-1` | QSFP-DD | 4 (1-4) | 200G* | spark1 via breakout |
| `qsfp56-dd-1-5` | QSFP-DD | 4 (5-8) | 200G* | spark2 via breakout |
| `qsfp56-dd-2-1` | QSFP-DD | 8 (1-8) | 400G | spare (unused) |

*\*Breakout cable: NADDOD Q2Q56-400G-CU2 (QSFP-DD → 2x QSFP56, 2m DAC)*

## QSFP-DD Sub-Port Naming

Each QSFP-DD port exposes **8 sub-interfaces** (8 SerDes lanes, 50G PAM4 each):

```
qsfp56-dd-X-1  ← primary, lanes 1-4 (up to 400G when all 8 lanes aggregate)
qsfp56-dd-X-2  ← lane 2 only (50G)
qsfp56-dd-X-3  ← lanes 3-4 (up to 100G)
qsfp56-dd-X-4  ← lane 4 only (50G)
qsfp56-dd-X-5  ← lanes 5-8 (up to 200G) ← second port in 2x200G breakout
qsfp56-dd-X-6  ← lane 6 only (50G)
qsfp56-dd-X-7  ← lanes 7-8 (up to 100G)
qsfp56-dd-X-8  ← lane 8 only (50G)
```

With a **2x200G breakout cable**, lanes are split:
- **Lanes 1-4** → `qsfp56-dd-X-1` → first QSFP56 connector (e.g. spark1)
- **Lanes 5-8** → `qsfp56-dd-X-5` → second QSFP56 connector (e.g. spark2)

## Gotcha #1: QSFP-DD breakout requires `advertise` restriction

### Symptom

Breakout cable plugged in but no link:
- `qsfp56-dd-1-1`: `auto-negotiation: failed`
- `qsfp56-dd-1-5`: `advertising: (empty)`
- Sparks: `Link detected: no (Autoneg, No partner detected)`

### Root cause

The CRS812 treats QSFP-DD ports as **1x400G** by default (8-lane aggregation). It advertises `400G-baseCR8` and attempts to use all 8 lanes as a single link. Since each Spark only sees 4 lanes through the breakout cable, negotiation fails. Sub-port `-5` (lanes 5-8) remains completely inactive.

**The cable EEPROM does not help** — the NADDOD Q2Q56-400G-CU2 identifies as `sfp-type: QSFPDD` with `400G-baseCR8` in `sfp-supported`. The switch does NOT auto-detect that it is a breakout cable.

### Fix

Restrict `advertise` on the primary sub-port (`dd-X-1`) to max 200G — remove 400G speeds:

```
/interface/ethernet/set qsfp56-dd-1-1 advertise=10M-baseT-half,10M-baseT-full,\
100M-baseT-half,100M-baseT-full,1G-baseT-half,1G-baseT-full,1G-baseX,\
2.5G-baseT,2.5G-baseX,5G-baseT,10G-baseT,10G-baseSR-LR,10G-baseCR,\
40G-baseSR4-LR4,40G-baseCR4,25G-baseSR-LR,25G-baseCR,50G-baseSR2-LR2,\
50G-baseCR2,100G-baseSR4-LR4,100G-baseCR4,50G-baseSR-LR,50G-baseCR,\
100G-baseSR2-LR2,100G-baseCR2,200G-baseSR4-LR4,200G-baseCR4
```

Once `dd-1-1` is restricted to max 200G, it only claims lanes 1-4. **`dd-1-5` auto-activates** as a second 200G port (lanes 5-8).

After that, `dd-1-5` still needs:
- `l2mtu=9500` (jumbo frames / RoCE)
- QoS: `pfc=pfc-tc3`, `trust-l3=keep` (PFC for RoCE)

## Gotcha #2: `egress-rate-queue` hardware cap at 100G

The QoS egress rate shaper (`egress-rate-queueN`) on the 98DX7335 accepts at most **100.0Gbps**:

```
/interface/ethernet/switch/qos/port/set qsfp56-dd-1-1 egress-rate-queue3=100.0Gbps
# → OK

/interface/ethernet/switch/qos/port/set qsfp56-dd-1-1 egress-rate-queue3=100.1Gbps
# → failure: max bit rate is 100G

/interface/ethernet/switch/qos/port/set qsfp56-dd-1-1 egress-rate-queue3=200.0Gbps
# → failure: max bit rate is 100G
```

For ports with >=200G link speed this means: the shaper cannot cover the full bandwidth and **caps RoCE traffic at 100G** (50% of a 200G link).

**Solution:** Set `egress-rate-queue3=0` (= no shaping / unlimited). PFC continues to work — pause-frame calculations use the PHY link speed, not the shaper value.

## Gotcha #3: Firmware vs. RouterOS version

```
> /system routerboard print
  current-firmware: 7.19.5
  upgrade-firmware: 7.22.1
```

RouterOS can be newer than the firmware. Some features (e.g. `lossless-traffic-class`, `lossless-buffers`) require firmware updates. After a RouterOS upgrade, upgrade firmware if needed:

```
/system routerboard upgrade
# then reboot
```

---

## Cheatsheet: Diagnostic Commands

### Link Status & Speed

```bash
# All running interfaces (R = RUNNING)
/interface print where running=yes

# Detailed port status (speed, FEC, duplex)
/interface ethernet monitor qsfp56-dd-1-1 once
/interface ethernet monitor qsfp56-dd-1-5 once
/interface ethernet monitor qsfp56-1-1 once

# All ethernet interfaces with flags
/interface ethernet print
```

### Cable / Transceiver Detection

```bash
# SFP module info (vendor, type, serial, cable length)
/interface ethernet monitor qsfp56-dd-1-1 once
# Key fields:
#   sfp-module-present: yes/no
#   sfp-type: QSFPDD / QSFP28/QSFP56
#   sfp-cmis-module-state: ready / not-ready
#   sfp-connector-type: no-separable-connector (= DAC)
#   sfp-vendor-name: NADDOD / Amphenol / ...
#   sfp-vendor-part-number: Q2Q56-400G-CU2
#   sfp-link-length-cable-assembly: 2m
#   sfp-encoding: pam4 / nrz

# Check if cable is detected on a sub-port
/interface ethernet monitor qsfp56-dd-1-5 once
# If sfp-module-present=yes + same serial as dd-1-1
# → same physical cable, both lane groups see the transceiver
```

### Auto-Negotiation & Advertise

```bash
# Configured vs. actually advertised speeds
/interface ethernet print detail where name=qsfp56-dd-1-1
# advertise=...      ← configured speeds
# (monitor shows: advertising=...  ← actually on the wire)

# Check current advertising (key field!)
/interface ethernet monitor qsfp56-dd-1-1 once
# advertising: ...          ← what the switch sends
# link-partner-advertising: ← what the remote end sends
# auto-negotiation: done / failed

# Change advertise (remove 400G for breakout)
/interface ethernet set qsfp56-dd-1-1 advertise=<speed-list-without-400G>
```

### Bridge & L2

```bash
# Bridge port status (H = HW-OFFLOAD, I = INACTIVE)
/interface bridge port print detail

# MAC table (which MAC learned on which port)
/interface bridge host print where bridge=bridge

# Bridge info
/interface bridge print detail
```

### QoS / RoCE / PFC

```bash
# QoS port config (PFC, trust, egress-rate)
/interface ethernet switch qos port print detail

# Only QSFP ports with PFC enabled
/interface ethernet switch qos port print where pfc!=disabled

# QoS profiles (roce=tc3, cnp=tc6)
/interface ethernet switch qos profile print

# DSCP mapping
/interface ethernet switch qos map ip print

# Tx queue scheduling
/interface ethernet switch qos tx-manager queue print

# PFC profiles
/interface ethernet switch qos priority-flow-control print

# Global QoS settings
/interface ethernet switch qos settings print

# Check L2MTU (must be 9500 for RoCE/jumbo)
/interface ethernet print proplist=name,l2mtu where name~"qsfp"
```

### Switch Chip Info

```bash
# Chip type and capabilities
/interface ethernet switch print detail
# → type=Marvell-98DX7335

# Switch port status (R = RUNNING)
/interface ethernet switch port print

# System info
/system resource print
/system routerboard print
```

### Ansible Playbook

```bash
# Full switch run
ansible-playbook mikrotik.yml -l mikrotik-CRS812-8DS-2DQ-2DDQ-RM

# RoCE/QoS tags only
ansible-playbook mikrotik.yml -l mikrotik-CRS812-8DS-2DQ-2DDQ-RM --tags mikrotik_roce

# Bridge ports only
ansible-playbook mikrotik.yml -l mikrotik-CRS812-8DS-2DQ-2DDQ-RM --tags mikrotik_ports
```

### Spark Side (DGX Spark)

```bash
# Link status on ConnectX-7 (QSFP interface)
ssh root@<spark-ip> 'ethtool enp1s0f0np0'
# Key fields:
#   Speed: 200000Mb/s / Unknown!
#   Link detected: yes / no
#   Port: Direct Attach Copper
#   Advertised FEC modes: ...

# IP address on QSFP
ssh root@<spark-ip> 'ip addr show enp1s0f0np0'

# L2 connectivity test (IPv6 link-local multicast through switch)
ssh root@<spark1-ip> 'ping -c 3 -I enp1s0f0np0 ff02::1%enp1s0f0np0'
```
