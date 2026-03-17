#!/bin/bash
set -e

# Install ping for ARP priming (not included in sglang image)
apt-get update -qq && apt-get install -y -qq tini iproute2 iputils-ping net-tools >/dev/null 2>&1

# Prime ARP table on the QSFP P2P link before NCCL tries to connect.
if [ "$NODE_RANK" = "0" ]; then
  peer="$QSFP_IP_SPARK2"
else
  peer="$QSFP_IP_SPARK1"
fi
echo "Waiting for QSFP peer ${peer} ..."
until ping -c10 -W1 "$peer" ; do
  sleep 1
done
echo "QSFP peer ${peer} reachable."

exec tini -s -- python3 /scripts/save_sharded.py
