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

# Patch SGLang 0.5.9 API mismatch: the RPC dispatcher calls func(**kwargs) but
# the scheduler mixin's save_sharded_model(self, params) expects a single dict.
# Rewrite the signature to accept kwargs directly.
MIXIN="/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler_update_weights_mixin.py"
if grep -q 'def save_sharded_model(self.*params)' "$MIXIN" 2>/dev/null; then
  sed -i 's/def save_sharded_model(self: Scheduler, params):/def save_sharded_model(self, path=None, pattern=None, max_size=None):/' "$MIXIN"
  sed -i 's/path=params\["path"\]/path=path/' "$MIXIN"
  sed -i 's/pattern=params\["pattern"\]/pattern=pattern/' "$MIXIN"
  sed -i 's/max_size=params\["max_size"\]/max_size=max_size/' "$MIXIN"
  echo "Patched save_sharded_model mixin for kwargs dispatch"
fi

exec tini -s -- python3 /scripts/save_sharded.py
