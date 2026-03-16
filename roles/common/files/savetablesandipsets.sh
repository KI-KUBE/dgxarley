#!/bin/bash

set -x

ipset save > /root/ipset.save
# iptables-save > /root/iptables.save
iptables-save | egrep -iv '(kubernetes|flannel|KUBE-|CNI)' > /root/iptables.save
ip6tables-save  > /root/ip6tables.save

