#!/bin/bash

bl=/root/badhosts.txt
if ! [ -e ${bl} ] ; then
  echo ${bl} does not exist... creating empty file...
  touch ${bl}
fi

# calculate performance parameters for the new set
if [ "${RANDOM}" ]; then
	# bash
	tmp_set_name="tmp_${RANDOM}"
else
	# non-bash
        tmp_set_name="tmp_$$"
fi


new_list_size=$(wc -l "${bl}" | awk '{print $1;}' )
hash_size=$(expr $new_list_size / 2)
set_name=manual-blacklist
#echo tmp_set_name: $tmp_set_name

if ! ipset -q list ${set_name} >/dev/null ; then
	ipset create ${set_name} hash:net family inet
fi

# start writing new set file
ipset create ${tmp_set_name} hash:net family inet hashsize ${hash_size} maxelem ${new_list_size}

i=0
while read line ; do
	#echo line: $line
	#echo iptables -I INPUT -s ${line} -p udp -m udp --dport 1:65535 -j REJECT
#	echo ipset add ${set_name} ${line}
	ipset add ${tmp_set_name} ${line}
	if [ $? -eq 0 ] ; then
		i=$(($i+1))
	fi

	#echo iptables -D INPUT -s ${line} -p udp -m udp --dport 1:65535 -j REJECT
	#iptables -D INPUT -s ${line} -p udp -m udp --dport 1:65535 -j REJECT
done < ${bl}

# replace old set with the new, temp one - this guarantees an atomic update
#echo ipset swap ${tmp_set_name} ${set_name}
ipset swap ${tmp_set_name} ${set_name}

# clear old set (now under temp name)
ipset destroy ${tmp_set_name} 

echo added $i entries to $set_name

exit 0