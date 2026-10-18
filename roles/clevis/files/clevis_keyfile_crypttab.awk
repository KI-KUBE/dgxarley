# clevis_keyfile_crypttab.awk
# Rewrites ONLY the /etc/crypttab line of mapper m (dracut path of the keyfile fallback,
# see tasks/dracut_crypttab.yml, which passes the program in via lookup('file')).
# Variables (awk -v): m = mapper, want = field 3 to set ("" = reset to none if managed),
# emb / usb = the managed field-3 values (embedded path, USB form), bootv = the boot-partition
# form "/<file>:UUID=<uuid>" and bootp = its prefix "/<file>:UUID=" (any UUID counts as ours, e.g.
# after a reinstall), to = keyfile-timeout value.
# Exit 3 = unmanaged keyfile / malformed line, exit 4 = mapper line missing.
function isnone(v) { return v == "none" || v == "-" || v == "" }
function managed(v) { return v == emb || v == usb || v == bootv || (bootp != "" && index(v, bootp) == 1) }
/^[[:space:]]*#/ || NF == 0 { print; next }
$1 == m {
  found = 1
  if (NF < 2) { print "crypttab: line of " m " has no device field" > "/dev/stderr"; exit 3 }
  f3 = (NF >= 3) ? $3 : "none"
  f4 = (NF >= 4) ? $4 : "none"
  if (!isnone(f3) && !managed(f3)) {
    print "crypttab: " m " has an unmanaged keyfile field (" f3 ") -- refusing to touch it" > "/dev/stderr"
    exit 3
  }
  new3 = want
  if (new3 == "") new3 = isnone(f3) ? f3 : "none"
  n = split(f4, a, ",")
  opts = ""
  for (i = 1; i <= n; i++) {
    if (a[i] == "" || a[i] == "none" || a[i] == "-" || a[i] ~ /^keyfile-timeout=/) continue
    opts = opts (opts == "" ? "" : ",") a[i]
  }
  if (!isnone(new3)) opts = opts (opts == "" ? "" : ",") "keyfile-timeout=" to
  if (opts == "") opts = "none"
  printf "%s %s %s %s\n", $1, $2, new3, opts
  next
}
{ print }
END { if (!found) { print "crypttab: no line for mapper " m > "/dev/stderr"; exit 4 } }
