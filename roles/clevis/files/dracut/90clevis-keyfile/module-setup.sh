#!/bin/bash
# Managed by Ansible (clevis role)
#
# dracut module clevis-keyfile: ships a systemd password agent that answers systemd-cryptsetup's
# ask-password requests with a key from the enabled sources (initrd-embedded key, key file on the
# unencrypted /boot partition, USB stick) while the console prompt (and clevis-dracut/tang) keeps
# working in parallel. crypttab stays untouched. Configuration: /etc/clevis-keyfile/keyfile.conf.

# Only part of the initrd when the role deployed its config (hostonly-friendly).
check() {
    [[ -r /etc/clevis-keyfile/keyfile.conf ]] || return 1
    return 0
}

depends() {
    echo systemd systemd-ask-password crypt
    return 0
}

installkernel() {
    # USB mass storage and the filesystems of stick / boot partition (hostonly would omit them)
    instmods usb_storage uas sd_mod vfat exfat nls_cp437 nls_iso8859_1 ext4
}

install() {
    local conf=/etc/clevis-keyfile/keyfile.conf
    local EMBED=0 BOOT=0 USB=0 KEYFILE=/etc/clevis-keyfile/fallback.key
    local BOOT_DEV='' BOOT_FILE='' USB_LABEL='' USB_FILE=''
    local unitdir=${systemdsystemunitdir:-/usr/lib/systemd/system}
    # an explicit add_dracutmodules skips check()
    if [[ ! -r $conf ]]; then
        dwarn "clevis-keyfile: $conf not readable, module not installed"
        return 0
    fi
    # shellcheck disable=SC1090
    . "$conf"

    inst_multiple cryptsetup mount umount blkid findmnt mktemp cmp head cat sleep cksum grep rm mkdir
    # the systemd-ask-password module normally ships these; belt and braces
    inst_multiple -o systemd-reply-password systemd-tty-ask-password-agent \
        /usr/lib/systemd/systemd-reply-password

    inst_simple "$moddir/clevis-keyfile-agent.sh" /usr/libexec/clevis-keyfile-agent
    chmod 0755 "${initdir}/usr/libexec/clevis-keyfile-agent"
    inst_simple "$conf" /etc/clevis-keyfile/keyfile.conf

    if [[ ${EMBED:-0} == 1 ]]; then
        if [[ -r $KEYFILE ]]; then
            inst_simple "$KEYFILE" /etc/clevis-keyfile/fallback.key
            chmod 0400 "${initdir}/etc/clevis-keyfile/fallback.key"
        else
            dwarn "clevis-keyfile: $KEYFILE not readable, NOT embedding the fallback key"
        fi
    fi

    # Wants= only (cryptsetup.target.wants): the agent is never a hard dependency of unlocking
    inst_simple "$moddir/clevis-keyfile-agent.service" "$unitdir/clevis-keyfile-agent.service"
    mkdir -p "${initdir}${unitdir}/cryptsetup.target.wants"
    ln_r "$unitdir/clevis-keyfile-agent.service" "$unitdir/cryptsetup.target.wants/clevis-keyfile-agent.service"
}
