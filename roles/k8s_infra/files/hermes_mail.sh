#!/usr/bin/env bash
# hermes-mail — open the Hermes email-gateway IMAP account in neomutt, fully
# preconfigured from the gateway's own credentials. Purpose: hands-on debugging
# of the email gateway (watch the INBOX -> Hermes_Working -> Hermes_Done moves,
# \Seen flags, and the Sent APPENDs) without copying creds into files or shell
# history. Config: /opt/maildebug/hermes_neomuttrc (see that file's header).
#
# Deployed as a subPath mount at /usr/local/bin/hermes-mail in the hermes
# (dashboard) container AND the hermes-gateway sidecar, gated on email.enabled.
# Just `kubectl exec ... -- hermes-mail` (default container is `hermes`; use
# `-c hermes-gateway` to debug from the exact process that runs the adapter —
# both share the same /opt/data/.env).
set -euo pipefail

ENV_FILE="${HERMES_MAIL_ENV_FILE:-/opt/data/.env}"
RC_FILE="${HERMES_MAIL_RC:-/opt/maildebug/hermes_neomuttrc}"

[ -r "$ENV_FILE" ] || { echo "hermes-mail: $ENV_FILE not readable — no gateway env here?" >&2; exit 1; }

# Pull ONLY the EMAIL_* keys out of .env into this process env. We deliberately
# do NOT `source` the file (arbitrary lines must not execute). Values can hold
# '=', so split on the first one only.
while IFS= read -r line; do
  case "$line" in
    EMAIL_*=*) export "${line%%=*}=${line#*=}" ;;
  esac
done < "$ENV_FILE"

: "${EMAIL_ADDRESS:?hermes-mail: EMAIL_ADDRESS missing — is the email gateway configured for this user?}"
: "${EMAIL_PASSWORD:?hermes-mail: EMAIL_PASSWORD missing}"
: "${EMAIL_IMAP_HOST:?hermes-mail: EMAIL_IMAP_HOST missing}"
export EMAIL_IMAP_PORT="${EMAIL_IMAP_PORT:-143}"

# Composed for the rc (keeps its backticks to one var each). Folder names mirror
# the adapter's defaults (see hermes_email_gateway_patched.py / config.yaml).
export HM_FOLDER="imap://${EMAIL_IMAP_HOST}:${EMAIL_IMAP_PORT}/"
export HM_WORKING="${EMAIL_WORKING_FOLDER:-Hermes_Working}"
export HM_DONE="${EMAIL_DONE_FOLDER:-Hermes_Done}"
export HM_SENT="${EMAIL_SENT_FOLDER:-Sent}"

# neomutt isn't in the base image; install on first use. exec into the pod is
# root and the pod has egress, so apt works. This is ephemeral (container
# rootfs) → it repeats after a pod restart, which is fine for an on-demand debug
# tool. ca-certificates is needed for STARTTLS peer verification.
if ! command -v neomutt >/dev/null 2>&1; then
  echo "hermes-mail: installing neomutt (one-time for this pod) ..." >&2
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends neomutt ca-certificates >/dev/null
fi

# neomutt writes its header cache / history under $HOME; /opt/data is the
# writable NFS home (a bare `kubectl exec` shell may not have HOME set).
export HOME="${HOME:-/opt/data}"

exec neomutt -F "$RC_FILE" "$@"
