"""
Email platform adapter for the Hermes gateway.

Allows users to interact with Hermes by sending emails.
Uses IMAP to receive and SMTP to send messages.

Environment variables:
    EMAIL_IMAP_HOST     — IMAP server host (e.g., imap.gmail.com)
    EMAIL_IMAP_PORT     — IMAP server port (default: 993)
    EMAIL_SMTP_HOST     — SMTP server host (e.g., smtp.gmail.com)
    EMAIL_SMTP_PORT     — SMTP server port (default: 587)
    EMAIL_ADDRESS       — Email address for the agent
    EMAIL_PASSWORD      — Email password or app-specific password
    EMAIL_POLL_INTERVAL — Seconds between mailbox checks (default: 15)
    EMAIL_ALLOWED_USERS — Comma-separated list of allowed sender addresses

------------------------------------------------------------------------------
LOCAL PATCH (dgxarley) — synced to upstream tag v2026.8.16
(plugins/platforms/email/adapter.py, blob 704524e4, 62120 bytes). Current for
the pinned image (hermes.image_tag v2026.8.16).

Re-synced 2026-08-17 (v2026.8.13 -> v2026.8.16, v0.20.2). Small but NOT
byte-identical: exactly one upstream commit touched this file, 480342232a
("fix(gateway): close leaked poller sockets in weixin/email adapters",
#79889), and both of its call sites land on our anchors.

  1. NEW module-level _close_imap(imap) (upstream, verbatim, placed where
     upstream put it: right after SMTP_CONNECT_TIMEOUT). It calls logout()
     and, on ANY exception, chases it with shutdown(). Rationale (upstream's
     own docstring, kept): IMAP4.logout() only guards against OSError, but a
     broken connection makes _simple_command('LOGOUT') raise IMAP4.abort,
     which is not an OSError -- so logout() propagates BEFORE its own
     shutdown() and the TCP socket stays open, leaking one fd per failed
     poll/connect until the process hits "[Errno 24] Too many open files".

  2. connect() -- ANCHOR MOVE, [PATCH-4] rewoven. Upstream wrapped the whole
     IMAP-test body in an inner try/finally (``imap = None`` ... ``finally:
     if imap is not None: _close_imap(imap)``) and DROPPED the three
     per-branch ``imap.logout()`` calls. Our block (the [PATCH-4]
     folder-ensure CREATEs, the self._open_imap() routing, and the
     process_existing conditional composed as the ``else`` of upstream's
     is_reconnect/snapshot branch) was re-indented into that inner try
     unchanged, and its three logout() calls were removed alongside
     upstream's. The ``self._seen_uids_snapshot[...] = ...`` assignment stays
     AFTER the inner try/finally, exactly where upstream put it.

  3. _fetch_new_messages() -- ``imap: Optional[imaplib.IMAP4] = None`` added
     above the outer try (upstream, for the annotation), and the finally's
     ``try: imap.logout() except: pass`` replaced by ``_close_imap(imap)``.
     [PATCH-3]'s self._open_imap() routing and [PATCH-5]'s Working-folder
     MOVE sit on context this diff does not touch and are unchanged.

  4. [PATCH-1] (this docstring), [PATCH-2], [PATCH-6], [PATCH-7] and
     [PATCH-8] sit on code the upstream diff does not touch and reapplied
     unchanged at identical anchors.

  5. dgxarley EXTENSION of the fix: our OWN two IMAP teardowns -- the
     module-level _imap_append_to_sent() ([PATCH-3]) and the instance
     _finalize_message() ([PATCH-3]/[PATCH-6] lifecycle) -- carried the
     identical leaky ``try: imap.logout() except: pass`` pattern. Upstream
     never saw those call sites (they do not exist upstream), so both were
     routed through _close_imap() for parity: same bug class, one leaked fd
     per Sent-APPEND / per finalize MOVE against a broken connection.

  6. Verification performed for this re-sync: ast.parse() on the new file;
     black --check clean; a full diff of the new file against the v2026.8.16
     baseline inspected hunk-by-hunk to confirm it contains only
     [PATCH-1]..[PATCH-8] plus black reformatting and the [UPSTREAM] comment
     markers (nothing upstream dropped or reverted); plugin.yaml and
     __init__.py confirmed byte-identical to v2026.8.13, so the ConfigMap
     subPath mount target is unchanged.

Re-synced 2026-08-15 (v2026.8.3 -> v2026.8.13, v0.20.1). This was a real
re-sync, not a byte-identical check: upstream restructured _fetch_new_messages
and connect() significantly. Summary of what changed and how it was handled:

  1. PATCH-9 RETIRED. The v2026.8.13 baseline now contains upstream commit
     65f407184d verbatim (module-level _CHARSET_ALIASES + _safe_decode(),
     consumed by _decode_header_value() and all three _extract_text_body()
     payload-decode sites) -- confirmed by diffing those three functions
     against the new baseline before deleting anything. The forward-port
     block (helpers + inline [PATCH-9] call-site comments) has been removed
     from this file. No behavior changed: the baseline's version is
     byte-identical to what we were carrying.

  2. Three unrelated upstream fixes landed in this bump, none colliding with
     our patches:
       - a7f0abc845: partial-batch dispatch (a mid-fetch exception now
         returns whatever was parsed so far instead of dropping the batch),
         seen-after-fetch UID marking (a UID is only added to _seen_uids once
         an IMAP response for it has arrived, not right after SEARCH), and a
         reconnect UID-baseline restore (new class-level _seen_uids_snapshot
         dict, keyed by address, restored on connect(is_reconnect=True) so a
         same-process reconnect does not re-mark the whole mailbox seen and
         silently skip mail that arrived during the outage).
       - 9b8da52f41: IMAP fetch failures (not just IMAP connect failures) now
         route through the fatal-error hook (_last_fetch_failed /
         _last_fetch_error, surfaced from _check_inbox() via
         _set_fatal_error() + _notify_fatal_error()), so the gateway's
         reconnect/backoff machinery reacts to a broken mailbox check instead
         of treating it as "nothing new".
       - 91bc822330: connect() now classifies terminal vs transient failures
         explicitly (smtplib.SMTPAuthenticationError -> non-retryable
         email_auth_error; generic IMAP/SMTP failures -> retryable).
     None of this is touched by our patches -- it is preserved verbatim.

  3. ANCHOR MOVES caused by (2), and how each PATCH-N section was rewoven:
       - _fetch_new_messages() split the per-message parsing out into a new
         method, _parse_fetched_message(uid, raw_email), which returns
         Optional[Dict] (None = silently-skipped automated sender) and can
         raise (caller logs the UID and continues -- a poison message no
         longer aborts the batch). [PATCH-5]'s INBOX -> Working MOVE used to
         sit inline between body/attachment extraction and the results.append
         call; it cannot live inside _parse_fetched_message() any more
         because that method no longer has access to the open `imap` handle
         nor a name that overlaps with the caller's `uid`. It was moved to
         the CALLER (_fetch_new_messages), right after
         "if parsed is not None:" and before "results.append(parsed)" --
         same open `imap` connection, same gating (done_folder AND
         working_folder AND a Message-ID present), same non-fatal
         warn-and-continue on a failed MOVE, source_folder still injected
         into the dict before it is appended. This preserves upstream's
         per-message poison guard (a MOVE never runs for a message that
         failed to parse) and the seen-after-fetch marking (untouched, still
         happens before the parse/move step).
       - connect()'s conditional pre-fill ([PATCH-4], process_existing) used
         to be the only branch inside the try block; upstream now ALSO
         branches on is_reconnect + a same-process _seen_uids_snapshot to
         decide between "restore the previous baseline" and "mark everything
         seen". These are orthogonal decisions -- is_reconnect/snapshot is
         about surviving a same-process outage, process_existing is about
         what a COLD start should do with a pre-existing backlog -- so they
         were composed as outer/inner: is_reconnect+snapshot stays the
         OUTER branch (upstream's new reconnect-restore behavior, verbatim,
         untouched), and our process_existing conditional was moved INTO the
         upstream "else" (first connect, or no snapshot yet) branch, in place
         of upstream's unconditional mark-all-seen. Folder-ensure (Working /
         Done / Sent CREATE) and routing the connection open through
         self._open_imap() ([PATCH-4]'s other half) sit unchanged, just above
         the imap.select("INBOX") call, before the branch.
       - _fetch_new_messages() and connect() now build the IMAP connection
         via self._open_imap() ([PATCH-3]'s wrapper) instead of upstream's
         inline imaplib.IMAP4_SSL(...) + login() + _send_imap_id() sequence,
         same as before this bump.
       - _dispatch_message() is untouched by the upstream diff, so [PATCH-6]
         (the try/finally around the drop-checks + handle_message, finalizing
         the mail out of Working on every path) reapplied at the identical
         anchor with no changes.
       - [PATCH-1] (this docstring), [PATCH-2] (__init__ attributes),
         [PATCH-3] (the _open_imap_conn / _imap_append_to_sent module-level
         helpers and the _open_imap / _ensure_folder / _imap_move /
         _search_message_id / _finalize_message / _append_to_sent instance
         wrappers), [PATCH-7] (_send_email{,_with_attachment,_with_attachments}
         Sent-folder APPEND) and [PATCH-8] (_standalone_send Sent-folder
         APPEND) all sit on upstream code this diff does not touch and
         reapplied unchanged at the same anchors.

  4. Verification performed for this re-sync: `ast.parse()` on the new file;
     a diff of the new file against the v2026.8.13 baseline was inspected
     hunk-by-hunk to confirm it contains only [PATCH-1]..[PATCH-8] (no
     [PATCH-9], nothing upstream reverted); grepped for the working_folder /
     done_folder / sent_folder / process_existing config.extra reads (still
     present, unchanged); confirmed _safe_decode / _CHARSET_ALIASES appear
     exactly once (from the baseline, no leftover [PATCH-9] duplicate).

The v2026.7.30 -> v2026.8.3 bump was the first real re-sync since v2026.7.7.2:
the divergence the earlier header warned about (upstream main 2026-08-02,
commit ff89f1b862, +1744 bytes) landed in that tag. It was purely the
profile-scoped secret refactor (see the fold-in block below) and touched NO
[PATCH-N] section. plugin.yaml (name: email-platform, hence the runtime
module hermes_plugins.email_platform.adapter) and __init__.py were
byte-identical at v2026.8.3 and remain so at v2026.8.13, so the ConfigMap
subPath mount target is unchanged. The re-sync check stays mandatory on
EVERY bump. See HERMES_EMAIL_UPSTREAM.md.

The v2026.7.7.2 -> v2026.7.20 -> v2026.7.30 bumps were all BYTE-IDENTICAL
re-checks (md5 39ed5d135762806451a944a9b279b8ad, 50848 bytes) and forced no
[PATCH-N] work.

FILE MOVED at the v2026.7.1 bump: upstream #41112/#3823 landed the plugin
refactor -- the adapter moved from gateway/platforms/email.py to
plugins/platforms/email/adapter.py and the static _PLATFORMS["email"] dict
was replaced by a register(ctx)->ctx.register_platform() plugin entry point
(see the "Plugin migration glue" block at the bottom of this file). The
ConfigMap subPath mount in hermes_webui_deployment.yaml.j2 was re-targeted to
/opt/hermes/plugins/platforms/email/adapter.py to match. Previous sync target
was v2026.6.19 (gateway/platforms/email.py, md5 a3f7dc61f40388bf806481b189b48e00).

Upstream changes folded in during the v2026.6.19 -> v2026.7.1 re-sync (all are
upstream-only; none collide with the [PATCH-N] logic):
  - Plugin migration: the register()/_build_adapter/_is_connected/
    _standalone_send glue block at end of file (untouched -- our patch never
    referenced the old _PLATFORMS dict).
  - SENDER AUTHENTICATION (GHSA-rxqh-5572-8m77): new module-level
    _domain_of / _domains_aligned / _verify_sender_authentication +
    _AUTH_METHOD_RE / _AUTH_PROP_RE regexes, EmailAdapter fields
    _require_authenticated_sender (env EMAIL_TRUST_FROM_HEADER / config
    require_authenticated_sender) + _authserv_id, the _allow_all_senders /
    _allowlist_in_effect statics, the sender_authenticated/auth_reason keys in
    _fetch_new_messages' results dict, and the reject-gate in _dispatch_message.
    Our [PATCH-5] source_folder key sits ALONGSIDE the two auth keys in the
    same results dict; our [PATCH-6] try/finally WRAPS the reject-gate so an
    unauthenticated-From drop still finalizes the mail out of Working.
  - __init__ now parses ports via utils.env_int / env_bool and falls back to
    config.extra for address/imap_host/smtp_host; our [PATCH-2] extra.get()
    reads reuse the same `extra` local.
  - connect() gained a `*, is_reconnect` kwarg + a missing-config fail-closed
    guard (_set_fatal_error); [PATCH-4] only rewrites the IMAP-test body below
    that guard.
  - check_email_requirements() now .strip()s and treats blank as missing.
  - `import time` was REMOVED upstream -- re-added below (our _append_to_sent
    needs time.time() for imaplib.Time2Internaldate).

Upstream changes folded in during the v2026.7.1 -> v2026.7.7.2 re-sync (all are
upstream-only robustness fixes; none collide with the [PATCH-N] logic, each sits
on original context our patches leave untouched):
  - _fetch_new_messages: guard `raw_email = msg_data[0][1]` against
    IndexError/TypeError + non-bytes payloads (skip the UID, don't abort the
    batch). Sits ABOVE our [PATCH-5] Working-MOVE, on original context.
  - new EmailAdapter._message_id_domain() helper: EMAIL_ADDRESS without an `@`
    now falls back to "localhost" instead of crashing send with IndexError.
  - the three _send_email{,_with_attachment,_with_attachments} msg_id sites now
    call _message_id_domain() instead of self._address.split('@')[1]. These are
    the same three methods our [PATCH-7] Sent-APPEND lives in; the msg_id line
    sits ABOVE each [PATCH-7] block, on original context.
  Our three PRs (#28697/#28699/#28702) were still OPEN at that tag, so no
  [PATCH-N] section could be dropped.

Upstream changes folded in during the v2026.7.30 -> v2026.8.3 re-sync (one
mechanical refactor, PR #50094 / the #59076 hunks; no [PATCH-N] section touched):
  - PROFILE-SCOPED SECRETS: every ``EMAIL_*`` read now goes through the new
    module-level _get_esecret() (alias _get_secret) / _esecret_int() /
    _esecret_bool() helpers instead of os.getenv / utils.env_int / utils.env_bool,
    so a secondary profile under gateway multiplexing reads ITS OWN credentials
    (agent.secret_scope.get_secret) while the default profile still falls back to
    os.environ on UnscopedSecretError. The `from utils import env_int, env_bool`
    import was replaced by `from utils import is_truthy_value` accordingly.
    ``GATEWAY_*`` reads deliberately stay on os.getenv (upstream does the same).
    agent/secret_scope.py already exists at v2026.7.30, so the new import is not
    a hard forward-only dependency.
  - dgxarley EXTENSION of that refactor: our [PATCH-8] _standalone_send() block
    reads EMAIL_IMAP_HOST / EMAIL_IMAP_PORT for the Sent-folder APPEND -- upstream
    has no such reads there, so they were converted to _get_secret() by hand for
    parity (an unscoped IMAP host under multiplexing would archive a secondary
    profile's reply into the default profile's mailbox).

Forward-ported ahead of the pinned tag (2026-08-09 -> RETIRED 2026-08-15):
upstream commit 65f407184d (2026-08-08, "fix(email): never let unknown or
malformed charsets abort the IMAP fetch", closes #35901/#55381/#55383) was not
in any release tag as of v2026.8.3, so it was forward-ported byte-for-byte as
[PATCH-9] (module-level _CHARSET_ALIASES + _safe_decode(), consumed by
_decode_header_value() and the three _extract_text_body() payload-decode call
sites). The v2026.8.13 baseline now contains this commit natively (verified by
diffing those three functions against the new baseline), so [PATCH-9] has been
REMOVED from this file as of the 2026-08-15 re-sync -- see item 1 above.

Adds three behaviours that upstream lacks:

  1.  Two-stage IMAP folder lifecycle:
        INBOX  -- fetch -->  Hermes_Working  -- handle_message() done -->  Hermes_Done
      so that anything sitting in Hermes_Working after a crash is visible
      as "interrupted in mid-processing", and INBOX stays empty of work
      already acknowledged.

  2.  Sent-mail archival via IMAP APPEND to Sent folder (upstream only
      pushes via SMTP and never writes to the user's IMAP).

  3.  Opt-in processing of pre-existing INBOX mail on startup (upstream
      hard-codes "ignore everything already there").

All behavioural knobs are configured via config.yaml (NOT env), mirroring the
upstream PRs -- see the Upstreaming note below. They MUST be nested under an
explicit ``extra:`` block (the loader only folds ``platforms.<name>.extra`` into
config.extra; bare keys are dropped):

  platforms.email.extra.working_folder     default "Hermes_Working"  ("" skips the
                                                              Working stage → INBOX→Done)
  platforms.email.extra.done_folder        default "Hermes_Done"  ("" disables all
                                                              moves; mail stays in INBOX
                                                              with \\Seen -- also skips
                                                              the Working stage)
  platforms.email.extra.sent_folder        default "Sent"   ("" disables IMAP APPEND)
  platforms.email.extra.process_existing   default true     (false = mark all
                                                              existing UNSEEN INBOX
                                                              UIDs as seen on startup
                                                              and only process truly
                                                              new ones)

When this file is bumped, the upstream source must be re-downloaded and the
patch sections re-applied:

  [PATCH-1] module docstring (this block)
  [PATCH-2] __init__: new self._* attributes
  [PATCH-3] new helpers: module-level _open_imap_conn (port-based SSL/STARTTLS)
            + _imap_append_to_sent (shared Sent APPEND); EmailAdapter._open_imap
            and _append_to_sent are thin instance wrappers over them. Plus
            _ensure_folder, _imap_move, _search_message_id, _finalize_message
  [PATCH-4] connect(): conditional pre-fill (composed with upstream's
            is_reconnect/_seen_uids_snapshot restore, see item 3 above) +
            folder ensure + route through _open_imap so 143/STARTTLS
            endpoints work
  [PATCH-5] _fetch_new_messages(): route through _open_imap + INBOX→Working
            MOVE per UID (now applied in the caller, after
            _parse_fetched_message() returns, see item 3 above)
  [PATCH-6] _dispatch_message(): finalize MOVE after handle_message returns
  [PATCH-7] _send_email{,_with_attachment,_with_attachments}(): APPEND to Sent
  [PATCH-8] _standalone_send() (plugin glue): APPEND to Sent via the shared
            helper, so the out-of-process cron / `hermes send` path archives
            too (parity with EmailAdapter; imap_tls preserved via _open_imap_conn)

  [PATCH-9] RETIRED 2026-08-15 -- was the 65f407184d forward-port, now native
            in the v2026.8.13 baseline. See the "Forward-ported" note above.

Upstreaming note: all of these behaviours are being upstreamed --
  - Sent-folder APPEND ([PATCH-3] shared _imap_append_to_sent helper +
    [PATCH-7] adapter call sites + [PATCH-8] standalone path) in
    PR NousResearch/hermes-agent#28697.
  - process-existing ([PATCH-4] conditional connect()-time pre-fill) in
    PR NousResearch/hermes-agent#28699.
  - INBOX→Working→Done lifecycle ([PATCH-3/4/5/6]) in
    PR NousResearch/hermes-agent#28702.
All three PRs' review-driven changes are adopted here so the patch matches what
we submitted:
  - APPEND status-tuple check (warn on NO/BAD instead of assuming success):
    backported into _imap_append_to_sent below.
  - Shared-helper refactor + standalone-path parity (#28697 review): the Sent
    APPEND is factored into the module-level _imap_append_to_sent, called from
    both EmailAdapter._append_to_sent AND _standalone_send (cron / `hermes
    send`), so the "every SMTP send archives" guarantee holds off the live
    adapter too. Unlike upstream's helper (unconditional IMAP4_SSL) ours routes
    through _open_imap_conn, preserving the 143/STARTTLS imap_tls path.
  - Lifecycle bug-fixes (#28702 review): the Working MOVE in _fetch_new_messages
    is gated on done_folder AND working_folder AND a Message-ID being present
    (no Done → no moves; no Message-ID → cannot relocate after MOVE), and
    _dispatch_message wraps its drop-checks + handle_message in one try/finally
    so an early drop (self / automated / non-allowlisted / unauthenticated) can't
    strand mail in Working.
  - ALL behavioural knobs (working_folder, done_folder, sent_folder,
    process_existing) are read from config.yaml `platforms.email.extra.*`
    (config.extra), NOT env vars. The loader only folds an explicit ``extra:``
    sub-block into config.extra (bare keys are dropped), so the keys MUST be
    nested under ``extra:`` in hermes_config.yaml.j2. NOTE our process_existing
    default is True (process the backlog), unlike upstream's False.
------------------------------------------------------------------------------
"""

import asyncio
import email as email_lib
import imaplib
import logging
import os
import re
import smtplib
import socket

# Profile-scoped secret reader for multiplexing support (PR #50094)
from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret
import ssl
import time  # dgxarley: re-added (upstream dropped it); _append_to_sent needs time.time()
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.config import Platform, PlatformConfig
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _get_esecret(name: str, default: str = "") -> str:
    """Scope-aware ``EMAIL_*`` read with the default-profile startup fallback.

    Secondary profiles run under ``_profile_runtime_scope`` — the scope is
    authoritative and a scoped miss returns ``default`` (no cross-profile
    borrow). The DEFAULT profile's adapter constructs and sends *unscoped*
    under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash its email path; there ``os.environ``
    is that profile's own value, so fall back to it. Same pattern as the
    Slack ``SLACK_APP_TOKEN`` read (#59739) and the WhatsApp
    ``_get_wsecret`` fix (5438e9c629).
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


# Backwards-compatible alias for the name used by the original #59076 hunks.
_get_secret = _get_esecret


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read (``env_int`` variant of ``_get_esecret``)."""
    raw = str(_get_esecret(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read (``env_bool`` variant of ``_get_esecret``)."""
    return is_truthy_value(_get_esecret(name, ""), default=default)


# Automated sender patterns — emails from these are silently ignored
_NOREPLY_PATTERNS = (
    "noreply",
    "no-reply",
    "no_reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "postmaster",
    "bounce",
    "notifications@",
    "automated@",
    "auto-confirm",
    "auto-reply",
    "automailer",
)

# RFC headers that indicate bulk/automated mail
_AUTOMATED_HEADERS = {
    "Auto-Submitted": lambda v: v.lower() != "no",
    "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
    "X-Auto-Response-Suppress": lambda v: bool(v),
    "List-Unsubscribe": lambda v: bool(v),
}

# Gmail-safe max length per email body
MAX_MESSAGE_LENGTH = 50_000

SMTP_CONNECT_TIMEOUT = 30


def _close_imap(imap: "imaplib.IMAP4") -> None:
    """Best-effort teardown that guarantees the underlying socket is closed.

    [UPSTREAM v2026.8.16]

    ``IMAP4.logout()`` only guards against ``OSError`` internally: a broken
    connection makes ``_simple_command('LOGOUT')`` raise ``IMAP4.abort``
    (which is *not* an ``OSError``), so ``logout()`` propagates before its
    own ``shutdown()`` call and the TCP socket stays open. On macOS, where
    the default soft fd limit is 256 and pollers may run through a local
    proxy, these abandoned sockets accumulate one per failed poll until the
    gateway hits ``[Errno 24] Too many open files`` (#79889). Always chase a
    failed ``logout()`` with ``shutdown()``, which closes the socket
    unconditionally.
    """
    try:
        imap.logout()
    except Exception:
        try:
            imap.shutdown()
        except Exception:
            pass


def _create_ipv4_connection(
    host: str,
    port: int,
    timeout: float,
    source_address: Any = None,
) -> socket.socket:
    """Create a TCP connection using only IPv4 addresses.

    This mirrors ``socket.create_connection`` but constrains DNS resolution to
    ``AF_INET``.  It avoids mutating process-global socket functions, which
    matters because email sends run in executor threads.
    """
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
        host, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(
            host,
            port,
            timeout,
            source_address=self.source_address,
        )


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        raw_sock = _create_ipv4_connection(
            host,
            port,
            timeout,
            source_address=self.source_address,
        )
        return self.context.wrap_socket(
            raw_sock,
            server_hostname=getattr(self, "_host", host),
        )


# Supported image extensions for inline detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _send_imap_id(imap: "imaplib.IMAP4") -> None:
    """Send RFC 2971 IMAP ID command identifying this client.

    Required by 163/NetEase mailbox after LOGIN: without it, every UID
    SEARCH/FETCH returns ``BYE Unsafe Login`` and disconnects.  Other
    IMAP servers either honor it silently or reject the unknown command;
    we swallow failures so non-supporting servers keep working.
    """
    try:
        try:
            from hermes_cli import __version__ as _hermes_version
        except Exception:  # noqa: BLE001 — keep ID best-effort if import fails
            _hermes_version = "0"
        imap.xatom(
            "ID",
            f'("name" "hermes-agent" "version" "{_hermes_version}" '
            '"vendor" "NousResearch" '
            '"support-email" "noreply@nousresearch.com")',
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


def _open_imap_conn(
    imap_host: str,
    imap_port: int,
    address: str,
    password: str,
) -> imaplib.IMAP4:
    """Open an authenticated IMAP connection (caller logout()s).

    Port-based auto-detect: 993 → implicit SSL (``IMAP4_SSL``); any other port
    → plain ``IMAP4`` + ``STARTTLS`` upgrade — required for Dovecot/Postfix
    setups that only expose 143. Module-level twin of
    :meth:`EmailAdapter._open_imap` so the out-of-process ``_standalone_send``
    opens IMAP over the SAME TLS logic.

    dgxarley divergence from upstream PR #28697: the upstream shared APPEND
    helper hard-codes ``IMAP4_SSL`` unconditionally, which fails on our
    143/STARTTLS endpoints with ``[SSL: RECORD_LAYER_FAILURE]`` (the server
    greets in plaintext before any handshake). Routing every APPEND through
    here keeps ``imap_tls`` working.
    """
    if imap_port == 993:
        imap: imaplib.IMAP4 = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=30)
    else:
        imap = imaplib.IMAP4(imap_host, imap_port, timeout=30)
        imap.starttls(ssl_context=ssl.create_default_context())
    imap.login(address, password)
    _send_imap_id(imap)
    return imap


def _imap_append_to_sent(
    *,
    imap_host: str,
    imap_port: int,
    address: str,
    password: str,
    sent_folder: str,
    raw_bytes: bytes,
) -> None:
    """IMAP-APPEND a freshly-sent outbound mail to ``sent_folder``.

    No-op when the folder is unset (empty string) or no IMAP host is
    configured. Best-effort: failures are logged as warnings and never
    re-raised — losing the Sent-folder copy must NOT roll back an SMTP send
    that already succeeded.

    Shared by the live :meth:`EmailAdapter._append_to_sent` and the
    out-of-process ``_standalone_send`` so both SMTP paths archive identically
    (parity with upstream PR #28697's review-driven refactor). Uses the
    port-based :func:`_open_imap_conn` (NOT upstream's unconditional
    ``IMAP4_SSL``) so ``imap_tls`` on 143/STARTTLS keeps working.
    """
    if not sent_folder or not imap_host:
        return
    try:
        imap = _open_imap_conn(imap_host, imap_port, address, password)
        try:
            # CREATE is idempotent; most servers return NO on "already exists".
            try:
                imap.create(sent_folder)
            except Exception:  # noqa: BLE001 — ignore "already exists" and similar
                pass
            # imaplib returns ("NO"/"BAD", ...) on a rejected APPEND WITHOUT
            # raising — inspect the status tuple explicitly so a silent failure
            # isn't logged as success.
            typ, data = imap.append(
                sent_folder,
                "(\\Seen)",
                imaplib.Time2Internaldate(time.time()),
                raw_bytes,
            )
            if typ != "OK":
                detail = b" ".join(p for p in data if isinstance(p, bytes)).decode("utf-8", "replace")
                logger.warning(
                    "[Email] APPEND to %r returned %s: %s",
                    sent_folder,
                    typ,
                    detail,
                )
            else:
                logger.debug("[Email] APPEND to %r ok", sent_folder)
        finally:
            # Same fd-leak guard upstream applied to its own IMAP teardowns in
            # v2026.8.16 (#79889): logout() lets IMAP4.abort escape on a broken
            # connection, leaving the socket open — one leaked fd per send.
            _close_imap(imap)
    except Exception as e:  # noqa: BLE001 — Sent-folder mirror is best-effort
        logger.warning("[Email] APPEND to %r failed: %s", sent_folder, e)


def _is_automated_sender(address: str, headers: dict) -> bool:
    """Return True if this email is from an automated/noreply source."""
    addr = address.lower()
    if any(pattern in addr for pattern in _NOREPLY_PATTERNS):
        return True
    for header, check in _AUTOMATED_HEADERS.items():
        value = headers.get(header, "")
        if value and check(value):
            return True
    return False


def check_email_requirements() -> bool:
    """Check if email platform settings are available and non-blank.

    Treats blank/whitespace-only values as missing so an abandoned setup that
    left empty ``EMAIL_*`` keys in ``.env`` does not enable the platform (#40715).
    """
    addr = _get_secret("EMAIL_ADDRESS", "").strip()
    pwd = _get_secret("EMAIL_PASSWORD", "").strip()
    imap = _get_secret("EMAIL_IMAP_HOST", "").strip()
    smtp = _get_secret("EMAIL_SMTP_HOST", "").strip()
    return all([addr, pwd, imap, smtp])


_CHARSET_ALIASES = {
    # Aliases seen in the wild that Python's codec registry doesn't know.
    # "unknown-8bit" / "x-unknown" are RFC 1428 placeholders some MTAs (QQ
    # Mail among them) emit when the original charset was lost (#35901).
    "unknown-8bit": "utf-8",
    "unknown": "utf-8",
    "x-unknown": "utf-8",
    "default": "utf-8",
    "ansi_x3.110-1983": "latin-1",
    "cp-850": "cp850",
    "gb2312": "gb18030",  # superset; avoids failures on GBK extensions
    "gbk": "gb18030",
    "ks_c_5601-1987": "cp949",
}


def _safe_decode(payload: bytes, charset: "Optional[str]") -> str:
    """Decode *payload* without ever raising.

    Unknown or malformed charset labels (``unknown-8bit``, misspelled names,
    attacker-controlled garbage) previously raised ``LookupError`` from
    ``bytes.decode`` — ``errors="replace"`` only guards decode errors, not a
    missing codec — which aborted the whole IMAP fetch and dropped every
    message in the batch (#35901, #55381, #55383). Fall back through a small
    alias table, then UTF-8, then latin-1 (which never fails).
    """
    label = (charset or "utf-8").strip().strip("\"'").lower() or "utf-8"
    label = _CHARSET_ALIASES.get(label, label)
    for candidate in (label, "utf-8"):
        try:
            return payload.decode(candidate, errors="replace")
        except (LookupError, ValueError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 encoded email header into a plain string.

    Never raises: malformed encoded-words or unknown charsets degrade to
    replacement characters instead of crashing the fetch loop (#55381).
    """
    try:
        parts = decode_header(raw)
    except Exception:  # malformed RFC 2047 structure
        return raw
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(_safe_decode(part, charset))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            # Skip attachments
            if "attachment" in disposition:
                continue
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return _safe_decode(payload, part.get_content_charset())
        # Fallback: try text/html and strip tags
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                continue
            if content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html = _safe_decode(payload, part.get_content_charset())
                    return _strip_html(html)
        return ""
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = _safe_decode(payload, msg.get_content_charset())
            if msg.get_content_type() == "text/html":
                return _strip_html(text)
            return text
        return ""


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <addr>' format."""
    match = re.search(r"<([^>]+)>", raw)
    if match:
        return match.group(1).strip().lower()
    return raw.strip().lower()


def _domain_of(address: str) -> str:
    """Return the lowercased domain part of an email address, or ''."""
    _, _, domain = address.rpartition("@")
    return domain.strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Return True if two domains are equal or in an organizational
    parent/subdomain relationship (relaxed DMARC alignment).

    DMARC relaxed alignment treats ``mail.example.com`` as aligned with
    ``example.com``. We approximate organizational alignment by checking
    exact equality or that one domain is a dot-suffix of the other.
    """
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith("." + b) or b.endswith("." + a)


# Match a single "method=result" token in an Authentication-Results header,
# e.g. ``dmarc=pass`` or ``spf=fail``.
_AUTH_METHOD_RE = re.compile(r"\b(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE)
# Match a property value like ``header.from=example.com`` or
# ``smtp.mailfrom=user@example.com``.
_AUTH_PROP_RE = re.compile(
    r"\b(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*([^\s;]+)",
    re.IGNORECASE,
)


def _verify_sender_authentication(
    msg: email_lib.message.Message,
    from_addr: str,
    *,
    authserv_id: str = "",
) -> Tuple[bool, str]:
    """Verify that the message's ``From:`` domain is authenticated.

    The ``From:`` header is attacker-controlled and is never authenticated by
    IMAP delivery, so an allowlist keyed on ``From:`` alone is trivially
    spoofable (GHSA-rxqh-5572-8m77). The only trustworthy signal is the
    ``Authentication-Results`` header that the *receiving* mail server (the one
    we IMAP into) stamps after running SPF/DKIM/DMARC. That header is prepended
    by our own server, so the topmost instance is the one we trust; any
    ``Authentication-Results`` an attacker injected into the body of their
    message sorts below it.

    Returns ``(authenticated, reason)``. ``authenticated`` is True when:
      * a DMARC pass is recorded for the From domain, OR
      * an SPF pass aligned with the From domain, OR
      * a DKIM pass aligned (``header.d``) with the From domain.

    When no ``Authentication-Results`` header is present at all, we return
    ``(False, "no Authentication-Results header")`` — fail-closed. Operators
    whose mail server does not stamp this header can opt out of the check
    (see ``EmailAdapter._require_authenticated_sender``).
    """
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"

    # get_all preserves header order; the receiving server prepends its result,
    # so the FIRST Authentication-Results is the trusted one. We pin to the
    # configured authserv-id when provided to defend against an injected header
    # that happens to sort first.
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return False, "no Authentication-Results header"

    trusted = None
    for raw in headers:
        value = " ".join(str(raw).split())
        if authserv_id:
            # authserv-id is the first token before the first ';'
            serv = value.split(";", 1)[0].strip().lower()
            if not _domains_aligned(serv, authserv_id) and serv != authserv_id.lower():
                continue
        trusted = value
        break
    if trusted is None:
        return False, "no Authentication-Results from trusted authserv-id"

    methods = {m.lower(): r.lower() for m, r in _AUTH_METHOD_RE.findall(trusted)}
    props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(trusted)}

    # 1) DMARC pass is the strongest signal — DMARC already enforces From
    #    alignment, so a pass means the From domain is authenticated.
    if methods.get("dmarc") == "pass":
        return True, "dmarc=pass"

    # 2) SPF pass aligned with the From domain (the envelope/MAIL FROM domain
    #    must match the From domain).
    if methods.get("spf") == "pass":
        spf_domain = (
            _domain_of(props.get("smtp.mailfrom", "")) or props.get("smtp.from", "") or props.get("envelope-from", "")
        )
        spf_domain = _domain_of(spf_domain) if "@" in spf_domain else spf_domain
        if _domains_aligned(spf_domain, from_domain):
            return True, "spf=pass aligned"

    # 3) DKIM pass aligned with the From domain (the signing domain header.d
    #    must align with the From domain).
    if methods.get("dkim") == "pass":
        dkim_domain = props.get("header.d", "") or _domain_of(props.get("header.from", ""))
        if _domains_aligned(dkim_domain, from_domain):
            return True, "dkim=pass aligned"

    return False, f"authentication failed ({trusted[:120]})"


def _extract_attachments(
    msg: email_lib.message.Message,
    skip_attachments: bool = False,
) -> List[Dict[str, Any]]:
    """Extract attachment metadata and cache files locally.

    When *skip_attachments* is True, all attachment/inline parts are ignored
    (useful for malware protection or bandwidth savings).
    """
    attachments = []
    if not msg.is_multipart():
        return attachments

    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", ""))
        if skip_attachments and ("attachment" in disposition or "inline" in disposition):
            continue
        if "attachment" not in disposition and "inline" not in disposition:
            continue
        # Skip text/plain and text/html body parts
        content_type = part.get_content_type()
        if content_type in {"text/plain", "text/html"} and "attachment" not in disposition:
            continue

        filename = part.get_filename()
        if filename:
            filename = _decode_header_value(filename)
        else:
            ext = part.get_content_subtype() or "bin"
            filename = f"attachment.{ext}"

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        ext = Path(filename).suffix.lower()
        if ext in _IMAGE_EXTS:
            try:
                cached_path = cache_image_from_bytes(payload, ext)
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
            attachments.append(
                {
                    "path": cached_path,
                    "filename": filename,
                    "type": "image",
                    "media_type": content_type,
                }
            )
        else:
            cached_path = cache_document_from_bytes(payload, filename)
            attachments.append(
                {
                    "path": cached_path,
                    "filename": filename,
                    "type": "document",
                    "media_type": content_type,
                }
            )

    return attachments


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    # Per-account snapshot of seen UIDs, surviving adapter recreation.
    # The gateway's reconnect watcher builds a FRESH adapter instance for
    # each retry; without this, connect(is_reconnect=True) would re-mark the
    # entire mailbox seen and silently skip every message that arrived
    # during the outage. Keyed by account address (multiplex gateways can
    # run several email accounts in one process). Same-process only by
    # design — after a full restart the usual mark-all-seen baseline applies.
    _seen_uids_snapshot: Dict[str, set] = {}

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)

        # Resolve connection settings from the env vars first, then fall back to
        # PlatformConfig.extra (address/imap_host/smtp_host) — the canonical dict
        # gateway.config populates and that the "connected" check, the
        # send-helper, and `hermes config show` already read. Without the
        # fallback a config.yaml-only setup left these empty. Host/address values
        # are stripped: a stray space or newline made IMAP4_SSL raise the
        # misleading ``[Errno 8] nodename nor servname`` (an unresolvable name)
        # instead of an obvious "host not set" error.
        extra = config.extra or {}
        self._address = (_get_secret("EMAIL_ADDRESS", "") or extra.get("address", "")).strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = (_get_secret("EMAIL_IMAP_HOST", "") or extra.get("imap_host", "")).strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        self._smtp_host = (_get_secret("EMAIL_SMTP_HOST", "") or extra.get("smtp_host", "")).strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)

        # Skip attachments — configured via config.yaml:
        #   platforms:
        #     email:
        #       skip_attachments: true
        self._skip_attachments = extra.get("skip_attachments", False)

        # [PATCH-2] dgxarley: folder-lifecycle + APPEND-to-Sent + process-existing,
        # all configured via config.yaml under an explicit ``extra:`` block (the
        # loader only folds ``platforms.<name>.extra`` into config.extra; bare keys
        # are dropped):
        #   platforms:
        #     email:
        #       extra:
        #         skip_attachments: true
        #         working_folder: "Hermes_Working"  # "" to skip the Working stage
        #         done_folder: "Hermes_Done"        # "" to disable all folder moves
        #         sent_folder: "Sent"               # "" to disable IMAP APPEND
        #         process_existing: true            # process pre-existing backlog
        # All four mirror the upstream PRs (NousResearch/hermes-agent #28697,
        # #28699, #28702). Empty string is a deliberate opt-out for the folder
        # vars — do NOT collapse with `or`. Lifecycle semantics:
        #   done_folder="" .............. no moves at all (INBOX, \Seen)
        #   working_folder="", done set . INBOX -> Done directly
        #   both set .................... INBOX -> Working -> Done
        # NOTE: our default for process_existing is True (process the backlog),
        # unlike upstream's False.
        self._working_folder = extra.get("working_folder", "Hermes_Working")
        self._done_folder = extra.get("done_folder", "Hermes_Done")
        self._sent_folder = extra.get("sent_folder", "Sent")
        self._process_existing = bool(extra.get("process_existing", True))

        # Require the sender's From: domain to be authenticated (SPF/DKIM/DMARC)
        # before trusting it for authorization. The From: header is
        # attacker-controlled and unauthenticated by IMAP, so an allowlist keyed
        # on it alone is spoofable (GHSA-rxqh-5572-8m77). Default ON (fail-closed).
        #
        # Operators whose receiving mail server does not stamp an
        # Authentication-Results header can opt out via config.yaml:
        #   platforms:
        #     email:
        #       require_authenticated_sender: false
        # or the EMAIL_TRUST_FROM_HEADER=true env mirror (parity with the other
        # EMAIL_* access-control vars). When allow-all is in effect the operator
        # has already chosen to accept any sender, so the check is moot and the
        # gate below is skipped.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        elif _esecret_bool("EMAIL_TRUST_FROM_HEADER", False):
            self._require_authenticated_sender = False
        else:
            self._require_authenticated_sender = True

        # Optional authserv-id to pin Authentication-Results to the operator's
        # own receiving server (defends against an injected header that sorts
        # first). Defaults to the From-domain of the agent's own address.
        self._authserv_id = (extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")).strip().lower()

        # Track message IDs we've already processed to avoid duplicates
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000  # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None

        # Track the last IMAP fetch attempt so the poll loop can distinguish
        # "checked, nothing new" from "the check itself failed" (#80016).
        self._last_fetch_failed: bool = False
        self._last_fetch_error: str = ""

        # Map chat_id (sender email) -> last subject + message-id for threading
        self._thread_context: Dict[str, Dict[str, str]] = {}

        logger.info("[Email] Adapter initialized for %s", self._address)
        logger.info(
            "[Email] Folder lifecycle: working=%r done=%r sent=%r process_existing=%s",
            self._working_folder,
            self._done_folder,
            self._sent_folder,
            self._process_existing,
        )

    def _trim_seen_uids(self) -> None:
        """Keep only the most recent UIDs to prevent unbounded memory growth.

        IMAP UIDs are monotonically increasing integers. When the set grows
        beyond the cap, we keep only the highest half — old UIDs are safe to
        drop because new messages always have higher UIDs and IMAP's UNSEEN
        flag prevents re-delivery regardless.
        """
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            # UIDs are bytes like b'1234' — sort numerically and keep top half
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))
            keep = self._seen_uids_max // 2
            self._seen_uids = set(sorted_uids[-keep:])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            # Fallback: just clear old entries if sort fails
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2 :])

    def _connect_smtp(self) -> smtplib.SMTP:
        """Create an SMTP connection, selecting the correct protocol for the port.

        Port 465 uses implicit TLS (``SMTP_SSL``).  All other ports use
        ``SMTP`` + ``STARTTLS``.

        When the host resolves to an IPv6 address that is unreachable
        (common on networks without IPv6 routing), the default connection can
        hang until the socket timeout expires.  We retry connection-level
        failures through an IPv4-only socket path, without mutating global
        resolver state.  TLS verification errors are not retried.

        Returns a connected SMTP object with TLS established — callers
        can proceed directly to ``login()``.
        """
        ctx = ssl.create_default_context()
        host = self._smtp_host
        port = self._smtp_port

        def _connect(*, ipv4_only: bool = False) -> smtplib.SMTP:
            """Attempt one SMTP connection."""
            smtp_cls = _IPv4SMTP if ipv4_only else smtplib.SMTP
            smtp_ssl_cls = _IPv4SMTP_SSL if ipv4_only else smtplib.SMTP_SSL
            if port == 465:
                return smtp_ssl_cls(host, port, timeout=SMTP_CONNECT_TIMEOUT, context=ctx)
            smtp = smtp_cls(host, port, timeout=SMTP_CONNECT_TIMEOUT)
            try:
                smtp.starttls(context=ctx)
            except Exception:
                smtp.close()
                raise
            return smtp

        try:
            return _connect()
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            # Connection-level failure (may be unreachable IPv6).
            # Retry with IPv4 only.
            return _connect(ipv4_only=True)

    # ------------------------------------------------------------------
    # [PATCH-3] IMAP folder-lifecycle helpers (dgxarley patch).
    # ------------------------------------------------------------------

    def _open_imap(self) -> imaplib.IMAP4:
        """Open an authenticated IMAP connection (caller logout()s).

        Thin instance wrapper over the module-level :func:`_open_imap_conn`
        (port-based SSL/STARTTLS auto-detect) so the live adapter and the
        out-of-process ``_standalone_send`` open IMAP over identical TLS logic.
        The STARTTLS branch is required for Dovecot/Postfix-style setups that
        only expose 143 — upstream uses ``IMAP4_SSL`` unconditionally, which
        fails there with ``[SSL: RECORD_LAYER_FAILURE]``.
        """
        return _open_imap_conn(self._imap_host, self._imap_port, self._address, self._password)

    @staticmethod
    def _ensure_folder(imap: imaplib.IMAP4, name: str) -> None:
        """Idempotently CREATE *name* on the IMAP server.

        ``IMAP CREATE`` returns NO if the folder already exists, which we
        accept silently — there is no portable EXISTS check across servers.
        """
        if not name:
            return
        try:
            status, _ = imap.create(name)
            if status == "OK":
                logger.info("[Email] Created IMAP folder %r", name)
            # NO usually means "already exists" — fine.
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.debug("[Email] CREATE %r ignored: %s", name, e)

    @staticmethod
    def _imap_move(
        imap: imaplib.IMAP4,
        uid: bytes,
        dst_folder: str,
    ) -> bool:
        """MOVE *uid* (in the currently SELECTed folder) to *dst_folder*.

        Tries the RFC 6851 ``UID MOVE`` first; falls back to
        ``UID COPY`` + ``UID STORE +FLAGS \\Deleted`` + ``EXPUNGE`` on
        servers that don't advertise MOVE. Returns True on apparent success.
        """
        # Try native MOVE first.
        try:
            status, data = imap.uid("MOVE", uid, dst_folder)
            if status == "OK":
                return True
            logger.debug("[Email] UID MOVE %s → %r: %s %s", uid, dst_folder, status, data)
        except Exception as e:  # noqa: BLE001 — fall through to COPY+EXPUNGE
            logger.debug("[Email] UID MOVE %s → %r raised: %s", uid, dst_folder, e)

        # Fallback: COPY + flag deleted + EXPUNGE the moved UID.
        try:
            status, _ = imap.uid("COPY", uid, dst_folder)
            if status != "OK":
                logger.warning("[Email] UID COPY %s → %r failed", uid, dst_folder)
                return False
            imap.uid("STORE", uid, "+FLAGS", "(\\Deleted)")
            # Prefer UID EXPUNGE (RFC 4315) so we don't expunge collateral
            # deleted-flagged mails the user has in the same folder.
            try:
                imap.uid("EXPUNGE", uid)
            except Exception:  # noqa: BLE001 — server lacks UIDPLUS
                imap.expunge()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("[Email] COPY+EXPUNGE move %s → %r failed: %s", uid, dst_folder, e)
            return False

    @staticmethod
    def _search_message_id(
        imap: imaplib.IMAP4,
        message_id: str,
    ) -> List[bytes]:
        """Return UIDs in the currently SELECTed folder whose Message-ID matches.

        Used to re-locate a mail after we MOVE'd it (UIDs change on MOVE,
        but the RFC 2822 Message-ID header is stable).
        """
        if not message_id:
            return []
        # The header value must be IMAP-string-quoted; imaplib does this for us
        # when we pass it as a separate argument.
        try:
            status, data = imap.uid("SEARCH", None, "HEADER", "Message-ID", message_id)
            if status != "OK" or not data or not data[0]:
                return []
            return data[0].split()
        except Exception as e:  # noqa: BLE001
            logger.debug("[Email] SEARCH Message-ID %r failed: %s", message_id, e)
            return []

    def _finalize_message(self, message_id: str, source_folder: str) -> None:
        """Move a now-processed mail to ``self._done_folder`` (best-effort).

        Runs in an executor thread (synchronous IMAP). If ``_done_folder`` is
        empty, this is a no-op — the mail stays where it is, with the IMAP
        \\Seen flag set automatically by the earlier RFC822 FETCH.
        """
        if not self._done_folder:
            return
        if self._done_folder == source_folder:
            # Already in the destination (e.g. user disabled Working and we
            # moved straight to Done during fetch). Nothing to do.
            return
        if not message_id:
            logger.debug(
                "[Email] Skipping finalize MOVE — no Message-ID for mail "
                "in %r (cannot relocate after handle_message)",
                source_folder,
            )
            return
        try:
            imap = self._open_imap()
            try:
                imap.select(source_folder)
                uids = self._search_message_id(imap, message_id)
                if not uids:
                    logger.warning(
                        "[Email] Cannot find Message-ID %s in %r — " "skipping MOVE → %r",
                        message_id,
                        source_folder,
                        self._done_folder,
                    )
                    return
                for uid in uids:
                    self._imap_move(imap, uid, self._done_folder)
                logger.info(
                    "[Email] Finalized %s: %r → %r",
                    message_id,
                    source_folder,
                    self._done_folder,
                )
            finally:
                # Same fd-leak guard upstream applied to its own IMAP teardowns
                # in v2026.8.16 (#79889).
                _close_imap(imap)
        except Exception as e:  # noqa: BLE001
            logger.error("[Email] Finalize MOVE failed: %s", e)

    def _append_to_sent(self, raw_bytes: bytes) -> None:
        """IMAP-APPEND a freshly-sent outbound mail to ``self._sent_folder``.

        Thin wrapper over the shared :func:`_imap_append_to_sent` helper so the
        live adapter and the out-of-process ``_standalone_send`` archive
        identically. Best-effort — see the helper for failure semantics
        (no-op on empty folder; status-tuple checked; never re-raised).
        """
        _imap_append_to_sent(
            imap_host=self._imap_host,
            imap_port=self._imap_port,
            address=self._address,
            password=self._password,
            sent_folder=self._sent_folder,
            raw_bytes=raw_bytes,
        )

    # ------------------------------------------------------------------
    # End of [PATCH-3] block.
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host surfaces as an actionable config
        # error instead of IMAP4_SSL("") raising the cryptic
        # ``[Errno 8] nodename nor servname provided, or not known``.
        missing = [
            name
            for name, value in (
                ("EMAIL_ADDRESS", self._address),
                ("EMAIL_PASSWORD", self._password),
                ("EMAIL_IMAP_HOST", self._imap_host),
                ("EMAIL_SMTP_HOST", self._smtp_host),
            )
            if not value
        ]
        if missing:
            message = (
                "Not configured — missing "
                + ", ".join(missing)
                + ". Set it via `hermes gateway setup` (env) or platforms.email "
                "in config.yaml."
            )
            logger.error("[Email] %s", message)
            # Mark non-retryable so the gateway does NOT keep reconnecting against
            # an empty host. A blank-but-present env var (e.g. ``EMAIL_IMAP_HOST=``)
            # used to slip past the startup gate and drive an indefinite retry
            # loop that leaked memory until the host OOM-killed (#40715).
            self._set_fatal_error("email_missing_configuration", message, retryable=False)
            return False

        try:
            # [UPSTREAM v2026.8.16] The handle is closed in ``finally`` —
            # before this, a failure in login/select/search left the TCP socket
            # open with no owner, leaking one fd per connect attempt. Under the
            # gateway's reconnect watcher (fresh adapter instance per retry)
            # against an unreachable/proxied host this grew monotonically until
            # fd exhaustion on macOS's 256 soft limit (#79889). The three
            # per-branch ``imap.logout()`` calls this replaces are gone.
            imap = None
            try:
                # Test IMAP connection — uses port-based SSL/STARTTLS auto-detect
                # so Dovecot-style 143-with-STARTTLS endpoints work alongside the
                # 993-implicit-SSL providers upstream targets exclusively.
                imap = self._open_imap()

                # [PATCH-4] Ensure our managed folders exist before any MOVE
                # touches them. CREATE is idempotent (NO on "already exists"),
                # so this is safe to run every reconnect. Working/Done are only
                # created when the lifecycle is active (done_folder set) — with no
                # Done there are no moves, so nothing to create. The Sent folder is
                # independent (used by APPEND regardless of the lifecycle).
                # _ensure_folder itself no-ops on an empty name.
                if self._done_folder:
                    self._ensure_folder(imap, self._working_folder)
                    self._ensure_folder(imap, self._done_folder)
                self._ensure_folder(imap, self._sent_folder)

                imap.select("INBOX")
                snapshot = self._seen_uids_snapshot.get(self._address)
                if is_reconnect and snapshot is not None:
                    # [UPSTREAM v2026.8.13] Reconnect within the same process:
                    # restore the previous adapter's seen-UID baseline instead of
                    # re-marking the whole mailbox. Mail that arrived during the
                    # outage stays UNSEEN relative to the baseline and is
                    # dispatched by the next poll instead of being silently
                    # skipped. Orthogonal to our process_existing knob below —
                    # this branch only fires on a same-process RECONNECT, never
                    # on a cold start, so it composes cleanly with [PATCH-4].
                    self._seen_uids = set(snapshot)
                    self._trim_seen_uids()
                    logger.info(
                        "[Email] IMAP reconnect test passed. Restored %d seen UIDs; "
                        "messages received during the outage will be processed.",
                        len(self._seen_uids),
                    )
                else:
                    # [PATCH-4] First connect (or no snapshot yet): pre-fill
                    # _seen_uids ONLY when not opted in to processing existing
                    # INBOX mail. Upstream's fallback here always pre-fills (=
                    # ignore everything already there); with process_existing=true
                    # (config.yaml) we leave the set empty so the next poll picks
                    # up the historical UNSEEN backlog.
                    if not self._process_existing:
                        status, data = imap.uid("search", None, "ALL")
                        if status == "OK" and data and data[0]:
                            for uid in data[0].split():
                                self._seen_uids.add(uid)
                        self._trim_seen_uids()
                        logger.info(
                            "[Email] IMAP connection test passed. %d existing messages skipped.",
                            len(self._seen_uids),
                        )
                    else:
                        logger.info(
                            "[Email] IMAP connection test passed. process_existing=true — "
                            "will process pre-existing UNSEEN mail on first poll."
                        )
            finally:
                if imap is not None:
                    _close_imap(imap)
            # [UPSTREAM v2026.8.13] Keep the reconnect snapshot current after
            # either branch above, so a later same-process reconnect restores
            # whatever baseline this connect (cold-start or reconnect) landed
            # on.
            self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        except Exception as e:
            logger.error("[Email] IMAP connection failed: %s", e)
            # Always set an explicit fatal code (OOF-156): returning False
            # with no error info made the gateway treat every IMAP failure —
            # including permanently bad credentials — as transient, retrying
            # forever with zero owner signal ("stuck retrying 22h").
            # Kept retryable=True deliberately: imaplib raises the same
            # generic IMAP4.error for bad credentials AND transient server
            # NOs (e.g. Gmail's "too many simultaneous connections"), so a
            # type-based terminal classification isn't safe here. Long-lived
            # loops surface via the reconnect watcher's NEEDS_ATTENTION
            # escalation instead.
            self._set_fatal_error(
                "email_imap_connect_error",
                f"IMAP connection to {self._imap_host}:{self._imap_port} failed: {e}",
                retryable=True,
            )
            return False

        try:
            # Test SMTP connection
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
        except smtplib.SMTPAuthenticationError as e:
            logger.error("[Email] SMTP authentication failed: %s", e)
            # Typed auth failure (535 & friends): bad or revoked credentials
            # can never self-heal, so drop out of the reconnect queue instead
            # of retrying a dead password forever (OOF-156). Type-based only —
            # SMTPAuthenticationError is unambiguous, unlike IMAP4.error above.
            self._set_fatal_error(
                "email_auth_error",
                f"SMTP authentication failed for {self._address}: {e}. "
                "Check EMAIL_PASSWORD (for Gmail/Outlook this must be an "
                "app password, not the account password).",
                retryable=False,
            )
            return False
        except Exception as e:
            logger.error("[Email] SMTP connection failed: %s", e)
            self._set_fatal_error(
                "email_smtp_connect_error",
                f"SMTP connection to {self._smtp_host} failed: {e}",
                retryable=True,
            )
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("[Email] Disconnected.")

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages at regular intervals."""
        while self._running:
            try:
                await self._check_inbox()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Email] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unseen messages and dispatch them."""
        # Run IMAP operations in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, self._fetch_new_messages)
        # [UPSTREAM v2026.8.13] Dispatch whatever the fetch managed to return
        # BEFORE escalating a failure: on a mid-batch exception
        # _fetch_new_messages returns the partial results, and dropping them
        # here would lose those messages (their processing already marked
        # them seen).
        for msg_data in messages:
            await self._dispatch_message(msg_data)
        if self._last_fetch_failed:
            # [UPSTREAM v2026.8.13] The IMAP check itself failed
            # (connect/login/select/search/fetch), not just an empty inbox.
            # Surface it through the fatal-error hook so the gateway's
            # existing reconnect/backoff/status machinery re-establishes the
            # mailbox instead of silently treating every failed check as
            # "nothing new" (#80016). The handler runs in a detached task
            # (gateway/run.py), so awaiting it from our own poll task is safe
            # even though teardown cancels this task.
            self._last_fetch_failed = False
            self._set_fatal_error(
                "email_imap_fetch_failed",
                self._last_fetch_error or "IMAP fetch failed",
                retryable=True,
            )
            await self._notify_fatal_error()

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch new (unseen) messages from IMAP. Runs in executor thread."""
        results = []
        imap: Optional[imaplib.IMAP4] = None
        try:
            imap = self._open_imap()
            try:
                imap.select("INBOX")

                status, data = imap.uid("search", None, "UNSEEN")
                if status != "OK" or not data or not data[0]:
                    return results

                for uid in data[0].split():
                    if uid in self._seen_uids:
                        continue

                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        # [UPSTREAM v2026.8.13] Transient per-UID fetch
                        # refusal: leave the UID out of _seen_uids so the
                        # next poll retries it.
                        continue

                    # [UPSTREAM v2026.8.13] IMAP fetch can return unexpected
                    # structures (e.g. a single bytes item instead of a list
                    # of tuples). Mark the UID seen once a response arrived
                    # (even a malformed one) so a garbage response is skipped
                    # once, not retried forever — but NOT before the fetch: a
                    # connection failure above must leave the remaining batch
                    # eligible for the next poll instead of permanently
                    # skipping it (#80032 review).
                    self._seen_uids.add(uid)
                    # Trim periodically to prevent unbounded memory growth
                    if len(self._seen_uids) > self._seen_uids_max:
                        self._trim_seen_uids()

                    try:
                        raw_email = msg_data[0][1]
                    except (IndexError, TypeError):
                        logger.warning(
                            "[Email] Unexpected IMAP response structure for UID %s, skipping",
                            uid,
                        )
                        continue
                    if not isinstance(raw_email, (bytes, bytearray)):
                        logger.warning("[Email] Non-bytes IMAP payload for UID %s, skipping", uid)
                        continue
                    # [UPSTREAM v2026.8.13] Per-message processing guard: one
                    # poison message (unparseable headers, pathological
                    # attachment, DNS hiccup in SPF/DKIM verification) must
                    # not abort the batch or escalate to a reconnect — it is
                    # already marked seen above, so log the UID and move on
                    # (#80032 review).
                    try:
                        parsed = self._parse_fetched_message(uid, raw_email)
                    except Exception as parse_exc:
                        logger.error(
                            "[Email] Failed to process message UID %s, skipping: %s",
                            uid,
                            parse_exc,
                        )
                        continue
                    if parsed is not None:
                        # [PATCH-5] Two-stage move: park the mail in
                        # self._working_folder while the agent runs, so a
                        # crash mid-processing leaves a visible "in-flight"
                        # trail. Gated on the FULL lifecycle being enabled:
                        #   - done_folder must be set, else there are no
                        #     moves at all (done_folder="" = INBOX + \Seen);
                        #     moving to Working with no Done would strand the
                        #     mail there.
                        #   - a Message-ID must be present: the mail is
                        #     re-located in Working by Message-ID for the
                        #     final MOVE -> Done (UIDs do not survive a
                        #     MOVE), so without one it could not advance.
                        # Mail that skips the Working move stays in INBOX and
                        # is moved straight to Done by _finalize_message
                        # (when done is set). Runs here (the caller), not
                        # inside _parse_fetched_message, because it needs the
                        # still-open `imap` connection and must not run for a
                        # message that failed to parse or was silently
                        # skipped (automated senders return None above).
                        source_folder = "INBOX"
                        if self._done_folder and self._working_folder and parsed["message_id"]:
                            if self._imap_move(imap, uid, self._working_folder):
                                source_folder = self._working_folder
                            else:
                                logger.warning(
                                    "[Email] Working-folder MOVE failed for UID %s "
                                    "— continuing with mail still in INBOX",
                                    uid,
                                )
                        # [PATCH-5] Carries the folder where the mail now
                        # lives so _dispatch_message -> _finalize_message
                        # knows where to look it up for the final MOVE ->
                        # Done.
                        parsed["source_folder"] = source_folder
                        results.append(parsed)
            finally:
                # [UPSTREAM v2026.8.16] _close_imap guarantees the socket dies
                # even when logout() raises IMAP4.abort on a broken connection
                # (#79889).
                _close_imap(imap)
        except Exception as e:
            logger.error("[Email] IMAP fetch error: %s", e)
            # [UPSTREAM v2026.8.13] Surfaced via _check_inbox()'s fatal-error
            # hook after this batch's partial results have been dispatched.
            self._last_fetch_failed = True
            self._last_fetch_error = str(e)
        # [UPSTREAM v2026.8.13] Keep the reconnect snapshot current with every
        # poll so a mid-outage adapter recreation restores an up-to-date
        # baseline: stale snapshots would re-dispatch messages this instance
        # already processed.
        self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        return results

    def _parse_fetched_message(self, uid: bytes, raw_email: "bytes | bytearray") -> Optional[Dict[str, Any]]:
        """Parse one fetched RFC822 payload into a dispatchable dict.

        [UPSTREAM v2026.8.13] Returns ``None`` for messages that should be
        silently skipped (automated/noreply senders). Raises on pathological
        input — the caller's per-message guard logs the UID and continues, so
        a poison message never aborts the batch or escalates to a reconnect.
        Split out of the inline _fetch_new_messages loop body by the upstream
        a7f0abc845 refactor; the [PATCH-5] Working-folder MOVE stays in the
        caller (see the comment there) since this method has no `imap` handle.
        """
        msg = email_lib.message_from_bytes(raw_email)

        sender_raw = msg.get("From", "")
        sender_addr = _extract_email_address(sender_raw)
        sender_name = _decode_header_value(sender_raw)
        # Remove email from name if present
        if "<" in sender_name:
            sender_name = sender_name.split("<")[0].strip().strip('"')

        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        message_id = msg.get("Message-ID", "")
        in_reply_to = msg.get("In-Reply-To", "")
        # Skip automated/noreply senders before any processing
        msg_headers = dict(msg.items())
        if _is_automated_sender(sender_addr, msg_headers):
            logger.debug("[Email] Skipping automated sender: %s", sender_addr)
            return None

        # Verify the From: domain is authenticated (SPF/DKIM/DMARC)
        # while the raw message — and its trusted
        # Authentication-Results header — is still in scope. The
        # verdict is consumed at dispatch where authorization is
        # decided. From: is attacker-controlled, so this is the only
        # place a spoof can be caught (GHSA-rxqh-5572-8m77).
        sender_authenticated, auth_reason = _verify_sender_authentication(
            msg, sender_addr, authserv_id=self._authserv_id
        )

        body = _extract_text_body(msg)
        attachments = _extract_attachments(msg, skip_attachments=self._skip_attachments)

        return {
            "uid": uid,
            "sender_addr": sender_addr,
            "sender_name": sender_name,
            "subject": subject,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "body": body,
            "attachments": attachments,
            "date": msg.get("Date", ""),
            "sender_authenticated": sender_authenticated,
            "auth_reason": auth_reason,
        }

    @staticmethod
    def _allow_all_senders() -> bool:
        """Return True when the operator opted into accepting any sender.

        Mirrors the gateway authz allow-all resolution: the per-platform
        EMAIL_ALLOW_ALL_USERS flag or the global GATEWAY_ALLOW_ALL_USERS flag.
        When either is set, sender identity is moot, so the From: authentication
        gate is skipped.
        """
        truthy = {"true", "1", "yes"}
        return (
            _get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in truthy
            or os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in truthy
        )

    @staticmethod
    def _allowlist_in_effect() -> bool:
        """Return True when a sender allowlist gates email access.

        Authorization keys on the From: address only when an allowlist is
        configured — the per-platform EMAIL_ALLOWED_USERS or the global
        GATEWAY_ALLOWED_USERS. When neither is set the gateway default-denies
        every sender regardless, so the spoofable From: identity grants nothing
        and the authentication gate is unnecessary.
        """
        return bool(_get_secret("EMAIL_ALLOWED_USERS", "").strip() or os.getenv("GATEWAY_ALLOWED_USERS", "").strip())

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        message_id = msg_data["message_id"]
        # [PATCH-6] Folder the mail currently lives in (set by
        # _fetch_new_messages). The finally below ALWAYS finalizes it, so an
        # early drop (self / automated / non-allowlisted / unauthenticated) after
        # a Working move can't strand the mail in Working.
        source_folder = msg_data.get("source_folder", "INBOX")

        try:
            # Skip self-messages
            if sender_addr == self._address.lower():
                return

            # Never reply to automated senders
            if _is_automated_sender(sender_addr, {}):
                logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
                return

            # Skip senders not in EMAIL_ALLOWED_USERS — prevents the adapter
            # from creating a MessageEvent (and thus thread context) for senders
            # that the gateway will never authorize.  Without this early guard,
            # a race between dispatch and authorization can result in the adapter
            # sending a reply even though the handler returned None.
            allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
            if not allowed_raw:
                if _get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() not in {"true", "1", "yes"} and (
                    os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() not in {"true", "1", "yes"}
                ):
                    logger.debug(
                        "[Email] Dropping sender at dispatch — EMAIL_ALLOWED_USERS is unset "
                        "and open access is not opted in: %s",
                        sender_addr,
                    )
                    return
            else:
                allowed = {addr.strip().lower() for addr in allowed_raw.split(",") if addr.strip()}
                if sender_addr.lower() not in allowed:
                    logger.debug("[Email] Dropping non-allowlisted sender at dispatch: %s", sender_addr)
                    return

            # Reject spoofed senders. The allowlist (and the gateway's own authz)
            # key on sender_addr, which comes straight from the attacker-controlled
            # From: header — so an attacker can forge From: an-allowlisted@addr to
            # get authorized (GHSA-rxqh-5572-8m77). This only matters when an
            # allowlist is actually being used to GRANT access: if no allowlist is
            # configured the gateway default-denies everyone anyway, and if allow-all
            # is on the operator already accepts any sender. So enforce From:
            # authentication exactly when an allowlist is in effect and allow-all is
            # off. Fail-closed: an unauthenticated From: is dropped before it can be
            # matched against the allowlist.
            if (
                self._require_authenticated_sender
                and self._allowlist_in_effect()
                and not self._allow_all_senders()
                and not msg_data.get("sender_authenticated", False)
            ):
                logger.warning(
                    "[Email] Dropping sender with unauthenticated From: %s (%s). "
                    "If your mail server does not stamp Authentication-Results, set "
                    "platforms.email.require_authenticated_sender: false (or "
                    "EMAIL_TRUST_FROM_HEADER=true) to accept the risk.",
                    sender_addr,
                    msg_data.get("auth_reason", "no verdict"),
                )
                return

            subject = msg_data["subject"]
            body = msg_data["body"].strip()
            attachments = msg_data["attachments"]

            # Build message text: include subject as context
            text = body
            if subject and not subject.startswith("Re:"):
                text = f"[Subject: {subject}]\n\n{body}"

            # Determine message type and media
            media_urls = []
            media_types = []
            msg_type = MessageType.TEXT

            for att in attachments:
                media_urls.append(att["path"])
                media_types.append(att["media_type"])
                if att["type"] == "image" and msg_type == MessageType.TEXT:
                    msg_type = MessageType.PHOTO
                elif att["type"] == "document":
                    # Document wins over PHOTO for mixed attachments: run.py's
                    # image handling keys off the per-path image/* mime type
                    # regardless of message_type, but document-context injection
                    # gates strictly on MessageType.DOCUMENT — so DOCUMENT is the
                    # only classification that surfaces both.
                    msg_type = MessageType.DOCUMENT

            # Store thread context for reply threading
            self._thread_context[sender_addr] = {
                "subject": subject,
                "message_id": message_id,
            }

            source = self.build_source(
                chat_id=sender_addr,
                chat_name=msg_data["sender_name"] or sender_addr,
                chat_type="dm",
                user_id=sender_addr,
                user_name=msg_data["sender_name"] or sender_addr,
            )

            event = MessageEvent(
                text=text or "(empty email)",
                message_type=msg_type,
                source=source,
                message_id=message_id,
                media_urls=media_urls,
                media_types=media_types,
                reply_to_message_id=msg_data["in_reply_to"] or None,
            )

            logger.info("[Email] New message from %s: %s", sender_addr, subject)
            await self.handle_message(event)
        finally:
            # [PATCH-6] Always advance the mail out of its current folder to
            # Done — on a successful reply, an early drop (self / automated /
            # non-allowlisted / unauthenticated sender), OR a handle_message
            # exception. Without this, mail that _fetch_new_messages already
            # moved into Working would be stranded there on any non-success path.
            # _finalize_message is a no-op when done_folder is unset, when the
            # mail never moved, or when there is no Message-ID to re-locate it by.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._finalize_message,
                message_id,
                source_folder,
            )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an email reply to the given address."""
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(None, self._send_email, chat_id, content, reply_to)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[Email] Send failed to %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    def _message_id_domain(self) -> str:
        """Domain part for generated Message-IDs.

        EMAIL_ADDRESS may lack an ``@`` (misconfiguration); fall back to
        ``localhost`` instead of crashing send with an IndexError.
        """
        if "@" in self._address:
            return self._address.rsplit("@", 1)[-1] or "localhost"
        return "localhost"

    def _send_email(
        self,
        to_addr: str,
        body: str,
        reply_to_msg_id: Optional[str] = None,
    ) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        # Thread context for reply
        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        # Threading headers
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        msg["Message-ID"] = msg_id

        msg.attach(MIMEText(body, "plain", "utf-8"))

        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        # [PATCH-7] Mirror outbound mail to Sent (no-op when configured empty).
        self._append_to_sent(msg.as_bytes())
        return msg_id

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Email has no typing indicator — no-op."""

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image URL as part of an email body.

        ``metadata`` is accepted to honor the base-class contract; the
        email body send doesn't use it.
        """
        text = caption or ""
        text += f"\n\nImage: {image_url}"
        return await self.send(chat_id, text.strip(), reply_to)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images as a single email with multiple MIME attachments.

        Local files are attached directly. URL images have their URL
        appended to the body (email adapter does not download remote
        images). No hard cap — email clients handle dozens of
        attachments fine, subject to SMTP message size limits.
        """
        if not images:
            return

        from urllib.parse import unquote as _unquote

        body_parts: List[str] = []
        local_paths: List[str] = []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if image_url.startswith("file://"):
                local_path = _unquote(image_url[7:])
                if Path(local_path).exists():
                    local_paths.append(local_path)
                else:
                    logger.warning("[Email] Skipping missing image: %s", local_path)
            else:
                # Remote URLs just get linked in the body (parity with send_image)
                body_parts.append(f"Image: {image_url}")

        if not local_paths and not body_parts:
            return

        body = "\n\n".join(body_parts)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._send_email_with_attachments,
                chat_id,
                body,
                local_paths,
            )
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    def _send_email_with_attachments(
        self,
        to_addr: str,
        body: str,
        file_paths: List[str],
    ) -> str:
        """Send an email with multiple file attachments via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        msg["Message-ID"] = msg_id

        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        for file_path in file_paths:
            p = Path(file_path)
            try:
                with open(p, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={p.name}")
                    msg.attach(part)
            except Exception as e:
                logger.warning("[Email] Failed to attach %s: %s", file_path, e)

        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
        # [PATCH-7] Mirror outbound mail to Sent (no-op when configured empty).
        self._append_to_sent(msg.as_bytes())
        return msg_id

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file as an email attachment."""
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None,
                self._send_email_with_attachment,
                chat_id,
                caption or "",
                file_path,
                file_name,
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[Email] Send document failed: %s", e)
            return SendResult(success=False, error=str(e))

    def _send_email_with_attachment(
        self,
        to_addr: str,
        body: str,
        file_path: str,
        file_name: Optional[str] = None,
    ) -> str:
        """Send an email with a file attachment via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        msg["Message-ID"] = msg_id

        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attach file
        p = Path(file_path)
        fname = file_name or p.name
        with open(p, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={fname}")
            msg.attach(part)

        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

        # [PATCH-7] Mirror outbound mail to Sent (no-op when configured empty).
        self._append_to_sent(msg.as_bytes())
        return msg_id

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the email chat."""
        ctx = self._thread_context.get(chat_id, {})
        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "subject": ctx.get("subject", ""),
        }


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the Email adapter moved from gateway/platforms/email.py into this
# bundled plugin. register() exposes the platform via the registry, replacing
# the Platform.EMAIL elif in gateway/run.py, the _PLATFORM_CONNECTED_CHECKERS
# entry in gateway/config.py, the _PLATFORMS["email"] static dict in
# hermes_cli/gateway.py, and the _send_email dispatch in
# tools/send_message_tool.py. EMAIL_* env→PlatformConfig seeding stays in core.
# ──────────────────────────────────────────────────────────────────────────


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process Email delivery via SMTP (one-shot). Implements the
    standalone_sender_fn contract; replaces the legacy _send_email helper."""
    import smtplib
    import ssl as _ssl
    from email.mime.text import MIMEText
    from email.utils import formatdate

    extra = getattr(pconfig, "extra", {}) or {}
    address = extra.get("address") or _get_secret("EMAIL_ADDRESS", "")
    password = _get_secret("EMAIL_PASSWORD", "")
    smtp_host = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", "")
    try:
        smtp_port = int(_get_secret("EMAIL_SMTP_PORT", "587") or "587")
    except (ValueError, TypeError):
        smtp_port = 587

    # [PATCH-8] IMAP Sent-folder archival config (parity with EmailAdapter and
    # upstream PR #28697's standalone-path review fix). The standalone path only
    # requires SMTP, so IMAP may be unconfigured — the shared helper no-ops on a
    # missing host or an empty folder. Port-based SSL/STARTTLS is handled inside
    # the helper via _open_imap_conn, so imap_tls on 143 keeps working here too.
    imap_host = extra.get("imap_host") or _get_secret("EMAIL_IMAP_HOST", "")
    try:
        imap_port = int(_get_secret("EMAIL_IMAP_PORT", "993") or "993")
    except (ValueError, TypeError):
        imap_port = 993
    # Empty string is a deliberate opt-out — do NOT collapse with `or`.
    sent_folder = extra.get("sent_folder", "Sent")

    if not all([address, password, smtp_host]):
        return {"error": "Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)"}

    try:
        msg = MIMEText(message, "plain", "utf-8")
        msg["From"] = address
        msg["To"] = chat_id
        msg["Subject"] = "Hermes Agent"
        msg["Date"] = formatdate(localtime=True)

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls(context=_ssl.create_default_context())
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        # [PATCH-8] Best-effort Sent-folder mirror — never fails the
        # (already-completed) SMTP send. Shares the archival helper (and thus
        # the port-based SSL/STARTTLS logic) with EmailAdapter.
        _imap_append_to_sent(
            imap_host=imap_host,
            imap_port=imap_port,
            address=address,
            password=password,
            sent_folder=sent_folder,
            raw_bytes=msg.as_bytes(),
        )
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as e:
        try:
            from tools.send_message_tool import _error as _e

            return _e(f"Email send failed: {e}")
        except Exception:
            return {"error": f"Email send failed: {e}"}


def _is_connected(config) -> bool:
    """Email is connected when an address is configured (in PlatformConfig.extra
    or via EMAIL_ADDRESS). Mirrors the legacy
    _PLATFORM_CONNECTED_CHECKERS[Platform.EMAIL] = bool(extra.get('address'))."""
    extra = getattr(config, "extra", {}) or {}
    if extra.get("address"):
        return True
    import hermes_cli.gateway as gateway_mod

    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs EmailAdapter from a PlatformConfig."""
    return EmailAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="email",
        label="Email",
        adapter_factory=_build_adapter,
        check_fn=check_email_requirements,
        is_connected=_is_connected,
        required_env=["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"],
        install_hint="Email uses the Python stdlib (smtplib/imaplib) — no extra deps",
        allowed_users_env="EMAIL_ALLOWED_USERS",
        allow_all_env="EMAIL_ALLOW_ALL_USERS",
        cron_deliver_env_var="EMAIL_HOME_ADDRESS",
        standalone_sender_fn=_standalone_send,
        max_message_length=50_000,
        pii_safe=True,
        emoji="📧",
        allow_update_command=True,
    )
