"""Email platform adapter for the Hermes gateway: users talk to Hermes by sending email; IMAP (polled)
receives, SMTP sends. Configured via EMAIL_* env vars or ``platforms.email`` in config.yaml (see website docs).

------------------------------------------------------------------------------
LOCAL PATCH (dgxarley) — synced to upstream tag v2026.9.24
(plugins/platforms/email/adapter.py, 50705 bytes, blob 7d68f2dfbe). Must move in
lockstep with hermes.image_tag: the baseline calls base-class methods that only
exist from its own tag on. Re-sync procedure: HERMES_EMAIL_UPSTREAM.md.

Adds four behaviours that upstream lacks:

  1.  Two-stage IMAP folder lifecycle:
        INBOX  -- fetch -->  Hermes_Working  -- handle_message() done -->  Hermes_Done
      so anything left in Hermes_Working after a crash is visibly "interrupted".
  2.  Sent-mail archival via IMAP APPEND (upstream only sends via SMTP).
  3.  Opt-in processing of pre-existing INBOX mail on startup (upstream marks
      everything already there as seen).
  4.  Opt-in quoting of the received email below the turn's final reply to it.

Knobs live in config.yaml under an explicit ``extra:`` block (an explicit
``extra:`` value wins over a bare platform key). The two quote knobs also accept
an env override (EMAIL_QUOTE_ORIGINAL / EMAIL_QUOTE_MAX_CHARS win):

  platforms.email.extra.working_folder     default "Hermes_Working"  ("" → INBOX→Done)
  platforms.email.extra.done_folder        default "Hermes_Done"     ("" → no moves at all)
  platforms.email.extra.sent_folder        default "Sent"            ("" → no APPEND)
  platforms.email.extra.process_existing   default true              (upstream PR: false)
  platforms.email.extra.quote_original     default false
  platforms.email.extra.quote_max_chars    default 10000             (quote body only)
  platforms.email.extra.quote_header       default "On {date}, {name} <{address}> wrote:"

Patch sections (grep for the markers):

  [PATCH-1] module docstring (this block)
  [PATCH-2] __init__: new self._* attributes (+ lifecycle log line)
  [PATCH-3] helpers: module-level _open_imap_conn (twin of upstream's
            _connect_imap) + _imap_append_to_sent; EmailAdapter._ensure_folder,
            _imap_move, _search_message_id, _finalize_message, _append_to_sent
  [PATCH-4] _probe_imap(): folder ensure + process_existing ``elif`` between
            upstream's reconnect-restore and mark-all-seen branches
  [PATCH-5] _fetch_new_messages(): INBOX→Working MOVE after
            _parse_fetched_message(), only if done_folder AND working_folder AND
            a Message-ID are present
  [PATCH-6] _dispatch_message(): try/finally around _sender_accepted + event
            build + handle_message, so every drop path finalizes out of Working
  [PATCH-7] _smtp_send(): APPEND to Sent (single sink for all adapter sends)
  [PATCH-8] _standalone_send(): APPEND to Sent via the shared helper (cron /
            `hermes send` path)
  [PATCH-9] retired (charset fix, native upstream since v2026.8.13)
  [PATCH-10] _imap_default_security(): port-derived IMAP security default
            (993 → tls, else starttls); upstream defaults to tls on every port,
            which breaks port-143 mailboxes. An explicit setting still wins.
  [PATCH-11] quoting: _format_quote_date() / _build_quote(); _remember_original()
            inside the [PATCH-6] try, only for msg_data["sender_granted"] (else
            _forget_sender_original()); _claim_quote() quotes only final-reply
            sends (metadata["notify"]), _claim_quote()/_settle_quote() wrap the
            single _smtp_send() call site. _standalone_send never quotes.

dgxarley-only deviations from upstream code paths: our own IMAP teardowns go
through _close_imap() (same fd-leak class as upstream #79889), and
_standalone_send's IMAP reads use the profile-scoped _get_secret().

Upstreaming (all OPEN; drop the matching sections once merged):
  - #28697 Sent-folder APPEND          [PATCH-3/7/8]
  - #28699 process_existing            [PATCH-4]
  - #28702 Working/Done lifecycle      [PATCH-3/4/5/6]
  - #113192 quote original             [PATCH-11]
[PATCH-10] stays dgxarley-only (per-user ``imap_security: starttls`` would make
it unnecessary).
------------------------------------------------------------------------------
"""

import asyncio
import email as email_lib
from contextlib import contextmanager, suppress
import imaplib
import logging
import os
import re
import smtplib
import socket
import ssl
import threading  # [PATCH-11] not imported upstream; _quote_lock guards claim/settle across executor threads
import time  # dgxarley: not imported upstream; _imap_append_to_sent needs time.time()
import uuid
from collections import OrderedDict  # [PATCH-11] not imported upstream; bounded per-Message-ID quote lookup
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate, parsedate_to_datetime
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.platforms.helpers import cancel_task
from gateway.platforms.event import MessageEvent, MessageType
from gateway.config import Platform, PlatformConfig
from utils import is_truthy_value
from gateway.platforms._shared import (
    get_scoped_secret as _get_secret,
    coerce_port,
    decode_json_list_literal,
    send_error,
)

logger = logging.getLogger(__name__)

_SECURITY_ALIASES = {
    "tls": "tls",
    "ssl": "tls",
    "implicit": "tls",
    "starttls": "starttls",
    "plain": "plain",
    "none": "plain",
}
# Automated senders (address substrings / bulk-mail headers) are silently ignored.
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
_AUTOMATED_HEADERS = {
    "Auto-Submitted": lambda v: v.lower() != "no",
    "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
    "X-Auto-Response-Suppress": lambda v: bool(v),
    "List-Unsubscribe": lambda v: bool(v),
}
MAX_MESSAGE_LENGTH = 50_000  # Gmail-safe max length per email body
SMTP_CONNECT_TIMEOUT = 30
_TRUTHY = {"true", "1", "yes"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
# Charset labels seen in the wild that Python's codec registry doesn't know: "unknown-8bit"/"x-unknown" are
# RFC 1428 placeholders (QQ Mail emits them); gb2312/gbk map to the gb18030 superset so GBK extensions decode.
_CHARSET_ALIASES = {
    "unknown-8bit": "utf-8",
    "unknown": "utf-8",
    "x-unknown": "utf-8",
    "default": "utf-8",
    "ansi_x3.110-1983": "latin-1",
    "cp-850": "cp850",
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "ks_c_5601-1987": "cp949",
}
# Ordered (pattern, replacement) substitutions for _strip_html.
_HTML_SUBS = (
    (re.compile(r"<br\s*/?>", re.IGNORECASE), "\n"),
    (re.compile(r"<p[^>]*>", re.IGNORECASE), "\n"),
    (re.compile(r"</p>", re.IGNORECASE), "\n"),
    (re.compile(r"<[^>]+>"), ""),
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"\n{3,}"), "\n\n"),
)
# "method=result" tokens (``dmarc=pass``) and property values (``header.from=x``) in Authentication-Results.
_AUTH_METHOD_RE = re.compile(r"\b(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE)
_AUTH_PROP_RE = re.compile(
    r"\b(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*([^\s;]+)", re.IGNORECASE
)
# [PATCH-11] Quoting the inbound mail under replies (opt-in, platforms.email.extra.quote_original / EMAIL_QUOTE_ORIGINAL).
_DEFAULT_QUOTE_HEADER = "On {date}, {name} <{address}> wrote:"
_DEFAULT_QUOTE_MAX_CHARS = 10_000
_QUOTE_TRUNCATED = "> [...]"
_QUOTE_LOOKUP_MAX = 500  # remembered originals (oldest evicted first)


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read."""
    return coerce_port(str(_get_secret(name, "")).strip() or default, default)


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read."""
    return is_truthy_value(raw, default=default) if (raw := str(_get_secret(name, "")).strip()) else default


def _normalize_security(value: Any, default: str = "tls") -> str:
    """Map to ``tls`` | ``starttls`` | ``plain``; unknown values warn and fall back to *default* (a typo never downgrades to plaintext)."""
    raw = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if raw and raw not in _SECURITY_ALIASES:
        logger.warning("Unknown email security mode %r; using %r", value, default)
    return _SECURITY_ALIASES.get(raw, default)


def _tls_context(verify: bool, host: str) -> ssl.SSLContext:
    """Verified context by default; unverified only when explicitly opted out."""
    if verify:
        return ssl.create_default_context()
    if host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("TLS verification disabled for non-loopback host %s", host)
    return ssl._create_unverified_context()


def _close_imap(imap: "imaplib.IMAP4") -> None:
    """Teardown that guarantees the socket closes: ``logout()`` only guards ``OSError``, so ``IMAP4.abort`` on a
    broken connection skipped ``shutdown()`` and leaked one fd per failed poll (fatal on macOS's 256 soft limit).

    ``IMAP4.logout()`` only guards against ``OSError`` internally: a broken connection makes
    ``_simple_command('LOGOUT')`` raise ``IMAP4.abort`` (which is *not* an ``OSError``), so ``logout()``
    propagates before its own ``shutdown()`` call and the TCP socket stays open. On macOS, where the default
    soft fd limit is 256 and pollers may run through a local proxy, these abandoned sockets accumulate one
    per failed poll until the gateway hits ``[Errno 24] Too many open files`` (#79889).
    """
    try:
        imap.logout()
    except Exception:
        with suppress(Exception):
            imap.shutdown()


def _create_ipv4_connection(host: str, port: int, timeout: float, source_address: Any = None) -> socket.socket:
    """``socket.create_connection`` constrained to ``AF_INET`` (no process-global socket mutation — sends run in executor threads)."""
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
    raise last_error if last_error is not None else OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(host, port, timeout, source_address=self.source_address)


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return self.context.wrap_socket(
            _create_ipv4_connection(host, port, timeout, source_address=self.source_address),
            server_hostname=getattr(self, "_host", host),
        )


def _open_smtp(
    host: str, port: int, security: str, ctx: ssl.SSLContext, smtp_cls: type, smtp_ssl_cls: type, **kwargs: Any
) -> smtplib.SMTP:
    """Open one SMTP connection with TLS established per *security*; *kwargs* go to the constructor."""
    if security == "tls":
        return smtp_ssl_cls(host, port, context=ctx, **kwargs)
    smtp = smtp_cls(host, port, **kwargs)
    if security == "starttls":
        try:
            smtp.starttls(context=ctx)
        except Exception:
            smtp.close()
            raise
    return smtp


def _send_imap_id(imap: "imaplib.IMAP4") -> None:
    """Send RFC 2971 IMAP ID: 163/NetEase require it after LOGIN (else every UID command
    returns ``BYE Unsafe Login``); other servers may reject it, so failures are swallowed.

    Sent only when the server advertises ``ID`` (RFC 2971 requires advertising it): a server
    without the extension can answer an untagged ``* BYE Unknown command.`` and close the
    connection, which imaplib cannot surface here — the failure appears one command later
    as a misleading SELECT error and the adapter retries forever (Purelymail, #39856).
    ``imap.capabilities`` is populated by imaplib at connect, so the check is free."""
    if "ID" not in imap.capabilities:
        logger.debug("[Email] Server does not advertise IMAP ID capability; skipping ID")
        return
    try:
        try:
            from hermes_cli import __version__ as _hermes_version
        except Exception:  # noqa: BLE001 — keep ID best-effort if import fails
            _hermes_version = "0"
        imap.xatom(
            "ID",
            f'("name" "hermes-agent" "version" "{_hermes_version}" '
            '"vendor" "NousResearch" "support-email" "noreply@nousresearch.com")',
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


def _imap_default_security(imap_port: int) -> str:
    """[PATCH-10] Port-derived IMAP security default when none is configured.

    Upstream defaults ``imap_security`` to ``tls`` regardless of the port, which
    breaks every existing 143/STARTTLS setup that never had to set the key (the
    server greets in plaintext, so the implicit-TLS handshake dies with
    ``[SSL: RECORD_LAYER_FAILURE]``). Mirror upstream's own SMTP rule instead
    (465 → ``tls``, else ``starttls``): 993 → ``tls``, any other port →
    ``starttls``. An explicit EMAIL_IMAP_SECURITY / ``imap_security`` still wins.
    This keeps the port-based behaviour our former ``_open_imap_conn`` had.
    """
    return "tls" if imap_port == 993 else "starttls"


def _open_imap_conn(host: str, port: int, security: str, tls_verify: bool) -> imaplib.IMAP4:
    """[PATCH-3] Unauthenticated IMAP connection (tls / starttls / plain).

    Module-level twin of upstream's ``EmailAdapter._connect_imap`` (same
    branches, same timeout) so the out-of-process ``_standalone_send`` Sent
    APPEND opens IMAP over the SAME transport rules as the live adapter.
    Kept as a copy rather than rewiring ``_connect_imap`` so the upstream
    method stays byte-identical for future re-syncs.
    """
    if security == "tls":
        return imaplib.IMAP4_SSL(host, port, timeout=30, ssl_context=_tls_context(tls_verify, host))
    imap = imaplib.IMAP4(host, port, timeout=30)
    if security == "starttls":
        try:
            imap.starttls(ssl_context=_tls_context(tls_verify, host))
        except Exception:
            _close_imap(imap)
            raise
    return imap


def _imap_append_to_sent(
    *,
    imap_host: str,
    imap_port: int,
    imap_security: str,
    imap_tls_verify: bool,
    address: str,
    password: str,
    sent_folder: str,
    raw_bytes: bytes,
) -> None:
    """[PATCH-3] IMAP-APPEND a freshly-sent outbound mail to ``sent_folder``.

    No-op when the folder is unset (empty string) or no IMAP host is
    configured. Best-effort: failures are logged as warnings and never
    re-raised — losing the Sent-folder copy must NOT roll back an SMTP send
    that already succeeded.

    Shared by the live :meth:`EmailAdapter._append_to_sent` and the
    out-of-process ``_standalone_send`` so both SMTP paths archive identically
    (parity with upstream PR #28697's review-driven refactor).
    """
    if not sent_folder or not imap_host:
        return
    try:
        imap = _open_imap_conn(imap_host, imap_port, imap_security, imap_tls_verify)
        try:
            imap.login(address, password)
            _send_imap_id(imap)
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
                logger.warning("[Email] APPEND to %r returned %s: %s", sent_folder, typ, detail)
            else:
                logger.debug("[Email] APPEND to %r ok", sent_folder)
        finally:
            _close_imap(imap)
    except Exception as e:  # noqa: BLE001 — Sent-folder mirror is best-effort
        logger.warning("[Email] APPEND to %r failed: %s", sent_folder, e)


def _is_automated_sender(address: str, headers: dict) -> bool:
    """True if this email is from an automated/noreply source."""
    addr = address.lower()
    return any(pattern in addr for pattern in _NOREPLY_PATTERNS) or any(
        (value := headers.get(header, "")) and check(value) for header, check in _AUTOMATED_HEADERS.items()
    )


def check_email_requirements() -> bool:
    """True when all email settings are present and non-blank (blank keys left by an abandoned setup must not enable the platform).

    Treats blank/whitespace-only values as missing so an abandoned setup that left empty ``EMAIL_*`` keys in
    ``.env`` does not enable the platform (#40715).
    """
    return all(
        _get_secret(name, "").strip()
        for name in ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST")
    )


def _safe_decode(payload: bytes, charset: "Optional[str]") -> str:
    """Decode without ever raising: ``errors="replace"`` does not guard a missing codec (``LookupError``), so fall back alias → UTF-8 → latin-1.

    Unknown or malformed charset labels (``unknown-8bit``, misspelled names, attacker-controlled garbage)
    previously raised ``LookupError`` from ``bytes.decode`` — ``errors="replace"`` only guards decode
    errors, not a missing codec — which aborted the whole IMAP fetch and dropped every message in the batch
    (#35901, #55381, #55383). Fall back through a small alias table, then UTF-8, then latin-1 (which never
    fails).
    """
    label = (charset or "utf-8").strip().strip("\"'").lower() or "utf-8"
    for candidate in (_CHARSET_ALIASES.get(label, label), "utf-8"):
        try:
            return payload.decode(candidate, errors="replace")
        except (LookupError, ValueError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 header into a plain string; never raises.

    Never raises: malformed encoded-words or unknown charsets degrade to replacement characters instead of
    crashing the fetch loop (#55381).
    """
    try:
        parts = decode_header(raw)
    except Exception:  # malformed RFC 2047 structure
        return raw
    return " ".join(_safe_decode(part, charset) if isinstance(part, bytes) else part for part, charset in parts)


def _first_body_part(msg: email_lib.message.Message, content_type: str) -> str:
    """Decoded text of the first non-attachment part of *content_type*, or ''."""
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")) or part.get_content_type() != content_type:
            continue
        if payload := part.get_payload(decode=True):
            return _safe_decode(payload, part.get_content_charset())
    return ""


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        html = _first_body_part(msg, "text/html")
        return _first_body_part(msg, "text/plain") or (_strip_html(html) if html else "")
    text = _safe_decode(payload, msg.get_content_charset()) if (payload := msg.get_payload(decode=True)) else ""
    return _strip_html(text) if msg.get_content_type() == "text/html" else text


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    for pattern, repl in _HTML_SUBS:
        html = pattern.sub(repl, html)
    return html.strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <addr>' format."""
    match = re.search(r"<([^>]+)>", raw)
    return (match.group(1) if match else raw).strip().lower()


def _domain_of(address: str) -> str:
    """Lowercased domain part of an email address, or ''."""
    return address.rpartition("@")[2].strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Relaxed DMARC alignment: equal, or one is a dot-suffix of the other."""
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    return bool(a and b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def _verify_sender_authentication(
    msg: email_lib.message.Message, from_addr: str, *, authserv_id: str = ""
) -> Tuple[bool, str]:
    """Verify the ``From:`` domain is authenticated; returns ``(authenticated, reason)``.
    ``From:`` is attacker-controlled (GHSA-rxqh-5572-8m77); the only trustworthy signal is the
    ``Authentication-Results`` header stamped by the *receiving* server. It prepends, so the FIRST
    instance is trusted and an injected copy sorts below it; pinned to *authserv_id* when given.
    True on DMARC pass, aligned SPF pass, or aligned DKIM (``header.d``) pass. No header → fail-closed
    (opt out via ``EmailAdapter._require_authenticated_sender``)."""
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"
    if not (headers := msg.get_all("Authentication-Results")):
        return False, "no Authentication-Results header"
    values = (" ".join(str(raw).split()) for raw in headers)  # authserv-id precedes the first ';'
    trusted = next(
        (
            v
            for v in values
            if not authserv_id
            or (serv := v.split(";", 1)[0].strip().lower()) == authserv_id.lower()
            or _domains_aligned(serv, authserv_id)
        ),
        None,
    )
    if trusted is None:
        return False, "no Authentication-Results from trusted authserv-id"
    methods = {m.lower(): r.lower() for m, r in _AUTH_METHOD_RE.findall(trusted)}
    props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(trusted)}
    if methods.get("dmarc") == "pass":  # DMARC already enforces From alignment
        return True, "dmarc=pass"
    if methods.get("spf") == "pass":  # envelope/MAIL FROM domain must align with From
        spf_domain = (
            _domain_of(props.get("smtp.mailfrom", "")) or props.get("smtp.from", "") or props.get("envelope-from", "")
        )
        if _domains_aligned(_domain_of(spf_domain) if "@" in spf_domain else spf_domain, from_domain):
            return True, "spf=pass aligned"
    if methods.get("dkim") == "pass":  # signing domain header.d must align with From
        dkim_domain = props.get("header.d", "") or _domain_of(props.get("header.from", ""))
        if _domains_aligned(dkim_domain, from_domain):
            return True, "dkim=pass aligned"
    return False, f"authentication failed ({trusted[:120]})"


def _extract_attachments(msg: email_lib.message.Message, skip_attachments: bool = False) -> List[Dict[str, Any]]:
    """Extract attachment metadata and cache files locally (nothing when *skip_attachments*)."""
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        disposition, content_type = str(part.get("Content-Disposition", "")), part.get_content_type()
        if skip_attachments or (
            "attachment" not in disposition
            and ("inline" not in disposition or content_type in {"text/plain", "text/html"})
        ):
            continue  # not an attachment, or an inline text/html body part
        filename = (
            _decode_header_value(fn)
            if (fn := part.get_filename())
            else f"attachment.{part.get_content_subtype() or 'bin'}"
        )
        if not (payload := part.get_payload(decode=True)):
            continue
        if (ext := Path(filename).suffix.lower()) in _IMAGE_EXTS:
            try:
                cached_path, kind = cache_image_from_bytes(payload, ext), "image"
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
        else:
            cached_path, kind = cache_document_from_bytes(payload, filename), "document"
        attachments.append({"path": cached_path, "filename": filename, "type": kind, "media_type": content_type})
    return attachments


def _format_quote_date(raw: str) -> str:
    """[PATCH-11] Human-readable ``Date:`` value; the raw header when unparseable, '' when missing."""
    if not (raw := (raw or "").strip()):
        return ""
    try:
        return parsedate_to_datetime(raw).strftime("%a, %d %b %Y %H:%M %z").strip()
    except (TypeError, ValueError, IndexError, OverflowError):
        return raw


def _build_quote(original_body: str, *, date: str, name: str, address: str, header_fmt: str, max_chars: int) -> str:
    """[PATCH-11] Classic ``> `` quote block for appending under a reply: ``"\\n\\n" + header + "\\n" + quoted lines``.

    *max_chars* bounds the quoted lines (prefixes included, header excluded); the cut lands on a line end and is
    marked with ``> [...]``. Already-quoted lines just gain another level (``>> ``). Empty body (or no line fits) → ''.
    """
    lines = [line.rstrip() for line in (original_body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    if not lines or max_chars <= 0:
        return ""
    quoted = [f">{line}" if line.startswith(">") else f"> {line}" if line else ">" for line in lines]
    text = "\n".join(quoted)
    if len(text) > max_chars:
        kept: List[str] = []
        used = len(_QUOTE_TRUNCATED)
        for line in quoted:
            if used + len(line) + 1 > max_chars:
                break
            kept.append(line)
            used += len(line) + 1
        if not kept:
            return ""
        text = "\n".join([*kept, _QUOTE_TRUNCATED])
    shown_date, shown_name = _format_quote_date(date), (name or "").strip() or address
    fmt = header_fmt or _DEFAULT_QUOTE_HEADER
    if not shown_date:  # "On {date}, X wrote:" -> "X wrote:"
        fmt = re.sub(r"On\s*\{date\},?\s*", "", fmt).replace("{date}", "").strip()
    try:
        header = fmt.format(date=shown_date, name=shown_name, address=address)
    except (KeyError, IndexError, ValueError):  # operator typo in quote_header must not break sending
        header = (
            _DEFAULT_QUOTE_HEADER.format(date=shown_date, name=shown_name, address=address)
            if shown_date
            else f"{shown_name} <{address}> wrote:"
        )
    return f"\n\n{header}\n{text}"


def _attach_file(msg: MIMEMultipart, path: Path, filename: str) -> None:
    """Attach *path* to *msg* as base64 application/octet-stream."""
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    # Per-account seen-UID snapshot surviving adapter recreation: the reconnect watcher builds a FRESH
    # adapter per retry; without this connect(is_reconnect=True) would re-mark the mailbox seen and skip
    # mail that arrived during the outage. Keyed by address (multiplex runs several accounts); same-process only.
    _seen_uids_snapshot: Dict[str, set] = {}

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)
        # Env first, then PlatformConfig.extra (config.yaml-only setups). Host/address are stripped: a stray
        # newline made IMAP4_SSL raise ``[Errno 8] nodename nor servname`` instead of "host not set".
        extra = config.extra or {}
        setting = lambda env, key: _get_secret(env, "") or extra.get(key, "")  # noqa: E731
        tls_verify = lambda env, key: _esecret_bool(env, is_truthy_value(extra.get(key), default=True))  # noqa: E731
        self._address = setting("EMAIL_ADDRESS", "address").strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = setting("EMAIL_IMAP_HOST", "imap_host").strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        # [PATCH-10] default derived from the port (upstream: always "tls"), see _imap_default_security.
        self._imap_security = _normalize_security(
            setting("EMAIL_IMAP_SECURITY", "imap_security"), default=_imap_default_security(self._imap_port)
        )
        self._imap_tls_verify = tls_verify("EMAIL_IMAP_TLS_VERIFY", "imap_tls_verify")
        self._smtp_host = setting("EMAIL_SMTP_HOST", "smtp_host").strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._smtp_security = _normalize_security(
            setting("EMAIL_SMTP_SECURITY", "smtp_security"), default="tls" if self._smtp_port == 465 else "starttls"
        )
        self._smtp_tls_verify = tls_verify("EMAIL_SMTP_TLS_VERIFY", "smtp_tls_verify")
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)
        self._skip_attachments = extra.get("skip_attachments", False)  # platforms.email.skip_attachments
        # [PATCH-2] dgxarley: folder-lifecycle + APPEND-to-Sent + process-existing,
        # all configured via config.yaml ``platforms.email.extra.*``:
        #   platforms:
        #     email:
        #       extra:
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
        # Require an authenticated From: domain (SPF/DKIM/DMARC) before trusting it for authorization
        # (GHSA-rxqh-5572-8m77). Default ON; opt out via require_authenticated_sender: false / EMAIL_TRUST_FROM_HEADER=true.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        else:
            self._require_authenticated_sender = not _esecret_bool("EMAIL_TRUST_FROM_HEADER", False)
        # Optional authserv-id pinning Authentication-Results to the operator's own server (defeats an injected header sorting first).
        self._authserv_id = (extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")).strip().lower()
        # [PATCH-11] Quote the inbound mail under the turn's final reply to it (default off; env wins
        # over config.yaml, same precedence as every other EMAIL_*/extra pair above).
        self._quote_original = _esecret_bool(
            "EMAIL_QUOTE_ORIGINAL", is_truthy_value(extra.get("quote_original"), default=False)
        )
        self._quote_max_chars = _esecret_int(
            "EMAIL_QUOTE_MAX_CHARS", coerce_port(extra.get("quote_max_chars"), _DEFAULT_QUOTE_MAX_CHARS)
        )
        self._quote_header = str(extra.get("quote_header") or _DEFAULT_QUOTE_HEADER)
        # Originals by Message-ID so parallel mails from one sender quote the right one; the per-sender entry
        # covers mails without a Message-ID. Only populated while quoting is on.
        self._original_by_msg_id: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
        self._last_original_by_sender: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
        self._quote_lock = threading.Lock()  # sends run concurrently in executor threads
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000  # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        self._last_fetch_failed, self._last_fetch_error = (
            False,
            "",
        )  # "checked, nothing new" vs "the check itself failed"
        # chat_id (sender email) -> last subject + message-id for threading
        # Track the last IMAP fetch attempt so the poll loop can distinguish "checked, nothing new" from
        # "the check itself failed" (#80016).
        self._thread_context: Dict[str, Dict[str, str]] = {}
        logger.info("[Email] Adapter initialized for %s", self._address)
        logger.info(
            "[Email] Folder lifecycle: working=%r done=%r sent=%r process_existing=%s imap_security=%s",
            self._working_folder,
            self._done_folder,
            self._sent_folder,
            self._process_existing,
            self._imap_security,
        )

    def _trim_seen_uids(self) -> None:
        """Keep only the highest half of UIDs once over the cap (UIDs are monotonic; UNSEEN prevents re-delivery)."""
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))  # UIDs are bytes like b'1234'
            self._seen_uids = set(sorted_uids[-(self._seen_uids_max // 2) :])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2 :])

    def _connect_imap(self) -> imaplib.IMAP4:
        """Create an IMAP connection using implicit TLS, STARTTLS, or plaintext."""
        if self._imap_security == "tls":
            return imaplib.IMAP4_SSL(
                self._imap_host,
                self._imap_port,
                timeout=30,
                ssl_context=_tls_context(self._imap_tls_verify, self._imap_host),
            )
        imap = imaplib.IMAP4(self._imap_host, self._imap_port, timeout=30)
        if self._imap_security == "starttls":
            try:
                imap.starttls(ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
            except Exception:
                _close_imap(imap)
                raise
        return imap

    @contextmanager
    def _inbox(self):
        """Logged-in IMAP handle on INBOX; always ``_close_imap``-ed on exit (a login/select failure used to leak one fd per reconnect)."""
        # Test IMAP connection. The handle is closed in ``finally`` — before this, a failure in
        # login/select/search left the TCP socket open with no owner, leaking one fd per connect attempt.
        # Under the gateway's reconnect watcher (fresh adapter instance per retry) against an
        # unreachable/proxied host this grew monotonically until fd exhaustion on macOS's 256 soft limit
        # (#79889).
        imap = self._connect_imap()
        try:
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            imap.select("INBOX")
            yield imap
        finally:
            _close_imap(imap)

    def _connect_smtp(self) -> smtplib.SMTP:
        """SMTP connection with TLS established (callers go straight to ``login()``). An unreachable IPv6 address can
        hang until the socket timeout, so connection-level failures retry through an IPv4-only socket path (no global
        resolver mutation); TLS verification errors are not retried."""
        host, port, security, ctx = (
            self._smtp_host,
            self._smtp_port,
            self._smtp_security,
            _tls_context(self._smtp_tls_verify, self._smtp_host),
        )
        try:
            return _open_smtp(host, port, security, ctx, smtplib.SMTP, smtplib.SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            return _open_smtp(host, port, security, ctx, _IPv4SMTP, _IPv4SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)

    # [PATCH-3] IMAP folder-lifecycle helpers (dgxarley patch). All IMAP opens
    # go through upstream's _inbox() (login + ID + SELECT INBOX, _close_imap on
    # exit) or the module-level _open_imap_conn twin, so upstream's
    # imap_security / imap_tls_verify apply everywhere.

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
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.debug("[Email] CREATE %r ignored: %s", name, e)

    @staticmethod
    def _imap_move(imap: imaplib.IMAP4, uid: bytes, dst_folder: str) -> bool:
        """MOVE *uid* (in the currently SELECTed folder) to *dst_folder*.

        Tries the RFC 6851 ``UID MOVE`` first; falls back to
        ``UID COPY`` + ``UID STORE +FLAGS \\Deleted`` + ``EXPUNGE`` on
        servers that don't advertise MOVE. Returns True on apparent success.
        """
        try:
            status, data = imap.uid("MOVE", uid, dst_folder)
            if status == "OK":
                return True
            logger.debug("[Email] UID MOVE %s → %r: %s %s", uid, dst_folder, status, data)
        except Exception as e:  # noqa: BLE001 — fall through to COPY+EXPUNGE
            logger.debug("[Email] UID MOVE %s → %r raised: %s", uid, dst_folder, e)

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
    def _search_message_id(imap: imaplib.IMAP4, message_id: str) -> List[bytes]:
        """Return UIDs in the currently SELECTed folder whose Message-ID matches.

        Used to re-locate a mail after we MOVE'd it (UIDs change on MOVE,
        but the RFC 2822 Message-ID header is stable).
        """
        if not message_id:
            return []
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
        if not self._done_folder or self._done_folder == source_folder:
            return
        if not message_id:
            logger.debug(
                "[Email] Skipping finalize MOVE — no Message-ID for mail in %r (cannot relocate after handle_message)",
                source_folder,
            )
            return
        try:
            with self._inbox() as imap:
                imap.select(source_folder)
                uids = self._search_message_id(imap, message_id)
                if not uids:
                    logger.warning(
                        "[Email] Cannot find Message-ID %s in %r — skipping MOVE → %r",
                        message_id,
                        source_folder,
                        self._done_folder,
                    )
                    return
                for uid in uids:
                    self._imap_move(imap, uid, self._done_folder)
                logger.info("[Email] Finalized %s: %r → %r", message_id, source_folder, self._done_folder)
        except Exception as e:  # noqa: BLE001
            logger.error("[Email] Finalize MOVE failed: %s", e)

    def _append_to_sent(self, raw_bytes: bytes) -> None:
        """IMAP-APPEND an outbound mail to ``self._sent_folder`` via the shared helper (best-effort)."""
        _imap_append_to_sent(
            imap_host=self._imap_host,
            imap_port=self._imap_port,
            imap_security=self._imap_security,
            imap_tls_verify=self._imap_tls_verify,
            address=self._address,
            password=self._password,
            sent_folder=self._sent_folder,
            raw_bytes=raw_bytes,
        )

    # End of [PATCH-3] block.

    def _fail(self, log_fmt: str, err: object, code: str, detail: str, *, retryable: bool) -> bool:
        """Log *err*, record a fatal error for the gateway's reconnect machinery, return False."""
        logger.error(log_fmt, err)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    def _probe_imap(self, is_reconnect: bool) -> bool:
        """Connection test + seen-UID baseline. Sets a fatal error and returns False on failure."""
        try:
            with self._inbox() as imap:
                # [PATCH-4] Ensure our managed folders exist before any MOVE
                # touches them. CREATE is idempotent (NO on "already exists"),
                # so this is safe on every reconnect. Working/Done are only
                # created when the lifecycle is active (done_folder set); the
                # Sent folder is independent (APPEND regardless of lifecycle).
                if self._done_folder:
                    self._ensure_folder(imap, self._working_folder)
                    self._ensure_folder(imap, self._done_folder)
                self._ensure_folder(imap, self._sent_folder)
                snapshot = self._seen_uids_snapshot.get(self._address)
                if is_reconnect and snapshot is not None:
                    # Same-process reconnect: restore the previous adapter's baseline so mail that
                    # arrived during the outage stays eligible for the next poll.
                    self._seen_uids = set(snapshot)
                    passed = "[Email] IMAP reconnect test passed. Restored %d seen UIDs; messages received during the outage will be processed."
                elif self._process_existing:
                    # [PATCH-4] First connect with process_existing=true (config.yaml):
                    # leave _seen_uids empty so the next poll picks up the pre-existing
                    # UNSEEN backlog. Orthogonal to the reconnect branch above, which
                    # only fires on a same-process RECONNECT, never on a cold start.
                    passed = "[Email] IMAP connection test passed. process_existing=true — %d seen UIDs, pre-existing UNSEEN mail is processed on first poll."
                else:  # first connect (or no snapshot): mark all existing messages seen
                    status, data = imap.uid("search", None, "ALL")
                    self._seen_uids.update(data[0].split() if status == "OK" and data and data[0] else ())
                    passed = "[Email] IMAP connection test passed. %d existing messages skipped."
                self._trim_seen_uids()
                logger.info(passed, len(self._seen_uids))
            self._seen_uids_snapshot[self._address] = set(self._seen_uids)
            return True
        except Exception as e:
            # Always set an explicit fatal code, else the gateway treats every failure as transient with zero
            # owner signal. retryable=True because imaplib raises the same generic IMAP4.error for bad credentials
            # AND transient NOs (Gmail "too many simultaneous connections"); loops surface via NEEDS_ATTENTION.
            return self._fail(
                "[Email] IMAP connection failed: %s",
                e,
                "email_imap_connect_error",
                f"IMAP connection to {self._imap_host}:{self._imap_port} failed: {e}",
                retryable=True,
            )

    def _probe_smtp(self) -> bool:
        """SMTP connect + login test. Sets a fatal error and returns False on failure."""
        try:
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            # Typed auth failure (535 & friends) can never self-heal, so drop out of the reconnect queue — unambiguous, unlike IMAP4.error.
            return self._fail(
                "[Email] SMTP authentication failed: %s",
                e,
                "email_auth_error",
                f"SMTP authentication failed for {self._address}: {e}. Check EMAIL_PASSWORD (for Gmail/Outlook "
                "this must be an app password, not the account password).",
                retryable=False,
            )
        except Exception as e:
            return self._fail(
                "[Email] SMTP connection failed: %s",
                e,
                "email_smtp_connect_error",
                f"SMTP connection to {self._smtp_host} failed: {e}",
                retryable=True,
            )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host is an actionable config error, not IMAP4_SSL("") raising ``[Errno 8]``.
        required = (
            ("EMAIL_ADDRESS", self._address),
            ("EMAIL_PASSWORD", self._password),
            ("EMAIL_IMAP_HOST", self._imap_host),
            ("EMAIL_SMTP_HOST", self._smtp_host),
        )
        if missing := [name for name, value in required if not value]:
            message = f"Not configured — missing {', '.join(missing)}. Set it via `hermes gateway setup` (env) or platforms.email in config.yaml."
            # Non-retryable: a blank-but-present env var used to drive an indefinite retry loop that leaked until OOM.
            return self._fail("[Email] %s", message, "email_missing_configuration", message, retryable=False)
        if not self._probe_imap(is_reconnect) or not self._probe_smtp():
            return False
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        await cancel_task(self._poll_task)
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
        messages = await asyncio.get_running_loop().run_in_executor(None, self._fetch_new_messages)
        # Dispatch partial results BEFORE escalating a failure — a mid-batch exception returns what was fetched (already marked seen).
        for msg_data in messages:
            await self._dispatch_message(msg_data)
        if self._last_fetch_failed:
            # The IMAP check itself failed (not an empty inbox): route through the fatal-error hook so the gateway's
            # reconnect/backoff re-establishes the mailbox. The handler runs detached (gateway/run.py), so awaiting it is safe.
            # The handler runs in a detached task (gateway/run.py), so awaiting it from our own poll task is
            # safe even though teardown cancels this task. See #80016.
            self._last_fetch_failed = False
            self._set_fatal_error(
                "email_imap_fetch_failed", self._last_fetch_error or "IMAP fetch failed", retryable=True
            )
            await self._notify_fatal_error()

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch new (unseen) messages from IMAP. Runs in executor thread."""
        results = []
        try:
            with self._inbox() as imap:
                status, data = imap.uid("search", None, "UNSEEN")
                for uid in (data[0].split() if status == "OK" and data and data[0] else []):
                    if uid in self._seen_uids:
                        continue
                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        continue  # transient per-UID refusal: leave unseen so the next poll retries
                    # Mark seen once a response arrived (even malformed) so garbage is skipped once, not retried forever —
                    # but NOT before the fetch: a connection failure must leave the rest of the batch eligible for the next poll.
                    # IMAP fetch can return unexpected structures (e.g. a single bytes item instead of a
                    # list of tuples). See #80032.
                    self._seen_uids.add(uid)
                    self._trim_seen_uids()
                    try:
                        raw_email = msg_data[0][1]
                    except (IndexError, TypeError):
                        logger.warning("[Email] Unexpected IMAP response structure for UID %s, skipping", uid)
                        continue
                    if not isinstance(raw_email, (bytes, bytearray)):
                        logger.warning("[Email] Non-bytes IMAP payload for UID %s, skipping", uid)
                        continue
                    # One poison message (unparseable headers, pathological attachment, DNS hiccup) must not abort the batch or force a reconnect.
                    try:
                        # See #80032.
                        parsed = self._parse_fetched_message(uid, raw_email)
                    except Exception as parse_exc:
                        logger.error("[Email] Failed to process message UID %s, skipping: %s", uid, parse_exc)
                        continue
                    if parsed is not None:
                        # [PATCH-5] Two-stage move: park the mail in Working while the
                        # agent runs, so a crash mid-processing leaves a visible trail.
                        # Gated on done_folder (no Done → no moves; Working-without-Done
                        # would strand the mail) AND a Message-ID (the final MOVE → Done
                        # re-locates by Message-ID, UIDs do not survive a MOVE). Runs here,
                        # not in _parse_fetched_message, because it needs the open `imap`
                        # and must not run for unparseable or skipped (None) messages.
                        source_folder = "INBOX"
                        if self._done_folder and self._working_folder and parsed["message_id"]:
                            if self._imap_move(imap, uid, self._working_folder):
                                source_folder = self._working_folder
                            else:
                                logger.warning(
                                    "[Email] Working-folder MOVE failed for UID %s — continuing with mail still in INBOX",
                                    uid,
                                )
                        # [PATCH-5] Tells _dispatch_message -> _finalize_message where to look.
                        parsed["source_folder"] = source_folder
                        results.append(parsed)
        except Exception as e:
            # _close_imap guarantees the socket dies even when logout() raises IMAP4.abort on a broken
            # connection (#79889).
            logger.error("[Email] IMAP fetch error: %s", e)
            self._last_fetch_failed, self._last_fetch_error = True, str(e)
        # Keep the reconnect snapshot current so a mid-outage adapter recreation does not re-dispatch messages already processed.
        self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        return results

    def _parse_fetched_message(self, uid: bytes, raw_email: "bytes | bytearray") -> Optional[Dict[str, Any]]:
        """Parse one RFC822 payload into a dispatchable dict; ``None`` for automated senders. Raises on pathological input (caller logs + continues)."""
        msg = email_lib.message_from_bytes(raw_email)
        sender_addr, sender_name = _extract_email_address(msg.get("From", "")), _decode_header_value(
            msg.get("From", "")
        )
        if "<" in sender_name:
            sender_name = sender_name.split("<")[0].strip().strip('"')
        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        if _is_automated_sender(sender_addr, dict(msg.items())):
            logger.debug("[Email] Skipping automated sender: %s", sender_addr)
            return None
        # Verify From: while the trusted Authentication-Results header is in scope; the verdict is consumed at dispatch (GHSA-rxqh-5572-8m77).
        sender_authenticated, auth_reason = _verify_sender_authentication(
            msg, sender_addr, authserv_id=self._authserv_id
        )
        return {
            "uid": uid,
            "sender_addr": sender_addr,
            "sender_name": sender_name,
            "subject": subject,
            "message_id": msg.get("Message-ID", ""),
            "in_reply_to": msg.get("In-Reply-To", ""),
            "body": _extract_text_body(msg),
            "attachments": _extract_attachments(msg, skip_attachments=self._skip_attachments),
            "date": msg.get("Date", ""),
            "sender_authenticated": sender_authenticated,
            "auth_reason": auth_reason,
        }

    @staticmethod
    def _allow_all_senders() -> bool:
        """True when the operator opted into any sender (EMAIL_ or GATEWAY_ALLOW_ALL_USERS).

        Both names go through the scoped reader: under multiplex ``os.environ`` is the DEFAULT
        profile's opt-in, and borrowing it opened every secondary mailbox to any sender."""
        return any(
            _get_secret(name, "").strip().lower() in _TRUTHY
            for name in ("EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS")
        )

    @staticmethod
    def _open_access() -> bool:
        """True when the gateway admits any sender, so a forged From: gains nothing. The gateway's own order:
        EMAIL_ALLOW_ALL_USERS wins over a list, GATEWAY_ALLOW_ALL_USERS applies only while no list is set."""
        if _get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in _TRUTHY:
            return True
        return _get_secret("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in _TRUTHY and not any(
            _get_secret(name, "").strip() for name in ("EMAIL_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS")
        )

    def _answers_unknown_senders(self) -> bool:
        """True when ``platforms.email.unauthorized_dm_behavior`` opts into ``pair`` or ``decline``."""
        behavior = (self.config.extra or {}).get("unauthorized_dm_behavior")
        return isinstance(behavior, str) and behavior.strip().lower() in {"pair", "decline"}

    def _sender_accepted(self, sender_addr: str, msg_data: Dict[str, Any]) -> bool:
        """Pre-dispatch sender gate: self, automated, authorization, From: authentication.

        Records the authorization verdict as ``msg_data["sender_granted"]`` (False for pair/decline answers)."""
        if sender_addr == self._address.lower():
            return False
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            return False
        allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
        # Parsed like the gateway's allowlists (JSON list literals included), or '["alice"]' would dodge the guard below.
        listed = set()
        for raw in (allowed_raw, _get_secret("GATEWAY_ALLOWED_USERS", "")):
            raw = decode_json_list_literal(raw)
            listed.update(
                str(a).strip().lower()
                for a in (raw if isinstance(raw, list) else str(raw).split(","))
                if str(a).strip()
            )
        if sender_addr.lower() in listed:
            granted = True
        elif sender_addr.split("@", 1)[0].lower() in listed:
            # The gateway's check also matches an address by its bare local part (#119446), so an entry like "alice"
            # would admit, or pair, alice@<any domain>; the domain is the sender's to choose.
            logger.debug("[Email] Dropping sender whose local part alone matches an allowlist entry: %s", sender_addr)
            return False
        else:
            # Approved pairings grant access too, and only the gateway's own check sees them. Its verdict also decides
            # open access: GATEWAY_ALLOW_ALL_USERS beside a GATEWAY_ALLOWED_USERS list grants a stranger nothing there.
            verdict = self._is_sender_authorized(sender_addr, "dm", sender_addr)
            granted = verdict if verdict is not None else (not allowed_raw and self._allow_all_senders())
        # Drop senders the gateway would neither authorize nor answer (pair/decline) before a MessageEvent (and thread
        # context) exists — otherwise a dispatch/authorization race can send a reply even though the handler returned None.
        msg_data["sender_granted"] = granted  # [PATCH-11]
        if not granted and not self._answers_unknown_senders():
            logger.debug(
                "[Email] Dropping unauthorized sender at dispatch (unknown senders are ignored): %s", sender_addr
            )
            return False
        # Reject spoofed senders (GHSA-rxqh-5572-8m77): short of open access, every grant keys on the attacker-controlled
        # From:, and a pairing code or decline is mailed back to it, open access or not; fail-closed. Only a granted
        # sender's drop warns: forged mail from strangers is routine, and the opt-out hint would be wrong advice for it.
        if self._require_authenticated_sender and not msg_data.get("sender_authenticated", False):
            if not granted:
                logger.debug(
                    "[Email] Not answering unknown sender with unauthenticated From: %s (%s)",
                    sender_addr,
                    msg_data.get("auth_reason", "no verdict"),
                )
                return False
            if self._open_access():
                return True
            logger.warning(
                "[Email] Dropping sender with unauthenticated From: %s (%s). If your mail server does not "
                "stamp Authentication-Results, set platforms.email.require_authenticated_sender: false "
                "(or EMAIL_TRUST_FROM_HEADER=true) to accept the risk.",
                sender_addr,
                msg_data.get("auth_reason", "no verdict"),
            )
            return False
        return True

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        # [PATCH-6] The finally below ALWAYS finalizes the mail out of its current
        # folder, so a _sender_accepted() drop (self / automated / non-allowlisted /
        # unauthenticated) or a handle_message exception after a Working move can't
        # strand it in Working. _finalize_message no-ops when done_folder is unset,
        # the mail never moved, or there is no Message-ID to re-locate it by.
        source_folder = msg_data.get("source_folder", "INBOX")
        try:
            if not self._sender_accepted(sender_addr, msg_data):
                return
            subject, body, attachments = msg_data["subject"], msg_data["body"].strip(), msg_data["attachments"]
            text = (
                f"[Subject: {subject}]\n\n{body}" if subject and not subject.startswith("Re:") else body
            )  # subject unless reply
            # DOCUMENT wins over PHOTO for mixed attachments: run.py keys image handling off the per-path mime type regardless
            # of message_type, but document-context injection gates strictly on MessageType.DOCUMENT — so DOCUMENT surfaces both.
            kinds = {att["type"] for att in attachments}
            self._thread_context[sender_addr] = {"subject": subject, "message_id": msg_data["message_id"]}
            # [PATCH-11] After _sender_accepted (already returned above on a drop) and inside the [PATCH-6]
            # try; pair/decline replies to not-granted senders never quote.
            if self._quote_original:
                if msg_data.get("sender_granted"):
                    self._remember_original(msg_data)
                else:
                    self._forget_sender_original(sender_addr)
            name = msg_data["sender_name"] or sender_addr
            event = MessageEvent(
                text=text or "(empty email)",
                message_id=msg_data["message_id"],
                message_type=(
                    MessageType.DOCUMENT
                    if "document" in kinds
                    else MessageType.PHOTO if "image" in kinds else MessageType.TEXT
                ),
                source=self.build_source(
                    chat_id=sender_addr,
                    chat_name=name,
                    chat_type="dm",
                    user_id=sender_addr,
                    user_name=name,
                    message_id=msg_data["message_id"],
                ),
                media_urls=[att["path"] for att in attachments],
                media_types=[att["media_type"] for att in attachments],
                reply_to_message_id=msg_data["in_reply_to"] or None,
            )
            logger.info("[Email] New message from %s: %s", sender_addr, subject)
            await self.handle_message(event)
        finally:
            await asyncio.get_running_loop().run_in_executor(
                None, self._finalize_message, msg_data["message_id"], source_folder
            )

    # [PATCH-11] Quote-the-original-email helpers (dgxarley patch, opt-in).

    def _remember_original(self, msg_data: Dict[str, Any]) -> None:
        """[PATCH-11] Keep what a later reply needs to quote this mail (bounded)."""
        message_id = (msg_data.get("message_id") or "").strip()
        record = {
            "message_id": message_id,
            "body": msg_data.get("body") or "",
            "date": msg_data.get("date") or "",
            "name": msg_data.get("sender_name") or "",
            "address": msg_data["sender_addr"],
            "state": "",
        }
        with self._quote_lock:
            if message_id:
                self._original_by_msg_id[message_id] = record
                self._original_by_msg_id.move_to_end(message_id)
                while len(self._original_by_msg_id) > _QUOTE_LOOKUP_MAX:
                    self._original_by_msg_id.popitem(last=False)
            self._last_original_by_sender[msg_data["sender_addr"]] = record
            self._last_original_by_sender.move_to_end(msg_data["sender_addr"])
            while len(self._last_original_by_sender) > _QUOTE_LOOKUP_MAX:
                self._last_original_by_sender.popitem(last=False)

    def _forget_sender_original(self, sender_addr: str) -> None:
        """[PATCH-11] Drop the per-sender fallback so a reply to a not-granted mail cannot quote an earlier one."""
        with self._quote_lock:
            self._last_original_by_sender.pop(sender_addr, None)

    def _claim_quote(
        self, to_addr: str, body: str, reply_to_msg_id: Optional[str], *, final: bool
    ) -> Tuple[str, Optional[Dict[str, str]]]:
        """[PATCH-11] ``(body with quote, claimed record)``; the record is ``None`` when nothing was quoted.

        Only a turn's final-reply send (``final``, from ``metadata["notify"]``) quotes: the claim marks the
        record pending so a concurrent send skips it, ``_settle_quote`` finalizes it after SMTP success or
        releases it on failure. The agent's own text is never shortened; the quote shrinks (or is dropped) to
        fit MAX_MESSAGE_LENGTH.
        """
        if not self._quote_original or not final:
            return body, None
        original_msg_id = (reply_to_msg_id or self._thread_context.get(to_addr, {}).get("message_id") or "").strip()
        with self._quote_lock:
            record = self._original_by_msg_id.get(original_msg_id) if original_msg_id else None
            if record is None:
                # Fall back to the sender's last mail only when it is the same mail (or carried no Message-ID),
                # never quote a different mail than the one being replied to.
                last = self._last_original_by_sender.get(to_addr)
                if last is not None and (not original_msg_id or last["message_id"] == original_msg_id):
                    record = last
            if record is None or record["address"] != to_addr or record["state"]:
                return body, None
            fixed = len(
                _build_quote(
                    "x",
                    date=record["date"],
                    name=record["name"],
                    address=record["address"],
                    header_fmt=self._quote_header,
                    max_chars=3,
                )
            ) - len("> x")
            budget = min(self._quote_max_chars, MAX_MESSAGE_LENGTH - len(body) - fixed)
            quote = _build_quote(
                record["body"],
                date=record["date"],
                name=record["name"],
                address=record["address"],
                header_fmt=self._quote_header,
                max_chars=budget,
            )
            if not quote:
                return body, None
            record["state"] = "pending"
        return (body + quote if body else quote.lstrip("\n")), record

    def _settle_quote(self, record: Optional[Dict[str, str]], sent: bool) -> None:
        """[PATCH-11] Finalize a claimed quote after SMTP success, or release it so a retry quotes again."""
        if record is not None:
            with self._quote_lock:
                record["state"] = "done" if sent else ""

    async def _run_send(self, fn, args: tuple, log_fmt: str, *log_args) -> SendResult:
        """Run a blocking SMTP sender in the executor; wrap its Message-ID in a SendResult."""
        try:
            return SendResult(
                success=True, message_id=await asyncio.get_running_loop().run_in_executor(None, fn, *args)
            )
        except Exception as e:
            logger.error(log_fmt, *log_args, e)
            return SendResult(success=False, error=str(e))

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send an email reply to the given address; [PATCH-11] only a turn's final-reply send quotes."""
        final = bool((metadata or {}).get("notify"))  # [PATCH-11]
        return await self._run_send(
            self._send_email, (chat_id, content, reply_to, final), "[Email] Send failed to %s: %s", chat_id
        )

    def _message_id_domain(self) -> str:
        """Domain for generated Message-IDs; ``localhost`` when EMAIL_ADDRESS lacks ``@``."""
        return (self._address.rsplit("@", 1)[-1] if "@" in self._address else "") or "localhost"

    def _new_reply(
        self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, *, attach_empty_body: bool = False
    ) -> Tuple[MIMEMultipart, str, str]:
        """Build a threaded reply skeleton. Returns ``(msg, msg_id, subject)``."""
        msg, ctx = MIMEMultipart(), self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        threading = (("In-Reply-To", original_msg_id), ("References", original_msg_id)) if original_msg_id else ()
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        for key, value in (
            ("From", self._address),
            ("To", to_addr),
            ("Subject", subject),
            *threading,
            ("Date", formatdate(localtime=True)),
            ("Message-ID", msg_id),
        ):
            msg[key] = value
        if body or attach_empty_body:
            msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg, msg_id, subject

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        """Login, send, and always release the SMTP connection (quit, else close)."""
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
        # _smtp_send is the single SMTP sink for _send_email and _send_with_files
        # (= _send_email_with_attachment{,s}), so one call site covers all three.
        self._append_to_sent(msg.as_bytes())

    def _send_email(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, final: bool = False) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        # [PATCH-11] Claim before the MIME skeleton is built (so the quote lands in the same body the
        # Sent-folder APPEND archives); settle after _smtp_send so a retry after a failed send quotes again.
        body, quoted = self._claim_quote(to_addr, body, reply_to_msg_id, final=final)
        try:
            msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, attach_empty_body=True)
            self._smtp_send(msg)
        except BaseException:
            self._settle_quote(quoted, sent=False)
            raise
        self._settle_quote(quoted, sent=True)
        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        return msg_id

    def _send_with_files(
        self,
        to_addr: str,
        body: str,
        files: List[Tuple[Path, str]],
        *,
        lenient: bool,
        reply_to_msg_id: Optional[str] = None,
        final: bool = False,
    ) -> str:
        """Send a reply with attachments; *lenient* logs-and-skips unattachable files instead of raising.
        An explicit *reply_to_msg_id* threads the mail like ``_send_email`` does (#10131)."""
        # [PATCH-11] Same claim/settle contract as _send_email (see there); wraps the attach loop too so a
        # failed attachment (non-lenient) also releases the claim for the retry.
        body, quoted = self._claim_quote(to_addr, body, reply_to_msg_id, final=final)
        try:
            msg, msg_id, _ = self._new_reply(to_addr, body, reply_to_msg_id)
            for path, name in files:
                try:
                    _attach_file(msg, path, name)
                except Exception as e:
                    if not lenient:
                        raise
                    logger.warning("[Email] Failed to attach %s: %s", path, e)
            self._smtp_send(msg)
        except BaseException:
            self._settle_quote(quoted, sent=False)
            raise
        self._settle_quote(quoted, sent=True)
        return msg_id

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image URL as part of an email body."""
        return await self.send(chat_id, f"{caption or ''}\n\nImage: {image_url}".strip(), reply_to, metadata)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> SendResult:
        """One email per batch: local files attached, URL images linked in the body (no remote download); base-class fallback on failure."""
        if not images:
            return SendResult(success=False, error="no images to send")
        from urllib.parse import unquote as _unquote

        body_parts, local_paths = [], []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if not image_url.startswith("file://"):
                body_parts.append(f"Image: {image_url}")  # parity with send_image
            elif Path(local_path := _unquote(image_url[7:])).exists():
                local_paths.append(local_path)
            else:
                logger.warning("[Email] Skipping missing image: %s", local_path)
        if not local_paths and not body_parts:
            return SendResult(success=False, error="no valid images in batch")
        final = bool((metadata or {}).get("notify"))  # [PATCH-11]
        try:
            message_id = await asyncio.get_running_loop().run_in_executor(
                None, self._send_email_with_attachments, chat_id, "\n\n".join(body_parts), local_paths, final
            )
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            return await super().send_multiple_images(chat_id, images, metadata, human_delay)
        return SendResult(success=True, message_id=message_id)

    def _send_email_with_attachments(self, to_addr: str, body: str, file_paths: List[str], final: bool = False) -> str:
        """Send an email with multiple file attachments via SMTP (unattachable files are skipped)."""
        msg_id = self._send_with_files(
            to_addr, body, [(Path(f), Path(f).name) for f in file_paths], lenient=True, final=final
        )
        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
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
        """Send a file as an email attachment; [PATCH-11] only a turn's final-reply send quotes."""
        final = bool((kwargs.get("metadata") or {}).get("notify"))  # [PATCH-11]
        return await self._run_send(
            self._send_email_with_attachment,
            (chat_id, caption or "", file_path, file_name, reply_to, final),
            "[Email] Send document failed: %s",
        )

    def _send_email_with_attachment(
        self,
        to_addr: str,
        body: str,
        file_path: str,
        file_name: Optional[str] = None,
        reply_to_msg_id: Optional[str] = None,
        final: bool = False,
    ) -> str:
        """Send an email with a single file attachment via SMTP (raises if unattachable)."""
        return self._send_with_files(
            to_addr,
            body,
            [(Path(file_path), file_name or Path(file_path).name)],
            lenient=False,
            reply_to_msg_id=reply_to_msg_id,
            final=final,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the email chat."""
        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "subject": self._thread_context.get(chat_id, {}).get("subject", ""),
        }


# Plugin glue: register() exposes the platform via the registry; EMAIL_* env → PlatformConfig seeding stays in core.
async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process Email delivery via SMTP (one-shot); standalone_sender_fn contract."""
    extra = getattr(pconfig, "extra", {}) or {}
    address, password = extra.get("address") or _get_secret("EMAIL_ADDRESS", ""), _get_secret("EMAIL_PASSWORD", "")
    smtp_host, smtp_port = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", ""), _esecret_int(
        "EMAIL_SMTP_PORT", 587
    )
    smtp_security = _normalize_security(
        _get_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"),
        default="tls" if smtp_port == 465 else "starttls",
    )
    smtp_tls_verify = _esecret_bool(
        "EMAIL_SMTP_TLS_VERIFY", is_truthy_value(extra.get("smtp_tls_verify"), default=True)
    )
    # [PATCH-8] IMAP Sent-folder archival config (parity with EmailAdapter and
    # upstream PR #28697's standalone-path review fix). The standalone path only
    # requires SMTP, so IMAP may be unconfigured — the shared helper no-ops on a
    # missing host or an empty folder. Security/verify resolved exactly like
    # EmailAdapter.__init__, incl. the [PATCH-10] port-derived default.
    imap_host = (extra.get("imap_host") or _get_secret("EMAIL_IMAP_HOST", "")).strip()
    imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
    imap_security = _normalize_security(
        _get_secret("EMAIL_IMAP_SECURITY", "") or extra.get("imap_security"),
        default=_imap_default_security(imap_port),
    )
    imap_tls_verify = _esecret_bool(
        "EMAIL_IMAP_TLS_VERIFY", is_truthy_value(extra.get("imap_tls_verify"), default=True)
    )
    # Empty string is a deliberate opt-out — do NOT collapse with `or`.
    sent_folder = extra.get("sent_folder", "Sent")
    if not all([address, password, smtp_host]):
        return send_error("Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)")
    try:
        msg = MIMEText(message, "plain", "utf-8")
        for key, value in (
            ("From", address),
            ("To", chat_id),
            ("Subject", "Hermes Agent"),
            ("Date", formatdate(localtime=True)),
        ):
            msg[key] = value
        server = _open_smtp(
            smtp_host,
            smtp_port,
            smtp_security,
            _tls_context(smtp_tls_verify, smtp_host),
            smtplib.SMTP,
            smtplib.SMTP_SSL,
        )
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        # [PATCH-8] Best-effort Sent-folder mirror — never fails the (already-completed) SMTP send.
        _imap_append_to_sent(
            imap_host=imap_host,
            imap_port=imap_port,
            imap_security=imap_security,
            imap_tls_verify=imap_tls_verify,
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
            return send_error(f"Email send failed: {e}")


def _is_connected(config) -> bool:
    """Connected when an address is configured (PlatformConfig.extra or EMAIL_ADDRESS)."""
    if (getattr(config, "extra", {}) or {}).get("address"):
        return True
    import hermes_cli.gateway as gateway_mod

    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="email",
        label="Email",
        adapter_factory=EmailAdapter,
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
