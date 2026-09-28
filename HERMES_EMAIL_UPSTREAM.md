# Hermes email gateway: local patch and upstream PRs

## Why this exists

The upstream email adapter (`plugins/platforms/email/adapter.py` in
`NousResearch/hermes-agent`) has gaps that make it awkward as a long-lived agent mailbox:

1. **No folder lifecycle.** Processed mail stays in `INBOX` (only `\Seen`). A busy INBOX
   becomes unreadable, and after a crash nothing shows which mail was in progress.
2. **No Sent-folder copy.** Replies go out over SMTP and are never written back to IMAP,
   so there is no audit trail unless the SMTP provider auto-captures sent mail.
3. **Existing INBOX is ignored on startup.** Everything already in INBOX is marked seen and
   dropped, which is wrong for first boot, backlog ingestion or restart after downtime.
4. **Replies do not quote the received mail.**

All four are fixed in a local patch; each fix is also submitted upstream as its own PR.

## Local patch

- File: `roles/k8s_infra/files/hermes_email_gateway_patched.py`, a full copy of upstream
  `adapter.py` with `[PATCH-N]` sections. Its header records the synced upstream tag + blob
  and the per-section account of the last re-sync.
- Delivery (no fork, no rebuild): `roles/k8s_infra/tasks/apps/hermes.yml` renders it into
  ConfigMap `hermes-email-patch` (key `adapter.py`), subPath-mounted over
  `/opt/hermes/plugins/platforms/email/adapter.py` in the `hermes-gateway` sidecar (dashboard
  pod, or the webui pod when the dashboard is off). The `checksum/email-patch` pod annotation
  rolls the pod when the file changes.
- Must stay in lockstep with `hermes.image_tag` in `roles/k8s_infra/defaults/main/hermes.yml`.

### Config keys

All are read from `config.yaml` `platforms.email.extra.*` (not env), rendered by
`roles/k8s_infra/templates/hermes/hermes_config.yaml.j2` from `hermes_users[*].email.*` in
`group_vars/all/vault/hermes.yml`. An empty string opts out of that stage for that user.

| Key                | dgxarley default | Effect                                                                           |
|--------------------|------------------|----------------------------------------------------------------------------------|
| `working_folder`   | `Hermes_Working` | INBOX → Working at fetch. `""` → INBOX → Done directly.                          |
| `done_folder`      | `Hermes_Done`    | → Done after `handle_message()`. `""` → no moves at all (INBOX + `\Seen`).       |
| `sent_folder`      | `Sent`           | IMAP APPEND (`\Seen`) after each successful SMTP send. `""` → SMTP only.         |
| `process_existing` | `true`           | Process the pre-existing UNSEEN backlog on first poll (upstream PR default: `false`). |
| `quote_original`   | `hermes.email.quote_original_default` | Quote the received mail below the first successful reply to it. |

### Folder lifecycle

```
INBOX ──UID MOVE──▶ Hermes_Working ──handle_message() returns (try/finally)──▶ Hermes_Done
                    (visible after a crash =
                     "interrupted mid-processing")
```

The Working MOVE only fires when both folders are set AND the mail has a Message-ID, so mail
is never stranded in Working. The `try/finally` covers every drop path (self, automated,
not allowlisted, unauthenticated From) as well as exceptions.

### Move semantics

`_imap_move` prefers `UID MOVE` (RFC 6851), falls back to `UID COPY` + `UID STORE +FLAGS
\Deleted` + `UID EXPUNGE` (UIDPLUS), and finally to a global `EXPUNGE`, which expunges every
`\Deleted` mail in the folder (only legacy servers; Dovecot, Gmail, mailcow, M365, Cyrus 2.5+
support both extensions).

### dgxarley-only sections

- `[PATCH-10]`: upstream defaults `imap_security` to `tls` on every port, which breaks our
  port-143 mailboxes. The patch restores the port-derived default (993 → tls, else starttls);
  an explicit setting still wins.

## Upstream PRs

Opened against `NousResearch/hermes-agent:main` from the fork `vroomfondel/hermes-agent`, one
feature per PR so they can merge independently. All still **open** (no merge yet).

| PR | Branch | Adds |
|----|--------|------|
| [#28697](https://github.com/NousResearch/hermes-agent/pull/28697) | `feat/email-sent-folder` | `platforms.email.sent_folder`, `_imap_append_to_sent()`, APPEND in the adapter senders and `_standalone_send` |
| [#28699](https://github.com/NousResearch/hermes-agent/pull/28699) | `feat/email-process-existing` | `platforms.email.process_existing`, gating the startup pre-fill (default keeps upstream behaviour) |
| [#28702](https://github.com/NousResearch/hermes-agent/pull/28702) | `feat/email-folder-lifecycle` | `working_folder` + `done_folder`, `_ensure_folder` / `_imap_move` / `_search_message_id` / `_finalize_message`, `try/finally` around dispatch |
| [#113192](https://github.com/NousResearch/hermes-agent/pull/113192) | `feat/email-quote-original` | `quote_original` (`[PATCH-11]`), incl. the `sender_granted` gate for pair/decline senders |

The PRs drift into `CONFLICTING`/`DIRTY` whenever upstream refactors the adapter; they are
then rebased onto `main` and pushed `--force-with-lease`.

### Fork clones

One clone per PR (not scratch worktrees): `~/hermes-fork-sent` (#28697),
`~/hermes-fork-existing` (#28699), `~/hermes-fork-folders` (#28702), `~/hermes-fork-quote`
(#113192); `~/hermes-fork` is the base. `origin` = fork, `upstream` = NousResearch. Run
tests with `env -u VIRTUAL_ENV` (dgxarley's own venv otherwise confuses `uv`):

```bash
cd ~/hermes-fork-<feature>
env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/gateway/test_email.py -v
```

## Re-sync procedure

### On every `hermes.image_tag` bump

Check the live pod image first: keel (`policy: minor`, 24h poll) usually pulls a new tag
before the defaults move, and a `--tags hermes` run with the old pin rolls the pods back.

1. Fetch the new baseline and compare its blob with the one in the patch header:

   ```bash
   gh api "repos/NousResearch/hermes-agent/contents/plugins/platforms/email/adapter.py?ref=<tag>" \
     --jq '.content' | base64 -d > /tmp/adapter_new.py
   ```

   Also check `plugin.yaml` / `__init__.py` in the same directory (the mount target depends on
   the layout).
2. Byte-identical: only update the "synced to upstream tag" line in the patch header.
3. Changed: 3-way merge (`git merge-file`) of the black-formatted old baseline, our file and
   the black-formatted new baseline; re-apply conflicting `[PATCH-N]` sections by hand. Record
   what moved in the patch header.
4. Verify: `black --check`, `ast.parse`, pyflakes, diff vs the new baseline shows only
   `[PATCH-N]` sections. The patch must add no non-stdlib import beyond upstream's own.
5. Rollout happens with the next `--tags hermes` run (`checksum/email-patch` rolls the pod).

The same bump also re-checks `hermes_health_patch.py` (checklist in its header) and the
contracts listed above `hermes.image_tag`.

### When an upstream PR merges

1. Bump `hermes.image_tag` to a release containing the merge.
2. Drop the corresponding `[PATCH-N]` section(s) and adapt the `config.yaml` keys if the
   merged shape differs.
3. Once every PR is merged and `[PATCH-10]` is no longer needed: drop the
   `hermes-email-patch` ConfigMap task, the subPath mount, the `checksum/email-patch`
   annotation and the patch file.

### Open watch item

- Upstream `c13ea774e6` (on `main`, not yet in a tag) replaces `from hermes_cli import
  __version__` in `_send_imap_id()` with `hermes_cli.version_info.get_version_info()`. It is
  non-fatal (`except Exception` falls back to `"0"`), but take it along at the next re-sync.

## Operational notes

- `imap.create()` runs on every connect; "already exists" returns `NO` and is swallowed.
- Sent-folder APPEND failures are logged as warnings and swallowed; they never turn into a
  failed SMTP send.
- A mail stuck in `Hermes_Working` means the gateway process died before the `finally` ran.
