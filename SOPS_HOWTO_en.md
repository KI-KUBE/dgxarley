# SOPS HOWTO — Encrypt secrets & deploy with Ansible

A practical guide: set up SOPS, store secrets encrypted in Git, and decrypt them
transparently during an Ansible deploy. As the key backend we use **age**
(modern, no GPG-keyring pain).

> **SOPS** = *Secrets OPerationS*, a CNCF project (formerly Mozilla).
> It encrypts **only the values** in YAML/JSON/ENV/INI files, not the keys.
> This keeps the file diffable and mergeable — you can see *which* key changes,
> just not the value.

---

## 0. Overview — how it fits together

```
                 ┌─────────────┐   encrypted        ┌──────────────┐
   you (editor) ─▶│ sops edit    │ ─────────────────▶ │ vault.yml    │──▶ git
                 └─────────────┘   (age public key)  │ (plaintext,  │
                                                     │  only values │
                                                     │  ciphered)   │
   ansible ◀── decrypted ◀── sops (age secret) ◀────┴──────────────┘
```

- **Public key** → encrypts. Allowed in the repo (`.sops.yaml`).
- **Secret key** (`keys.txt`) → decrypts. Stays **local / in a secret store**, never in Git.
- Anyone who needs to deploy needs the secret key. Anyone who only needs to
  write gets by with the public key.

---

## 1. Installation

**sops** (recommended via Homebrew, keeps itself up to date):

```bash
brew install sops
sops --version    # >= 3.9
```

Alternatively, the official release binary:

```bash
curl -L -o ~/.local/bin/sops \
  https://github.com/getsops/sops/releases/latest/download/sops-v3.13.2.linux.amd64
chmod +x ~/.local/bin/sops
```

**age** (key generation + crypto backend):

```bash
# Debian/Ubuntu:
sudo apt install age
# or brew install age
age --version
```

---

## 2. Generate an age key (once per person/machine)

```bash
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
chmod 600 ~/.config/sops/age/keys.txt
```

Output (example):

```
Public key: changeme_age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8j
```

- `~/.config/sops/age/keys.txt` is the **default path** that sops finds
  automatically — no need to set environment variables.
- You'll need the **public key** (the line above, or
  `age-keygen -y ~/.config/sops/age/keys.txt`) shortly for `.sops.yaml`.
- **Back up `keys.txt`** to a safe place (password manager / vault).
  If you lose it, all secrets encrypted only with this key are gone.

Print the public key again at any time:

```bash
age-keygen -y ~/.config/sops/age/keys.txt
```

---

## 3. Create `.sops.yaml` in the repo root

This file tells sops **which files are encrypted with which keys**.
It belongs **in Git** (it contains only public keys). Example for an
Ansible layout:

```yaml
# .sops.yaml
keys:
  # >>> every person allowed to decrypt, with their age PUBLIC key <<<
  - &thiess age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8j
  - &ci     age1cizzz...ci-deploy-key...zzz

creation_rules:
  # All Ansible vault files
  - path_regex: (group_vars|host_vars)/.*/vault\.ya?ml$
    key_groups:
      - age:
          - *thiess
          - *ci

  # Standalone secret files
  - path_regex: secrets/.*\.(ya?ml|json|env)$
    key_groups:
      - age:
          - *thiess
          - *ci

  # Fallback for anything else you encrypt explicitly
  - key_groups:
      - age:
          - *thiess
```

Note:
- **Only the first matching `creation_rule`** applies (order = priority).
- YAML anchors (`&thiess` / `*thiess`) avoid copy-pasting the long keys.
- Only the keys listed here can decrypt later. Adding a new person =
  add their key + `sops updatekeys` (see §8).

---

## 3b. SSH keys as recipients (instead of dedicated age keys)

If the team already has **SSH ed25519 keys** (`~/.ssh/id_ed25519`, registered in
GitHub/GitLab, present on the deploy hosts), you can use them directly as
recipients — no one has to create and manage a separate age key. There are two
ways.

> Verified with **sops 3.13.2** on this system. Native SSH support has existed
> since sops ~3.9. Only **`ssh-ed25519` and `ssh-rsa`** work — no ecdsa/`sk-`
> keys.

### Way 1 — SSH pubkey directly (native, no extra tool) ✅ recommended

age *and* sops understand SSH keys directly. You enter the **complete pubkey
line** as an age recipient in `.sops.yaml` — **not** an `age1…` string:

```yaml
# .sops.yaml
keys:
  - &thiess_ssh ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB9xtjTmS6oCcWrgN/8MBrZqcuwmnx9E... thiess@hyperion
  - &ci_ssh     ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA...ci-deploy...                    ci@runner

creation_rules:
  - path_regex: (group_vars|host_vars)/.*/vault\.ya?ml$
    key_groups:
      - age:
          - *thiess_ssh
          - *ci_ssh
```

Encryption then happens as usual (`sops group_vars/prod/vault.yml`). In the
encrypted file the recipient is the SSH line, e.g.
`recipient: ssh-ed25519 AAAA...`.

**Decrypting** — sops finds the matching SSH **private key** like this:

- **automatically** at `~/.ssh/id_ed25519` or `~/.ssh/id_rsa` — *no env
  variable needed* (tested).
- or explicitly, if the key lives elsewhere:

  ```bash
  export SOPS_AGE_SSH_PRIVATE_KEY_FILE=/path/to/id_ed25519
  sops decrypt group_vars/prod/vault.yml
  ```

Pitfalls:
- **`SOPS_AGE_KEY_FILE` does NOT work with an SSH key** (the variable expects
  the age format `AGE-SECRET-KEY-1…`). For SSH, use the default path or
  `SOPS_AGE_SSH_PRIVATE_KEY_FILE`.
- **Passphrase-protected** SSH keys are prompted for interactively by age — so
  for automated deploys/CI use a **passwordless deploy key** (ssh-agent doesn't
  help here, age doesn't talk to the agent).

### Way 2 — Convert SSH → age with `ssh-to-age`

Some setups (e.g. **sops-nix**) or tooling that only understands `age1…`
recipients want uniform age keys. `ssh-to-age` converts ed25519 SSH keys
**deterministically** into age keys.

```bash
brew install ssh-to-age            # currently not installed on your system

# SSH pubkey  -> age recipient (age1...) for .sops.yaml:
ssh-to-age -i ~/.ssh/id_ed25519.pub
curl -s https://github.com/YOUR_USER.keys | ssh-to-age     # straight from GitHub

# SSH privkey -> age identity (for decrypting) at the default path:
ssh-to-age -private-key -i ~/.ssh/id_ed25519 -o ~/.config/sops/age/keys.txt
```

Then enter the resulting `age1…` recipient in `.sops.yaml` as in §3; the
`keys.txt` at the default path is found automatically when decrypting.

### Which way when

| | Way 1 (SSH direct) | Way 2 (`ssh-to-age`) |
|---|---|---|
| Extra tool | none | `ssh-to-age` |
| Recipient in `.sops.yaml` | `ssh-ed25519 AAAA…` | `age1…` |
| Single source of truth | the SSH key itself | converted age key |
| Good when | simplest start, SSH keys already present | uniform `age1…` recipients (sops-nix, mixed with pure age keys) |

**Recommendation:** Way 1, unless you have a concrete reason for uniform
`age1…` recipients.

### Deploy with SSH keys

Nothing extra is needed on the deploy runner — the SSH private key usually lives
on ops machines anyway. The Ansible deploy is then exactly as in §6; sops
accesses `~/.ssh/id_ed25519` automatically.

CI with an SSH deploy key:

```yaml
- name: Provide SSH deploy key
  run: |
    mkdir -p ~/.ssh
    echo "${{ secrets.DEPLOY_SSH_KEY }}" > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
- name: Deploy
  run: ansible-playbook -i inventory/prod site.yml   # sops uses ~/.ssh/id_ed25519 automatically
```

---

## 4. Encrypt & edit secrets

**Create a new file / edit an existing one in the editor** (opens `$EDITOR`,
encrypts automatically on save according to the `.sops.yaml` rules):

```bash
sops group_vars/prod/vault.yml
```

You type plaintext YAML:

```yaml
db_password: super-secret-123
api_token: ghp_xxxxxxxxxxxx
```

After saving, only this remains on disk / in Git:

```yaml
db_password: ENC[AES256_GCM,data:9f3k...,tag:...]
api_token:  ENC[AES256_GCM,data:a1b2...,tag:...]
sops:
    age:
        - recipient: age1ql3z...
          enc: |
            -----BEGIN AGE ENCRYPTED FILE-----
            ...
    lastmodified: "2026-07-07T..."
    mac: ENC[...]
```

Other useful commands:

```bash
# Encrypt an existing plaintext file in place
sops encrypt --in-place secrets/api.yml

# Print decrypted to stdout (don't save) — for a quick check
sops decrypt group_vars/prod/vault.yml

# Set/get a single value without an editor
sops set   group_vars/prod/vault.yml '["db_password"]' '"new-pw"'
sops decrypt --extract '["db_password"]' group_vars/prod/vault.yml
```

---

## 5. Ansible integration

The clean way is the **`community.sops` collection** — with it, Ansible decrypts
`vault.yml` files automatically at load time, without you calling
`sops decrypt` manually anywhere.

### 5.1 Setup (once)

```bash
ansible-galaxy collection install community.sops
```

`requirements.yml` for the repo (so CI/colleagues can reproduce it):

```yaml
# requirements.yml
collections:
  - name: community.sops
```

### 5.2 Variant A — `vars_files` / `include_vars` (explicit)

In a playbook you load an encrypted file via the lookup plugin:

```yaml
- hosts: dbservers
  vars:
    vault: "{{ lookup('community.sops.sops', 'group_vars/prod/vault.yml') | from_yaml }}"
  tasks:
    - name: Set DB password
      ansible.builtin.debug:
        msg: "pw is {{ vault.db_password }}"
```

### 5.3 Variant B — vars plugin (automatic, recommended)

Enable the **vars plugin**, then `group_vars`/`host_vars` that are
sops-encrypted are loaded just like unencrypted vars — transparently, without a
lookup:

```ini
# ansible.cfg
[defaults]
vars_plugins_enabled = host_group_vars,community.sops.sops
```

After that, plain Ansible is enough:

```yaml
- hosts: dbservers
  tasks:
    - ansible.builtin.debug:
        msg: "pw is {{ db_password }}"   # comes from group_vars/prod/vault.yml
```

Ansible calls sops in the background; sops finds your age key at
`~/.config/sops/age/keys.txt` and decrypts on the fly.

---

## 6. Deploy workflow

**Prerequisite on the deploy machine (laptop or CI runner):**
`sops` + `age` installed, and the **age secret key** available.

```bash
# 1. Fetch collections
ansible-galaxy collection install -r requirements.yml

# 2. Make sure the key is there
ls -l ~/.config/sops/age/keys.txt

# 3. Deploy — secrets are decrypted at runtime, never written in plaintext
ansible-playbook -i inventory/prod site.yml
```

Quick pre-check that decryption works, before running the whole playbook:

```bash
sops decrypt group_vars/prod/vault.yml | head
# or on the Ansible side:
ansible -i inventory/prod dbservers -m debug -a "var=db_password"
```

### 6.1 CI/CD (GitLab / GitHub Actions)

Store the age secret key as a **CI secret** and write it to the default path at
runtime — never in the repo:

```yaml
# GitHub Actions example
- name: Provide age key
  run: |
    mkdir -p ~/.config/sops/age
    echo "${{ secrets.SOPS_AGE_KEY }}" > ~/.config/sops/age/keys.txt
    chmod 600 ~/.config/sops/age/keys.txt

- name: Deploy
  run: |
    ansible-galaxy collection install -r requirements.yml
    ansible-playbook -i inventory/prod site.yml
```

As an alternative to the file path, the env var `SOPS_AGE_KEY` (the key content
directly) also works — handy for containers:

```bash
export SOPS_AGE_KEY="AGE-SECRET-KEY-1..."
ansible-playbook -i inventory/prod site.yml
```

---

## 7. Git integration (nice diffs)

So that `git diff` shows the **plaintext diff** for encrypted files instead of
ciphertext garbage:

```bash
# .gitattributes
group_vars/**/vault.yml diff=sops
host_vars/**/vault.yml  diff=sops
group_vars/**/vault/** diff=sops
host_vars/**/vault/**  diff=sops
secrets/**              diff=sops
```

```bash
# configure once, locally
git config diff.sops.textconv "sops decrypt"
```

Now `git diff` shows the decrypted values — locally, only for people with the
key.

> Optional: a `pre-commit` hook that prevents accidentally committing an
> **unencrypted** `vault.yml` (checks whether the `sops:` field is present in
> the file).

---

## 8. Key management & rotation

**Add a new person / new CI key:**

1. Enter their age **public key** in `.sops.yaml` under `keys:` + in the
   appropriate `creation_rule`.
2. Re-encrypt existing files with the new keys:

   ```bash
   sops updatekeys group_vars/prod/vault.yml
   # or all at once:
   find . -name 'vault.yml' -exec sops updatekeys -y {} \;
   ```

   `updatekeys` re-encrypts **only the data key**, not your plaintext — fast and
   diff-friendly.

**Remove a person:** delete the public key from `.sops.yaml` → `sops updatekeys`.
Note: anyone with the old Git history can still decrypt old versions → on a real
leak, **rotate the secrets themselves** (new passwords).

**Rotate the data key** (fresh encryption key, e.g. after suspicion):

```bash
sops rotate --in-place group_vars/prod/vault.yml
```

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `no matching creation rules found` | The path matches no `creation_rule` in `.sops.yaml`. Check the regex. |
| `failed to get the data key ... no identity matched` | Your public key isn't in the file. Have someone with access run `sops updatekeys`. |
| Ansible doesn't see the vars | `community.sops` not installed **or** `vars_plugins_enabled` missing in `ansible.cfg`. |
| sops can't find the key | `keys.txt` not under `~/.config/sops/age/`, or `SOPS_AGE_KEY(_FILE)` not set. |
| SSH key is not accepted | `SOPS_AGE_KEY_FILE` points at an SSH key → wrong. Use `SOPS_AGE_SSH_PRIVATE_KEY_FILE` or place the key at `~/.ssh/id_ed25519`. Only ed25519/rsa supported. |
| CI: `age: no identity` | CI secret `SOPS_AGE_KEY` missing/empty, or the file isn't `chmod 600`. |

---

## 10. Cheat sheet (the 6 commands you need daily)

```bash
sops secrets/foo.yml                       # create/edit (encrypts on save)
sops decrypt secrets/foo.yml               # view plaintext
sops encrypt --in-place plain.yml          # encrypt a plaintext file
sops updatekeys secrets/foo.yml            # re-apply keys from .sops.yaml
sops rotate --in-place secrets/foo.yml     # rotate the data key
ansible-playbook -i inventory/prod site.yml  # deploy (decrypts automatically)
```

---

### Appendix: SOPS vs. git-crypt (this repo currently uses git-crypt)

This repo encrypts vault files today with **git-crypt** (see `.gitattributes`).
A short comparison, in case a switch is up for debate:

| | git-crypt | SOPS |
|---|---|---|
| Granularity | whole file (binary in Git) | individual **values**, keys stay readable |
| Diff/merge | file = blob, barely mergeable | diffable line by line (which key changes) |
| Key backend | GPG / symmetric | age, GPG, KMS, Vault, … |
| Granular access | per repo | per file regex + key groups |
| Ansible | file must be checked out decrypted | `community.sops` decrypts on the fly |

**Do not apply both to the same files at once.** A switch would mean: remove the
git-crypt filter for `vault.yml`, decrypt the files, re-encrypt with `sops`,
adjust `.gitattributes`/`.sops.yaml` — done deliberately as its own step, not on
the side.
