# k3shelperstuff

Standalone helper scripts for operating the K3s cluster:

- [`update_local_k3s_keys.py`](#update_local_k3s_keyspy) — keep the local kubeconfig in sync with a remote K3s server
- [`keel_drift.py`](#keel_driftpy) — find Keel-tracked workloads whose running image lags behind its tag
- [`pin_drift.py`](#pin_driftpy) — find version pins in the repo that lag behind the upstream release

## update_local_k3s_keys.py

K3s kubeconfig credential synchronization utility. Keeps local Kubernetes authentication credentials (`~/.kube/config`) in sync with a remote K3s server.

### What it does

Fetches the kubeconfig from a remote K3s server via SSH, compares user credentials and cluster CA data against the local kubeconfig, and interactively updates any differences.

- Extracts client certificates, client keys, and cluster CA data from both remote and local kubeconfig
- Shows truncated diffs without exposing full secrets
- Prompts before writing any changes
- Auto-detects remote host and context from the current-context in `~/.kube/config`

### Usage

```bash
k3s-keys-sync [OPTIONS]                                   # installed entry point
python -m dgxarley.k3shelperstuff.update_local_k3s_keys   # or as a module
```

| Option                    | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `-u`, `--user USER`       | SSH user (default: `root`)                                    |
| `-H`, `--host HOST`       | Remote host (auto-detected from kubeconfig server URL)        |
| `-c`, `--context CONTEXT` | Local kubeconfig context (auto-detected from current-context) |

The remote kubeconfig is read from `/etc/rancher/k3s/k3s.yaml` on the target host.

## keel_drift.py

Finds Keel-tracked workloads whose running image is older than the image its tag currently points at.

### Why it exists

On every poll, Keel compares the registry digest of right now against the digest it memorised during the previous poll. That memo lives in memory only and is seeded from the registry at startup. What actually runs in the cluster therefore never enters Keel's decision: if a tag is moved while Keel restarts, Keel sets its baseline to the new digest without ever touching the Deployment, and the change stays invisible until the next push.

This script performs exactly the comparison Keel does not: the digest of the running pod against the digest the tag currently points at.

### What it does

- Collects every Deployment, StatefulSet and DaemonSet carrying an active `keel.sh/policy` (annotations beat labels, `never` and empty count as inactive, the same order Keel itself uses)
- Reads the running digest per container from the `imageID` of the running pods, **including initContainers** (their status still carries the digest they ran with long after they finished)
- Marks a stale initContainer that Keel would not touch: tracking is opt-in via `keel.sh/initContainers: "true"` (labels first, then annotations), so without it such a container stays stale indefinitely with nothing pointing at it
- Resolves the tag against the registry, accepting both the index digest and any per-platform manifest digest of a multi-arch tag
- Authenticates with the workload's `imagePullSecrets`, falling back to the local Docker login (`DOCKER_CONFIG` or `~/.docker/config.json`) so Docker Hub does not count against the anonymous 100/h per-IP limit
- Distinguishes **index drift from image drift**: if the running digest does not match the tag, it is resolved as an index and its platform manifests are compared against the tag's. A registry that re-pushes an index (changed attestations, say) without moving the manifests underneath leaves the running bits identical, so this counts as current rather than stale
- Flags containers with `imagePullPolicy != Always`, since a restart cannot renew an unchanged tag there

### Usage

```bash
keel-drift [OPTIONS]                          # installed entry point
python -m dgxarley.k3shelperstuff.keel_drift  # or as a module
```

Needs the optional dependencies: `pip install 'dgxarley[k3s]'`.

| Option                   | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| `-n`, `--namespace NS`   | Check only this namespace (default: all)                      |
| `-c`, `--context CTX`    | kubeconfig context to check (default: current, env `KUBE_CONTEXT`) |
| `--drift-only`           | Show only stale and unclear workloads                         |
| `--fix-command`          | Print the `kubectl rollout restart` commands to straighten out |
| `-q`, `--quiet`          | Suppress the table, print only the summary                    |
| `-v`, `--verbose`        | Log every namespace, workload and registry access             |
| `--no-local-credentials` | Ignore the local Docker login, query everything anonymously   |

The table goes to stdout, progress and diagnostics to stderr, so the output stays pipe-friendly.

### Exit codes

| Code | Meaning                                                      |
|------|--------------------------------------------------------------|
| `0`  | No workload is stale (or none is tracked by Keel at all)      |
| `1`  | At least one workload is stale, usable as a pipeline gate     |
| `2`  | The named context is unknown, or no kubeconfig / in-cluster context could be used |

### Examples

```bash
keel-drift                          # every tracked workload
keel-drift --context ht@dgxarley    # a specific kubeconfig context
keel-drift --namespace somestuff    # a single namespace
keel-drift --drift-only --quiet     # drift only, terse
keel-drift --fix-command            # print rollout-restart commands
```

## pin_drift.py

Finds version pins in the repo that are older than the newest upstream release. Counterpart to `keel_drift.py`: that one compares the digest of a rolling tag (`latest`, `main-stable`), this one compares a fixed pin (`v1.2.3`) against the release list of the project it comes from. Read-only: it reads repo files and release APIs, nothing else.

### What it does

- Loads the pin declarations from `pin_drift.yml` (searched upwards from the current directory, so it works from anywhere inside the repo)
- Extracts each pin from the declared file(s) by regex. If a pin lives in several files (role defaults plus `group_vars` copies, manifest plus update script), **all occurrences must agree**, otherwise the pin is reported as `unclear`
- Fetches the upstream releases (GitHub or Forgejo/Gitea API), skipping pre-releases and drafts, and keeps the tags matching `tag_pattern`
- Classifies the gap as `patch`, `minor` or `MAJOR` by the first version component that differs. For a major gap it also names the newest release within the pinned major
- Treats a pin shorter than the upstream version (`5.1` against `5.1.3`) as a floating tag: current until a newer `5.2` appears
- Reports a git-crypt locked file, a pattern that no longer matches, or an unreachable upstream as `unclear` instead of guessing

### Declaring a pin

One entry in `pin_drift.yml` per pin:

| Key           | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `name`        | Unique label                                                            |
| `file` / `files` | Repo-relative file(s) carrying the pin                               |
| `var`         | YAML key whose scalar value is the version (`hindsight_version`)        |
| `image`       | Image repo; the tag after the colon is the version (`prom/prometheus`)  |
| `pattern`     | Raw regex with exactly one capture group, for everything else           |
| `github`      | Upstream as `owner/repo`                                                |
| `forgejo`     | Upstream as `host/owner/repo` (Codeberg and other Forgejo/Gitea hosts)  |
| `source`      | `releases` (default) or `tags`, for projects that tag without releasing |
| `tag_pattern` | Regex an upstream tag must match (default `^v?\d+(\.\d+)+$`)            |

Exactly one of `var` / `image` / `pattern` and exactly one of `github` / `forgejo`. Versions are compared by their numeric components, so a pin whose suffix contains digits (`0.5.21-sm121`) needs a `pattern` that captures the version part only.

### Usage

```bash
pin-drift [OPTIONS]                          # installed entry point
python -m dgxarley.k3shelperstuff.pin_drift  # or as a module
```

Needs the optional dependencies: `pip install 'dgxarley[k3s]'`.

| Option                | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `-c`, `--config PATH` | Pin declarations (default: `pin_drift.yml` found upwards, env `PIN_DRIFT_CONFIG`) |
| `-o`, `--only NAME`   | Check only this pin (repeatable)                                         |
| `--updates-only`      | Hide the pins that are current                                           |
| `-q`, `--quiet`       | Suppress the table, print only the summary                               |

GitHub allows 60 anonymous API requests per hour, fewer than two full runs. A token is taken from `GITHUB_TOKEN` / `GH_TOKEN` or, failing that, from `gh auth token`; it is sent to `api.github.com` only.

### Exit codes

| Code | Meaning                                                  |
|------|----------------------------------------------------------|
| `0`  | Every pin is current (unclear pins do not fail the run)  |
| `1`  | At least one pin has an update, usable as a pipeline gate |
| `2`  | No `pin_drift.yml` found, it is invalid, or `--only` names an unknown pin |

### Examples

```bash
pin-drift                        # every declared pin
pin-drift --updates-only         # hide the pins that are current
pin-drift --only hermes-agent    # a single pin
pin-drift -o k3s -o sglang -q    # two pins, summary only
```
