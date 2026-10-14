# k3shelperstuff

Standalone helper scripts for operating the K3s cluster:

- [`update_local_k3s_keys.py`](#update_local_k3s_keyspy) — keep the local kubeconfig in sync with a remote K3s server
- [`keel_drift.py`](#keel_driftpy) — find Keel-tracked workloads whose running image lags behind its tag
- [`pin_drift.py`](#pin_driftpy) — find version pins in the repo that lag behind the upstream release
- [`drift.py`](#driftpy) — `pin_drift` and `keel_drift` as subcommands `pin`/`keel` of one CLI

## update_local_k3s_keys.py

Fetches the kubeconfig from a remote K3s server via SSH, compares user
credentials and cluster CA data against the local kubeconfig, and interactively
updates any differences.

- Extracts client certificates, client keys, and cluster CA data from both remote and local kubeconfig
- Shows truncated diffs without exposing full secrets
- Prompts before writing any changes
- Auto-detects remote host and context from the current-context in `~/.kube/config`

Also runnable as `python -m dgxarley.k3shelperstuff.update_local_k3s_keys`.

```bash
k3s-keys-sync [OPTIONS]
```

| Option                    | Description                                                       |
|---------------------------|-------------------------------------------------------------------|
| `-u`, `--user USER`       | SSH user (default: `root`)                                        |
| `-H`, `--host HOST`       | Remote host (auto-detected from kubeconfig server URL)            |
| `-c`, `--context CONTEXT` | Local kubeconfig context (auto-detected from current-context)     |
| `--create`                | Create context/cluster/user if the context does not exist locally |
| `-y`, `--yes`             | Non-interactive mode (skip confirmation prompts)                  |
| `--server URL`            | Override the K3s API server URL (default: `https://{host}:6443`)  |

The remote kubeconfig is read from `/etc/rancher/k3s/k3s.yaml` on the target host.

## keel_drift.py

### Why it exists

On every poll [Keel](https://keel.sh) compares **only** the registry digest of
right now against the digest it memorised during its previous poll
(`trigger/poll/single_tag_watcher.go`). That memo lives in memory alone and is
seeded from the registry at startup (`trigger/poll/watcher.go`, `addJob`).

Which means: **what actually runs in the cluster never enters Keel's decision.**
If a tag is moved while Keel restarts, Keel sets its baseline to the new digest
without ever touching the corresponding deployment — the change stays invisible
until the next push.

`keel_drift.py` makes exactly the comparison Keel does not: **running pod digest
against the digest the tag currently points at.**

### What it checks

- All `Deployment`/`StatefulSet`/`DaemonSet` objects carrying a `keel.sh/policy`
  annotation or label (`never`/empty is treated as untracked, exactly as Keel does)
- Regular containers **and** initContainers — the latter are flagged when they are
  stale while `keel.sh/initContainers: "true"` is missing, i.e. where Keel would
  never act
- `imagePullPolicy != Always` on a stale container — there a restart (and Keel's
  force policy) cannot re-pull the unchanged tag at all
- Multi-arch tags: both the index digest and its platform manifests count as a
  match, and an older index pointing at the same platform manifests is reported
  as current rather than as a false positive

Credentials come from the workload's `imagePullSecrets`; the local
`~/.docker/config.json` (or `$DOCKER_CONFIG`) is used as a fallback, because
anonymous Docker Hub access is capped at 100 manifest requests per hour and IP.

### Usage

Needs the optional dependencies: `pip install 'dgxarley[k3s]'`. Also runnable as
`python -m dgxarley.k3shelperstuff.keel_drift`.

```bash
keel-drift                        # all tracked workloads
keel-drift --namespace somestuff  # a single namespace
keel-drift --drift-only --quiet   # only deviations, terse
keel-drift --fix-command          # emit rollout-restart commands
```

| Option                   | Env var                     | Description                                      |
|--------------------------|-----------------------------|--------------------------------------------------|
| `-n`, `--namespace NS`   | `KEEL_NAMESPACE`            | Only check this namespace                        |
| `-c`, `--context CTX`    | `KEEL_CONTEXT`, `KUBE_CONTEXT` | kubeconfig context instead of the active one  |
| `--drift-only`           | `KEEL_DRIFT_ONLY`           | Only show stale and unclear workloads            |
| `--fix-command`          | `KEEL_FIX_COMMAND`          | Emit `kubectl rollout restart` commands (with `--context`, if given) |
| `-q`, `--quiet`          | `KEEL_QUIET`                | Suppress the table, print the summary only       |
| `-v`, `--verbose`        | `KEEL_VERBOSE`              | Log every workload and registry access (DEBUG)   |
| `--no-local-credentials` | `KEEL_NO_LOCAL_CREDENTIALS` | Ignore the local Docker login, query anonymously |

The table and the summary go to **stdout**, logging and the progress display to
**stderr** — so the result can be piped while the diagnostics stay visible.

Without a kubeconfig the in-cluster service account is used, so the tool also
runs as a `Job`/`CronJob` inside the cluster.

### Exit codes

| Code | Meaning                                                     |
|------|-------------------------------------------------------------|
| `0`  | Nothing stale (or no tracked workloads at all)              |
| `1`  | At least one container is stale — usable as a pipeline gate |
| `2`  | Unusable `--context` (the available ones are listed), or no kubeconfig and no in-cluster context |

## pin_drift.py

Counterpart to `keel_drift`: that one checks a rolling tag by digest, this one
checks a **fixed pin** (`v1.2.3`) against the release list of the project it
comes from. Read-only — it reads the repo files, the release APIs and, if a
kubeconfig is usable, the pods and nodes of the cluster, nothing else.

The pins are declared in `pin_drift.yml`, searched upwards from the current
directory (or given via `--config`). Paths in it are relative to the file
itself.

```yaml
pins:
  - name: mosquitto
    file: mosquitto-2.1/Dockerfile
    pattern: 'mosquitto/archive/refs/tags/v([\d.]+)\.tar\.gz'
    github: eclipse-mosquitto/mosquitto
    source: tags
  - name: some-chart
    files: [values.yaml, values-prod.yaml]
    var: app_version                 # matches `app_version: 1.4.2`
    forgejo: git.example.com/owner/repo
  - name: grafana
    file: deploy/grafana.yaml
    image: grafana/grafana           # matches `grafana/grafana:11.2.0`
    github: grafana/grafana
    tag_pattern: '^v\d+\.\d+\.\d+$'
```

| Key                      | Meaning                                                                         |
|--------------------------|---------------------------------------------------------------------------------|
| `name`                   | Unique name, used by `--only`                                                   |
| `file` / `files`         | File(s) carrying the pin; several files must agree on the same value            |
| `var`                    | Extract from a `key: value` line                                                |
| `image`                  | Extract the tag after `image:`                                                  |
| `pattern`                | Raw regex with exactly one capture group                                        |
| `github` / `forgejo`     | `owner/repo` or `host/owner/repo` the releases come from                        |
| `source`                 | `releases` (default, drafts/prereleases skipped) or `tags`                      |
| `tag_pattern`            | Filter for upstream tags (default: `^v?\d+(?:\.\d+)+$`)                         |
| `running_image`          | Image(s) whose tags in live pods are the running version (default: `image`)     |
| `running_kubelet`        | `true`: the running version is the nodes' kubelet version (k3s, k8s)            |

Exactly one of `var`/`image`/`pattern` and one of `github`/`forgejo` is required.
A pin shorter than the upstream version (`5.1` against `5.1.3`) is a floating
tag and counts as current until a `5.2` appears. Files encrypted with git-crypt
are reported as `unclear` instead of failing.

### Running column

With a usable kubeconfig (or in-cluster context) the table gets a `running`
column: the distinct tags of the matching images across all live pods
(containers and initContainers, finished pods ignored), or the kubelet versions
for `running_kubelet` pins. Image names are compared normalised (`redis` ==
`docker.io/library/redis`), tags ignoring a leading `v`. A pin whose running
version differs is noted as `not rolled out`, or `partially rolled out` if the
pinned version runs alongside older ones; this does not affect the exit code.
`-` means no matching pod — expected for pins outside the cluster.

GitHub allows 60 anonymous API requests per hour; a token is taken from
`GITHUB_TOKEN` / `GH_TOKEN` or `gh auth token`.

### Usage

Needs the optional dependencies: `pip install 'dgxarley[k3s]'`. Also runnable as
`python -m dgxarley.k3shelperstuff.pin_drift`.

```bash
pin-drift                        # every declared pin
pin-drift --updates-only         # hide the pins that are current
pin-drift --only mosquitto       # a single pin (repeatable)
pin-drift --config ../pin_drift.yml
pin-drift --context ht@dgxarley  # running versions from this context
pin-drift --no-running           # repo against upstream only
```

| Option               | Env var                  | Description                                   |
|----------------------|--------------------------|-----------------------------------------------|
| `-c`, `--config P`   | `PIN_DRIFT_CONFIG`       | Pin declarations (default: search upwards)    |
| `-o`, `--only NAME`  | `PIN_DRIFT_ONLY`         | Check only this pin (repeatable)              |
| `--updates-only`     | `PIN_DRIFT_UPDATES_ONLY` | Hide the pins that are current and rolled out |
| `--context CTX`      | `PIN_DRIFT_CONTEXT`, `KUBE_CONTEXT` | kubeconfig context (default: active one) |
| `--no-running`       | `PIN_DRIFT_NO_RUNNING`   | Do not query the cluster                      |
| `-q`, `--quiet`      | `PIN_DRIFT_QUIET`        | Suppress the table, print the summary only    |
| `-v`, `--verbose`    | `PIN_DRIFT_VERBOSE`      | DEBUG logging, third-party libraries included |

### Exit codes

| Code | Meaning                                                          |
|------|------------------------------------------------------------------|
| `0`  | No update available (pins current or unclear)                    |
| `1`  | At least one pin has a major/minor/patch update — pipeline gate  |
| `2`  | No usable `pin_drift.yml`, an unknown `--only` name, or an unusable explicit `--context` |

## drift.py

Both checks behind one CLI, so `--help` lists them side by side instead of only
showing the options of whichever tool happens to be the default.

Needs the optional dependencies: `pip install 'dgxarley[k3s]'`. Also runnable as
`python -m dgxarley.k3shelperstuff.drift`.

```bash
k3s-drift --help               # lists pin and keel
k3s-drift pin [pin options]    # same options as pin_drift
k3s-drift keel [keel options]  # same options as keel_drift
```

`DRIFT_PROG_NAME` overrides the program name shown in `--help`, for wrapper
scripts (`./drift.sh`, `scripts/check_pin_drift.sh`) that call the module.
