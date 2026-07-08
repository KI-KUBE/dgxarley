# Strategic merge of ops-managed keys into /opt/data/config.yaml +
# /opt/data/.env, run by the hermes-agent `merge-config` initContainer.
#
# We can NOT mount the seed files as subPath of a ConfigMap/Secret directly —
# that makes them bind-mounts, and Hermes' `atomic_yaml_write()` does rename(2)
# over the target which fails with EBUSY on bind-mounts (the original 2026 trace
# was triggered by a Settings -> Theme save).
#
# Strategy: every top-level key present in the seed (provider, default-model,
# base_url for config.yaml; OPENAI_API_KEY for .env) is enforced on every pod
# start. Keys the user added (theme, tool toggles, extra provider keys) are
# preserved. Lets ops update the cluster default and propagate to existing pods
# on next restart, while still letting users customise non-managed bits in the
# WebUI without losing their work.
#
# Runs under the agent image's venv (PyYAML available, guaranteed cached on the
# node). Target ownership comes from the MERGE_UID/MERGE_GID env vars so this
# file stays static/cluster-wide (no per-user Jinja2 templating).
#
# Shared by BOTH the hermes-agent and hermes-webui pods (the webui spawns its
# own `hermes` subprocesses reading the same config.yaml/.env). The webui pod
# additionally passes WEBUI_SETTINGS_B64 (base64 of the per-user webui_settings
# JSON) to also merge ops-managed WebUI preferences into
# /opt/data/.webui/settings.json; the agent pod never sets it and skips that.
import base64
import copy
import json
import os
import re
from typing import Any

import yaml

UID, GID = int(os.environ["MERGE_UID"]), int(os.environ["MERGE_GID"])


def deep_override(user: dict[str, Any], ops: dict[str, Any]) -> None:
    for k, v in ops.items():
        if isinstance(v, dict) and isinstance(user.get(k), dict):
            deep_override(user[k], v)
        else:
            user[k] = v


# .env — KEY=VALUE merge (ops wins for managed keys, others kept)
def parse_env(p: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v
    return out


# config.yaml — strategic merge
ops = yaml.safe_load(open("/seed-config/config.yaml")) or {}
try:
    user = yaml.safe_load(open("/opt/data/config.yaml")) or {}
except FileNotFoundError:
    user = {}
# custom_providers is a LIST, so deep_override would blunt-replace it (losing any
# provider the user added via `hermes model`). Capture the user's list first and
# merge BY NAME after: our ops-managed entries win, user-added ones are kept.
_user_cps_raw = user.get("custom_providers")
_user_cps = _user_cps_raw if isinstance(_user_cps_raw, list) else []
deep_override(user, ops)
_ops_cps = ops.get("custom_providers") if isinstance(ops.get("custom_providers"), list) else []
if _ops_cps:
    _ops_names = {cp.get("name") for cp in _ops_cps if isinstance(cp, dict)}
    user["custom_providers"] = list(_ops_cps) + [
        cp for cp in _user_cps if isinstance(cp, dict) and cp.get("name") not in _ops_names
    ]
# Patch the LiteLLM virtual key from the (Secret-backed) .env into
# model.api_key so Hermes' custom-provider resolver picks it up as
# cfg_api_key (#1760). REQUIRED: Hermes host-gates OPENAI_API_KEY to
# openai.com/azure (GHSA / #28660), so the env fallback never
# authenticates against our litellm.* base_url. Kept out of the
# ConfigMap (secret-free); injected here at startup only.
ops_env = parse_env("/seed-secret/.env")
_openai_key = (ops_env.get("OPENAI_API_KEY") or "").strip()
if _openai_key:
    user.setdefault("model", {})["api_key"] = _openai_key
    # Same patch for the auxiliary vision model (image tasks) — ONLY when the
    # seed config actually declares auxiliary.vision (custom provider → our
    # litellm base_url). Hermes host-gates OPENAI_API_KEY to openai.com/azure,
    # so the aux vision client also needs cfg_api_key set explicitly. Guarded so
    # we never invent an auxiliary block when no vision model is configured.
    if isinstance(user.get("auxiliary"), dict) and isinstance(user["auxiliary"].get("vision"), dict):
        user["auxiliary"]["vision"]["api_key"] = _openai_key
# Custom providers — inject each entry's inline api_key (config-var, NOT key_env,
# per the deployment design) from the Secret-backed .env so keys stay out of the
# non-secret ConfigMap. Generic + provider-agnostic: the key for entry <name> is
# read from CUSTOM_PROVIDER_<NAME>_API_KEY, where <NAME> is name.upper() with every
# non-alphanumeric char replaced by "_" (MUST match hermes_env.j2's sanitizer).
# Entries that carry their own key_env (or already have an api_key) are left alone.
for _cp in user.get("custom_providers") or []:
    if not isinstance(_cp, dict) or _cp.get("key_env") or _cp.get("api_key"):
        continue
    _cp_san = re.sub(r"[^A-Z0-9]", "_", str(_cp.get("name", "")).upper())
    _cp_key = (ops_env.get(f"CUSTOM_PROVIDER_{_cp_san}_API_KEY") or "").strip()
    if _cp_key:
        _cp["api_key"] = _cp_key
with open("/opt/data/config.yaml", "w") as f:
    yaml.safe_dump(user, f, sort_keys=False)
os.chown("/opt/data/config.yaml", UID, GID)
os.chmod("/opt/data/config.yaml", 0o600)
user_env = parse_env("/opt/data/.env")
user_env.update(ops_env)
with open("/opt/data/.env", "w") as f:
    for k, v in user_env.items():
        f.write(f"{k}={v}\n")
os.chown("/opt/data/.env", UID, GID)
os.chmod("/opt/data/.env", 0o600)

# --- Named profiles: seed /opt/data/profiles/<name>/{config.yaml,.env,SOUL.md} ---
# Each profile is its own HERMES_HOME; Hermes' list_profiles()/dashboard switcher
# discover them by scanning this dir (no registry). A profile's config is the
# DEFAULT profile's final config (deep-copied) deep-merged with the per-profile
# overrides from profiles.json, with model.api_key RE-injected from the Secret .env.
# So a "custom"-provider profile's main model talks DIRECT to its own endpoint (own
# key) while the inherited auxiliary.vision stays on LiteLLM (separate api_key field
# — no conflict). Same ops-wins / user-edits-preserved semantics as the default profile.
try:
    _profiles_spec = json.loads(open("/seed-config/profiles.json").read()) or []
except (FileNotFoundError, json.JSONDecodeError):
    _profiles_spec = []
try:
    _profiles_secrets = json.loads(open("/seed-secret/profiles-secrets.json").read()) or {}
except (FileNotFoundError, json.JSONDecodeError):
    _profiles_secrets = {}
if _profiles_spec:
    os.makedirs("/opt/data/profiles", exist_ok=True)
    os.chown("/opt/data/profiles", UID, GID)
for _p in _profiles_spec:
    _name = str(_p.get("name") or "").strip()
    if not _name:
        continue
    _pdir = os.path.join("/opt/data/profiles", _name)
    os.makedirs(_pdir, exist_ok=True)
    # .env — inherit the default profile's .env, then add this profile's own secrets
    _penv = dict(user_env)
    _penv.update(_profiles_secrets.get(_name, {}))
    # config.yaml — default config (deep copy) + profile overrides
    _pops = copy.deepcopy(user)
    _pmodel = _pops.setdefault("model", {})
    if _p.get("model_kind") == "custom":
        _pmodel["provider"] = "custom"
        if _p.get("base_url"):
            _pmodel["base_url"] = _p["base_url"]
        if _p.get("model_default"):
            _pmodel["default"] = _p["model_default"]
        else:
            _pmodel.pop("default", None)
    elif _p.get("model_default"):
        _pmodel["default"] = _p["model_default"]
    if _p.get("terminal_cwd"):
        _pops.setdefault("terminal", {})["cwd"] = _p["terminal_cwd"]
    try:
        _puser = yaml.safe_load(open(os.path.join(_pdir, "config.yaml"))) or {}
    except (FileNotFoundError, yaml.YAMLError):
        _puser = {}
    deep_override(_puser, _pops)
    # Re-inject this profile's model.api_key (overrides the inherited default's key)
    _mkey = _penv.get(_p.get("model_api_key_env") or "", "").strip()
    if _mkey:
        _puser.setdefault("model", {})["api_key"] = _mkey
    with open(os.path.join(_pdir, "config.yaml"), "w") as f:
        yaml.safe_dump(_puser, f, sort_keys=False)
    with open(os.path.join(_pdir, ".env"), "w") as f:
        for _k, _v in _penv.items():
            f.write(f"{_k}={_v}\n")
    # SOUL.md — seed only when absent (never clobber a personality the user edited)
    _soul = str(_p.get("soul") or "").strip()
    _soul_path = os.path.join(_pdir, "SOUL.md")
    if _soul and not os.path.exists(_soul_path):
        with open(_soul_path, "w") as f:
            f.write(_soul + "\n")
    # ownership + perms (whole profile subtree to the runtime uid/gid)
    for _root, _dirs, _files in os.walk(_pdir):
        os.chown(_root, UID, GID)
        for _fn in _files:
            os.chown(os.path.join(_root, _fn), UID, GID)
    os.chmod(os.path.join(_pdir, "config.yaml"), 0o600)
    os.chmod(os.path.join(_pdir, ".env"), 0o600)

# WebUI-only: merge STATE_DIR/settings.json with the per-user webui_settings
# (passed as base64 JSON via WEBUI_SETTINGS_B64 by the webui pod only). Mirrors
# the config.yaml/.env merge above — ops-managed keys WIN, all other keys the
# user has set via the Preferences panel (POST /api/settings) are preserved.
# The ops-relevant toggles live in api/config.py:_SETTINGS_DEFAULTS upstream
# (sidebar_density, show_cli_sessions, simplified_tool_calling, busy_input_mode,
# …); none has an env/config override. base64 round-trip because the rendered
# JSON contains lowercase true/false/null which are not valid Python literals.
_settings_b64 = os.environ.get("WEBUI_SETTINGS_B64", "").strip()
ops_settings = json.loads(base64.b64decode(_settings_b64)) if _settings_b64 else {}
if ops_settings:
    state_dir = "/opt/data/.webui"
    settings_path = os.path.join(state_dir, "settings.json")
    os.makedirs(state_dir, exist_ok=True)
    os.chown(state_dir, UID, GID)
    os.chmod(state_dir, 0o700)
    try:
        user_settings = json.load(open(settings_path))
        if not isinstance(user_settings, dict):
            user_settings = {}
    except (FileNotFoundError, json.JSONDecodeError):
        user_settings = {}
    user_settings.update(ops_settings)  # ops keys win
    with open(settings_path, "w") as f:
        json.dump(user_settings, f, indent=2, sort_keys=True)
    os.chown(settings_path, UID, GID)
    os.chmod(settings_path, 0o600)
