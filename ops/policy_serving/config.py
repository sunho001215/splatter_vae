"""``configs/realworld/policy_serving.yaml`` reader + the staged policy CATALOG (``<checkpoint_root>/catalog.json``):
defaults, key validation, path resolution. No model imports (torch-free).

PyYAML is imported only when a YAML file is actually read, so the robot-side client works without it (pass host/port
on the command line, or a JSON file with the same structure).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "configs/realworld/policy_serving.yaml"

SERVER_DEFAULTS = {
    "bind_host": "127.0.0.1",
    "port": 47100,
    "device": "cuda",
    "checkpoint_root": None,
    "max_queue": 4,
    "session_idle_timeout_s": 30.0,
    "response_cache": 32,
    "io_timeout_s": 5.0,
    "max_header_bytes": 1 << 20,
    "max_payload_bytes": 32 << 20,
    "max_frames_per_step": 64,
    "history_capacity": None,
    "max_sessions_per_policy": 1,
    "max_connections": 16,
    "gpu_uuid": None,
    "warmup_iterations": 3,
    "warmup_native_hw": [480, 640],
    "default_policies": [],
    "torch_threads": None,
    "catalog": None,
    "preload": [],
    "gpu_memory_budget_mb": None,
    "max_resident_policies": None,
}
CLIENT_DEFAULTS = {
    "host": "127.0.0.1",
    "port": 47100,
    "connect_timeout_s": 3.0,
    "io_timeout_s": 5.0,
    "step_timeout_s": 0.5,
    "reconnect_backoff_s": [0.05, 0.1, 0.25, 0.5, 1.0],
    "ttl_ms": 100.0,
    "max_action_age_ms": 200.0,
}
POLICY_KEYS = {"checkpoint", "export", "dataset", "training_contract", "sha256", "condition", "task", "note"}

# Staged policy catalog (written by the staging tool next to the staged files, read by the server at start, on SIGHUP
# and when a HELLO names an unknown id). Paths are relative to the checkpoint root unless absolute.
CATALOG_SCHEMA = "s4d-policy-catalog/1"
CATALOG_TOP_KEYS = frozenset({"schema", "generated_utc", "criteria_sha256", "policies", "aliases", "note"})
CATALOG_ENTRY_KEYS = frozenset(POLICY_KEYS | {"seed", "run", "status", "flags", "reasons", "input_spec", "criteria",
                                             "usage", "usage_reasons", "execution_eligible"})
CATALOG_REQUIRED_KEYS = frozenset({"checkpoint", "sha256", "condition", "task", "status"})
CATALOG_STATUSES = ("deployable", "restricted", "quarantined")  # restricted: served, a criterion FAILED (see reasons)
SERVABLE_STATUSES = ("deployable", "restricted")
CATALOG_USAGES = ("dry_run_replay_only", "none", "execution")  # usage restriction published to the robot-PC client


def _load_file(path: Path) -> dict:
    text = Path(path).read_text()
    if Path(path).suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text) or {}


def load_config(path: str | Path | None = None) -> dict:
    """Full config with defaults; unknown keys are refused (a typo must not silently fall back to a default)."""
    raw = _load_file(Path(path) if path else DEFAULT_PATH)
    unknown = set(raw) - {"protocol_version", "server", "client", "policies"}
    if unknown:
        raise ValueError(f"unknown top-level policy_serving keys {sorted(unknown)}")
    cfg = {"protocol_version": raw.get("protocol_version"), "server": copy.deepcopy(SERVER_DEFAULTS),
           "client": copy.deepcopy(CLIENT_DEFAULTS), "policies": {}}
    for section, defaults in (("server", SERVER_DEFAULTS), ("client", CLIENT_DEFAULTS)):
        given = raw.get(section) or {}
        bad = set(given) - set(defaults)
        if bad:
            raise ValueError(f"unknown {section} keys {sorted(bad)}")
        cfg[section].update(given)
    for pid, entry in (raw.get("policies") or {}).items():
        bad = set(entry) - POLICY_KEYS
        if bad:
            raise ValueError(f"policy {pid}: unknown keys {sorted(bad)}")
        if "checkpoint" not in entry:
            raise ValueError(f"policy {pid}: checkpoint required")
        if "dataset" in entry and "training_contract" in entry:
            raise ValueError(f"policy {pid}: give dataset OR training_contract, not both")
        cfg["policies"][str(pid)] = dict(entry)
    return cfg


def validate_catalog(raw: dict, source="catalog") -> dict:
    """Checked catalog dict (see ``CATALOG_*``): unknown keys refused, status values checked, aliases point at existing
    ids and never shadow one. Returns ``{"schema", ..., "policies": {id: entry}, "aliases": {alias: id}}``."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: a catalog is a JSON object")
    bad = set(raw) - CATALOG_TOP_KEYS
    if bad:
        raise ValueError(f"{source}: unknown catalog keys {sorted(bad)}")
    if raw.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"{source}: schema {raw.get('schema')!r} != {CATALOG_SCHEMA!r}")
    policies = raw.get("policies") or {}
    if not isinstance(policies, dict):
        raise ValueError(f"{source}: policies must be an object")
    out = {k: copy.deepcopy(v) for k, v in raw.items() if k not in ("policies", "aliases")}
    out["policies"], out["aliases"] = {}, {}
    for pid, entry in policies.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: policy {pid}: entry must be an object")
        unknown, missing = set(entry) - CATALOG_ENTRY_KEYS, CATALOG_REQUIRED_KEYS - set(entry)
        if unknown or missing:
            raise ValueError(f"{source}: policy {pid}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        if entry["status"] not in CATALOG_STATUSES:
            raise ValueError(f"{source}: policy {pid}: status {entry['status']!r} not in {CATALOG_STATUSES}")
        if entry.get("dataset") is not None and entry.get("training_contract") is not None:
            raise ValueError(f"{source}: policy {pid}: give dataset OR training_contract, not both")
        for key in ("flags", "reasons"):
            if not isinstance(entry.get(key, []), list) or not all(isinstance(x, str) for x in entry.get(key, [])):
                raise ValueError(f"{source}: policy {pid}: {key} must be a list of strings")
        if entry["status"] in ("quarantined", "restricted") and not entry.get("reasons"):
            raise ValueError(f"{source}: policy {pid}: a {entry['status']} entry states its reasons")
        if "execution_eligible" in entry and not isinstance(entry["execution_eligible"], bool):
            raise ValueError(f"{source}: policy {pid}: execution_eligible must be true or false")
        if entry.get("usage", "dry_run_replay_only") not in CATALOG_USAGES:
            raise ValueError(f"{source}: policy {pid}: usage {entry.get('usage')!r} not in {CATALOG_USAGES}")
        if entry.get("execution_eligible") is True and (entry["status"] != "deployable"
                                                         or entry.get("usage") != "execution"):
            raise ValueError(f"{source}: policy {pid}: only a deployable entry with usage 'execution' can be "
                             "execution-eligible")
        if entry["status"] == "restricted" and entry.get("usage") not in ("dry_run_replay_only", "none"):
            raise ValueError(f"{source}: policy {pid}: a restricted entry is dry-run/replay only")
        if not isinstance(entry.get("usage_reasons", []), list) or \
                not all(isinstance(x, str) for x in entry.get("usage_reasons", [])):
            raise ValueError(f"{source}: policy {pid}: usage_reasons must be a list of strings")
        if not isinstance(entry.get("criteria", {}), dict):
            raise ValueError(f"{source}: policy {pid}: criteria must be an object")
        out["policies"][str(pid)] = copy.deepcopy(entry)
    aliases = raw.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError(f"{source}: aliases must be an object")
    for alias, pid in aliases.items():
        if alias in out["policies"]:
            raise ValueError(f"{source}: alias {alias!r} shadows a policy id")
        if pid not in out["policies"]:
            raise ValueError(f"{source}: alias {alias!r} points at unknown policy {pid!r}")
        out["aliases"][str(alias)] = str(pid)
    return out


def load_catalog(path: str | Path) -> dict:
    """``validate_catalog`` of a catalog JSON file."""
    return validate_catalog(json.loads(Path(path).read_text()), source=str(path))


def resolve(root: str | Path | None, value: str | None) -> Path | None:
    """Policy file path: absolute as given, else relative to ``checkpoint_root`` (``~`` expanded)."""
    if value is None:
        return None
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    if root is None:
        raise ValueError(f"relative policy path {value!r} needs server.checkpoint_root (or --checkpoint-root)")
    return Path(root).expanduser() / p
