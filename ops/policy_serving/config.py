"""``configs/realworld/policy_serving.yaml`` reader + the staged policy CATALOG (``<checkpoint_root>/catalog.json``):
defaults, key validation, path resolution. No model imports (torch-free).

PyYAML is imported only when a YAML file is actually read, so the robot-side client works without it (pass host/port
on the command line, or a JSON file with the same structure).
"""

from __future__ import annotations

import copy
import json
import math
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
# Catalog-only key ``actor`` (default onestep = FQL's deployed policy): ``bc_flow`` serves the checkpoint's BC flow
# instead (s4d.rwlearn.inference.ACTORS). EXPERIMENTAL (DIAG 2026-11-11): such an entry never replaces a policy id - it
# has its own id ``<base id>+bcflow``, the flag ``experimental_bc_flow``, the base entry's exact artefacts, no alias
# points at it, its status follows the base (never servable without a servable base), and it is execution-eligible
# only for a deployable base under a user decision that names bc_flow and is bound to the checkpoint sha256. Static
# config entries cannot select an actor.
BCFLOW_ARTEFACT_KEYS = ("checkpoint", "sha256", "export", "training_contract", "dataset", "condition", "task")
BCFLOW_SUFFIX, BCFLOW_FLAG = "+bcflow", "experimental_bc_flow"
ACTORS = ("onestep", "bc_flow")  # == s4d.rwlearn.inference.ACTORS (pinned by a test; no torch import here)

# Staged policy catalog (written by the staging tool next to the staged files, read by the server at start, on SIGHUP
# and when a HELLO names an unknown id). Paths are relative to the checkpoint root unless absolute.
CATALOG_SCHEMA = "s4d-policy-catalog/1"
CATALOG_TOP_KEYS = frozenset({"schema", "generated_utc", "criteria_sha256", "policies", "aliases", "note"})
CATALOG_ENTRY_KEYS = frozenset(POLICY_KEYS | {"seed", "run", "status", "flags", "reasons", "input_spec", "criteria",
                                             "usage", "usage_reasons", "execution_eligible",
                                             "execution_decision", "actor"})
CATALOG_REQUIRED_KEYS = frozenset({"checkpoint", "sha256", "condition", "task", "status"})
CATALOG_STATUSES = ("deployable", "restricted", "quarantined")  # restricted: served, a criterion FAILED (see reasons)
SERVABLE_STATUSES = ("deployable", "restricted")
CATALOG_USAGES = ("execution", "dry_run_replay_only", "none")  # usage restriction published to the robot-PC client
EXECUTION_DECISION_KEYS = ("decided_by", "at", "text")


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


def _finite_json(x) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(_finite_json(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_finite_json(v) for v in x)
    return True


def _check_eligibility(source, pid, entry: dict) -> None:
    """``execution_eligible: true`` only for a ``deployable`` entry with usage ``execution`` and a non-empty user
    ``execution_decision`` {decided_by: user, at, text} (user decision 2026-11-10 21:10); restricted / quarantined
    entries can never be eligible; usage ``execution`` without eligibility is refused (fail closed)."""
    eligible, usage, decision = entry.get("execution_eligible") is True, entry.get("usage"), entry.get("execution_decision")
    if eligible:
        if entry["status"] != "deployable":
            raise ValueError(f"{source}: policy {pid}: a {entry['status']} entry is never execution-eligible")
        if usage != "execution":
            raise ValueError(f"{source}: policy {pid}: an execution-eligible entry has usage 'execution'")
        if not isinstance(decision, dict) or decision.get("decided_by") != "user" or \
                not all(isinstance(decision.get(k), str) and decision.get(k).strip() for k in EXECUTION_DECISION_KEYS):
            raise ValueError(f"{source}: policy {pid}: execution eligibility needs an execution_decision record "
                             f"{{decided_by: user, at, text}}")
    elif usage == "execution":
        raise ValueError(f"{source}: policy {pid}: usage 'execution' requires execution_eligible: true")
    elif decision is not None and not eligible:
        raise ValueError(f"{source}: policy {pid}: an execution_decision on a non-eligible entry")


def _check_actor(source, pid: str, entry: dict) -> None:
    """``actor`` is onestep (default) or bc_flow; a bc_flow entry is a separately named experimental variant
    (``BCFLOW_SUFFIX`` id, ``BCFLOW_FLAG``) and eligible only under a user decision whose text names bc_flow; a
    onestep entry never carries the suffix (no silent replacement of a policy by its BC flow)."""
    actor = entry.get("actor", "onestep")
    if actor not in ACTORS:
        raise ValueError(f"{source}: policy {pid}: actor {actor!r} not in {ACTORS}")
    if (actor == "bc_flow") != str(pid).endswith(BCFLOW_SUFFIX):
        raise ValueError(f"{source}: policy {pid}: a bc_flow entry (and only it) has an id ending in {BCFLOW_SUFFIX!r}")
    if actor == "bc_flow":
        if BCFLOW_FLAG not in entry.get("flags", []):
            raise ValueError(f"{source}: policy {pid}: a bc_flow entry carries the flag {BCFLOW_FLAG!r}")
        decision = entry.get("execution_decision") or {}
        if entry.get("execution_eligible") is True and ("bc_flow" not in str(decision.get("text", ""))
                                                         or decision.get("checkpoint_sha256") != entry.get("sha256")):
            raise ValueError(f"{source}: policy {pid}: an execution-eligible bc_flow entry needs a user decision "
                             "naming bc_flow, bound to its checkpoint_sha256")


def _check_bcflow_base(source, pid: str, entry: dict, policies: dict) -> None:
    """A bc_flow variant serves exactly its base entry's artefacts and never outlives / outranks it: base present,
    same artefacts, servable only while the base is (restricted base -> restricted variant), eligible only on a
    deployable base."""
    base = policies.get(pid[:-len(BCFLOW_SUFFIX)])
    if base is None or base.get("actor", "onestep") != "onestep":
        if entry["status"] != "quarantined":
            raise ValueError(f"{source}: policy {pid}: a bc_flow variant without its one-step base is quarantined")
        return
    if any(entry.get(k) != base.get(k) for k in BCFLOW_ARTEFACT_KEYS):
        raise ValueError(f"{source}: policy {pid}: a bc_flow variant serves exactly its base entry's artefacts")
    rank = {"deployable": 2, "restricted": 1, "quarantined": 0}
    if rank[entry["status"]] > rank[base["status"]]:
        raise ValueError(f"{source}: policy {pid}: a bc_flow variant cannot be {entry['status']} while its base is "
                         f"{base['status']}")
    if entry.get("execution_eligible") is True and base["status"] != "deployable":
        raise ValueError(f"{source}: policy {pid}: a bc_flow variant is eligible only on a deployable base")


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
        _check_eligibility(source, pid, entry)
        _check_actor(source, pid, entry)
        if not _finite_json(entry):
            raise ValueError(f"{source}: policy {pid}: non-finite number (NaN / Infinity) in the entry")
        if not isinstance(entry.get("usage_reasons", []), list) or \
                not all(isinstance(x, str) for x in entry.get("usage_reasons", [])):
            raise ValueError(f"{source}: policy {pid}: usage_reasons must be a list of strings")
        if not isinstance(entry.get("criteria", {}), dict):
            raise ValueError(f"{source}: policy {pid}: criteria must be an object")
        out["policies"][str(pid)] = copy.deepcopy(entry)
    for pid, entry in out["policies"].items():
        if entry.get("actor", "onestep") == "bc_flow":
            _check_bcflow_base(source, pid, entry, out["policies"])
    aliases = raw.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError(f"{source}: aliases must be an object")
    for alias, pid in aliases.items():
        if alias in out["policies"]:
            raise ValueError(f"{source}: alias {alias!r} shadows a policy id")
        if pid not in out["policies"]:
            raise ValueError(f"{source}: alias {alias!r} points at unknown policy {pid!r}")
        if out["policies"][pid].get("actor", "onestep") != "onestep":
            raise ValueError(f"{source}: alias {alias!r} points at the experimental {pid!r} (select it by its id)")
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
