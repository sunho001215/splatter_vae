"""``configs/realworld/policy_serving.yaml`` reader: defaults, key validation, path resolution. No model imports.

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
