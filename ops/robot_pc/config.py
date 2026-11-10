"""``configs/robot_pc/robot_pc.yaml`` reader (unknown keys refused, paths resolved against ``root``) and the factories
the entry points share: action limits of a task, the persistent ``PolicyClient``, the M5 helper."""

from __future__ import annotations

import copy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "configs/robot_pc/robot_pc.yaml"
SECTIONS = {"root", "server", "cameras", "assets", "tasks", "limits", "executor", "static_latency_state"}
TASK_KEYS = {"rig_document", "m5_max_sync_ns", "workspace_min_m", "workspace_max_m"}


def load(path: str | Path | None = None, *, root: str | Path | None = None) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path or DEFAULT_PATH).read_text()) or {}
    unknown = set(cfg) - SECTIONS
    if unknown:
        raise ValueError(f"unknown robot-PC config sections {sorted(unknown)}")
    for task, entry in (cfg.get("tasks") or {}).items():
        if set(entry) - TASK_KEYS:
            raise ValueError(f"task {task}: unknown keys {sorted(set(entry) - TASK_KEYS)}")
    cfg = copy.deepcopy(cfg)
    cfg["root"] = str(Path(root or cfg.get("root") or ".").expanduser())
    return cfg


def path(cfg: dict, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(cfg["root"]) / p


def task_entry(cfg: dict, task: str) -> dict:
    tasks = cfg.get("tasks") or {}
    if task not in tasks:
        raise ValueError(f"task {task!r} not configured ({sorted(tasks)})")
    return tasks[task]


def limits(cfg: dict, task: str):
    from ops.robot_pc.validation import ActionLimits

    t = task_entry(cfg, task)
    return ActionLimits.from_config({**(cfg.get("limits") or {}), "workspace_min_m": t["workspace_min_m"],
                                     "workspace_max_m": t["workspace_max_m"]})


def client(cfg: dict, policy_id: str, choices, *, name: str, token_file: str | Path | None = None, **overrides):
    """The persistent link to the policy server (``ops.policy_serving.client.PolicyClient``)."""
    from ops.policy_serving.client import PolicyClient

    from ops.robot_pc.safety import admit_endpoint

    s = {**(cfg.get("server") or {}), **{k: v for k, v in overrides.items() if v is not None}}
    admit_endpoint(s.get("host", "127.0.0.1"), int(s.get("port", 47100)))
    tok_path = path(cfg, str(token_file)) if token_file else path(cfg, s.get("token_file"))
    token = tok_path.read_bytes().strip() if tok_path is not None and tok_path.is_file() else None
    return PolicyClient(s.get("host", "127.0.0.1"), int(s.get("port", 47100)), policy_id=policy_id, choices=choices,
                        token=token, connect_timeout_s=s.get("connect_timeout_s", 3.0),
                        io_timeout_s=s.get("io_timeout_s", 5.0), step_timeout_s=s.get("step_timeout_s", 0.5),
                        reconnect_backoff_s=s.get("reconnect_backoff_s", (0.05, 0.1, 0.25, 0.5, 1.0)),
                        ttl_ms=s.get("ttl_ms", 100.0), hello_timeout_s=s.get("hello_timeout_s", 20.0), name=name)


def m5(cfg: dict, task: str, cameras, period_s: float):
    """The validated live M5 helper of a multi-view session (asset + unmodified URDF from the bundle)."""
    from s4d.rwlearn.deploy.m5_live import LiveM5, SurfaceMotion

    a = cfg.get("assets") or {}
    motion = SurfaceMotion(path(cfg, a["m5_surface"]), path(cfg, a["urdf"]))
    return LiveM5(motion, cameras, period_s, int(task_entry(cfg, task)["m5_max_sync_ns"]))
