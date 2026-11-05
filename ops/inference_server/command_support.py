"""Configuration and source helpers shared by inference commands."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def spec(cfg: dict, name: str, **override):
    from s4d.realworld.workers import WorkerSpec

    w = {**cfg["workers"][name], **override}
    return WorkerSpec(name, w["python"], repo_dir=w.get("repo_dir"), expected_commit=w.get("expected_commit"),
                      checkpoint=w.get("checkpoint"), timeout_s=float(w.get("timeout_s", 3600)))


def remote_pipeline_config(pipeline: dict, server: dict) -> dict:
    """Resolve worker locations in the container while leaving adapter and output paths on the host.

    Server worker fields override host-specific worker paths; model parameters merge so inference settings from
    the episode config survive; server-only workers (e.g. the kernel interpreter) are added.
    """
    from copy import deepcopy

    cfg = deepcopy(pipeline)
    workers = cfg.setdefault("workers", {})
    for name, remote in server["workers"].items():
        local = workers.get(name, {})
        workers[name] = {**local, **remote, "params": {**local.get("params", {}), **remote.get("params", {})}}
    return cfg


def local_head(*, repo=REPO, subprocess_module=subprocess) -> str:
    # The command wrapper passes its globals so existing overrides still apply.
    dirty = subprocess_module.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"local checkout {repo} has uncommitted changes; commit before a remote run")
    return subprocess_module.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 check=True).stdout.strip()


def pool_config(config: dict, head: str) -> dict:
    pool_cfg = {k: config[k] for k in ("ssh", "container", "workdir", "source_root", "retries", "backoff_s",
                                      "call_timeout_s", "hello_timeout_s", "workers", "env")}
    pool_cfg["expected_source_commit"] = head
    return pool_cfg
