"""Preparation-only PNP v2 spec resolution. No operational imports or writes.

The tokens belong to this resolver, not s4d.config.load_config. Never pass an
unresolved template directly to a trainer. Legacy task modules stay untouched.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

TASK = "pick-and-place-v2"
SPEC = "configs/realworld/tasks/pick-and-place-v2.yaml"
STEM = "pnpv2-20261117"
METHODS = (
    "ours_orig",
    "ours_ch01",
    "dynamo",
    "reviwo",
    "sincro",
    "snerl",
    "mvp3d_single",
    "mvp3d_multiview",
    "e2e_cnn",
    "e2e_cnn_noaug",
)
RGB_METHODS = ("dynamo", "reviwo", "e2e_cnn", "e2e_cnn_noaug")
TOKEN = re.compile(r"\$\{pnpv2\.([a-z0-9_]+)\}")


def row_id(stage, method, seed=None):
    return f"{STEM}-{stage}-{method}-{TASK}-S" + (f"-s{seed}" if seed is not None else "")


def load_spec(path):
    spec = yaml.safe_load(Path(path).read_text())
    if (
        spec.get("schema") != "pick-and-place-v2-task/1"
        or spec.get("task") != TASK
        or spec.get("variants") != ["S"]
        or spec.get("seeds") != [0, 1, 2]
    ):
        raise ValueError("only pick-and-place-v2 S, seeds 0/1/2 are prepared")
    c = spec["collection"]
    if (
        c.get("failures_enabled") is not False
        or type(c.get("expected_success")) is not int
        or c["expected_success"] != 100
        or c.get("split_seed") != 0
        or c.get("validation_fraction") != 0.1
    ):
        raise ValueError("S-only, full 100-success collection and seeded 90/10 split required; no inherited amendment")
    if spec.get("calibration") != {"criteria": "rig_loma-release/2", "acceptance_override": None}:
        raise ValueError("standard task calibration required; no acceptance override for pick-and-place-v2")
    if set(spec["roots"]) != {"raw", "processed", "publication", "calibration", "robot_assets"}:
        raise ValueError("explicit task roots required")
    for path in spec["roots"].values():
        if not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts or "${" in path:
            raise ValueError("roots must be absolute paths without traversal or tokens")
    if set(spec["scene"]) != {"lookat", "pose_scale", "bounds"}:
        raise ValueError("explicit pending/measured scene parameters required")
    return spec


def values(spec):
    r = spec["roots"]
    base = f"{r['publication']}/{TASK}/S"
    return {
        **spec["scene"],
        "sam3_prompts": spec["sam3_prompts"],
        "raw_success": f"{r['raw']}/{TASK}/success",
        "processed": f"{r['processed']}/{TASK}",
        "manifest": f"{base}/manifest.json",
        "split": f"{base}/split_seed0.json",
        "outcomes": f"{base}/outcomes.json",
        **{
            f"{view}_{kind}": f"{base}/{view}/{name}"
            for view in ("rgb", "geometry")
            for kind, name in (("store", "store.yaml"), ("gate", "gate_report.json"))
        },
        "rig": f"{r['calibration']}/{TASK}/rig-{TASK}-accepted.json",
        "robot_urdf": f"{r['robot_assets']}/{TASK}/confirmed-robot.urdf",
        "workspace_stats": f"runs/{row_id('stats', 'ours')}/workspace_stats.json",
    }


def resolve(obj, spec):
    """Resolve recursively, preserving YAML types for whole-value tokens (including null)."""
    lookup = values(spec)

    def visit(value):
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, str):
            return value
        match = TOKEN.fullmatch(value)
        if match:
            if match[1] not in lookup:
                raise ValueError(f"unknown PNP v2 token: {match[1]}")
            return copy.deepcopy(lookup[match[1]])
        if "${" in value:
            raise ValueError(f"unknown or embedded template token: {value}")
        return value

    return visit(obj)


def config(path, spec):
    return resolve(yaml.safe_load(Path(path).read_text()), spec)


def require_collected(spec):
    if any("__TODO_" in p for p in spec["roots"].values()):
        raise ValueError("dataset not collected: resolve TODO roots in the single task spec first")
