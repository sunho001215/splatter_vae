"""pick-and-place-v2 task spec, identities and template resolution. Pure: no ops imports, no writes.

The ``${pnpv2.*}`` tokens belong to this resolver, not ``s4d.config.load_config``. Queue rows pass the templates
together with ``--set`` overrides for every token-bearing field; the pnpv2 entrypoints refuse a leftover token.
Legacy task registries stay untouched (``pnpv2_runtime.install`` is process-local).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import yaml

TASK = "pick-and-place-v2"
SPEC = "configs/realworld/tasks/pick-and-place-v2.yaml"
ARRIVAL_SCHEMA = "pick-and-place-v2-arrival/1"
# Every new queue row and every new prep-dispatcher job uses this stem. The v1 families (rwv2-20261113-*,
# rwv3-20261115-*, rwnt-/rwnq-/rwa100-/rwcal-*) are discontinued by prefix and must never match a v2 identity.
STEM = "pnpv2-20261117"
# Declared disk of pnp-v2 remote prep jobs (Phase B, geometry publication, future Phase A), user-approved measured
# charge 2026-11-18: 52 finished Phase-A episodes wrote 0.81-1.40 GiB each (median 1.01; workers are read-only with
# RAM tmpfs, so disk = outputs) and drawer-open-v2 Phase B adds <= 0.09 GiB per episode; 4 GiB = 2.9x the maximum.
# The established compiler default (32 GiB) let the 15.5 % disk floor cap the remote route at ~13 concurrent jobs.
PREP_DISK_GIB = 4
PAUSE = "pnpv2-await-placement-20261117"
V1_PAUSE = "pick-and-place-v1-discontinued-20261117"
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
# Calibration-independent (s4d/rwlearn/gate.py CONSUMERS geometry=False): own RGB publication and wave.
RGB_METHODS = ("dynamo", "reviwo", "e2e_cnn", "e2e_cnn_noaug")
GEOMETRY_METHODS = tuple(m for m in METHODS if m not in RGB_METHODS)
VIEWS = ("rgb", "geometry")
BUNDLE_FILES = (
    ("store", "store.yaml"),
    ("manifest", "manifest.json"),
    ("split", "split_seed0.json"),
    ("outcomes", "outcomes.json"),
    ("gate", "gate_report.json"),
)
TOKEN = re.compile(r"\$\{pnpv2\.([a-z0-9_]+)\}")
# Queue-row entrypoints: each installs the new task process-locally, then runs the established script unchanged.
WRAPPERS = {
    f"scripts/pnpv2_{name}": f"scripts/{name}"
    for name in (
        "realworld_dsreset_check.py",
        "compute_workspace_stats.py",
        "realworld_native_depth.py",
        "train.py",
        "train_dynamo.py",
        "train_reviwo.py",
        "train_sincro.py",
        "train_snerl.py",
        "train_mvp3d.py",
        "export_encoder.py",
        "realworld_dsreset_artifact.py",
        "realworld_fql_dataset.py",
        "train_realworld_fql.py",
        "dsreset_evaluate_ours_task_length.py",
    )
}


def select_task(task):
    """Select a supported prep task BEFORE importing consumers, in a dedicated CLI process.

    Defaults remain the established pnp-v2 identities. Reconfiguration of an imported pipeline is refused;
    this is not a task switch inside a worker or a shared Python service.
    """
    import sys

    global TASK, SPEC, STEM, PAUSE
    if task != "ball-hiding":
        raise ValueError("unsupported prep task")
    consumers = [name for name in sys.modules if name.startswith(("ops.jobs.pnpv2_", "s4d.rwlearn.pnpv2_"))
                 and name != __name__]
    if consumers:
        raise ValueError("select task before importing pipeline consumers: " + ", ".join(consumers))
    TASK, SPEC, STEM, PAUSE = ("ball-hiding", "configs/realworld/tasks/ball-hiding.yaml",
                              "bh-20261120", "bh-await-placement-20261120")


def row_id(stage, method, seed=None):
    return f"{STEM}-{stage}-{method}-{TASK}-S" + (f"-s{seed}" if seed is not None else "")


def dispatcher_stem(stage):
    """Prep-dispatcher job-id stem (Phase A / calibration / Phase B / follow-ups); never a v1 prefix."""
    return f"{STEM}-{stage}"


def _absolute(path):
    return isinstance(path, str) and Path(path).is_absolute() and ".." not in Path(path).parts and "${" not in path


def load_spec(path, overlay=None):
    """Validate the repo spec; ``overlay`` is the per-arrival confirmation document (arrival.json) or None."""
    spec = yaml.safe_load(Path(path).read_text())
    if (
        spec.get("schema") != "pick-and-place-v2-task/2"
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
    roots = {"raw", "remote_raw", "processed", "remote_processed", "publication", "work", "robot_assets"}
    if set(spec["roots"]) != roots or not all(_absolute(p) for p in spec["roots"].values()):
        raise ValueError("explicit absolute task roots required (no traversal or tokens)")
    if not spec["roots"]["remote_processed"].startswith("/home/compu/kaist/sunho/realworld/") or not spec["roots"][
        "publication"
    ].startswith("/home/compu/kaist/sunho/realworld/"):
        raise ValueError("published inputs must live under the verified remote real-world root")
    a = spec.get("arrival") or {}
    if (
        a.get("completion_marker") != "success.VERIFIED"
        or type(a.get("stable_seconds")) is not int
        or a["stable_seconds"] < 60
    ):
        raise ValueError("arrival needs the VERIFIED marker name and a stable-inventory window >= 60 s")
    if set(spec["scene"]) != {"lookat", "pose_scale", "bounds"}:
        raise ValueError("explicit pending/measured scene parameters required")
    if spec.get("sam3_prompts") is not None:
        raise ValueError("SAM3 prompts are confirmed per arrival (arrival.json), never in the repo spec")
    if overlay is not None:
        spec = apply_arrival(spec, overlay)
    return spec


def apply_arrival(spec, doc):
    """Merge the per-arrival confirmations; refuses anything else (no root/science override through this file)."""
    if not isinstance(doc, dict) or doc.get("schema") != ARRIVAL_SCHEMA or doc.get("task") != TASK:
        raise ValueError("arrival.json must be a pick-and-place-v2-arrival/1 document")
    extra = set(doc) - {"schema", "task", "sam3_prompts", "robot_urdf_sha256", "confirmed_by", "notes"}
    if extra:
        raise ValueError(f"arrival.json may only confirm prompts/robot asset: {sorted(extra)}")
    prompts = doc.get("sam3_prompts")
    if (
        not isinstance(prompts, dict)
        or not prompts
        or not all(isinstance(k, str) and k for k in prompts)
        or set(prompts.values()) - {"robot", "object", "fixture"}
        or "robot" not in prompts.values()
        or "object" not in prompts.values()
    ):
        raise ValueError("arrival.json needs reviewed SAM3 prompts with robot and object classes")
    if not re.fullmatch(r"[0-9a-f]{64}", str(doc.get("robot_urdf_sha256", ""))) or not str(
        doc.get("confirmed_by", "")
    ).strip():
        raise ValueError("arrival.json needs the confirmed robot URDF sha256 and a named confirmer")
    out = copy.deepcopy(spec)
    out["sam3_prompts"] = dict(prompts)
    out["robot_urdf_sha256"] = doc["robot_urdf_sha256"]
    return out


def read_arrival(spec):
    """The confirmation document if present (None otherwise); never required for read-only status."""
    path = Path(values(spec)["arrival_doc"])
    return json.loads(path.read_text()) if path.is_file() else None


def values(spec):
    r = spec["roots"]
    w = r["work"]
    out = {
        **spec["scene"],
        "sam3_prompts": spec["sam3_prompts"],
        "raw_task": f"{r['raw']}/{TASK}",
        "raw_success": f"{r['raw']}/{TASK}/success",
        "remote_raw_success": f"{r['remote_raw']}/{TASK}/success",
        "processed": r["processed"],
        "remote_processed": r["remote_processed"],
        "work": w,
        "arrival_doc": f"{w}/arrival.json",
        "ingest": f"{w}/ingest",
        "manifest": f"{w}/ingest/S/manifest.json",
        "prep_plan": f"{w}/ingest/prep-plan.json",
        "phase_a": f"{w}/phaseA",
        "calib": f"{w}/calib",
        "phase_b": f"{w}/phaseB",
        "snapshot": f"{w}/ingest-snapshot",
        "rig": f"{r['remote_processed']}/calibration/{TASK}/rig-{TASK}-accepted.json",
        "rig_local": f"{r['processed']}/calibration/{TASK}/rig-{TASK}-accepted.json",
        "scene_report": f"{r['processed']}/measurements/{TASK}/scene.json",
        "robot_urdf": f"{r['robot_assets']}/{spec['robot_urdf']}",
        "workspace_stats": f"runs/{row_id('stats', 'ours')}/workspace_stats.json",
    }
    if TASK == "ball-hiding":
        out["rgb_source"] = f"{w}/normalize"
    for view in VIEWS:
        out[f"{view}_dir"] = f"{r['publication']}/{TASK}/S/{view}"
        out[f"{view}_dir_local"] = f"{w}/publication/S/{view}"
        for key, name in BUNDLE_FILES:
            out[f"{view}_{key}"] = f"{out[f'{view}_dir']}/{name}"
            out[f"{view}_{key}_local"] = f"{out[f'{view}_dir_local']}/{name}"
    return out


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


def leftover_tokens(cfg):
    """Dotted keys whose value is still an unresolved ``${pnpv2.*}`` template token."""
    found = []

    def visit(value, key):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, f"{key}.{k}" if key else k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(v, f"{key}[{i}]")
        elif isinstance(value, str) and "${pnpv2." in value:
            found.append(key)

    visit(cfg, "")
    return found
