"""Read-only checks of pick-and-place-v2 membership and publication evidence; no ops imports, no writes."""

import json
from pathlib import Path

from s4d.rwlearn.pnpv2_spec import TASK

RGB_CONSUMERS = ("dynamo", "reviwo", "e2e_cnn")
GEOMETRY_CONSUMERS = ("ours", "sincro", "snerl", "mvp3d")
STAGE_VIEW = {"prep": "rgb", "calib": "geometry"}  # gate rows reuse the established --stage spellings


def no_override(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("acceptance_override", "acceptance_overrides") and value:
                raise ValueError("pick-and-place-v2 has no calibration acceptance override")
            no_override(value)
    elif isinstance(obj, list):
        for value in obj:
            no_override(value)


def manifest_check(manifest, spec):
    from s4d.rwlearn.pnpv2_arrival import task_known

    with task_known() as dsreset:
        dsreset.validate_manifest(manifest)
    if (
        manifest["task"] != TASK
        or manifest["variant"] != "S"
        or len(manifest["episodes"]) != spec["collection"]["expected_success"]
        or any(r["label"] != "success" for r in manifest["episodes"])
        or manifest.get("excluded")
        or manifest.get("amendment")
        or manifest.get("seed") != 0
        or manifest.get("validation_fraction") != 0.1
        or len(manifest["split"]["validation"]) != 10
    ):
        raise ValueError("need full 100-success PNP v2 S membership, 90/10 seed-0 split, no exclusions or amendment")
    return manifest


def readiness(stage, manifest_path, split_path, outcomes_path, gate_path, store_path, spec, *, open_store=None):
    """Runtime barrier of one view: exact store/raw membership, split, outcomes and every consumer's gate."""
    from s4d.rwlearn.baseline_data import check_gate
    from s4d.rwlearn.dsreset_identity import store_identities

    if open_store is None:
        from s4d.rwlearn.store import open_store
    view = STAGE_VIEW[stage]
    m = manifest_check(json.loads(Path(manifest_path).read_text()), spec)
    split, outcomes = json.loads(Path(split_path).read_text()), json.loads(Path(outcomes_path).read_text())
    report = json.loads(Path(gate_path).read_text())
    no_override(report)
    store = open_store(str(store_path))
    ids = store_identities(store)
    by_sha = {r["sha256"]: r for r in m["episodes"]}
    if len(ids) != len(by_sha) or {r["raw_sha256"] for r in ids.values()} != set(by_sha):
        raise ValueError("store raw hashes must exactly match the task manifest")
    if set(outcomes) != set(ids) or any(v is not True for v in outcomes.values()):
        raise ValueError("S-only outcomes must equal exactly the store episode set")
    for ep, rec in ids.items():
        row = by_sha[rec["raw_sha256"]]
        if rec["task"] != TASK or ep not in split[row["split"]]:
            raise ValueError("store task or split differs from immutable membership")
    consumers = RGB_CONSUMERS if view == "rgb" else GEOMETRY_CONSUMERS
    for consumer in consumers:
        check_gate(store, report, steps=None, split=split, target=TASK, consumer=consumer)
    return dict(schema="pick-and-place-v2-readiness/1", task=TASK, view=view, stage=stage, passed=True,
                episodes=len(ids), consumers=list(consumers), robot_io=False)
