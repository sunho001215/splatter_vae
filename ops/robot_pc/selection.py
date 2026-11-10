"""Task / policy / camera selection: what the server offers (WELCOME: policy id -> unbound input spec + catalog info)
and the session choices a policy's contract requires (cameras, native grids, static calibration, native depth
correction), from the robot-PC config's per-task rig document. Contract-driven, never guessed:

* single-view conditions: exactly one TRAINED camera (``--camera`` or the first trained one);
* multi-view conditions: all trained cameras in training order, live only with the validated M5 helper asset;
* calibrated conditions (3D-MVP, SNeRL): the task's published rig document (``realsense.load_rig_session``: c2w, the
  stream's K_native; refused when a camera is not validated);
* depth conditions (3D-MVP): native RealSense depth x the rig document's ``depth_scale['sensor_depth']`` (never a
  FoundationStereo factor); identity only when explicitly requested and labelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SelectionError(ValueError):
    pass


def describe(policies: dict) -> list[dict]:
    """WELCOME ``policies`` -> one row per id with its contract requirements (sorted by id)."""
    rows = []
    for pid, entry in sorted(policies.items()):
        spec, info = entry.get("input_spec") or {}, entry.get("info") or {}
        rows.append({"policy_id": pid, "condition": spec.get("condition") or info.get("condition"),
                     "task": info.get("task"), "seed": info.get("seed"), "status": info.get("status", "deployable"),
                     "fixture": bool(spec.get("fixture")), "views": spec.get("views"),
                     "modalities": spec.get("modalities"), "needs_calibration": spec.get("needs_calibration"),
                     "trained_cameras": [list(o) for o in spec.get("trained_camera_orders", [])],
                     "flags": info.get("flags", [])})
    return rows


def select_policy(policies: dict, policy_id: str | None = None, *, condition: str | None = None,
                  task: str | None = None, seed: int | None = None) -> str:
    """An explicit id, or the unique deployable id matching condition / task / seed."""
    if policy_id is not None:
        if policy_id not in policies:
            raise SelectionError(f"the server does not offer {policy_id!r}; it offers {sorted(policies)}")
        return policy_id
    match = [r["policy_id"] for r in describe(policies)
             if (condition is None or r["condition"] == condition) and (task is None or r["task"] == task)
             and (seed is None or r["seed"] == seed) and r["status"] == "deployable" and not r["fixture"]]
    if len(match) != 1:
        raise SelectionError(f"{len(match)} deployable policies match condition={condition} task={task} seed={seed}: "
                             f"{match}; give --policy")
    return match[0]


def session_cameras(spec: dict, camera: str | None = None) -> list[str]:
    orders = spec["trained_camera_orders"]
    if spec["views"] > 1:
        if camera is not None:
            raise SelectionError("multi-view policies use every trained camera in training order (no --camera)")
        return list(orders[0])
    trained = sorted({c for o in orders for c in o})
    cam = camera or orders[0][0]
    if cam not in trained:
        raise SelectionError(f"camera {cam} is not a trained camera of this policy ({trained})")
    return [cam]


def depth_corrections(spec: dict, cams: list, rig_document: Path | None, *, identity: bool = False) -> dict | None:
    """Native depth correction per camera (only for depth conditions): the rig document's sensor_depth scale, or an
    explicitly requested, labelled identity. Independent of the stream intrinsics (start the depth streams with it)."""
    if "depth" not in spec["modalities"]:
        return None
    from s4d.rwlearn.deploy.realsense import load_rig_session, native_correction

    if identity:
        return {c: native_correction() for c in cams}
    if rig_document is None:
        raise SelectionError("a depth condition needs the task's rig document (validated sensor_depth scale)")
    return load_rig_session(rig_document, cams, {c: np.eye(3) for c in cams})["native_depth_correction"]


def session_choices(spec: dict, cams: list, *, native_hw: dict, K_native: dict, rig_document: Path | None,
                    identity_depth: bool = False) -> dict:
    """HELLO choices for the session cameras (the client verifies every later frame against them)."""
    choices = {"native_hw": {c: [int(x) for x in native_hw[c]] for c in cams}}
    if spec["views"] == 1:
        choices["camera_id"] = cams[0]
    if spec["needs_calibration"]:
        if rig_document is None:
            raise SelectionError("this condition needs the task's static rig calibration (rig document)")
        from s4d.rwlearn.deploy.realsense import load_rig_session

        choices["calibration"] = load_rig_session(rig_document, cams, K_native)["calibration"]
    corr = depth_corrections(spec, cams, rig_document, identity=identity_depth)
    if corr is not None:
        choices["native_depth_correction"] = corr
    return choices
