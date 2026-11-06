"""Episode outcomes and the ending (reset / reversal) check of the real-world demonstrations.

User decision 2026-11-06 ~11:20 KST (rw-decisions-calibration-upload-20261106, item 1): every uploaded episode is a
SUCCESSFUL demonstration. Its outcome is ``success`` and its success-state proxy is the recording's FINAL observation
(the last recorded frame of every camera; ``s4d.rwlearn.transitions``). Endings that include a reset or reversal of the
completed task are FLAGGED for the coordinator; a flag never changes the success label.

Where the labels live: ``outcome_labels`` writes an ``rw-outcomes`` v1 document (``outcomes.json`` next to the gate
report and split of a learning-inputs directory, ``scripts/realworld_learning.py inputs``); the FQL dataset builder
reads it (``--outcomes``) and records it in the dataset attributes, so every transition's reward traces back to it.

Ending check (``check_episode``; robot data first, read-only h5py reads of the raw ``panda_teleop_episode`` file, no
images). Signals: measured flange pose ``robot/flange_pose`` (100 Hz), the sent gripper command ``action/gripper``
(closure c in [0, 1]), the Robotiq status ``gripper/object`` (gOBJ: 0 moving, 1 contact while opening, 2 contact while
closing = object held, 3 at the requested position), the external joint-torque estimate ``robot/tau_ext`` and every
camera's frame times (frame indices of the evidence).

* drawer-open: the pull is the motion from the minimum to the following maximum of the flange position along the
  episode's drawer axis (the principal horizontal direction of motion, sign-aligned with base +y; measured within
  18 deg of +y on every recording). A push-back has to move the flange back along the axis while it is still at the
  handle: ``back_at_handle`` = how far the flange moves back before it first rises ``CLEARANCE_M`` above its height at
  the pull end. Flagged (``reversal_suspected``) when that exceeds ``PUSH_BACK_M``, or when the external torque during
  the backward motion reaches the pull's contact level (``torque_contact_fraction`` of the way from the free-motion
  approach to the pull median) while the pull itself shows contact. The operators' usual return toward the start pose
  after lifting off the handle is recorded as information (``return_toward_start``), not as a reversal.
* mug-hanging: the hang release is the end of the last closure interval during which the gripper held an object
  (gOBJ 2). Flagged (``reversal_suspected``) when the episode ends with the gripper commanded closed or still holding
  an object, when gOBJ reports a held object after the hang release, or when an object is grasped again within
  ``REVISIT_LEAVE_M`` of where the previous one was released (the mug taken off the hook). Several holding intervals
  elsewhere, and a return of the flange to within ``REVISIT_M`` of the release position after leaving it (a possible
  knock-off), cannot be decided from robot data: ``needs_image_review``. A later closure without object contact and
  the usual return toward the start pose are information.

``decision`` per episode: ``no_reversal`` (robot data decide), ``reversal_suspected`` or ``needs_image_review``. Image
evidence is used only for the latter (``review_sheet``: first / task-event / final colour frames of every camera, read
from the original HDF5, for a human or the coordinator to inspect).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

OUTCOME_SCHEMA, OUTCOME_VERSION = "rw-outcomes", 1
REPORT_SCHEMA, REPORT_VERSION = "rw-ending-flags", 1
DECISION = ("user decision 2026-11-06 ~11:20 KST: every uploaded episode is a successful demonstration "
            "(outcome success); ending flags never change the label")
SUCCESS_STATE = ("success-state proxy = the recording's FINAL observation: the last recorded frame of every camera "
                 "(s4d.rwlearn.transitions: r_k = success(s_k+1) - 1, mask_k = 1 - success(s_k+1))")
DECISIONS = ("no_reversal", "reversal_suspected", "needs_image_review")

# thresholds (metres, N m, fractions); every one is reported with the evidence
CLEARANCE_M = 0.03  # drawer: rise above the pull-end height that takes the fingers off the handle
PUSH_BACK_M = 0.05  # drawer: backward travel at handle height that counts as pushing the drawer back
MIN_PULL_M = 0.15  # drawer: a pull shorter than this is not a completed opening (review)
SPEED_M_S = 0.03  # motion threshold of the approach / pull / backward phases
TORQUE_CONTACT_FRACTION = 0.5
MIN_PULL_CONTACT_NM = 0.2  # the torque test applies only when the pull itself shows contact above free motion
RETURN_FRACTION = 0.3  # information: the flange ends this fraction of the pull back toward the start
REVISIT_LEAVE_M, REVISIT_M = 0.05, 0.03  # mug: left the release position by 5 cm, then came back within 3 cm
GOBJ_HELD = 2

_ROBOT = ("robot/t", "robot/flange_pose", "robot/tau_ext", "action/gripper", "action/segment")
_GRIPPER = ("gripper/t", "gripper/position", "gripper/object")


def episode_key(task: str, path) -> str:
    """Store / gate episode id ``<task>/episode_NNNNNN`` of a raw recording."""
    return f"{task}/{Path(path).stem}"


# ----------------------------------------------------------------------------------------------- labels
def outcome_labels(episodes, *, ending_report: dict | None = None) -> dict:
    """The ``rw-outcomes`` v1 document: outcome ``success`` for every episode (user decision), with its ending flags
    (codes only; evidence in the ending report) for traceability."""
    flags = (ending_report or {}).get("episodes", {})
    out = {"schema": OUTCOME_SCHEMA, "schema_version": OUTCOME_VERSION, "decision": DECISION,
           "success_state": SUCCESS_STATE, "flags_change_labels": False, "episodes": {}}
    if ending_report is not None:
        out["ending_report"] = {"schema": ending_report.get("schema"), "sha256": _sha(ending_report)}
    for ep in sorted(episodes):
        rec = flags.get(ep) or {}
        out["episodes"][ep] = {"outcome": "success",
                               "ending_decision": rec.get("decision"),
                               "ending_flags": [f["code"] for f in rec.get("flags", []) if f["severity"] != "info"]}
    return out


def load_outcomes(doc_or_path) -> dict[str, bool]:
    """{episode: True (success) | False (failure)} from an ``rw-outcomes`` document (or its path). Legacy plain
    ``{episode: bool}`` mappings are accepted; anything else is refused."""
    doc = doc_or_path
    if isinstance(doc, (str, Path)):
        doc = json.loads(Path(doc).read_text())
    if not isinstance(doc, dict):
        raise ValueError("outcomes must be a JSON object")
    if "schema" not in doc:
        bad = {k: v for k, v in doc.items() if not isinstance(v, bool)}
        if bad:
            raise ValueError(f"outcome labels must be JSON booleans, got {dict(list(bad.items())[:3])}")
        return dict(doc)
    if (doc.get("schema"), doc.get("schema_version")) != (OUTCOME_SCHEMA, OUTCOME_VERSION):
        raise ValueError(f"not an {OUTCOME_SCHEMA} v{OUTCOME_VERSION} document")
    out = {}
    for ep, rec in (doc.get("episodes") or {}).items():
        o = (rec or {}).get("outcome")
        if o not in ("success", "failure"):
            raise ValueError(f"{ep}: outcome {o!r} is neither 'success' nor 'failure'")
        out[ep] = o == "success"
    return out


def _sha(obj) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- signals
def read_signals(path) -> dict:
    """The small arrays the ending check needs (no images; read-only)."""
    import h5py

    with h5py.File(path, "r") as f:
        sig = {k: f[k][()] for k in _ROBOT + _GRIPPER}
        sig["cameras"] = {name[4:]: f["cameras"][name]["t"][()] for name in f["cameras"] if name.startswith("cam_")}
        sig["task_name"] = str(f.attrs.get("task_name"))
    return sig


def _frames(sig: dict, times) -> dict:
    """Colour frame index of every camera at (the first frame at or after) each time."""
    times = np.atleast_1d(np.asarray(times, np.float64))
    return {serial: np.clip(np.searchsorted(t, times), 0, len(t) - 1).astype(int).tolist()
            for serial, t in sorted(sig["cameras"].items())}


def _smooth(a: np.ndarray, k: int = 11) -> np.ndarray:
    return np.convolve(np.pad(a, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def _flag(code, severity, message, sig, rows, **evidence) -> dict:
    rows = [int(r) for r in np.atleast_1d(rows)]
    t = sig["robot/t"]
    return {"code": code, "severity": severity, "message": message, "robot_rows": rows,
            "t_s": [round(float(t[r] - t[0]), 3) for r in rows], "frames": _frames(sig, t[rows]),
            "evidence": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                         for k, v in evidence.items()}}


def _return_info(sig, start_row: int, ref_row: int, extent: float) -> dict:
    p = sig["robot/flange_pose"][:, :3]
    d_ref_start = float(np.linalg.norm(p[ref_row] - p[start_row]))
    d_end_start = float(np.linalg.norm(p[-1] - p[start_row]))
    return {"start_xyz": p[start_row].round(4).tolist(), "task_event_xyz": p[ref_row].round(4).tolist(),
            "final_xyz": p[-1].round(4).tolist(), "event_to_start_m": round(d_ref_start, 4),
            "final_to_start_m": round(d_end_start, 4), "extent_m": round(float(extent), 4)}


# ----------------------------------------------------------------------------------------------- drawer-open
def check_drawer(sig: dict) -> dict:
    t, p = sig["robot/t"], sig["robot/flange_pose"][:, :3]
    tau = np.linalg.norm(sig["robot/tau_ext"], axis=1)
    n = len(t)
    flags = []
    c = p[:, :2] - p[:, :2].mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    axis = axis * np.sign(axis[1] or 1.0)
    axis_angle = float(np.degrees(np.arccos(np.clip(abs(axis[1]), 0, 1))))
    proj = p[:, :2] @ axis
    lo = int(proj[: max(int(0.8 * n), 1)].argmin())
    hi = lo + int(proj[lo:].argmax())
    span = float(proj[hi] - proj[lo])
    z_ref = float(p[hi, 2])
    after = np.arange(n) > hi
    above = np.flatnonzero(after & (p[:, 2] >= z_ref + CLEARANCE_M))
    stop = int(above[0]) if len(above) else n
    back_at_handle = float(proj[hi] - proj[hi:stop].min()) if stop > hi else 0.0
    behind = np.flatnonzero(after & (proj < proj[hi] - PUSH_BACK_M))
    lift_at_push_back = float(p[behind[0], 2] - z_ref) if len(behind) else None
    v = np.gradient(_smooth(proj), t)
    speed = np.abs(np.gradient(_smooth(np.linalg.norm(p - p[0], axis=1)), t))
    approach = np.flatnonzero((np.arange(n) < lo) & (speed > SPEED_M_S))
    pull = np.arange(lo, hi + 1)[v[lo:hi + 1] > SPEED_M_S]
    back = np.flatnonzero(after & (v < -SPEED_M_S))
    med = lambda idx: float(np.median(tau[idx])) if len(idx) else float("nan")  # noqa: E731
    tq = {"approach": med(approach), "pull": med(pull), "backward": med(back)}
    pull_contact = tq["pull"] - tq["approach"]
    torque_decides = bool(np.isfinite(pull_contact) and pull_contact >= MIN_PULL_CONTACT_NM and len(back))
    back_contact = (tq["backward"] - tq["approach"]) / pull_contact if torque_decides else None
    phases = {"drawer_axis_xy": axis.round(4).tolist(), "axis_angle_from_base_y_deg": round(axis_angle, 2),
              "pull_rows": [lo, hi], "pull_frames": _frames(sig, t[[lo, hi]]), "pull_m": round(span, 4),
              "pull_end_z_m": round(z_ref, 4), "back_at_handle_m": round(back_at_handle, 4),
              "lift_when_back_5cm_m": None if lift_at_push_back is None else round(lift_at_push_back, 4),
              "torque_median_nm": {k: round(x, 3) for k, x in tq.items()}, "torque_decides": torque_decides,
              "backward_contact_fraction": None if back_contact is None else round(float(back_contact), 3)}
    if span < MIN_PULL_M:
        flags.append(_flag("pull_not_found", "review", f"largest pull along the drawer axis {span:.3f} m < "
                           f"{MIN_PULL_M} m", sig, [lo, hi], pull_m=span))
    if back_at_handle > PUSH_BACK_M:
        rows = [hi, int(np.argmin(np.where((np.arange(n) >= hi) & (np.arange(n) < stop), proj, np.inf)))]
        flags.append(_flag("push_back_at_handle_height", "suspect",
                           f"flange moved {back_at_handle:.3f} m back along the drawer axis before rising "
                           f"{CLEARANCE_M} m above the pull-end height", sig, rows, back_at_handle_m=back_at_handle,
                           pull_m=span))
    if torque_decides and back_contact >= TORQUE_CONTACT_FRACTION:
        flags.append(_flag("contact_torque_while_moving_back", "suspect",
                           f"external torque while moving back is {back_contact:.0%} of the way from free motion to "
                           "the pull's contact level", sig, [int(back[0]), int(back[-1])], **tq))
    ret = float(proj[hi] - proj[-1])
    info = _return_info(sig, 0, hi, span)
    info["return_along_axis_m"] = round(ret, 4)
    if ret >= RETURN_FRACTION * span:
        flags.append(_flag("return_toward_start", "info",
                           f"after the pull the flange returns {ret:.3f} m ({ret / max(span, 1e-9):.0%} of the pull) "
                           f"along the axis, lifted {lift_at_push_back if lift_at_push_back is not None else 0:.3f} m "
                           "above the pull height before moving back", sig, [hi, n - 1],
                           return_m=ret, pull_m=span))
    return {"phases": phases, "return": info, "flags": flags}


# ----------------------------------------------------------------------------------------------- mug-hanging
def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def check_mug(sig: dict) -> dict:
    t, p = sig["robot/t"], sig["robot/flange_pose"][:, :3]
    c = sig["action/gripper"]
    gt, gobj = sig["gripper/t"], sig["gripper/object"]
    n = len(t)
    flags = []
    closures = _intervals(c > 0.5)
    held = gobj == GOBJ_HELD

    def holds(a, b):  # an object held (gOBJ 2) at a gripper sample within robot rows [a, b]
        m = (gt >= t[a]) & (gt <= t[b] + 0.1)  # gripper registers lag the command by up to ~0.1 s
        return bool((held & m).any())

    holding = [iv for iv in closures if holds(*iv)]
    phases = {"closures_rows": closures, "holding_closures_rows": holding, "final_command": round(float(c[-1]), 3),
              "final_gobj": int(gobj[-1]), "final_register": round(float(sig["gripper/position"][-1]), 1)}
    if not holding:
        flags.append(_flag("no_grasp", "review", "no closure interval with an object held (gOBJ 2)", sig, [0, n - 1]))
        return {"phases": phases, "return": _return_info(sig, 0, n - 1, 0.0), "flags": flags}
    grasp, release = holding[-1][0], holding[-1][1] + 1 if holding[-1][1] + 1 < n else holding[-1][1]
    phases.update(grasp_row=grasp, release_row=release, grasp_frames=_frames(sig, t[grasp]),
                  release_frames=_frames(sig, t[release]))
    if c[-1] > 0.5:
        flags.append(_flag("ends_closed", "suspect", "the episode ends with the gripper commanded closed", sig, [n - 1],
                           final_command=float(c[-1])))
    if gobj[-1] == GOBJ_HELD:
        flags.append(_flag("holding_at_end", "suspect", "the gripper still holds an object at the end", sig, [n - 1]))
    for (_, b0), (a1, _) in zip(holding[:-1], holding[1:]):  # a later grasp: the mug taken off the hook again?
        gap = float(np.linalg.norm(p[a1] - p[min(b0 + 1, n - 1)]))
        if gap < REVISIT_LEAVE_M:
            flags.append(_flag("regrasp_at_release_location", "suspect",
                               f"an object is grasped again {gap:.3f} m from where the previous one was released",
                               sig, [b0, a1], distance_m=gap))
        else:
            flags.append(_flag("multiple_grasps", "review", f"{len(holding)} holding intervals; robot data cannot "
                               "tell which release is the hang", sig, [b0, a1], distance_m=gap))
    for a, b in (iv for iv in closures if iv[0] > holding[-1][1]):  # closures after the hang release hold nothing
        flags.append(_flag("reclose_without_object", "info", "closure after the hang release without object contact "
                           "(reopened)", sig, [a, b], duration_s=float(t[b] - t[a])))
    after_release = gt > t[release] + 0.1
    if (held & after_release).any():
        k = int(np.flatnonzero(held & after_release)[0])
        row = int(np.clip(np.searchsorted(t, gt[k]), 0, n - 1))
        flags.append(_flag("object_contact_after_release", "suspect", "gOBJ reports a held object after the hang "
                           "release", sig, [row]))
    d = np.linalg.norm(p[release:] - p[release], axis=1)
    left = np.flatnonzero(d > REVISIT_LEAVE_M)
    if len(left):
        back = np.flatnonzero(d[left[0]:] < REVISIT_M)
        if len(back):
            row = release + int(left[0]) + int(back[0])
            flags.append(_flag("revisit_release_location", "review",
                               f"the flange returns within {REVISIT_M} m of the release position after leaving it by "
                               f"{REVISIT_LEAVE_M} m (possible contact with the hung mug)", sig, [release, row]))
    tau = np.linalg.norm(sig["robot/tau_ext"], axis=1)
    phases["torque_nm"] = {"pre_grasp_median": round(float(np.median(tau[:grasp])), 3) if grasp else None,
                           "post_release_median": round(float(np.median(tau[release:])), 3),
                           "post_release_max": round(float(tau[release:].max()), 3)}
    info = _return_info(sig, 0, release, float(np.linalg.norm(p[release] - p[grasp])))
    info["final_to_release_m"] = round(float(d[-1]), 4)
    info["final_to_grasp_m"] = round(float(np.linalg.norm(p[-1] - p[grasp])), 4)
    if info["final_to_start_m"] < 0.5 * info["event_to_start_m"]:
        flags.append(_flag("return_toward_start", "info", "after the hang release the open gripper returns toward the "
                           f"start pose (final {info['final_to_start_m']:.3f} m from it, release "
                           f"{info['event_to_start_m']:.3f} m)", sig, [release, n - 1]))
    return {"phases": phases, "return": info, "flags": flags}


CHECKS = {"drawer-open": check_drawer, "mug-hanging": check_mug}


def check_episode(path, task: str) -> dict:
    sig = read_signals(path)
    res = CHECKS[task](sig)
    sev = {f["severity"] for f in res["flags"]}
    decision = ("reversal_suspected" if "suspect" in sev else "needs_image_review" if "review" in sev
                else "no_reversal")
    t = sig["robot/t"]
    return {"episode": episode_key(task, path), "path": str(path), "recorder_task_name": sig["task_name"],
            "outcome": "success", "decision": decision, "duration_s": round(float(t[-1] - t[0]), 3),
            "final_frames": {s: len(ct) - 1 for s, ct in sorted(sig["cameras"].items())}, **res}


def scan(raw_root, tasks=("drawer-open", "mug-hanging"), episodes=None) -> dict:
    """Ending check of every ``episode_*.h5`` under ``<raw_root>/<task>/`` (read-only)."""
    raw_root = Path(raw_root)
    out = {"schema": REPORT_SCHEMA, "schema_version": REPORT_VERSION, "decision": DECISION,
           "success_state": SUCCESS_STATE, "raw_root": str(raw_root),
           "thresholds": {"clearance_m": CLEARANCE_M, "push_back_m": PUSH_BACK_M, "min_pull_m": MIN_PULL_M,
                          "speed_m_s": SPEED_M_S, "torque_contact_fraction": TORQUE_CONTACT_FRACTION,
                          "min_pull_contact_nm": MIN_PULL_CONTACT_NM, "return_fraction": RETURN_FRACTION,
                          "revisit_leave_m": REVISIT_LEAVE_M, "revisit_m": REVISIT_M},
           "counts": {}, "episodes": {}}
    for task in tasks:
        files = sorted((raw_root / task).glob("episode_*.h5"))
        if episodes is not None:
            files = [f for f in files if episode_key(task, f) in episodes]
        counts = {"episodes": 0, **{d: 0 for d in DECISIONS}, "info_flags": {}}
        for f in files:
            rec = check_episode(f, task)
            out["episodes"][rec["episode"]] = rec
            counts["episodes"] += 1
            counts[rec["decision"]] += 1
            for fl in rec["flags"]:
                if fl["severity"] == "info":
                    counts["info_flags"][fl["code"]] = counts["info_flags"].get(fl["code"], 0) + 1
        out["counts"][task] = counts
    return out


def summary(report: dict) -> dict:
    """Compact per-episode view of an ending report (decision, flag codes, the deciding robot-data margins)."""
    keep = {"drawer-open": ("pull_m", "back_at_handle_m", "lift_when_back_5cm_m", "backward_contact_fraction",
                            "axis_angle_from_base_y_deg"),
            "mug-hanging": ("final_command", "final_gobj")}
    out = {k: report[k] for k in ("schema", "schema_version", "decision", "success_state", "thresholds", "counts")}
    out["episodes"] = {}
    for ep, r in sorted(report["episodes"].items()):
        task = ep.split("/")[0]
        rec = {"decision": r["decision"], "flags": [f"{f['code']}:{f['severity']}" for f in r["flags"]],
               **{k: r["phases"].get(k) for k in keep.get(task, ())}}
        if task == "mug-hanging":
            rec["holding_intervals"] = len(r["phases"].get("holding_closures_rows", []))
            rec["final_to_release_m"] = r["return"].get("final_to_release_m")
        out["episodes"][ep] = rec
    return out


def flagged(report: dict) -> dict:
    """Episodes whose decision is not ``no_reversal``, with their non-informational flags."""
    return {ep: {"decision": r["decision"], "flags": [f for f in r["flags"] if f["severity"] != "info"]}
            for ep, r in report["episodes"].items() if r["decision"] != "no_reversal"}


# ----------------------------------------------------------------------------------------------- image evidence
def review_sheet(path, rows: list[int], out_png, width: int = 213) -> Path:
    """Image evidence for an undecided ending: the first frame, the colour frame at each robot row in ``rows`` and the
    final frame of every camera, from the ORIGINAL recording (read-only), tiled into one PNG (one row per camera)."""
    import h5py
    from PIL import Image, ImageDraw

    tiles = []
    with h5py.File(path, "r") as f:
        t = f["robot/t"][()]
        for name in sorted(k for k in f["cameras"] if k.startswith("cam_")):
            ct = f["cameras"][name]["t"][()]
            picks = [0, *[int(np.clip(np.searchsorted(ct, t[r]), 0, len(ct) - 1)) for r in rows], len(ct) - 1]
            row = []
            for i in picks:
                img = Image.open(io.BytesIO(bytes(f["cameras"][name]["color"][i]))).convert("RGB")
                img = img.resize((width, int(width * img.height / img.width)))
                ImageDraw.Draw(img).text((3, 3), f"{Path(path).stem} {name[-4:]} f{i}", fill=(255, 255, 0))
                row.append(np.asarray(img))
            tiles.append(np.concatenate(row, 1))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(tiles, 0)).save(out_png)
    return out_png
