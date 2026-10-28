"""Camera suite v2 (user directive 2026-10-29): method-independent visibility analysis, pre-registered selection of
the held-out cameras vis0..vis3, validation of the stronger moving-camera paths, and media. Evaluation only.

No policy, return, reconstruction or representation score is read anywhere in this tool: cameras are chosen from
simulator geometry, segmentation and depth on representative scripted-expert / uniform-random trajectories
(``s4d.rl.visibility``). Run the stages in order (one allowed GPU UUID, MUJOCO_GL=egl, ``python -E -s``):

    measure --set reference          train0..5 and the original held-out eval0..3 (calibration and documentation)
    register-rule                    write the selection rule (runs/setup/camera-suite-v2-rule-20261029.json)
    measure --set candidates         the rule's candidate grid (after the rule is registered)
    select                           apply the registered rule -> runs/camera-suite-v2/selection.json
    measure --set paths              train1, the 1x reference paths and the stronger levels
    paths                            kinematics + visibility validation -> runs/camera-suite-v2/paths.json
    freeze                           runs/setup/camera-suite-v2-definition-20261029.{json,md}
    measure --set v2                 the frozen vis0..vis3 (all frames, for the published numbers)
    media                            contact sheets, preview videos, camera layout (runs/camera-suite-v2/media)

Work files go to runs/camera-suite-v2/ (untracked); the rule and the frozen definition go to runs/setup/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from _bootstrap import REPO  # noqa: E402  (imports no torch)

from s4d.data.metaworld.cameras import EVAL_CAMERAS, TRAIN_CAMERAS  # noqa: E402
from s4d.rl import camera_suites as CS  # noqa: E402  (pure geometry, no MuJoCo)

WORK = REPO / "runs/camera-suite-v2"
TASKS = (
    "hammer", "bin-picking", "door-open", "peg-insert-side", "peg-unplug-side", "pick-place",
    "stick-push", "coffee-push", "drawer-open", "window-open", "soccer", "sweep-into",
)
REFERENCE_PATHS = ("lateral", "circular")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- view sets
def static_view(tilt: float, azimuth: float):
    from s4d.rl.env import orbit_camera  # noqa: PLC0415

    return [orbit_camera(tilt, azimuth)]


def view_set(name: str) -> dict[str, list]:
    """name -> {view name: list of FreeCamera poses (looping by render count)}."""
    from s4d.rl.env import HELDOUT_PATHS, HELDOUT_V2_PATHS, TRAIN_PATHS, trajectory_path  # noqa: PLC0415

    if name == "reference":
        return {**TRAIN_PATHS, **HELDOUT_PATHS}
    if name == "candidates":
        rule = load_rule()
        return {candidate_name(t, a): static_view(t, a) for t, a in rule_candidates(rule)}
    if name == "v2":
        if not HELDOUT_V2_PATHS:
            raise SystemExit("camera suite v2 is not frozen yet (s4d.rl.camera_suites.HELDOUT_V2_CAMERAS is empty)")
        return dict(HELDOUT_V2_PATHS)
    if name == "paths":
        views = {"train1": TRAIN_PATHS["train1"]}
        views |= {f"traj_{k}": trajectory_path(k) for k in REFERENCE_PATHS}
        views |= {f"traj_{k}": trajectory_path(k) for k in CS.STRONGER_TRAJECTORIES}
        return views
    if name.startswith("paths-try:"):  # reduced trial levels, e.g. paths-try:lateral=0.30,circular=27/16
        views = {}
        for item in name.split(":", 1)[1].split(","):
            kind, value = item.split("=")
            if kind == "lateral":
                views[f"traj_lateral_{value}m"] = trajectory_path("lateral", lateral_amplitude=float(value))
            else:
                az, el = map(float, value.split("/"))
                views[f"traj_circular_{az:g}_{el:g}"] = trajectory_path(
                    "circular", circular_azimuth_deg=az, circular_elevation_deg=el)
        return views
    raise SystemExit(f"unknown view set {name!r}")


def candidate_name(tilt: float, azimuth: float) -> str:
    return f"t{tilt:g}_a{azimuth:g}"


# --------------------------------------------------------------------------- measurement (worker processes)
def _worker_init() -> None:
    from _bootstrap import guard_mujoco  # noqa: PLC0415

    guard_mujoco()


def _measure_job(job: tuple) -> dict:
    task, views, size, stride = job
    from s4d.rl.visibility import measure_task  # noqa: PLC0415

    return measure_task(task, views, size=size, frame_stride=stride)


def measure(args) -> None:
    from s4d.rl.visibility import METRICS, summarize  # noqa: PLC0415

    views = view_set(args.set)
    names = list(views)
    chunks = [names[i:i + args.views_per_job] for i in range(0, len(names), args.views_per_job)]
    jobs = [(task, {n: views[n] for n in chunk}, args.size, args.frame_stride) for task in args.tasks for chunk in chunks]
    out_dir = WORK / "measurements" / args.set.replace(":", "_").replace("/", "-").replace(",", "+").replace("=", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[camera_visibility] {len(names)} views x {len(args.tasks)} tasks in {len(jobs)} jobs, "
          f"{args.workers} workers, stride {args.frame_stride}", flush=True)
    per_task: dict[str, list[dict]] = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_measure_job, jobs):
            per_task.setdefault(result["task"], []).append(result)
            print(f"[camera_visibility] {result['task']}: {len(result['views'])} views done", flush=True)
    summary: dict = {"set": args.set, "frame_stride": args.frame_stride, "image_size": args.size,
                     "metrics": list(METRICS), "views": {}, "tasks": list(args.tasks)}
    for task, results in per_task.items():
        frames = results[0]["frames"]
        if any(not np.array_equal(r["frames"], frames) for r in results):
            raise RuntimeError(f"{task}: jobs replayed different frames")
        metrics = np.concatenate([r["metrics"] for r in results], axis=0)
        order = [v for r in results for v in r["views"]]
        np.savez_compressed(out_dir / f"{task}.npz", metrics=metrics, frames=frames, views=np.asarray(order),
                            episode_kinds=np.asarray(results[0]["episode_kinds"]))
        for v, view in enumerate(order):
            summary["views"].setdefault(view, {})[task] = summarize(metrics[v], frames, results[0]["image_pixels"])
        summary.setdefault("expert_success", {})[task] = results[0]["expert_success"]
    summary["views"] = {v: {t: summary["views"][v][t] for t in args.tasks} for v in names}
    summary["aggregate"] = {v: aggregate(summary["views"][v]) for v in names}
    json_dump(out_dir / "summary.json", summary)
    print(f"[camera_visibility] wrote {out_dir / 'summary.json'}", flush=True)


def aggregate(per_task: dict) -> dict:
    """Across tasks: worst task and mean of the per-task summaries of one view."""
    scores = {t: s["score"] for t, s in per_task.items()}
    worst = min(scores, key=scores.get)
    out = {"score_min": scores[worst], "score_mean": float(np.mean(list(scores.values()))), "worst_task": worst}
    for target in ("obj", "ee", "goal"):
        vis = [s[target]["visibility"] for s in per_task.values()]
        hid = [s[target]["hidden_fraction"] for s in per_task.values()]
        arm = [s[target]["arm_occlusion_mean"] for s in per_task.values()]
        out[target] = {"visibility_min": float(min(vis)), "visibility_mean": float(np.mean(vis)),
                       "hidden_fraction_max": float(max(hid)), "hidden_fraction_mean": float(np.mean(hid)),
                       "arm_occlusion_max": float(np.nanmax(arm)), "arm_occlusion_mean": float(np.nanmean(arm))}
    ws = [s["workspace"] for s in per_task.values()]
    out["workspace"] = {"in_view_min": float(min(w["in_view_min"] for w in ws)),
                        "visible_mean_min": float(min(w["visible_mean"] for w in ws)),
                        "visible_mean": float(np.mean([w["visible_mean"] for w in ws])),
                        "arm_occlusion_mean": float(np.mean([w["arm_occlusion_mean"] for w in ws])),
                        "longest_lost_run_max": int(max(w["longest_lost_run"] for w in ws))}
    out["min_depth_m"] = float(min(s["min_depth_m"] for s in per_task.values()))
    out["min_clearance_m"] = float(min(s["min_clearance_m"] for s in per_task.values()))
    return out


# --------------------------------------------------------------------------- the selection rule
def _checked(path: Path, key: str) -> dict:
    payload = json.loads(path.read_text())
    body = {k: v for k, v in payload.items() if k != key}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != payload[key]:
        raise SystemExit(f"{path} was modified after registration")
    return payload


def load_rule() -> dict:
    """The registered rule, with registered amendments applied (each file is hash-checked)."""
    path = REPO / CS.RULE_FILE
    if not path.is_file():
        raise SystemExit(f"register the selection rule first ({CS.RULE_FILE})")
    rule = _checked(path, "rule_sha256")
    amendment = REPO / CS.RULE_AMENDMENT_FILE
    if amendment.is_file():
        a = _checked(amendment, "amendment_sha256")
        if a["amends_rule_sha256"] != rule["rule_sha256"]:
            raise SystemExit("the amendment does not amend the registered rule")
        rule = {**rule, "eligibility": {**rule["eligibility"], **a["eligibility"]},
                "amendment": {k: v for k, v in a.items() if k != "eligibility"}}
        a2_path = REPO / CS.RULE_AMENDMENT_A2_FILE
        if a2_path.is_file():
            a2 = _checked(a2_path, "amendment_sha256")
            if a2["amends_amendment_sha256"] != a["amendment_sha256"]:
                raise SystemExit("amendment A2 does not follow amendment A1")
            rule = {**rule, "selection": {**rule["selection"], **a2["selection"]}, "amendment_a2": a2}
    return rule


AMENDMENT_A1 = """\
Amendment A1 to the camera suite v2 selection rule (2026-10-29), registered after the candidate measurement and
before computing its own outcome. Reason: under step 4 no candidate (0 of 138) was eligible, and the step-7 fallback
needs task-specific exceptions on 6 of the 12 tasks (coffee-push, door-open, drawer-open, hammer, stick-push,
window-open), which defeats the common rig. Step 4 required matching the least observable training camera on each of
~15 separate criteria of every task (180 thresholds), an envelope only the training cameras themselves can meet.
Inputs are unchanged (visibility only; no returns, reconstruction or representation scores were ever computed).

Step 4 is replaced by (every one of the 12 tasks must pass):
  (a) score >= its minimum over the training cameras on that task (the registered score floor, unchanged): the
      camera is at least as observable as the least observable training camera, by the registered aggregate of
      object, gripper and goal visibility net of fully hidden frames (arm occlusion enters through visibility);
  (b) for each of object, gripper and goal: hidden-frame share <= max(0.10, the registered step-4 limit), i.e. no
      target is invisible in more than 10% of frames unless a training camera loses it as often;
  (c) at least 90% of the workspace grid inside the image at every frame.
The per-criterion partial-occlusion and unoccluded-workspace envelopes are dropped as separate constraints (they are
reported, and partial arm occlusion is already inside the score). Steps 1-3 and 5-8 are unchanged.
"""


AMENDMENT_A2 = """\
Amendment A2 to the camera suite v2 selection rule (2026-10-29), registered after amendment A1 also admitted no
camera (0 of 138: the registered per-task score floors, each task's minimum over the six training cameras, 0.62-0.92,
are not met on all 12 tasks by any viewpoint at least 25 deg from the training rig), and before computing its own
outcome. Threshold eligibility anchored per task to the training rig is replaced by a maximin choice that returns the
most observable admissible rig. Inputs remain visibility only.

  1. Admissible: the registered candidates (>= 25 deg from every training camera); measurement and per-task score
     unchanged (steps 1-3).
  2. S(c) = minimum over the 12 tasks of score(c, task).
  3. The common rig is the set of 4 admissible candidates, pairwise >= 30 deg apart, that maximises the smallest
     S(c) of its members; ties: larger sum of S, then larger minimum pairwise separation, then smaller
     (tilt, azimuth) tuples. vis0..vis3 are numbered by increasing azimuth.
  4. Task-specific exceptions use the training rig's global worst case (largest value over all training cameras and
     all tasks, + 0.02 slack for hidden shares; smallest score): a chosen camera whose object, gripper or goal
     hidden-frame share on a task exceeds that envelope, or whose score on a task is below it, is replaced for that
     task only by the admissible candidate closest in viewing direction that meets the envelope on that task.
  5. Confirmation on every frame (step 6, checked against step 4 of this amendment) and the moving-path validation
     (step 8) are unchanged.
"""


def register_amendment_a2(args) -> None:
    path = REPO / CS.RULE_AMENDMENT_A2_FILE
    if path.exists():
        raise SystemExit(f"{CS.RULE_AMENDMENT_A2_FILE} is already registered")
    a1 = _checked(REPO / CS.RULE_AMENDMENT_FILE, "amendment_sha256")
    ref = json.loads((WORK / "measurements/reference/summary.json").read_text())
    train = [f"train{i}" for i in range(len(TRAIN_CAMERAS))]
    per = [ref["views"][v][t] for v in train for t in ref["tasks"]]
    envelope = {f"{k}_hidden_fraction_max": round(max(x[k]["hidden_fraction"] for x in per) + 0.02, 4)
                for k in ("obj", "ee", "goal")}
    envelope["score_min"] = round(min(x["score"] for x in per), 4)
    a1_outcome = WORK / "selection.json"
    amendment = {
        "name": "camera suite v2 selection rule, amendment A2",
        "date": "2026-10-29",
        "amends_amendment_sha256": a1["amendment_sha256"],
        "statement": AMENDMENT_A2,
        "evidence_under_amendment_a1": {"eligible_candidates": 0, "candidates": 138,
                                        "selection_json_sha256": sha256_file(a1_outcome)},
        "selection": {"mode": "maximin", "count": 4, "min_pairwise_separation_deg": 30.0,
                      "exception_envelope": envelope,
                      "envelope_from": "training cameras train0..5 over all 12 tasks, reference measurement"},
    }
    amendment["amendment_sha256"] = hashlib.sha256(json.dumps(amendment, sort_keys=True).encode()).hexdigest()
    json_dump(path, amendment)
    a1_outcome.rename(WORK / "selection-amendment-A1.json")
    print(f"[camera_visibility] registered {path} ({amendment['amendment_sha256'][:12]}) envelope {envelope}")


def maximin_select(cams: dict, per_view: dict, rule: dict) -> dict:
    """Amendment A2 steps 2-4: the 4-camera set maximising the smallest worst-task score, and task exceptions."""
    sel = rule["selection"]
    tasks = rule["tasks"]
    S = {n: min(per_view[n][t]["score"] for t in tasks) for n in cams}
    order = sorted(cams, key=lambda n: (-S[n], cams[n]))
    sep = {(a, b): CS.angular_separation_deg(cams[a], cams[b]) for a in cams for b in cams if a < b}

    def separation(a, b):
        return sep[(a, b)] if a < b else sep[(b, a)]

    best = None
    for k in range(sel["count"], len(order) + 1):
        top = order[:k]
        feasible = [c for c in itertools.combinations(top, sel["count"])
                    if all(separation(a, b) >= sel["min_pairwise_separation_deg"]
                           for a, b in itertools.combinations(c, 2))]
        if feasible:
            def key(c):
                return (min(S[n] for n in c), sum(S[n] for n in c),
                        min(separation(a, b) for a, b in itertools.combinations(c, 2)),
                        tuple(-x for n in sorted(c, key=lambda n: cams[n]) for x in cams[n]))
            best = max(feasible, key=key)
            break
    if best is None:
        raise SystemExit("no admissible set of four cameras")
    chosen = sorted(best, key=lambda n: (cams[n][1], cams[n][0]))  # vis0..vis3 by azimuth
    env = sel["exception_envelope"]

    def meets(n, t):
        x = per_view[n][t]
        return (all(x[k]["hidden_fraction"] <= env[f"{k}_hidden_fraction_max"] for k in ("obj", "ee", "goal"))
                and x["score"] >= env["score_min"])

    exceptions = {}
    for i, n in enumerate(chosen):
        for t in tasks:
            if meets(n, t):
                continue
            others = [m for m in chosen if m != n]
            options = [m for m in cams if meets(m, t)
                       and all(separation(m, o) >= sel["min_pairwise_separation_deg"] for o in others)]
            replacement = min(options, key=lambda m: (separation(n, m), cams[m])) if options else None
            exceptions.setdefault(f"vis{i}", {})[t] = {
                "camera": n, "reason": {k: per_view[n][t][k]["hidden_fraction"] for k in ("obj", "ee", "goal")}
                | {"score": per_view[n][t]["score"]},
                "replacement": replacement,
                "replacement_tilt_azimuth": cams[replacement] if replacement else None,
                "replacement_separation_deg": separation(n, replacement) if replacement else None,
            }
    return {"chosen": chosen, "S": {n: S[n] for n in chosen}, "set_min_S": min(S[n] for n in chosen),
            "pairwise_separation_deg": {f"{a}-{b}": separation(a, b) for a, b in itertools.combinations(chosen, 2)},
            "top_k_needed": k, "exceptions": exceptions}


def register_amendment(args) -> None:
    path = REPO / CS.RULE_AMENDMENT_FILE
    if path.exists():
        raise SystemExit(f"{CS.RULE_AMENDMENT_FILE} is already registered")
    if (WORK / "selection.json").exists():
        previous = json.loads((WORK / "selection.json").read_text())
        if previous.get("rule_sha256") != json.loads((REPO / CS.RULE_FILE).read_text())["rule_sha256"]:
            raise SystemExit("unexpected selection state")
    rule = _checked(REPO / CS.RULE_FILE, "rule_sha256")
    e = rule["eligibility"]
    amendment = {
        "name": "camera suite v2 selection rule, amendment A1",
        "date": "2026-10-29",
        "amends_rule_sha256": rule["rule_sha256"],
        "statement": AMENDMENT_A1,
        "evidence_under_original_rule": {
            "eligible_candidates": 0, "candidates": 138,
            "fallback_min_tasks_excluded": 6,
            "fallback_tasks": ["coffee-push", "door-open", "drawer-open", "hammer", "stick-push", "window-open"],
            "selection_json_sha256": sha256_file(WORK / "selection.json"),
        },
        "eligibility": {
            "mode": "A1",
            "hidden_fraction_max": {k: {t: round(max(0.10, v), 4) for t, v in per.items()}
                                    for k, per in e["hidden_fraction_max"].items()},
            "workspace_in_view_min": {t: 0.90 for t in rule["tasks"]},
        },
    }
    amendment["amendment_sha256"] = hashlib.sha256(json.dumps(amendment, sort_keys=True).encode()).hexdigest()
    json_dump(path, amendment)
    (WORK / "selection.json").rename(WORK / "selection-original-rule.json")
    print(f"[camera_visibility] registered {path} ({amendment['amendment_sha256'][:12]})")


def rule_candidates(rule: dict) -> list[tuple[float, float]]:
    grid = rule["candidates"]
    cams = CS.candidate_grid(range(*grid["tilt_range"]), range(*grid["azimuth_range"]))
    train = [tuple(map(float, c)) for c in TRAIN_CAMERAS]
    return [c for c in cams if CS.min_separation_deg(c, train) >= grid["min_separation_from_training_deg"]]


RULE_STATEMENT = """\
Camera suite v2 held-out selection rule (registered 2026-10-29, before any candidate camera was measured; inputs are
simulator geometry, MuJoCo segmentation and depth only -- no policy, baseline return, reconstruction quality or
representation score of any method is read).

1. Candidates: the training rig's orbit (look-at (0, 0.6, 0), radius 1.0 m, fovy 45 deg, 128x128), tilt 25..65 deg in
   5 deg steps x azimuth 0..350 deg in 10 deg steps, keeping those whose viewing direction is at least 25 deg from every
   training camera (the training cameras are 23-28 deg from their nearest neighbour, so a held-out camera is at
   least as far from the training set as the training cameras are from each other).
2. Measurement: for every candidate and each of the 12 campaign tasks, the representative trajectories (10 scripted-
   expert episodes, 2 at each collection noise level 0/0.05/0.15/0.30/0.50 with collect.py's smooth noise, and 5
   uniform-random episodes; fixed seeds; RL timing: one render at reset and one per agent step, 125 agent steps of 2
   simulator steps) are rendered every 5th render (26 frames per episode) in the normal scene and in a diagnostic
   render without the robot's proximal links (every robot body except the wrist and gripper). Per frame: visible
   pixels of the manipulated object(s) and the gripper, visible points of the goal location (5 points; visible if
   unoccluded within 2 cm or covered by the object / goal body), each with and without the arm, and the table-surface
   workspace grid's in-image and unoccluded fractions (task definitions in s4d/rl/visibility.py).
3. Per (camera, task): visibility_k = mean over frames of (visible / arm-free pixels or points of target k; 0 when
   the target is not visible even without the arm), hidden_k = share of frames with nothing visible, arm_occlusion_k =
   mean arm-hidden share of the arm-free footprint, arm_heavy_k = share of frames with more than half of it hidden;
   score = mean over k in {object, gripper, goal} of (visibility_k - hidden_k).
4. Eligibility (every one of the 12 tasks must pass; thresholds are computed from the training cameras' own
   measurement on the same trajectories and recorded below, per task): for each target k, hidden_k <= its maximum
   over the training cameras + 0.02, arm_occlusion_k <= that maximum + 0.02 and arm_heavy_k <= that maximum + 0.02;
   workspace in-image fraction at every frame >= its minimum over the training cameras - 0.05; mean unoccluded
   workspace fraction >= that minimum - 0.05; score >= its minimum over the training cameras. A held-out camera is
   thus, on every task and criterion, at least as observable as the least observable training camera on that task
   (up to the stated slack). An undefined arm occlusion (target never visible even without the arm) is caught by
   hidden_k.
5. Selection of four static cameras for a common rig: vis0 = the eligible candidate with the best worst-task score
   (ties: mean score, then lower tilt, then lower azimuth). vis1..vis3: repeatedly the eligible candidate farthest
   (largest minimum angular separation) from the training cameras and the cameras already chosen, among those at
   least 30 deg from every chosen camera; candidates within 1 deg of the largest separation are tied and the best
   worst-task score wins. This spreads the suite around the workspace and maximises novelty subject to visibility.
6. Confirmation: the chosen cameras are re-measured on every frame (stride 1); each must still pass step 4, otherwise
   it is replaced by the next pick of the same procedure.
7. Task-specific exceptions: only if fewer than four eligible cameras can be chosen. Then the fewest tasks whose
   exclusion from step 4 allows four are named; the common rig is chosen without them, and for each such task every
   failing vis camera is replaced, for that task only, by the eligible candidate (on that task) closest in viewing
   direction. Every exception is recorded with its numbers before any policy evaluation and applied to every method.
8. Moving paths (lateral/circular at 2x and 3x, train1 base, 72-pose period, original starting phase) are validated
   with the same measurement on every frame: no frame with the workspace grid wholly outside the image or wholly
   occluded; >= 50% of the grid inside the image at every frame; nothing closer than 0.10 m to the camera (near plane
   0.023 m: no clipping); camera centre >= 0.05 m outside every rendered geom's bounding box; per task, hidden-frame
   shares of object, gripper and goal at most 0.10 above the static base camera train1, and no hidden run longer
   than 12 renders (object, gripper) or 24 renders (goal) unless train1 itself has a longer one. A failing level is
   reduced in steps of 10% of its target amplitudes until it passes, and the reduction is recorded.
"""

RULE_SLACK = {"hidden": 0.02, "arm": 0.02, "arm_heavy": 0.02, "workspace": 0.05}


def rule_definition(ref: dict) -> dict:
    """The registered rule with thresholds computed from the training cameras' reference measurement."""
    if ref["frame_stride"] != 1:
        raise SystemExit("the reference measurement must cover every frame")
    train = [f"train{i}" for i in range(len(TRAIN_CAMERAS))]
    views = ref["views"]
    tasks = ref["tasks"]
    targets = ("obj", "ee", "goal")

    def worst(metric: str, slack: float) -> dict:
        """Per target and task: the largest value over the training cameras plus the slack."""
        return {k: {t: round(float(np.nanmax([views[v][t][k][metric] for v in train])) + slack, 4) for t in tasks}
                for k in targets}

    hidden = worst("hidden_fraction", RULE_SLACK["hidden"])
    arm = worst("arm_occlusion_mean", RULE_SLACK["arm"])
    heavy = worst("arm_heavy_fraction", RULE_SLACK["arm_heavy"])
    return {
        "name": "camera suite v2 held-out selection rule",
        "date": "2026-10-29",
        "statement": RULE_STATEMENT,
        "rig": {"lookat": [0.0, 0.6, 0.0], "radius_m": 1.0, "fovy_deg": 45.0, "image_size": 128},
        "tasks": list(tasks),
        "candidates": {"tilt_range": [25, 66, 5], "azimuth_range": [0, 360, 10],
                       "min_separation_from_training_deg": 25.0},
        "search_frame_stride": 5,
        "confirmation_frame_stride": 1,
        "trajectories": {"expert_noise_levels": [0.0, 0.05, 0.15, 0.30, 0.50], "expert_per_level": 2,
                         "random_episodes": 5, "construction_seed": 1, "expert_seed_base": 900000,
                         "random_seed_base": 900100, "agent_steps": 125, "action_repeat": 2,
                         "renders_per_episode": 126},
        "eligibility": {
            "hidden_fraction_max": hidden,
            "arm_occlusion_mean_max": arm,
            "arm_heavy_fraction_max": heavy,
            "workspace_in_view_min": {t: round(min(views[v][t]["workspace"]["in_view_min"] for v in train)
                                              - RULE_SLACK["workspace"], 4) for t in tasks},
            "workspace_visible_mean_min": {t: round(min(views[v][t]["workspace"]["visible_mean"] for v in train)
                                                   - RULE_SLACK["workspace"], 4) for t in tasks},
            "score_floor": {t: round(min(views[v][t]["score"] for v in train), 4) for t in tasks},
            "slack": RULE_SLACK,
            "thresholds_from": "training cameras train0..5, reference measurement on every frame",
        },
        "selection": {"count": 4, "min_pairwise_separation_deg": 30.0, "tie_deg": 1.0,
                      "first": "best worst-task score", "next": "farthest from training + chosen"},
        "moving_paths": {
            "levels_target": {k: {"kind": kind, **amps} for k, (kind, amps) in CS.STRONGER_TRAJECTORIES.items()},
            "validation": {"workspace_longest_lost_run_max": 0, "workspace_in_view_min": 0.5, "min_depth_m": 0.10,
                           "min_clearance_m": 0.05, "hidden_fraction_excess_over_base_max": 0.10,
                           "longest_hidden_run_max": {"obj": 12, "ee": 12, "goal": 24}},
            "reduction_step": "10% of the target amplitudes",
        },
        "reference_measurement_sha256": sha256_file(WORK / "measurements/reference/summary.json"),
        "forbidden_inputs": ["returns or success of any method", "baseline returns", "reconstruction quality",
                             "representation scores"],
    }


def eligibility(per_task: dict, rule: dict) -> tuple[bool, dict]:
    """Per-task pass/fail of one camera's summaries against the registered thresholds; returns (all pass, failures)."""
    e = rule["eligibility"]
    failures: dict[str, list[str]] = {}
    if e.get("mode") == "A1":  # amendment A1: score floor, hidden-frame caps, workspace in view
        for task, s in per_task.items():
            bad = [f"{t} hidden {s[t]['hidden_fraction']:.3f} > {e['hidden_fraction_max'][t][task]:.3f}"
                   for t in ("obj", "ee", "goal") if s[t]["hidden_fraction"] > e["hidden_fraction_max"][t][task]]
            if s["workspace"]["in_view_min"] < e["workspace_in_view_min"][task]:
                bad.append(f"workspace in view {s['workspace']['in_view_min']:.3f}")
            if s["score"] < e["score_floor"][task]:
                bad.append(f"score {s['score']:.3f} < {e['score_floor'][task]:.3f}")
            if bad:
                failures[task] = bad
        return not failures, failures
    for task, s in per_task.items():
        bad = []
        for target in ("obj", "ee", "goal"):
            for metric, key, label in (("hidden_fraction", "hidden_fraction_max", "hidden"),
                                       ("arm_occlusion_mean", "arm_occlusion_mean_max", "arm occlusion"),
                                       ("arm_heavy_fraction", "arm_heavy_fraction_max", ">50%-arm-occluded frames")):
                value, limit = s[target][metric], e[key][target][task]
                if not np.isnan(value) and value > limit:
                    bad.append(f"{target} {label} {value:.3f} > {limit:.3f}")
        if s["workspace"]["in_view_min"] < e["workspace_in_view_min"][task]:
            bad.append(f"workspace in view {s['workspace']['in_view_min']:.3f}")
        if s["workspace"]["visible_mean"] < e["workspace_visible_mean_min"][task]:
            bad.append(f"workspace visible {s['workspace']['visible_mean']:.3f}")
        floor = e["score_floor"][task]
        if s["score"] < floor:
            bad.append(f"score {s['score']:.3f} < {floor:.3f}")
        if bad:
            failures[task] = bad
    return not failures, failures


def greedy_select(scores: dict[str, float], means: dict[str, float], cams: dict[str, tuple[float, float]],
                  eligible: list[str], rule: dict) -> list[dict]:
    """The registered greedy procedure: first the eligible camera with the best worst-task score; then repeatedly the
    eligible camera farthest (min angular separation) from the training cameras and the cameras already chosen,
    among those at least ``min_pairwise_separation_deg`` from the chosen ones; ties within ``tie_deg`` go to the
    better worst-task score."""
    sel = rule["selection"]
    train = [tuple(map(float, c)) for c in TRAIN_CAMERAS]
    order = sorted(eligible, key=lambda n: (-round(scores[n], 6), -round(means[n], 6), cams[n][0], cams[n][1]))
    chosen = [order[0]] if order else []
    steps = [{"pick": order[0], "reason": "best worst-task score", "score_min": scores[order[0]]}] if order else []
    while len(chosen) < sel["count"]:
        pool = [n for n in order if n not in chosen
                and CS.min_separation_deg(cams[n], [cams[c] for c in chosen]) >= sel["min_pairwise_separation_deg"]]
        if not pool:
            break
        spread = {n: CS.min_separation_deg(cams[n], train + [cams[c] for c in chosen]) for n in pool}
        best = max(spread.values())
        tied = [n for n in pool if spread[n] >= best - sel["tie_deg"]]
        pick = tied[0]  # pool keeps the score order
        chosen.append(pick)
        steps.append({"pick": pick, "reason": "farthest from training + chosen", "separation_deg": spread[pick],
                      "tied_within_deg": tied, "score_min": scores[pick]})
    return steps


def select(args) -> None:
    rule = load_rule()
    summary = json.loads((WORK / "measurements/candidates/summary.json").read_text())
    if summary["frame_stride"] != rule["search_frame_stride"]:
        raise SystemExit("candidate measurement does not use the registered frame stride")
    cams = {candidate_name(t, a): (t, a) for t, a in rule_candidates(rule)}
    missing = set(cams) - set(summary["views"])
    if missing:
        raise SystemExit(f"candidates not measured: {sorted(missing)[:5]}...")
    table, eligible = {}, []
    for name, (tilt, az) in cams.items():
        ok, failures = eligibility(summary["views"][name], rule)
        agg = summary["aggregate"][name]
        table[name] = {"tilt_deg": tilt, "azimuth_deg": az, "eligible": ok, "failures": failures,
                       "score_min": agg["score_min"], "score_mean": agg["score_mean"], "worst_task": agg["worst_task"],
                       "min_separation_from_training_deg": CS.min_separation_deg(
                           (tilt, az), [tuple(map(float, c)) for c in TRAIN_CAMERAS])}
        if ok:
            eligible.append(name)
    scores = {n: table[n]["score_min"] for n in cams}
    means = {n: table[n]["score_mean"] for n in cams}
    if rule["selection"].get("mode") == "maximin":  # amendment A2
        result = maximin_select(cams, summary["views"], rule)
        steps = [{"pick": n, "reason": "maximin set (amendment A2)", "score_min": result["S"][n]}
                 for n in result["chosen"]]
        out = {"rule_sha256": rule["rule_sha256"], "amendment_sha256": rule["amendment"]["amendment_sha256"],
               "amendment_a2_sha256": rule["amendment_a2"]["amendment_sha256"], "candidates": len(cams),
               "eligible": len(eligible), "selection": steps, "maximin": result,
               "exceptions": result["exceptions"], "table": table,
               "candidate_summary_sha256": sha256_file(WORK / "measurements/candidates/summary.json")}
        json_dump(WORK / "selection.json", out)
        for st in steps:
            n = st["pick"]
            print(json.dumps({"pick": n, "tilt_azimuth": cams[n], "S": round(result["S"][n], 4),
                              "min_sep_training": round(table[n]["min_separation_from_training_deg"], 1),
                              "worst_task": table[n]["worst_task"]}), flush=True)
        print(json.dumps({"pairwise": {k: round(v, 1) for k, v in result["pairwise_separation_deg"].items()},
                          "top_k_needed": result["top_k_needed"], "exceptions": result["exceptions"]}), flush=True)
        return
    steps = greedy_select(scores, means, cams, eligible, rule)
    exceptions = {}
    if len(steps) < rule["selection"]["count"]:
        exceptions = {"status": "common rig impossible under the rule; see the exception procedure",
                      "blocking": {n: table[n]["failures"] for n in sorted(cams, key=lambda n: -scores[n])[:10]}}
    result = {"rule_sha256": rule["rule_sha256"],
              "amendment_sha256": rule.get("amendment", {}).get("amendment_sha256"),
              "candidates": len(cams), "eligible": len(eligible),
              "selection": steps, "exceptions": exceptions, "table": table,
              "candidate_summary_sha256": sha256_file(WORK / "measurements/candidates/summary.json")}
    json_dump(WORK / "selection.json", result)
    for st in steps:
        print(json.dumps({k: v for k, v in st.items() if k != "tied_within_deg"}), flush=True)
    print(f"[camera_visibility] {len(eligible)}/{len(cams)} eligible; selection -> {WORK / 'selection.json'}")


def register_rule(args) -> None:
    path = REPO / CS.RULE_FILE
    if path.exists():
        raise SystemExit(f"{CS.RULE_FILE} is already registered; it is never overwritten")
    if (WORK / "measurements/candidates").exists():
        raise SystemExit("candidate measurements exist already: the rule must be registered before them")
    ref = json.loads((WORK / "measurements/reference/summary.json").read_text())
    rule = rule_definition(ref)
    rule["rule_sha256"] = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
    json_dump(path, rule)
    print(f"[camera_visibility] registered {path} ({rule['rule_sha256'][:12]})")


def paths(args) -> None:
    """Kinematics and visibility validation of the moving paths against the registered path criteria."""
    from s4d.data.metaworld.cameras import LOOKAT  # noqa: PLC0415
    from s4d.rl.env import camera_c2w, trajectory_path  # noqa: PLC0415

    rule = load_rule()
    crit = rule["moving_paths"]["validation"]
    summary = json.loads((WORK / "measurements/paths/summary.json").read_text())
    extra = {}
    for f in sorted((WORK / "measurements").glob("paths-try*/summary.json")):
        extra.update(json.loads(f.read_text())["views"])
    base = summary["views"]["train1"]
    out = {"criteria": crit, "levels": {}, "sim_dt_s": SIM_DT, "action_repeat": 2, "episode_agent_steps": 125}
    for kind in [*REFERENCE_PATHS, *CS.STRONGER_TRAJECTORIES]:
        name = f"traj_{kind}"
        per_task = summary["views"][name]
        c2w = np.stack([camera_c2w(p) for p in trajectory_path(kind)])
        kin = CS.path_kinematics(c2w, LOOKAT, action_repeat=2, sim_dt=SIM_DT, episode_agent_steps=125)
        checks, failures = path_checks(per_task, base, crit)
        out["levels"][name] = {"kinematics": kin, "passed": not failures, "failures": failures, "checks": checks,
                               "aggregate": summary["aggregate"][name]}
        print(f"{name:22s} passed={not failures} lat/az range {kin['azimuth_deg']} tilt {kin['tilt_deg']} "
              f"v_max {kin['translation_m_per_agent_step']['max']:.4f} m/step rot_max "
              f"{kin['rotation_deg_per_agent_step']['max']:.2f} deg/step {failures if failures else ''}", flush=True)
    for name, per_task in extra.items():
        checks, failures = path_checks(per_task, base, crit)
        out.setdefault("trials", {})[name] = {"passed": not failures, "failures": failures}
    out["train1_aggregate"] = summary["aggregate"]["train1"]
    json_dump(WORK / "paths.json", out)


SIM_DT = 0.0125  # Meta-World v3: frame_skip 5 x timestep 0.0025 s (checked by `paths` against the env)


def path_checks(per_task: dict, base: dict, crit: dict) -> tuple[dict, dict]:
    """Moving-path validation per task against the registered criteria (and the static base camera train1)."""
    checks, failures = {}, {}
    for task, s in per_task.items():
        b = base[task]
        c = {
            "workspace_longest_lost_run": s["workspace"]["longest_lost_run"],
            "workspace_in_view_min": s["workspace"]["in_view_min"],
            "min_depth_m": s["min_depth_m"],
            "min_clearance_m": s["min_clearance_m"],
            **{f"{t}_hidden_excess": s[t]["hidden_fraction"] - b[t]["hidden_fraction"] for t in ("obj", "ee", "goal")},
            **{f"{t}_longest_hidden_run": s[t]["longest_hidden_run"] for t in ("obj", "ee", "goal")},
        }
        bad = []
        if c["workspace_longest_lost_run"] > crit["workspace_longest_lost_run_max"]:
            bad.append("workspace lost")
        if c["workspace_in_view_min"] < crit["workspace_in_view_min"]:
            bad.append(f"workspace in view {c['workspace_in_view_min']:.2f}")
        if c["min_depth_m"] < crit["min_depth_m"]:
            bad.append(f"near-plane clipping risk {c['min_depth_m']:.3f}")
        if c["min_clearance_m"] < crit["min_clearance_m"]:
            bad.append(f"camera clearance {c['min_clearance_m']:.3f}")
        for t in ("obj", "ee", "goal"):
            if c[f"{t}_hidden_excess"] > crit["hidden_fraction_excess_over_base_max"]:
                bad.append(f"{t} hidden +{c[f'{t}_hidden_excess']:.3f}")
            if c[f"{t}_longest_hidden_run"] > crit["longest_hidden_run_max"][t] and \
                    c[f"{t}_longest_hidden_run"] > b[t]["longest_hidden_run"]:
                bad.append(f"{t} hidden run {c[f'{t}_longest_hidden_run']}")
        checks[task] = c
        if bad:
            failures[task] = bad
    return checks, failures


def _compact(per_task: dict) -> dict:
    keep = ("visibility", "hidden_fraction", "arm_occlusion_mean", "arm_heavy_fraction", "median_count",
            "longest_hidden_run")
    return {t: {**{k: {m: round(float(s[k][m]), 4) for m in keep} for k in ("obj", "ee", "goal")},
                "workspace": {m: round(float(v), 4) for m, v in s["workspace"].items()},
                "score": round(float(s["score"]), 4), "min_depth_m": round(s["min_depth_m"], 4),
                "min_clearance_m": round(s["min_clearance_m"], 4)} for t, s in per_task.items()}


def camera_record(name: str, tilt: float, az: float) -> dict:
    from s4d.data.metaworld.cameras import FOVY_DEG, LOOKAT, RADIUS  # noqa: PLC0415
    from s4d.rl.env import (  # noqa: PLC0415
        HELDOUT_V2_PATHS,
        camera_c2w,  # noqa: PLC0415
        orbit_camera,
    )
    from s4d.rl.evaluate import nearest_training_camera  # noqa: PLC0415

    train = [tuple(map(float, c)) for c in TRAIN_CAMERAS]
    seps = [CS.angular_separation_deg((tilt, az), c) for c in train]
    c2w = camera_c2w(orbit_camera(tilt, az))
    return {"name": name, "tilt_deg": tilt, "azimuth_deg": az, "mujoco_azimuth_deg": az, "mujoco_elevation_deg": -tilt,
            "distance_m": RADIUS, "lookat": list(LOOKAT), "fovy_deg": FOVY_DEG, "position_m": c2w[:3, 3].round(6).tolist(),
            "c2w_opencv": c2w.round(8).tolist(),
            "separation_from_training_deg": {f"train{i}": round(x, 2) for i, x in enumerate(seps)},
            "min_separation_from_training_deg": round(min(seps), 2),
            "separation_from_old_heldout_deg": {
                f"eval{i}": round(CS.angular_separation_deg((tilt, az), tuple(map(float, c))), 2)
                for i, c in enumerate(EVAL_CAMERAS)},
            "multiview_replaces_slot_of": f"train{nearest_training_camera(HELDOUT_V2_PATHS[name])}"
            if name in HELDOUT_V2_PATHS else None}


def code_hashes() -> dict:
    files = ["s4d/rl/visibility.py", "s4d/rl/camera_suites.py", "scripts/camera_visibility.py", "s4d/rl/env.py",
             "s4d/rl/evaluate.py", "s4d/data/metaworld/cameras.py", "s4d/data/metaworld/collect.py"]
    return {f: sha256_file(REPO / f) for f in files}


def freeze(args) -> None:
    rule = load_rule()
    selection = json.loads((WORK / "selection.json").read_text())
    if selection["rule_sha256"] != rule["rule_sha256"]:
        raise SystemExit("selection.json was made under a different rule")
    picks = [(selection["table"][st["pick"]]["tilt_deg"], selection["table"][st["pick"]]["azimuth_deg"])
             for st in selection["selection"]]
    if tuple(picks) != tuple(CS.HELDOUT_V2_CAMERAS):
        raise SystemExit(f"camera_suites.HELDOUT_V2_CAMERAS {CS.HELDOUT_V2_CAMERAS} != selection {picks}")
    v2 = json.loads((WORK / "measurements/v2/summary.json").read_text())
    ref = json.loads((WORK / "measurements/reference/summary.json").read_text())
    if v2["frame_stride"] != rule["confirmation_frame_stride"] or ref["frame_stride"] != 1:
        raise SystemExit("confirmation measurements must cover every frame")
    confirmation = {}
    for name in CS.HELDOUT_V2_NAMES:
        ok, failures = eligibility(v2["views"][name], rule)
        confirmation[name] = {"eligible_every_frame": ok, "failures": failures}
        if not ok and not CS.HELDOUT_V2_TASK_EXCEPTIONS:
            raise SystemExit(f"{name} fails confirmation: {failures}; apply rule step 6")
    old = {}
    for name in [f"eval{i}" for i in range(len(EVAL_CAMERAS))] + [f"train{i}" for i in range(len(TRAIN_CAMERAS))]:
        ok, failures = eligibility(ref["views"][name], rule)
        old[name] = {"would_pass_v2_rule": ok, "failures": failures}
    paths_result = json.loads((WORK / "paths.json").read_text())
    for name in [f"traj_{k}" for k in CS.STRONGER_TRAJECTORIES]:
        if not paths_result["levels"][name]["passed"]:
            raise SystemExit(f"{name} failed validation; reduce it per rule step 8 first")
    cams = dict(zip(CS.HELDOUT_V2_NAMES, CS.HELDOUT_V2_CAMERAS))
    measurement_files = sorted((WORK / "measurements").glob("*/*"))
    definition = {
        "suite": CS.SUITE_V2,
        "version": 1,
        "frozen": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "directive": "user 2026-10-29: better-defined held-out cameras and stronger moving cameras (evaluation only)",
        "rule_file": CS.RULE_FILE,
        "rule_sha256": rule["rule_sha256"],
        "frozen_before_policy_evaluation": True,
        "policy_inputs": "normal, unmodified scene; diagnostic arm-free renders are analysis-only",
        "heldout_v2": {
            "cameras": [camera_record(n, t, a) for n, (t, a) in cams.items()],
            "common_rig": not CS.HELDOUT_V2_TASK_EXCEPTIONS,
            "task_exceptions": CS.HELDOUT_V2_TASK_EXCEPTIONS,
            "selection_steps": selection["selection"],
            "candidates": selection["candidates"],
            "eligible_candidates": selection["eligible"],
            "confirmation_every_frame": confirmation,
            "visibility": {n: {"aggregate": v2["aggregate"][n], "per_task": _compact(v2["views"][n])} for n in cams},
        },
        "reference_cameras": {
            "training": {f"train{i}": {"tilt_deg": float(t), "azimuth_deg": float(a)}
                         for i, (t, a) in enumerate(TRAIN_CAMERAS)},
            "old_heldout_eval0_3": {
                "label": "occlusion-heavy stress test (suite v1, unchanged; results kept in eval.jsonl, never mixed)",
                "cameras": {f"eval{i}": {"tilt_deg": float(t), "azimuth_deg": float(a)}
                            for i, (t, a) in enumerate(EVAL_CAMERAS)},
            },
            "under_v2_rule": old,
            "visibility": {n: {"aggregate": ref["aggregate"][n], "per_task": _compact(ref["views"][n])}
                           for n in ref["views"]},
        },
        "moving_paths": {
            "levels": {k: {"kind": kind, **amps} for k, (kind, amps) in CS.STRONGER_TRAJECTORIES.items()},
            "target_levels": rule["moving_paths"]["levels_target"],
            "base_camera": "train1", "period_poses": 72, "starting_phase": "pose 0 at the reset render, as the 1x paths",
            "note": "amplitude grows at a fixed 72-pose period, so camera speed grows with it: combined range-and-speed "
                    "stress tests",
            "validation": {n: {k: v for k, v in r.items() if k != "checks"} for n, r in paths_result["levels"].items()},
            "trials": paths_result.get("trials", {}),
        },
        "evaluation": {
            "output": CS.SUITE_OUTPUTS["v2"],
            "groups": "ref_train0..5 (optional), vis0..3, ref_traj_lateral/circular (optional), "
                      "traj_{lateral,circular}_{2x,3x}",
            "episodes_per_group": "historical per-camera counts (20)",
            "seeds": f"paired: numpy default_rng([run_seed, eval_index, 7, {CS.V2_SEED_TAG}]); group g uses the first n",
            "multiview": "vis* replace their nearest training camera's slot; trajectories replace train1's slot",
            "reevaluation": "final (1M) and last-5 (960k..1M) snapshots of finished runs",
        },
        "code_sha256": code_hashes(),
        "measurement_sha256": {str(f.relative_to(WORK)): sha256_file(f) for f in measurement_files if f.is_file()},
    }
    path = REPO / CS.DEFINITION_FILE
    if path.exists() and not args.refresh:
        raise SystemExit(f"{CS.DEFINITION_FILE} is frozen; it is never overwritten")
    json_dump(path, definition)
    (path.with_suffix(".md")).write_text(definition_markdown(definition, ref, v2, paths_result, rule))
    print(f"[camera_visibility] froze {path}")


def _fmt(x, pct=False):
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def definition_markdown(d: dict, ref: dict, v2: dict, paths_result: dict, rule: dict) -> str:
    lines = [f"# Camera suite v2 (`{d['suite']}`): frozen definition", "",
             f"Frozen {d['frozen']} before any policy evaluation, under the registered rule `{d['rule_file']}` "
             f"(sha256 `{d['rule_sha256'][:16]}`). Evaluation only: training cameras, training data, pretraining and RL "
             "training are unchanged; the original held-out cameras eval0..3 and their `eval.jsonl` history are kept and "
             "reported separately as an **occlusion-heavy stress test**.", "",
             "## Held-out cameras `heldout_v2` (common rig, all 12 tasks)", "",
             "| camera | tilt | azimuth | min sep. from training | nearest training (multi-view slot) "
             "| sep. from eval0..3 |",
             "|---|---|---|---|---|---|"]
    for c in d["heldout_v2"]["cameras"]:
        lines.append(f"| {c['name']} | {c['tilt_deg']:g} | {c['azimuth_deg']:g} "
                     f"| {c['min_separation_from_training_deg']:.1f} deg "
                     f"| {c['multiview_replaces_slot_of']} | "
                     + ", ".join(f"{k} {v:.0f}" for k, v in c["separation_from_old_heldout_deg"].items()) + " |")
    lines += ["", f"Task-specific exceptions: {d['heldout_v2']['task_exceptions'] or 'none'}. "
              f"{d['heldout_v2']['eligible_candidates']} of {d['heldout_v2']['candidates']} candidates were eligible.", "",
              "## Visibility (all frames of 10 expert + 5 random episodes per task, 12 tasks)", "",
              "visibility = mean visible share of the arm-free footprint; hidden = frames with nothing visible; arm = mean "
              "arm-occluded share; worst task = minimum over tasks.", "",
              "| camera | set | worst-task score | obj vis (min task) | obj hidden (max task) "
              "| obj arm occl. (mean / max task) | gripper vis (min) | gripper arm occl. (mean / max) | goal vis (min) "
              "| goal hidden (max) | workspace visible (mean) | v2 rule |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(n, "v2 held-out", v2["aggregate"][n], "pass") for n in CS.HELDOUT_V2_NAMES]
    verdict = {n: "pass" if r["would_pass_v2_rule"] else "fail" for n, r in d["reference_cameras"]["under_v2_rule"].items()}
    rows += [(n, "training", ref["aggregate"][n], verdict[n]) for n in ref["aggregate"] if n.startswith("train")]
    rows += [(n, "v1 held-out (stress test)", ref["aggregate"][n], verdict[n])
             for n in ref["aggregate"] if n.startswith("eval")]
    for n, kind, a, passed in rows:
        lines.append(f"| {n} | {kind} | {a['score_min']:.3f} ({a['worst_task']}) | {a['obj']['visibility_min']:.3f} "
                     f"| {_fmt(a['obj']['hidden_fraction_max'], True)} | {_fmt(a['obj']['arm_occlusion_mean'], True)} / "
                     f"{_fmt(a['obj']['arm_occlusion_max'], True)} | {a['ee']['visibility_min']:.3f} | "
                     f"{_fmt(a['ee']['arm_occlusion_mean'], True)} / {_fmt(a['ee']['arm_occlusion_max'], True)} | "
                     f"{a['goal']['visibility_min']:.3f} | {_fmt(a['goal']['hidden_fraction_max'], True)} | "
                     f"{a['workspace']['visible_mean']:.3f} | {passed} |")
    lines += ["", "Why eval0..3 fail the v2 rule (task: first failures):", ""]
    for n in [k for k in d["reference_cameras"]["under_v2_rule"] if k.startswith("eval")]:
        f = d["reference_cameras"]["under_v2_rule"][n]["failures"]
        reasons = "; ".join(f"{t}: {', '.join(v[:2])}" for t, v in list(f.items())[:4])
        lines.append(f"- {n}: " + ("passes" if not f else reasons + (f" (+{len(f) - 4} more tasks)" if len(f) > 4 else "")))
    lines += ["", "## Moving paths (train1 base, 72-pose period, starting phase pose 0 at reset)", "",
              "Increasing the amplitude at a fixed period increases the camera speed too: the 2x/3x levels are combined "
              "range-and-speed stress tests. Episode: 126 renders (reset + 125 agent steps of 2 simulator steps, "
              f"{1000 * paths_result['sim_dt_s']:.1f} ms each), pose = render index mod 72 (1.75 periods).", "",
              "| path | amplitude | azimuth range | tilt range | distance range | max / mean translation per agent step "
              "| max / mean rotation per agent step | per sim step (max) | max speed | validation |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in paths_result["levels"].items():
        k = r["kinematics"]
        kind = name.removeprefix("traj_")
        reference = {"lateral": {"lateral_amplitude": 0.12},
                     "circular": {"circular_azimuth_deg": 10.0, "circular_elevation_deg": 6.0}}
        amps = CS.STRONGER_TRAJECTORIES.get(kind, ("", {}))[1] or reference[kind]
        amp = (f"+/-{amps['lateral_amplitude']:g} m" if "lateral_amplitude" in amps
               else f"+/-{amps['circular_azimuth_deg']:g} deg az, +/-{amps['circular_elevation_deg']:g} deg tilt")
        lines.append(
            f"| {name} | {amp} | {k['azimuth_deg'][0]:.1f}..{k['azimuth_deg'][1]:.1f} "
            f"| {k['tilt_deg'][0]:.1f}..{k['tilt_deg'][1]:.1f} | {k['distance_m'][0]:.3f}..{k['distance_m'][1]:.3f} "
            f"| {1000 * k['translation_m_per_agent_step']['max']:.1f} / "
            f"{1000 * k['translation_m_per_agent_step']['mean']:.1f} mm | {k['rotation_deg_per_agent_step']['max']:.2f} / "
            f"{k['rotation_deg_per_agent_step']['mean']:.2f} deg | {1000 * k['translation_m_per_sim_step']['max']:.1f} mm, "
            f"{k['rotation_deg_per_sim_step']['max']:.2f} deg | {k['translation_m_per_s']['max']:.2f} m/s, "
            f"{k['rotation_deg_per_s']['max']:.0f} deg/s "
            f"| {'pass' if r['passed'] else 'FAIL: ' + json.dumps(r['failures'])[:120]} |")
    lines += ["", "## Evaluation", "",
              f"`scripts/eval_rl.py --suites v2` writes `{CS.SUITE_OUTPUTS['v2']}` (never `eval.jsonl`). Groups: "
              "`ref_train0..5` and `ref_traj_{lateral,circular}` (matched references, `--v2-reference`), `vis0..3`, "
              "`traj_{lateral,circular}_{2x,3x}`; 20 episodes each; paired seeds "
              f"`default_rng([run_seed, eval_index, 7, {CS.V2_SEED_TAG}])` shared by every v2 group, so every method with "
              "the same run seed sees identical initial conditions and camera paths. `--snapshots last5` re-evaluates "
              "960k..1M of a finished run (final = 1M). Multi-view methods: a vis camera replaces its nearest training "
              "camera's slot; trajectories replace train1's slot; per-frame K and c2w come from the rendered pose.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- media
MEDIA_TASKS = ("hammer", "door-open", "peg-insert-side")  # preview videos: paper-study tasks (2026-10-28 directive)
SHEET_FRAMES = (0, 62)  # reset frame and mid-episode frame of the noise-free expert episode
DISPLAY = 256  # display resolution of the media; policies see 128x128 renders of the same cameras


def _font(size: int):
    from PIL import ImageFont  # noqa: PLC0415

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _rgb_job(job: tuple) -> dict:
    """RGB frames of the noise-free expert episode (representative episode 0) for every view at chosen renders."""
    task, views, size, wanted = job
    import mujoco  # noqa: PLC0415

    from s4d.rl.visibility import TaskEnv, free_camera, representative_episodes  # noqa: PLC0415

    env = TaskEnv(task)
    renderer = mujoco.Renderer(env.model, height=size, width=size)
    frames: dict[str, dict[int, np.ndarray]] = {name: {} for name in views}

    def on_frame(k: int) -> None:
        if k not in wanted:
            return
        for name, path in views.items():
            renderer.update_scene(env.data, camera=free_camera(path[k % len(path)]))
            frames[name][k] = renderer.render().copy()

    try:
        env.rollout(representative_episodes(task)[0], on_frame)
    finally:
        renderer.close()
        env.close()
    return {"task": task, "frames": frames}


def _label(img: np.ndarray, lines: list[str], size: int = 12, color=(20, 20, 20), height: int | None = None):
    from PIL import Image, ImageDraw  # noqa: PLC0415

    h = height or (len(lines) * (size + 3) + 4)
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + h), (250, 250, 250))
    canvas.paste(Image.fromarray(img), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((3, img.shape[0] + 2 + i * (size + 3)), line, fill=color, font=_font(size))
    return np.asarray(canvas)


def _stack(rows: list[list[np.ndarray]], pad: int = 4, bg=(255, 255, 255)) -> np.ndarray:
    h = max(t.shape[0] for r in rows for t in r)
    w = max(t.shape[1] for r in rows for t in r)
    out = np.full((len(rows) * (h + pad) + pad, max(len(r) for r in rows) * (w + pad) + pad, 3), bg, np.uint8)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            y0, x0 = pad + i * (h + pad), pad + j * (w + pad)
            out[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    return out


def _header(text: str, width: int, size: int = 22, sub: str | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    h = size + 10 + (size if sub else 0)
    canvas = Image.new("RGB", (width, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), text, fill=(0, 0, 0), font=_font(size))
    if sub:
        draw.text((6, 6 + size), sub, fill=(70, 70, 70), font=_font(int(size * 0.65)))
    return np.asarray(canvas)


def _vis_lines(s: dict) -> list[str]:
    def line(k: str) -> str:
        arm = s[k]["arm_occlusion_mean"]
        arm_text = "n/a" if arm != arm else f"{100 * arm:.0f}%"
        return f"{k} {s[k]['visibility']:.2f} hid {100 * s[k]['hidden_fraction']:.0f}% arm {arm_text}"

    return [line("obj"), line("ee"), line("goal") + f"  ws {s['workspace']['visible_mean']:.2f}"]


def contact_sheet(title: str, subtitle: str, views: dict, cameras: dict, summary: dict, frames: dict, out: Path):
    """Rows: tasks; columns: each camera at the reset and the mid-episode frame; visibility over all frames."""
    rows = []
    head = []
    for name in views:
        tilt, az = cameras[name]
        agg = summary["aggregate"][name]
        for k in SHEET_FRAMES:
            when = "reset frame" if k == 0 else f"mid-episode (render {k})"
            text = [f"{name} (tilt {tilt:g}, az {az:g})", f"{when}; worst-task score {agg['score_min']:.2f}"]
            head.append(_label(np.full((1, DISPLAY, 3), 255, np.uint8), text, 14))
    rows.append(head)
    for task in summary["tasks"]:
        row = []
        for name in views:
            s = summary["views"][name][task]
            for k in SHEET_FRAMES:
                lines = [f"{task}  [{name}, render {k}]"] + (_vis_lines(s) if k == SHEET_FRAMES[-1] else [
                    f"score {s['score']:.2f}", f"obj px {s['obj']['median_count']:.0f} ee px {s['ee']['median_count']:.0f}",
                    f"goal in view {s['goal_point_in_view']:.2f}"])
                row.append(_label(frames[task][name][k], lines, 12))
        rows.append(row)
    grid = _stack(rows)
    sheet = np.concatenate([_header(title, grid.shape[1], 26, subtitle), grid], axis=0)
    from PIL import Image  # noqa: PLC0415

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out, optimize=True)
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8,
                            ffmpeg_log_level="error") as writer:
        for f in frames:
            writer.append_data(f)


def layout_figure(out: Path, vis: dict, paths: dict) -> None:
    """Top-down view of the camera positions and a tilt/azimuth map (PIL; matplotlib is not installed)."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from s4d.data.metaworld.cameras import LOOKAT  # noqa: PLC0415

    W, H = 1500, 720
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f, fs, fl = _font(15), _font(12), _font(20)
    colors = {"train": (31, 119, 180), "eval": (214, 39, 40), "vis": (44, 160, 44)}
    path_colors = {"1x": (120, 120, 120), "2x": (255, 127, 14), "3x": (148, 103, 189)}

    def position(tilt, az, dist=1.0):
        return np.asarray(LOOKAT) + dist * CS.view_direction(tilt, az)

    # panel 1: top-down (x right, y up), centred on the look-at point
    cx, cy, scale = 360, 380, 260

    def top(p):
        return cx + scale * p[0], cy - scale * (p[1] - LOOKAT[1])

    d.text((20, 15), "Top-down view (world x right, y up; robot base at the origin)", fill=(0, 0, 0), font=fl)
    tx0, ty0 = top((-0.7, 0.2))
    tx1, ty1 = top((0.7, 1.0))
    d.rectangle([tx0, ty1, tx1, ty0], outline=(160, 120, 80), width=2)
    d.text((tx0 + 4, ty1 + 4), "table", fill=(160, 120, 80), font=fs)
    wx0, wy0 = top((-0.35, 0.40))
    wx1, wy1 = top((0.35, 0.90))
    d.rectangle([wx0, wy1, wx1, wy0], outline=(200, 200, 200), width=1)
    d.text((wx0 + 4, wy0 - 16), "workspace grid", fill=(150, 150, 150), font=fs)
    bx, by = top((0.0, 0.0))
    d.ellipse([bx - 9, by - 9, bx + 9, by + 9], fill=(90, 90, 90))
    d.text((bx + 12, by - 6), "robot base", fill=(60, 60, 60), font=fs)
    lx, ly = top(LOOKAT)
    d.line([lx - 6, ly, lx + 6, ly], fill=(0, 0, 0))
    d.line([lx, ly - 6, lx, ly + 6], fill=(0, 0, 0))
    for level, kinds in paths.items():
        for c2w in kinds.values():
            pts = [top(m[:3, 3]) for m in c2w] + [top(c2w[0][:3, 3])]
            d.line(pts, fill=path_colors[level], width=3 if level != "1x" else 2)
    for group, cams in (("train", {f"train{i}": c for i, c in enumerate(TRAIN_CAMERAS)}),
                        ("eval", {f"eval{i}": c for i, c in enumerate(EVAL_CAMERAS)}), ("vis", vis)):
        for name, (tilt, az) in cams.items():
            x, y = top(position(tilt, az))
            r = 8
            if group == "train":
                d.ellipse([x - r, y - r, x + r, y + r], fill=colors[group])
            elif group == "eval":
                d.rectangle([x - r, y - r, x + r, y + r], fill=colors[group])
            else:
                d.polygon([(x, y - r - 3), (x + r + 2, y + r), (x - r - 2, y + r)], fill=colors[group])
            d.line([x, y, lx, ly], fill=colors[group] + (0,), width=1)
            d.text((x + 10, y - 8), name, fill=colors[group], font=fs)
    # panel 2: azimuth x tilt map
    ox, oy, pw, ph = 800, 90, 660, 520
    d.text((ox, 15), "Viewing directions: azimuth (deg) x tilt (deg)", fill=(0, 0, 0), font=fl)

    def sky(tilt, az):
        return ox + pw * ((az % 360) / 360.0), oy + ph * (1 - (tilt - 20) / 50.0)

    d.rectangle([ox, oy, ox + pw, oy + ph], outline=(0, 0, 0))
    for az in range(0, 361, 45):
        x, _ = sky(20, az if az < 360 else 359.999)
        d.line([x, oy, x, oy + ph], fill=(230, 230, 230))
        d.text((x - 10, oy + ph + 4), str(az), fill=(0, 0, 0), font=fs)
    for tilt in range(20, 71, 10):
        _, y = sky(tilt, 0)
        d.line([ox, y, ox + pw, y], fill=(230, 230, 230))
        d.text((ox - 26, y - 7), str(tilt), fill=(0, 0, 0), font=fs)
    train = [tuple(map(float, c)) for c in TRAIN_CAMERAS]
    for tilt in np.arange(20, 70.1, 1.0):
        for az in np.arange(0, 360, 2.0):
            if CS.min_separation_deg((tilt, az), train) < 25.0:
                x, y = sky(tilt, az)
                d.point((x, y), fill=(198, 219, 239))
    for tilt, az in CS.candidate_grid():
        x, y = sky(tilt, az)
        d.point((x, y), fill=(120, 120, 120))
    for level, kinds in paths.items():
        for c2w in kinds.values():
            pts = []
            for m in c2w:
                rel = m[:3, 3] - np.asarray(LOOKAT)
                tilt = math.degrees(math.asin(rel[2] / np.linalg.norm(rel)))
                az = math.degrees(math.atan2(-rel[1], -rel[0])) % 360
                pts.append(sky(tilt, az))
            d.line(pts + [pts[0]], fill=path_colors[level], width=2)
    for group, cams in (("train", {f"train{i}": c for i, c in enumerate(TRAIN_CAMERAS)}),
                        ("eval", {f"eval{i}": c for i, c in enumerate(EVAL_CAMERAS)}), ("vis", vis)):
        for name, (tilt, az) in cams.items():
            x, y = sky(tilt, az)
            r = 7
            if group == "train":
                d.ellipse([x - r, y - r, x + r, y + r], fill=colors[group])
            elif group == "eval":
                d.rectangle([x - r, y - r, x + r, y + r], fill=colors[group])
            else:
                d.polygon([(x, y - r - 3), (x + r + 2, y + r), (x - r - 2, y + r)], fill=colors[group])
            d.text((x + 9, y - 7), name, fill=colors[group], font=fs)
    ly0 = oy + ph + 30
    items = [("train0..5 (training)", "train"), ("eval0..3 (v1 held-out: occlusion-heavy stress test)", "eval"),
             ("vis0..3 (v2 held-out, visibility-selected)", "vis")]
    for i, (text, g) in enumerate(items):
        d.rectangle([ox + 4, ly0 + i * 20, ox + 16, ly0 + i * 20 + 12], fill=colors[g])
        d.text((ox + 22, ly0 + i * 20 - 2), text, fill=(0, 0, 0), font=f)
    for i, (lvl, col) in enumerate(path_colors.items()):
        d.line([ox + 420, ly0 + i * 20 + 6, ox + 450, ly0 + i * 20 + 6], fill=col, width=3)
        d.text((ox + 456, ly0 + i * 20 - 2), f"moving paths {lvl} (lateral, circular)", fill=(0, 0, 0), font=f)
    d.text((ox, ly0 + 66), "light blue: within 25 deg of a training camera (excluded); grey dots: candidate grid",
           fill=(90, 90, 90), font=fs)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def media(args) -> None:
    from s4d.rl.env import HELDOUT_PATHS, HELDOUT_V2_PATHS, TRAIN_PATHS, camera_c2w, trajectory_path  # noqa: PLC0415

    out = WORK / "media"
    vis = dict(zip(CS.HELDOUT_V2_NAMES, CS.HELDOUT_V2_CAMERAS))
    v2 = json.loads((WORK / "measurements/v2/summary.json").read_text())
    ref = json.loads((WORK / "measurements/reference/summary.json").read_text())
    cams = {**reference_cameras_dict(), **vis}
    static = {**TRAIN_PATHS, **HELDOUT_PATHS, **HELDOUT_V2_PATHS}
    jobs = [(task, static, DISPLAY, set(SHEET_FRAMES)) for task in TASKS]
    level_names = {"1x": ("lateral", "circular"), "2x": ("lateral_2x", "circular_2x"),
                   "3x": ("lateral_3x", "circular_3x")}
    moving = {f"traj_{k}": trajectory_path(k) for ks in level_names.values() for k in ks}
    jobs += [(task, {**moving, **HELDOUT_V2_PATHS}, DISPLAY, set(range(126))) for task in MEDIA_TASKS]
    sheets: dict[str, dict] = {}
    clips: dict[str, dict] = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_rgb_job, jobs):
            if len(next(iter(result["frames"].values()))) == len(SHEET_FRAMES):
                sheets[result["task"]] = result["frames"]
            else:
                clips[result["task"]] = result["frames"]
            print(f"[camera_visibility] media frames {result['task']}", flush=True)
    written = []
    written.append(contact_sheet(
        "Camera suite v2: held-out cameras vis0..vis3 (heldout_v2)",
        "Selected before any policy evaluation by the registered visibility rule; numbers over all frames of 10 expert "
        "+ 5 random episodes (obj/ee/goal: mean visible share of the arm-free footprint, hidden-frame %, arm-occluded %)",
        HELDOUT_V2_PATHS, cams, v2, sheets, out / "contact_sheet_heldout_v2.png"))
    written.append(contact_sheet(
        "Original held-out cameras eval0..eval3: OCCLUSION-HEAVY STRESS TEST (suite v1, unchanged)",
        "Kept and reported separately; same measurements as the v2 sheet", HELDOUT_PATHS, cams, ref, sheets,
        out / "contact_sheet_eval0-3_occlusion_heavy_stress_test.png"))
    written.append(contact_sheet(
        "Training cameras train0..train5 (reference)", "Same measurements; unchanged training rig",
        TRAIN_PATHS, cams, ref, sheets, out / "contact_sheet_train0-5_reference.png"))
    font = _font(14)
    for task, frames in clips.items():
        for kind in ("lateral", "circular"):
            names = [f"traj_{kind}", f"traj_{kind}_2x", f"traj_{kind}_3x"]
            video = []
            for k in range(72):  # one full period from the reset pose
                tiles = [_label(frames[n][k], [f"{n}  pose {k}/72"], 14) for n in names]
                video.append(_stack([tiles]))
            path = out / f"preview_{task}_{kind}_1x-2x-3x.mp4"
            write_video(path, video)
            written.append(path)
        video = []
        for k in range(126):
            tiles = [_label(frames[n][k], [f"{n}  render {k}"], 14) for n in HELDOUT_V2_PATHS]
            video.append(_stack([tiles]))
        path = out / f"preview_{task}_heldout_v2.mp4"
        write_video(path, video)
        written.append(path)
    del font
    paths = {lvl: {k: np.stack([camera_c2w(p) for p in trajectory_path(k)]) for k in ks} for lvl, ks in level_names.items()}
    layout_figure(out / "camera_layout.png", vis, paths)
    written.append(out / "camera_layout.png")
    json_dump(out / "index.json", {"files": [str(p.relative_to(REPO)) for p in written],
                                   "display_resolution": DISPLAY, "policy_resolution": 128,
                                   "sheet_frames": SHEET_FRAMES, "video_tasks": MEDIA_TASKS,
                                   "video_fps": 20, "real_time_fps": 40})
    for p in written:
        print(p, flush=True)


def reference_cameras_dict() -> dict[str, tuple[float, float]]:
    return CS.reference_cameras()


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--set", required=True)
    m.add_argument("--tasks", nargs="+", default=list(TASKS))
    m.add_argument("--frame-stride", type=int, default=1, help="measure every n-th render of each episode")
    m.add_argument("--size", type=int, default=128, help="analysis resolution (the policies' 128x128)")
    m.add_argument("--workers", type=int, default=2, help="render processes (~0.2-0.45 GB GPU memory each)")
    m.add_argument("--views-per-job", type=int, default=1000)
    for name in ("register-rule", "register-amendment", "register-amendment-a2", "select", "paths"):
        sub.add_parser(name)
    fz = sub.add_parser("freeze")
    fz.add_argument("--refresh", action="store_true", help="rewrite after a documented correction (never the cameras)")
    md = sub.add_parser("media")
    md.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    # MuJoCo-only tool: guard_mujoco enforces the allowlisted single UUID and pins the EGL device; torch/CUDA is never
    # initialised, which keeps a CUDA context (~0.5 GB) off the shared MIG instance.
    from _bootstrap import guard_mujoco  # noqa: PLC0415

    guard_mujoco()
    {"measure": measure, "media": media, "register-rule": register_rule, "select": select, "paths": paths,
     "freeze": freeze, "register-amendment": register_amendment,
     "register-amendment-a2": register_amendment_a2}[args.command](args)


if __name__ == "__main__":  # spawned workers re-import this file
    main()
