"""Development-host tool of the live M5 helper (``s4d.rwlearn.deploy.m5_live``). Never runs on the robot PC, never talks
to a robot.

    # sample the training store's M5 surface once (needs the URDF meshes) -> asset shipped to the robot PC
    .venv/bin/python -I scripts/robot_pc/m5_asset.py export --recording /home/ws/data/real_world/mug-hanging/episode_000001.h5 \
        --out outputs/robot_pc/m5/panda_robotiq_2f85_droid_iso50.m5.npz
    # validate the helper against released training-store episodes (pre-registered criterion, ROB-STATUS.md)
    .venv/bin/python -I scripts/robot_pc/m5_asset.py validate --asset <npz> --urdf <urdf> \
        --pair <store episode.h5> <raw recording.h5> [--pair ...] --json m5-validation.json

``validate`` checks, per episode, (a) the store-trajectory path: the helper's surface + ``pipeline.robot_state_at`` +
``train_store.set_surface_motion`` on the store's own robot rows and frame times, and (b) the LIVE path: the raw
recording's robot rows, gripper register samples and colour frame times replayed in arrival order through
``m5_live.LiveM5`` (sets emitted when final, states when final), both against the stored ``sync/frames`` and
``sync/surface_disp_m``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

TOLERANCE_M = 1e-9  # pre-registered (ROB-STATUS.md): numerical equivalence of the same surface FK
FRAME_DELAY_NS = 65_000_000  # frames reach the host ~65 ms after their device stamp (recorder t_recv - t)


def robot_description(recording: Path, pipeline_config: Path):
    """The recording's RobotDescription exactly as the adapter builds it from the pipeline config's robot section."""
    import yaml

    from s4d.realworld.adapters.panda_teleop import PandaTeleopH5Adapter

    rc = yaml.safe_load(pipeline_config.read_text())["adapter"]["config"]["robot"]
    return PandaTeleopH5Adapter({"episodes": {"e": str(recording)}, "robot": rc}).read_meta("e").robot


def cmd_export(args) -> dict:
    from s4d.rwlearn.deploy.m5_live import export_surface

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = export_surface(robot_description(args.recording, args.pipeline_config), args.out)
    return {"asset": str(args.out), "header": header}


def _compare(name: str, frames_ref, disp_ref, frames, disp) -> dict:
    complete = (frames_ref >= 0).all(1)
    same_frames = bool(np.array_equal(frames, frames_ref))
    fin = np.isfinite(disp_ref) & complete
    err = np.abs(disp[fin] - disp_ref[fin]) if fin.any() else np.zeros(0)
    inf_same = bool(np.array_equal(np.isinf(disp[complete]), np.isinf(disp_ref[complete])))
    acc_ref, acc = disp_ref[complete] <= 0.005, disp[complete] <= 0.005
    return {"path": name, "sets": int(complete.sum()), "frames_identical": same_frames,
            "max_abs_err_m": float(err.max()) if len(err) else 0.0, "inf_pattern_identical": inf_same,
            "decisions_identical": bool(np.array_equal(acc, acc_ref)),
            "pass": bool(same_frames and inf_same and np.array_equal(acc, acc_ref)
                         and (not len(err) or err.max() <= TOLERANCE_M))}


def validate_pair(motion, store_path: Path, raw_path: Path) -> dict:
    import h5py

    from s4d.realworld.pipeline import robot_state_at
    from s4d.realworld.train_store import mutual_sync, set_surface_motion
    from s4d.rwlearn import actions as A
    from s4d.rwlearn.deploy.m5_live import LiveM5
    from s4d.rwlearn.deploy.replay import RecordedEpisode

    with h5py.File(store_path, "r") as f:
        cams = json.loads(f.attrs["cameras"])
        stamps = [np.asarray(f[f"cameras/{c}/timestamps_ns"][()], np.int64) for c in cams]
        r = f["robot"]
        traj = SimpleNamespace(timestamps_ns=np.asarray(r["timestamps_ns"][()], np.int64),
                               joint_positions=np.asarray(r["joint_positions"][()], np.float64),
                               gripper_state=np.asarray(r["gripper_state"][()], np.float64))
        max_sync = int(r.attrs["robot_sync_max_ns"])
        frames_ref = np.asarray(f["sync/frames"][()], np.int64)
        disp_ref = np.asarray(f["sync/surface_disp_m"][()], np.float64)
        ident = {"episode": str(f.attrs["episode_id"]), "raw_sha256": str(f.attrs["raw_sha256"]),
                 "store_code_commit": str(f.attrs["code_commit"]), "robot_sync_max_ns": max_sync,
                 "store_surface": str(f["sync"].attrs.get("surface"))}
    # (a) store trajectory -> helper surface (authoritative state + motion functions)
    meta = SimpleNamespace(trajectory=traj)
    states, ok = [], []
    for s in stamps:
        q, g, inside, _ = robot_state_at(meta, s, max_sync)
        states.append(list(zip(q, g)))
        ok.append(np.asarray(inside, bool))
    frames_a, _ = mutual_sync(stamps)
    disp_a = set_surface_motion(motion.points, frames_a, states, ok)
    out = {**ident, "cameras": cams, "store_path": _compare("store_trajectory", frames_ref, disp_ref, frames_a, disp_a)}

    # (b) live path from the raw recording, in arrival order
    rec = RecordedEpisode(raw_path)
    if [list(rec.timestamps_ns(c)) for c in cams] != [list(s) for s in stamps]:
        raise SystemExit(f"{raw_path}: colour frame times differ from the store's")
    raw = rec.raw
    with h5py.File(raw_path, "r") as f:
        q_rows = np.asarray(f["robot/q"][()], np.float64)
    events = [(int(t) + 100_000, 0, "robot", (int(t), q)) for t, q in zip(raw.robot_t, q_rows)]
    events += [(int(t) + 1_000_000, 1, "gripper", (int(t), float(p)))
               for t, p in zip(raw.gripper_t, raw.gripper_position)]
    for c, s in zip(cams, stamps):
        events += [(int(t) + FRAME_DELAY_NS, 2, "frame", (c, i, int(t))) for i, t in enumerate(s)]
    events.sort(key=lambda e: (e[0], e[1]))
    live = LiveM5(motion, cams, A.NOMINAL_PERIOD_S, max_sync)
    emitted, waits = [], []
    for when, _, kind, payload in events:
        if kind == "robot":
            live.add_robot(*payload)
        elif kind == "gripper":
            live.add_gripper(*payload)
        else:
            live.add_frame(*payload)
        for s in live.poll():
            emitted.append(s)
            waits.append((when - s["observation_time_ns"]) / 1e6)
    emitted += live.poll(end_of_stream=True)
    frames_b = np.full_like(frames_ref, -1)
    disp_b = np.zeros(len(frames_ref))
    for s in emitted:
        k = s["frame_ids"][cams[0]]
        frames_b[k] = [s["frame_ids"][c] for c in cams]
        disp_b[k] = s["surface_disp_m"]
    complete = (frames_ref >= 0).all(1)
    frames_b[~complete] = frames_ref[~complete]  # incomplete sets are dropped live (counted), as the loaders drop them
    out["live_path"] = _compare("live_raw_recording", frames_ref, disp_ref, frames_b, disp_b)
    out["live_path"].update(emitted=len(emitted), incomplete_dropped=int(live.counts["incomplete_sets"]),
                            incomplete_in_store=int((~complete).sum()),
                            emit_delay_after_newest_member_ms=None if not waits else
                            {"p50": float(np.percentile(waits, 50)), "p95": float(np.percentile(waits, 95)),
                             "max": float(np.max(waits)),
                             "note": f"arrival model: frames +{FRAME_DELAY_NS / 1e6:.0f} ms, robot rows +0.1 ms, "
                                     "gripper +1 ms after their stamps; includes waiting for set finality"})
    out["pass"] = bool(out["store_path"]["pass"] and out["live_path"]["pass"])
    return out


# ------------------------------------------------------------------------------------------------ negative cases
def _store_arrays(store_path: Path) -> dict:
    import h5py

    with h5py.File(store_path, "r") as f:
        cams = json.loads(f.attrs["cameras"])
        r = f["robot"]
        return {"episode": str(f.attrs["episode_id"]), "cams": cams,
                "stamps": [np.asarray(f[f"cameras/{c}/timestamps_ns"][()], np.int64) for c in cams],
                "t": np.asarray(r["timestamps_ns"][()], np.int64), "q": np.asarray(r["joint_positions"][()], np.float64),
                "g": np.asarray(r["gripper_state"][()], np.float64), "max_sync": int(r.attrs["robot_sync_max_ns"])}


def _reference(motion, a: dict):
    """The training rule on (perturbed) arrays: mutual_sync + robot_state_at + set_surface_motion."""
    from s4d.realworld.pipeline import robot_state_at
    from s4d.realworld.train_store import mutual_sync, set_surface_motion

    meta = SimpleNamespace(trajectory=SimpleNamespace(timestamps_ns=a["t"], joint_positions=a["q"], gripper_state=a["g"]))
    states, ok = [], []
    for s in a["stamps"]:
        q, g, inside, _ = robot_state_at(meta, s, a["max_sync"])
        states.append(list(zip(q, g)))
        ok.append(np.asarray(inside, bool))
    frames, _ = mutual_sync(a["stamps"])
    return frames, set_surface_motion(motion.points, frames, states, ok)


def _live(motion, a: dict):
    """The live helper fed in arrival order (frames +65 ms, robot rows +0.1 ms, register samples +1 ms; the register
    samples are the store's register on the robot rows, so the adapter's interpolation is the identity)."""
    from s4d.rwlearn import actions as A
    from s4d.rwlearn.deploy.m5_live import LiveM5

    events = [(int(t) + 100_000, 0, "robot", (int(t), q)) for t, q in zip(a["t"], a["q"])]
    events += [(int(t) + 1_000_000, 1, "gripper", (int(t), float(g))) for t, g in zip(a["t"], a["g"])]
    for c, st in zip(a["cams"], a["stamps"]):
        events += [(int(t) + FRAME_DELAY_NS, 2, "frame", (c, i, int(t))) for i, t in enumerate(st)]
    live = LiveM5(motion, a["cams"], A.NOMINAL_PERIOD_S, a["max_sync"])
    emitted = []
    for _, _, kind, payload in sorted(events, key=lambda e: (e[0], e[1])):
        getattr(live, f"add_{kind}")(*payload)
        emitted += live.poll()
    emitted += live.poll(end_of_stream=True)
    n = len(a["stamps"][0])
    frames, disp = np.full((n, len(a["cams"])), -1, np.int64), np.zeros(n)
    for s in emitted:
        k = s["frame_ids"][a["cams"][0]]
        frames[k] = [s["frame_ids"][c] for c in a["cams"]]
        disp[k] = s["surface_disp_m"]
    return frames, disp, dict(live.counts), live.sets.irregular_intervals


def _perturb(a: dict, case: str) -> dict:
    a = {k: (v.copy() if isinstance(v, np.ndarray) else [x.copy() for x in v] if k == "stamps" else v)
         for k, v in a.items()}
    period = int(np.median(np.diff(a["stamps"][0])))
    n = len(a["t"])
    if case == "large_motion":  # fast joint oscillation: most sets move > 5 mm
        a["q"] = a["q"] + 0.05 * np.sin(2 * np.pi * 3.0 * (a["t"] - a["t"][0])[:, None] / 1e9)
    elif case in ("camera_offset_0.45P", "camera_offset_0.55P"):  # members near / beyond half a period
        a["stamps"][2] = a["stamps"][2] + int(float(case.split("_")[-1][:-1]) * period)
    elif case == "camera_drops":  # every 5th frame of camera 2 missing: non-mutual (-1) sets
        a["stamps"][1] = np.delete(a["stamps"][1], np.arange(3, len(a["stamps"][1]), 5))
    elif case == "state_gap":  # 0.5 s without robot rows mid-episode: members without a valid state
        mid = a["t"][n // 2]
        keep = (a["t"] < mid) | (a["t"] > mid + 500_000_000)
        a["t"], a["q"], a["g"] = a["t"][keep], a["q"][keep], a["g"][keep]
    elif case == "state_stops_early":  # the state stream ends at 70 %: later frames are outside (no extrapolation)
        a["t"], a["q"], a["g"] = a["t"][:int(0.7 * n)], a["q"][:int(0.7 * n)], a["g"][:int(0.7 * n)]
    elif case == "state_starts_late":
        a["t"], a["q"], a["g"] = a["t"][int(0.2 * n):], a["q"][int(0.2 * n):], a["g"][int(0.2 * n):]
    elif case == "frozen_state":  # joints held constant for 1 s with advancing times (the M5 rule sees no motion)
        sel = (a["t"] > a["t"][n // 3]) & (a["t"] < a["t"][n // 3] + 1_000_000_000)
        a["q"][sel] = a["q"][np.flatnonzero(sel)[0]]
    elif case == "nan_joints":
        a["q"][n // 2:n // 2 + 20, 3] = np.nan
    elif case != "baseline":
        raise ValueError(case)
    return a


NEGATIVE_CASES = ("baseline", "large_motion", "camera_offset_0.45P", "camera_offset_0.55P", "camera_drops",
                  "state_gap", "state_stops_early", "state_starts_late", "frozen_state", "nan_joints")


def cmd_negative(args) -> dict:
    from s4d.rwlearn.deploy.m5_live import SurfaceMotion, StateTimeline

    motion = SurfaceMotion(args.asset, args.urdf)
    out = {"criterion": "helper frames / inf pattern / 5 mm decisions identical to the training rule on the same "
                        f"(perturbed, in-memory) arrays; finite |diff| <= {TOLERANCE_M} m", "episodes": []}
    for store in args.store:
        base = _store_arrays(Path(store))
        ep = {"episode": base["episode"], "cases": {}}
        for case in NEGATIVE_CASES:
            a = _perturb(base, case)
            f_ref, d_ref = _reference(motion, a)
            f_live, d_live, counts, irregular = _live(motion, a)
            complete = (f_ref >= 0).all(1)
            f_live[~complete] = f_ref[~complete]  # incomplete sets are dropped live (counted), as the loaders do
            r = _compare(case, f_ref, d_ref, f_live, d_live)
            r.update(exercised={"sets": int(complete.sum()), "incomplete": int((~complete).sum()),
                                "rejected_gt_5mm": int((np.isfinite(d_ref) & (d_ref > 0.005) & complete).sum()),
                                "no_valid_state_inf": int((np.isinf(d_ref) & complete).sum())},
                     live_counts=counts, irregular_intervals=irregular)
            ep["cases"][case] = r
        t = StateTimeline(base["max_sync"])  # non-advancing state times: refused (the training rule is undefined)
        t.add_robot(int(base["t"][0]), base["q"][0])
        try:
            t.add_robot(int(base["t"][0]), base["q"][1])
            ep["cases"]["non_advancing_state_time"] = {"pass": False, "detail": "accepted a repeated time"}
        except ValueError as exc:
            ep["cases"]["non_advancing_state_time"] = {"pass": True, "detail": f"refused: {exc}"}
        out["episodes"].append(ep)
    out["pass"] = all(c["pass"] for e in out["episodes"] for c in e["cases"].values())
    return out


def cmd_validate(args) -> dict:
    from s4d.rwlearn.deploy.m5_live import SurfaceMotion

    motion = SurfaceMotion(args.asset, args.urdf)
    episodes = [validate_pair(motion, Path(s), Path(r)) for s, r in args.pair]
    return {"criterion": {"tolerance_m": TOLERANCE_M, "rule": "pre-registered in ROB-STATUS.md 2026-11-10 13:55 KST"},
            "asset": str(args.asset), "asset_sha256": motion.asset_sha256, "urdf": str(args.urdf),
            "header": motion.header, "episodes": episodes, "pass": all(e["pass"] for e in episodes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="m5_asset.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--recording", type=Path, required=True, help="any raw recording of the rig (robot metadata)")
    e.add_argument("--pipeline-config", type=Path, default=REPO / "configs/realworld/mug_hanging.yaml")
    e.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("--asset", type=Path, required=True)
    v.add_argument("--urdf", type=Path, required=True)
    v.add_argument("--pair", nargs=2, action="append", required=True, metavar=("STORE_H5", "RAW_H5"))
    n = sub.add_parser("negative", help="perturbed in-memory copies of release-store episodes vs the training rule")
    n.add_argument("--asset", type=Path, required=True)
    n.add_argument("--urdf", type=Path, required=True)
    n.add_argument("--store", action="append", required=True)
    for p in (e, v, n):
        p.add_argument("--json", type=Path, default=None, help="write the result here")
    args = ap.parse_args(argv)
    result = {"export": cmd_export, "validate": cmd_validate, "negative": cmd_negative}[args.cmd](args)
    text = json.dumps(result, indent=1, allow_nan=False, default=str)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
