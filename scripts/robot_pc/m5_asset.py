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
    for p in (e, v):
        p.add_argument("--json", type=Path, default=None, help="write the result here")
    args = ap.parse_args(argv)
    result = cmd_export(args) if args.cmd == "export" else cmd_validate(args)
    text = json.dumps(result, indent=1, allow_nan=False, default=str)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
