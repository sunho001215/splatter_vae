"""ROBOT PC, live dry run (never commands a robot): live RealSense cameras + read-only robot state -> policy server ->
robot-side validation -> DryRunSink (JSONL log, nothing is sent).

    venv/bin/python -I src/scripts/robot_pc/dry_run.py --policy dynamo-drawer-open --state static --steps 300 \
        --log runs/dryrun-dynamo.jsonl --summary runs/dryrun-dynamo.json

Refuses (before opening any camera) when a recorder / teleop process, a device holder or the teleop collector's IPC
socket exists, i.e. whenever live acquisition could interfere with a recording. ``--state live`` is refused before
anything connects: the read-only switch_controller RobotState reader exists (``switch_controller.LiveStateReader``) but
no read-only gripper register source does, so no observation could be bound. ``--state static`` binds a FIXED
labelled state for latency measurements only (its actions are meaningless and only logged).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc import config as C  # noqa: E402
from ops.robot_pc import selection  # noqa: E402


def main(argv=None) -> int:
    from ops.robot_pc import dryrun
    from ops.robot_pc.cameras import LiveCameras
    from ops.robot_pc.loop import PolicyLoop, summarize
    from ops.robot_pc.robot_state import StaticLatencyState
    from ops.robot_pc.safety import execution_eligible
    from ops.robot_pc.sinks import DryRunSink

    ap = argparse.ArgumentParser(prog="dry_run.py", description=__doc__.split("\n")[0])
    cli.add_common(ap)
    ap.add_argument("--state", choices=("live", "static"), required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--route", default="", help="label of the network route")
    ap.add_argument("--log", type=Path, required=True, help="DryRunSink JSONL (commands that were NOT sent)")
    ap.add_argument("--rows", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.steps is None and args.duration_s is None:
        ap.error("give --steps or --duration-s")
    cfg = cli.load_config(args)
    st = cfg.get("static_latency_state") or {}
    if args.state == "live":  # refused BEFORE any connection: no read-only gripper source exists in this revision
        print("dry run REFUSED: --state live has no read-only gripper register source (the teleop collector owns the "
              "Robotiq port); its observations could never be bound. Nothing was connected.", file=sys.stderr)
        return 3
    state = StaticLatencyState(st["pose7"], st["q7"], st["gripper_register"])
    pid, task, spec, info = cli.choose_policy(cfg, args)
    cams = selection.session_cameras(spec, args.camera)
    rig = cli.rig_document(cfg, task)
    corrections = selection.depth_corrections(spec, cams, rig, identity=args.identity_depth)
    cc = cfg.get("cameras") or {}
    cameras = LiveCameras(cams, native_hw=tuple(cc.get("native_hw", (480, 640))), fps=int(cc.get("fps", 15)),
                          depth="depth" in spec["modalities"], check_devices=bool(cc.get("check_devices", True)))
    client = sink = None
    rows_out = cli.JsonlRows(args.rows)
    try:
        K = cameras.start(corrections)
        native_hw = {c: tuple(cc.get("native_hw", (480, 640))) for c in cams}
        choices = selection.session_choices(spec, cams, native_hw=native_hw, K_native=K, rig_document=rig,
                                            identity_depth=args.identity_depth)
        client = cli.connect(cfg, args, pid, choices, "robot-pc-dry-run")
        live_m5 = C.m5(cfg, task, cams, float(client.spec["frame_period_s"])) if client.spec["views"] > 1 else None
        cameras.bind(K)
        sink = DryRunSink(args.log)
        loop = PolicyLoop(client, state, C.limits(cfg, task), sink, pad_run_start=not args.no_pad)
        rows = dryrun.run(client, loop, cameras, state, cams=cams, steps=args.steps, duration_s=args.duration_s,
                          live_m5=live_m5, on_row=rows_out)
        summary = {**cli.SAFETY, "policy": pid, "task": task, "cameras": cams, "state_source": state.kind,
                   "catalog_info": info, "execution_eligible": execution_eligible(info),
                   "state_label": state.label, "sink": "DryRunSink", "dry_run_log": str(args.log),
                   **summarize(rows, route=args.route, mode="live-dry-run"), "loop_counts": dict(loop.counts),
                   "camera_errors": dict(cameras.errors), "client_stats": dict(client.stats)}
        if live_m5 is not None:
            summary["m5"] = dict(live_m5.counts)
    finally:
        rows_out.close()
        cameras.close()
        if client is not None:
            client.close()
        if sink is not None:
            sink.close()
        state.close()
    cli.write_json(args.summary, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
