"""ROBOT PC launcher: live inference with switch_controller, the measured Robotiq gripper, the cameras and the 5090 policy
server; every session recorded under inference_outputs/. DRY RUN by default (commands computed, validated, recorded,
NOT sent); ``--execute`` arms the execution mode, which starts only when the operator presses ENTER.

    venv/bin/python -I src/scripts/robot_pc/run.py --policy reviwo-drawer-open-s0 --duration-s 20          # dry run
    sg dialout -c "venv/bin/python -I src/scripts/robot_pc/run.py --policy reviwo-drawer-open-s0 --execute"  # live

Stop at any time: Ctrl-C, ``s`` + ENTER, or SIGTERM (the robot then holds the measured pose and switch_controller is
asked for IDLE); the hardware e-stop stays the operator's. Preflight (console): tunnel / server WELCOME, no teleop
collector or recorder running, switch_controller state live and healthy, gripper single owner (measured), cameras free,
policy (execution: catalog ``execution_eligible`` only), controller subscribers connected (execution).
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc import config as C  # noqa: E402
from ops.robot_pc import selection  # noqa: E402


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None, *, operator_input=input) -> int:
    from ops.robot_pc import devices, live, tunnel
    from ops.robot_pc.cameras import LiveCameras
    from ops.robot_pc.executor import HighRateExecutor
    from ops.robot_pc.safety import ExecutionRefused, authorize_execution, execution_eligible
    from ops.robot_pc.sinks import DryRunSink, LiveSink
    from ops.robot_pc.switch_controller import LiveStateReader, state_blockers

    ap = argparse.ArgumentParser(prog="run.py", description=__doc__.split("\n")[0])
    cli.add_common(ap)
    ap.add_argument("--execute", action="store_true", help="arm the execution mode (starts on the operator's ENTER)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--gripper", choices=("measured", "static"), default="measured",
                    help="static = a labelled fixed register, DRY RUN only (no gripper access)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--route", default="direct robotpc->5090 ssh -L")
    args = ap.parse_args(argv)
    if args.execute and args.gripper != "measured":
        ap.error("execution needs the measured gripper")
    cfg = cli.load_config(args)
    root = Path(cfg["root"])
    opened = []  # cleanup stack

    def cleanup():
        for close in reversed(opened):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - shutdown continues
                say(f"cleanup: {type(exc).__name__}: {exc}")

    try:
        # ------------------------------------------------------------------ preflight
        st = tunnel.status(root, int(args.port or (cfg.get("server") or {}).get("port", 47100)))
        say(f"server route: {'OK' if st['welcome'] else 'NO WELCOME'} {st}")
        if not st["welcome"]:
            return 2
        devices.require_free(allow_controller=True)  # no teleop collector / recorder / camera holder
        if not devices.scan(ipc_paths=())["controllers"]:
            say("switch_controller is not running")
            return 2
        reader = LiveStateReader(approved=True)
        opened.append(reader.close)
        time.sleep(0.5)
        blockers = state_blockers(list(reader.samples), time.monotonic_ns())
        say(f"robot state: {len(reader.samples)} samples in 0.5 s, blockers {blockers or 'none'}")
        if blockers:
            return 2
        if args.gripper == "measured":
            from ops.robot_pc.gripper import GripperWorker, RobotiqRTU

            gripper = GripperWorker(RobotiqRTU, read_only=not args.execute)
            gripper.start()
            opened.append(gripper.stop)
            time.sleep(0.3)
            if gripper.latest is None:
                say(f"gripper: no status ({gripper.error})")
                return 2
            say(f"gripper: register {gripper.latest.position}, activated {gripper.latest.status == 3}, "
                f"fault {gripper.latest.fault}, {'COMMANDS ENABLED' if args.execute else 'status reads only'}")
        else:
            gripper = live.StaticGripper()
            say("gripper: STATIC labelled register (dry run only)")
        state = live.LiveState(reader, gripper)
        pid, task, spec, info = cli.choose_policy(cfg, args)
        say(f"policy {pid} ({task}); execution_eligible={execution_eligible(info)}")
        if args.execute and not execution_eligible(info):
            say("REFUSED: the catalog does not mark this policy execution_eligible")
            return 3
        cams = selection.session_cameras(spec, args.camera)
        cc = cfg.get("cameras") or {}
        hw = tuple(cc.get("native_hw", (480, 640)))
        rig = cli.rig_document(cfg, task)
        corrections = selection.depth_corrections(spec, cams, rig, identity=args.identity_depth)
        cameras = LiveCameras(cams, native_hw=hw, fps=int(cc.get("fps", 15)), depth="depth" in spec["modalities"],
                              keep_native=not args.no_record, allow_controller=True)
        opened.append(cameras.close)
        K = cameras.start(corrections)
        choices = selection.session_choices(spec, cams, native_hw={c: hw for c in cams}, K_native=K, rig_document=rig,
                                            identity_depth=args.identity_depth)
        client = cli.connect(cfg, args, pid, choices, "robot-pc-run")
        opened.append(client.close)
        cameras.bind(K)
        recorder = None
        if not args.no_record:
            from ops.robot_pc.recorder import SessionRecorder

            meta = {"policy": pid, "task": task, "catalog_info": info, "mode": "execute" if args.execute else "dry_run",
                    "cameras": cams, "route": args.route, "gripper_origin": getattr(gripper, "origin", None),
                    "config": cfg, "bundle": str(root / "BUNDLE.json")}
            recorder = SessionRecorder(root, meta=meta, cameras={c: {"native_hw": list(hw), "depth": False,
                                                                     "K_native": K[c].tolist(),
                                                                     "calibration": choices.get("calibration", {}).get(c)}
                                                                 for c in cams})
            opened.append(lambda: recorder.close("error", "launcher exited abnormally"))  # no-op after close()
            say(f"recording to {recorder.session_dir}")
        ex = cfg.get("executor") or {}
        executor = HighRateExecutor(_Discard(), **{k: ex[k] for k in ex})  # the session thread sends
        stop = threading.Event()
        session_box = {}

        def on_signal(signum, frame):
            stop.set()
            if "s" in session_box:
                session_box["s"].stop(f"signal {signum}")

        signal.signal(signal.SIGTERM, on_signal)
        if args.execute:
            say("READY. Operator: hand on the e-stop. Press ENTER to START robot motion "
                "(during the run: s + ENTER or Ctrl-C stops).")
            operator_input()
            auth = authorize_execution(cli_execute=True, operator_start=True, policy_info=info)
            from ops.robot_pc.controller import ControllerLink

            link = ControllerLink(auth)
            if not link.wait_subscribers(3.0):
                link.close()
                say("REFUSED: switch_controller did not connect to the command/action channels")
                return 3
            link.engage()
            opened.append(lambda: link.release(state.latest()["flange_pose7"] if state.latest() else None,
                                               still=lambda: live.still(state)))
            sink, sent = LiveSink(auth, link, gripper), True
            say("controller: mode 3 requested; STREAMING")
        else:
            sink, sent = DryRunSink(), False
            say("DRY RUN: commands computed, validated and recorded, NOT sent")
        session = live.LiveSession(client=client, cameras=cameras, cams=cams, state=state,
                                   limits=C.limits(cfg, task), executor=executor, sink=sink, recorder=recorder,
                                   sent=sent, on_status=lambda r: _status(r))
        session_box["s"] = session

        def keys():
            for line in sys.stdin:
                if line.strip().lower() in ("s", "q", "stop"):
                    session.stop("operator key")
                    return

        threading.Thread(target=keys, daemon=True, name="stop-key").start()
        say("RUNNING (s + ENTER or Ctrl-C to stop)")
        rows = session.run(steps=args.steps, duration_s=args.duration_s)
        say(f"STOPPED: {session.stop_reason}; {len(rows)} steps, executor {executor.counts}")
        if recorder is not None:
            summary = recorder.close("complete" if session.stop_reason in ("budget reached", "operator key",
                                                                             "operator Ctrl-C") else "stopped",
                                     session.stop_reason)
            say(f"recorded: {summary.get('session_dir')}")
        return 0
    except ExecutionRefused as exc:
        say(f"REFUSED: {exc}")
        return 3
    except KeyboardInterrupt:
        say("interrupted")
        return 130
    finally:
        cleanup()


class _Discard:
    """The executor sends nothing itself; the session's 500 Hz thread hands its setpoints to the real sink."""

    def send(self, item) -> None:
        pass


def _status(row: dict) -> None:
    c = row.get("client_ms") or {}
    cmd = "cmd" if row.get("command") else (row.get("rejected") or row.get("code") or row.get("status"))
    if c:
        print(f"  step {row['reference_id']}: {row['status']} {cmd} e2e {c.get('e2e_ms', 0):.1f} ms "
              f"capture->cmd {c.get('capture_to_command_ms', 0):.1f} ms", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
