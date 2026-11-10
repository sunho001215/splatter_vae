"""POLICY-FREE controlled hardware tests (Directive J, P3): gripper or zero-motion hold; ENTER starts real tests.

    venv/bin/python -I src/scripts/robot_pc/hwtest.py gripper --closures 0.3 0.0 --dwell-s 2
    venv/bin/python -I src/scripts/robot_pc/hwtest.py hold --duration-s 10

Use --rehearse for status reads and dry-run setpoints only. Stop: s + ENTER, Ctrl-C or SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli, conservative  # noqa: E402
from ops.robot_pc.hwtests import Cancelled, HardwareTest  # noqa: E402


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None, *, operator_input=input) -> int:
    from ops.robot_pc import devices, live
    from ops.robot_pc.cameras import LiveCameras
    from ops.robot_pc.gripper import GripperWorker, RobotiqRTU
    from ops.robot_pc.safety import ExecutionRefused, authorize_hardware_test
    from ops.robot_pc.switch_controller import LiveStateReader, state_blockers

    ap = argparse.ArgumentParser(prog="hwtest.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="test", required=True)
    for name in ("gripper", "hold"):
        p = sub.add_parser(name)
        p.add_argument("--rehearse", action="store_true")
        p.add_argument("--camera", default=None, help="optional scene camera serial (native RGB recording only)")
        p.add_argument("--no-record", action="store_true")
        p.add_argument("--config", type=Path, default=None)
        p.add_argument("--root", default=None)
        if name == "gripper":
            p.add_argument("--closures", type=float, nargs="+", default=[0.3, 0.0])
            p.add_argument("--dwell-s", type=float, default=2.0)
        else:
            p.add_argument("--duration-s", type=float, default=10.0)
            p.add_argument("--max-drift-mm", type=float, default=5.0)
            p.add_argument("--max-drift-deg", type=float, default=2.0)
            p.add_argument("--max-dq", type=float, default=0.05)
    args = ap.parse_args(argv)
    if args.test == "gripper":
        try:
            closures = conservative.gripper_closures(args.closures)
        except ValueError as exc:
            ap.error(str(exc))
        if not math.isfinite(args.dwell_s) or not 0.5 <= args.dwell_s <= 5:
            ap.error("--dwell-s must be in [0.5, 5]")
        parameters = {"closures": closures, "dwell_s": args.dwell_s}
    else:
        if not math.isfinite(args.duration_s) or not 1 <= args.duration_s <= 30:
            ap.error("--duration-s must be in [1, 30]")
        if any(not math.isfinite(v) or v <= 0 for v in (args.max_drift_mm, args.max_drift_deg, args.max_dq)):
            ap.error("drift and dq limits must be finite and positive")
        parameters = {"duration_s": args.duration_s, "max_drift_mm": args.max_drift_mm,
                      "max_drift_deg": args.max_drift_deg, "max_dq": args.max_dq}
    cfg, applied = conservative.tighten(cli.load_config(args), "slow")
    opened, previous_signals, session_box = [], {}, {}
    stop = threading.Event()  # one event throughout preflight, ENTER, engagement, streaming and shutdown
    shutdown = live.MotionShutdown(stop, report=say)
    session = None
    result = {"verdict": None, "error": "launcher exited abnormally"}

    def check_cancel():
        if stop.is_set():
            raise Cancelled()

    def on_signal(signum, frame):
        stop.set()
        if "s" in session_box:
            session_box["s"].abort("operator Ctrl-C (SIGINT)" if signum == signal.SIGINT else f"signal {signum}")
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    def keys():
        for line in sys.stdin:
            if line.strip().lower() in ("s", "q", "stop"):
                stop.set()
                if "s" in session_box:
                    session_box["s"].abort("operator key")
                return

    def close_recorder():
        verdict = result["verdict"] or {}
        recorder.close("complete" if verdict.get("pass") else "stopped",
                       None if verdict.get("pass") else
                       ((session.stop_reason if session is not None else result["error"]) or "test did not pass"))

    def cleanup():
        shutdown.finish()  # executor join -> gripper clear/stop -> fresh measured-ee hold / still wait / IDLE
        if session is not None:
            try:
                result["verdict"] = session.finalize()
                say("verdict " + json.dumps(result["verdict"], allow_nan=False))
            except Exception as exc:
                say(f"finalize: {type(exc).__name__}: {exc}")
        for close in reversed(opened):
            try:
                close()
            except Exception as exc:
                say(f"cleanup: {type(exc).__name__}: {exc}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_signals[sig] = signal.signal(sig, on_signal)
        check_cancel()
        devices.require_free(allow_controller=True)
        say("devices: free (switch_controller allowed)")
        if not devices.scan(ipc_paths=())["controllers"]:
            raise ExecutionRefused("switch_controller is not running")
        say("switch_controller: present")
        check_cancel()
        reader = LiveStateReader(approved=True)
        opened.append(reader.close)
        stop.wait(0.5)
        check_cancel()
        blockers = state_blockers(list(reader.samples), time.monotonic_ns())
        say(f"robot state: {len(reader.samples)} samples in 0.5 s, blockers {blockers or 'none'}")
        if blockers:
            raise ExecutionRefused("; ".join(blockers))
        gripper = GripperWorker(RobotiqRTU, read_only=args.rehearse or args.test != "gripper",
                                speed=conservative.GRIPPER_TEST["speed"], force=conservative.GRIPPER_TEST["force"])
        shutdown.gripper = gripper
        gripper.start()
        deadline = time.monotonic() + 1.0
        while not stop.is_set() and time.monotonic() < deadline:
            if gripper.healthy():  # initial idle-bus 0x09 may clear
                break
            stop.wait(0.02)
        check_cancel()
        if not gripper.healthy():
            raise ExecutionRefused(f"gripper is not healthy ({gripper.error})")
        say(f"gripper: register {gripper.latest.position}, fault {gripper.latest.fault}, "
            f"ownership unverifiable={gripper.unverifiable}, "
            f"{'status reads only' if gripper.read_only else 'small targets enabled after ENTER'}")
        state = live.LiveState(reader, gripper)
        shutdown.state = state
        cams, cameras, K = ([args.camera] if args.camera else []), None, {}
        cc = cfg.get("cameras") or {}
        hw = tuple(cc.get("native_hw", (480, 640)))
        check_cancel()
        if cams:
            cameras = LiveCameras(cams, native_hw=hw, fps=int(cc.get("fps", 15)), depth=False,
                                  keep_native=True, allow_controller=True)
            opened.append(cameras.close)
            K = cameras.start()
            cameras.bind(K)
            say(f"scene camera: {args.camera} (native RGB only)")
        recorder = None
        if not args.no_record:
            from ops.robot_pc.recorder import SessionRecorder

            meta = {"mode": f"hwtest-{args.test}" + ("-rehearsal" if args.rehearse else ""), "test": args.test,
                    "parameters": parameters, "config": cfg, "gripper_origin": gripper.origin, "cameras": cams}
            recorder = SessionRecorder(Path(cfg["root"]), meta=meta,
                                       cameras={c: {"native_hw": list(hw), "depth": False,
                                                    "K_native": K[c].tolist()} for c in cams})
            opened.append(close_recorder)
            say(f"recording to {recorder.session_dir}")
        session = HardwareTest(args.test, parameters=parameters, cfg=cfg, state=state, rehearse=args.rehearse,
                               stop_event=stop, shutdown=shutdown, recorder=recorder, cameras=cameras, cams=cams)
        say(f"profile slow: {applied}")
        session.prepare()
        check_cancel()
        if args.rehearse:
            say("REHEARSAL: status reads only; hold setpoints use a dry-run ideal follower")
            session.capture_start()
        else:
            if args.test == "gripper":
                from ops.robot_pc.gripper import closure_to_position

                say(f"ACTION: gripper closures {closures}, registers {[closure_to_position(c) for c in closures]}, "
                    f"dwell {args.dwell_s} s each, speed {gripper.speed}, force {gripper.force}; no arm commands")
            else:
                say(f"ACTION: request mode 3; hold the measured flange pose at ENTER for {args.duration_s} s "
                    f"through the executor; drift limits {args.max_drift_mm} mm / {args.max_drift_deg} deg, "
                    f"|dq| limit {args.max_dq} rad/s; release to measured-ee hold then IDLE")
            say("READY. Operator: hand on the e-stop. Press ENTER to START (s/q + ENTER cancels).")
            response = operator_input()
            if isinstance(response, str) and response.strip().lower() in ("s", "q", "stop"):
                stop.set()
            check_cancel()
            session.capture_start()  # snapshot at ENTER, before subscriber wait / engage settle
            auth = authorize_hardware_test(test=args.test, cli_flag=True, operator_start=True)
            check_cancel()
            threading.Thread(target=keys, daemon=True, name="stop-key").start()
            if args.test == "hold":
                from ops.robot_pc.controller import ControllerLink as Link
                from ops.robot_pc.live import LiveSink as HoldSink

                shutdown.link = link = Link(auth)  # register release IMMEDIATELY, including engage failures
                session.link = link
                check_cancel()
                connected = link.wait_subscribers(3.0, cancel=stop)
                check_cancel()
                if not connected:
                    raise ExecutionRefused("switch_controller did not connect to command/action channels")
                session._healthy_gripper()
                link.engage(cancel=stop)
                check_cancel()
                session.sink = HoldSink(auth, link, gripper=None)
                say(f"controller: mode 3 requested ({link.mode_requests_sent} sends); confirm acceptance on UI")
        session_box["s"] = session
        if args.rehearse:
            threading.Thread(target=keys, daemon=True, name="stop-key").start()
        check_cancel()
        say("RUNNING (s + ENTER or Ctrl-C to stop)")
        result["verdict"] = session.run()
        return 0 if result["verdict"]["pass"] else 1
    except Cancelled:
        result["error"] = "cancelled during startup"
        if session is not None:
            session.abort(result["error"])
        say(result["error"])
        return 130
    except ExecutionRefused as exc:
        if session is not None:
            session.abort(f"refused: {exc}")
        say(f"REFUSED: {exc}")
        return 3
    except KeyboardInterrupt:
        stop.set()
        if session is not None:
            session.abort("operator Ctrl-C")
        say("interrupted")
        return 130
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if session is not None:
            session.abort(result["error"])
        say(f"ERROR: {result['error']}")
        return 1
    finally:
        try:
            cleanup()
        finally:
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
