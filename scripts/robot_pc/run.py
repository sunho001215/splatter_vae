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
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli, selection  # noqa: E402
from ops.robot_pc import config as C  # noqa: E402


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None, *, operator_input=input) -> int:
    from ops.robot_pc import devices, live, tunnel
    from ops.robot_pc.cameras import LiveCameras
    from ops.robot_pc.executor import HighRateExecutor
    from ops.robot_pc.safety import ExecutionRefused, authorize_execution, execution_eligible
    from ops.robot_pc.sinks import DryRunSink
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
    ap.add_argument("--profile", choices=("slow", "none"), default=None,
                    help="tighten-only motion profile (ops/robot_pc/conservative.py); default: slow with --execute, "
                         "none for a dry run (pass --profile slow to rehearse the execution limits)")
    ap.add_argument("--fault-after-s", type=float, default=None,
                    help="DRY RUN ONLY: raise an injected error after N s (verifies error-terminated recording)")
    args = ap.parse_args(argv)
    if args.execute and args.gripper != "measured":
        ap.error("execution needs the measured gripper")
    if args.execute and args.fault_after_s is not None:
        ap.error("--fault-after-s is a dry-run test option")
    cfg = cli.load_config(args)
    profile = args.profile or ("slow" if args.execute else "none")
    if args.execute and profile != "slow":
        ap.error("execution runs with the conservative 'slow' profile (P3)")
    applied = {}
    if profile != "none":
        from ops.robot_pc import conservative

        cfg, applied = conservative.tighten(cfg, profile)
    root = Path(cfg["root"])
    opened = []  # cleanup stack
    failure = {"error": "launcher exited abnormally"}
    recorder = None
    recording_status = "error"
    cleaning = prompting = False
    stop = threading.Event()  # the same cancellation event during preflight, engagement and streaming

    def record_release(text):
        say(text)
        if recorder is not None:
            recorder.event(time.monotonic_ns(), text)

    shutdown = live.MotionShutdown(stop, report=say, on_release=record_release)
    session_box = {}
    previous_signals = {}

    class Cancelled(Exception):
        pass

    def check_cancel():
        if stop.is_set():
            raise Cancelled()

    def on_signal(signum, frame):
        if cleaning or shutdown._thread is not None:
            say("shutdown in progress")
            return
        stop.set()
        if prompting:
            raise Cancelled()
        if "s" in session_box:
            session_box["s"].stop("operator Ctrl-C (SIGINT)" if signum == signal.SIGINT else f"signal {signum}")
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    def keys():
        for line in sys.stdin:
            if line.strip().lower() in ("s", "q", "stop"):
                stop.set()
                if "s" in session_box:
                    session_box["s"].stop("operator key")
                return

    def cleanup():
        nonlocal cleaning
        cleaning = True
        shutdown.finish()  # executor join -> gripper clear -> controller release -> gripper stop/join
        for close in reversed(opened):
            try:
                close()
            except BaseException as exc:  # shutdown continues through every resource and recorder finalization
                say(f"cleanup: {type(exc).__name__}: {exc}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_signals[sig] = signal.signal(sig, on_signal)
        # ------------------------------------------------------------------ preflight
        port = int(args.port or (cfg.get("server") or {}).get("port", 47100))
        w = tunnel.welcome(port)  # (the tunnel's pid identity is not checkable under `sg dialout`: other gid)
        say(f"server route 127.0.0.1:{port}: {'WELCOME ' + str(w) if w else 'NO WELCOME (tunnel.py status / start)'}")
        if not w:
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
        check_cancel()
        if args.gripper == "measured":
            from ops.robot_pc.gripper import GripperWorker, RobotiqRTU

            from ops.robot_pc.conservative import GRIPPER_TEST

            gripper = GripperWorker(RobotiqRTU, read_only=not args.execute,
                                    **({"speed": GRIPPER_TEST["speed"], "force": GRIPPER_TEST["force"]}
                                       if profile == "slow" else {}))
            shutdown.gripper = gripper
            gripper.start()
            # 0x09 on the first reads of an idle bus can clear normally. Give readiness up to 1 s.
            deadline = time.monotonic() + 1.0
            while not stop.is_set() and time.monotonic() < deadline:
                if (gripper.healthy() if args.execute else gripper.latest is not None):
                    break
                stop.wait(0.02)
            check_cancel()
            if gripper.latest is None:
                say(f"gripper: no status ({gripper.error})")
                return 2
            if args.execute and not gripper.healthy():
                say(f"REFUSED: gripper is not healthy ({gripper.error})")
                return 3
            say(f"gripper: register {gripper.latest.position}, activated {gripper.latest.status == 3}, "
                f"fault {gripper.latest.fault}, ownership unverifiable={gripper.unverifiable}, "
                f"{'COMMANDS ENABLED' if args.execute else 'status reads only'}")
        else:
            gripper = live.StaticGripper()
            say("gripper: STATIC labelled register (dry run only)")
        state = live.LiveState(reader, gripper)
        shutdown.state = state
        pid, task, spec, info = cli.choose_policy(cfg, args)
        say(f"policy {pid} ({task}); execution_eligible={execution_eligible(info)}")
        if args.execute and not execution_eligible(info):
            say("REFUSED: the catalog does not mark this policy execution_eligible")
            return 3
        cams = selection.session_cameras(spec, args.camera)
        if spec.get("views", 1) > 1 or len(cams) != 1:
            raise ExecutionRefused("live multiview sync sets are not implemented; select a single-camera policy")
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
                    "profile": profile,
                    "cameras": cams, "route": args.route, "gripper_origin": getattr(gripper, "origin", None),
                    "config": cfg, "bundle": str(root / "BUNDLE.json")}
            recorder = SessionRecorder(root, meta=meta, cameras={c: {"native_hw": list(hw), "depth": False,
                                                                     "K_native": K[c].tolist(),
                                                                     "calibration": choices.get("calibration", {}).get(c)}
                                                                 for c in cams})
            def close_recorder():
                summary = recorder.close(recording_status, failure["error"])
                say(f"recorded: {summary.get('session_dir')}")
            opened.append(close_recorder)
            say(f"recording to {recorder.session_dir}")
        limits = C.limits(cfg, task)
        def start_limits():
            base = C.limits(cfg, task)
            if profile == "none":
                return base
            from ops.robot_pc import conservative

            start = state.latest()
            if start is None:
                raise ExecutionRefused("no robot state for the start box")
            boxed = conservative.test_box(base, start["flange_pose7"],
                                          conservative.PROFILES[profile]["test_radius_m"])
            say(f"profile {profile}: executor {applied}; goals within {conservative.PROFILES[profile]['test_radius_m']}"
                f" m of the start: box {[round(x, 3) for x in boxed.workspace_min_m]}.."
                f"{[round(x, 3) for x in boxed.workspace_max_m]}")
            if recorder is not None:
                recorder.event(time.monotonic_ns(), json.dumps({"profile": profile, "applied": applied,
                                                                "start_flange_pose7": list(start["flange_pose7"]),
                                                                "workspace_min_m": boxed.workspace_min_m,
                                                                "workspace_max_m": boxed.workspace_max_m}))
            return boxed
        if not args.execute:
            limits = start_limits()
        ex = cfg.get("executor") or {}
        executor = HighRateExecutor(_Discard(), **{k: ex[k] for k in ex})  # the session thread sends
        check_cancel()  # before engagement (including the operator prompt)
        if args.execute:
            prompting = True
            try:
                check_cancel()
                say("READY. Operator: hand on the e-stop. Press ENTER to START robot motion "
                    "(during the run: s + ENTER or Ctrl-C stops).")
                response = operator_input()
            finally:
                prompting = False
            if isinstance(response, str) and response.strip().lower() in ("s", "q", "stop"):
                stop.set()
            check_cancel()  # after ENTER
            blockers = state_blockers(list(reader.samples), time.monotonic_ns())
            if blockers or not live.still(state):
                raise ExecutionRefused("robot is not healthy and still at START: " + "; ".join(blockers))
            limits = start_limits()
            auth = authorize_execution(cli_execute=True, operator_start=True, policy_info=info)
            if auth.hardware_test is not None:
                raise ExecutionRefused("policy execution requires hardware_test is None")
            from ops.robot_pc.controller import ControllerLink

            shutdown.link = link = ControllerLink(auth)  # register release IMMEDIATELY upon construction
            threading.Thread(target=keys, daemon=True, name="stop-key").start()
            check_cancel()
            connected = link.wait_subscribers(3.0, cancel=stop)
            check_cancel()  # after wait_subscribers; never engage after cancellation
            if not connected:
                say("REFUSED: switch_controller did not connect to the command/action channels")
                return 3
            if not gripper.healthy():
                raise ExecutionRefused(f"gripper became unhealthy before engagement ({gripper.error})")
            blockers = state_blockers(list(reader.samples), time.monotonic_ns())
            if blockers or not live.still(state):
                raise ExecutionRefused("robot became unhealthy or moving before engagement: " + "; ".join(blockers))
            if recorder is not None:
                recorder.event(time.monotonic_ns(), "engage")
            link.engage(cancel=stop)
            check_cancel()  # after engage (also covers cancellation during settle)
            sink, sent = live.LiveSink(auth, link, gripper), True
            say(f"controller: mode 3 requested ({link.mode_requests_sent} sends); "
                "operator confirms acceptance on controller UI (not observable in RobotState)")
        else:
            threading.Thread(target=keys, daemon=True, name="stop-key").start()
            sink, sent = DryRunSink(), False
            say("DRY RUN: commands computed, validated and recorded, NOT sent")

        def on_status(row):
            _status(row)
            if t_fault is not None and time.monotonic() > t_fault:
                raise RuntimeError(f"injected fault after {args.fault_after_s} s (--fault-after-s, dry run)")

        session = live.LiveSession(client=client, cameras=cameras, cams=cams, state=state,
                                   limits=limits, executor=executor, sink=sink, recorder=recorder,
                                   sent=sent, on_status=on_status, stop_event=stop, shutdown=shutdown)
        session_box["s"] = session
        t_fault = None if args.fault_after_s is None else time.monotonic() + args.fault_after_s

        check_cancel()  # before streaming; run() checks again before starting its executor
        say("RUNNING (s + ENTER or Ctrl-C to stop)")
        rows = session.run(steps=args.steps, duration_s=args.duration_s)
        say(f"STOPPED: {session.stop_reason}; {len(rows)} steps, executor {executor.counts}")
        say(f"loop rates: {session.loop_stats()}")
        recording_status = "complete" if session.stop_reason in ("budget reached", "operator key",
                                                                 "operator Ctrl-C") else "stopped"
        failure["error"] = session.stop_reason
        return 0
    except Cancelled:
        failure["error"] = "cancelled during startup"
        say(failure["error"])
        return 130
    except ExecutionRefused as exc:
        say(f"REFUSED: {exc}")
        return 3
    except KeyboardInterrupt:
        stop.set()
        failure["error"] = "interrupted (Ctrl-C outside the session loop)"
        say("interrupted")
        return 130
    except Exception as exc:
        failure["error"] = f"{type(exc).__name__}: {exc}"
        say(f"ERROR: {failure['error']}")
        return 1
    finally:
        try:
            cleanup()
        finally:
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)


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
