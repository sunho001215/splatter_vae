"""ROBOT PC inference: ONE entry point. Choose the policy by task / method / seed (or --policy), the cameras by NAME
(config/camera_names.yaml: left / middle / right), run it on the robot (default: starts on the operator's ENTER) or as
a --dry-run (commands computed and recorded, NOT sent). Every run is recorded to
inference_outputs/<task>/<method>/seed<N>/<cameras>/runNNN/ (inference.h5, <camera>.mp4, config.json, summary.json).

    venv/bin/python -I src/scripts/robot_pc/infer.py --list                                   # what can run
    sg dialout -c "venv/bin/python -I src/scripts/robot_pc/infer.py --task drawer-open --method e2e_cnn --seed 0 \
        --cameras right --duration-s 30"                                                     # on the robot
    sg dialout -c "venv/bin/python -I src/scripts/robot_pc/infer.py --task drawer-open --method reviwo --seed 1 --dry-run"

No workspace box, speed / acceleration cap or lead limit is applied (user directive K): the policy's goal is
interpolated at 500 Hz over one policy period and streamed; only setpoints switch_controller would reject (> 0.2 m /
0.5 rad from the measured pose) are not sent.

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
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli, selection  # noqa: E402
from ops.robot_pc import config as C  # noqa: E402


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None, *, operator_input=input) -> int:
    from ops.robot_pc import devices, live, tunnel
    from ops.robot_pc.camera_warmup import WarmupConfig, wait_until_stable
    from ops.robot_pc.cameras import LiveCameras
    from ops.robot_pc.executor import HighRateExecutor
    from ops.robot_pc.safety import ExecutionRefused, authorize_execution, execution_eligible
    from ops.robot_pc.sinks import DryRunSink
    from ops.robot_pc.switch_controller import LiveStateReader, state_blockers

    ap = argparse.ArgumentParser(prog="infer.py", description=__doc__.split("\n")[0])
    cli.add_common(ap)
    ap.add_argument("--method", "--baseline", dest="condition", default=None,
                    help="method / baseline name of the catalog (reviwo, e2e_cnn, sincro, dynamo, ...); with --task "
                         "and --seed it selects the policy (or give --policy <id>)")
    ap.add_argument("--cameras", default=None,
                    help="camera NAMES, comma separated (config/camera_names.yaml), e.g. right or left,middle,right; "
                         "default: what the policy contract requires (its first trained camera)")
    ap.add_argument("--dry-run", action="store_true", help="compute, validate and record commands; send NOTHING")
    ap.add_argument("--list", action="store_true", help="list tasks / methods / seeds and execution eligibility")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--gripper", choices=("measured", "static"), default="measured",
                    help="static = a labelled fixed register, DRY RUN only (no gripper access)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--camera-warmup-timeout-s", type=float, default=None,
                    help="colour stability timeout in seconds (0 disables warmup; recorded in run config)")
    ap.add_argument("--route", default="direct robotpc->5090 ssh -L")
    ap.add_argument("--fault-after-s", type=float, default=None,
                    help="DRY RUN ONLY: raise an injected error after N s (verifies error-terminated recording)")
    args = ap.parse_args(argv)
    args.execute = not args.dry_run
    if args.camera is not None and args.cameras is not None:
        ap.error("give camera names with --cameras (or one serial with --camera), not both")
    if args.execute and args.gripper != "measured":
        ap.error("execution needs the measured gripper")
    if args.execute and args.fault_after_s is not None:
        ap.error("--fault-after-s is a dry-run test option")
    cfg = cli.load_config(args)
    warmup_options = dict(cfg.get("camera_warmup") or {})
    if args.camera_warmup_timeout_s is not None:
        warmup_options["timeout_s"] = args.camera_warmup_timeout_s
    try:
        warmup_cfg = WarmupConfig.from_config(warmup_options)
    except ValueError as exc:
        ap.error(str(exc))
    cfg["camera_warmup"] = asdict(warmup_cfg)
    if args.list:
        return list_policies(cfg, args)
    names = camera_names(cfg)
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

            gcfg = cfg.get("gripper") or {}
            gripper = GripperWorker(RobotiqRTU, read_only=not args.execute, speed=int(gcfg.get("speed", 146)),
                                    force=int(gcfg.get("force", 255)))
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
        for line in restriction_notes(info):
            say(line)
        if args.execute and not execution_eligible(info):
            say("REFUSED: the catalog does not mark this policy execution_eligible")
            return 3
        cams = resolve_cameras(spec, names, args.cameras, args.camera)
        cam_label = "-".join(_name(names, c) for c in cams)
        say(f"cameras: {', '.join(f'{_name(names, c)}={c}' for c in cams)}")
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
        client = cli.connect(cfg, args, pid, choices, "robot-pc-infer")
        opened.append(client.close)
        selected = info
        # the BOUND policy's current labels (SPEC), never only the earlier WELCOME snapshot (Codex review of 1847fdd)
        info = confirm_policy(selected, getattr(client, "spec_info", None), execute=args.execute, stage="binding")
        if any(info.get(k) != selected.get(k) for k in POLICY_LABELS):
            say("catalog labels at binding (SPEC) differ from the selection (WELCOME); current labels:")
            for line in restriction_notes(info):
                say(line)
        cameras.bind(K)
        recorder = None
        if not args.no_record:
            from ops.robot_pc.recorder import SessionRecorder

            method, seed = info.get("condition") or "policy", info.get("seed")
            if info.get("actor", "onestep") != "onestep":  # EXPERIMENTAL variants never share a folder with FQL runs
                method = f"{method}+{info['actor'].replace('_', '')}"
            experiment = {"policy": pid, "task": task, "method": method, "seed": seed,
                          "mode": "execute" if args.execute else "dry_run", "duration_s": args.duration_s,
                          "steps": args.steps, "cameras": [{"name": _name(names, c), "serial": c} for c in cams],
                          "camera_names_file": str(C.path(cfg, (cfg.get("cameras") or {}).get("names_file"))),
                          "camera_names": names, "catalog_info": info, "catalog_info_selected": selected,
                          "input_spec": spec, "route": args.route,
                          "gripper_origin": getattr(gripper, "origin", None),
                          "gripper": {"speed": getattr(gripper, "speed", None), "force": getattr(gripper, "force", None)},
                          "config": cfg, "bundle": str(running_bundle(root)), "argv": list(argv or sys.argv[1:])}
            prefix = f"{task}/{method}/seed{seed if seed is not None else 'x'}/{cam_label}"
            recorder = SessionRecorder(root, meta=experiment, run_prefix=prefix,
                                       camera_names={c: _name(names, c) for c in cams},
                                       files={"config.json": experiment},
                                       cameras={c: {"native_hw": list(hw), "depth": "depth" in spec["modalities"],
                                                    "K_native": K[c].tolist(),
                                                    "calibration": choices.get("calibration", {}).get(c)}
                                                for c in cams})
            def close_recorder():
                summary = recorder.close(recording_status, failure["error"])
                say(f"recorded: {summary.get('session_dir')}")
            opened.append(close_recorder)
            say(f"recording to {recorder.session_dir}")
        warmup = wait_until_stable(cameras, cams, warmup_cfg, cancel=stop, fps=int(cc.get("fps", 15)))
        say(f"camera warmup: {warmup['status']} after {warmup['waited_s']:.2f} s; "
            f"cameras {json.dumps(warmup['cameras'])}")
        if recorder is not None:
            recorder.event(time.monotonic_ns(), "camera_warmup " + json.dumps(warmup))
        if warmup["status"] == "cancelled":
            raise Cancelled()
        if warmup["status"] == "no_frames" or (warmup["status"] == "timeout" and warmup_cfg.on_timeout == "refuse"):
            failure["error"] = f"camera warmup {warmup['status']}: {warmup.get('reason', 'colour did not settle')}"
            raise ExecutionRefused(failure["error"])
        if warmup["status"] == "timeout":
            say("WARNING: camera warmup timed out; proceeding (camera_warmup.on_timeout=proceed)")
        limits = C.limits(cfg, task)
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
            if blockers:
                raise ExecutionRefused("robot state is not healthy at START: " + "; ".join(blockers))
            # a grant revoked / a checkpoint re-staged since the selection refuses here, before any motion
            info = confirm_policy(info, current_catalog_info(cfg, args, pid), execute=True, stage="START re-check")
            if recorder is not None:  # the full label snapshot execution was authorized on
                recorder.event(time.monotonic_ns(), "start_recheck " + json.dumps(info, default=str))
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
            if blockers:
                raise ExecutionRefused("robot state became unhealthy before engagement: " + "; ".join(blockers))
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
                                                                 "operator Ctrl-C", "operator Ctrl-C (SIGINT)") \
            else "stopped"
        failure["error"] = session.stop_reason
        rel = shutdown.release_outcome if sent else None
        if sent and not (isinstance(rel, dict) and rel.get("hold") == "sent" and rel.get("idle") == "sent"):
            recording_status = "error"  # an engaged run whose release did not hold + IDLE is never "complete"
            failure["error"] = f"{session.stop_reason}; release incomplete: {rel}"
            say(f"WARNING: release incomplete: {rel} (operator: check the controller UI / e-stop)")
            return 4
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


def camera_names(cfg: dict) -> dict:
    """{name: serial} from the mapping file (config/camera_names.yaml; the repo template if the copy is missing)."""
    import yaml

    p = C.path(cfg, (cfg.get("cameras") or {}).get("names_file") or "config/camera_names.yaml")
    if p is None or not p.is_file():
        p = Path(__file__).resolve().parents[2] / "configs/robot_pc/camera_names.yaml"
    m = {str(k): str(v) for k, v in (yaml.safe_load(p.read_text()) or {}).items()}
    bad = [k for k in m if not k or not all(ch.isalnum() or ch == "_" for ch in k)]
    if bad or len(set(m.values())) != len(m):
        raise SystemExit(f"{p}: camera names must be [a-z0-9_] and every serial used once ({m})")
    return m


def _name(names: dict, serial: str) -> str:
    inv = {v: k for k, v in names.items()}
    return inv.get(serial, serial)


def resolve_cameras(spec: dict, names: dict, requested: str | None, serial: str | None = None) -> list:
    """Requested camera names -> serials, checked against the policy contract (count and trained cameras)."""
    from ops.robot_pc.safety import ExecutionRefused

    orders = spec["trained_camera_orders"]
    trained = sorted({c for o in orders for c in o})
    tnames = ",".join(_name(names, c) for c in trained)
    if requested is not None:
        want = [x.strip() for x in requested.split(",") if x.strip()]
        unknown = [w for w in want if w not in names]
        if unknown:
            raise ExecutionRefused(f"unknown camera name(s) {unknown}; known: {sorted(names)} (config/camera_names.yaml)")
        cams = [names[w] for w in want]
    elif serial is not None:
        cams = [serial]
    else:
        cams = list(orders[0]) if spec["views"] > 1 else [orders[0][0]]
    if spec["views"] == 1:
        if len(cams) != 1:
            raise ExecutionRefused(f"this policy is single-view: give exactly ONE camera (trained: {tnames})")
        if cams[0] not in trained:
            raise ExecutionRefused(f"camera {_name(names, cams[0])} is not a trained camera of this policy ({tnames})")
        return cams
    if [list(o) for o in orders].count(cams) == 0:
        need = " or ".join(",".join(_name(names, c) for c in o) for o in orders)
        raise ExecutionRefused(f"this policy needs {spec['views']} cameras in training order: {need}")
    raise ExecutionRefused("live multi-view (synchronised camera sets) is not implemented yet: select a single-view "
                           "policy (infer.py --list shows the views)")


def restriction_notes(info: dict) -> list:
    """Console lines for a ``restricted`` policy (a criterion FAILED, not waived): its reasons, usage notes and, when
    the catalog marks it execution-eligible, the user decision that acknowledges the failure (recorded with the
    session's catalog_info as well). [] for every other status."""
    if (info or {}).get("status") != "restricted":
        return []
    lines = [f"RESTRICTED policy (criterion FAILED, not waived): {r}" for r in info.get("reasons", [])]
    lines += [f"note: {r}" for r in info.get("usage_reasons", []) if r not in info.get("reasons", [])]
    d = info.get("execution_decision") or {}
    if info.get("execution_eligible") is True and d:
        lines.append(f"execution allowed by the user decision of {d.get('at')}: acknowledged FAILED criteria "
                     f"{d.get('acknowledged_failures')} (checkpoint {str(d.get('checkpoint_sha256'))[:12]})")
    return lines


POLICY_IDENTITY = ("checkpoint_sha256", "condition", "task", "seed", "run", "actor")
# for execution, every label the operator was shown (or that gates execution) must be unchanged since the selection
POLICY_LABELS = ("status", "usage", "execution_eligible", "execution_decision", "reasons", "usage_reasons",
                 "criteria", "requires_client")


def confirm_policy(selected: dict, current, *, execute: bool, stage: str) -> dict:
    """The CURRENT catalog labels of the selected policy (the SPEC of the bound session; the server's WELCOME again at
    START), checked against the selection: the same weights and identity (``POLICY_IDENTITY``); for execution a
    published checkpoint sha256, unchanged eligibility labels (``POLICY_LABELS``: a grant revoked or changed since the
    selection refuses - run again to see the current notes) and ``execution_eligible`` true. Returns the current info;
    raises ExecutionRefused otherwise (fail closed: missing info refuses execution)."""
    from ops.policy_serving.config import valid_sha256
    from ops.robot_pc.safety import ExecutionRefused, execution_eligible

    if not isinstance(current, dict) or not current:
        if execute:
            raise ExecutionRefused(f"{stage}: the server published no catalog info for the policy; cannot confirm "
                                   "the selected policy")
        return dict(selected)
    diff = {k: [selected.get(k), current.get(k)] for k in POLICY_IDENTITY if selected.get(k) != current.get(k)}
    if diff:
        raise ExecutionRefused(f"{stage}: the policy differs from the one selected: {diff}")
    if execute:
        if not valid_sha256(current.get("checkpoint_sha256")):
            raise ExecutionRefused(f"{stage}: the policy's checkpoint sha256 is not published")
        changed = sorted(k for k in POLICY_LABELS if selected.get(k) != current.get(k))
        if changed:  # never execute on labels (warnings, criteria, requirements) the operator was not shown
            raise ExecutionRefused(f"{stage}: the catalog labels {changed} changed since the selection; run again "
                                   "to see the current notes")
        from ops.policy_serving.client import CAPABILITIES

        lacking = sorted(set(current.get("requires_client") or []) - CAPABILITIES)
        if lacking:
            raise ExecutionRefused(f"{stage}: the policy requires client capabilities {lacking} this client lacks")
        if not execution_eligible(current):
            raise ExecutionRefused(f"{stage}: the catalog does not mark this policy execution_eligible")
    return dict(current)


def current_catalog_info(cfg: dict, args, pid: str):
    """The policy's labels in a fresh WELCOME (no session is opened); None when the server does not offer it."""
    from ops.robot_pc.safety import ExecutionRefused

    try:
        offered = cli.welcome(cfg, args).get("policies") or {}
    except Exception as exc:  # noqa: BLE001 - an unreachable server refuses the start (fail closed)
        raise ExecutionRefused(f"START re-check: no WELCOME from the policy server ({type(exc).__name__}: {exc})") \
            from None
    return (offered.get(pid) or {}).get("info")


def running_bundle(root: Path) -> Path:
    """BUNDLE.json of the release this code runs from (``releases/<name>/src`` -> ``releases/<name>/BUNDLE.json``;
    a release run side by side records its own bundle), else the workspace's."""
    own = Path(__file__).resolve().parents[2].parent / "BUNDLE.json"
    return own if own.is_file() else root / "BUNDLE.json"


def list_policies(cfg: dict, args) -> int:
    from ops.robot_pc.safety import execution_eligible

    offered = cli.welcome(cfg, args)["policies"]
    names = camera_names(cfg)
    rows, aliases = [], 0
    for pid, p in sorted(offered.items()):
        info, spec = p.get("info") or {}, p.get("input_spec") or {}
        if info.get("alias_of"):  # legacy alias of a canonical <method>-<task>-s<seed> id (same checkpoint)
            aliases += 1
            continue
        cams = sorted({c for o in spec.get("trained_camera_orders", []) for c in o})
        rows.append((info.get("task", "?"), info.get("condition", "?"), str(info.get("seed", "?")), pid,
                     "yes" if execution_eligible(info) else "no", info.get("status", "?"), str(spec.get("views", "?")),
                     "+".join(spec.get("modalities", [])), ",".join(_name(names, c) for c in cams)))
    head = ("task", "method", "seed", "policy id", "eligible", "status", "views", "inputs", "cameras")
    w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(x.ljust(w[i]) for i, x in enumerate(r)))
    print(f"\ncamera names: {names} (config/camera_names.yaml); 'eligible' = may run on the robot (catalog); "
          f"{aliases} legacy alias ids not shown")
    flagged = sorted(pid for pid, p in offered.items() if not (p.get("info") or {}).get("alias_of")
                     and (p.get("info") or {}).get("status") == "restricted"
                     and execution_eligible(p.get("info") or {}))
    if flagged:
        print("\nrestricted but eligible (a criterion FAILED, not waived; executable by user decision; NOT a "
              "validated-performance claim):")
        for pid in flagged:
            info = offered[pid].get("info") or {}
            failed = sorted(k for k, v in (info.get("criteria") or {}).items() if isinstance(v, dict)
                            and v.get("pass") is False)
            d = info.get("execution_decision") or {}
            print(f"  {pid}: FAILED {failed}; decision {d.get('at')} (checkpoint "
                  f"{str(d.get('checkpoint_sha256'))[:12]}); requires client {info.get('requires_client')}")
            for r in info.get("usage_reasons", []):
                if r.startswith("WARNING"):
                    print(f"    {r}")
    return 0


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
