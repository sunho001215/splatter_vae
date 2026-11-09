"""DRY-RUN recorded-episode demo of network policy serving. NEVER sends anything to a robot.

Recorded ``panda_teleop_episode`` H5 (read-only) -> the robot-side client exactly as live (training crop + 128 px
resize on this CPU; for 3D-MVP the recorded native z16 depth aligned to colour by ``--alignment``: ``rs_align``
(default, the validated emulation of the live rs.align + colour-z path) or ``sensor_depth`` (the release stage
kernel), each frame labelled with the alignment actually used; the rig calibration loaded once) -> policy server ->
actions JSONL, compared with the recorded commands (``actions.step_table``)
and, with ``--direct-checkpoint``, with ``DeployPolicy.infer`` run here on the identical frames, binding and FQL noise
(validation mode: ``--noise-seed``; per step seed = base + step).

    .venv/bin/python scripts/policy_replay_demo.py --episode /home/ws/data/real_world/drawer-open/episode_000000.h5 \\
        --policy dynamo-drawer-open --host 127.0.0.1 --port 47100 --steps 60 --noise-seed 0 --out actions.jsonl
    # 3D-MVP: + --rig-document realworld-ops/calib-loma/rel/rig/rig-loma-drawer-open.json
    # multi-view: + --m5-store <released training-store episode.h5 of this recording> (M5 set evidence)
    # direct comparison: + --direct-checkpoint runs/<run>/policy.pt [--direct-export ...] --direct-training <sidecar|.h5>

Science comes from ``s4d.rwlearn.deploy`` (``replay.RecordedEpisode``: decoding, depth alignment, state binding;
``realsense.load_rig_session``); networking from ``ops.policy_serving``. Output: JSONL rows + a JSON summary on stdout.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ops.policy_serving import protocol as P  # noqa: E402
from ops.policy_serving.bench import _percentiles  # noqa: E402
from ops.policy_serving.client import PolicyClient  # noqa: E402


def session_plan(rec, spec: dict, camera: str | None, rig_document: Path | None) -> tuple[list, dict]:
    """(session cameras, HELLO choices) for the recording; calibration / native correction from the rig document."""
    from s4d.rwlearn.deploy.realsense import load_rig_session, native_correction

    orders = spec["trained_camera_orders"]
    cams = list(orders[0]) if spec["views"] > 1 else [camera or orders[0][0]]
    missing = [c for c in cams if c not in rec.cameras()]
    if missing:
        raise SystemExit(f"cameras {missing} not in the recording {rec.cameras()}")
    choices = {"native_hw": {c: list(rec.native_hw(c)) for c in cams}}
    if spec["views"] == 1:
        choices["camera_id"] = cams[0]
    depth = "depth" in spec["modalities"]
    if spec["needs_calibration"] or depth:
        if rig_document is None and spec["needs_calibration"]:
            raise SystemExit("this condition needs the static rig calibration: --rig-document")
        rig = load_rig_session(rig_document, cams, {c: rec.K_native(c) for c in cams}) if rig_document else None
        if spec["needs_calibration"]:
            choices["calibration"] = rig["calibration"]
        if depth:
            choices["native_depth_correction"] = rig["native_depth_correction"] if rig else \
                {c: native_correction() for c in cams}
    return cams, P.jsonable(choices)


def timeline(rec, cams: list, spec: dict, m5_store: Path | None) -> tuple[list, dict | None]:
    """Steps [{camera: stream index}] in recording order (+ M5 evidence {reference seq: metres} for multi-view)."""
    if spec["views"] == 1:
        return [{cams[0]: i} for i in range(len(rec.timestamps_ns(cams[0])))], None
    if m5_store is None:
        raise SystemExit("multi-view replay needs the recording's M5 set evidence: --m5-store")
    from s4d.rwlearn.deploy.replay import m5_from_store

    m5 = m5_from_store(m5_store, rec)
    if list(m5["cameras"]) != list(cams):
        raise SystemExit(f"store camera order {m5['cameras']} != trained order {cams}")
    return m5["steps"], m5["disp"]


def recorded_commands(rec, cams: list, steps: list, spec: dict) -> dict:
    """The recorded command of each step (training contract): action_phys / transition_ok on the step times."""
    from s4d.rwlearn import actions as A

    raw = rec.raw
    times = np.array([max(int(rec.timestamps_ns(c)[s[c]]) for c in cams) for s in steps], np.int64)
    ref = cams[0]
    seg = np.asarray(raw.cameras[ref]["segment"], np.int64)[raw.cameras[ref]["paired_colour"]]
    segment = np.array([seg[s[ref]] for s in steps], np.int64)
    c = spec["action"]
    tab = A.step_table(raw, times, segment, nominal_period_s=c["nominal_period_s"],
                       max_step_factor=c["max_step_s"] / c["nominal_period_s"])
    return {"ok": tab.transition_ok, "action_phys": tab.action_phys}


def depth_metadata(native: dict) -> dict:
    """Frame provenance of the recorded native depth, labelled from the alignment ``RecordedEpisode.native`` used."""
    if "depth" not in native:
        return {}
    return {"alignment": f"{native['alignment']} (recorded replay)", "depth_timestamp_ns": native["depth_timestamp_ns"]}


def first_request(rows: list) -> int | None:
    """Bytes of the first request actually sent (rows skipped before a request carry none)."""
    return next((r["request_bytes"] for r in rows if r["status"] != "skipped"), None)


def compare_recorded(out: dict, phys: np.ndarray) -> dict:
    from s4d.rwlearn import actions as A

    Rp = A.apply_delta(np.eye(4), np.asarray(out["delta_phys"], np.float64))[:3, :3]
    Rr = A.apply_delta(np.eye(4), phys[:6])[:3, :3]
    return {"position_error_mm": 1e3 * float(np.linalg.norm(np.asarray(out["delta_phys"])[:3] - phys[:3])),
            "rotation_error_deg": float(np.degrees(np.arccos(np.clip((np.trace(Rp.T @ Rr) - 1) / 2, -1, 1)))),
            "gripper_abs_error": abs(float(out["gripper_closure"]) - float(phys[6]))}


def same_outputs(net: dict, ref: dict) -> list:
    """Keys whose network value differs from the direct value (timings and the session id excluded)."""
    bad = []
    for key, want in ref.items():
        got = net.get(key)
        if key == "timings":
            continue
        if key in ("obs_ids", "metadata"):
            same = {k: v for k, v in got.items() if k != "session_id"} == P.jsonable(
                {k: v for k, v in want.items() if k != "session_id"})
        elif isinstance(want, (np.ndarray, np.generic)):
            same = np.asarray(got).dtype == np.asarray(want).dtype and np.array_equal(got, want)
        else:
            same = P.jsonable(got) == P.jsonable(want)
        if not same:
            bad.append(key)
    return bad


class Direct:
    """The same requests through ``DeployPolicy`` in this process, for the equality check (bit equality needs the
    server's device: run the demo next to a GPU server with ``--direct-device cuda``)."""

    def __init__(self, checkpoint, export, training, choices, device="cpu"):
        from s4d.rwlearn.deploy import DeployPolicy, HistoryWindow

        self.policy = DeployPolicy(checkpoint, device=device, export_path=export, training=training)
        self.spec = self.policy.reset_session(session_id=f"direct-{time.monotonic_ns()}", **choices)
        self.history = HistoryWindow(self.spec)

    def push(self, cam, frame):
        self.history.push(cam, frame)

    def infer(self, end_ids, disp, pad, ref, state, noise):
        from ops.policy_serving.server import classify
        from s4d.rwlearn.deploy import DeployRefusal

        try:
            w = self.history.assemble(end_ids, surface_disp_m=disp, pad_run_start=pad)
            ids = w.binding(ref, pose_time_ns=state["pose_time_ns"], gripper_time_ns=state["gripper_time_ns"])
            return self.policy.infer(w, state["pose7"], state["gripper_register"], ids, noise=noise), None
        except DeployRefusal as exc:
            return None, classify(exc)


def main(argv=None) -> int:
    from s4d.rwlearn.deploy import Frame
    from s4d.rwlearn.deploy.replay import RecordedEpisode

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--camera", default=None, help="single-view camera serial (default: first trained camera)")
    ap.add_argument("--rig-document", type=Path, default=None)
    ap.add_argument("--alignment", default="rs_align", choices=("rs_align", "sensor_depth"),
                    help="recorded native depth -> colour grid: rs_align (live-path emulation) or sensor_depth")
    ap.add_argument("--m5-store", type=Path, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--no-pad", action="store_true", help="refuse run starts instead of the training padding")
    ap.add_argument("--noise-seed", type=int, default=None, help="validation mode: explicit FQL noise per step")
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--takeover", action="store_true")
    ap.add_argument("--direct-checkpoint", type=Path, default=None)
    ap.add_argument("--direct-export", type=Path, default=None)
    ap.add_argument("--direct-training", default=None, help="deploy-contract sidecar .json or the FQL dataset .h5")
    ap.add_argument("--direct-device", default="cpu", help="device of the direct run (the server's, for bit equality)")
    ap.add_argument("--out", type=Path, default=None, help="actions JSONL")
    args = ap.parse_args(argv)
    if args.direct_checkpoint is not None and args.noise_seed is None:
        ap.error("--direct-checkpoint compares exactly only with explicit FQL noise: give --noise-seed")

    rec = RecordedEpisode(args.episode)
    plan = {}

    def choose(spec):
        plan["cams"], plan["choices"] = session_plan(rec, spec, args.camera, args.rig_document)
        return plan["choices"]

    token = Path(args.token_file).expanduser().read_bytes().strip() if args.token_file else None
    client = PolicyClient(args.host, args.port, policy_id=args.policy, choices=choose, token=token,
                          step_timeout_s=args.timeout_s, takeover=args.takeover, name="replay-demo")
    client.connect()
    spec, cams, choices = client.spec, plan["cams"], plan["choices"]
    depth = "depth" in spec["modalities"]
    steps, disp = timeline(rec, cams, spec, args.m5_store)
    steps = steps[args.start:args.start + args.steps]
    recorded = recorded_commands(rec, cams, steps, spec)
    direct = Direct(args.direct_checkpoint, args.direct_export, args.direct_training, choices, args.direct_device) \
        if args.direct_checkpoint else None
    out = args.out.open("w") if args.out else None
    rows, added = [], {c: -1 for c in cams}
    try:
        for i, step in enumerate(steps):
            end_ids, t_obs = {}, 0
            for c in cams:
                idx = step[c]
                seq, t = int(rec.frame_numbers(c)[idx]), int(rec.timestamps_ns(c)[idx])
                end_ids[c], t_obs = seq, max(t_obs, t)
                if seq <= added[c]:
                    continue
                n = rec.native(c, idx, depth=depth, alignment=args.alignment)
                corr = choices.get("native_depth_correction", {}).get(c)
                meta = depth_metadata(n)
                try:  # per-frame measured state: informational frame metadata (the request binds its own)
                    state_f = rec.state_at(t)
                except ValueError:
                    state_f = {"pose7": rec.raw.measured[0], "gripper_register": float(rec.raw.gripper_position[0])}
                    meta["state_note"] = "no robot sample at or before this frame; first sample shown (informational)"
                pre = client.add_frame_native(c, seq, t, n["rgb"], pose7=state_f["pose7"],
                                              gripper_register=state_f["gripper_register"], depth=n.get("depth"),
                                              valid=n.get("valid"), native_correction=corr if depth else None,
                                              K_native=rec.K_native(c), metadata=meta)
                added[c] = seq
                if direct is not None:
                    fmeta = {"depth_source": "native_realsense", "native_correction": corr, **meta} if depth else {}
                    direct.push(c, Frame(seq, t, pre["rgb"], pre.get("depth"), pre.get("valid"), fmeta))
            if disp is not None:
                client.add_sync_set(end_ids, disp.get(end_ids[cams[0]]))
            ref = f"{args.episode.stem}:{args.start + i}"
            try:
                state = rec.state_at(t_obs)
            except ValueError as exc:  # no measured state to bind: no request (training has no transition either)
                rows.append({"step": args.start + i, "reference_id": ref, "status": "skipped", "code": "no_state",
                             "detail": str(exc), "client_ms": {}, "server_ms": {}, "request_bytes": 0,
                             "dry_run": True})
                continue
            seed = None if args.noise_seed is None else args.noise_seed + args.start + i
            try:
                r = client.step(end_ids, ref, state["pose7"], state["gripper_register"],
                                pose_time_ns=state["pose_time_ns"], gripper_time_ns=state["gripper_time_ns"],
                                pad_run_start=not args.no_pad, noise_seed=seed)
            except ValueError as exc:  # the recorded state is too old for the training binding rule: no request
                rows.append({"step": args.start + i, "reference_id": ref, "status": "skipped",
                             "code": "binding_refused", "detail": str(exc), "client_ms": {}, "server_ms": {},
                             "request_bytes": 0, "dry_run": True})
                continue
            row = {"step": args.start + i, "reference_id": ref, "end_ids": end_ids, "observation_time_ns": t_obs,
                   "status": r.status, "code": r.code, "detail": r.detail, "frames_sent": r.frames_sent,
                   "request_bytes": r.request_bytes, "response_bytes": r.response_bytes,
                   "client_ms": r.timings_ms, "server_ms": r.server_ms, "dry_run": True}
            if r.ok:
                o = r.outputs
                row.update({k: P.jsonable(o[k]) for k in ("action_norm", "delta_phys", "flange_pose",
                                                          "execution_flange_pose", "gripper_closure",
                                                          "gripper_register", "measured_pose7") if k in o})
                row["padded"] = o["metadata"].get("padded")
                if recorded["ok"][i]:
                    row["recorded_action_phys"] = P.jsonable(recorded["action_phys"][i])
                    row["vs_recorded"] = compare_recorded(o, recorded["action_phys"][i])
            if direct is not None:
                d, code = direct.infer(end_ids, disp, not args.no_pad, ref, state, P.validation_noise(seed))
                row["direct_equal"] = (code == r.code) if d is None else (r.ok and not same_outputs(r.outputs, d))
                if d is not None and r.ok:
                    row["direct_mismatch_keys"] = same_outputs(r.outputs, d)
            rows.append(row)
            if out:
                out.write(json.dumps(row, allow_nan=False) + "\n")
    finally:
        client.close()
        if out:
            out.close()
    ok = [r for r in rows if r["status"] == "ok"]
    vs = [r["vs_recorded"] for r in ok if "vs_recorded" in r]
    summary = {
        "dry_run": True, "robot_commands_sent": 0, "episode": str(args.episode), "policy": args.policy,
        "condition": spec["condition"], "cameras": cams, "steps": len(rows), "ok": len(ok),
        "errors": dict(collections.Counter(r["code"] or r["status"] for r in rows if r["status"] != "ok")),
        "noise": "explicit validation noise (seed base + step)" if args.noise_seed is not None else "server FQL RNG",
        "direct": None if direct is None else {"compared": sum("direct_equal" in r for r in rows),
                                               "equal": sum(bool(r.get("direct_equal")) for r in rows)},
        "vs_recorded": None if not vs else {k: float(np.median([v[k] for v in vs])) for k in vs[0]},
        "vs_recorded_n": len(vs),
        "client_ms": {k: _percentiles(r["client_ms"].get(k) for r in ok)
                      for k in ("e2e_ms", "preprocess_ms", "serialize_ms", "rtt_ms", "network_ms")},
        "server_ms": {k: _percentiles(r["server_ms"].get(k) for r in ok)
                      for k in ("queue_ms", "encoder_ms", "fql_ms", "total_ms")},
        "payload_bytes": {"first_request": first_request(rows),
                          "steady_request_median": float(np.median([r["request_bytes"] for r in ok[1:]]))
                          if len(ok) > 1 else None},
        "client_stats": dict(client.stats),
    }
    print(json.dumps(summary, indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
