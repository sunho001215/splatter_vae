"""Read back one recorded robot-PC session (inference_outputs/inference_YYYYMMDD_HHMMSS/) and verify it - READ ONLY.

    venv/bin/python -I src/scripts/robot_pc/verify_session.py inference_outputs/inference_20261110_190758 [--json out]

Checks (each reported; ``ok`` false on any failure): summary.json <-> inference.h5 status/counts, zero drops, equal
dataset lengths per group, monotonic timestamps per stream, robot-state rate / sequence gaps, gripper rate, camera
frame intervals + RealSense frame-number gaps, MP4 decodable with frame count == H5 rows (and PSNR of frame 0 vs the H5
frame), observation<->state binding ages (camera / robot / gripper sync on the host clock), latency percentiles
(preprocessing->action and capture->action separately), action transformations (command goal == policy execution
pose, translation / rotation recomputed, closure -> register), executed setpoints within the executor's speed limits,
dry-run sessions sent nothing, and - when robot_msg is installed - the RobotAction bytes of the setpoints are identical
to the collector's construction and decode back to the controller-ee pose. Nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def _q_angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(2 * np.arccos(min(1.0, d)))


def _pct(v) -> dict | None:
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()), "mean": float(v.mean())}


def verify(session: Path, *, max_setpoints: int = 2000) -> dict:
    import h5py

    from ops.robot_pc import switch_controller as sc
    from ops.robot_pc.gripper import closure_to_position

    out: dict = {"session": str(session), "checks": {}, "failures": []}

    def check(name, ok, **info):
        out["checks"][name] = {"ok": bool(ok), **info}
        if not ok:
            out["failures"].append(name)

    summary = json.loads((session / "summary.json").read_text())
    h5p = session / "inference.h5"
    check("files", h5p.exists() and not (session / "inference.h5.partial").exists(),
          files={p.name: p.stat().st_size for p in sorted(session.iterdir())})
    check("drops", sum(summary["dropped"].values()) == 0, dropped=summary["dropped"])
    with h5py.File(h5p, "r") as h:
        meta = json.loads(h.attrs["meta_json"])
        mode = meta.get("mode")
        out["mode"], out["policy"] = mode, meta.get("policy")
        check("status", h.attrs["status"] == summary["status"], h5_status=str(h.attrs["status"]),
              summary_status=summary["status"], error=summary.get("error"))
        # ---- per-group equal lengths and counts
        lengths, mism = {}, []
        def walk(g, path=""):
            ds = {k: len(v) for k, v in g.items() if isinstance(v, h5py.Dataset)}
            if ds:
                lens = {k: n for k, n in ds.items() if k not in ("depth", "depth_frame_index", "depth_scale_m")}
                lengths[path or "/"] = lens
                if len(set(lens.values())) > 1:
                    mism.append(path)
            for k, v in g.items():
                if isinstance(v, h5py.Group):
                    walk(v, f"{path}/{k}" if path else k)
        walk(h)
        counts_ok = all(len(h[f"{s}/t_ns"] if f"{s}/t_ns" in h else h[f"{s}/t_step_ns"]) == n
                        for s, n in summary["counts"].items())
        check("group_lengths", not mism and counts_ok, mismatched=mism, counts=summary["counts"])
        # ---- monotonic timestamps
        mono = {}
        for s in summary["counts"]:
            col = f"{s}/t_ns" if f"{s}/t_ns" in h else f"{s}/t_step_ns"
            t = h[col][()]
            mono[s] = bool(len(t) < 2 or np.all(np.diff(t) >= 0))
        check("monotonic", all(mono.values()), per_stream=mono)
        # ---- robot state
        rt, seq = h["robot/t_ns"][()], h["robot/seq"][()]
        if len(rt) > 1:
            dt = np.diff(rt) / 1e6
            info = {"n": int(len(rt)), "rate_hz": float((len(rt) - 1) / ((rt[-1] - rt[0]) / 1e9)),
                    "dt_ms_p50": float(np.median(dt)), "dt_ms_p99": float(np.percentile(dt, 99)),
                    "dt_ms_max": float(dt.max()), "seq_gaps": int((np.diff(seq) != 1).sum()),
                    "seq_missing": int((np.diff(seq) - 1).clip(0).sum())}
            fl, ee = h["robot/flange_pose"][()], h["robot/ee_pose"][()]
            k = np.linspace(0, len(fl) - 1, min(200, len(fl))).astype(int)
            info["flange_vs_ee_frame_max_m"] = float(max(np.linalg.norm(sc.flange_from_ee(ee[i])[:3] - fl[i][:3])
                                                         for i in k))
            info["max_abs_dq"] = float(np.abs(h["robot/dq"][()]).max())
            # streaming window = from the first executed setpoint (the executor's state-age check is what matters:
            # a receipt gap > state_timeout_s latches a stop); gaps before it (preflight: camera / gripper / policy
            # start-up starving the reader) are reported separately
            ex_t = h["executed/t_ns"][()]
            stale_ms = 1e3 * float(meta.get("config", {}).get("executor", {}).get("state_timeout_s", 0.05))
            if not len(ex_t):  # no setpoints (gripper test): the window starts at the test's "start" event
                ev_t = h["events/t_ns"][()]
                ev = [x.decode() if isinstance(x, bytes) else x for x in h["events/text"][()]]
                starts = [t for t, e in zip(ev_t, ev) if e.startswith("start ")]
                ex_t = np.array([starts[0], rt[-1]], np.int64) if starts else ex_t
            if len(ex_t):
                w = (rt[:-1] >= ex_t[0]) & (rt[1:] <= ex_t[-1])
                dts = np.diff(rt)[w] / 1e6
                info.update(streaming_dt_ms_max=float(dts.max()) if len(dts) else None,
                            streaming_seq_missing=int((np.diff(seq)[w] - 1).clip(0).sum()),
                            preflight_seq_missing=int((np.diff(seq)[rt[:-1] < ex_t[0]] - 1).clip(0).sum()),
                            stale_threshold_ms=stale_ms)
                ok_rs = info["rate_hz"] > 900 and (info["streaming_dt_ms_max"] or 0) < stale_ms
            else:
                ok_rs = info["rate_hz"] > 900 and info["dt_ms_max"] < stale_ms
            check("robot_state", ok_rs, **info)
        else:
            check("robot_state", False, n=int(len(rt)))
        gt = h["gripper/t_ns"][()]
        gpos = h["gripper/position"][()]
        if len(gt) > 1:
            st = [json.loads(x) for x in h["gripper/status_json"][()]]
            check("gripper", (len(gt) - 1) / ((gt[-1] - gt[0]) / 1e9) > 40, n=int(len(gt)),
                  rate_hz=float((len(gt) - 1) / ((gt[-1] - gt[0]) / 1e9)), positions=sorted(set(gpos.tolist())),
                  faults=sorted({s.get("fault") for s in st}), status=sorted({s.get("status") for s in st}))
        else:
            check("gripper", meta.get("gripper_origin", "").startswith("static"), n=int(len(gt)),
                  origin=meta.get("gripper_origin"))
        # ---- cameras + video
        cam_info = {}
        for serial in h["cameras"]:
            g = h[f"cameras/{serial}"]
            t = g["t_ns"][()]
            fn = [json.loads(m).get("color_frame_number") for m in g["meta_json"][()]]
            fn = [x for x in fn if x is not None]
            ci = {"frames": int(len(t)),
                  "fps": float((len(t) - 1) / ((t[-1] - t[0]) / 1e9)) if len(t) > 1 else None,
                  "interval_ms": _pct(np.diff(t) / 1e6) if len(t) > 1 else None,
                  "frame_number_gaps": int((np.diff(fn) != 1).sum()) if len(fn) > 1 else None,
                  "frame_number_missing": int((np.diff(fn) - 1).clip(0).sum()) if len(fn) > 1 else None}
            vid = session / f"camera_{serial}.mp4"
            try:
                import cv2

                cap = cv2.VideoCapture(str(vid))
                n, first = 0, None
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if first is None:
                        first = fr
                    n += 1
                cap.release()
                ci["mp4_frames"] = n
                if first is not None and len(t):
                    ref = g["rgb"][0][..., ::-1].astype(float)
                    mse = float(np.mean((first.astype(float) - ref) ** 2))
                    ci["mp4_frame0_psnr_db"] = float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
                ci["mp4_ok"] = n == len(t) and n == summary["videos"][serial]["frames"]
            except ImportError:
                ci["mp4_ok"] = None
            cam_info[serial] = ci
        check("cameras_video", all(c.get("mp4_ok") is not False and (c["frame_number_missing"] or 0) == 0
                                   for c in cam_info.values()), per_camera=cam_info)
        if not str(mode).startswith("hwtest"):
            # ---- steps: latency, sync, transformations
            steps = [json.loads(x) for x in h["steps/json"][()]]
            ok_steps = [s for s in steps if s.get("status") == "ok"]
            lat = {k: _pct(s["client_ms"].get(k) for s in ok_steps)
                   for k in ("e2e_ms", "preprocess_ms", "rtt_ms", "network_ms", "acquire_ms", "capture_to_action_ms",
                             "capture_to_command_ms")}
            lat["server_total_ms"] = _pct(s["server_ms"].get("total_ms") for s in ok_steps)
            span = (steps[-1]["t_step_ns"] - steps[0]["t_step_ns"]) / 1e9 if len(steps) > 1 else None
            out["latency_ms"] = lat
            check("steps", len(ok_steps) > 0, steps=len(steps), ok_steps=len(ok_steps),
                  statuses={k: sum((s.get("code") or s["status"]) == k for s in steps)
                            for k in {s.get("code") or s["status"] for s in steps}},
                  rejected={k: sum(s.get("rejected") == k for s in steps) for k in {s.get("rejected") for s in steps} if k},
                  policy_hz=(len(steps) - 1) / span if span else None)
            # sync: observation time vs the newest robot / gripper sample at or before it (host monotonic clock)
            obs_t = np.array([s["observation_time_ns"] for s in ok_steps], np.int64)
            def ages(t_stream):
                if not len(t_stream) or not len(obs_t):
                    return None
                i = np.searchsorted(t_stream, obs_t, side="right") - 1
                good = i >= 0
                return _pct(((obs_t[good] - t_stream[i[good]]) / 1e6).tolist())
            check("sync", True, robot_age_at_observation_ms=ages(rt), gripper_age_at_observation_ms=ages(gt),
                  note="host CLOCK_MONOTONIC: RealSense global_time frame time vs the latest RobotState receipt / Modbus "
                       "status read; the policy binds the newest state at or before the frame")
            # transformations
            bad = []
            cmds = [s for s in ok_steps if s.get("command")]
            for s in cmds:
                c = s["command"]
                g, m = np.asarray(c["goal_pose7"]), np.asarray(c["measured_pose7"])
                ex = np.asarray(s["execution_flange_pose"])
                errs = {"goal_vs_policy_m": float(np.linalg.norm(g[:3] - ex[:3])), "goal_vs_policy_rad": _q_angle(g[3:], ex[3:]),
                        "translation_m": abs(float(np.linalg.norm(g[:3] - m[:3])) - c["translation_m"]),
                        "rotation_deg": abs(np.degrees(_q_angle(g[3:], m[3:])) - c["rotation_deg"]),
                        "register": abs(closure_to_position(c["gripper_closure"]) - c["gripper_register"])}
                if errs["goal_vs_policy_m"] > 1e-9 or errs["goal_vs_policy_rad"] > 1e-6 or errs["translation_m"] > 1e-9 \
                        or errs["rotation_deg"] > 1e-4 or errs["register"] > 0:
                    bad.append({"ref": c["reference_id"], **errs})
            lim = meta.get("config", {}).get("limits", {})
            check("action_transformations", not bad and all(
                s["command"]["translation_m"] <= lim.get("max_translation_m", 1) + 1e-9 and
                s["command"]["rotation_deg"] <= lim.get("max_rotation_deg", 360) + 1e-9 for s in cmds),
                  commands=len(cmds), mismatches=bad[:5],
                  max_translation_m=max((s["command"]["translation_m"] for s in cmds), default=None),
                  max_rotation_deg=max((s["command"]["rotation_deg"] for s in cmds), default=None))
        # executed setpoints
        et, ep, sent = h["executed/t_ns"][()], h["executed/pose7"][()], h["executed/sent"][()]
        ex_cfg = meta.get("config", {}).get("executor", {})
        info = {"n": int(len(et)), "sent_true": int(sent.sum())}
        if len(et) > 1:
            dt = np.diff(et) / 1e9
            cont = dt < 3 * (1.0 / ex_cfg.get("rate_hz", 500))  # consecutive ticks of one engagement
            v = np.linalg.norm(np.diff(ep[:, :3], axis=0), axis=1)[cont] / dt[cont]
            w = np.array([_q_angle(ep[i, 3:], ep[i + 1, 3:]) for i in np.nonzero(cont)[0]]) / dt[cont]
            info.update(rate_hz=float((len(et) - 1) / ((et[-1] - et[0]) / 1e9)), interval_ms=_pct(dt * 1e3),
                        max_speed_m_s=float(v.max()) if len(v) else 0.0,
                        max_ang_speed_rad_s=float(w.max()) if len(w) else 0.0)
            # the speed is per-tick distance / host tick interval: a late tick shows a lower speed, never higher
            speed_ok = info["max_speed_m_s"] <= ex_cfg.get("max_linear_speed", 0.3) * 1.05 + 1e-6 and \
                info["max_ang_speed_rad_s"] <= ex_cfg.get("max_angular_speed", 1.5) * 1.05 + 1e-6
        else:
            speed_ok = True
        check("executed", speed_ok and
              (not (mode == "dry_run" or str(mode).endswith("-rehearsal")) or info["sent_true"] == 0), **info)
        events = [x.decode() if isinstance(x, bytes) else x for x in h["events/text"][()]]
        out["events"] = events
        if str(mode).startswith("hwtest"):
            verdicts, malformed = [], []
            for e in events:
                if e.startswith("verdict "):
                    try:
                        verdicts.append(json.loads(e[len("verdict "):]))
                    except (ValueError, TypeError):
                        malformed.append(e)
            verdict = verdicts[-1] if verdicts and isinstance(verdicts[-1], dict) else {}
            rehearsal = str(mode).endswith("-rehearsal")
            check("hwtest_verdict", len(verdicts) == 1 and not malformed and verdict.get("pass") is True and
                  verdict.get("test") == meta.get("test") and verdict.get("rehearse") is rehearsal and
                  not verdict.get("abort_reasons") and summary["status"] == "complete" and
                  (not rehearsal or (verdict.get("writes", 0) == 0 and verdict.get("actions_sent", 0) == 0 and
                                     verdict.get("mode_requests_sent", 0) == 0)),
                  verdict=verdict, malformed=malformed)
        for e in events:
            if e.startswith("loop_stats "):
                out["loop_stats"] = json.loads(e[len("loop_stats "):])
        # ---- message bytes (needs robot_msg)
        try:
            import robot_msg
        except ImportError:
            check("robot_action_bytes", True, skipped="robot_msg not installed")
        else:
            k = np.linspace(0, len(ep) - 1, min(max_setpoints, len(ep))).astype(int) if len(ep) else []
            mism, dec = 0, 0.0
            for i in k:
                ours = robot_msg.encode(sc.robot_action(robot_msg, ep[i], sc.TARGET_DT_S))
                ee = sc.ee_from_flange(ep[i])  # collector verbatim: robot.py:61-62, :127-128
                pose = robot_msg.Pose(robot_msg.Vec3(*map(float, ee[:3])), robot_msg.Quaternion(*map(float, ee[3:7])))
                theirs = robot_msg.encode(robot_msg.RobotAction(frame_id="world", real_time_action=(
                    robot_msg.RobotActionRealTimeAction(ee_pose_rt=[pose], target_dt=float(sc.TARGET_DT_S)))))
                mism += ours != theirs
                d = robot_msg.decode(ours).real_time_action.ee_pose_rt[0]
                got = np.array([d.position.x, d.position.y, d.position.z, d.orientation.x, d.orientation.y,
                                d.orientation.z, d.orientation.w])
                dec = max(dec, float(np.abs(got - ee).max()))
            hold_ok = None
            if len(rt):
                eem = h["robot/ee_pose"][-1]
                a = robot_msg.encode(sc.robot_action_ee(robot_msg, eem, sc.TARGET_DT_S))
                pose = robot_msg.Pose(robot_msg.Vec3(*map(float, eem[:3])), robot_msg.Quaternion(*map(float, eem[3:7])))
                b = robot_msg.encode(robot_msg.RobotAction(frame_id="world", real_time_action=(
                    robot_msg.RobotActionRealTimeAction(ee_pose_rt=[pose], target_dt=float(sc.TARGET_DT_S)))))
                hold_ok = a == b
            check("robot_action_bytes", mism == 0 and dec < 1e-12 and hold_ok is not False, checked=len(k),
                  byte_mismatches=int(mism), decode_max_abs_err=dec, measured_hold_identical=hold_ok)
    out["ok"] = not out["failures"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args(argv)
    rep = verify(Path(args.session))
    txt = json.dumps(rep, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(txt + "\n")
    print(txt)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
