"""Recorded replay on the robot PC: recorded ``panda_teleop_episode`` H5 (read-only) -> the client path exactly as live
(training crop + 128 px resize here; 3D-MVP: recorded native z16 aligned like the live rs.align path; multi-view: the
LIVE M5 helper on the recorded joint/gripper stream) -> policy server -> robot-side validation -> ``MockSink``. No
camera, no robot, no torch.

Times: every recorded time (frames, robot rows, gripper samples) is moved onto this host's monotonic clock by ONE
offset, so the trained spacing, the binding ages and the action ages are the recorded ones. ``pace="realtime"``
releases each frame at its (shifted) recorded time; ``pace="max"`` releases as fast as the path allows, and the
validation clock then runs on the frame clock (``ReplayClock``) so action ages stay processing latencies.
"""

from __future__ import annotations

import time

import numpy as np

from ops.robot_pc.robot_state import StateUnavailable

FRAME_ARRIVAL_NS = 65_000_000  # recorded t_recv - t of every camera (p50 64.9-65.2 ms): frames arrive after robot rows


class ReplayClock:
    """Validation / age clock of a replay. realtime: the wall clock (frames are released at their shifted times).
    max: the frame clock, i.e. ages = wall time since the newest frame's release + its recorded offset from the
    observation, so action ages and capture->action stay processing latencies (no negative ages, no backlog)."""

    def __init__(self, first_recorded_ns: int, pace: str):
        if pace not in ("realtime", "max"):
            raise ValueError("pace must be realtime or max")
        self.pace = pace
        self.offset_ns = time.monotonic_ns() + 50_000_000 - int(first_recorded_ns)
        self._skew = 0

    def release(self, recorded_ns: int) -> int:
        """Wait (realtime) until the frame's shifted time; returns it."""
        t = int(recorded_ns) + self.offset_ns
        if self.pace == "realtime":
            delay = (t - time.monotonic_ns()) / 1e9
            if delay > 0:
                time.sleep(delay)
        else:  # frame clock = wall clock + (newest shifted frame time - its wall release time)
            self._skew = t - time.monotonic_ns()
        return t

    def now(self) -> int:
        return time.monotonic_ns() + self._skew


def frame_events(cameras, cams: list, start: int, count: int | None) -> list:
    """(recorded t, camera order, cam, stream index) of the session cameras from the reference camera's frame
    ``start`` on, for ``count`` reference frames (+ the other cameras' frames inside that span)."""
    ref = cameras.timestamps_ns(cams[0])
    stop = len(ref) if count is None else min(len(ref), start + count)
    lo, hi = int(ref[start]) - FRAME_ARRIVAL_NS, int(ref[stop - 1]) + FRAME_ARRIVAL_NS
    out = []
    for v, c in enumerate(cams):
        for i, t in enumerate(cameras.timestamps_ns(c)):
            if (c == cams[0] and start <= i < stop) or (c != cams[0] and lo <= t <= hi):
                out.append((int(t), v, c, i))
    return sorted(out)


def run(client, loop, cameras, state, clock: ReplayClock, *, cams: list, start: int = 0, count: int | None = None,
        live_m5=None, noise_seed: int | None = None, on_row=None) -> list:
    """Replay; returns the rows (``on_row(row)`` per step). Single-view: one step per reference frame. Multi-view: one
    step per final M5 set (sets reach the server with their displacement; steps whose history lacks evidence are
    answered ``incomplete_history`` by the server, as in training). realtime pace behaves like the live loop: when the
    next reference frame is already due, the current observation is sent as history only and the step is taken on the
    newest one (counted in ``skipped_behind``), so a slow path lowers the step rate instead of building a backlog."""
    from ops.robot_pc.syncsets import StateFeed, step_sets

    spec = client.spec
    depth = "depth" in spec["modalities"]
    corr = client.choices.get("native_depth_correction") or {}
    rows, acquired = [], {c: 0.0 for c in cams}
    feed = None if live_m5 is None else StateFeed(live_m5, state)  # the live loops' feed (syncsets)
    ref_t = cameras.timestamps_ns(cams[0])

    def behind(idx) -> bool:  # realtime: the next reference frame is already due on the replay clock
        return clock.pace == "realtime" and idx + 1 < len(ref_t) and \
            time.monotonic_ns() >= int(ref_t[idx + 1]) + clock.offset_ns

    def emit(end_ids, t_obs, ref_id, **extra):
        seed = None if noise_seed is None else noise_seed + len(rows)
        row = {**loop.step(end_ids, t_obs, ref_id, noise_seed=seed, acquire_ms=sum(acquired.values())), **extra}
        for c in acquired:
            acquired[c] = 0.0
        rows.append(row)
        if on_row is not None:
            on_row(row)

    def take(end_ids, t_obs, ref_id, disp):
        emit(end_ids, t_obs, ref_id, surface_disp_m=disp)

    end_of_recording = count is None or start + count >= len(cameras.timestamps_ns(cams[0]))
    for t_rec, _, cam, idx in frame_events(cameras, cams, start, count):
        n = cameras.native(cam, idx)
        t = clock.release(t_rec)
        acquired[cam] += n["acquire_ms"]
        try:
            fs = state.bound_state(t)  # informational per-frame sample (the request binds its own)
            pose, grip = fs.pose7, fs.gripper_register
        except StateUnavailable:
            pose, grip = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.0
        meta = {"alignment": f"{n['alignment']} (recorded replay)",
                "depth_timestamp_ns": int(n["depth_timestamp_ns"]) + clock.offset_ns} if depth else {}
        client.add_frame_native(cam, int(n["seq"]), t, n["rgb"], pose7=pose, gripper_register=grip,
                                depth=n.get("depth"), valid=n.get("valid"), native_correction=corr.get(cam),
                                K_native=cameras.K_native(cam), metadata=meta)
        if live_m5 is None:
            if cam == cams[0]:
                if behind(idx) and not (count is not None and idx + 1 >= start + count):
                    loop.counts["skipped_behind"] += 1
                else:
                    emit({cam: int(n["seq"])}, t, f"{cam}:{idx}")
            continue
        feed.start(t)
        live_m5.add_frame(cam, int(n["seq"]), t)
        feed.feed(t + FRAME_ARRIVAL_NS)  # robot rows / gripper samples reach the host before the frame
        step_sets(client, live_m5.poll(), cams, take, newest_only=clock.pace == "realtime", counts=loop.counts)
    if live_m5 is not None and end_of_recording:  # the recording ends: flush what the stream end decides
        feed.feed(2**62)
        step_sets(client, live_m5.poll(end_of_stream=True), cams, take, newest_only=False, counts=loop.counts)
    return rows


def recorded_comparison(recording, cams: list, rows: list, offset_ns: int, spec: dict) -> dict:
    """Per ok row: the recorded command of that step (``actions.step_table`` at the unshifted observation time) vs
    the policy's delta (training-split episodes: a consistency check, not an evaluation)."""
    from s4d.rwlearn import actions as A

    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": 0}
    raw, ref = recording.raw, cams[0]
    times = np.array([r["observation_time_ns"] - offset_ns for r in ok], np.int64)
    cam_t = recording.timestamps_ns(ref)
    seg_all = np.asarray(raw.cameras[ref]["segment"], np.int64)[raw.cameras[ref]["paired_colour"]]
    seg = seg_all[np.clip(np.searchsorted(cam_t, times, side="right") - 1, 0, len(cam_t) - 1)]
    c = spec["action"]
    tab = A.step_table(raw, times, seg, nominal_period_s=c["nominal_period_s"],
                       max_step_factor=c["max_step_s"] / c["nominal_period_s"])
    pos, rot, grip = [], [], []
    for k, r in enumerate(ok):
        if not tab.transition_ok[k]:
            continue
        phys = tab.action_phys[k]
        mine = A.pose_delta(A.pose7_to_matrix(np.asarray(r["measured_pose7"])),
                            A.pose7_to_matrix(np.asarray(r["flange_pose"])))
        pos.append(1e3 * float(np.linalg.norm(mine[:3] - phys[:3])))
        Rp, Rr = (A.apply_delta(np.eye(4), x)[:3, :3] for x in (mine, phys[:6]))
        rot.append(float(np.degrees(np.arccos(np.clip((np.trace(Rp.T @ Rr) - 1) / 2, -1, 1)))))
        grip.append(abs(r["gripper_closure"] - float(phys[6])))
    if not pos:
        return {"n": 0}
    return {"n": len(pos), "position_error_mm_median": float(np.median(pos)),
            "rotation_error_deg_median": float(np.median(rot)), "gripper_abs_error_median": float(np.median(grip)),
            "note": "training-split recordings: consistency with the recorded commands, not an evaluation"}


def replay_episode(cfg: dict, args, *, episode, pace: str, start: int = 0, count: int | None = None,
                   name: str = "robot-pc-replay", m5_store=None, on_row=None, route: str = "", warmup: int = 0,
                   pings: int = 0) -> tuple[list, dict]:
    """Everything of one recorded replay: policy selection from the server, session choices from the recording and
    the task's rig document, the persistent link, the live M5 helper (multi-view), validation and the MockSink."""
    from ops.robot_pc import cli, safety
    from ops.robot_pc import config as C
    from ops.robot_pc import selection
    from ops.robot_pc.cameras import RecordedCameras
    from ops.robot_pc.loop import PolicyLoop, percentiles, summarize
    from ops.robot_pc.robot_state import RecordedRobotState
    from ops.robot_pc.sinks import MockSink
    from s4d.rwlearn.deploy.replay import RecordedEpisode

    pid, task, _, info = cli.choose_policy(cfg, args)  # info["resident"]: loaded on the server before this session
    rec = RecordedEpisode(episode)
    plan = {}

    def choose(spec):
        cams = selection.session_cameras(spec, args.camera)
        missing = [c for c in cams if c not in rec.cameras()]
        if missing:
            raise SystemExit(f"cameras {missing} not in the recording {rec.cameras()}")
        plan["cams"] = cams
        return selection.session_choices(spec, cams, native_hw={c: rec.native_hw(c) for c in cams},
                                         K_native={c: rec.K_native(c) for c in cams},
                                         rig_document=cli.rig_document(cfg, task), identity_depth=args.identity_depth)

    t_hello = time.monotonic_ns()
    client = cli.connect(cfg, args, pid, choose, name)
    hello_ms = (time.monotonic_ns() - t_hello) / 1e6
    try:
        ping_ms = [client.ping() for _ in range(int(pings))]
        spec, cams = client.spec, plan["cams"]
        cameras = RecordedCameras(rec, cams, depth="depth" in spec["modalities"])
        live_m5 = C.m5(cfg, task, cams, float(spec["frame_period_s"])) if spec["views"] > 1 else None
        clock = ReplayClock(int(cameras.timestamps_ns(cams[0])[start]), pace)
        state = RecordedRobotState(rec, clock.offset_ns)
        sink = MockSink()
        loop = PolicyLoop(client, state, C.limits(cfg, task), sink, clock=clock.now, pad_run_start=not args.no_pad)
        rows = run(client, loop, cameras, state, clock, cams=cams, start=start, count=count, live_m5=live_m5,
                   noise_seed=args.noise_seed, on_row=on_row)
        summary = {"policy": pid, "task": task, "condition": spec["condition"], "cameras": cams,
                   "catalog_info": info, "execution_eligible": safety.execution_eligible(info),
                   "episode": str(episode), "pace": pace, "sink": "MockSink", "commands_recorded": len(sink.items),
                   "session_open_ms": hello_ms, "ping_ms": percentiles(ping_ms),
                   **summarize(rows, route=route, mode=f"replay-{pace}", warmup=warmup),
                   "vs_recorded": recorded_comparison(rec, cams, rows, clock.offset_ns, spec),
                   "loop_counts": dict(loop.counts), "client_stats": dict(client.stats),
                   "input_note": "recorded JPEG decoded (bit-exact with training); depth: recorded z16 through the "
                                 "rs.align emulation + colour-z + native scale; capture time = replay release time "
                                 "(no camera-to-host latency)"}
        if live_m5 is not None:
            summary["m5"] = {"source": "live helper on the recorded joint/gripper stream", **dict(live_m5.counts),
                             "asset_sha256": live_m5.motion.asset_sha256}
            if m5_store is not None:
                from s4d.rwlearn.deploy.replay import m5_from_store

                stored = m5_from_store(m5_store, rec)["disp"]
                live = {r["end_ids"][cams[0]]: r["surface_disp_m"] for r in rows if r.get("surface_disp_m") is not None}
                diffs = [abs(float(stored[k]) - float(v)) for k, v in live.items() if k in stored]
                summary["m5"]["vs_store"] = {"compared": len(diffs), "max_abs_diff_m": max(diffs) if diffs else None,
                                             "not_in_store": sum(k not in stored for k in live),
                                             "store": str(m5_store)}
    finally:
        client.close()
    return rows, summary
