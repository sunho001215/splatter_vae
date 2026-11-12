"""Recorded replay on the robot PC: recorded ``panda_teleop_episode`` H5 (read-only) -> the client path exactly as live
(training crop + 128 px resize here; 3D-MVP: recorded native z16 aligned like the live rs.align path; multi-view: the
LIVE M5 helper on the recorded joint/gripper stream) -> policy server -> robot-side validation -> ``MockSink``. No
camera, no robot, no torch.

Times: every recorded time (frames, robot rows, gripper samples) is moved onto this host's monotonic clock by ONE
offset, so the trained spacing, the binding ages and the action ages are the recorded ones. ``pace="realtime"``
releases each frame at its (shifted) recorded time; ``pace="max"`` releases as fast as the path allows, and the
validation clock then runs on the frame clock (``ReplayClock``) so action ages stay processing latencies.

Multi-view ``sync_mode``: ``latest`` (default, the live rule: ``syncsets.LatestFrames``, one step per reference frame on
every camera's latest released frame, misalignment = warning) or ``training-sets`` (the training-equivalent mutual-
nearest M5 sets of the live helper; the robot-PC vs in-container bitwise comparison). ``inject`` (synthetic, labelled
in the summary): ``offset_ms`` {camera: ms} shifts a camera's whole capture timeline (cross-camera misalignment, its
own spacing unchanged); ``delay_ms`` {camera: ms} delivers a camera's frames that much later (a lagging camera);
``arrival_jitter_ms`` delays each frame's delivery by a seeded uniform [0, jitter] (which frame is the latest at a
step). Delays never change capture times, and a camera still delivers in order.
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


def frame_events(cameras, cams: list, start: int, count: int | None, shifts: dict | None = None) -> list:
    """(release t, camera order, cam, stream index, capture t) on the recorded clock, in release order, of the session
    cameras from the reference camera's frame ``start`` on, for ``count`` reference frames (+ the other cameras' frames
    inside that span). ``shifts`` (synthetic injection): {cam: (capture shift ns, per-frame release delay ns array)};
    without it release t = capture t = the recorded time."""
    shifts = shifts or {}

    def times(c):
        cap = np.asarray(cameras.timestamps_ns(c), np.int64) + int(shifts.get(c, (0, None))[0])
        delay = shifts.get(c, (0, None))[1]
        # a camera delivers in order (one USB queue): a delayed frame holds back the next one
        return cap, cap if delay is None else np.maximum.accumulate(cap + np.asarray(delay, np.int64)[:len(cap)])

    ref, _ = times(cams[0])
    stop = len(ref) if count is None else min(len(ref), start + count)
    lo, hi = int(ref[start]) - FRAME_ARRIVAL_NS, int(ref[stop - 1]) + FRAME_ARRIVAL_NS
    out = []
    for v, c in enumerate(cams):
        cap, rel = times(c)
        for i, t in enumerate(cap):
            if (c == cams[0] and start <= i < stop) or (c != cams[0] and lo <= t <= hi):
                out.append((int(rel[i]), v, c, i, int(t)))
    return sorted(out)


def injection_shifts(cameras, cams: list, inject: dict | None) -> dict | None:
    """``inject`` {"offset_ms": {cam: ms}, "delay_ms": {cam: ms}, "arrival_jitter_ms": ms, "seed": int} ->
    ``frame_events`` shifts (capture shift ns, per-frame delivery delay ns or None)."""
    if not inject:
        return None
    offsets, delays = inject.get("offset_ms") or {}, inject.get("delay_ms") or {}
    unknown = (set(offsets) | set(delays)) - set(cams)
    if unknown:
        raise ValueError(f"injection for cameras {sorted(unknown)} not in the session {cams}")
    rng = np.random.default_rng(int(inject.get("seed", 0)))
    jitter = int(round(float(inject.get("arrival_jitter_ms", 0.0)) * 1e6))
    if jitter < 0 or any(float(v) < 0 for v in delays.values()):
        raise ValueError("delivery delays and jitter must be >= 0")
    out = {}
    for c in cams:
        n = len(cameras.timestamps_ns(c))
        delay = np.full(n, int(round(float(delays.get(c, 0.0)) * 1e6)), np.int64)
        if jitter:
            delay = delay + rng.integers(0, jitter + 1, n)
        out[c] = (int(round(float(offsets.get(c, 0.0)) * 1e6)), delay if delay.any() else None)
    return out


def run(client, loop, cameras, state, clock: ReplayClock, *, cams: list, start: int = 0, count: int | None = None,
        live_m5=None, noise_seed: int | None = None, on_row=None, sync_mode: str = "latest",
        inject: dict | None = None) -> list:
    """Replay; returns the rows (``on_row(row)`` per step). Single-view: one step per reference frame. Multi-view
    ``latest``: one step per reference frame on every camera's latest released frame (``syncsets.LatestFrames``, the
    live rule; rows carry ``sync``); ``training-sets``: one step per final M5 set (sets reach the server with their
    displacement). realtime pace behaves like the live loop: when the next reference frame is already due, the current
    observation is sent as history only and the step is taken on the newest one (``skipped_behind`` /
    ``sync_superseded``), so a slow path lowers the step rate instead of building a backlog."""
    from ops.robot_pc.syncsets import LatestFrames, StateFeed, step_sets

    if sync_mode not in ("latest", "training-sets"):
        raise ValueError("sync_mode must be latest or training-sets")
    spec = client.spec
    depth = "depth" in spec["modalities"]
    corr = client.choices.get("native_depth_correction") or {}
    rows, acquired = [], {c: 0.0 for c in cams}
    multiview = len(cams) > 1
    if multiview and sync_mode == "training-sets" and live_m5 is None:
        raise ValueError("training-sets replay needs the live M5 helper")
    latest = LatestFrames(client, live_m5, state, cams, period_s=float(spec["frame_period_s"]), counts=loop.counts,
                          state_lead_ns=FRAME_ARRIVAL_NS) if multiview and sync_mode == "latest" else None
    feed = StateFeed(live_m5, state) if multiview and sync_mode == "training-sets" else None
    shifts = injection_shifts(cameras, cams, inject)
    events = frame_events(cameras, cams, start, count, shifts)
    ref_rel = sorted(e[0] for e in events if e[2] == cams[0])  # reference release times (recorded clock)
    ref_k = -1  # position of the newest released reference frame in ref_rel

    def behind(idx) -> bool:  # realtime: the next reference frame is already due on the replay clock
        return clock.pace == "realtime" and ref_k + 1 < len(ref_rel) and \
            time.monotonic_ns() >= int(ref_rel[ref_k + 1]) + clock.offset_ns

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

    def due(rel_ns: int) -> bool:  # realtime: that release time has already passed on the wall clock
        return clock.pace == "realtime" and time.monotonic_ns() >= int(rel_ns) + clock.offset_ns

    def service_deadline(next_rel_ns: int | None) -> None:
        """Latest-frame rule: a step whose partner wait runs out STRICTLY before the next release is taken at its
        deadline (as the live loop does), on the frames released so far. A frame released at the deadline, or
        already due after a late wake-up, is drained first (the live loop drains every due frame before deciding)."""
        dl = latest.deadline_ns() if latest is not None else None
        if dl is None or (next_rel_ns is not None and dl >= int(next_rel_ns) + clock.offset_ns):
            return
        clock.release(dl - clock.offset_ns)  # realtime: wait until the deadline; max: the frame clock moves there
        if (next_rel_ns is not None and due(next_rel_ns)) or behind(None):
            return  # overslept into the next release / a newer reference frame is due: drain it first
        s = latest.newest_step(clock.now())
        if s is not None:
            emit(*s[:3], sync=s[3])

    end_of_recording = count is None or start + count >= len(cameras.timestamps_ns(cams[0]))
    for i, (t_rel, _, cam, idx, t_cap) in enumerate(events):
        service_deadline(t_rel)
        nxt = events[i + 1][0] if i + 1 < len(events) else None
        n = cameras.native(cam, idx)
        clock.release(t_rel)
        t = t_cap + clock.offset_ns  # capture time on this host's clock (= release time without injection)
        if cam == cams[0]:
            ref_k += 1
        acquired[cam] += n["acquire_ms"]
        try:
            fs = state.bound_state(t)  # informational per-frame sample (the request binds its own)
            pose, grip = fs.pose7, fs.gripper_register
        except StateUnavailable:
            pose, grip = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.0
        meta = {"alignment": f"{n['alignment']} (recorded replay)",
                "depth_timestamp_ns": int(n["depth_timestamp_ns"]) + clock.offset_ns + t_cap - int(
                    cameras.timestamps_ns(cam)[idx])} if depth else {}
        client.add_frame_native(cam, int(n["seq"]), t, n["rgb"], pose7=pose, gripper_register=grip,
                                depth=n.get("depth"), valid=n.get("valid"), native_correction=corr.get(cam),
                                K_native=cameras.K_native(cam), metadata=meta)
        if not multiview:
            if cam == cams[0]:
                if behind(idx) and not (count is not None and idx + 1 >= start + count):
                    loop.counts["skipped_behind"] += 1
                else:
                    emit({cam: int(n["seq"])}, t, f"{cam}:{idx}")
            continue
        if latest is not None:
            latest.add_frame(cam, int(n["seq"]), t, clock.now())
            if behind(idx):
                continue  # a newer reference frame is already due: it supersedes this one (sync_superseded)
            if nxt is not None and (nxt == t_rel or due(nxt)):
                continue  # drain every frame released at the same time / already due before deciding (live drain)
            s = latest.newest_step(clock.now())
            if s is not None:
                emit(*s[:3], sync=s[3])
            continue
        feed.start(t)
        live_m5.add_frame(cam, int(n["seq"]), t)
        feed.feed(t + FRAME_ARRIVAL_NS)  # robot rows / gripper samples reach the host before the frame
        step_sets(client, live_m5.poll(), cams, take, newest_only=clock.pace == "realtime", counts=loop.counts)
    if latest is not None and end_of_recording:  # the recording ends: the newest reference frame steps on what was
        service_deadline(None)  # released, at its deadline if it waits for a partner (a live stream has no end:
        s = latest.newest_step(clock.now(), flush=True)  # no flush mid-recording)
        if s is not None:
            emit(*s[:3], sync=s[3])
    elif feed is not None and end_of_recording:  # the recording ends: flush what the stream end decides
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
                   pings: int = 0, sync_mode: str = "latest", inject: dict | None = None) -> tuple[list, dict]:
    """Everything of one recorded replay: policy selection from the server, session choices from the recording and
    the task's rig document, the persistent link, the live M5 helper (multi-view), validation and the MockSink."""
    from ops.robot_pc import cli, safety, selection
    from ops.robot_pc import config as C
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
                   noise_seed=args.noise_seed, on_row=on_row, sync_mode=sync_mode, inject=inject)
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
        if inject:
            summary["injection"] = {**inject, "note": "SYNTHETIC: capture timelines shifted / release order jittered "
                                                      "on purpose (sync validation); not a recording property"}
        if len(cams) > 1:
            from ops.robot_pc.syncsets import sync_summary

            summary["multiview_sync"] = sync_mode
            if sync_mode == "latest":
                summary["sync"] = sync_summary(rows, loop.counts)
        if live_m5 is not None:
            summary["m5"] = {"source": "live helper on the recorded joint/gripper stream", **dict(live_m5.counts),
                             "asset_sha256": live_m5.motion.asset_sha256,
                             "use": "training-sets: the step sets" if sync_mode == "training-sets" else
                                    "informational: displacement of each step's latest frames"}
            if m5_store is not None and sync_mode == "training-sets":
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
