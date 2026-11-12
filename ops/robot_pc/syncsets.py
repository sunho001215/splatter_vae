"""Multi-view camera steps on the robot PC (CPU, no torch), ONE implementation for recorded replay (``replay.run``), the
live dry run (``dryrun.run``) and the live session (``live.LiveSession``):

* ``LatestFrames`` (the live rule since the user directive of 2026-11-12 evening): one step per new reference-camera
  frame on the LATEST valid frame of every camera at that instant: each other camera contributes its newest frame
  captured no later than half a frame period after the reference frame when the client still holds such a frame with
  its whole history (a larger client ``frame_buffer``; with the default buffer = the trained span, the newest
  frame). A step waits at most ``wait_s`` (default half a frame period) for a camera whose newest frame is more than
  half a period older than the reference, then goes ahead with that camera's newest (stale) frame. Cross-camera
  misalignment is a WARNING, never a refusal: every step carries a ``sync`` record
  (per-camera frame ids, capture times, ages at the step, pairwise offsets, the reference offset vs the training
  mutual-nearest bound, the M5 robot-surface displacement of exactly these frames when the live helper can compute
  it) and mismatches are counted (``sync_mismatch`` / ``sync_motion_over_bound``). Per-camera checks (trained
  spacing, timestamp regularity, frame validity, calibration), the camera set / training order and every timeout are
  unchanged (server deploy history, ``live`` / ``dryrun`` idle timeouts);
* ``StateFeed``: every robot row (time, q7) and gripper register sample of the state source reaches the M5 helper
  exactly once and in time order, from one second before the first frame on. Each stream remembers the newest time it
  fed, so a live sample stamped before a poll but appended to the reader's buffer after it is fed by the next poll
  (feeding up to a wall-clock bound would drop it for good);
* ``step_sets`` / ``LiveSyncSets``: the former strict rule (training-equivalent mutual-nearest sets finalised by the
  live M5 helper, every final set declared with its displacement). Kept for recorded replay's ``training-sets`` mode
  (the bitwise robot-PC vs in-container comparison on the training store's sets) and tests; live loops no longer use it.
"""

from __future__ import annotations

import collections

import numpy as np

from s4d.rwlearn.store import MAX_SET_MOTION_M

STATE_LEAD_NS = 10**9  # rows from one second before the first frame (the recorded replay's start)

# Server answers without an action for a multi-view window (a server older than the 2026-11-12 relaxation refusing a
# moving / unpaired set; any server: a camera's own history incomplete, padding not step-aligned): an ANSWER, never a
# route or server failure.
SYNC_REFUSALS = frozenset({"unsynchronized", "incomplete_history"})
SYNC_COUNTS = ("sync_steps", "sync_mismatch", "sync_motion_over_bound", "sync_partner_wait_expired",
               "sync_superseded", "sync_cap_fallback")


class StateFeed:
    """Feeds ``state.rows`` (``RobotStateSource.rows``: robot [(t, q7)], gripper [(t, register)], each in time order)
    to ``live_m5`` exactly once per sample."""

    def __init__(self, live_m5, state):
        self.m5, self.state = live_m5, state
        self.robot_t: int | None = None  # newest time fed per stream (None: nothing asked yet)
        self.gripper_t: int | None = None

    def start(self, t_ns: int) -> None:
        """At the first frame: feed from ``STATE_LEAD_NS`` before it."""
        if self.robot_t is None:
            self.robot_t = self.gripper_t = int(t_ns) - STATE_LEAD_NS

    def feed(self, up_to_ns: int) -> None:
        if self.robot_t is None:
            return
        robot, gripper = self.state.rows(min(self.robot_t, self.gripper_t), int(up_to_ns))
        for t, q in robot:
            if t > self.robot_t:
                self.m5.add_robot(t, q)
                self.robot_t = t
        for t, g in gripper:
            if t > self.gripper_t:
                self.m5.add_gripper(t, g)
                self.gripper_t = t


def step_sets(client, sets: list, cams: list, step, *, newest_only: bool, counts) -> None:
    """``client.add_sync_set`` for each final set (oldest first), then ``step(end_ids, observation_time_ns,
    reference_id, surface_disp_m)`` on it (``newest_only``: on the last one only; the others count
    ``counts["skipped_behind"]``)."""
    for k, s in enumerate(sets):
        disp = s["surface_disp_m"] if np.isfinite(s["surface_disp_m"]) else None  # None: no valid robot state
        client.add_sync_set(s["frame_ids"], disp)
        if newest_only and k + 1 < len(sets):
            counts["skipped_behind"] += 1
            continue
        step(s["frame_ids"], s["observation_time_ns"], f"set:{s['frame_ids'][cams[0]]}", disp)


class LiveSyncSets:
    """A live loop's multi-view sets: ``add_frame`` per drained camera frame (any camera order), ``newest_step`` once
    per loop iteration -> ``(end_ids, observation_time_ns, reference_id, surface_disp_m)`` of the newest final set or
    None. The helper must have the session's cameras in training order (checked)."""

    def __init__(self, client, live_m5, state, cams):
        cams = [str(c) for c in cams]
        if list(live_m5.sets.cameras) != cams:
            raise ValueError(f"the live M5 helper's cameras {live_m5.sets.cameras} are not the session cameras {cams} "
                             "in training order")
        self.client, self.m5, self.cams = client, live_m5, cams
        self.feed = StateFeed(live_m5, state)
        self._fresh = False

    def add_frame(self, camera_id: str, frame_id: int, capture_ns: int) -> None:
        self.feed.start(capture_ns)
        self.m5.add_frame(camera_id, int(frame_id), int(capture_ns))
        self._fresh = True

    def newest_step(self, now_ns: int, counts):
        """Robot / gripper rows up to ``now_ns`` -> the helper's newly final sets -> ``add_sync_set`` each; the newest
        set's step arguments (None when no set became final). Polls only after new frames or while sets wait for their
        member states (nothing else can finalise a set)."""
        if not (self._fresh or self.m5.pending):
            return None
        self._fresh = False
        self.feed.feed(now_ns)
        out = []
        step_sets(self.client, self.m5.poll(), self.cams, lambda *a: out.append(a), newest_only=True, counts=counts)
        return out[-1] if out else None


def sync_record(cams: list, end_ids: dict, times: dict, now_ns: int, half_ns: float, disp, max_motion_m: float) -> dict:
    """The per-observation sync log of one multi-view step (JSON-safe): frame ids and capture times per camera, the
    age of every frame at the step (``age_ms``, this host's monotonic clock) and relative to the observation time =
    newest frame (``age_rel_ms``), pairwise offsets ``a|b`` = t_a - t_b, the largest offset from the reference camera
    vs the training mutual-nearest bound (half a frame period; ``mismatch`` = the former strict refusal) and the M5
    displacement of exactly these frames (metres; ``"no_state"`` = no valid robot state; None = not computed)."""
    ref_t, obs = times[cams[0]], max(times.values())
    ref_off = max(abs(times[c] - ref_t) for c in cams) / 1e6
    finite = disp is not None and bool(np.isfinite(disp))
    return {"rule": "latest", "frame_ids": {c: int(end_ids[c]) for c in cams},
            "capture_ns": {c: int(times[c]) for c in cams},
            "age_ms": {c: (int(now_ns) - int(times[c])) / 1e6 for c in cams},
            "age_rel_ms": {c: (int(obs) - int(times[c])) / 1e6 for c in cams},
            "offsets_ms": {f"{a}|{b}": (int(times[a]) - int(times[b])) / 1e6
                           for i, a in enumerate(cams) for j, b in enumerate(cams) if i < j},
            "max_offset_ms": ref_off, "bound_ms": half_ns / 1e6, "mismatch": ref_off * 1e6 >= half_ns,
            "surface_disp_m": (float(disp) if finite else "no_state") if disp is not None else None,
            "motion_over_bound": disp is not None and (not finite or float(disp) > max_motion_m)}


class LatestFrames:
    """A multi-view loop's steps on the latest valid frame of every camera (module docstring): ``add_frame`` per drained
    camera frame (any camera order; ``now_ns`` = when it reached the loop on the loop's clock), ``newest_step`` once per
    loop iteration -> ``(end_ids, observation_time_ns, reference_id, sync_record)`` or None. ``live_m5`` (optional,
    informational): the validated live M5 helper on the session cameras in training order (checked); with it, each
    step's displacement is computed when the member states are final and declared to the server
    (``add_sync_set``). Counts go to ``counts`` (the loop's)."""

    def __init__(self, client, live_m5, state, cams, *, period_s: float, counts, wait_s: float | None = None,
                 state_lead_ns: int = 0):
        cams = [str(c) for c in cams]
        if live_m5 is not None and list(live_m5.sets.cameras) != cams:
            raise ValueError(f"the live M5 helper's cameras {live_m5.sets.cameras} are not the session cameras {cams} "
                             "in training order")
        self.client, self.m5, self.cams, self.counts = client, live_m5, cams, counts
        self.feed = None if live_m5 is None else StateFeed(live_m5, state)
        self.half_ns = float(period_s) * 1e9 / 2
        self.wait_ns = int(self.half_ns if wait_s is None else float(wait_s) * 1e9)
        self.latest: dict = dict.fromkeys(cams)  # camera -> newest (frame id, capture ns)
        # end-frame candidates per camera: the newest ``retain`` frames whose WHOLE trained history is among the
        # frames both the client retains (``frame_capacity``: sent with the step or already sent) and the server
        # still holds (its advertised ``limits.history_capacity``), so the chosen end frame can always be assembled:
        # retain = min(client, server) - span + 1 (the newest frame only with the default client buffer = span)
        spec = getattr(client, "spec", None) or {}
        span = (int(spec["history_length"]) - 1) * int(spec.get("frame_spacing", 1)) + 1 \
            if "history_length" in spec else 1
        held = int(getattr(client, "frame_capacity", 1) or 1)
        server = (getattr(client, "limits", None) or {}).get("history_capacity")
        if server is not None:
            held = min(held, int(server))
        self.retain = max(1, held - span + 1)
        self._recent = {c: collections.deque(maxlen=self.retain) for c in cams}
        self._ref_seen_ns = 0
        self._stepped = None  # reference frame id of the last step
        self.state_lead_ns = int(state_lead_ns)  # recorded replay: robot rows reach the host before the frames

    def add_frame(self, camera_id: str, frame_id: int, capture_ns: int, now_ns: int) -> None:
        camera_id = str(camera_id)
        if camera_id not in self.latest:
            raise ValueError(f"camera {camera_id} is not in this session {self.cams}")
        if self.feed is not None:
            self.feed.start(capture_ns)
        if camera_id == self.cams[0]:
            prev = self.latest[camera_id]
            if prev is not None and prev[0] != self._stepped:
                self.counts["sync_superseded"] += 1  # a newer reference frame before this one could step
            self._ref_seen_ns = int(now_ns)
        self.latest[camera_id] = (int(frame_id), int(capture_ns))
        self._recent[camera_id].append(self.latest[camera_id])

    def _member(self, cam: str, t_ref: int) -> tuple:
        """Among the camera's newest ``retain`` frames (assemblable with their whole history), the newest captured no
        later than half a period after the reference; when there is none (every retained frame is from a later
        instant: the reference camera delivers that late), its newest frame, counted ``sync_cap_fallback`` (the
        step's record shows the offset)."""
        fit = [f for f in self._recent[cam] if f[1] <= t_ref + self.half_ns]
        if fit:
            return fit[-1]
        self.counts["sync_cap_fallback"] += 1
        return self._recent[cam][-1]

    def deadline_ns(self) -> int | None:
        """When the pending reference frame's partner wait runs out (loop clock), or None: no step is waiting for a
        late partner. Recorded replay steps at this time when no frame is released before it, as a live loop
        polling every millisecond would."""
        ref = self.latest[self.cams[0]]
        if ref is None or ref[0] == self._stepped or any(v is None for v in self.latest.values()):
            return None
        if not any(self.latest[c][1] < ref[1] - self.half_ns for c in self.cams[1:]):
            return None
        return self._ref_seen_ns + self.wait_ns

    def newest_step(self, now_ns: int, *, flush: bool = False):
        """The step on the newest reference frame once its partners are no older than half a period before it or the
        wait ran out (``flush``: no frame will follow, step now). None: no new reference frame, a camera without any
        frame yet (the loops' idle timeouts refuse a missing camera), or still waiting."""
        ref = self.latest[self.cams[0]]
        if ref is None or ref[0] == self._stepped or any(v is None for v in self.latest.values()):
            return None
        late = [c for c in self.cams[1:] if self.latest[c][1] < ref[1] - self.half_ns]
        if late and not flush and int(now_ns) - self._ref_seen_ns < self.wait_ns:
            return None
        if late:
            self.counts["sync_partner_wait_expired"] += 1
        self._stepped = ref[0]
        members = {c: ref if c == self.cams[0] else self._member(c, ref[1]) for c in self.cams}
        end_ids = {c: members[c][0] for c in self.cams}
        times = {c: members[c][1] for c in self.cams}
        disp = self._displacement(times, now_ns)
        if disp is not None:
            self.client.add_sync_set(end_ids, float(disp) if np.isfinite(disp) else None)
        limit = MAX_SET_MOTION_M
        rec = sync_record(self.cams, end_ids, times, now_ns, self.half_ns, disp, limit)
        self.counts["sync_steps"] += 1
        self.counts["sync_mismatch"] += rec["mismatch"]
        self.counts["sync_motion_over_bound"] += rec["motion_over_bound"]
        return end_ids, max(times.values()), f"latest:{ref[0]}", rec

    def _displacement(self, times: dict, now_ns: int):
        """M5 displacement of exactly these frames (informational): None without the helper or while a member's
        state is not final yet (counted ``not_ready``)."""
        if self.m5 is None:
            return None
        self.feed.feed(int(now_ns) + self.state_lead_ns)
        stamps = [times[c] for c in self.cams]
        if not all(self.m5.timeline.ready(t) for t in stamps):
            self.m5.counts["not_ready"] += 1
            return None
        states, ok = self.m5.timeline.states(stamps)
        disp = self.m5.motion.displacement(states, ok)
        self.m5.counts["sets"] += 1
        self.m5.counts["over_bound" if not disp <= MAX_SET_MOTION_M else "within_bound"] += 1
        return disp


def sync_summary(rows: list, counts) -> dict:
    """Session summary of the multi-view sync records (rows' ``sync``): the counts and p50 / p95 / max of the
    reference offset and of every camera's frame age at the step."""
    recs = [r["sync"] for r in rows if isinstance(r.get("sync"), dict)]

    def pct(values):
        v = np.asarray([x for x in values if x is not None], np.float64)
        return None if not len(v) else {"n": int(len(v)), "p50": float(np.percentile(v, 50)),
                                         "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    cams = list(recs[0]["capture_ns"]) if recs else []
    return {"rule": "latest valid frame per camera; mismatch = warning (counted, logged), never a refusal",
            **{k: int(counts.get(k, 0)) for k in SYNC_COUNTS},
            "bound_ms": recs[0]["bound_ms"] if recs else None,
            "max_offset_ms": pct(r["max_offset_ms"] for r in recs),
            "age_ms": {c: pct(r["age_ms"][c] for r in recs) for c in cams},
            "server_mismatch": sum(bool((r.get("server_sync") or {}).get("mismatch")) for r in rows)}
