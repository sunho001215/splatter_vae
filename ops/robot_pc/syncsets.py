"""Multi-view synchronised camera sets on the robot PC (CPU, no torch), ONE implementation for recorded replay
(``replay.run``), the live dry run (``dryrun.run``) and the live session (``live.LiveSession``), so all three hand the
live M5 helper (``s4d.rwlearn.deploy.m5_live.LiveM5``) and the policy client exactly the same data in the same order:

* ``StateFeed``: every robot row (time, q7) and gripper register sample of the state source reaches the helper exactly
  once and in time order, from one second before the first frame on. Each stream remembers the newest time it fed, so
  a live sample stamped before a poll but appended to the reader's buffer after it is fed by the next poll (feeding up
  to a wall-clock bound would drop it for good);
* ``step_sets``: every final set reaches the client with its M5 robot-surface displacement (``add_sync_set``; ``None``
  = no valid robot state, refused by the server as in training) in the order the helper finalises them, each followed
  by a policy step on it - except, with ``newest_only``, sets that already have a newer final set: they are history
  only (counted ``skipped_behind``), the rule of single-view live and realtime replay (no backlog of stale
  observations);
* ``LiveSyncSets``: the two above for a live loop (frames as they are drained, state up to now, the newest step).

Training contract kept unchanged (``s4d.rwlearn.deploy.contracts`` / ``train_store.mutual_sync``): all trained cameras
in TRAINING ORDER (reference = first), mutual-nearest sets, M5 gating by the server's history (moving or stateless sets
are refused there), 3-step history at the trained spacing assembled by the server from the declared sets.
"""

from __future__ import annotations

import numpy as np

STATE_LEAD_NS = 10**9  # rows from one second before the first frame (the recorded replay's start)

# Server answers of a multi-view window refused by the training sync contract (a set in its history moved > 5 mm, had
# no valid robot state, or lacks a mutual partner / evidence): an ANSWER without an action, like training's excluded
# sets - never a route or server failure.
SYNC_REFUSALS = frozenset({"unsynchronized", "incomplete_history"})


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
