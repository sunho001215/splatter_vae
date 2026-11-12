"""Live dry run on the robot PC: live cameras (+ read-only robot state) -> policy server -> robot-side validation ->
``DryRunSink`` (logs, never sends). One step per new reference frame (single-view) or per final M5 set (multi-view,
``syncsets.LiveSyncSets``: every final set reaches the server with its displacement), always on the NEWEST observation:
older frames / sets are sent once as history but never stepped (no backlog of stale observations). Capture times are
the recorder's (``DeviceClock``), so capture->action includes the camera-to-host latency (recordings: ~65 ms).
``acquire_ms`` here = device stamp -> frame aligned, preprocessed and queued (newest frame of the step); in replay it
is the replay-only JPEG decode + rs.align emulation."""

from __future__ import annotations

import time

from ops.robot_pc.robot_state import StateUnavailable


def run(client, loop, cameras, state, *, cams: list, steps: int | None = None, duration_s: float | None = None,
        live_m5=None, on_row=None, idle_timeout_s: float = 2.0) -> list:
    from ops.robot_pc.syncsets import LiveSyncSets

    rows = []
    sync = None if live_m5 is None else LiveSyncSets(client, live_m5, state, cams)
    t_end = None if duration_s is None else time.monotonic() + float(duration_s)
    last_frame = dict.fromkeys(cams, time.monotonic())  # per required camera
    last_step = time.monotonic()
    while (steps is None or len(rows) < steps) and (t_end is None or time.monotonic() < t_end):
        newest, acquire, extra = None, 0.0, {}
        for cam in cams:
            for cap in cameras.drain(cam):
                last_frame[cam] = time.monotonic()
                t = cap["capture_ns"]
                try:
                    fs = state.bound_state(t)
                    pose, grip = fs.pose7, fs.gripper_register
                except StateUnavailable:
                    pose, grip = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.0
                client.add_capture(cam, cap, pose7=pose, gripper_register=grip,
                                   metadata={"state_source": state.kind})
                acquire = max(acquire, (cap["arrival_ns"] - cap["capture_ns"]) / 1e6)  # device stamp -> processed
                if sync is not None:
                    sync.add_frame(cam, cap["frame_id"], t)
                elif cam == cams[0]:
                    newest = ({cam: cap["frame_id"]}, t, f"{cam}:{cap['frame_id']}")
        if sync is not None:  # every final set reaches the server; the step is on the newest one
            s = sync.newest_step(time.monotonic_ns(), loop.counts)
            if s is not None:
                newest, extra = s[:3], {"surface_disp_m": s[3]}
        now_s = time.monotonic()
        stale = [c for c in cams if now_s - last_frame[c] > idle_timeout_s]
        if stale or now_s - last_step > 3 * idle_timeout_s:
            raise RuntimeError(f"stopped: cameras {stale} sent no frame for {idle_timeout_s} s / no step for "
                               f"{3 * idle_timeout_s} s ({dict(cameras.errors)})")
        if newest is None:
            time.sleep(0.001)
            continue
        last_step = now_s
        row = loop.step(*newest, acquire_ms=acquire)
        row.update(extra)
        row["state_label"] = getattr(state, "label", "")
        rows.append(row)
        if on_row is not None:
            on_row(row)
        if row.get("terminal"):
            break
    return rows
