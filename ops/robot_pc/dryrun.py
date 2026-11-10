"""Live dry run on the robot PC: live cameras (+ read-only robot state) -> policy server -> robot-side validation ->
``DryRunSink`` (logs, never sends). One step per new reference frame (single-view) or per final M5 set (multi-view),
always on the NEWEST observation: older frames are sent once as history but never stepped (no backlog of stale
observations). Capture times are the recorder's (``DeviceClock``), so capture->action includes the camera-to-host
latency (recordings: ~65 ms). ``acquire_ms`` here = device stamp -> frame aligned, preprocessed and queued (newest
frame of the step); in replay it is the replay-only JPEG decode + rs.align emulation."""

from __future__ import annotations

import time

from ops.robot_pc.robot_state import StateUnavailable


def run(client, loop, cameras, state, *, cams: list, steps: int | None = None, duration_s: float | None = None,
        live_m5=None, on_row=None, idle_timeout_s: float = 2.0) -> list:
    rows, fed_to = [], None
    t_end = None if duration_s is None else time.monotonic() + float(duration_s)
    last_frame = dict.fromkeys(cams, time.monotonic())  # per required camera
    last_step = time.monotonic()
    while (steps is None or len(rows) < steps) and (t_end is None or time.monotonic() < t_end):
        newest, acquire = None, 0.0
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
                if live_m5 is not None:
                    live_m5.add_frame(cam, cap["frame_id"], t)
                elif cam == cams[0]:
                    newest = ({cam: cap["frame_id"]}, t, f"{cam}:{cap['frame_id']}")
        if live_m5 is not None:
            now = time.monotonic_ns()
            robot, grip_rows = state.rows(fed_to if fed_to is not None else now - 10**9, now)
            fed_to = now
            for r in robot:
                live_m5.add_robot(*r)
            for g in grip_rows:
                live_m5.add_gripper(*g)
            for s in live_m5.poll():
                disp = s["surface_disp_m"]
                client.add_sync_set(s["frame_ids"], disp if disp != float("inf") else None)  # None: no valid state
                newest = (s["frame_ids"], s["observation_time_ns"], f"set:{s['frame_ids'][cams[0]]}")
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
        row["state_label"] = getattr(state, "label", "")
        rows.append(row)
        if on_row is not None:
            on_row(row)
        if row.get("terminal"):
            break
    return rows
