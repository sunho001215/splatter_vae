"""Following detection (ops/robot_pc/following.py) on the REAL test-3 session and synthetic stalls (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ops.robot_pc.following import FollowMonitor
from ops.robot_pc.poses import rotate_left

# per executed tick of inference_20261110_211938 (real hardware, 2026-11-10 21:19, e2e_cnn-drawer-open-s0, profile slow):
# t_ns, sent setpoint flange pose7, latest measured flange pose7 (exported read-only from the session H5)
DATA = Path(__file__).parent / "data" / "robot_pc" / "follow_inference_20261110_211938.npz"


def _load():
    z = np.load(DATA)
    return z["t_ns"], z["setpoint"], z["measured"], json.loads(str(z["executor"]))


def _monitor(ex):
    return FollowMonitor(lead_m=ex["lead_m"], lead_rad=np.radians(ex["lead_rot_deg"]), max_speed=ex["max_linear_speed"],
                         max_ang_speed=ex["max_angular_speed"])


def _run(mon, t, sp, m):
    for k in range(len(t)):
        reason = mon.update(t[k], sp[k], m[k])
        if reason:
            return (t[k] - t[0]) / 1e9, reason
    return None, None


def _extend(t, sp, m, seconds):
    n = int(seconds / 0.002)
    return (np.r_[t, t[-1] + np.arange(1, n + 1) * 2_000_000], np.r_[sp, np.repeat(sp[-1:], n, 0)],
            np.r_[m, np.repeat(m[-1:], n, 0)])


def test_real_session_old_rule_aborted_new_rule_does_not():
    t, sp, m, ex = _load()
    assert len(t) == 640 and ex["lead_m"] == 0.02
    err = np.linalg.norm(sp[:, :3] - m[:, :3], axis=1)
    assert 0.018 < err.max() < ex["lead_m"]  # the error sat at the lead (leash) - normal impedance lag
    since, old_stop = None, None  # the rule that ran on the hardware: err >= 0.5 lead for > 1 s
    for k in range(len(t)):
        since = (since or t[k]) if err[k] >= 0.5 * ex["lead_m"] else None
        if since is not None and t[k] - since > 1_000_000_000:
            old_stop = (t[k] - t[0]) / 1e9
            break
    assert old_stop is not None and 1.2 < old_stop < 1.3
    mon = _monitor(ex)
    assert _run(mon, t, sp, m) == (None, None)
    assert mon.robot_moved


def test_real_session_stalled_robot_is_caught():
    t, sp, m, ex = _load()
    k0 = int(np.searchsorted(t, t[0] + 600_000_000))
    m2, sp2 = m.copy(), sp.copy()
    m2[k0:], sp2[k0:] = m[k0], sp[k0]  # the robot stops at 0.6 s; the leash holds the setpoint at its lead
    at, reason = _run(_monitor(ex), *_extend(t, sp2, m2, 2.0))
    assert reason.startswith("robot not progressing toward the command: position error") and 1.5 < at < 1.8


def test_mode_not_accepted_robot_never_moves():
    t, sp, m, ex = _load()
    at, reason = _run(_monitor(ex), t, sp, np.repeat(m[:1], len(t), 0))
    assert reason.startswith("robot has not moved since engagement") and reason.endswith("mode not accepted?")
    assert 1.0 < at < 1.3


def test_rotation_stall_caught_and_rotation_lag_tolerated():
    _, _, m, ex = _load()
    n = 2000
    t = m[:1, 0].astype(np.int64) * 0 + np.arange(n, dtype=np.int64) * 2_000_000
    q0 = m[0, 3:]
    lead = ex["lead_rot_deg"]
    meas = np.repeat(m[:1], n, 0)
    meas[:, 0] += np.minimum(0.01, np.arange(n) * 0.00005)  # the robot translates (no mode problem)

    def sp_rot(deg_of_k):
        # position: the setpoint tracks the measured position exactly; only the orientation lags
        return np.array([np.r_[meas[k, :3], rotate_left(q0, np.array([0.0, 0.0, np.radians(deg_of_k(k))]))]
                         for k in range(n)])

    # orientation pinned at the leash (7.9 deg of 8) without progress -> stop
    at, reason = _run(_monitor(ex), t, sp_rot(lambda k: min(lead - 0.1, 0.05 * k)), meas)
    assert reason is not None and "commanded orientation" in reason
    # orientation error drifting at ~5 deg without progress (as observed in test 3) -> tolerated
    at, reason = _run(_monitor(ex), t, sp_rot(lambda k: min(5.0, 0.05 * k)), meas)
    assert reason is None


def test_expired_command_pause_restarts_the_interval():
    _, _, m, ex = _load()
    mon = _monitor(ex)
    sp = m[0].copy()
    sp[0] += 0.015
    for k in range(400):  # 0.8 s without motion, command 15 mm away -> not yet
        assert mon.update(k * 2_000_000, sp, m[0]) is None
    mon.pause()
    for k in range(400, 800):
        assert mon.update(k * 2_000_000, sp, m[0]) is None


@pytest.mark.parametrize("v_frac", [0.5, 1.0])
def test_slow_but_progressing_robot_is_not_stopped(v_frac):
    _, _, m, ex = _load()
    mon = _monitor(ex)
    v = v_frac * 0.1 * ex["max_linear_speed"] * 2  # 0.005-0.01 m/s toward the setpoint, error held at 15 mm
    for k in range(1500):
        meas = m[0].copy()
        meas[0] += v * k * 0.002
        sp = meas.copy()
        sp[0] += 0.015
        assert mon.update(k * 2_000_000, sp, meas) is None
