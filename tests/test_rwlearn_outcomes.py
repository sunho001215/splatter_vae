"""Outcome labels and the ending (reset / reversal) check (s4d/rwlearn/outcomes.py) on synthetic robot signals shaped
like the recordings (100 Hz flange pose, gripper command + Robotiq gOBJ, external torque, 15 Hz camera frames), plus
a read-only smoke run on two real recordings when the raw data are present (local host only)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from s4d.rwlearn import outcomes as O

RAW = Path("/home/ws/data/real_world")
CAMS = ("135122070988", "135122073201", "135122076054")


def _path(waypoints, hz=100):
    """Piecewise-linear flange trajectory through (t_s, xyz) waypoints."""
    t = np.arange(0, waypoints[-1][0], 1 / hz)
    ts = np.array([w[0] for w in waypoints])
    xyz = np.stack([np.interp(t, ts, [w[1][i] for w in waypoints]) for i in range(3)], 1)
    return t, xyz


def signals(t, xyz, command=None, gobj=None, tau=None):
    n = len(t)
    pose = np.concatenate([xyz, np.tile([1.0, 0, 0, 0], (n, 1))], 1)
    tau = np.full((n, 7), 0.35) if tau is None else tau
    gt = t[::2]
    return {"robot/t": t + 1000.0, "robot/flange_pose": pose, "robot/tau_ext": tau,
            "action/gripper": np.zeros(n) if command is None else command, "action/segment": np.zeros(n, int),
            "gripper/t": gt + 1000.0, "gripper/position": np.full(len(gt), 3.0),
            "gripper/object": np.full(len(gt), 3.0) if gobj is None else gobj(gt),
            "cameras": {c: np.arange(0, t[-1], 1 / 15) + 1000.0 + 0.01 * i for i, c in enumerate(CAMS)},
            "task_name": "synthetic"}


START = (0.42, 0.0, 0.57)


def drawer(return_z=0.60, push_torque=False):
    wp = [(0, START), (2.5, (0.49, -0.10, 0.52)), (3.0, (0.49, -0.10, 0.51)), (7.0, (0.46, 0.20, 0.51)),
          (8.0, (0.46, 0.20, return_z)), (11.0, (0.48, 0.02, return_z)), (12.0, (0.49, 0.01, return_z))]
    t, xyz = _path(wp)
    tau = np.full((len(t), 7), 0.35)
    tau[(t > 3.0) & (t < 7.0)] = 0.9  # contact while pulling
    if push_torque:
        tau[(t > 8.0) & (t < 11.0)] = 0.9
    return signals(t, xyz, tau=tau)


def test_drawer_return_lifted_off_the_handle_is_not_a_reversal():
    res = O.check_drawer(drawer())
    assert [f["code"] for f in res["flags"]] == ["return_toward_start"]
    assert res["flags"][0]["severity"] == "info"
    ph = res["phases"]
    assert ph["pull_m"] == pytest.approx(0.30, abs=0.01) and ph["back_at_handle_m"] < 0.01
    assert ph["torque_decides"] and ph["backward_contact_fraction"] < 0.1


def test_drawer_pushed_back_at_handle_height_is_flagged_with_frames():
    res = O.check_drawer(drawer(return_z=0.51))
    codes = {f["code"]: f for f in res["flags"]}
    f = codes["push_back_at_handle_height"]
    assert f["severity"] == "suspect" and f["evidence"]["back_at_handle_m"] > 0.15
    assert set(f["frames"]) == set(CAMS) and all(len(v) == 2 for v in f["frames"].values())
    assert f["t_s"][0] == pytest.approx(7.0, abs=0.05)  # the pull end
    res = O.check_drawer(drawer(push_torque=True))  # lifted, but pressing on something while moving back
    assert "contact_torque_while_moving_back" in {f["code"] for f in res["flags"]}


def test_drawer_without_a_pull_needs_review():
    t, xyz = _path([(0, START), (5, (0.45, 0.02, 0.55)), (10, START)])
    res = O.check_drawer(signals(t, xyz))
    assert res["flags"][0]["code"] == "pull_not_found" and res["flags"][0]["severity"] == "review"


def mug(regrasp=False, end_closed=False, reclose=False, revisit=False):
    grasp, hook = (0.51, 0.09, 0.58), (0.47, -0.09, 0.71)
    wp = [(0, START), (3, grasp), (4.5, grasp), (9, hook), (10, hook), (13, (0.48, 0.04, 0.55)), (16, (0.47, 0.045, 0.54))]
    if revisit:
        wp = wp[:5] + [(12, (0.47, 0.02, 0.71)), (13.5, hook), (16, (0.48, 0.04, 0.55))]
    if regrasp:
        wp = wp[:5] + [(11, (0.47, 0.0, 0.71)), (12, hook), (13, hook), (16, grasp)]
    t, xyz = _path(wp)
    c = ((t >= 3.5) & (t < 10.0)).astype(float)
    holding = [(4.0, 10.0)]
    if regrasp:
        c[(t >= 12.5) & (t < 15.0)] = 1.0
        holding.append((13.0, 15.0))
    if reclose:
        c[(t >= 12.0) & (t < 12.2)] = 1.0
    if end_closed:
        c[t >= 15.0] = 1.0

    def gobj(gt):
        g = np.full(len(gt), 3.0)
        for a, b in holding:
            g[(gt >= a) & (gt < b)] = 2.0  # gt: seconds from the start
        return g
    return signals(t, xyz, command=c, gobj=gobj)


def test_mug_hang_release_and_return_is_not_a_reversal():
    res = O.check_mug(mug())
    assert {f["severity"] for f in res["flags"]} <= {"info"}
    assert len(res["phases"]["holding_closures_rows"]) == 1


@pytest.mark.parametrize("kw,code,severity", [
    ({"regrasp": True}, "regrasp_at_release_location", "suspect"),
    ({"end_closed": True}, "ends_closed", "suspect"),
    ({"reclose": True}, "reclose_without_object", "info"),
    ({"revisit": True}, "revisit_release_location", "review"),
])
def test_mug_endings(kw, code, severity):
    res = O.check_mug(mug(**kw))
    f = {x["code"]: x for x in res["flags"]}[code]
    assert f["severity"] == severity and f["frames"] and f["robot_rows"]


def _write_raw(path: Path, sig: dict):
    with h5py.File(path, "w") as f:
        for k in O._ROBOT + O._GRIPPER:
            f.create_dataset(k, data=sig[k])
        for c, t in sig["cameras"].items():
            f.create_dataset(f"cameras/cam_{c}/t", data=t)
        f.attrs["task_name"] = "synthetic"


def test_scan_decisions_labels_and_report(tmp_path):
    (tmp_path / "drawer-open").mkdir()
    (tmp_path / "mug-hanging").mkdir()
    _write_raw(tmp_path / "drawer-open" / "episode_000001.h5", drawer())
    _write_raw(tmp_path / "drawer-open" / "episode_000002.h5", drawer(return_z=0.51))
    _write_raw(tmp_path / "mug-hanging" / "episode_000003.h5", mug(revisit=True))
    rep = O.scan(tmp_path)
    assert rep["counts"]["drawer-open"] == {"episodes": 2, "no_reversal": 1, "reversal_suspected": 1,
                                            "needs_image_review": 0, "info_flags": {"return_toward_start": 2}}
    assert rep["counts"]["mug-hanging"]["needs_image_review"] == 1
    flagged = O.flagged(rep)
    assert set(flagged) == {"drawer-open/episode_000002", "mug-hanging/episode_000003"}
    # flags never change the label: every episode stays a success
    doc = O.outcome_labels(sorted(rep["episodes"]), ending_report=rep)
    assert all(v["outcome"] == "success" for v in doc["episodes"].values()) and doc["flags_change_labels"] is False
    assert doc["episodes"]["drawer-open/episode_000002"]["ending_flags"] == ["push_back_at_handle_height"]
    assert O.load_outcomes(doc) == {ep: True for ep in rep["episodes"]}
    assert O.load_outcomes({"a/b": True}) == {"a/b": True}
    with pytest.raises(ValueError, match="booleans"):
        O.load_outcomes({"a/b": "true"})
    bad = {**doc, "episodes": {"x/y": {"outcome": "maybe"}}}
    with pytest.raises(ValueError, match="neither"):
        O.load_outcomes(bad)


@pytest.mark.skipif(not (RAW / "drawer-open" / "episode_000000.h5").is_file()
                    or not (RAW / "mug-hanging" / "episode_000001.h5").is_file(), reason="raw recordings not here")
def test_real_recordings_read_only_smoke():
    d = O.check_episode(RAW / "drawer-open" / "episode_000000.h5", "drawer-open")
    m = O.check_episode(RAW / "mug-hanging" / "episode_000001.h5", "mug-hanging")
    assert d["decision"] == "no_reversal" and m["decision"] == "no_reversal"
    assert d["outcome"] == m["outcome"] == "success"
    assert 0.25 < d["phases"]["pull_m"] < 0.40 and d["phases"]["axis_angle_from_base_y_deg"] < 20
