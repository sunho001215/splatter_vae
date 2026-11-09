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
            "gripper/segment": np.zeros(len(gt), int),
            "cameras": {c: np.arange(0, t[-1], 1 / 15) + 1000.0 + 0.01 * i for i, c in enumerate(CAMS)},
            "task_name": "synthetic"}


START = (0.42, 0.0, 0.57)


def drawer(return_z=0.60, push_torque=None):
    """Synthetic drawer-open: pull 0.30 m at 0.51 m, then a return at ``return_z`` (0.51 = at handle height, i.e.
    pushing the drawer back: contact torque by default, as a real push would show)."""
    wp = [(0, START), (2.5, (0.49, -0.10, 0.52)), (3.0, (0.49, -0.10, 0.51)), (7.0, (0.46, 0.20, 0.51)),
          (8.0, (0.46, 0.20, return_z)), (11.0, (0.48, 0.02, return_z)), (12.0, (0.49, 0.01, return_z))]
    t, xyz = _path(wp)
    tau = np.full((len(t), 7), 0.35)
    tau[(t > 3.0) & (t < 7.0)] = 0.9  # contact while pulling
    if push_torque if push_torque is not None else return_z < 0.53:
        tau[(t > 7.0) & (t < 11.0)] = 0.9
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
    assert f["severity"] == "suspect" and f["evidence"]["travel_m"] > 0.15
    assert set(f["frames"]) == set(CAMS) and all(len(v) == 2 for v in f["frames"].values())
    assert f["t_s"][0] == pytest.approx(7.0, abs=0.05)  # the pull end
    res = O.check_drawer(drawer(push_torque=True))  # lifted, but pressing on something while moving back
    assert "contact_torque_while_moving_back" in {f["code"] for f in res["flags"]}
    # at the front and at handle height, but the torque shows free motion: geometry and torque disagree -> review
    res = O.check_drawer(drawer(return_z=0.51, push_torque=False))
    f = {x["code"]: x for x in res["flags"]}["push_back_at_handle_height"]
    assert f["severity"] == "review" and f["evidence"]["contact_fraction"] < 0.5


def test_drawer_push_back_is_found_after_later_motion_and_after_a_lift():
    """Codex review: (a) open, push back at handle height, then move the disengaged arm beyond the original pull end
    (lifted): the pull end is the FIRST completed pull, so the push-back is still found; (b) open, lift 5 cm, lower back
    to the handle and push 18 cm: every handle-height interval after the pull counts."""
    wp = [(0, START), (2.5, (0.49, -0.10, 0.52)), (3.0, (0.49, -0.10, 0.51)), (7.0, (0.46, 0.20, 0.51)),
          (9.0, (0.46, 0.02, 0.51)), (10.0, (0.46, 0.02, 0.62)), (12.0, (0.46, 0.30, 0.62)), (13.0, (0.46, 0.30, 0.62))]
    t, xyz = _path(wp)
    res = O.check_drawer(signals(t, xyz))
    f = {x["code"]: x for x in res["flags"]}["push_back_at_handle_height"]
    assert f["evidence"]["travel_m"] > 0.15 and res["phases"]["pull_rows"][1] == pytest.approx(700, abs=15)
    wp = [(0, START), (2.5, (0.49, -0.10, 0.52)), (3.0, (0.49, -0.10, 0.51)), (7.0, (0.46, 0.20, 0.51)),
          (7.5, (0.46, 0.20, 0.56)), (8.0, (0.46, 0.20, 0.51)), (10.0, (0.46, 0.02, 0.51)), (11.0, (0.46, 0.02, 0.6))]
    t, xyz = _path(wp)
    res = O.check_drawer(signals(t, xyz))
    f = {x["code"]: x for x in res["flags"]}["push_back_at_handle_height"]
    assert f["evidence"]["travel_m"] > 0.15


def test_mug_partial_release_and_regrasp_under_a_closed_command_is_split():
    """Codex review: the command stays above 0.5 while the fingers open (gOBJ 2 -> 3), the mug is re-grasped at the
    hook (gOBJ 2 again) and carried away: two holds, re-grasp at the release location -> suspect."""
    grasp, hook = (0.51, 0.09, 0.58), (0.47, -0.09, 0.71)
    wp = [(0, START), (3, grasp), (4.5, grasp), (9, hook), (11, hook), (15, grasp), (16, grasp)]
    t, xyz = _path(wp)
    c = (t >= 3.5).astype(float) * 0.9
    c[t >= 15.5] = 0.0

    def gobj(gt):
        g = np.full(len(gt), 3.0)
        g[(gt >= 4.0) & (gt < 9.5)] = 2.0  # held, released at the hook (fingers open: position drops)
        g[(gt >= 10.5) & (gt < 15.5)] = 2.0  # held again
        return g
    sig = signals(t, xyz, command=c, gobj=gobj)
    sig["gripper/position"] = np.where(sig["gripper/object"] == 2, 210.0, 60.0)
    res = O.check_mug(sig)
    assert len(res["phases"]["holding_rows"]) == 2 and len(res["phases"]["closures_rows"]) == 1
    assert "regrasp_at_release_location" in {f["code"] for f in res["flags"]}


def test_mug_hold_merge_rule_on_a_subsecond_gap():
    """A 0.3 s gap between two gOBJ-2 runs: merged when the fingers stay closed (flicker), split when they open;
    each gap is judged against the run right before it (an accepted flicker never lowers the reference)."""
    t, xyz = _path([(0, START), (10, START)])
    sig = signals(t, xyz)
    gt = sig["gripper/t"] - 1000.0
    runs = [(2.0, 4.0), (4.3, 6.0), (6.3, 8.0)]
    sig["gripper/object"] = np.where(np.any([(gt >= a) & (gt < b) for a, b in runs], 0), 2.0, 0.0)
    pos = np.where(sig["gripper/object"] == 2, 210.0, 3.0)
    gap1, gap2 = (gt >= 4.0) & (gt < 4.3), (gt >= 6.0) & (gt < 6.3)
    pos[gap1], pos[gap2] = 208.0, 207.0  # fingers closed in both gaps: one hold
    sig["gripper/position"] = pos
    assert O._holds(sig) == [(int(np.flatnonzero(gt >= 2.0)[0]), int(np.flatnonzero(gt < 8.0)[-1]))]
    pos[gap1] = 191.0  # accepted flicker (within 20 of 210) ...
    pos[gap2] = 175.0  # ... must not lower the reference: 35 below the held 210 is an opening -> split
    sig["gripper/position"] = pos
    holds = O._holds(sig)
    assert len(holds) == 2 and gt[holds[0][1]] < 6.0 <= gt[holds[1][0]] + 0.31
    pos[gap1] = 150.0  # an opening in the first gap splits there too
    assert len(O._holds(sig)) == 3


def test_mug_release_row_is_the_first_sample_after_the_hold():
    res = O.check_mug(mug())
    t = mug()["robot/t"] - 1000.0
    assert t[res["phases"]["release_row"]] == pytest.approx(10.0, abs=0.011)


def test_drawer_normal_variants_are_not_flagged():
    """Codex review: (a) a lift on a plateau then forward motion while raised (pull end stays at 7 s); (b) a pull that
    descends 2 cm while pulling, then a return lifted 4 cm above the final contact height; (c) after lifting, moving
    sideways away from the handle, lowering and returning 10 cm (not at the drawer front: review, not suspect)."""
    base = [(0, START), (2.5, (0.49, -0.10, 0.53)), (3.0, (0.49, -0.10, 0.53))]
    wp = base + [(7.0, (0.46, 0.20, 0.51)), (8.0, (0.46, 0.20, 0.60)), (9.0, (0.46, 0.25, 0.60)),
                 (12.0, (0.48, 0.02, 0.60)), (13.0, (0.49, 0.01, 0.60))]
    res = O.check_drawer(signals(*_path(wp)))
    assert {f["severity"] for f in res["flags"]} == {"info"} and res["phases"]["pull_rows"][1] == pytest.approx(700, 2)
    wp = base + [(7.0, (0.46, 0.20, 0.51)), (8.0, (0.46, 0.20, 0.55)), (11.0, (0.48, 0.02, 0.55)),
                 (12.0, (0.49, 0.01, 0.55))]
    res = O.check_drawer(signals(*_path(wp)))
    assert {f["severity"] for f in res["flags"]} == {"info"}
    wp = base + [(7.0, (0.46, 0.20, 0.51)), (8.0, (0.46, 0.20, 0.60)), (9.0, (0.60, 0.20, 0.60)),
                 (10.0, (0.60, 0.20, 0.51)), (12.0, (0.60, 0.10, 0.51)), (13.0, (0.60, 0.10, 0.6))]
    res = O.check_drawer(signals(*_path(wp)))
    assert {f["code"] for f in res["flags"] if f["severity"] != "info"} == set()  # sideways: outside the corridor
    wp = base + [(7.0, (0.46, 0.20, 0.51)), (8.0, (0.46, 0.20, 0.60)), (9.0, (0.46, 0.10, 0.60)),
                 (10.0, (0.46, 0.10, 0.51)), (12.0, (0.46, 0.0, 0.51)), (13.0, (0.46, 0.0, 0.6))]
    res = O.check_drawer(signals(*_path(wp)))
    f = {x["code"]: x for x in res["flags"]}
    assert f["low_backward_motion_away_from_front"]["severity"] == "review"
    assert "push_back_at_handle_height" not in f


def test_drawer_diagonal_lift_and_motion_beyond_the_front_are_not_push_backs():
    """Codex review 3: (a) the lift also advances 2 cm (diagonal): the pull end stays at the last level progress;
    (b) after lifting, lowering 12 cm BEYOND the open front and moving back 6 cm (never reaching the front)."""
    base = [(0, START), (2.5, (0.49, -0.10, 0.52)), (3.0, (0.49, -0.10, 0.51)), (7.0, (0.46, 0.20, 0.51))]
    wp = base + [(8.0, (0.46, 0.22, 0.60)), (11.0, (0.48, 0.02, 0.60)), (12.0, (0.49, 0.01, 0.60))]
    res = O.check_drawer(signals(*_path(wp)))
    assert {f["severity"] for f in res["flags"]} == {"info"}
    assert res["phases"]["pull_rows"][1] <= 740  # at most the part of the lift within the 3 cm clearance
    assert res["phases"]["contact_height_z_m"] == pytest.approx(0.51, abs=0.005)
    wp = base + [(8.0, (0.46, 0.20, 0.60)), (9.0, (0.46, 0.32, 0.60)), (10.0, (0.46, 0.32, 0.51)),
                 (11.0, (0.46, 0.26, 0.51)), (12.0, (0.46, 0.26, 0.60))]
    res = O.check_drawer(signals(*_path(wp)))
    assert {f["severity"] for f in res["flags"]} <= {"info"}
    # the same lowering, but moving back 6 cm PAST the front at handle height: a push of 6 - 12 + ... = from 0.32 to
    # 0.14 is 6 cm behind the front -> suspect
    wp = base + [(8.0, (0.46, 0.20, 0.60)), (9.0, (0.46, 0.32, 0.60)), (10.0, (0.46, 0.32, 0.51)),
                 (12.0, (0.46, 0.14, 0.51)), (13.0, (0.46, 0.14, 0.60))]
    res = O.check_drawer(signals(*_path(wp)))
    f = {x["code"]: x for x in res["flags"]}["push_back_at_handle_height"]
    assert f["severity"] == "suspect" and f["evidence"]["travel_m"] == pytest.approx(0.06, abs=0.003)


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
    assert len(res["phases"]["holding_rows"]) == 1


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
    assert doc["episodes"]["drawer-open/episode_000002"]["ending_flags"] == ["push_back_at_handle_height",
                                                                         "contact_torque_while_moving_back"]
    assert O.load_outcomes(doc) == {ep: True for ep in rep["episodes"]}
    assert O.load_outcomes({"a/b": True}) == {"a/b": True}
    with pytest.raises(ValueError, match="booleans"):
        O.load_outcomes({"a/b": "true"})
    bad = {**doc, "episodes": {"x/y": {"outcome": "maybe"}}}
    with pytest.raises(ValueError, match="neither"):
        O.load_outcomes(bad)


# Runs in the main checkout with /home/ws/data, alongside the URDF used by M5 validation.
if ((RAW / "drawer-open" / "episode_000000.h5").is_file()
        and (RAW / "mug-hanging" / "episode_000001.h5").is_file()):
    def test_real_recordings_read_only_smoke():
        d = O.check_episode(RAW / "drawer-open" / "episode_000000.h5", "drawer-open")
        m = O.check_episode(RAW / "mug-hanging" / "episode_000001.h5", "mug-hanging")
        assert d["decision"] == "no_reversal" and m["decision"] == "no_reversal"
        assert d["outcome"] == m["outcome"] == "success"
        assert 0.25 < d["phases"]["pull_m"] < 0.40 and d["phases"]["axis_angle_from_base_y_deg"] < 20


def transport(task='covering', *, reset=False, revisit=False):
    """One release, optionally followed by taking the cloth/object back or revisiting it open."""
    grasp, destination = (0.40, 0.0, 0.48), (0.50, -0.20, 0.53)
    if task == 'pick-and-place':
        destination = (0.58, 0.10, 0.50)
    wp = [(0, START), (3, grasp), (4, grasp), (7, destination), (8.2, destination), (9, START)]
    wp += [(9.5, destination), (10, destination), (12, grasp), (13, START)] if reset else (
        [(11, destination), (12, START)] if revisit else [(12, START)])
    t, xyz = _path(wp)
    c = np.where((t >= 3.5) & (t < 8.2), 0.9, 0.0)
    if reset:
        c[(t >= 9.5) & (t < 12)] = 0.9

    def gobj(gt):
        held = (gt >= 4) & (gt < 8)
        if reset:
            held |= (gt >= 9.6) & (gt < 12)
        return np.where(held, 2.0, 3.0)

    sig = signals(t, xyz, command=c, gobj=gobj)
    sig['gripper/position'] = np.where(sig['gripper/object'] == 2, 180.0, 3.0)
    return sig


@pytest.mark.parametrize('task', ['covering', 'pick-and-place'])
def test_new_task_release_and_open_retreat_preserve_success(task):
    result = O.CHECKS[task](transport(task))
    assert not [f for f in result['flags'] if f['severity'] != 'info']
    assert len(result['phases']['holding_rows']) == 1 and result['phases']['transport_xy_m'] > 0.15
    assert result['thresholds_provisional'] == (task == 'pick-and-place')
    ep = task + '/episode_000000'
    report = {'episodes': {ep: {'decision': 'no_reversal', **result}}}
    assert O.load_outcomes(O.outcome_labels([ep], ending_report=report)) == {ep: True}


@pytest.mark.parametrize('task', ['covering', 'pick-and-place'])
def test_new_task_reset_regrasps_first_release_and_never_relabels(task):
    sig = transport(task, reset=True)
    result = O.CHECKS[task](sig)
    f = next(f for f in result['flags'] if f['code'] == 'regrasp_at_release_location')
    assert f['severity'] == 'suspect' and set(f['frames']) == set(CAMS)
    assert sig['robot/t'][result['phases']['release_row']] - 1000 == pytest.approx(8, abs=0.011)
    ep = task + '/reset'
    report = {'episodes': {ep: {'decision': 'reversal_suspected', **result}}}
    labels = O.outcome_labels([ep], ending_report=report)
    assert labels['episodes'][ep]['outcome'] == 'success' and not labels['flags_change_labels']
    assert 'regrasp_at_release_location' in labels['episodes'][ep]['ending_flags']


@pytest.mark.parametrize('task', ['covering', 'pick-and-place'])
def test_new_task_holding_or_closed_end_and_ambiguous_return_are_flagged(task):
    sig = transport(task)
    sig['action/gripper'][-10:] = 0.9
    sig['gripper/object'][-5:] = 2
    sig['gripper/position'][-5:] = 180
    result = O.CHECKS[task](sig)
    codes = {f['code']: f['severity'] for f in result['flags']}
    assert codes['ends_closed'] == codes['holding_at_end'] == 'suspect'
    result = O.CHECKS[task](transport(task, revisit=True))
    assert any(f['code'] == 'revisit_release_location' and f['severity'] == 'review' for f in result['flags'])


@pytest.mark.parametrize('task', ['covering', 'pick-and-place'])
def test_new_task_missing_grasp_short_transport_and_unobserved_release_need_review(task):
    sig = transport(task)
    sig['gripper/object'][:] = 3
    assert 'no_grasp' in {f['code'] for f in O.CHECKS[task](sig)['flags']}
    sig = transport(task)
    sig['robot/flange_pose'][:, :3] = START
    assert 'short_transport' in {f['code'] for f in O.CHECKS[task](sig)['flags']}
    sig = transport(task)
    sig['gripper/segment'][sig['gripper/t'] >= 1008] = 1
    result = O.CHECKS[task](sig)
    assert any(f['code'] == 'release_across_pause' and f['severity'] == 'review' for f in result['flags'])
    sig = transport(task)
    sig['gripper/position'][sig['gripper/t'] >= 1008] = 180
    assert 'release_not_confirmed_open' in {f['code'] for f in O.CHECKS[task](sig)['flags']}


def test_outcome_scan_defaults_use_the_authoritative_task_registry(tmp_path):
    from s4d.rwlearn.tasks import TARGETS

    report = O.scan(tmp_path)
    assert tuple(report['counts']) == TARGETS and not report['episodes']
    assert tuple(O.CHECKS) == TARGETS
