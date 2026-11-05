"""Action contract and transitions (s4d/rwlearn/actions.py, transitions.py) on a compact fixture extracted from a real
recording (tests/fixtures/rwlearn/raw_mug_hanging_ep3_actions.npz: mug-hanging episode_000003, two recording segments,
gripper closing and opening; action/robot/gripper/camera-timestamp arrays only) and on synthetic edge cases."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from s4d.rwlearn import actions as A
from s4d.rwlearn import transitions as T

FIXTURE = Path(__file__).parent / "fixtures" / "rwlearn" / "raw_mug_hanging_ep3_actions.npz"


@pytest.fixture(scope="module")
def raw():
    return A.raw_from_arrays(np.load(FIXTURE))


def _tables(raw):
    contract = A.contract_for(raw)
    for serial in raw.serials:
        cam = raw.cameras[serial]
        yield serial, contract, A.step_table(raw, cam["t"], cam["segment"], nominal_period_s=contract.nominal_period_s)


def test_recorded_command_is_the_lead_limited_target(raw):
    """Evidence for the source choice: the sent command stays within max_position_lead of the measured pose on every
    row; rows flagged lead_limited are those where the operator target is ~one lead further away."""
    lead = float(raw.teleop["max_position_lead"])
    d_cmd = np.linalg.norm(raw.command[:, :3] - raw.measured[:, :3], axis=1)
    assert d_cmd.max() <= lead + 1e-6
    d_tgt = np.linalg.norm(raw.target[:, :3] - raw.command[:, :3], axis=1)
    assert np.median(d_tgt[raw.lead_limited]) > 0.75 * lead
    assert np.median(d_tgt[~raw.lead_limited]) < 0.1 * lead
    assert np.allclose(raw.T_flange_tcp, np.eye(4)) and raw.status == "complete"


def test_aggregation_and_reference_follow_the_contract(raw):
    for serial, _contract, tab in _tables(raw):
        cam = raw.cameras[serial]
        for k in np.flatnonzero(tab.transition_ok)[::17]:
            last = np.flatnonzero(raw.robot_t < cam["t"][k + 1])[-1]  # last command strictly before tau_{k+1}
            ref = np.flatnonzero(raw.robot_t <= cam["t"][k])[-1]  # latest measured state at tau_k
            assert raw.robot_t[last] > cam["t"][k]
            np.testing.assert_allclose(tab.command_pose[k], A.pose7_to_matrix(raw.command[last]), atol=1e-12)
            np.testing.assert_allclose(tab.measured_pose[k], A.pose7_to_matrix(raw.measured[ref]), atol=1e-12)
            assert tab.action_phys[k, 6] == raw.gripper_command[last]
        # the pause between the two recording segments is never a transition
        seg = cam["segment"]
        assert not tab.transition_ok[np.flatnonzero(np.diff(seg))].any()
        assert tab.reasons["segment_break"] == len(np.flatnonzero(np.diff(seg)))


def test_round_trip_policy_action_to_recorded_command(raw):
    """normalise(recorded) -> to_robot_command -> exactly the recorded sent command (pose and gripper)."""
    n = 0
    for _serial, contract, tab in _tables(raw):
        ok = np.flatnonzero(tab.transition_ok)
        norm = contract.normalise(tab.action_phys[ok])
        assert np.abs(norm).max() <= 1.0  # every recorded command inside the contract bounds
        out = A.to_robot_command(A.matrix_to_pose7(tab.measured_pose[ok]), norm, contract)
        np.testing.assert_allclose(A.pose7_to_matrix(out["flange_pose"]), tab.command_pose[ok], atol=1e-9)
        np.testing.assert_allclose(out["gripper_closure"], tab.action_phys[ok, 6], atol=1e-12)
        np.testing.assert_array_equal(out["gripper_register"], A.register_from_closure(tab.action_phys[ok, 6]))
        n += len(ok)
    assert n > 600


def test_contract_scales_come_from_the_recorder_limits(raw):
    c = A.contract_for(raw)
    t = raw.teleop
    assert c.nominal_period_s == pytest.approx(1 / 15, rel=0.01)
    assert c.pos_scale_m == pytest.approx(t["max_position_lead"] + t["max_linear_speed"] * 1.5 * c.nominal_period_s)
    assert c.rot_scale_rad == pytest.approx(np.deg2rad(t["max_rotation_lead_deg"])
                                            + t["max_angular_speed"] * 1.5 * c.nominal_period_s)
    assert A.ActionContract.from_json(c.to_json()) == c
    # denormalise(normalise(x)) is the identity inside the bounds; outputs outside [-1, 1] are clipped
    x = np.array([0.03, -0.02, 0.01, 0.1, -0.2, 0.05, 0.7])
    np.testing.assert_allclose(c.denormalise(c.normalise(x)), x, atol=1e-12)
    assert c.denormalise(np.full(7, 3.0))[0] == pytest.approx(c.pos_scale_m)


def test_pose_algebra_and_gripper_mapping():
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation

    ref = np.eye(4)
    ref[:3, :3] = Rotation.random(random_state=1).as_matrix()
    ref[:3, 3] = [0.4, -0.1, 0.5]
    d = np.concatenate([rng.normal(0, 0.02, 3), rng.normal(0, 0.1, 3)])
    goal = A.apply_delta(ref, d)
    np.testing.assert_allclose(A.pose_delta(ref, goal), d, atol=1e-12)
    # the rotation increment is in the BASE frame (left-multiplied), the translation is a base-frame offset
    np.testing.assert_allclose(goal[:3, :3], Rotation.from_rotvec(d[3:]).as_matrix() @ ref[:3, :3], atol=1e-12)
    np.testing.assert_allclose(goal[:3, 3], ref[:3, 3] + d[:3])
    assert A.register_from_closure([0.0, 1.0, 0.5]).tolist() == [3, 230, 116]
    np.testing.assert_allclose(A.closure_from_register([3, 230, 255, 0]), [0, 1, 1, 0])
    R = Rotation.random(random_state=2).as_matrix()
    np.testing.assert_allclose(A.rotation_6d(R), np.r_[R[:, 0], R[:, 1]])


def test_limit_lead_reproduces_the_recorded_clamp(raw):
    """The recorder clamps the operator target to the lead around the measured pose (inferred; checked on rows
    flagged lead_limited, where it reproduces the sent command to a few mm - timing within the 500 Hz loop)."""
    rows = np.flatnonzero(raw.lead_limited)[::25]
    lim = A.limit_lead(raw.measured[rows], raw.target[rows].astype(np.float64), raw.teleop)
    err = np.linalg.norm(lim[:, :3] - raw.command[rows, :3], axis=1)
    assert np.median(err) < 0.005


def _synthetic_raw(times_cam, segment_cam, robot_t, segment_robot, sent=None):
    n = len(robot_t)
    pose = np.tile([0.4, 0.0, 0.5, 0, 0, 0, 1.0], (n, 1))
    pose[:, 0] += np.linspace(0, 0.1, n)
    cmd = pose.copy()
    cmd[:, 0] += 0.02
    return A.RawEpisode(
        path="synthetic", task_name="x", episode_index=0, robot_t=np.asarray(robot_t, np.int64), measured=pose,
        command=cmd, target=cmd, gripper_command=np.linspace(0, 1, n), engaged=np.ones(n, bool),
        sent=np.ones(n, bool) if sent is None else sent, lead_limited=np.zeros(n, bool),
        segment=np.asarray(segment_robot), gripper_t=np.asarray(robot_t, np.int64), gripper_position=np.full(n, 3.0),
        gripper_segment=np.asarray(segment_robot),
        cameras={"c": {"t": np.asarray(times_cam, np.int64), "segment": np.asarray(segment_cam)}},
        segments={"t_start": np.array([0]), "t_end": np.array([robot_t[-1]])},
        teleop={"max_position_lead": 0.04, "max_linear_speed": 0.3, "max_rotation_lead_deg": 20,
                "max_angular_speed": 1.5}, T_flange_tcp=np.eye(4), status="complete")


def test_invalid_steps_dropped_frames_and_unsent_commands():
    ms = 1_000_000
    robot_t = np.arange(0, 1000 * ms, 10 * ms)
    cam_t = np.array([5, 72, 139, 206, 340, 407, 474]) * ms  # frame at 273 ms dropped
    raw = _synthetic_raw(cam_t, np.zeros(7, int), robot_t, np.zeros(len(robot_t), int))
    raw.sent[(robot_t > 407 * ms) & (robot_t < 474 * ms)] = False
    tab = A.step_table(raw, cam_t, np.zeros(7, int), nominal_period_s=0.0667)
    assert tab.transition_ok.tolist() == [True, True, True, False, True, False, False]
    assert tab.reasons["long_step"] == 1 and tab.reasons["not_sent_or_engaged"] == 1
    # a frame before the first robot sample has no state
    tab = A.step_table(raw, np.r_[-3 * ms, cam_t[:3]], np.zeros(4, int), nominal_period_s=0.0667)
    assert not tab.state_ok[0] and tab.reasons["no_state"] == 1
    # a stale gripper register (no sample within 100 ms) is not a current state
    raw.gripper_t = raw.gripper_t[:1]
    raw.gripper_position, raw.gripper_segment = raw.gripper_position[:1], raw.gripper_segment[:1]
    tab = A.step_table(raw, cam_t, np.zeros(7, int), nominal_period_s=0.0667)
    assert tab.state_ok.tolist() == [True, True, False, False, False, False, False]


def test_contract_is_shared_across_episodes_despite_period_jitter(raw):
    import copy

    other = copy.deepcopy(raw)
    for cam in other.cameras.values():
        cam["t"] = cam["t"][0] + ((cam["t"] - cam["t"][0]) * 1.0004).astype(np.int64)  # 0.04 % slower clock
    assert A.contract_for(other) == A.contract_for(raw)
    for cam in other.cameras.values():
        cam["t"] = cam["t"][0] + (cam["t"] - cam["t"][0]) * 2  # 7.5 fps: not the nominal rate
    with pytest.raises(ValueError, match="frame period"):
        A.contract_for(other)


def test_execution_pose_respects_the_radial_lead(raw):
    c = A.contract_for(raw)
    out = A.to_robot_command(raw.measured[100], np.ones(7), c)  # the corner of the normalisation box
    assert np.linalg.norm(out["delta_phys"][:3]) == pytest.approx(np.sqrt(3) * c.pos_scale_m)
    lead = np.linalg.norm(out["execution_flange_pose"][:3] - raw.measured[100, :3])
    assert lead == pytest.approx(c.lead_m)


def test_rewards_masks_and_terminals(raw):
    contract = A.contract_for(raw)
    serial = raw.serials[0]
    cam = raw.cameras[serial]
    from s4d.rwlearn.store import runs_from

    tl = T.Timeline(serial, cam["t"], cam["segment"], runs_from(cam["t"], cam["segment"]),
                    np.arange(len(cam["t"]))[:, None], [serial])
    tr = T.episode_transitions(raw, tl, contract)
    ok = tr["transition_ok"]
    last = np.flatnonzero(ok)[-1]
    assert last + 1 == len(cam["t"]) - 1 and tr["label"] == "success"
    assert tr["success"].sum() == 1 and tr["success"][last + 1]  # the recording's final observation
    assert tr["reward"][last] == 0.0 and tr["mask"][last] == 0.0
    assert (tr["reward"][ok][:-1] == -1).all() and (tr["mask"][ok][:-1] == 1).all()
    # the last transition before the pause bootstraps (not a success) and is a frame-stack boundary
    brk = np.flatnonzero(np.diff(cam["segment"]))[0]
    assert ok[brk - 1] and tr["mask"][brk - 1] == 1 and tr["terminal"][brk - 1] and not ok[brk]
    sparse = T.episode_transitions(raw, tl, contract, reward="sparse_final")
    assert sparse["reward"][last] == 1 and sparse["reward"][ok].sum() == 1
    np.testing.assert_array_equal(sparse["mask"], tr["mask"])
    # a failed demonstration has no success state: its last transition is a truncation (bootstraps)
    failed = T.episode_transitions(raw, tl, contract, success=False)
    assert failed["label"] == "failure" and not failed["success"].any() and failed["mask"][last] == 1
    # an unusable tail never moves the success label earlier
    cut = T.Timeline(serial, cam["t"][:-1], cam["segment"][:-1], tl.run[:-1], tl.frames[:-1], [serial])
    raw_tail = copy.deepcopy(raw)
    raw_tail.sent[raw_tail.robot_t >= cam["t"][-3]] = False
    tail = T.episode_transitions(raw_tail, tl, contract)
    assert tail["label"] == "success_frame_unreachable" and not tail["success"].any()
    sliced = T.episode_transitions(raw, cut, contract)  # a store / sync that dropped the recording's last frame
    assert sliced["label"] == "success_frame_unreachable" and not sliced["success"].any()
    # observation histories restart (pad) at the start of the second segment
    h = T.history_indices(tl.run, np.array([brk + 1, brk + 2, brk + 3]))
    assert h.tolist() == [[brk + 1] * 3, [brk + 1, brk + 1, brk + 2], [brk + 1, brk + 2, brk + 3]]
