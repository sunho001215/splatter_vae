"""Conservative limits, hardware-test authorization and the hardware-test launcher (fakes only; nothing real)."""

from __future__ import annotations

import pytest

from ops.robot_pc import conservative
from ops.robot_pc.safety import ExecutionAuthorization, ExecutionRefused, Mode, authorize_execution, \
    authorize_hardware_test
from ops.robot_pc.validation import ActionLimits

BASE = {"executor": {"rate_hz": 500, "max_linear_speed": 0.3, "max_angular_speed": 1.5, "max_linear_accel": 4.0,
                     "max_angular_accel": 15.0, "lead_m": 0.04, "lead_rot_deg": 20.0, "state_timeout_s": 0.05,
                     "frozen_state_s": 0.25}}


def test_tighten_only_lowers():
    cfg, applied = conservative.tighten(BASE, "slow")
    ex = cfg["executor"]
    assert ex["max_linear_speed"] == 0.05 and ex["max_angular_speed"] == 0.25 and ex["lead_m"] == 0.02
    assert ex["rate_hz"] == 500 and ex["state_timeout_s"] == 0.05  # untouched
    assert BASE["executor"]["max_linear_speed"] == 0.3  # input not mutated
    assert applied["max_linear_speed"] == (0.3, 0.05)


def test_tighten_never_loosens():
    strict = {"executor": {**BASE["executor"], "max_linear_speed": 0.01, "lead_m": 0.005}}
    cfg, _ = conservative.tighten(strict, "slow")
    assert cfg["executor"]["max_linear_speed"] == 0.01 and cfg["executor"]["lead_m"] == 0.005


def test_tighten_unknown_profile():
    with pytest.raises(ValueError):
        conservative.tighten(BASE, "fast")


def test_test_box_intersects():
    lim = ActionLimits.from_config({"workspace_min_m": [0.257, -0.242, 0.425], "workspace_max_m": [0.621, 0.321, 0.694]})
    box = conservative.test_box(lim, [0.42, 0.0, 0.66, 0, 0, 0, 1], 0.08)
    assert box.workspace_min_m == pytest.approx((0.34, -0.08, 0.58))
    assert box.workspace_max_m == pytest.approx((0.50, 0.08, 0.694))  # clipped by the task box
    assert box.max_translation_m == lim.max_translation_m
    with pytest.raises(ValueError):
        conservative.test_box(lim, [0.9, 0.0, 0.6, 0, 0, 0, 1], 0.08)  # start outside the task box


def test_gripper_closures_bounded():
    assert conservative.gripper_closures([0.3, 0.0]) == [0.3, 0.0]
    for bad in ([0.8], [-0.1], [], [float("nan")]):
        with pytest.raises(ValueError):
            conservative.gripper_closures(bad)


def test_hardware_test_authorization():
    a = authorize_hardware_test(test="hold", cli_flag=True, operator_start=True)
    assert a.mode is Mode.EXECUTE and a.hardware_test == "hold"
    for kw in ({"test": "rollout", "cli_flag": True, "operator_start": True},
               {"test": "gripper", "cli_flag": False, "operator_start": True},
               {"test": "gripper", "cli_flag": True, "operator_start": False}):
        with pytest.raises(ExecutionRefused):
            authorize_hardware_test(**kw)
    with pytest.raises(ExecutionRefused):  # forged
        ExecutionAuthorization(Mode.EXECUTE, 0, (), None, hardware_test="hold")
    p = authorize_execution(cli_execute=True, operator_start=True, policy_info={"execution_eligible": True})
    assert p.hardware_test is None
    with pytest.raises(ExecutionRefused):
        authorize_execution(cli_execute=True, operator_start=True, policy_info={})
