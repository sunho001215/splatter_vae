"""Robot geometry and robot-grounded calibration on synthetic fixtures (stand-in geometry, known extrinsic, noise).

Nothing here uses real robot assets or images; it validates the kinematics conventions, the estimator and the
rejection logic only."""

import numpy as np
import pytest
from realworld_calibration_fixtures import (  # noqa: F401
    HW,
    JOINTS,
    LANDMARKS,
    T_TRUE,
    CalibrationFrame,
    K,
    RobotSurface,
    landmarks,
    look_at_T_cam_base,
    make_episode,
    project,
    splat_depth,
)
from scipy.spatial.transform import Rotation

from s4d.realworld.calibration import CalibrationConfig, calibrate_camera, overlay_image
from s4d.realworld.geometry import invert_T, rotation_angle_deg, transform_points
from s4d.realworld.robot import (
    URDFModel,
    gripper_joint_value,
    joint_dict,
    panda_dh_fk,
)
from s4d.realworld.synthetic import synthetic_panda_urdf, synthetic_robot_urdf


def test_urdf_fk_matches_franka_dh_and_mount_conventions():
    arm = URDFModel.from_string(synthetic_panda_urdf())
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-2, 2, 7)
        fk = arm.fk(joint_dict(JOINTS, q))
        dh = panda_dh_fk(q)
        for i in range(1, 9):
            assert np.allclose(fk[f"panda_link{i}"], dh[i - 1], atol=1e-9)
    robot = URDFModel.from_string(synthetic_robot_urdf(mount_yaw=np.pi / 2))
    fk = robot.fk(joint_dict(JOINTS, np.zeros(7), "finger_joint", 0.02))
    T8g = np.linalg.inv(fk["panda_link8"]) @ fk["gripper_base"]
    assert np.allclose(T8g[:3, 3], [0, 0, 0.011])
    assert np.allclose(T8g[:3, :3], Rotation.from_euler("z", np.pi / 2).as_matrix())
    # mimic joint: both fingers close symmetrically by the driven value
    gap = np.linalg.norm(fk["left_finger"][:3, 3] - fk["right_finger"][:3, 3])
    assert np.isclose(gap, 0.085 - 2 * 0.02)


def test_gripper_state_mapping_and_safe_urdf_numbers():
    assert gripper_joint_value(0.5, "normalized_closure", "robotiq_2f_85") == pytest.approx(0.4)
    assert gripper_joint_value(255, "robotiq_position_register", "robotiq_2f_140") == pytest.approx(0.7)
    with pytest.raises(ValueError):
        gripper_joint_value(0.5, "normalized_closure", "unknown")
    m = URDFModel.from_string('<robot name="r"><link name="a"/><link name="b"/><joint name="j" type="fixed">'
                              '<parent link="a"/><child link="b"/><origin xyz="0 0 1" rpy="0 0 pi/2"/></joint></robot>')
    assert np.allclose(m.fk({})["b"][:3, :3], Rotation.from_euler("z", np.pi / 2).as_matrix())
    with pytest.raises(ValueError):
        URDFModel.from_string('<robot name="r"><link name="a"/><link name="b"/><joint name="j" type="fixed">'
                              '<parent link="a"/><child link="b"/><origin xyz="__import__ 0 0"/></joint></robot>')


def test_calibration_recovers_known_extrinsic_with_noise_and_outliers():
    frames = make_episode(T_TRUE)
    res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW))
    assert res.status == "accepted", res.reasons
    err_rot = rotation_angle_deg(res.T_cam_base[:3, :3], T_TRUE[:3, :3])
    err_t = np.linalg.norm(invert_T(res.T_cam_base)[:3, 3] - invert_T(T_TRUE)[:3, 3])  # camera centre error
    assert err_rot < 0.3 and err_t < 0.005, (err_rot, err_t)
    assert res.diagnostics["depth_abs_median_m"] < 0.01
    assert res.diagnostics["heldout_reproj_median_px"] < 2.0
    ov = overlay_image(None, res.T_cam_base, frames[0], K, HW)
    assert ov.shape == (*HW, 3) and (ov[..., 1] == 255).sum() > 50


def test_refinement_improves_a_perturbed_prior_without_landmark_ransac():
    """Depth + reprojection refinement pulls a 2 deg / 3 cm-off prior back (init from the prior only)."""
    frames = make_episode(T_TRUE, outlier_rate=0.0)
    from s4d.realworld import calibration as cal

    prior = T_TRUE.copy()
    prior[:3, :3] = Rotation.from_euler("xyz", [0.02, -0.02, 0.01]).as_matrix() @ prior[:3, :3]
    prior[:3, 3] += [0.02, -0.01, 0.02]
    orig = cal._visual_init
    try:
        cal._visual_init = lambda frames, K, cfg: (None, None, None, None)
        res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW), T_init=prior)
    finally:
        cal._visual_init = orig
    assert res.init_method == "external_prior"
    assert rotation_angle_deg(res.T_cam_base[:3, :3], T_TRUE[:3, :3]) < 0.3
    assert res.status == "accepted", res.reasons


def test_rejects_insufficient_robot_visibility():
    frames = make_episode(T_TRUE, visible_frames={0, 1, 2})
    res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW))
    assert res.status == "rejected"
    assert any("visibility" in r for r in res.reasons)


def test_rejects_poor_motion_coverage():
    frames = make_episode(T_TRUE, static=True)
    res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW))
    assert res.status == "rejected"
    assert any("motion coverage" in r for r in res.reasons)


def test_rejects_when_no_visual_evidence():
    frames = make_episode(T_TRUE, visible_frames=set())
    res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW))
    assert res.status == "rejected" and res.T_cam_base is None
    assert "visual initialisation failed" in res.reasons[0]


def test_robot_mask_from_calibration_respects_occlusion():
    from s4d.realworld.geometry import point_mask

    frames = make_episode(T_TRUE, n=3, outlier_rate=0.0)
    f = frames[1]
    cam = transform_points(T_TRUE, f.surface_base)
    loose = point_mask(cam, K, HW)
    tight = point_mask(cam, K, HW, depth=f.depth, tol_m=0.02)
    assert tight.sum() > 50 and (tight <= loose).all() and tight.sum() < loose.sum()  # hidden samples dropped


def test_rejects_failed_optimisation():
    """A prior 5 deg / 8 cm off and an optimiser budget of 2 evaluations: the solver stops early and the result is
    rejected (never silently accepted)."""
    from s4d.realworld import calibration as cal

    frames = make_episode(T_TRUE, n=10, outlier_rate=0.0)
    prior = T_TRUE.copy()
    prior[:3, :3] = Rotation.from_euler("xyz", [0.05, -0.06, 0.04]).as_matrix() @ prior[:3, :3]
    prior[:3, 3] += [0.05, -0.04, 0.05]
    orig = cal._visual_init
    try:
        cal._visual_init = lambda frames, K, cfg: (None, None, None, None)
        res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW, max_nfev=2, outer_iters=1,
                                                                         min_visible_frames=3), T_init=prior)
    finally:
        cal._visual_init = orig
    assert res.status == "rejected"
    assert any("optimisation failed" in r for r in res.reasons), res.reasons


def test_non_finite_residuals_never_pass_the_gates():
    """Observed landmarks whose model points fall behind the camera give NaN projections; they must count as
    infinitely wrong (rejection), not vanish from medians."""
    frames = make_episode(T_TRUE, n=12, outlier_rate=0.0)
    cam_center = invert_T(T_TRUE)[:3, 3]
    for f in frames[:7]:
        # move every landmark of this frame behind the camera while keeping the 2D observations
        f.landmarks_base = cam_center + (cam_center - f.landmarks_base)
    res = calibrate_camera("rig0/left", frames, K, CalibrationConfig(image_hw=HW, depth_term=False))
    assert res.status == "rejected", res.diagnostics.get("reproj_median_px")
