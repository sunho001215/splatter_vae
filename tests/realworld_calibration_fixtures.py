"""Deterministic synthetic robot episodes shared by calibration and RGB-D tests."""

import numpy as np

from s4d.realworld.calibration import CalibrationFrame
from s4d.realworld.geometry import project, transform_points
from s4d.realworld.robot import RobotSurface, URDFModel, joint_dict, landmarks
from s4d.realworld.synthetic import look_at_T_cam_base, splat_depth, synthetic_robot_urdf

JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
LANDMARKS = ["panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link7", "panda_link8",
             "gripper_base", "left_finger", "right_finger"]
HW = (240, 320)
K = np.array([[280.0, 0, 161.0], [0, 280.0, 119.0], [0, 0, 1]])


def make_episode(T_cam_base, *, n=25, static=False, visible_frames=None, px_noise=0.5, depth_noise=0.003, seed=0,
                 outlier_rate=0.05):
    rng = np.random.default_rng(seed)
    model = URDFModel.from_string(synthetic_robot_urdf())
    surface = RobotSurface.build(model, points_per_link=150, seed=0)
    dense = RobotSurface.build(model, points_per_link=40000, seed=1)
    q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
    frames = []
    for t in range(n):
        phase = 0.0 if static else t / n * 2 * np.pi
        q = q0 + 0.45 * np.sin(phase + np.arange(7)) * (0 if static else 1)
        grip = 0.0425 * (0.5 + 0.5 * np.sin(phase))
        qd = joint_dict(JOINTS, q, "finger_joint", grip)
        P_dense, _ = dense.points(qd)
        depth = splat_depth(transform_points(T_cam_base, P_dense), K, HW, background=3.0)
        depth = np.where(depth > 0, depth + rng.normal(0, depth_noise, depth.shape), 0).astype(np.float32)
        L = landmarks(model, qd, LANDMARKS)
        uv, z = project(transform_points(T_cam_base, L), K)
        uv = uv + rng.normal(0, px_noise, uv.shape)
        out = rng.random(len(uv)) < outlier_rate
        uv[out] += rng.normal(0, 40, (out.sum(), 2))
        if visible_frames is not None and t not in visible_frames:
            uv[:] = np.nan
        conf = np.ones(len(L))
        surf, _ = surface.points(qd)
        fk = model.fk(qd)
        frames.append(CalibrationFrame(t, 1_000_000 * t, L, uv, conf, surf, fk["panda_link8"][:3, 3], depth, None))
    return frames


# Preserve callable metadata and pickle references through the legacy export.
make_episode.__module__ = "test_realworld_calibration"

T_TRUE = look_at_T_cam_base(eye=[1.3, 0.7, 0.8], target=[0.35, 0.0, 0.45])
