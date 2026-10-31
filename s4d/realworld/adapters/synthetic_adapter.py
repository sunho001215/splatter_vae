"""Fully synthetic episodes (ray-cast boxes, two metric infrared stereo rigs + one rig with missing stereo
geometry, a stand-in Panda/gripper URDF) for end-to-end tests. Nothing here is real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from s4d.realworld.adapters.base import EpisodeAdapter, register_adapter
from s4d.realworld.contract import (
    CameraStream,
    EpisodeMeta,
    Intrinsics,
    Provenance,
    RobotDescription,
    RobotTrajectory,
    StereoGeometry,
)
from s4d.realworld.geometry import make_T, transform_points
from s4d.realworld.synthetic import look_at_T_cam_base, render_box_scene, synthetic_robot_urdf

RIG_EYES = {"rig0": (1.25, 0.55, 0.65), "rig1": (1.25, -0.6, 0.6), "rig2": (0.55, 0.05, 1.2)}
TARGET = (0.5, 0.0, 0.1)


@register_adapter
class SyntheticAdapter(EpisodeAdapter):
    name = "synthetic"
    version = "1"

    def __init__(self, config: dict):
        super().__init__(config)
        self.hw = tuple(config.get("image_hw", (96, 144)))
        self.f = float(config.get("focal", 150.0))
        self.baseline = float(config.get("baseline_m", 0.095))
        self.frames = int(config.get("frames", 10))
        self.asset_dir = Path(config["asset_dir"])

    def K(self) -> np.ndarray:
        H, W = self.hw
        return np.array([[self.f, 0, W / 2], [0, self.f, H / 2], [0, 0, 1.0]])

    def T_cam_base(self, rig: str, eye: str) -> np.ndarray:
        T = look_at_T_cam_base(RIG_EYES[rig], TARGET)
        if eye == "right":
            T = make_T(np.eye(3), [-self.baseline, 0, 0]) @ T  # right camera centre at +x of the left
        return T

    def list_episodes(self) -> list[str]:
        return ["synthetic_ep000"]

    def time(self, index: int) -> float:
        return index * 0.5

    def read_meta(self, episode_id: str) -> EpisodeMeta:
        H, W = self.hw
        intr = Intrinsics(W, H, self.f, self.f, W / 2 - 0.5, H / 2 - 0.5, "integer_centers", "none", (), 0.0,
                          "synthetic")
        streams, stereo, stamps = [], [], {}
        for r, rig in enumerate(RIG_EYES):
            for eye in ("left", "right"):
                sid = f"{rig}/{eye}"
                streams.append(CameraStream(sid, rig, eye, "infrared", 1, "GRAY", "uint8", intr, True,
                                            device_serial=f"SYN{r:04d}", sensor_name="synthetic IR",
                                            timestamp_source="synthetic", ir_emitter="off"))
                stamps[sid] = (np.arange(self.frames) * 66_666_667 + 1_000_000 * r).astype(np.int64)
            if rig == "rig2":
                stereo.append(StereoGeometry(rig, f"{rig}/left", f"{rig}/right", "missing",
                                             notes="synthetic: deliberately missing to exercise explicit reporting"))
            else:
                T = make_T(np.eye(3), [-self.baseline, 0, 0])
                stereo.append(StereoGeometry(rig, f"{rig}/left", f"{rig}/right", "metric",
                                             tuple(map(tuple, T)), self.baseline, "synthetic"))
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        urdf = self.asset_dir / "synthetic_robot.urdf"
        urdf.write_text(synthetic_robot_urdf())
        robot = RobotDescription("franka_panda", "synthetic_two_finger", str(urdf), None,
                                 gripper_base_frame="gripper_base",
                                 T_flange_gripper=tuple(map(tuple, make_T(np.eye(3), [0, 0, 0.011]))),
                                 mount_source="synthetic", gripper_state_semantics="normalized_closure",
                                 gripper_joint="finger_joint")
        n = self.frames * 7
        ts = (np.arange(n) * 10_000_000).astype(np.int64)
        tt = ts / 1e9 / 0.0666 * 0.5
        q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
        q = q0 + 0.4 * np.sin(tt[:, None] * 0.6 + np.arange(7))
        traj = RobotTrajectory(ts, q, 0.5 + 0.5 * np.sin(tt), extra={"joint_velocities": np.gradient(q, axis=0)},
                               extra_units={"joint_velocities": "rad/sample"})
        return EpisodeMeta(episode_id, tuple(streams), tuple(stereo), robot, traj, stamps,
                           Provenance(self.name, self.version, "synthetic://", (), "", "synthetic fixture"),
                           clock="synthetic", assumptions=("synthetic data: every value is generated",))

    def render(self, rig: str, eye: str, index: int):
        return render_box_scene(self.T_cam_base(rig, eye), self.K(), self.hw, self.time(index))

    def iter_frames(self, episode_id, stream_id, start=0, stop=None):
        rig, eye = stream_id.split("/")
        stamps = self.read_meta(episode_id).stream_timestamps_ns[stream_id]
        for i in range(start, self.frames if stop is None else min(stop, self.frames)):
            image, _, _, _ = self.render(rig, eye, i)
            yield i, int(stamps[i]), np.clip(image, 0, 255).astype(np.uint8)

    # ---- synthetic-only ground truth helpers (tests / mock backends) ----
    def gt_points(self, stream_id: str, query: int, pixels_rc: np.ndarray, frames: list[int]) -> tuple:
        """Ground-truth base-frame positions (T,N,3) of the query pixels' surface points and their visibility."""
        rig, eye = stream_id.split("/")
        _, _, P, on_obj = self.render(rig, eye, query)
        r, c = pixels_rc[:, 0], pixels_rc[:, 1]
        P0, obj = P[r, c], on_obj[r, c]
        from s4d.realworld.synthetic import object_center

        out, vis = [], []
        for f in frames:
            Pt = np.where(obj[:, None], P0 + object_center(self.time(f)) - object_center(self.time(query)), P0)
            _, depth, _, _ = self.render(rig, eye, f)
            cam = transform_points(self.T_cam_base(rig, eye), Pt)
            uv = cam @ self.K().T
            front = cam[:, 2] > 1e-6
            uv = np.where(front[:, None], uv[:, :2] / np.where(front, uv[:, 2], 1)[:, None], -1)
            j, i = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
            H, W = self.hw
            inside = front & (i >= 0) & (i < H) & (j >= 0) & (j < W)
            v = np.zeros(len(Pt), bool)
            v[inside] = np.abs(depth[i[inside], j[inside]] - cam[inside, 2]) < 0.02
            out.append(Pt)
            vis.append(v)
        return np.stack(out), np.stack(vis)
