"""Subprocess check (one UUID) for camera suite v2: a multi-view env that receives a v2 group (a stronger moving path in
the first slot) reports, for every frame of its history, the K and OpenCV c2w of the pose actually rendered at that
frame (pose = render count mod 72, also across the loop boundary); a v2 static camera renders exactly the rig view."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import guard_mujoco  # noqa: E402

guard_mujoco()  # MuJoCo only: the allowlisted UUID and EGL device are enforced, no CUDA context is created

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from s4d.data.metaworld.cameras import FOVY_DEG, intrinsics_from_fovy, mujoco_free_camera  # noqa: E402
from s4d.rl.camera_suites import HELDOUT_V2_CAMERAS  # noqa: E402
from s4d.rl.env import HELDOUT_V2_PATHS, MetaWorldCameraEnv, camera_c2w  # noqa: E402
from s4d.rl.evaluate import evaluation_groups_v2  # noqa: E402


def main() -> None:
    kwargs = {"image_size": 32, "frame_stack": 3, "action_repeat": 2, "max_episode_steps": 250,
              "proprio_indices": [0, 1, 2, 3]}
    groups = evaluation_groups_v2({"episodes_per_train_camera": 1, "episodes_per_heldout_camera": 1,
                                   "episodes_per_trajectory": 1}, views=3)
    views = groups["traj_lateral_3x"][0]
    env = MetaWorldCameraEnv("hammer", 0, views=3, cameras=True, **kwargs)
    obs, _ = env.reset(views, 7)
    K = intrinsics_from_fovy(FOVY_DEG, 32, 32)
    ok, checked = True, 0
    renders = [0, 0, 0]  # render counts of the stacked frames (the reset frame pads the history)
    for step in range(80):
        expected = np.stack([[camera_c2w(view[r % len(view)]) for view in views] for r in renders])
        ok &= bool(np.allclose(obs["c2w"], expected, atol=1e-5) and np.allclose(obs["K"], K, atol=1e-4))
        checked += 1
        obs, *_ = env.step(np.zeros(4, np.float32))
        renders = renders[1:] + [step + 1]
    env.close()
    single = MetaWorldCameraEnv("hammer", 0, **kwargs)
    name = next(iter(HELDOUT_V2_PATHS))
    single.reset(HELDOUT_V2_PATHS[name], 3)
    tilt, azimuth = HELDOUT_V2_CAMERAS[0]
    reference = mujoco.Renderer(single.model, height=32, width=32)
    reference.update_scene(single.data, camera=mujoco_free_camera(azimuth, -tilt))
    static_ok = bool(np.abs(reference.render().transpose(2, 0, 1).astype(int) - single.latest_frame()).max() <= 1)
    reference.close()
    single.close()
    print(json.dumps({"c2w_matches_path_per_frame": ok, "frames_checked": checked, "static_view_matches_rig": static_ok}))


if __name__ == "__main__":
    main()
