"""Subprocess check (one UUID): RL frames equal collector-rig frames and the gapped stack picks the right frames."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import guard_gpus, guard_mujoco  # noqa: E402

guard_gpus()
guard_mujoco()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from s4d.data.metaworld.cameras import mujoco_free_camera  # noqa: E402
from s4d.rl.env import TRAIN_PATHS, MetaWorldCameraEnv, trajectory_path  # noqa: E402

env = MetaWorldCameraEnv("hammer", 0, image_size=64, frame_stack=3, frame_gap=3, action_repeat=2, max_episode_steps=250)
obs, proprio = env.reset(TRAIN_PATHS["train2"])
first = env.latest_frame()
assert obs.shape == (9, 64, 64) and proprio.shape == (4,)
assert all(np.array_equal(obs[3 * k : 3 * k + 3], first) for k in range(3)), "reset must pad with the first frame"
frames = [first]
for _ in range(7):
    obs, *_ = env.step(np.zeros(4, np.float32))
    frames.append(env.latest_frame())
history = frames[-7:]
assert np.array_equal(obs, np.concatenate([history[0], history[3], history[6]])), "gap-3 stack is t-6, t-3, t"
reference = mujoco.Renderer(env.model, height=64, width=64)
reference.update_scene(env.data, camera=mujoco_free_camera(-30.0, -60.0))
same_as_rig = bool(np.array_equal(reference.render().transpose(2, 0, 1), env.latest_frame()))
env.path, env.render_count = trajectory_path("lateral"), 0
centre = env._render()
env.render_count = 18  # reference path is 0.12 m to the right here
moved = not np.array_equal(env._render(), centre)
first_states = []
for _ in range(3):
    env.reset(TRAIN_PATHS["train0"])
    first_states.append(np.asarray(env.data.qpos).copy())
varied = not all(np.allclose(first_states[0], s) for s in first_states[1:])
print(json.dumps({"same_as_rig": same_as_rig, "trajectory_moves": moved, "reset_varies_objects": varied}))
