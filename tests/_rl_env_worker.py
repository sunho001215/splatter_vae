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


def act(obs, proprio):  # deterministic function of the observation, so any pixel difference changes the rollout
    return np.tanh(np.stack([obs.reshape(len(obs), -1)[:, :4].astype(np.float32) / 128 - 1, proprio], 0).sum(0))


def main() -> None:
    env = MetaWorldCameraEnv("hammer", 0, image_size=64, frame_stack=3, action_repeat=2, max_episode_steps=250)
    obs, proprio = env.reset(TRAIN_PATHS["train2"])
    first = env.latest_frame()
    assert obs.shape == (9, 64, 64) and proprio.shape == (4,)
    assert all(np.array_equal(obs[3 * k : 3 * k + 3], first) for k in range(3)), "reset must pad with the first frame"
    frames = [first]
    for _ in range(7):
        obs, *_ = env.step(np.zeros(4, np.float32))
        frames.append(env.latest_frame())
    assert np.array_equal(obs, np.concatenate(frames[-3:])), "stack is the last three agent-step frames"
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
    from s4d.rl.vecenv import EnvPool  # noqa: E402

    kwargs = {
        "image_size": 64,
        "frame_stack": 3,
        "action_repeat": 2,
        "max_episode_steps": 40,
        "proprio_indices": [0, 1, 2, 3],
    }
    episodes = [(TRAIN_PATHS["train0"], 11), (trajectory_path("circular"), 12), (TRAIN_PATHS["train3"], 13)]

    pool = EnvPool("hammer", 0, 2, 2, kwargs)
    pooled = pool.run(act, episodes)
    pool.close()
    single = MetaWorldCameraEnv("hammer", 0, **kwargs)
    returns = []
    for path, seed in episodes:
        obs, proprio = single.reset(path, seed)
        total, done = 0.0, False
        while not done:
            obs, proprio, reward, done, _ = single.step(act(obs[None], proprio[None])[0])
            total += reward
        returns.append(total)
    same = bool(np.allclose(pooled[1], returns))
    fresh = MetaWorldCameraEnv("hammer", 0, **kwargs)
    used_obs, _ = single.reset(TRAIN_PATHS["train0"], 99)  # single has already played three episodes
    fresh_obs, _ = fresh.reset(TRAIN_PATHS["train0"], 99)
    history_free = bool(np.array_equal(used_obs, fresh_obs))
    for e in (env, single, fresh):
        e.close()
    print(
        json.dumps(
            {
                "same_as_rig": same_as_rig,
                "trajectory_moves": moved,
                "reset_varies_objects": varied,
                "pool_matches_single_env": same,
                "seeded_reset_history_free": history_free,
            }
        )
    )


if __name__ == "__main__":  # spawned pool workers re-import this file
    main()
