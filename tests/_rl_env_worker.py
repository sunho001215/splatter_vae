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
    env.paths, env.render_count = [trajectory_path("lateral")], 0
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
    # multi-view + depth + cameras: view 0 equals the single-camera frame, depth equals the collector's quantised
    # metric depth, cameras equal the pretraining rig, and the env pool batches the dict observations
    from s4d.data.metaworld.cameras import camera_rig  # noqa: E402
    from s4d.data.metaworld.collect import DEPTH_UNIT_M, encode_depth  # noqa: E402
    from s4d.rl.env import camera_c2w, stack_observations  # noqa: E402
    from s4d.rl.evaluate import evaluation_groups  # noqa: E402

    multi = MetaWorldCameraEnv("hammer", 0, views=3, depth=True, cameras=True, **kwargs)
    views = (TRAIN_PATHS["train0"], TRAIN_PATHS["train2"], trajectory_path("circular"))
    mobs, _ = multi.reset(views, 5)
    plain_obs, _ = fresh.reset(TRAIN_PATHS["train0"], 5)
    shapes = {k: tuple(v.shape) for k, v in mobs.items()}
    multi_first_view = bool(np.array_equal(mobs["rgb"][-1, 0], plain_obs[-3:]))
    depth_r = mujoco.Renderer(multi.model, height=64, width=64)
    depth_r.enable_depth_rendering()
    depth_r.update_scene(multi.data, camera=mujoco_free_camera(0.0, -45.0))
    expected = encode_depth(depth_r.render()).astype(np.float32) * np.float32(DEPTH_UNIT_M)
    depth_matches = bool(np.array_equal(mobs["depth"][-1, 0], expected))
    rig = camera_rig(64, 64)
    cameras_match = bool(np.allclose(mobs["c2w"][-1, 1], rig["c2w"][2], atol=1e-5)
                         and np.allclose(mobs["K"][-1, 0], rig["K"][0], atol=1e-5)
                         and np.allclose(camera_c2w(TRAIN_PATHS["train3"][0]), rig["c2w"][3], atol=1e-6))
    before = mobs["c2w"][-1, 2].copy()
    mobs, *_ = multi.step(np.zeros(4, np.float32))
    camera_moves = not np.allclose(mobs["c2w"][-1, 2], before)
    groups = evaluation_groups({"episodes_per_train_camera": 1, "episodes_per_heldout_camera": 1,
                                "episodes_per_trajectory": 1}, views=3)
    pool = EnvPool("hammer", 0, 1, 2, {**kwargs, "views": 3, "depth": True, "cameras": True})
    seen = []

    def act_multi(obs, proprio):
        seen.append({k: v.shape for k, v in obs.items()})
        return np.zeros((len(proprio), 4), np.float32)

    pool.run(act_multi, [(groups["eval0"][0], 1), (groups["traj_lateral"][0], 2)])
    pool.close()
    pooled_dict = bool(seen) and seen[0]["rgb"][:3] == (2, 3, 3) and seen[0]["depth"][:3] == (2, 3, 3)
    batched = stack_observations([mobs, mobs])
    for e in (env, single, fresh, multi):
        e.close()
    print(
        json.dumps(
            {
                "same_as_rig": same_as_rig,
                "trajectory_moves": moved,
                "reset_varies_objects": varied,
                "pool_matches_single_env": same,
                "seeded_reset_history_free": history_free,
                "multi_shapes": shapes,
                "multi_first_view_matches_single": multi_first_view,
                "multi_depth_matches_collector": depth_matches,
                "multi_cameras_match_rig": cameras_match,
                "multi_trajectory_camera_moves": camera_moves,
                "multi_pool_batches_dicts": pooled_dict,
                "multi_stack_shape": list(batched["rgb"].shape),
            }
        )
    )


if __name__ == "__main__":  # spawned pool workers re-import this file
    main()
