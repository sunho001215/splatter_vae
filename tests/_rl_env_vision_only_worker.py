"""Subprocess check (one UUID): the real environment and the evaluation pool with ``proprio_indices=[]`` give the policy
only the RGB history; images, rewards and success equal the with-proprio environment's for the same seeds and actions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import guard_gpus, guard_mujoco  # noqa: E402

guard_gpus()
guard_mujoco()

import numpy as np  # noqa: E402

from s4d.rl.env import TRAIN_PATHS, MetaWorldCameraEnv, trajectory_path  # noqa: E402
from s4d.rl.vecenv import EnvPool  # noqa: E402


def rollout(indices):
    env = MetaWorldCameraEnv("hammer", 0, image_size=64, frame_stack=3, action_repeat=2, max_episode_steps=250,
                             proprio_indices=indices)
    obs, proprio = env.reset(TRAIN_PATHS["train1"], seed=7)
    out = {"obs": [obs], "proprio": [proprio], "reward": [], "success": []}
    rng = np.random.default_rng(3)
    for _ in range(6):
        obs, proprio, reward, done, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        out["obs"].append(obs)
        out["proprio"].append(proprio)
        out["reward"].append(reward)
        out["success"].append(info["success"])
    return out


def image_policy(seen):
    def act(obs, proprio):  # a deterministic function of the images only
        seen.append((tuple(np.asarray(obs).shape), tuple(np.asarray(proprio).shape)))
        return np.tanh(np.asarray(obs).reshape(len(obs), -1)[:, :4].astype(np.float32) / 128 - 1)
    return act


def main() -> None:
    vo, ref = rollout([]), rollout([0, 1, 2, 3])
    base = {"image_size": 64, "frame_stack": 3, "action_repeat": 2, "max_episode_steps": 40}
    episodes = [(TRAIN_PATHS["train0"], 11), (trajectory_path("circular"), 12), (TRAIN_PATHS["train3"], 13)]
    results = {}
    for name, indices in (("vision_only", []), ("with_proprio", [0, 1, 2, 3])):
        seen = []
        pool = EnvPool("hammer", 0, 2, 2, {**base, "proprio_indices": indices})
        try:
            results[name] = (pool.run(image_policy(seen), episodes), sorted(set(seen)))
        finally:
            pool.close()
    print(json.dumps({
        "proprio_shapes": sorted({tuple(p.shape) for p in vo["proprio"]}),
        "reference_proprio_shapes": sorted({tuple(p.shape) for p in ref["proprio"]}),
        "images_equal_reference": all(np.array_equal(a, b) for a, b in zip(vo["obs"], ref["obs"])),
        "rewards_equal_reference": vo["reward"] == ref["reward"],
        "success_equal_reference": vo["success"] == ref["success"],
        "pool_inputs": [list(map(list, x)) for x in results["vision_only"][1]],
        "pool_returns_equal_reference": bool(np.allclose(results["vision_only"][0][1], results["with_proprio"][0][1])),
        "pool_success_equal_reference": results["vision_only"][0][0] == results["with_proprio"][0][0],
    }))


if __name__ == "__main__":
    main()
