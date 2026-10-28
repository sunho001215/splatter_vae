"""PPO pilot (s4d/rl/sb3_ppo.py): pinned SB3, protocol spaces, normalised deterministic evaluation policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from s4d.rl import sb3_ppo
from s4d.rl.protocol import resolve_config


def test_pinned_stable_baselines3_is_used():
    sb3 = sb3_ppo.load_sb3()
    assert sb3.__version__ == "2.9.0" and str(sb3_ppo.UPSTREAM) in sb3.__file__


def test_algorithm_config_replaces_drm_sections():
    cfg = resolve_config("hammer", "cnn", 2000, [], algorithm="sb3_ppo")
    assert cfg["algorithm"] == "sb3_ppo" and "nstep" not in cfg["train"] and "expectile" not in cfg["agent"]
    assert cfg["agent"]["n_envs"] * cfg["agent"]["n_steps"] == 2048 and cfg["train"]["num_train_steps"] == 1000000
    assert cfg["env"] == resolve_config("hammer", "cnn", 2000, [])["env"]


def test_raw_spaces_follow_the_encoder_observation():
    cnn = resolve_config("hammer", "cnn", 2000, [], algorithm="sb3_ppo")
    from s4d.rl.env import env_kwargs

    space, plain = sb3_ppo.raw_spaces(cnn, env_kwargs(cnn), 4)
    assert plain and space["pixels"].shape == (9, 128, 128) and space["proprio"].shape == (4,)
    snerl = resolve_config("hammer", "snerl", 2000, [], algorithm="sb3_ppo")
    space, plain = sb3_ppo.raw_spaces(snerl, env_kwargs(snerl), 4)
    assert not plain and space["rgb"].shape == (3, 3, 3, 128, 128) and space["c2w"].shape == (3, 3, 4, 4)


def test_cnn_policy_acts_deterministically_with_the_snapshot_normaliser():
    device = torch.device("cuda")
    cfg = resolve_config("hammer", "cnn", 2000, [], algorithm="sb3_ppo")
    policy = sb3_ppo.PPOPolicy(cfg, 4, 4, device)
    stats = {"proprio": {"mean": torch.ones(4), "var": torch.full((4,), 4.0)}}
    policy.load_policy({"policy": policy.policy.state_dict(),
                        "normalizer": {"obs_rms": stats, "clip_obs": 10.0, "epsilon": 1e-8, "norm_obs": True}})
    assert np.allclose(policy._normalize("proprio", np.full((1, 4), 3.0, np.float32)), 1.0, atol=1e-6)
    pixels = torch.randint(0, 255, (3, 9, 128, 128), dtype=torch.uint8, device=device)
    proprio = np.zeros((3, 4), np.float32)
    a, b = policy.act(pixels, proprio, 0, eval_mode=True), policy.act(pixels, proprio, 0, eval_mode=True)
    assert a.shape == (3, 4) and np.array_equal(a, b) and np.abs(a).max() <= 1.0
