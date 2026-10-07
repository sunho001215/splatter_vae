"""DrQ-v2 port: replay semantics and resume, agent updates, frame-gap rule, cameras and env rendering."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from s4d.rl.agent import DrQv2Agent, schedule
from s4d.rl.env import TRAIN_PATHS, lateral_offsets, matching_frame_gap, stack_indices, trajectory_path
from s4d.rl.replay import MemmapReplayBuffer, MemmapReplayBufferStorage, make_replay_loader

REPO = Path(__file__).resolve().parents[1]


def fill(storage, episodes: int, length: int, start_value: int = 0) -> None:
    value = start_value
    for _ in range(episodes):
        storage.add_initial(np.full((3, 4, 4), value, np.uint8), np.zeros(4))
        for t in range(length):
            value += 1
            done = t == length - 1
            storage.add(np.zeros(4), 1.0, 0.0 if done else 1.0, np.full((3, 4, 4), value, np.uint8), np.zeros(4), done)
        value += 100


def test_replay_nstep_stacks_never_cross_episodes_and_resume(tmp_path):
    args = (tmp_path, (3, 4, 4), np.uint8, (4,), (4,), 1000, 3, 3)
    storage = MemmapReplayBufferStorage(*args)
    fill(storage, episodes=2, length=5)
    sampler = MemmapReplayBuffer(*args, discount=0.5)
    for _ in range(200):
        obs, _, _, reward, discount, next_obs, _ = sampler.sample()
        frames = obs.reshape(3, 3, 4, 4)[:, 0, 0, 0].astype(int)
        nxt = next_obs.reshape(3, 3, 4, 4)[:, 0, 0, 0].astype(int)
        assert nxt[-1] - frames[-1] == 3, "next state is n=3 steps later"
        assert np.all(np.diff(frames) >= 0) and frames[-1] - frames[0] <= 2, "stack is padded, never crosses episodes"
        assert reward[0] == pytest.approx(1.0 + 0.5 + 0.25)
        assert discount[0] in (pytest.approx(0.125), 0.0)
    reopened = MemmapReplayBufferStorage(*args, reset=False)
    reopened.resume()
    fill(reopened, episodes=1, length=4, start_value=50)
    episodes = np.unique(np.asarray(reopened._episode_id)[np.asarray(reopened._state_id) >= 0])
    assert episodes.tolist() == [0, 1, 2]
    assert len(reopened) == 14


def tiny_cfg(encoder_type="convnet"):
    return {
        "env": {"frame_stack": 3, "image_height": 16, "image_width": 16},
        "vision": {"encoder_type": encoder_type},
        "agent": {
            "lr": 1e-3,
            "feature_dim": 8,
            "hidden_dim": 16,
            "critic_target_tau": 0.01,
            "stddev_schedule": "linear(1.0,0.1,100)",
            "stddev_clip": 0.3,
        },
    }


def test_cnn_agent_update_changes_all_networks(tmp_path):
    agent = DrQv2Agent(tiny_cfg(), action_dim=4, proprio_dim=4, device=torch.device("cpu"))
    storage = MemmapReplayBufferStorage(tmp_path, (3, 16, 16), np.uint8, (4,), (4,), 100, 3, 3)
    rng = np.random.default_rng(0)
    for _ in range(3):
        storage.add_initial(rng.integers(0, 255, (3, 16, 16), dtype=np.uint8), rng.normal(size=4))
        for t in range(10):
            storage.add(
                rng.uniform(-1, 1, 4),
                rng.normal(),
                1.0,
                rng.integers(0, 255, (3, 16, 16), dtype=np.uint8),
                rng.normal(size=4),
                t == 9,
            )
    loader, _ = make_replay_loader(storage, batch_size=8, num_workers=0, discount=0.99)
    before = {k: [p.clone() for p in getattr(agent, k).parameters()] for k in ("encoder", "actor", "critic")}
    metrics = agent.update(iter(loader), step=10)
    assert all(math.isfinite(v) for v in metrics.values())
    for name, params in before.items():
        assert any(not torch.equal(a, b) for a, b in zip(params, getattr(agent, name).parameters())), name
    action = agent.act(torch.zeros(2, 9, 16, 16, dtype=torch.uint8), np.zeros((2, 4)), step=10, eval_mode=True)
    assert action.shape == (2, 4) and np.all(np.abs(action) <= 1)
    clone = DrQv2Agent(tiny_cfg(), action_dim=4, proprio_dim=4, device=torch.device("cpu"))
    clone.load_state_dict(agent.state_dict())
    assert torch.equal(clone.actor.policy[0].weight, agent.actor.policy[0].weight)


def test_schedule_and_frame_gap_rule():
    assert schedule("linear(1.0,0.1,100)", 50) == pytest.approx(0.55)
    assert schedule(0.2, 7) == 0.2
    assert matching_frame_gap([3, 6, 9], 3, 2) == 3
    assert matching_frame_gap([3, 6, 9], 3, 3) == 2
    assert matching_frame_gap([3, 6, 9], 1, 2) == 1
    with pytest.raises(ValueError):
        matching_frame_gap([3, 9], 3, 2)
    with pytest.raises(ValueError):
        matching_frame_gap(None, 3, 2)
    assert stack_indices(3, 3) == [0, 3, 6] and stack_indices(3, 1) == [0, 1, 2]


def test_reference_camera_trajectories():
    offsets = lateral_offsets(72, 0.12)
    assert offsets[0] == 0 and offsets[-1] == pytest.approx(0) and offsets.max() == pytest.approx(0.12, abs=0.01)
    lateral = trajectory_path("lateral")
    base = TRAIN_PATHS["train1"][0]
    assert lateral[0].azimuth == pytest.approx(base.azimuth) and lateral[0].elevation == pytest.approx(base.elevation)
    assert lateral[0].distance == pytest.approx(1.0)
    assert max(c.distance for c in lateral) > 1.0
    circular = trajectory_path("circular")
    assert circular[0].azimuth == pytest.approx(base.azimuth + 10) and circular[18].elevation == pytest.approx(-51)


def test_env_frames_match_collector_rig_in_one_uuid_subprocess():
    first_uuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / "tests/_rl_env_worker.py")],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": first_uuid, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"same_as_rig": True, "trajectory_moves": True, "reset_varies_objects": True}
