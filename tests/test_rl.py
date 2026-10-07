"""DrQ-v2 port: replay semantics, backings and resume, latent-vs-pixel update equivalence, cameras and env."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from s4d.model.encoder import Encoder, EncoderConfig
from s4d.rl.agent import DrQv2Agent, schedule
from s4d.rl.env import TRAIN_PATHS, check_pretraining_spacing, lateral_offsets, trajectory_path
from s4d.rl.evaluate import episode_seeds, evaluation_groups, policy_inputs
from s4d.rl.replay import Replay, replay_iterator

REPO = Path(__file__).resolve().parents[1]
GAMMA = 0.5


def frame(value: int) -> np.ndarray:
    return np.full((3, 4, 4), value % 256, np.uint8)


def fill(replay: Replay, episodes: int, length: int, start: int = 0) -> int:
    value = start
    for _ in range(episodes):
        replay.add_initial(frame(value), np.full(4, value, np.float32))
        for t in range(length):
            value += 1
            done = t == length - 1
            replay.add(np.full(4, value, np.float32), 1.0, 0.0 if done else 1.0, frame(value), np.full(4, value), done)
        value += 100
    return value


def check_batch(batch, nstep: int = 3) -> None:
    obs, proprio, action, reward, discount, next_obs, next_proprio = (np.asarray(x) for x in batch)
    frames = obs.reshape(len(obs), 3, 3, 4, 4)[:, :, 0, 0, 0].astype(int)
    nxt = next_obs.reshape(len(obs), 3, 3, 4, 4)[:, :, 0, 0, 0].astype(int)
    assert np.all(nxt[:, -1] - frames[:, -1] == nstep), "next state is n steps later in the same episode"
    assert np.all(np.diff(frames, axis=1) >= 0) and np.all(frames[:, -1] - frames[:, 0] <= 2), "padded, never crosses"
    assert np.all(proprio[:, 0] == frames[:, -1]) and np.all(next_proprio[:, 0] == nxt[:, -1])
    assert np.all(action[:, 0] == frames[:, -1] + 1), "action leaves the sampled state"
    full = discount[:, 0] > 0
    assert np.allclose(reward[full, 0], 1 + GAMMA + GAMMA**2) and np.allclose(discount[full, 0], GAMMA**3)


@pytest.mark.parametrize("backing", ["ram", "disk"])
def test_replay_semantics_both_backings(tmp_path, backing):
    directory = tmp_path / "replay" if backing == "disk" else None
    replay = Replay(directory, (3, 4, 4), np.uint8, 4, 4, 1000, 3, 3, snapshot_dir=tmp_path / "snap")
    fill(replay, episodes=3, length=6)
    assert len(replay) == 18
    check_batch(replay.sample(256, np.random.default_rng(0), GAMMA))
    batches = replay_iterator(replay, 64, GAMMA, num_workers=1 if backing == "disk" else 0, seed=0)
    check_batch(next(batches))


def test_ring_overwrite_never_returns_stale_rows(tmp_path):
    replay = Replay(None, (3, 4, 4), np.uint8, 4, 4, 20, 3, 3)
    fill(replay, episodes=12, length=6)
    assert len(replay) == 20
    for seed in range(20):
        check_batch(replay.sample(32, np.random.default_rng(seed), GAMMA))


@pytest.mark.parametrize("backing", ["ram", "disk"])
def test_checkpoint_restore_returns_to_checkpointed_contents(tmp_path, backing):
    directory = tmp_path / "replay" if backing == "disk" else None
    args = (directory, (3, 4, 4), np.uint8, 4, 4, 1000, 3, 3)
    replay = Replay(*args, snapshot_dir=tmp_path / "snap")
    value = fill(replay, episodes=2, length=6)
    replay.add_initial(frame(value), np.zeros(4))
    replay.add(np.zeros(4), 1.0, 1.0, frame(value + 1), np.zeros(4), False)  # episode in progress at checkpoint
    meta = replay.checkpoint()
    reference = replay.sample(128, np.random.default_rng(1), GAMMA)
    fill(replay, episodes=2, length=6, start=5000)  # written after the checkpoint, lost by the crash
    reopened = Replay(*args, mode="r+" if directory else "w+", snapshot_dir=tmp_path / "snap")
    reopened.restore(meta)
    assert len(reopened) == 13  # two full episodes plus the in-progress transition at the checkpoint
    for got, want in zip(reopened.sample(128, np.random.default_rng(1), GAMMA), reference):
        assert np.array_equal(got, want)
    fill(reopened, episodes=1, length=6, start=9000)
    check_batch(reopened.sample(256, np.random.default_rng(2), GAMMA))
    assert int(reopened.episode_id.max()) == 3, "resumed data starts a fresh episode id"


def tiny_cfg(encoder_type="convnet", export_path=None):
    return {
        "env": {"frame_stack": 3, "image_height": 32, "image_width": 32},
        "vision": {"encoder_type": encoder_type, "export_path": export_path},
        "agent": {
            "lr": 1e-3,
            "feature_dim": 8,
            "hidden_dim": 16,
            "critic_target_tau": 0.01,
            "stddev_schedule": "linear(1.0,0.1,100)",
            "stddev_clip": 0.3,
        },
    }


def test_cnn_agent_update_changes_all_networks_and_roundtrips(tmp_path):
    agent = DrQv2Agent(tiny_cfg(), action_dim=4, proprio_dim=4, device=torch.device("cpu"))
    replay = Replay(tmp_path / "replay", (3, 32, 32), np.uint8, 4, 4, 100, 3, 3)
    rng = np.random.default_rng(0)
    for _ in range(3):
        replay.add_initial(rng.integers(0, 255, (3, 32, 32), dtype=np.uint8), rng.normal(size=4))
        for t in range(10):
            image = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, image, rng.normal(size=4), t == 9)
    before = {k: [p.clone() for p in getattr(agent, k).parameters()] for k in ("encoder", "actor", "critic")}
    metrics = agent.update(replay_iterator(replay, 8, 0.99, num_workers=0, seed=0), step=10)
    assert all(math.isfinite(v) for v in metrics.values())
    for name, params in before.items():
        assert any(not torch.equal(a, b) for a, b in zip(params, getattr(agent, name).parameters())), name
    action = agent.act(torch.zeros(2, 9, 32, 32, dtype=torch.uint8), np.zeros((2, 4)), step=10, eval_mode=True)
    assert action.shape == (2, 4) and np.all(np.abs(action) <= 1)
    clone = DrQv2Agent(tiny_cfg(), action_dim=4, proprio_dim=4, device=torch.device("cpu"))
    clone.load_state_dict(agent.state_dict())
    assert torch.equal(clone.actor.policy[0].weight, agent.actor.policy[0].weight)


def write_export(path: Path, num_frames: int = 3) -> Path:
    cfg = EncoderConfig(image_height=32, image_width=32, width=32, depth=2, heads=2, slot_dim=16, num_frames=num_frames)
    torch.manual_seed(0)
    encoder = Encoder(cfg)
    payload = {"format": "splatter4d-encoder-v1", "encoder_config": asdict(cfg), "state_dict": encoder.state_dict()}
    torch.save({**payload, "step": 0, "frame_strides": [2, 4, 6]}, path)
    return path


def test_frozen_latent_replay_update_equals_encoding_stored_frames(tmp_path):
    """Online per-step encoding (batch 1, stored fp16 in RAM) vs encoding the stored frames at update time."""
    device = torch.device("cuda")
    cfg = tiny_cfg("splatter4d", str(write_export(tmp_path / "encoder.pt")))
    torch.manual_seed(1)
    agent = DrQv2Agent(cfg, 4, 4, device)
    assert not agent.use_pixels and not agent.augment_pixels, "frozen encoders use no augmentation (reference)"
    assert agent.encoder_opt is None and not any(p.requires_grad for p in agent.encoder.backbone.parameters())
    latents = Replay(None, agent.encoder.replay_atom_shape, np.float16, 4, 4, 1000, 1, 3)
    pixels = Replay(tmp_path / "pixels", (3, 32, 32), np.uint8, 4, 4, 1000, 3, 3)
    rng = np.random.default_rng(0)
    for _ in range(4):
        frames = [rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)] * 3
        prop = rng.normal(size=4).astype(np.float32)
        stack = np.concatenate(frames)
        latents.add_initial(policy_inputs(agent, stack[None])[0].cpu().numpy(), prop)
        pixels.add_initial(frames[-1], prop)
        for t in range(20):
            frames = frames[1:] + [rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)]
            action, reward, prop = rng.uniform(-1, 1, 4), float(rng.normal()), rng.normal(size=4).astype(np.float32)
            latent = policy_inputs(agent, np.concatenate(frames)[None])[0].cpu().numpy()
            latents.add(action, reward, 1.0, latent, prop, t == 19)
            pixels.add(action, reward, 1.0, frames[-1], prop, t == 19)
    from_ram = latents.sample(64, np.random.default_rng(5), 0.99)
    from_pixels = pixels.sample(64, np.random.default_rng(5), 0.99)
    for i in (1, 2, 3, 4, 6):
        assert np.array_equal(from_ram[i], from_pixels[i])
    encode = agent.encoder.extract_cacheable_feature
    batch_latents = [encode(torch.as_tensor(from_pixels[i], device=device)).cpu().numpy() for i in (0, 5)]
    for stored, recomputed in zip((from_ram[0], from_ram[5]), batch_latents):
        np.testing.assert_allclose(stored.astype(np.float32), recomputed.astype(np.float32), atol=2e-2, rtol=0)
    twin = copy.deepcopy(agent)
    rebuilt = list(from_ram)
    rebuilt[0], rebuilt[5] = batch_latents
    torch.manual_seed(7)
    agent.update(iter([tuple(torch.as_tensor(x) for x in from_ram)]), step=50)
    torch.manual_seed(7)
    twin.update(iter([tuple(torch.as_tensor(x) for x in rebuilt)]), step=50)
    for a, b in zip(agent.actor.parameters(), twin.actor.parameters()):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
    for a, b in zip(agent.critic.parameters(), twin.critic.parameters()):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_schedule_spacing_guard_and_eval_protocol():
    assert schedule("linear(1.0,0.1,100)", 50) == pytest.approx(0.55)
    assert schedule(0.2, 7) == 0.2
    check_pretraining_spacing([2, 4, 6], 3, 2)
    check_pretraining_spacing(None, 1, 2)  # the T=1 ablation has no temporal spacing
    for strides in ([3, 6, 9], None):
        with pytest.raises(ValueError, match="outside pretraining strides"):
            check_pretraining_spacing(strides, 3, 2)
    groups = evaluation_groups(
        {"episodes_per_train_camera": 20, "episodes_per_heldout_camera": 20, "episodes_per_trajectory": 20}
    )
    assert len(groups) == 12 and sum(n for _, n in groups.values()) == 240
    assert sum(groups[f"train{i}"][1] for i in range(6)) == 120
    assert episode_seeds(3, 4, 5) == episode_seeds(3, 4, 5) != episode_seeds(3, 5, 5)


def test_reference_camera_trajectories():
    offsets = lateral_offsets(72, 0.12)
    assert offsets[0] == 0 and offsets[-1] == pytest.approx(0) and offsets.max() == pytest.approx(0.12, abs=0.01)
    lateral = trajectory_path("lateral")
    base = TRAIN_PATHS["train1"][0]
    assert lateral[0].azimuth == pytest.approx(base.azimuth) and lateral[0].elevation == pytest.approx(base.elevation)
    assert lateral[0].distance == pytest.approx(1.0) and max(c.distance for c in lateral) > 1.0
    circular = trajectory_path("circular")
    assert circular[0].azimuth == pytest.approx(base.azimuth + 10) and circular[18].elevation == pytest.approx(-51)


def run_worker(name: str) -> dict:
    first_uuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / f"tests/{name}")],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": first_uuid, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_env_frames_match_collector_rig_in_one_uuid_subprocess():
    report = run_worker("_rl_env_worker.py")
    assert report == {
        "same_as_rig": True,
        "trajectory_moves": True,
        "reset_varies_objects": True,
        "pool_matches_single_env": True,
    }


EXPECTED_SCHEDULES = {  # MWM Appendix F category -> user-decided schedule (agent steps)
    "door-open": ("easy", "linear(1.0,0.1,100000)"),
    "peg-unplug-side": ("easy", "linear(1.0,0.1,100000)"),
    "hammer": ("medium", "linear(1.0,0.1,250000)"),
    "peg-insert-side": ("medium", "linear(1.0,0.1,250000)"),
    "bin-picking": ("medium", "linear(1.0,0.1,250000)"),
    "pick-place": ("hard", "linear(1.0,0.1,500000)"),
    "stick-push": ("very hard", "linear(1.0,0.1,500000)"),
    "shelf-place": ("very hard", "linear(1.0,0.1,500000)"),
}


@pytest.mark.parametrize("encoder", ["cnn", "splatter4d"])
def test_every_task_resolves_to_its_difficulty_schedule_for_every_method(tmp_path, encoder):
    from s4d.rl.protocol import resolve_config, task_exploration

    export = write_export(tmp_path / "encoder.pt")
    overrides = [f"vision.export_path={export}"] if encoder == "splatter4d" else []
    for task, (difficulty, expected) in EXPECTED_SCHEDULES.items():
        assert task_exploration(task) == (difficulty, expected)
        cfg = resolve_config(task, encoder, 0, overrides)
        assert cfg["agent"]["stddev_schedule"] == expected and cfg["task_difficulty"] == difficulty
        duration = float(expected.rstrip(")").split(",")[-1])
        assert schedule(cfg["agent"]["stddev_schedule"], 0) == 1.0
        assert schedule(cfg["agent"]["stddev_schedule"], int(duration)) == pytest.approx(0.1)
    with pytest.raises(KeyError):
        task_exploration("reach")
