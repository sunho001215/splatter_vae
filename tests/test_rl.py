"""DrM port: dormant ratio, perturbation, exploitation target, updates per encoder, replay, cameras and env."""

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
from s4d.rl.agent import DrMAgent, dormant_ratio, perturb, schedule
from s4d.rl.env import TRAIN_PATHS, check_pretraining_spacing, lateral_offsets, trajectory_path
from s4d.rl.evaluate import episode_seeds, evaluation_groups, policy_inputs
from s4d.rl.replay import Replay, replay_iterator

REPO = Path(__file__).resolve().parents[1]
GAMMA = 0.5
sys.path.insert(0, str(REPO / "tests"))
import _drm_official as official  # noqa: E402


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
    """Frames hold ``value % 256``; proprio and actions hold the exact float value of the same state."""
    obs, proprio, action, reward, discount, next_obs, next_proprio = (np.asarray(x) for x in batch)
    frames = obs.reshape(len(obs), 3, 3, 4, 4)[:, :, 0, 0, 0].astype(int)
    nxt = next_obs.reshape(len(obs), 3, 3, 4, 4)[:, :, 0, 0, 0].astype(int)
    now, later = proprio[:, 0].astype(int), next_proprio[:, 0].astype(int)
    assert np.all(frames[:, -1] == now % 256) and np.all(nxt[:, -1] == later % 256), "atoms match their states"
    assert np.all(later - now == nstep), "next state is n steps later in the same episode"
    back = [tuple(row) for row in (frames[:, -1:] - frames) % 256]  # frames looked back, oldest first
    assert set(back) <= {(2, 1, 0), (1, 1, 0), (0, 0, 0)}, "padded with the first frame, never crosses an episode"
    assert np.all(action[:, 0] == now + 1), "action leaves the sampled state"
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


def tiny_cfg(encoder_type="convnet", export_path=None, **agent):
    import yaml

    base = yaml.safe_load((REPO / "configs/rl/base.yaml").read_text())
    return {
        "env": {"frame_stack": 3, "image_height": 32, "image_width": 32},
        "vision": {"encoder_type": encoder_type, "export_path": export_path},
        "agent": {**base["agent"], "feature_dim": 8, "hidden_dim": 16, **agent},
    }


def pixel_replay(tmp_path, episodes=3, length=12) -> Replay:
    replay = Replay(tmp_path / "replay", (3, 32, 32), np.uint8, 4, 4, 1000, 3, 10)
    rng = np.random.default_rng(0)
    for _ in range(episodes):
        replay.add_initial(rng.integers(0, 255, (3, 32, 32), dtype=np.uint8), rng.normal(size=4))
        for t in range(length):
            image = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, image, rng.normal(size=4), t == length - 1)
    return replay


def write_export(path: Path, num_frames: int = 3) -> Path:
    cfg = EncoderConfig(image_height=32, image_width=32, width=32, depth=2, heads=2, slot_dim=16, num_frames=num_frames)
    torch.manual_seed(0)
    encoder = Encoder(cfg)
    payload = {"format": "splatter4d-encoder-v1", "encoder_config": asdict(cfg), "state_dict": encoder.state_dict()}
    torch.save({**payload, "step": 0, "frame_strides": [2, 4, 6]}, path)
    return path


def known_network() -> torch.nn.Sequential:
    """Layer 1 outputs |.| = (0, 1, 1, 1) on ones(2); layer 2 outputs (0, 3): 2 of 6 units are dormant."""
    net = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
        net[2].bias.zero_()
    return net


def test_dormant_ratio_on_a_known_network_and_against_the_official_code():
    net = known_network()
    assert dormant_ratio(net, torch.ones(5, 2)) == pytest.approx(2 / 6)
    assert official.cal_dormant_ratio(net, torch.ones(5, 2)) == pytest.approx(2 / 6)
    torch.manual_seed(0)
    agent = DrMAgent(tiny_cfg(), 4, 4, torch.device("cpu"))
    obs, proprio = torch.randn(64, agent.encoder.repr_dim), torch.randn(64, 4)
    for model, inputs in ((agent.actor, (obs, proprio, 0)), (agent.critic, (obs, proprio, torch.rand(64, 4)))):
        mine = dormant_ratio(model, *inputs, percentage=0.025)
        assert mine == official.cal_dormant_ratio(model, *inputs, percentage=0.025)
    assert not any(module._forward_hooks for module in agent.actor.modules()), "hooks are removed"


def test_perturbation_follows_the_official_formula():
    torch.manual_seed(1)
    agent = DrMAgent(tiny_cfg(), 4, 4, torch.device("cpu"))
    for ratio, expected in ((1.0, 0.2), (0.3, 0.4), (0.01, 0.95)):
        agent.dormant_ratio = ratio
        assert agent.perturb_factor == pytest.approx(expected)  # min(max(0.2, 1 - 2 r), 0.95)
    for net in (agent.actor, agent.critic, agent.value_predictor, agent.encoder):
        twin = copy.deepcopy(net)
        opt = torch.optim.Adam(net.parameters())
        twin_opt = torch.optim.Adam(twin.parameters())
        opt.state[next(net.parameters())]["step"] = torch.tensor(3.0)
        torch.manual_seed(5)
        perturb(net, opt, 0.4)
        torch.manual_seed(5)
        official.perturb(twin, twin_opt, 0.4)
        for (name, a), b in zip(net.named_parameters(), twin.parameters()):
            assert torch.equal(a, b), name
        assert len(opt.state) == 0, "optimizer state is reset"
    before = copy.deepcopy(agent.actor)
    torch.manual_seed(9)
    fresh = copy.deepcopy(agent.actor).apply(official.weight_init)
    torch.manual_seed(9)
    perturb(agent.actor, agent.actor_opt, 0.4)
    torch.testing.assert_close(agent.actor.trunk[0].weight, 0.4 * before.trunk[0].weight + 0.6 * fresh.trunk[0].weight)
    assert torch.equal(agent.actor.trunk[1].weight, before.trunk[1].weight), "LayerNorm is kept"
    conv = [p.clone() for p in agent.encoder.parameters()]
    agent.perturb()
    assert all(torch.equal(a, b) for a, b in zip(conv, agent.encoder.parameters())), "CNN has no Linear: unchanged"


def test_awake_exploration_schedule():
    agent = DrMAgent(tiny_cfg(), 4, 4, torch.device("cpu"))
    agent.dormant_ratio = 0.5
    assert agent.stddev(10) == pytest.approx(1 / (1 + math.exp(-10 * 0.3)))
    agent.dormant_ratio, agent.awaken_step = 0.1, 1000
    assert agent.stddev(1000) == pytest.approx(1.0)  # linear(1.0,0.1,500000) restarts at awakening
    assert agent.stddev(251000) == pytest.approx(0.55)
    assert agent.stddev(10**7) == pytest.approx(max(0.1, 1 / (1 + math.exp(1.0))))
    actions = agent.act(torch.zeros(64, 9, 32, 32, dtype=torch.uint8), np.zeros((64, 4)), step=10, eval_mode=False)
    assert actions.std() > 0.5, "uniform actions before num_expl_steps"


def test_exploitation_target_and_expectile_match_the_official_code_on_a_fixed_batch():
    torch.manual_seed(2)
    agent = DrMAgent(tiny_cfg(), 4, 4, torch.device("cpu"))
    agent.dormant_ratio, agent.awaken_step = 0.1, 0
    g = torch.Generator().manual_seed(3)
    obs, nxt = torch.randn(32, agent.encoder.repr_dim, generator=g), torch.randn(32, agent.encoder.repr_dim, generator=g)
    proprio, action = torch.randn(32, 4, generator=g), torch.rand(32, 4, generator=g) * 2 - 1
    reward, discount = torch.randn(32, 1, generator=g), torch.full((32, 1), 0.97**10)
    for module in (agent.critic, agent.critic_target):
        module.eval()  # disable dropout so both computations see the same network function
    torch.manual_seed(4)
    mine = agent.critic_target_value(nxt, proprio, reward, discount, step=100)
    torch.manual_seed(4)
    expected = official.target_q(agent, nxt, proprio, reward, discount, step=100)
    assert torch.equal(mine, expected)
    with torch.no_grad():
        explore = torch.min(*agent.critic_target(nxt, proprio, torch.zeros(32, 4)))
    assert agent.target_lambda == 0.5 and not torch.allclose(explore, agent.value_predictor(nxt, proprio))
    loss = official.predictor_loss(agent, obs, proprio, action)
    assert agent.update_predictor(obs, proprio, action)["predictor_loss"] == pytest.approx(float(loss), rel=1e-6)


@pytest.mark.parametrize("encoder", ["convnet", "splatter4d"])
def test_end_to_end_updates_with_perturbation_for_every_encoder_type(tmp_path, encoder):
    device = torch.device("cuda")
    export = str(write_export(tmp_path / "encoder.pt")) if encoder == "splatter4d" else None
    torch.manual_seed(0)
    agent = DrMAgent(tiny_cfg(encoder, export, dormant_perturb_interval=4), 4, 4, device)
    if encoder == "convnet":
        batches = replay_iterator(pixel_replay(tmp_path), 16, 0.97, num_workers=1, seed=0)
    else:
        latents = Replay(None, agent.encoder.replay_atom_shape, np.float16, 4, 4, 1000, 1, 10)
        rng = np.random.default_rng(0)
        for _ in range(3):
            frame_stack = rng.integers(0, 255, (1, 9, 32, 32), dtype=np.uint8)
            latents.add_initial(policy_inputs(agent, frame_stack)[0].cpu().numpy(), rng.normal(size=4))
            for t in range(12):
                frame_stack = rng.integers(0, 255, (1, 9, 32, 32), dtype=np.uint8)
                latent = policy_inputs(agent, frame_stack)[0].cpu().numpy()
                latents.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, latent, rng.normal(size=4), t == 11)
        batches = replay_iterator(latents, 16, 0.97, num_workers=0, seed=0)
    frozen = {k: v.clone() for k, v in agent.encoder.backbone.state_dict().items()}
    actor = [p.clone() for p in agent.actor.parameters()]
    perturbed = []
    for step in range(2, 10, 2):
        metrics = agent.update(batches, step)
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        assert 0 <= metrics["actor_dormant_ratio"] <= 1 and 0 <= metrics["critic_dormant_ratio"] <= 1
        perturbed.append("perturb_factor" in metrics)
    assert perturbed == [False, True, False, True]
    assert any(not torch.equal(a, b) for a, b in zip(actor, agent.actor.parameters()))
    if encoder == "splatter4d":
        assert agent.augment_pixels is False and not any(p.requires_grad for p in agent.encoder.backbone.parameters())
        for key, value in agent.encoder.backbone.state_dict().items():
            assert torch.equal(value, frozen[key]), f"frozen encoder changed: {key}"
    clone = DrMAgent(tiny_cfg(encoder, export), 4, 4, device)
    clone.load_state_dict(agent.state_dict())
    assert clone.awaken_step == agent.awaken_step and clone.dormant_ratio == agent.dormant_ratio
    action = clone.act(policy_inputs(clone, np.zeros((2, 9, 32, 32), np.uint8)), np.zeros((2, 4)), 5000, True)
    assert action.shape == (2, 4) and np.all(np.abs(action) <= 1)


def test_frozen_latent_replay_update_equals_encoding_stored_frames(tmp_path):
    """Online per-step encoding (batch 1, stored fp16 in RAM) vs encoding the stored frames at update time."""
    device = torch.device("cuda")
    cfg = tiny_cfg("splatter4d", str(write_export(tmp_path / "encoder.pt")))
    torch.manual_seed(1)
    agent = DrMAgent(cfg, 4, 4, device)
    latents = Replay(None, agent.encoder.replay_atom_shape, np.float16, 4, 4, 1000, 1, 10)
    pixels = Replay(tmp_path / "pixels", (3, 32, 32), np.uint8, 4, 4, 1000, 3, 10)
    rng = np.random.default_rng(0)
    for _ in range(4):
        frames = [rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)] * 3
        prop = rng.normal(size=4).astype(np.float32)
        latents.add_initial(policy_inputs(agent, np.concatenate(frames)[None])[0].cpu().numpy(), prop)
        pixels.add_initial(frames[-1], prop)
        for t in range(20):
            frames = frames[1:] + [rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)]
            action, reward, prop = rng.uniform(-1, 1, 4), float(rng.normal()), rng.normal(size=4).astype(np.float32)
            latent = policy_inputs(agent, np.concatenate(frames)[None])[0].cpu().numpy()
            latents.add(action, reward, 1.0, latent, prop, t == 19)
            pixels.add(action, reward, 1.0, frames[-1], prop, t == 19)
    from_ram = latents.sample(64, np.random.default_rng(5), 0.97)
    from_pixels = pixels.sample(64, np.random.default_rng(5), 0.97)
    for i in (1, 2, 3, 4, 6):
        assert np.array_equal(from_ram[i], from_pixels[i])
    encode = agent.encoder.extract_cacheable_feature
    rebuilt_latents = [encode(torch.as_tensor(from_pixels[i], device=device)).cpu().numpy() for i in (0, 5)]
    for stored, recomputed in zip((from_ram[0], from_ram[5]), rebuilt_latents):
        np.testing.assert_allclose(stored.astype(np.float32), recomputed.astype(np.float32), atol=2e-2, rtol=0)
    for module in (agent.critic, agent.critic_target):
        module.eval()  # dropout off: compare the update functions, not dropout masks
    twin = copy.deepcopy(agent)
    rebuilt = list(from_ram)
    rebuilt[0], rebuilt[5] = rebuilt_latents
    torch.manual_seed(7)
    agent.update(iter([tuple(torch.as_tensor(x) for x in from_ram)]), step=50)
    torch.manual_seed(7)
    twin.update(iter([tuple(torch.as_tensor(x) for x in rebuilt)]), step=50)
    for net in ("actor", "critic", "value_predictor"):
        for a, b in zip(getattr(agent, net).parameters(), getattr(twin, net).parameters()):
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
        "seeded_reset_history_free": True,
    }


@pytest.mark.parametrize("encoder", ["cnn", "splatter4d"])
def test_every_task_resolves_to_the_official_drm_metaworld_settings(tmp_path, encoder):
    from s4d.rl.protocol import resolve_config, task_settings

    export = write_export(tmp_path / "encoder.pt")
    overrides = [f"vision.export_path={export}"] if encoder == "splatter4d" else []
    official_agent = {
        "lr": 1e-4,
        "critic_target_tau": 0.01,
        "dormant_threshold": 0.025,
        "target_dormant_ratio": 0.2,
        "target_lambda": 0.5,
        "dormant_temp": 10,
        "dormant_perturb_interval": 100000,
        "min_perturb_factor": 0.2,
        "max_perturb_factor": 0.95,
        "perturb_rate": 2,
        "num_expl_steps": 2000,
        "stddev_schedule": "linear(1.0,0.1,500000)",
        "stddev_clip": 0.3,
        "expectile": 0.9,
        "hidden_dim": 1024,
    }
    tasks = (
        "door-open",
        "peg-unplug-side",
        "hammer",
        "peg-insert-side",
        "bin-picking",
        "pick-place",
        "stick-push",
        "shelf-place",
    )
    for task in tasks:
        assert task_settings(task)["drm_overrides"] == {}
        cfg = resolve_config(task, encoder, 0, overrides)
        assert {k: cfg["agent"][k] for k in official_agent} == official_agent
        assert cfg["train"]["nstep"] == 10 and cfg["train"]["discount"] == pytest.approx(0.97)
        assert cfg["train"]["num_seed_steps"] == 2000 and cfg["train"]["time_limit_continuation"] == 1.0
        assert cfg["agent"]["feature_dim"] == (50 if encoder == "cnn" else 256)
        assert cfg["algorithm"] == "drm"
