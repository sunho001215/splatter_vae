"""Vision-only DrM (2026-10-23 campaign): no proprioception anywhere in the policy (``env.proprio_indices=[]``)."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from s4d.rl.agent import DrMAgent
from s4d.rl.env import MetaWorldCameraEnv
from s4d.rl.evaluate import policy_inputs
from s4d.rl.replay import Replay, replay_iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_rl import tiny_cfg, write_export  # noqa: E402

VISION_ONLY = ["env.proprio_indices=[]", "wandb.tags=[vision_only,no_proprio]"]


def proprio_of(state, indices):
    """The environment's own proprio extraction (``MetaWorldCameraEnv._proprio``) for a given index list."""
    return MetaWorldCameraEnv._proprio(types.SimpleNamespace(proprio_indices=list(indices)), state)


def agent_for(encoder, tmp_path, proprio_dim, device="cpu"):
    export = str(write_export(tmp_path / "encoder.pt")) if encoder == "splatter4d" else None
    torch.manual_seed(0)
    return DrMAgent(tiny_cfg(encoder, export), 4, proprio_dim, torch.device(device))


@pytest.mark.parametrize("encoder", ["convnet", "splatter4d"])
def test_no_network_has_a_proprio_input(tmp_path, encoder):
    agent = agent_for(encoder, tmp_path, 0)
    for net in (agent.actor, agent.critic, agent.critic_target, agent.value_predictor):
        assert net.trunk[0].in_features == agent.encoder.repr_dim, type(net).__name__
    with_proprio = agent_for(encoder, tmp_path, 4)
    assert with_proprio.actor.trunk[0].in_features == agent.encoder.repr_dim + 4


@pytest.mark.parametrize("encoder", ["convnet", "splatter4d"])
def test_simulator_state_cannot_change_outputs_when_images_are_fixed(tmp_path, encoder):
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (2, 9, 32, 32), dtype=np.uint8)
    states = [rng.normal(size=(39,)).astype(np.float32) for _ in range(3)]  # Meta-World state vectors
    for indices, expect_equal in (([], True), ([0, 1, 2, 3], False)):  # vision-only vs the control with proprio
        agent = agent_for(encoder, tmp_path, len(indices))
        agent.train(False)  # the official critics have dropout: compare deterministic eval-mode outputs
        agent.critic_target.train(False)  # (DrMAgent.train leaves the target network as the official code does)
        x = policy_inputs(agent, obs)
        with torch.no_grad():
            feats = agent.encoder(x, is_feature=not agent.use_pixels)
        outs = []
        for state in states:
            p = np.stack([proprio_of(state, indices)] * 2)
            assert p.shape == (2, len(indices))
            pt = torch.as_tensor(p, dtype=torch.float32)
            action = agent.act(x, p, step=5000, eval_mode=True)
            with torch.no_grad():
                a = torch.as_tensor(action, dtype=torch.float32)
                q1, q2 = agent.critic(feats, pt, a)
                v = agent.value_predictor(feats, pt)
                qt = agent.critic_target(feats, pt, a)[0]
            outs.append((action, q1.numpy(), q2.numpy(), v.numpy(), qt.numpy()))
        same = all(all(np.array_equal(o, r) for o, r in zip(out, outs[0])) for out in outs[1:])
        assert same is expect_equal, (indices, encoder)


def latent_replay(agent, proprio_dim, directory=None):
    replay = Replay(directory, agent.encoder.replay_atom_shape, agent.encoder.replay_atom_dtype, proprio_dim, 4, 1000,
                    agent.encoder.replay_atom_frame_stack, 10)
    rng = np.random.default_rng(1)
    for _ in range(3):
        frames = rng.integers(0, 255, (1, 9, 32, 32), dtype=np.uint8)
        atom = policy_inputs(agent, frames)[0].cpu().numpy() if not agent.use_pixels else frames[0, -3:]
        replay.add_initial(atom, np.zeros(proprio_dim, np.float32))
        for t in range(12):
            frames = rng.integers(0, 255, (1, 9, 32, 32), dtype=np.uint8)
            atom = policy_inputs(agent, frames)[0].cpu().numpy() if not agent.use_pixels else frames[0, -3:]
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, atom, np.zeros(proprio_dim, np.float32), t == 11)
    return replay


@pytest.mark.parametrize("encoder", ["convnet", "splatter4d"])
def test_updates_dormant_ratios_and_replay_resume_without_proprio(tmp_path, encoder):
    device = torch.device("cuda")
    agent = agent_for(encoder, tmp_path, 0, "cuda")
    directory = tmp_path / "replay" if encoder == "convnet" else None  # CNN: disk memmaps; frozen: RAM + snapshots
    replay = Replay(directory, agent.encoder.replay_atom_shape, agent.encoder.replay_atom_dtype, 0, 4, 1000,
                    agent.encoder.replay_atom_frame_stack, 10, snapshot_dir=tmp_path / "snap")
    rng = np.random.default_rng(2)
    for _ in range(3):
        frames = rng.integers(0, 255, (1, 9, 32, 32), dtype=np.uint8)
        atom = frames[0, -3:] if agent.use_pixels else policy_inputs(agent, frames)[0].cpu().numpy()
        replay.add_initial(atom, np.zeros(0, np.float32))
        for t in range(12):
            frames = rng.integers(0, 255, (1, 9, 32, 32), dtype=np.uint8)
            atom = frames[0, -3:] if agent.use_pixels else policy_inputs(agent, frames)[0].cpu().numpy()
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, atom, np.zeros(0, np.float32), t == 11)
    assert replay.proprio.shape == (replay.capacity, 0) and not isinstance(replay.proprio, np.memmap)
    meta = replay.checkpoint()
    replay.restore(meta)
    batches = replay_iterator(replay, 16, 0.97, seed=0, device=device)
    for step in range(2, 8, 2):
        metrics = agent.update(batches, step)
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        assert 0 <= metrics["actor_dormant_ratio"] <= 1 and 0 <= metrics["critic_dormant_ratio"] <= 1


@pytest.mark.parametrize("encoder", ["splatter4d", "cnn"])
def test_vision_only_config_changes_only_proprio_and_labels(tmp_path, encoder):
    from s4d.rl.protocol import resolve_config

    over = [f"vision.export_path={write_export(tmp_path / 'e.pt')}"] if encoder == "splatter4d" else []
    ref = resolve_config("hammer", encoder, 2000, over)
    vo = resolve_config("hammer", encoder, 2000, over + VISION_ONLY)

    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
        return out

    a, b = flat(ref), flat(vo)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"env.proprio_indices", "wandb.tags"}
    assert b["env.proprio_indices"] == [] and a["env.proprio_indices"] == [0, 1, 2, 3]
    assert b["wandb.tags"] == ["vision_only", "no_proprio"]
    if encoder == "cnn":
        assert b.get("agent.random_shift", True) is True  # the CNN keeps its standard random-shift augmentation
