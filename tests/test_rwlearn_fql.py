"""Equivalence of the PyTorch FQL port (s4d/rwlearn/fql.py) with the official JAX agent (seohongpark/fql @ e8cd16e).

The fixture ``tests/fixtures/rwlearn/fql_official_small.npz`` was produced by the OFFICIAL code
(runs/setup/tools/realworld_fql_20261106/fql_reference_jax.py, separate JAX environment): official ``get_config()``
with 4 x 32 hidden layers (everything else official), one fixed batch of 256, five ``agent.update`` calls. It stores the
initial parameters, the random draws of every update, the official info dicts and the parameters after the first and
last update. Here the port starts from the same parameters, consumes the same draws and must reproduce every loss,
every logged statistic and every parameter (online and target) to float32 round-off.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from s4d.rwlearn import fql

FIXTURE = Path(__file__).parent / "fixtures" / "rwlearn" / "fql_official_small.npz"


def _load():
    z = np.load(FIXTURE)
    meta = json.loads(bytes(z["meta"]).decode())
    return z, meta


def _agent_from_fixture(z, meta):
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()
           if k not in ("action_dim", "ob_dims")}
    agent = fql.FQLAgent.create(0, meta["obs_dim"], meta["action_dim"], cfg)
    fql.load_official_params(agent, {k[len("init/"):]: z[k] for k in z.files if k.startswith("init/")})
    return agent


def _batch(z):
    return {k: torch.from_numpy(z[f"batch/{k}"]) for k in ("observations", "actions", "rewards", "masks",
                                                          "next_observations")}


def _noise(z, u):
    return fql.Noise(**{k: torch.from_numpy(z[f"draws/{u}/{k}"]) for k in ("next_actions", "x0", "t", "distill", "mse")})


def test_fixture_comes_from_the_pinned_official_commit():
    _, meta = _load()
    assert meta["commit"] == fql.FQL_COMMIT
    official = fql.official_config()
    for k, v in official.items():  # everything official except the hidden widths of this small fixture
        if k in ("actor_hidden_dims", "value_hidden_dims"):
            assert meta["config"][k] == [32, 32, 32, 32] and len(v) == 4 and v[0] == 512
        elif k != "encoder":
            assert meta["config"][k] == (list(v) if isinstance(v, tuple) else v), k


def test_parameter_layout_round_trips_with_the_official_tree():
    z, meta = _load()
    agent = _agent_from_fixture(z, meta)
    exported = fql.export_official_params(agent)
    init = {k[len("init/"):]: z[k] for k in z.files if k.startswith("init/")}
    assert set(exported) == set(init)
    for k, v in init.items():
        np.testing.assert_array_equal(exported[k], v, err_msg=k)


def test_updates_reproduce_the_official_losses_statistics_and_parameters():
    z, meta = _load()
    agent = _agent_from_fixture(z, meta)
    batch = _batch(z)
    for u, official in enumerate(meta["infos"]):
        info = agent.update(batch, _noise(z, u))
        assert set(info) == set(official), set(info) ^ set(official)
        for k, v in official.items():
            assert info[k] == pytest.approx(v, rel=2e-4, abs=2e-5), (u, k, info[k], v)
        key = f"params/{u}/"
        if any(name.startswith(key) for name in z.files):
            mine = fql.export_official_params(agent)
            for name in (n for n in z.files if n.startswith(key)):
                np.testing.assert_allclose(mine[name[len(key):]], z[name], rtol=2e-4, atol=2e-6, err_msg=name)


def test_policy_sampling_and_euler_flow_match_the_official_outputs():
    z, meta = _load()
    agent = _agent_from_fixture(z, meta)
    batch = _batch(z)
    for u in range(meta["updates"]):
        agent.update(batch, _noise(z, u))
    obs, noise = batch["observations"], torch.from_numpy(z["check/sample_noise"])
    np.testing.assert_allclose(agent.sample_actions(obs, noise).numpy(), z["check/sample_actions"], rtol=1e-4,
                               atol=1e-5)
    np.testing.assert_allclose(agent.compute_flow_actions(obs, noise).numpy(), z["check/flow_actions"], rtol=1e-4,
                               atol=1e-5)


def test_gradients_flow_only_where_the_official_agent_lets_them():
    torch.manual_seed(0)
    agent = fql.FQLAgent.create(0, 5, 2, {"actor_hidden_dims": (16, 16), "value_hidden_dims": (16, 16)})
    B = 8
    batch = {"observations": torch.randn(B, 5), "actions": torch.rand(B, 2) * 2 - 1, "rewards": torch.randn(B),
             "masks": torch.ones(B), "next_observations": torch.randn(B, 5)}
    noise = agent.draw_noise(batch)
    # critic loss alone: gradient only into the critic
    loss, _ = agent.critic_loss(batch, noise)
    loss.backward()
    assert all(p.grad is not None for p in agent.critic.parameters())
    assert all(p.grad is None for p in agent.actor_bc_flow.parameters())
    assert all(p.grad is None for p in agent.actor_onestep_flow.parameters())
    assert all(p.grad is None for p in agent.target_critic.parameters())
    agent.optimizer.zero_grad(set_to_none=True)
    # actor loss alone: gradients into both flows (Q term through the one-step actions), none into the critic
    loss, _ = agent.actor_loss(batch, noise)
    loss.backward()
    assert all(p.grad is None for p in agent.critic.parameters())
    assert all(p.grad is not None for p in agent.actor_bc_flow.parameters())
    assert all(p.grad is not None for p in agent.actor_onestep_flow.parameters())


def test_target_update_uses_the_pre_update_critic_and_checkpoint_round_trip(tmp_path):
    agent = fql.FQLAgent.create(3, 4, 2, {"actor_hidden_dims": (8,), "value_hidden_dims": (8,), "tau": 0.5})
    B = 4
    batch = {"observations": torch.randn(B, 4), "actions": torch.rand(B, 2) * 2 - 1, "rewards": torch.randn(B),
             "masks": torch.ones(B), "next_observations": torch.randn(B, 4)}
    before = [p.detach().clone() for p in agent.critic.parameters()]
    target_before = [p.detach().clone() for p in agent.target_critic.parameters()]
    agent.update(batch)
    for tp, old, tb in zip(agent.target_critic.parameters(), before, target_before):
        torch.testing.assert_close(tp, 0.5 * old + 0.5 * tb)
    torch.save(agent.state_dict(), tmp_path / "a.pt")
    other = fql.FQLAgent.create(99, 4, 2, {"actor_hidden_dims": (8,), "value_hidden_dims": (8,), "tau": 0.5})
    other.load_state_dict(torch.load(tmp_path / "a.pt", weights_only=False))
    a, b = agent.update(batch), other.update(batch)
    assert a == b  # same parameters, optimiser state and generator state -> identical next update


def test_toy_bandit_prefers_the_better_behaviour_mode():
    """Behavioural check on the toy task of the reference tool (bimodal behaviour; one mode is better). Small widths
    and few steps keep it fast; the full-size official-vs-port comparison is runs/setup/tools/realworld_fql_20261106/
    fql_toy_compare.py (report in docs/realworld_fql.md)."""
    from s4d.rwlearn.fql_toy import toy_bandit

    data = {k: torch.from_numpy(v) for k, v in toy_bandit(4000, 0).items()}
    agent = fql.FQLAgent.create(0, 4, 2, {"actor_hidden_dims": (32, 32), "value_hidden_dims": (32, 32),
                                          "batch_size": 128, "alpha": 1.0, "lr": 1e-3})
    rng = np.random.default_rng(1)
    for _ in range(600):
        idx = torch.from_numpy(rng.integers(len(data["actions"]), size=128))
        agent.update({k: v[idx] for k, v in data.items() if k != "terminals"})
    obs = torch.from_numpy(np.random.default_rng(2).normal(size=(2000, 4)).astype(np.float32))
    onestep = agent.sample_actions(obs)
    flow = agent.compute_flow_actions(obs, torch.randn(2000, 2, generator=torch.Generator().manual_seed(3)))

    def reward(a):
        return float((-((a - 0.5) ** 2).sum(1)).mean())

    assert 0.4 < float((flow.sum(1) > 0).float().mean()) < 0.6  # the BC flow keeps both behaviour modes
    assert float((onestep.sum(1) > 0).float().mean()) > 0.9  # the Q-steered one-step policy picks the better one
    assert reward(onestep) > reward(flow) + 0.5
