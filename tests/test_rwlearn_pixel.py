"""End-to-end CNN + FQL (s4d/rwlearn/pixel.py) against the OFFICIAL pixel code (seohongpark/fql @ e8cd16e).

Fixture ``tests/fixtures/rwlearn/fql_official_pixel.npz`` (runs/setup/tools/realworld_fql_20261106/fql_reference_jax.py
``pixel``, separate JAX environment): the parameter layout of ``impala_small`` at 128 x 128 x 9 and its output on a
small batch (deterministic parameter values), ``flax.linen.max_pool`` SAME, ``batched_random_crop``, the frame
stacking of ``Dataset.sample(frame_stack=3)``, and three ``agent.update`` calls of an official FQL agent with the
official ``ImpalaEncoder`` (4 x 4 stacks, (32,) MLP) on uint8 images. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from s4d.rwlearn import fql, pixel
from s4d.rwlearn.transitions import history_indices

FIXTURE = Path(__file__).parent / "fixtures" / "rwlearn" / "fql_official_pixel.npz"


@pytest.fixture(scope="module")
def fx():
    z = np.load(FIXTURE)
    return z, json.loads(bytes(z["meta"]).decode())


def _official_layout(tensor, kind):
    return tuple(tensor.permute(2, 3, 1, 0).shape) if kind == "conv_kernel" else \
        tuple(tensor.T.shape) if kind == "kernel" else tuple(tensor.shape)


def _load(module_slots, values):
    with torch.no_grad():
        for key, tensor, kind in module_slots:
            arr = torch.as_tensor(values[key])
            tensor.copy_(arr.T if kind == "kernel" else arr.permute(3, 2, 0, 1) if kind == "conv_kernel" else arr)


def deterministic_params(shapes: dict) -> dict:  # = fql_reference_jax.deterministic_params
    return {k: (0.2 * np.sin(0.37 * np.arange(int(np.prod(shp))) + j)).astype(np.float32).reshape(shp)
            for j, (k, shp) in enumerate(sorted(shapes.items()))}


def test_fixture_is_from_the_pinned_official_commit(fx):
    _, meta = fx
    assert meta["commit"] == fql.FQL_COMMIT and meta["mode"] == "pixel"
    assert meta["impala_small"]["stack_sizes"] == list(pixel.IMPALA_SMALL["stack_sizes"])
    assert meta["impala_small"]["num_blocks"] == pixel.IMPALA_SMALL["num_blocks"]
    assert meta["impala_small"]["mlp_hidden_dims"] == list(pixel.IMPALA_SMALL["mlp_hidden_dims"])


def test_impala_small_has_the_official_parameter_layout_at_128px(fx):
    _, meta = fx
    enc = pixel.ImpalaEncoder(9, (128, 128), **pixel.IMPALA_SMALL)
    mine = {k: list(_official_layout(t, kind)) for k, t, kind in enc.official_slots("")}
    official = {"/" + k: v for k, v in meta["impala_small"]["param_shapes_128x128x9"].items()}
    assert mine == official
    assert enc.flat_dim == 16 * 16 * 32 and enc.out_dim == 512


def test_impala_small_forward_equals_the_official_encoder(fx):
    z, meta = fx
    shapes = meta["impala_small"]["param_shapes_8x12x9"]
    enc = pixel.ImpalaEncoder(9, (8, 12), **pixel.IMPALA_SMALL)
    _load(enc.official_slots(""), {"/" + k: v for k, v in deterministic_params(shapes).items()})
    out = enc(torch.from_numpy(z["impala/input"]))
    np.testing.assert_allclose(out.detach().numpy(), z["impala/output"], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("hw", ["7x6", "8x8", "5x9"])
def test_max_pool_same_equals_flax(fx, hw):
    z, _ = fx
    x = torch.from_numpy(z[f"pool/{hw}/input"]).permute(0, 3, 1, 2)
    y = pixel.max_pool_same(x).permute(0, 2, 3, 1).numpy()
    np.testing.assert_array_equal(y, z[f"pool/{hw}/output"])


def test_random_crop_equals_the_official_augmentation(fx):
    z, _ = fx
    out = pixel.random_crop(torch.from_numpy(z["crop/input"]), torch.from_numpy(z["crop/froms"][:, :2]), 3)
    assert out.dtype == torch.uint8
    np.testing.assert_array_equal(out.numpy(), z["crop/output"])


def test_frame_stacking_equals_the_official_dataset(fx):
    """Official: [ob[t-2], ob[t-1], ob[t]] padded with the episode's first observation; next = [ob[t-1], ob[t],
    next_ob[t]]. Ours: the 3-step history of the run (episode) ending at k, next = the history ending at k + 1."""
    z, _ = fx
    term = z["stack/terminals"]
    run = np.concatenate([[0], np.cumsum(term[:-1] > 0)])
    ks = z["stack/idxs"]
    np.testing.assert_array_equal(history_indices(run, ks, 3), z["stack/observations"])
    nxt = history_indices(run, np.minimum(ks + 1, len(ks) - 1), 3)
    # the official next_observations of the toy store index min(t + 1, N - 1) as its last frame; inside an episode the
    # next history is the history ending at t + 1 (our transitions never cross a run end, where they differ)
    inside = np.r_[run[1:] == run[:-1], False]
    np.testing.assert_array_equal(nxt[inside], z["stack/next_observations"][inside])


def _agent(z, meta):
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()
           if k not in ("action_dim", "ob_dims", "encoder")}
    tiny = meta["impala_tiny"]
    factory = lambda: pixel.ImpalaEncoder(9, (8, 12), tiny["stack_sizes"], tiny["num_blocks"],  # noqa: E731
                                          tiny["mlp_hidden_dims"])
    agent = fql.FQLAgent.create(0, factory().out_dim, meta["action_dim"], cfg, encoder=factory)
    fql.load_official_params(agent, {k[len("init/"):]: z[k] for k in z.files if k.startswith("init/")})
    return agent


def test_pixel_agent_parameter_layout_round_trips(fx):
    z, meta = fx
    agent = _agent(z, meta)
    exported = fql.export_official_params(agent)
    init = {k[len("init/"):]: z[k] for k in z.files if k.startswith("init/")}
    assert set(exported) == set(init)
    for k, v in init.items():
        np.testing.assert_array_equal(exported[k], v, err_msg=k)


def test_pixel_agent_updates_reproduce_the_official_agent(fx):
    """Encoders trained jointly (every network's own encoder; the BC-flow encoder shared with the flow target; the
    target critic's encoder Polyak-updated): losses, gradient statistics and every parameter after three updates."""
    z, meta = fx
    agent = _agent(z, meta)
    batch = {k: torch.from_numpy(z[f"batch/{k}"]) for k in ("observations", "actions", "rewards", "masks",
                                                            "next_observations")}
    enc0 = [p.detach().clone() for p in agent.critic.encoder.parameters()]
    for u, official in enumerate(meta["infos"]):
        noise = fql.Noise(**{k: torch.from_numpy(z[f"draws/{u}/{k}"]) for k in ("next_actions", "x0", "t", "distill",
                                                                                "mse")})
        info = agent.update(batch, noise)
        assert set(info) == set(official)
        for k, v in official.items():
            assert info[k] == pytest.approx(v, rel=5e-4, abs=5e-5), (u, k, info[k], v)
    assert any(not torch.equal(a, b) for a, b in zip(enc0, agent.critic.encoder.parameters()))  # not frozen
    mine = fql.export_official_params(agent)
    last = meta["updates"] - 1
    for k in mine:
        ref = z[f"params/{last}/{k}"]
        np.testing.assert_allclose(mine[k], ref, rtol=2e-4, atol=2e-5, err_msg=k)
    obs = batch["observations"]
    np.testing.assert_allclose(agent.sample_actions(obs, torch.from_numpy(z["check/sample_noise"])).numpy(),
                               z["check/sample_actions"], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(agent.compute_flow_actions(obs, torch.from_numpy(z["check/sample_noise"])).numpy(),
                               z["check/flow_actions"], rtol=2e-4, atol=2e-5)


def test_augment_is_one_coin_per_batch_with_shared_offsets():
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (6, 10, 10, 9), dtype=torch.uint8, generator=g)
    batch = {"actions": torch.zeros(6, 7), "observations": {"pixels": img, "proprio": torch.ones(6, 10)},
             "next_observations": {"pixels": img.clone(), "proprio": torch.ones(6, 10)}}
    hits = 0
    for s in range(40):
        out = pixel.augment({k: (dict(v) if isinstance(v, dict) else v) for k, v in batch.items()},
                            torch.Generator().manual_seed(s), p_aug=0.5)
        torch.testing.assert_close(out["observations"]["pixels"], out["next_observations"]["pixels"])  # same offsets
        torch.testing.assert_close(out["observations"]["proprio"], batch["observations"]["proprio"])  # 2-D untouched
        hits += int(not torch.equal(out["observations"]["pixels"], img))
    assert 5 < hits < 35  # p_aug 0.5 per batch
    same = pixel.augment(dict(batch), torch.Generator().manual_seed(0), p_aug=None)
    assert same["observations"]["pixels"] is img
