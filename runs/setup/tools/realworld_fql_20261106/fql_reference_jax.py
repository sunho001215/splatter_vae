"""Reference outputs of the OFFICIAL Flow Q-Learning agent (JAX) for the PyTorch port's equivalence checks.

Runs only in a separate JAX environment (never the repository venv), on the official checkout pinned in
``s4d/rwlearn/fql.py::FQL_COMMIT``; the commit is verified before anything is imported. Nothing here trains a policy
on campaign data. Two modes:

* ``small``: official ``get_config()`` with narrower hidden layers (4 x 32 instead of 4 x 512, everything else
  official, batch 256); a fixed random batch; ``--updates`` official ``agent.update`` calls. Writes the initial and
  per-update parameters, the random draws each update consumed (reproduced from the agent's PRNG key in the official
  split order and cross-checked against the official losses by the torch test), the official info dicts, plus
  ``sample_actions`` / ``compute_flow_actions`` outputs. -> tests/fixtures/rwlearn/fql_official_small.npz
* ``pixel``: the official pixel path. (a) the parameter names and shapes of ``impala_small`` (``encoder_modules``) on
  128 x 128 x 9 inputs (the real-world configuration) and its output on a 8 x 12 x 9 uint8 batch; (b)
  ``flax.linen.max_pool`` 3x3 / 2 SAME on odd and even sizes; (c) the official ``batched_random_crop`` on a fixed batch
  and offsets; (d) the official ``Dataset.sample`` frame stacking (``frame_stack=3``) on a toy dataset whose
  observations encode their index; (e) a full official FQL agent with the official ``ImpalaEncoder`` class at
  stack sizes (4, 4), one block and a (32,) MLP (registered as ``encoder_modules['impala_tiny']`` in this process
  only, to keep the fixture small, like the 4 x 32 hidden layers) on 8 x 12 x 9 uint8 observations: initial
  parameters, per-update draws, infos and final parameters of three ``agent.update`` calls.
  -> tests/fixtures/rwlearn/fql_official_pixel.npz
* ``toy``: the full official configuration (4 x 512, batch 256, lr 3e-4, alpha 10, flow_steps 10, discount 0.99,
  tau 0.005, q_agg mean) on a one-step bimodal bandit (``toy_bandit``); ``--steps`` updates on batches drawn by
  ``numpy.random.default_rng(seed)``. Writes init params, the per-step draws and infos, and the final one-step-policy
  statistics -> a scratch .npz consumed by ``fql_toy_compare.py`` (torch side, repository venv).

    <jaxenv>/bin/python -I fql_reference_jax.py --fql-src <checkout> small \
        --out tests/fixtures/rwlearn/fql_official_small.npz
    <jaxenv>/bin/python -I fql_reference_jax.py --fql-src <checkout> toy --steps 3000 --out <scratch>/toy.npz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FQL_COMMIT = "e8cd16eb490332924dfa2492097219f181765933"


def toy_bandit(n: int, seed: int) -> dict:
    """One-step bandit with a bimodal behaviour policy: context ~ N(0, I_4); the behaviour picks mode +1 or -1 with
    probability 1/2 and acts m * (0.5, 0.5) + 0.1 N(0, I_2) (clipped to [-1, 1]); reward -|a - (0.5, 0.5)|^2, so the
    +1 mode is better; every transition terminal (mask 0). BC samples both modes, FQL should prefer +1."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 4)).astype(np.float32)
    mode = np.where(rng.random(n) < 0.5, 1.0, -1.0)[:, None]
    act = np.clip(mode * 0.5 + 0.1 * rng.normal(size=(n, 2)), -1, 1).astype(np.float32)
    rew = -np.sum((act - 0.5) ** 2, axis=1).astype(np.float32)
    return {"observations": obs, "actions": act, "rewards": rew, "masks": np.zeros(n, np.float32),
            "next_observations": obs.copy(), "terminals": np.ones(n, np.float32)}


def deterministic_params(shapes: dict) -> dict:
    """Parameter values from a closed formula (sorted names): 0.2 sin(0.37 i + j) for element i of the j-th array."""
    return {k: (0.2 * np.sin(0.37 * np.arange(int(np.prod(shp))) + j)).astype(np.float32).reshape(shp)
            for j, (k, shp) in enumerate(sorted(shapes.items()))}


def flat_params(tree) -> dict:
    import jax

    return {"/".join(str(getattr(p, "key", p)) for p in path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def update_draws(rng_key, batch_size: int, action_dim: int) -> dict:
    """The draws ``FQLAgent.update`` consumes from ``agent.rng``, in the official split order."""
    import jax

    _, rng = jax.random.split(rng_key)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    _, sample_rng = jax.random.split(critic_rng)
    action_seed, _ = jax.random.split(sample_rng)
    next_actions = jax.random.normal(action_seed, (batch_size, action_dim))
    rng, x_rng, t_rng = jax.random.split(actor_rng, 3)
    x0 = jax.random.normal(x_rng, (batch_size, action_dim))
    t = jax.random.uniform(t_rng, (batch_size, 1))
    rng, noise_rng = jax.random.split(rng)
    distill = jax.random.normal(noise_rng, (batch_size, action_dim))
    action_seed, _ = jax.random.split(rng)
    mse = jax.random.normal(action_seed, (batch_size, action_dim))
    return {k: np.asarray(v, np.float32) for k, v in
            dict(next_actions=next_actions, x0=x0, t=t, distill=distill, mse=mse).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fql-src", required=True, type=Path)
    ap.add_argument("mode", choices=("small", "toy", "pixel"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    head = subprocess.run(["git", "-C", str(args.fql_src), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(args.fql_src), "status", "--porcelain"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if head != FQL_COMMIT or dirty:
        raise SystemExit(f"official checkout must be clean at {FQL_COMMIT}, got {head} (dirty: {bool(dirty)})")
    sys.path.insert(0, str(args.fql_src.resolve()))
    import jax
    import jax.numpy as jnp
    from agents.fql import FQLAgent, get_config

    cfg = get_config()
    out: dict = {}
    if args.mode == "pixel":
        import functools

        import flax.linen as nn
        from utils.datasets import Dataset, batched_random_crop
        from utils.encoders import ImpalaEncoder, encoder_modules

        rng = np.random.default_rng(args.seed)
        # (a) impala_small: layout at the real-world input size, values on a small input
        enc = encoder_modules["impala_small"]()
        shapes = jax.tree_util.tree_map(lambda a: list(a.shape), jax.eval_shape(
            enc.init, jax.random.PRNGKey(0), jnp.zeros((1, 128, 128, 9), jnp.uint8))["params"])
        impala_small_shapes = {"/".join(str(getattr(q, "key", q)) for q in path): leaf for path, leaf in
                               jax.tree_util.tree_flatten_with_path(shapes, is_leaf=lambda a: isinstance(a, list))[0]}
        x = rng.integers(0, 256, (4, 8, 12, 9), dtype=np.uint8)
        tree = enc.init(jax.random.PRNGKey(7), x)["params"]
        # deterministic parameter VALUES (formula ``deterministic_params``, regenerated by the torch test), so the
        # fixture stores no parameters for this layout check
        flat = flat_params(tree)
        det = deterministic_params({k: v.shape for k, v in flat.items()})
        paths = jax.tree_util.tree_flatten_with_path(tree)[0]
        leaves = [jnp.asarray(det["/".join(str(getattr(q, "key", q)) for q in path)]) for path, _ in paths]
        params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tree), leaves)
        out["impala/input"] = x
        out["impala/output"] = np.asarray(enc.apply({"params": params}, x), np.float32)
        impala_small_small_input_shapes = {k: list(v.shape) for k, v in flat.items()}
        encoder_modules["impala_tiny"] = functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4),
                                                           mlp_hidden_dims=(32,))
        # (b) max_pool SAME
        for h, w in ((7, 6), (8, 8), (5, 9)):
            y = rng.normal(size=(2, h, w, 3)).astype(np.float32)
            out[f"pool/{h}x{w}/input"] = y
            out[f"pool/{h}x{w}/output"] = np.asarray(nn.max_pool(jnp.asarray(y), window_shape=(3, 3),
                                                                  padding="SAME", strides=(2, 2)))
        # (c) random crop
        imgs = rng.integers(0, 256, (4, 8, 8, 9), dtype=np.uint8)
        froms = np.concatenate([rng.integers(0, 7, (4, 2)), np.zeros((4, 1), np.int64)], 1)
        out["crop/input"], out["crop/froms"] = imgs, froms
        out["crop/output"] = np.asarray(batched_random_crop(imgs, froms, 3))
        # (d) frame stacking: observation value = its index; episodes end at terminals
        N = 40
        terminals = np.zeros(N, np.float32)
        terminals[[9, 10, 24, N - 1]] = 1  # episodes [0..9], [10], [11..24], [25..39]
        obs = np.arange(N, dtype=np.uint8).reshape(N, 1, 1, 1)
        nxt = np.minimum(np.arange(N) + 1, N - 1).astype(np.uint8).reshape(N, 1, 1, 1)
        ds = Dataset.create(observations=obs, next_observations=nxt, terminals=terminals,
                            actions=np.zeros((N, 1), np.float32))
        ds.frame_stack = 3
        idxs = np.arange(N)
        b = ds.sample(N, idxs=idxs)
        out["stack/terminals"], out["stack/idxs"] = terminals, idxs
        out["stack/observations"] = np.asarray(b["observations"]).reshape(N, -1)
        out["stack/next_observations"] = np.asarray(b["next_observations"]).reshape(N, -1)
        # (e) full agent with the official ImpalaEncoder (impala_tiny, registered above)
        cfg["actor_hidden_dims"] = (32, 32, 32, 32)
        cfg["value_hidden_dims"] = (32, 32, 32, 32)
        cfg["encoder"] = "impala_tiny"
        cfg["batch_size"] = 32
        B, action_dim = int(cfg["batch_size"]), 3
        batch = {"observations": rng.integers(0, 256, (B, 8, 12, 9), dtype=np.uint8),
                 "actions": rng.uniform(-1, 1, size=(B, action_dim)).astype(np.float32),
                 "rewards": rng.normal(size=B).astype(np.float32),
                 "masks": (rng.random(B) < 0.9).astype(np.float32),
                 "next_observations": rng.integers(0, 256, (B, 8, 12, 9), dtype=np.uint8)}
        agent = FQLAgent.create(args.seed, batch["observations"][:1], batch["actions"][:1], dict(cfg))
        out.update({f"init/{k}": v for k, v in flat_params(agent.network.params).items()})
        out.update({f"batch/{k}": v for k, v in batch.items()})
        infos = []
        jb = {k: jnp.asarray(v) for k, v in batch.items()}
        for u in range(args.updates):
            draws = update_draws(agent.rng, B, action_dim)
            out.update({f"draws/{u}/{k}": v for k, v in draws.items()})
            agent, info = agent.update(jb)
            infos.append({k: float(v) for k, v in info.items()})
        out.update({f"params/{args.updates - 1}/{k}": v for k, v in flat_params(agent.network.params).items()})
        key = jax.random.PRNGKey(123)
        action_seed, _ = jax.random.split(key)
        out["check/sample_noise"] = np.asarray(jax.random.normal(action_seed, (B, action_dim)), np.float32)
        out["check/sample_actions"] = np.asarray(agent.sample_actions(jb["observations"], seed=key), np.float32)
        out["check/flow_actions"] = np.asarray(agent.compute_flow_actions(jb["observations"],
                                                                          jnp.asarray(out["check/sample_noise"])))
        meta = {"mode": "pixel", "commit": head, "config": {k: (list(v) if isinstance(v, tuple) else v)
                                                           for k, v in dict(cfg).items() if k not in ("ob_dims",)},
                "action_dim": action_dim, "updates": args.updates, "infos": infos, "jax": jax.__version__,
                "impala_small": {"stack_sizes": [16, 32, 32], "num_blocks": 1, "mlp_hidden_dims": [512],
                                 "param_shapes_128x128x9": impala_small_shapes,
                                 "param_shapes_8x12x9": impala_small_small_input_shapes},
                "impala_tiny": {"stack_sizes": [4, 4], "num_blocks": 1, "mlp_hidden_dims": [32]}}
    elif args.mode == "small":
        cfg["actor_hidden_dims"] = (32, 32, 32, 32)
        cfg["value_hidden_dims"] = (32, 32, 32, 32)
        rng = np.random.default_rng(args.seed)
        B, obs_dim, action_dim = int(cfg["batch_size"]), 6, 3
        batch = {"observations": rng.normal(size=(B, obs_dim)).astype(np.float32),
                 "actions": rng.uniform(-1, 1, size=(B, action_dim)).astype(np.float32),
                 "rewards": rng.normal(size=B).astype(np.float32),
                 "masks": (rng.random(B) < 0.9).astype(np.float32),
                 "next_observations": rng.normal(size=(B, obs_dim)).astype(np.float32)}
        agent = FQLAgent.create(args.seed, batch["observations"][:1], batch["actions"][:1], dict(cfg))
        out.update({f"init/{k}": v for k, v in flat_params(agent.network.params).items()})
        out.update({f"batch/{k}": v for k, v in batch.items()})
        infos = []
        jb = {k: jnp.asarray(v) for k, v in batch.items()}
        for u in range(args.updates):
            draws = update_draws(agent.rng, B, action_dim)
            out.update({f"draws/{u}/{k}": v for k, v in draws.items()})
            agent, info = agent.update(jb)
            infos.append({k: float(v) for k, v in info.items()})
            if u in (0, args.updates - 1):  # first and last update (fixture size)
                out.update({f"params/{u}/{k}": v for k, v in flat_params(agent.network.params).items()})
        key = jax.random.PRNGKey(123)
        action_seed, _ = jax.random.split(key)
        out["check/sample_noise"] = np.asarray(jax.random.normal(action_seed, (B, action_dim)), np.float32)
        out["check/sample_actions"] = np.asarray(agent.sample_actions(jb["observations"], seed=key), np.float32)
        out["check/flow_actions"] = np.asarray(agent.compute_flow_actions(jb["observations"],
                                                                          jnp.asarray(out["check/sample_noise"])))
        meta = {"mode": "small", "commit": head, "config": {k: (list(v) if isinstance(v, tuple) else v)
                                                           for k, v in dict(cfg).items() if k not in ("ob_dims",)},
                "obs_dim": obs_dim, "action_dim": action_dim, "updates": args.updates, "infos": infos,
                "jax": jax.__version__}
    else:
        data = toy_bandit(20000, args.seed)
        B, obs_dim, action_dim = int(cfg["batch_size"]), 4, 2
        agent = FQLAgent.create(args.seed, data["observations"][:1], data["actions"][:1], dict(cfg))
        out.update({f"init/{k}": v for k, v in flat_params(agent.network.params).items()})
        idx_rng = np.random.default_rng(args.seed + 1)
        indices = np.stack([idx_rng.integers(len(data["actions"]), size=B) for _ in range(args.steps)])
        draws = {k: [] for k in ("next_actions", "x0", "t", "distill", "mse")}
        infos = []
        for s in range(args.steps):
            d = update_draws(agent.rng, B, action_dim)
            for k in draws:
                draws[k].append(d[k])
            batch = {k: jnp.asarray(v[indices[s]]) for k, v in data.items() if k != "terminals"}
            agent, info = agent.update(batch)
            if s % 50 == 0 or s == args.steps - 1:
                infos.append({"step": s, **{k: float(v) for k, v in info.items()}})
        out.update({f"draws/{k}": np.stack(v) for k, v in draws.items()})
        out["indices"] = indices.astype(np.int32)
        eval_obs = np.random.default_rng(args.seed + 2).normal(size=(4096, obs_dim)).astype(np.float32)
        key = jax.random.PRNGKey(args.seed + 3)
        action_seed, _ = jax.random.split(key)
        out["eval/observations"] = eval_obs
        out["eval/noise"] = np.asarray(jax.random.normal(action_seed, (4096, action_dim)), np.float32)
        acts = np.asarray(agent.sample_actions(jnp.asarray(eval_obs), seed=key), np.float32)
        flow = np.asarray(agent.compute_flow_actions(jnp.asarray(eval_obs), jnp.asarray(out["eval/noise"])))
        out["eval/onestep_actions"], out["eval/flow_actions"] = acts, flow
        out.update({f"final/{k}": v for k, v in flat_params(agent.network.params).items()})
        h = __import__("hashlib").sha256()
        for k in sorted(data):  # same digest as s4d/rwlearn/fql_toy.py::dataset_sha256
            h.update(k.encode())
            h.update(np.ascontiguousarray(data[k]).tobytes())
        meta = {"mode": "toy", "commit": head, "steps": args.steps, "seed": args.seed, "infos": infos,
                "data_sha256": h.hexdigest(), "data_n": len(data["actions"]),
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(cfg).items()
                           if k not in ("ob_dims",)},
                "eval": {"onestep_good_mode_fraction": float((acts.sum(1) > 0).mean()),
                         "onestep_mean_reward": float(-np.sum((acts - 0.5) ** 2, 1).mean()),
                         "flow_good_mode_fraction": float((flow.sum(1) > 0).mean())},
                "jax": jax.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **out)
    print(json.dumps({k: v for k, v in meta.items() if k != "infos"} | {"last_info": meta["infos"][-1]}, indent=1))


if __name__ == "__main__":
    main()
