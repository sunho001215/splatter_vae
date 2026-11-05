"""Torch side of the full-size FQL equivalence check (repository venv, CPU).

Replays the official run written by ``fql_reference_jax.py toy`` with the port (s4d/rwlearn/fql.py): the same official
initial parameters, the same batches (indices) and the same random draws per update, the full official configuration
(4 x 512 MLPs, batch 256, lr 3e-4, alpha 10, flow_steps 10, discount 0.99, tau 0.005, q_agg mean). Reports the logged
losses side by side every 50 updates, the final parameter differences and the final policies' behaviour on the toy
task. XLA and torch sum in different orders, so float32 round-off accumulates over thousands of updates; the report
states how far the two trajectories drift, it does not hide it.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> python fql_toy_compare.py --official <scratch>/toy.npz \
        --report docs/realworld_fql_toy_equivalence.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from s4d.rwlearn import fql
from s4d.rwlearn.fql_toy import dataset_sha256, toy_bandit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    z = np.load(args.official)
    meta = json.loads(bytes(z["meta"]).decode())
    if meta["commit"] != fql.FQL_COMMIT or meta["mode"] != "toy":
        raise SystemExit("not an official toy reference at the pinned commit")
    data = toy_bandit(meta["data_n"], meta["seed"])
    if dataset_sha256(data) != meta["data_sha256"]:
        raise SystemExit("rebuilt toy data differs from the official run's data")
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()
           if k not in ("action_dim", "ob_dims")}
    official_cfg = {k: v for k, v in fql.official_config().items() if k != "agent_name"}
    if {k: v for k, v in cfg.items() if k != "agent_name"} != official_cfg:
        raise SystemExit("the official run did not use the official default configuration")
    agent = fql.FQLAgent.create(meta["seed"], 4, 2, cfg)
    fql.load_official_params(agent, {k[len("init/"):]: z[k] for k in z.files if k.startswith("init/")})
    tensors = {k: torch.from_numpy(v) for k, v in data.items() if k != "terminals"}
    official = {row["step"]: row for row in meta["infos"]}
    rows = []
    for s in range(meta["steps"]):
        idx = torch.from_numpy(z["indices"][s].astype(np.int64))
        noise = fql.Noise(**{k: torch.from_numpy(z[f"draws/{k}"][s]) for k in
                             ("next_actions", "x0", "t", "distill", "mse")})
        info = agent.update({k: v[idx] for k, v in tensors.items()}, noise)
        if s in official:
            ref = official[s]
            rows.append({"step": s, **{k: {"official": ref[k], "port": info[k],
                                           "rel": abs(info[k] - ref[k]) / max(abs(ref[k]), 1e-8)}
                                       for k in ref if k != "step"}})
    final = {k[len("final/"):]: z[k] for k in z.files if k.startswith("final/")}
    mine = fql.export_official_params(agent)
    param_rel = {k: float(np.linalg.norm(mine[k] - v) / max(np.linalg.norm(v), 1e-12)) for k, v in final.items()}
    obs = torch.from_numpy(z["eval/observations"])
    noise = torch.from_numpy(z["eval/noise"])
    acts = agent.sample_actions(obs, noise).numpy()
    flow = agent.compute_flow_actions(obs, noise).numpy()
    port_eval = {"onestep_good_mode_fraction": float((acts.sum(1) > 0).mean()),
                 "onestep_mean_reward": float(-np.sum((acts - 0.5) ** 2, 1).mean()),
                 "flow_good_mode_fraction": float((flow.sum(1) > 0).mean())}

    def max_rel(key, upto):
        vals = [r[key]["rel"] for r in rows if r["step"] <= upto and key in r]
        return max(vals) if vals else None

    keys = [k for k in rows[0] if k != "step"]
    report = {
        "official_commit": meta["commit"], "jax": meta.get("jax"), "torch": torch.__version__,
        "config": meta["config"], "steps": meta["steps"], "data_sha256": meta["data_sha256"],
        "max_relative_difference_of_logged_values": {
            f"first_{n}_updates": {k: max_rel(k, n) for k in keys} for n in (1, 200, 1000, meta["steps"])},
        "final_parameter_relative_l2": {"median": float(np.median(list(param_rel.values()))),
                                        "max": float(max(param_rel.values()))},
        "final_policy": {"official": meta["eval"], "port": port_eval,
                         "onestep_actions_max_abs_diff": float(np.abs(acts - z["eval/onestep_actions"]).max()),
                         "flow_actions_max_abs_diff": float(np.abs(flow - z["eval/flow_actions"]).max())},
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
