"""Offline FQL on a frozen real-world baseline representation, or end-to-end with a randomly initialised IMPALA CNN
trained jointly (``condition=e2e_cnn``; s4d/rwlearn/pixel.py) (docs/realworld_fql.md). Offline only: no environment,
no robot. The agent is the PyTorch port of the official FQL (s4d/rwlearn/fql.py); the loop is the official
``main.py`` offline phase (uniform batches, one ``agent.update`` per step, validation losses every ``log_interval``)
with offline validation metrics in place of environment evaluation. alpha = 10 for every condition (user decision
2026-11-06; no sweep). The offline metrics are diagnostics only, never a selection criterion: later tuning follows a
predefined real-robot evaluation protocol.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/train_realworld_fql.py --config configs/realworld/fql/base.yaml \
        --name fql-dynamo-drawer-open-s0 --set condition=dynamo data.dataset=... data.split_manifest=... \
        data.gate_report=...

Outputs under ``<output-root>/<name>``: ``checkpoints/latest.pt`` (agent, optimiser, generators, step; resumed
losslessly), ``policy.pt`` (agent + the metadata ``s4d.rwlearn.inference.OfflinePolicy`` needs), ``metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, require_passed_tests  # noqa: E402

GPU_MAPPING = guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ops.jobs.run_identity import record_run_identity  # noqa: E402
from ops.resources.gpu_guard import GPUIsolationError, validate_runtime_path  # noqa: E402
from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.rwlearn import actions as A  # noqa: E402
from s4d.rwlearn.fql import FQLAgent, Noise  # noqa: E402
from s4d.rwlearn.fql_data import CONDITIONS, OfflineTransitions, sha256_file  # noqa: E402
from s4d.rwlearn.gate import check_dataset_gate  # noqa: E402
from s4d.rwlearn.inference import checkpoint_payload  # noqa: E402
from s4d.rwlearn.pixel import OfflinePixelTransitions  # noqa: E402

SHORT_RUN_STEPS = 10_000


@torch.no_grad()
def offline_validation(agent: FQLAgent, val: OfflineTransitions, batches: int, batch_size: int,
                       gen: torch.Generator) -> dict:
    """Official validation losses (``total_loss`` without update) + action errors of the one-step policy and of the
    BC flow against the recorded commands, in physical units."""
    infos, pos, rot, grip, bc_pos = [], [], [], [], []
    c = val.contract
    for _ in range(batches):  # every draw from ``gen``: validation never advances the training generators
        batch = val.sample(batch_size, gen)
        infos.append(agent.evaluate_losses(batch, Noise.draw(batch_size, A.ACTION_DIM, gen, agent.device)))
        noise = torch.randn(batch_size, A.ACTION_DIM, generator=gen).to(agent.device)
        pred = c.denormalise(agent.sample_actions(batch["observations"], noise).cpu().numpy())
        flow = c.denormalise(agent.compute_flow_actions(batch["observations"], noise).cpu().numpy())
        rec = c.denormalise(batch["actions"].cpu().numpy())
        pos.append(np.linalg.norm(pred[:, :3] - rec[:, :3], axis=1))
        bc_pos.append(np.linalg.norm(flow[:, :3] - rec[:, :3], axis=1))
        rot.append(np.linalg.norm(pred[:, 3:6] - rec[:, 3:6], axis=1))
        grip.append(np.abs(pred[:, 6] - rec[:, 6]))
    out = {f"validation/{k}": float(np.mean([i[k] for i in infos])) for k in infos[0]}
    out.update({"validation/onestep_pos_err_mm": 1e3 * float(np.median(np.concatenate(pos))),
                "validation/bcflow_pos_err_mm": 1e3 * float(np.median(np.concatenate(bc_pos))),
                "validation/onestep_rotvec_err_deg": float(np.degrees(np.median(np.concatenate(rot)))),
                "validation/onestep_gripper_abs_err": float(np.mean(np.concatenate(grip)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default=str(REPO / "runs/realworld_fql"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config([args.config], args.set)
    condition = cfg["condition"]
    if condition not in CONDITIONS:
        raise SystemExit(f"condition must be one of {sorted(CONDITIONS)}")
    train_cfg, data_cfg = dict(cfg["train"]), dict(cfg["data"])
    steps = int(train_cfg["offline_steps"])
    if steps > SHORT_RUN_STEPS:
        require_passed_tests()
    split = json.loads(Path(data_cfg["split_manifest"]).read_text())
    gate = check_dataset_gate(data_cfg["dataset"], data_cfg.get("gate_report"), steps=steps, split=split,
                              smoke_without_calibration=bool(data_cfg.get("smoke_without_calibration")))
    try:
        run_dir = validate_runtime_path(Path(args.output_root) / args.name, repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("outputs must stay inside an authorized runtime root") from exc
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"gpus": GPU_MAPPING, "dir": str(run_dir), "gate": gate}
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=steps)
    dump_config(cfg, run_dir / "config.yaml")

    device = torch.device(train_cfg["device"])
    clip = data_cfg.get("action_clip_eps")
    pixel_cfg = dict(cfg.get("pixel") or {})
    end_to_end = CONDITIONS[condition].encoder_type is None
    if end_to_end:  # frames of the store, loaded once per camera and frame; CNN trained jointly with FQL
        if pixel_cfg.get("encoder") != "impala_small":
            raise SystemExit("e2e_cnn uses the official impala_small encoder (pixel.encoder)")
        if not data_cfg.get("store"):
            raise SystemExit("e2e_cnn needs data.store (the training store the dataset was built from)")
        kw = {"device": device, "frames_device": data_cfg.get("frames_device") or device,
              "p_aug": pixel_cfg.get("p_aug"), "action_clip_eps": clip}
        train = OfflinePixelTransitions(data_cfg["dataset"], data_cfg["store"], split["train"], **kw)
        val = (OfflinePixelTransitions(data_cfg["dataset"], data_cfg["store"], split["validation"], **kw)
               if split["validation"] else None)  # official: p_aug is set on the validation dataset too
        enc = train.encoder_factory()
    else:
        train = OfflineTransitions(data_cfg["dataset"], split["train"], device, action_clip_eps=clip)
        val = (OfflineTransitions(data_cfg["dataset"], split["validation"], device, action_clip_eps=clip)
               if split["validation"] else None)
        enc = train.encoder_factory(int(cfg["encoder"]["proj_dim"]))
    if train.attrs["condition"] != condition:
        raise SystemExit(f"dataset condition {train.attrs['condition']} != {condition}")
    seed = int(train_cfg["seed"])
    agent_cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in dict(cfg["agent"]).items()}
    agent = FQLAgent.create(seed, enc().out_dim, A.ACTION_DIM, agent_cfg, device=device, encoder=enc)
    if device.type == "cuda" and end_to_end:
        torch.backends.cudnn.benchmark = True
    sample_gen = torch.Generator().manual_seed(seed + 1)
    aug_gen = torch.Generator().manual_seed(seed + 3)
    val_gen_seed = seed + 2
    latest = run_dir / "checkpoints" / "latest.pt"
    step = 0
    identity = {"condition": condition, "export_sha256": train.attrs["export_sha256"],
                "contract": json.loads(train.attrs["contract"]), "reward": train.attrs["reward"],
                "dataset_sha256": sha256_file(data_cfg["dataset"]), "split": split, "smoke": bool(gate["smoke"]),
                "agent": agent_cfg, "action_clip_eps": clip,
                "pixel": {**pixel_cfg, "store": str(data_cfg.get("store"))} if end_to_end else None}
    if latest.is_file():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)  # own checkpoint (RNG payload)
        if json.dumps(ckpt.get("identity"), sort_keys=True, default=list) != json.dumps(identity, sort_keys=True,
                                                                                       default=list):
            raise SystemExit(f"{latest} was trained on different inputs (encoder, contract, dataset, split, smoke "
                             "status or agent config); refusing to resume")
        agent.load_state_dict(ckpt["agent"])
        sample_gen.set_state(ckpt["sample_gen"])
        if "aug_gen" in ckpt:
            aug_gen.set_state(ckpt["aug_gen"])
        step = int(ckpt["step"])

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(checkpoint_payload(agent, train, condition, step, sample_gen=sample_gen.get_state(),
                                      aug_gen=aug_gen.get_state(), identity=identity), tmp)
        tmp.replace(path)

    def sample(batch_size: int):
        return train.sample(batch_size, sample_gen, aug_gen) if end_to_end else train.sample(batch_size, sample_gen)

    wandb_run = init_wandb(cfg, args.name, cfg["wandb"]["project"], not args.no_wandb, run_dir)
    metrics = open(run_dir / "metrics.jsonl", "a")
    batch_size = int(agent_cfg["batch_size"])
    agent.train_mode(True)
    while step < steps:
        info = agent.update(sample(batch_size))
        step += 1
        record = {}
        if step % int(train_cfg["log_interval"]) == 0:
            record.update({f"training/{k}": v for k, v in info.items()})
        if val is not None and (step == 1 or step % int(train_cfg["eval_interval"]) == 0):
            agent.train_mode(False)
            record.update(offline_validation(agent, val, int(train_cfg["val_batches"]), batch_size,
                                             torch.Generator().manual_seed(val_gen_seed)))
            agent.train_mode(True)
        if record:
            metrics.write(json.dumps({"step": step, **record}) + "\n")
            metrics.flush()
            if wandb_run is not None:
                wandb_run.log(record, step=step)
        if step % int(train_cfg["save_interval"]) == 0 or step == steps:
            save(latest)
            save(run_dir / "policy.pt")
    (run_dir / "completion.json").write_text(json.dumps({"status": "completed", "steps": step}))
    metrics.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
