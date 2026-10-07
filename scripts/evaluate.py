"""Evaluate a checkpoint on the full validation split (metrics, probes, panels) -> outputs/<run>/eval/step_X/summary.json.

CUDA_VISIBLE_DEVICES=$GPU5 python scripts/evaluate.py --run outputs/metaworld-hammer-full-0 [--checkpoint path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import REPO, guard_gpus

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.diag.local_log import RunLogger  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.model.render import require_prebuilt_renderer  # noqa: E402
from s4d.train.checkpoint import load_checkpoint  # noqa: E402
from s4d.train.evaluate import Evaluator  # noqa: E402
from s4d.train.loop import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing config.yaml and checkpoints/")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--wandb", action="store_true", help="also log to W&B (resumes nothing; separate eval run)")
    args = ap.parse_args()
    run_dir = Path(args.run).resolve(strict=True)
    if REPO.resolve() not in run_dir.parents:
        raise ValueError("evaluation outputs must remain inside the new repository")
    require_prebuilt_renderer()
    cfg = load_config([run_dir / "config.yaml"], args.set)
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "latest.pt"
    device = torch.device("cuda", 0)
    model = build_model(cfg).to(device)
    step = load_checkpoint(ckpt, model.encoder, model.decoder, restore_rng=False)
    _, val_loaders, probe_loaders, n_train = build_data(cfg)
    wandb_run = init_wandb(
        cfg,
        f"{cfg['run']['name']}-eval",
        cfg.get("wandb", {}).get("project", "splatter4d-metaworld"),
        enabled=args.wandb,
        run_dir=run_dir,
    )
    logger = RunLogger(run_dir, wandb_run)
    windows = {stride: len(loader.dataset) for stride, loader in val_loaders.items()}
    logger.text(f"evaluating {ckpt} (step {step}) on validation windows per stride {windows}; GPUs {GPU_MAPPING}")
    try:
        summary = Evaluator(cfg, val_loaders, probe_loaders, logger, device, n_train)(model, step, full=True)
        print(json.dumps(summary, indent=1))
    finally:
        logger.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
