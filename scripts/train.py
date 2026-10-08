"""Pretraining entry point (in-domain Meta-World or DROID cache).

    CUDA_VISIBLE_DEVICES=$GPU4 python scripts/train.py --config configs/metaworld/base.yaml \
        configs/metaworld/tasks/hammer.yaml --set train.steps=200000 --name metaworld-hammer-full-0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  (sys.path)
from _bootstrap import REPO, guard_gpus, require_passed_tests

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402

from s4d.config import dump_config, get, load_config  # noqa: E402
from s4d.diag.local_log import RunLogger  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.model.render import require_prebuilt_renderer  # noqa: E402
from s4d.train import ddp  # noqa: E402
from s4d.train.evaluate import Evaluator  # noqa: E402
from s4d.train.loop import train  # noqa: E402


def fixed_windows(dataset, indices):
    """Select one explicit, repeatable training batch without fabricating validation."""
    from torch.utils.data import Subset

    if not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices):
        raise ValueError("data.fixed_window_indices must be a nonempty list of integer training indices")
    if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("data.fixed_window_indices must be unique and within the training split")
    return Subset(dataset, indices)


def build_data(cfg: dict):
    """Return (train_dataset, val_loaders, probe_loaders, n_train_cams).

    ``val_loaders`` and ``probe_loaders`` map a frame stride (simulator steps) to a loader; the
    evaluator reports every metric separately per stride.
    """
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from s4d.data.contract import collate  # noqa: PLC0415

    regime = get(cfg, "data.regime", "metaworld")
    workers = int(get(cfg, "train.num_workers", 8))
    bs = int(get(cfg, "train.batch_size", 16))

    def loader(dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers // 2,
            collate_fn=collate,
            pin_memory=True,
            drop_last=shuffle,
        )

    if regime == "metaworld":
        from s4d.data.metaworld.dataset import MetaworldWindowDataset, split_episodes  # noqa: PLC0415

        path = Path(get(cfg, "data.root")) / f"{get(cfg, 'data.task')}.hdf5"
        episodes = get(cfg, "data.episodes")
        if episodes:  # overfit gate: explicit episode list used for both splits
            train_eps = val_eps = list(episodes)
        else:
            train_eps, val_eps = split_episodes(
                path, float(get(cfg, "data.train_ratio", 0.96)), int(get(cfg, "train.seed", 0))
            )
        strides = tuple(int(s) for s in get(cfg, "data.strides"))
        val_strides = tuple(int(s) for s in get(cfg, "data.val_strides"))
        train_ds = MetaworldWindowDataset(path, train_eps, strides=strides, seed=int(get(cfg, "train.seed", 0)))
        heldout = None
        if get(cfg, "eval.heldout_sets", False):  # near/trajectory held-out sets (validation episodes only)
            from s4d.data.metaworld.heldout import heldout_path  # noqa: PLC0415

            heldout = heldout_path(get(cfg, "data.task"), get(cfg, "data.root"))
            if not heldout.is_file():
                raise FileNotFoundError(f"eval.heldout_sets requested but {heldout} is missing (scripts/render_heldout_sets.py)")
        val_loaders = {
            s: loader(MetaworldWindowDataset(path, val_eps, strides=(s,), with_eval=True, heldout=heldout), False)
            for s in val_strides
        }
        probe_loaders = {s: loader(MetaworldWindowDataset(path, train_eps, strides=(s,)), True) for s in val_strides}
        n_train_cams = len(train_ds.train_cams)
    elif regime == "droid":
        from s4d.data.droid.dataset import DroidCacheDataset  # noqa: PLC0415

        root = Path(get(cfg, "data.root"))
        size = {
            "image_height": get(cfg, "model.encoder.image_height", 144),
            "image_width": get(cfg, "model.encoder.image_width", 256),
        }
        train_ds = DroidCacheDataset(root, split="train", **size)
        val_ds = DroidCacheDataset(root, split="validation", with_eval=True, **size)
        n_train_cams = 2
        indices = get(cfg, "data.fixed_window_indices")
        if indices is not None:
            train_ds = val_ds = fixed_windows(train_ds, indices)
        val_loaders, probe_loaders = {3: loader(val_ds, False)}, {}  # canonical DROID windows use stride 3
    else:
        raise ValueError(f"unsupported data regime {regime!r}")
    return train_ds, val_loaders, probe_loaders, n_train_cams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key.path=value overrides")
    ap.add_argument("--name", required=True, help="run name <regime>-<task>-<variant>-<seed>")
    ap.add_argument("--output-root", default=str(REPO / "outputs"))
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for outputs/<name>/checkpoints/latest.pt")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["run"] = {"name": args.name, "gpus": GPU_MAPPING}
    require_passed_tests()
    ctx = ddp.init_distributed()
    run_dir = Path(args.output_root) / args.name
    if REPO.resolve() not in run_dir.resolve().parents:
        raise ValueError("run outputs must remain inside the new repository")
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(f"run {run_dir} exists; use --resume auto explicitly")
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    if resume == "auto":
        latest = run_dir / "checkpoints" / "latest.pt"
        resume = str(latest) if latest.exists() else None
    wandb_run = None
    if ctx.is_main:
        dump_config(cfg, run_dir / "config.yaml")
        wandb_run = init_wandb(
            cfg,
            args.name,
            get(cfg, "wandb.project", "splatter4d-metaworld"),
            enabled=not args.no_wandb and bool(get(cfg, "wandb.enabled", True)),
            run_dir=run_dir,
        )
    logger = RunLogger(run_dir, wandb_run)
    if ctx.is_main:
        logger.text(f"GPU mapping: {json.dumps(GPU_MAPPING)}; world size {ctx.world_size}")
        logger.text(f"torch {torch.__version__}; config {args.config} overrides {args.set}")
    try:
        require_prebuilt_renderer()
        train_ds, val_loaders, probe_loaders, n_train_cams = build_data(cfg)
        evaluator = Evaluator(cfg, val_loaders, probe_loaders, logger, ctx.device, n_train_cams) if ctx.is_main else None
        train(cfg, run_dir, train_ds, ctx, logger, evaluator, resume=resume)
        if ctx.is_main:
            (run_dir / "completion.json").write_text(json.dumps({"status": "completed", "steps": get(cfg, "train.steps")}))
    except BaseException as exc:
        if ctx.is_main:
            (run_dir / "failure.json").write_text(json.dumps({"error": repr(exc)}))
        raise
    finally:
        logger.close()
        if wandb_run is not None:
            wandb_run.finish()
        ddp.cleanup()


if __name__ == "__main__":
    main()
