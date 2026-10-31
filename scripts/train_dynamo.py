"""DynaMo pretraining on one Meta-World task: official model, objective and optimisation; splatter4d data and outputs.

The encoder, inverse-dynamics projector and ``DynaMoSSL`` objective are the official classes from the pinned
submodule (``s4d/baselines/dynamo/official.py``); the optimisation step, per-epoch learning-rate schedule,
validation and epoch logging replicate ``train.py::Trainer`` (``s4d/baselines/dynamo/training.py``). This entry point
replaces only data loading (single-camera windows from our HDF5 and saved split manifest), output locations,
W&B/metrics logging, checkpoint/resume and the encoder exports. Hyperparameters come from
``configs/baselines/dynamo/<task>.yaml``; see docs/baselines_dynamo.md.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/train_dynamo.py --config configs/baselines/dynamo/hammer.yaml \
        --name dynamo-hammer [--resume auto]

Outputs in ``<output-root>/<name>/``: ``config.yaml``, ``provenance.json``, ``metrics.jsonl`` (step records every
``log_every`` steps, one record with the official epoch means per epoch), ``checkpoints/latest.pt`` (every
``save_every`` steps and every epoch; lossless resume with ``--resume auto``), ``encoder.pt`` (latest encoder),
``encoder_init.pt`` (the same encoder at its random initialisation), ``exports/encoder_epochNN.pt`` (upstream's
``encoder_{epoch}.pt`` every ``save_every_epochs``) and ``completion.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, require_passed_tests  # noqa: E402

GPU_MAPPING = guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from s4d.baselines.data import split_episodes  # noqa: E402
from s4d.baselines.dynamo.data import DynaMoWindows, to_official_obs  # noqa: E402
from s4d.baselines.dynamo.model import build_modules, encoder_kwargs, save_export  # noqa: E402
from s4d.baselines.dynamo.training import (  # noqa: E402
    EpochLog,
    EpochSampler,
    evaluate,
    set_seed_everywhere,
    start_epoch,
    train_step,
)
from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from ops.resources.gpu_guard import GPUIsolationError, validate_runtime_path  # noqa: E402
from ops.jobs.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 1000


def rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default=str(REPO / "runs/pretrain"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--resume", choices=["auto"], default=None, help="continue from <run>/checkpoints/latest.pt")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--init-export-only", action="store_true", help="write encoder_init.pt (random init) and exit")
    args = ap.parse_args()

    cfg = load_config([args.config], args.set)
    ds_cfg, train_cfg, interface = cfg["dataset"], cfg["train"], cfg["data_interface"]
    max_steps = train_cfg.get("max_global_steps")
    if not args.init_export_only and (max_steps is None or int(max_steps) > SHORT_RUN_STEPS):
        require_passed_tests()
    try:
        run_dir = validate_runtime_path(Path(args.output_root) / args.name, repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("pretraining outputs must stay inside an authorized runtime root") from exc
    latest = run_dir / "checkpoints" / "latest.pt"
    if latest.is_file() and args.resume is None and not args.init_export_only:
        raise FileExistsError(f"{run_dir} has a checkpoint; continue it with --resume auto")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"gpus": GPU_MAPPING, "dir": str(run_dir)}
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=max_steps)
    dump_config(cfg, run_dir / "config.yaml")

    seed = int(train_cfg["seed"])
    set_seed_everywhere(seed)  # train.py:36, before the datasets and models (train.py:50-61)
    device = torch.device(train_cfg["device"])
    frame_spacing, window = int(interface["frame_spacing"]), int(ds_cfg["window_size"])
    batch_size = int(ds_cfg["batch_size"])
    manifest = interface["split_manifest"]
    train_set = DynaMoWindows(
        ds_cfg["hdf5_path"], split_episodes(manifest, "train"), window, frame_spacing, ds_cfg.get("max_episodes")
    )
    val_set = DynaMoWindows(ds_cfg["hdf5_path"], split_episodes(manifest, "validation"), window, frame_spacing)
    image_size = int(train_set[0].shape[-1])
    modules = build_modules(cfg, device)
    kwargs = encoder_kwargs(cfg)
    if not (run_dir / "encoder_init.pt").is_file():  # the seed's initial weights, before any update
        save_export(run_dir / "encoder_init.pt", modules.encoder, kwargs, frame_spacing, image_size, 0, True)
    if args.init_export_only:
        return

    epoch, step_in_epoch, global_step = 0, 0, 0
    log, window_sums = EpochLog(), {}
    if latest.is_file():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)  # own checkpoint (RNG payload)
        modules.load_state_dict(ckpt["modules"])
        epoch, step_in_epoch, global_step = int(ckpt["epoch"]), int(ckpt["step_in_epoch"]), int(ckpt["global_step"])
        log, window_sums = EpochLog(ckpt["epoch_log"]), dict(ckpt["window_sums"])
        set_rng_state(ckpt["rng"])

    workers = int(ds_cfg["num_workers"])
    sampler = EpochSampler(len(train_set), seed)
    train_loader = DataLoader(  # train.py:152-163 (shuffle via the resumable sampler), drop_last False as upstream
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(ds_cfg["pin_memory"]),
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=bool(ds_cfg["pin_memory"])
    )
    steps_per_epoch = math.ceil(len(train_set) / batch_size)
    num_epochs = int(train_cfg["num_epochs"])
    log_every, save_every = int(train_cfg["log_every"]), int(train_cfg["save_every"])
    clip = train_cfg["clip_grad_norm"]
    to_obs = lambda batch: to_official_obs(batch, device)  # noqa: E731

    wandb_run = init_wandb(cfg, args.name, cfg["wandb"]["project"], not args.no_wandb, run_dir)
    metrics_file = open(run_dir / "metrics.jsonl", "a")
    print(
        f"DynaMo: {len(train_set)} training windows ({steps_per_epoch} steps/epoch, {num_epochs * steps_per_epoch} "
        f"steps in {num_epochs} epochs), {len(val_set)} validation windows; resuming at epoch {epoch} step "
        f"{step_in_epoch} (global {global_step})",
        flush=True,
    )

    def write(record: dict) -> None:
        metrics_file.write(json.dumps({"step": global_step, **record}) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(record, step=global_step)

    def save() -> None:
        tmp = latest.with_suffix(".tmp")
        torch.save(
            {
                "modules": modules.state_dict(),
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "epoch_log": log.state(),
                "window_sums": window_sums,
                "rng": rng_state(),
            },
            tmp,
        )
        tmp.replace(latest)
        save_export(run_dir / "encoder.pt", modules.encoder, kwargs, frame_spacing, image_size, global_step, False)

    def reached_max() -> bool:
        return max_steps is not None and global_step >= int(max_steps)

    tick, data_time = time.time(), 0.0
    while epoch < num_epochs and not reached_max():
        lr = start_epoch(modules, train_cfg, epoch)
        if step_in_epoch == 0 and lr is not None:
            log.append("metrics", 1, {"lr": lr})  # train.py:172
        sampler.set_epoch(epoch, step_in_epoch * batch_size)
        modules.encoder.train()
        modules.projector.train()
        fetch = time.time()
        for batch in train_loader:
            data_time += time.time() - fetch
            obs = to_obs(batch)
            components = train_step(modules, obs, clip)
            log.append("ssl_train", len(obs), components)  # train.py:190
            global_step += 1
            step_in_epoch += 1
            for key, value in components.items():
                count, total = window_sums.get(key, (0, 0.0))
                window_sums[key] = (count + 1, total + value)
            if global_step % log_every == 0:
                elapsed, tick = time.time() - tick, time.time()
                write(
                    {
                        **{f"train/{k}": total / count for k, (count, total) in window_sums.items()},
                        "train/lr_encoder": modules.encoder_optim.param_groups[0]["lr"],
                        "train/lr_forward_dynamics": modules.forward_dynamics_optim.param_groups[0]["lr"],
                        "train/epoch": epoch + step_in_epoch / steps_per_epoch,
                        "train/step_time": elapsed / log_every,
                        "train/data_wait": data_time / log_every,
                        "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                    }
                )
                window_sums, data_time = {}, 0.0
            if save_every > 0 and global_step % save_every == 0:
                save()
            if reached_max():
                break
            fetch = time.time()
        else:  # the epoch is complete: train.py:292-296
            evaluate(modules, val_loader, log, to_obs)
            write({"epoch": epoch, **log.flush()})
            if (epoch + 1) % int(train_cfg["save_every_epochs"]) == 0:
                (run_dir / "exports").mkdir(exist_ok=True)
                save_export(
                    run_dir / "exports" / f"encoder_epoch{epoch:02d}.pt",
                    modules.encoder,
                    kwargs,
                    frame_spacing,
                    image_size,
                    global_step,
                    False,
                )
            epoch, step_in_epoch = epoch + 1, 0
            save()
            tick = time.time()
    save()
    summary = {"status": "completed", "steps": global_step, "epochs_completed": epoch, "steps_per_epoch": steps_per_epoch}
    (run_dir / "completion.json").write_text(json.dumps(summary))
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
