"""ReViWo pretraining on one Meta-World task: reference trainer logic, splatter4d data interface and outputs.

The loss is the reference ``baselines/ReViWo/train.py::compute_reviwo_loss`` and the model the upstream
``MultiViewBetaVAE`` (``s4d/baselines/reviwo/``); this entry point replaces only data loading (our HDF5 layout and
saved split manifest), output locations, logging, checkpoint/resume and the encoder export. Every hyperparameter comes
from ``configs/baselines/reviwo/<task>.yaml`` (the reference configs). See docs/BASELINES.md.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/train_reviwo.py --config configs/baselines/reviwo/hammer.yaml \
        --name reviwo-hammer
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, require_passed_tests  # noqa: E402

GPU_MAPPING = guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from s4d.baselines.data import ReViWoStates, split_episodes  # noqa: E402
from s4d.baselines.reviwo.model import build_model, disable_kmeans_init, save_export  # noqa: E402
from s4d.baselines.reviwo.training import ReViWoTrainConfig, compute_reviwo_loss  # noqa: E402
from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402

SHORT_RUN_STEPS = 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default=str(REPO / "runs/pretrain"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config([args.config], args.set)
    ds_cfg, reviwo_cfg, interface = dict(cfg["dataset"]), dict(cfg["reviwo"]), dict(cfg["data_interface"])
    cfg_train = ReViWoTrainConfig(**cfg["train"])
    if cfg_train.max_global_steps is not None and cfg_train.max_global_steps > SHORT_RUN_STEPS:
        require_passed_tests()
    run_dir = (Path(args.output_root) / args.name).resolve()
    if REPO.resolve() not in run_dir.parents:
        raise ValueError("pretraining outputs must stay inside the repository")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"gpus": GPU_MAPPING, "dir": str(run_dir)}
    dump_config(cfg, run_dir / "config.yaml")

    seed = int(ds_cfg.get("seed", 42))  # reference set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg_train.device)

    num_views = int(ds_cfg.get("camera_num", 6))
    max_frames = ds_cfg.get("max_frames_per_demo")
    manifest = interface["split_manifest"]
    train_set = ReViWoStates(
        ds_cfg["hdf5_path"], split_episodes(manifest, "train"), num_views, ds_cfg.get("num_episodes"), max_frames
    )
    valid_set = ReViWoStates(ds_cfg["hdf5_path"], split_episodes(manifest, "validation"), num_views, None, max_frames)
    workers = int(ds_cfg.get("num_workers", 8))
    loader_args = {
        "batch_size": int(ds_cfg.get("batch_size", 128)),
        "pin_memory": bool(ds_cfg.get("pin_memory", True)),
        "drop_last": True,
        "shuffle": True,
    }
    train_loader = DataLoader(train_set, num_workers=workers, persistent_workers=workers > 0, **loader_args)
    valid_loader = DataLoader(valid_set, num_workers=max(1, workers // 2), **loader_args)
    img_size = int(train_set[0]["images"].shape[-1])
    cfg_train.camera_num = num_views  # reference: overwritten from the dataset

    model = build_model(reviwo_cfg, img_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    latest = run_dir / "checkpoints" / "latest.pt"
    global_step, epoch = 0, 0
    if latest.is_file():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)  # own checkpoint (RNG payload)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        disable_kmeans_init(model)
        global_step, epoch = int(ckpt["global_step"]), int(ckpt["epoch"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
        random.setstate(ckpt["python_rng"])

    wandb_run = init_wandb(cfg, args.name, cfg["wandb"]["project"], not args.no_wandb, run_dir)
    metrics_file = open(run_dir / "metrics.jsonl", "a")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps({"step": global_step, **record}) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(record, step=global_step)

    def save() -> None:
        tmp = latest.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "python_rng": random.getstate(),
            },
            tmp,
        )
        tmp.replace(latest)
        save_export(run_dir / "encoder.pt", model, reviwo_cfg, img_size, global_step)

    max_steps = cfg_train.max_global_steps
    tick = time.time()
    while max_steps is None or global_step < max_steps:
        for batch in train_loader:
            if max_steps is not None and global_step >= max_steps:
                break
            model.train()
            loss_dict = compute_reviwo_loss(model, batch, cfg_train, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            optimizer.step()
            if global_step % cfg_train.log_every == 0:
                elapsed = time.time() - tick
                tick = time.time()
                log(
                    {
                        **{f"train/{k}": float(v) for k, v in loss_dict.items()},
                        "train/epoch": epoch,
                        "train/step_time": elapsed / (cfg_train.log_every if global_step else 1),
                        "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                )
            if cfg_train.eval_every > 0 and global_step > 0 and global_step % cfg_train.eval_every == 0:
                # Reference log_validation_images_reviwo: validation losses and the reconstruction/shuffle grid.
                model.eval()
                with torch.no_grad():
                    val_batch = next(iter(valid_loader))
                    val_losses = compute_reviwo_loss(model, val_batch, cfg_train, device)
                    grid, _, _ = model.visualize(val_batch["images"][:4].to(device))
                out = run_dir / "eval" / f"step_{global_step:08d}.png"
                out.parent.mkdir(exist_ok=True)
                Image.fromarray(grid).save(out)
                log({f"val/{k}": float(v) for k, v in val_losses.items()})
            global_step += 1
            if cfg_train.save_every > 0 and global_step % cfg_train.save_every == 0:
                save()
        else:
            epoch += 1
            if max_steps is None and epoch >= cfg_train.num_epochs:
                break
    save()
    (run_dir / "completion.json").write_text(json.dumps({"status": "completed", "steps": global_step}))
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
