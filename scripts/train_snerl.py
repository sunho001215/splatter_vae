"""SNeRL pretraining on one Meta-World task: the official ``nerf_pretrain`` networks, renderer, loss and schedule
(``third_party/snerl``) on our HDF5 data, with our outputs, logging, checkpoint/resume and encoder export.

The step is the official loop body (``s4d/baselines/snerl/model.py::training_step``); hyperparameters come from
``configs/baselines/snerl/<task>.yaml`` (official ``configs/hammer.txt`` + parser defaults). See docs/baselines_snerl.md.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/train_snerl.py --config configs/baselines/snerl/hammer.yaml \
        --name snerl-hammer

Outputs under ``<output-root>/<name>``: ``checkpoints/latest.pt`` (networks, optimiser, step, RNG states; resumed
losslessly with ``--resume auto``), ``encoder.pt`` (export for RL), ``encoder_init.pt`` (the same encoder at its random
initialisation, for the pretrained-vs-random check), ``metrics.jsonl``, ``eval/step_*.png``.
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

from s4d.baselines.data import split_episodes  # noqa: E402
from s4d.baselines.snerl import viz  # noqa: E402
from s4d.baselines.snerl.data import SNeRLScenes  # noqa: E402
from s4d.baselines.snerl.model import (  # noqa: E402
    create,
    dino_target,
    live_density_fractions,
    render_camera,
    save_export,
    to_device_scene,
    training_step,
    update_learning_rate,
)
from s4d.baselines.snerl.upstream import dino_extractor, official  # noqa: E402
from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.gpu_guard import GPUIsolationError, validate_runtime_path  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 10_000  # smoke runs (< 0.5% of the official 2.4M iterations) may run without suite evidence
RAY_STREAM = 1  # per-step generator of the ray choice: default_rng([seed, step, RAY_STREAM])


@torch.no_grad()
def validate(model, extractor, val_set: SNeRLScenes, n: int, chunk: int, out_png: Path | None) -> dict:
    """Full-image renders of ``n`` fixed validation scenes at a camera that is not an encoder input."""
    model.train(False)
    metrics: dict[str, list[float]] = {"psnr": [], "semantic_acc": [], "dino_mse": [], "depth_mae_m": []}
    rows = []
    for k in range(n):
        ep, t, inputs, target = val_set.draw(k)
        if target in inputs:
            target = min(set(range(val_set.num_cams)) - set(inputs))
        scene = to_device_scene(val_set.scene(ep, t, inputs, target, with_depth=True), model.device)
        latent = model.encoder(scene["obs_rgb"].permute(0, 3, 1, 2), scene["obs_c2w"].reshape(-1, 16))
        H, W = scene["target_rgb"].shape[:2]
        out = render_camera(model, latent, val_set.K_cv[target], val_set.c2w_cv[target], H, W, chunk,
                            val_set.lookat, val_set.scale)
        mse = torch.mean((out["rgb"] - scene["target_rgb"]) ** 2)
        metrics["psnr"].append(float(-10.0 * torch.log10(mse)))
        metrics["semantic_acc"].append(float((out["semantic"].argmax(-1) == scene["target_semantic"]).float().mean()))
        teacher = dino_target(extractor, scene["target_rgb"])
        metrics["dino_mse"].append(float(torch.mean((out["dino"] - teacher) ** 2)))
        gt_depth = scene["target_depth"]
        valid = gt_depth > 0
        metrics["depth_mae_m"].append(float((out["depth"] - gt_depth).abs()[valid].mean()))
        if out_png is not None and k < 2:
            to_np = lambda x: (x.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # noqa: E731
            gt_np, pred_depth = gt_depth.cpu().numpy(), out["depth"].cpu().numpy()
            gt_pca, pred_pca = viz.dino_pca(teacher.cpu().numpy(), out["dino"].cpu().numpy())
            rows.append(
                [viz.tile(to_np(img), f"input cam {v}") for img, v in zip(scene["obs_rgb"], inputs)]
                + [
                    viz.tile(to_np(scene["target_rgb"]), f"GT cam {target}"),
                    viz.tile(to_np(out["rgb"]), "render"),
                    viz.tile(viz.depth_image(gt_np, 0.3, 1.8, gt_np <= 0), "GT depth"),
                    viz.tile(viz.depth_image(pred_depth, 0.3, 1.8), "depth"),
                    viz.tile(viz.class_image(scene["target_semantic"].cpu().numpy()), "GT classes"),
                    viz.tile(viz.class_image(out["semantic"].argmax(-1).cpu().numpy()), "classes"),
                    viz.tile(gt_pca, "DINO PCA"),
                    viz.tile(pred_pca, "rendered DINO"),
                ]
            )
    if rows:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viz.grid(rows)).save(out_png)
    model.train(True)
    return {f"val/{key}": float(np.mean(values)) for key, values in metrics.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default=str(REPO / "runs/pretrain"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--resume", choices=("auto", "never"), default="auto",
                    help="auto: continue from checkpoints/latest.pt when it exists")
    ap.add_argument("--export-init-only", action="store_true",
                    help="write encoder_init.pt (the encoder at its seeded random initialisation) and exit")
    ap.add_argument("--no-wandb", action="store_true")
    args_cli = ap.parse_args()

    cfg = load_config([args_cli.config], args_cli.set)
    ds_cfg, nerf_cfg, train_cfg, interface = (dict(cfg[k]) for k in ("dataset", "nerf", "train", "data_interface"))
    loss_cfg, dino_cfg = dict(cfg["loss"]), dict(cfg.get("dino") or {})
    max_steps = int(train_cfg["max_steps"])
    if max_steps > SHORT_RUN_STEPS and not args_cli.export_init_only:
        require_passed_tests()
    try:
        run_dir = validate_runtime_path(Path(args_cli.output_root) / args_cli.name, repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("pretraining outputs must stay inside an authorized runtime root") from exc
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"gpus": GPU_MAPPING, "dir": str(run_dir)}
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=max_steps)
    dump_config(cfg, run_dir / "config.yaml")

    official()  # load the upstream code before seeding (its import-time np.random.seed(0) is undone anyway)
    seed = int(train_cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(train_cfg["device"])
    lookat, scale = tuple(interface["lookat"]), float(interface["pose_scale"])
    num_inputs = int(ds_cfg["num_inputs"])

    manifest = interface["split_manifest"]
    train_set = SNeRLScenes(ds_cfg["hdf5_path"], split_episodes(manifest, "train"), seed, num_inputs, max_steps,
                            lookat, scale)
    val_set = SNeRLScenes(ds_cfg["hdf5_path"], split_episodes(manifest, "validation"), seed + 1, num_inputs, None,
                          lookat, scale)
    latest = run_dir / "checkpoints" / "latest.pt"
    global_step = 0
    resume = args_cli.resume == "auto" and latest.is_file()
    if resume and args_cli.export_init_only:
        raise SystemExit("--export-init-only needs a fresh run directory (or --resume never)")
    if not resume:
        model = create(nerf_cfg, device)  # one official initialisation at the configured seed; never redraw silently
        probe = to_device_scene(train_set[0], device)
        train_set.close()
        coarse, fine = live_density_fractions(model, probe)
        attempts = [{"torch_seed": seed, "live_coarse": coarse, "live_fine": fine}]
        (run_dir / "init.json").write_text(json.dumps({"attempts": attempts}, indent=1))
        if min(coarse, fine) == 0:
            print("warning: NeRF density is zero on the probe; initialization is kept unchanged", file=sys.stderr)
        save_export(run_dir / "encoder_init.pt", model.encoder, 0, num_inputs, lookat, scale, random_init=True)
        if args_cli.export_init_only:
            return
    else:
        model = create(nerf_cfg, device)
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)  # own checkpoint (RNG payload)
        model.load_state_dict(ckpt["model"])
        global_step = int(ckpt["global_step"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(ckpt["cuda_rng"])

    extractor = dino_extractor(device, dino_cfg.get("weights"), bool(dino_cfg.get("pretrained", True)))
    workers = int(ds_cfg["num_workers"])
    loader = DataLoader(
        train_set,
        batch_size=None,  # one scene per step (run_nerf.py:687)
        sampler=range(global_step, max_steps),
        num_workers=workers,
        prefetch_factor=int(ds_cfg["prefetch_factor"]) if workers > 0 else None,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),  # worker seeds never draw from the (checkpointed) global RNG
    )

    wandb_run = init_wandb(cfg, args_cli.name, cfg["wandb"]["project"], not args_cli.no_wandb, run_dir)
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
                "global_step": global_step,
                "model": model.state_dict(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
            },
            tmp,
        )
        tmp.replace(latest)
        save_export(run_dir / "encoder.pt", model.encoder, global_step, num_inputs, lookat, scale)

    print_every, save_every, eval_every = (int(train_cfg[k]) for k in ("i_print", "save_every", "eval_every"))
    weights = {"semantic_weight": float(loss_cfg["semantic_weight"]), "dino_weight": float(loss_cfg["dino_weight"])}
    tick, window = time.time(), 0
    model.train(True)
    for scene in loader:
        rng = np.random.default_rng([seed, global_step, RAY_STREAM])
        loss, terms = training_step(model, extractor, to_device_scene(scene, device), global_step, rng, **weights)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        lr = update_learning_rate(model.optimizer, float(nerf_cfg["lrate"]), int(nerf_cfg["lrate_decay"]), global_step)
        window += 1
        if global_step % print_every == 0:
            elapsed, tick = time.time() - tick, time.time()
            record = {f"train/{k}": float(v.detach()) for k, v in terms.items()}
            record.update({"train/lr": lr, "train/step_time": elapsed / window})
            if device.type == "cuda":
                record["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            log(record)
            window = 0
        if eval_every > 0 and global_step > 0 and global_step % eval_every == 0:
            png = run_dir / "eval" / f"step_{global_step:08d}.png"
            log(validate(model, extractor, val_set, int(train_cfg["eval_scenes"]), int(train_cfg["render_chunk"]), png))
        global_step += 1
        if save_every > 0 and global_step % save_every == 0:
            save()
    save()
    png = run_dir / "eval" / f"step_{global_step:08d}.png"
    log(validate(model, extractor, val_set, int(train_cfg["eval_scenes"]), int(train_cfg["render_chunk"]), png))
    (run_dir / "completion.json").write_text(json.dumps({"status": "completed", "steps": global_step}))
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
