"""Model bundle, loss computation, and the training loop (single GPU or DDP)."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from s4d.config import get
from s4d.data.contract import PAIRS, collate, validate_batch
from s4d.losses.depth import abs_rel, align_teacher, depth_gradient_loss, depth_l1
from s4d.losses.invariance import multi_positive_info_nce, slot_consistency, state_statistics
from s4d.losses.motion import expected_displacement, motion_loss
from s4d.losses.regularizers import visibility_loss
from s4d.losses.rgb import coverage_loss, masked_psnr, masked_ssim, pixel_weights, rgb_loss
from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
from s4d.model.encoder import Encoder, EncoderConfig
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet
from s4d.model.render import render_features, render_hard_depth, render_rgbd
from s4d.train import ddp
from s4d.train.checkpoint import gather_rng_states, load_checkpoint, save_checkpoint

MOVING_SCORE = 0.5
STATIC_SCORE = 0.05
ACTIVE_OPACITY = 0.05


class Model(nn.Module):
    """Encoder + decoder bundle so DDP wraps one module."""

    def __init__(self, encoder: Encoder, decoder: GaussianDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images_bv: torch.Tensor,
        score_bv: torch.Tensor | None,
        num_views: int,
        source: torch.Tensor,
        mask_ratio: float | None = None,
    ):
        enc = self.encoder(images_bv, score_bv, mask_ratio=mask_ratio)
        slots = enc["slots"]
        B = slots.shape[0] // num_views
        slots = slots.view(B, num_views, *slots.shape[1:])
        gs = self.decoder(slots[torch.arange(B, device=slots.device), source])
        return slots.float(), enc["visible"].view(B, num_views, -1), gs


def build_model(cfg: dict) -> Model:
    enc_cfg = dict(get(cfg, "model.encoder", {}))
    if enc_cfg.pop("backbone", "scratch") == "dinov2_vits14":
        encoder_cfg = EncoderConfig.dinov2_vits14(**enc_cfg)
    else:
        encoder_cfg = EncoderConfig(**enc_cfg)
    dec = dict(get(cfg, "model.decoder", {}))
    scene = GroupConfig(**dec.pop("scene"))
    dynamic = GroupConfig(**dec.pop("dynamic"))
    decoder_cfg = DecoderConfig(slot_dim=encoder_cfg.slot_dim, scene=scene, dynamic=dynamic, **dec)
    return Model(Encoder(encoder_cfg), GaussianDecoder(decoder_cfg))


def temporal_ramp(step: int, ramp_steps: int) -> float:
    return 1.0 if ramp_steps <= 0 else min(1.0, max(0.0, step / ramp_steps))


def _rows(x: torch.Tensor) -> torch.Tensor:
    """(B,T,V,C,H,W) -> (B*T*V,C,H,W)."""
    return x.reshape(-1, *x.shape[3:])


def _per_t(x: torch.Tensor, B: int, T: int, V: int) -> torch.Tensor:
    return x.view(B, T, V).mean(dim=(0, 2))


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    return torch.nanmean(x) if torch.isfinite(x).any() else x.new_tensor(float("nan"))


def forward_losses(
    model: Model,
    batch: dict,
    cfg: dict,
    step: int,
    *,
    source: torch.Tensor | None = None,
    mask_ratio: float | None = None,
    return_renders: bool = False,
) -> dict:
    """Compute every loss term and metric for one batch. Returns dict with 'total', 'losses', 'metrics', renders."""
    loss_cfg = cfg["loss"]
    near, far = float(cfg["render"]["near"]), float(cfg["render"]["far"])
    images = batch["images"]
    B, T, V, _, H, W = images.shape
    device = images.device
    images01 = images.float() / 255.0
    score = batch["motion_score"].float()
    enc_in = images01.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 3, H, W)
    score_in = score.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 1, H, W)
    single_frame = bool(get(cfg, "model.single_frame", False))
    if single_frame:
        enc_in, score_in = enc_in[:, :1], score_in[:, :1]
    if source is None:
        source = torch.randint(V, (B,), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(get(cfg, "train.bf16", True))):
        slots, visible, gs = model(enc_in, score_in, V, source, mask_ratio=mask_ratio)
    gs = GaussianSet(
        gs.xyz.float(),
        gs.scales.float(),
        gs.quats.float(),
        gs.opacity.float(),
        gs.rgb.float(),
        gs.delta01.float(),
        gs.delta12.float(),
        gs.group,
    )
    if single_frame:
        gs.delta01 = gs.delta01 * 0.0
        gs.delta12 = gs.delta12 * 0.0
    ramp = 0.0 if single_frame else temporal_ramp(step, int(loss_cfg.get("ramp_steps", 20000)))
    w2c, K = batch["w2c"], batch["K"]
    xyz_seq = gs.xyz_sequence()

    # ---- RGB + depth + coverage -------------------------------------------------------------
    r = render_rgbd(gs, xyz_seq, w2c, K, H, W, near, far)
    depth_t = batch["depth"].float()
    jitter = loss_cfg.get("teacher_scale_jitter")
    if jitter:
        factor = torch.empty(B, T, V, 1, 1, 1, device=device).uniform_(float(jitter[0]), float(jitter[1]))
        depth_t = depth_t * factor
    weights = pixel_weights(_rows(score), float(loss_cfg.get("lambda_dyn", 1.0)))
    valid = _rows(depth_t) > 0
    rgb_rows, tgt_rows, depth_rows, alpha_rows = _rows(r["rgb"]), _rows(images01), _rows(r["depth"]), _rows(r["alpha"])
    rgb_l = rgb_loss(rgb_rows, tgt_rows, weights, float(loss_cfg.get("ssim_weight", 0.2)))
    cov_l = coverage_loss(alpha_rows, valid, weights)
    align_mode = loss_cfg.get("depth_align", "none")
    if step < int(loss_cfg.get("depth_align_warmup_steps", 0)):
        align_mode = "none"
    aligned, align_stats = align_teacher(depth_rows, _rows(depth_t), alpha_rows, align_mode, weights=weights)
    dl1 = depth_l1(depth_rows, aligned, valid, weights)
    dgrad = depth_gradient_loss(depth_rows, aligned, valid, weights=weights)
    hard = render_hard_depth(gs, xyz_seq, w2c, K, H, W, near, far)
    hard_valid = valid
    dhard = depth_l1(_rows(hard["depth"]), aligned, hard_valid, weights)
    render_t = (
        float(loss_cfg.get("rgb", 1.0)) * _per_t(rgb_l, B, T, V)
        + float(loss_cfg.get("coverage", 0.1)) * _per_t(cov_l, B, T, V)
        + float(loss_cfg.get("depth_l1", 1.0)) * _per_t(dl1, B, T, V)
        + float(loss_cfg.get("depth_grad", 0.5)) * _per_t(dgrad, B, T, V)
        + float(loss_cfg.get("depth_hard", 0.5)) * _per_t(dhard, B, T, V)
    )
    render_loss = (render_t[0] + ramp * (render_t[1] + render_t[2])) / (1.0 + 2.0 * ramp)

    # ---- 3D motion -------------------------------------------------------------------------------
    is_dyn = (gs.group == DYNAMIC_GROUP).float()[None, :, None].expand(B, -1, 1)
    d01, d12 = gs.delta01, gs.delta12
    feat0 = torch.cat((d01, d01 + d12, is_dyn), -1)
    feat1 = torch.cat((d12, torch.zeros_like(d12), is_dyn), -1)
    feat2 = torch.cat((torch.zeros_like(d01), torch.zeros_like(d12), is_dyn), -1)
    states = gs.xyz_sequence().detach()
    fr = render_features(gs.detach_geometry(), states, torch.stack((feat0, feat1, feat2), 1), w2c, K, H, W, near, far)
    feats, cov = fr["features"], fr["alpha"].detach()
    pred_feat = torch.stack((feats[:, 0, :, 0:3], feats[:, 1, :, 0:3], feats[:, 0, :, 3:6]), 1)
    coverage = torch.stack((cov[:, 0], cov[:, 1], cov[:, 0]), 1)
    pred_disp = expected_displacement(pred_feat, coverage)
    m_loss, m_metrics = motion_loss(
        pred_disp, coverage, batch["motion3d"], batch["motion_weight"], float(loss_cfg.get("motion_huber", 0.01))
    )
    dyn_share_map = (feats[:, :, :, 6:7] / cov.clamp_min(1e-6)).detach()
    moving0 = score > MOVING_SCORE
    moving_count = moving0.float().sum()
    dyn_share = (dyn_share_map * moving0.float()).sum() / moving_count.clamp_min(1.0)
    dyn_share = torch.where(moving_count > 0, dyn_share, dyn_share.new_tensor(float("nan")))

    # ---- invariance + visibility -------------------------------------------------------------
    slots_mean = slots.mean(2)
    nce, nce_metrics = multi_positive_info_nce(
        slots_mean, float(loss_cfg.get("temperature", 0.1)), distributed=model.training
    )
    cons = slot_consistency(slots)
    vis_t = visibility_loss(xyz_seq, w2c, K, H, W, near, far)
    vis = (vis_t[0] + ramp * (vis_t[1] + vis_t[2])) / (1.0 + 2.0 * ramp)
    total = (
        render_loss
        + ramp * float(loss_cfg.get("motion", 1.0)) * m_loss
        + float(loss_cfg.get("invariance", 1.0)) * nce
        + float(loss_cfg.get("consistency", 0.5)) * cons
        + float(loss_cfg.get("visibility", 1.0)) * vis
    )

    # ---- metrics -------------------------------------------------------------------------------
    with torch.no_grad():
        score_rows = _rows(score)
        all_mask = torch.ones_like(score_rows, dtype=torch.bool)
        psnr_all = masked_psnr(rgb_rows, tgt_rows, all_mask).view(B, T, V)
        psnr_mov = masked_psnr(rgb_rows, tgt_rows, score_rows > MOVING_SCORE).view(B, T, V)
        psnr_sta = masked_psnr(rgb_rows, tgt_rows, score_rows < STATIC_SCORE).view(B, T, V)
        absrel = abs_rel(depth_rows, _rows(depth_t), valid).view(B, T, V)
        opacity = gs.opacity
        metrics = {
            "ramp": torch.tensor(ramp, device=device),
            "psnr": _nanmean(psnr_all),
            "psnr_moving": _nanmean(psnr_mov),
            "psnr_static": _nanmean(psnr_sta),
            "ssim": _nanmean(masked_ssim(rgb_rows, tgt_rows, all_mask)),
            "ssim_moving": _nanmean(masked_ssim(rgb_rows, tgt_rows, score_rows > MOVING_SCORE)),
            "ssim_static": _nanmean(masked_ssim(rgb_rows, tgt_rows, score_rows < STATIC_SCORE)),
            "depth_absrel": absrel.mean(),
            "coverage_mean": alpha_rows.mean(),
            "coverage_valid_mean": (alpha_rows * valid).sum() / valid.sum().clamp_min(1),
            "align_scale_mean": align_stats["scale"].mean(),
            "align_scale_std": align_stats["scale"].std(unbiased=False),
            "opacity_mean": opacity.mean(),
            "opacity_std": opacity.std(unbiased=False),
            "active_fraction": (opacity > ACTIVE_OPACITY).float().mean(),
            "align_shift_mean": align_stats["shift"].mean(),
            "align_scale_min": align_stats["scale"].min(),
            "align_scale_max": align_stats["scale"].max(),
            "dyn_alpha_share_moving": dyn_share,
            "dynamic_score_moving_count": moving_count,
            "translation01_dyn_mean": d01[:, gs.group == DYNAMIC_GROUP].norm(dim=-1).mean(),
            "translation12_dyn_mean": d12[:, gs.group == DYNAMIC_GROUP].norm(dim=-1).mean(),
            "translation01_dyn_max": d01[:, gs.group == DYNAMIC_GROUP].norm(dim=-1).amax(),
        }
        for gid, name in ((0, "scene"), (1, "dynamic")):
            sel = gs.group == gid
            if sel.any():
                metrics[f"active_fraction_{name}"] = (opacity[:, sel] > ACTIVE_OPACITY).float().mean()
                metrics[f"opacity_mean_{name}"] = opacity[:, sel].mean()
        for t in range(T):
            metrics[f"psnr_t{t}"] = _nanmean(psnr_all[:, t])
            metrics[f"psnr_moving_t{t}"] = _nanmean(psnr_mov[:, t])
        metrics.update(m_metrics)
        metrics.update(nce_metrics)
        if not model.training or step == 0 or (step + 1) % int(get(cfg, "train.log_every", 50)) == 0:
            metrics.update(state_statistics(slots_mean))
    losses = {
        "total": total.detach(),
        "render": render_loss.detach(),
        "motion": m_loss.detach(),
        "infonce": nce.detach(),
        "consistency": cons.detach(),
        "visibility": vis.detach(),
        **{f"rgb_t{t}": v.detach() for t, v in enumerate(_per_t(rgb_l, B, T, V))},
        **{f"coverage_t{t}": v.detach() for t, v in enumerate(_per_t(cov_l, B, T, V))},
        **{f"depth_l1_t{t}": v.detach() for t, v in enumerate(_per_t(dl1, B, T, V))},
        **{f"depth_grad_t{t}": v.detach() for t, v in enumerate(_per_t(dgrad, B, T, V))},
        **{f"depth_hard_t{t}": v.detach() for t, v in enumerate(_per_t(dhard, B, T, V))},
        **{f"visibility_t{t}": v.detach() for t, v in enumerate(vis_t)},
    }
    out = {"total": total, "losses": losses, "metrics": metrics, "slots": slots, "source": source, "gs": gs}
    if return_renders:
        out.update(
            {
                "rgb": r["rgb"].detach(),
                "depth": r["depth"].detach(),
                "alpha": r["alpha"].detach(),
                "pred_disp": pred_disp.detach(),
                "coverage": coverage.detach(),
                "visible": visible,
                "dyn_share_map": dyn_share_map,
                "aligned_depth": aligned.view(B, T, V, 1, H, W).detach(),
                "pairs": PAIRS,
            }
        )
    return out


# ------------------------------------------------------------------------------------------ training
def build_optimizer(model: nn.Module, cfg: dict):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("anchors") or "token" in name or "embed" in name else decay).append(p)
    lr = float(get(cfg, "train.lr", 5e-4))
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(get(cfg, "train.weight_decay", 0.01))},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        fused=True,
    )
    total, warmup = int(get(cfg, "train.steps", 200000)), int(get(cfg, "train.warmup_steps", 10000))
    min_ratio = float(get(cfg, "train.min_lr", 1e-5)) / lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def grad_norms(model: Model) -> dict[str, float]:
    out = {}
    for name, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        sq = sum(float(p.grad.detach().float().norm() ** 2) for p in module.parameters() if p.grad is not None)
        out[f"grad_norm_{name}"] = math.sqrt(sq)
    return out


def make_train_loader(dataset, cfg: dict, ctx: ddp.DistContext, seed: int) -> DataLoader:
    sampler = DistributedSampler(
        dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=seed, drop_last=True
    )
    workers = int(get(cfg, "train.num_workers", 8))
    return DataLoader(
        dataset,
        batch_size=int(get(cfg, "train.batch_size", 16)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )


def infinite(loader: DataLoader, start_step: int = 0):
    if len(loader) == 0:
        raise ValueError("training dataset is smaller than one full batch")
    epoch, skip = divmod(start_step, len(loader))
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if i >= skip:
                yield batch
        skip = 0
        epoch += 1


def train(
    cfg: dict, run_dir: Path, train_dataset, val_loader, ctx: ddp.DistContext, logger, evaluate_fn, resume: str | None = None
) -> None:
    """Main loop. ``logger`` has .scalars(step, dict) and .text(msg); ``evaluate_fn(model, step)`` runs eval."""
    seed = int(get(cfg, "train.seed", 0))
    seed_everything(seed + ctx.rank)
    device = ctx.device
    model = build_model(cfg).to(device)
    stats_path = get(cfg, "model.anchor_stats")
    if stats_path:
        model.decoder.set_anchor_statistics(json.loads(Path(stats_path).read_text()))
    optimizer, scheduler = build_optimizer(model, cfg)
    step = 0
    if resume:
        step = load_checkpoint(resume, model.encoder, model.decoder, optimizer, scheduler, rank=ctx.rank)
        logger.text(f"resumed from {resume} at step {step}")
    wrapped = DistributedDataParallel(model, device_ids=[ctx.local_rank]) if ctx.world_size > 1 else model
    loader = make_train_loader(train_dataset, cfg, ctx, seed)
    batches = infinite(loader, start_step=step)
    total_steps = int(get(cfg, "train.steps", 200000))
    log_every, eval_every = int(get(cfg, "train.log_every", 50)), int(get(cfg, "train.eval_every", 5000))
    save_every, clip = int(get(cfg, "train.save_every", 10000)), float(get(cfg, "train.grad_clip", 1.0))
    ckpt_dir = run_dir / "checkpoints"

    def run_evaluation():
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[ctx.local_rank]):
                ddp.rank_zero_call((lambda: evaluate_fn(model, step)) if evaluate_fn is not None else None)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    if step == 0 and bool(get(cfg, "train.eval_at_start", True)):
        run_evaluation()
    model.train()
    t_last = time.time()
    data_time = 0.0
    last_log_step = step
    while step < total_steps:
        t0 = time.time()
        batch = move_batch(next(batches), device)
        data_time += time.time() - t0
        if step == 0:
            validate_batch(batch)
        out = forward_losses(wrapped, batch, cfg, step)
        total = out["total"]
        if not torch.isfinite(total):
            logger.text(f"non-finite loss at step {step}: {out['losses']}")
            raise FloatingPointError(f"non-finite total loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        norms = grad_norms(model) if (step + 1) % log_every == 0 or step == 0 else {}
        optimizer.step()
        scheduler.step()
        step += 1
        if step % log_every == 0 or step == 1:
            scalars = ddp.reduce_mean(
                {
                    **{f"loss/{k}": v for k, v in out["losses"].items()},
                    **{f"metric/{k}": v for k, v in out["metrics"].items()},
                }
            )
            if ctx.is_main:
                now = time.time()
                interval = step - last_log_step
                scalars.update({f"train/{k}": v for k, v in norms.items()})
                scalars.update(
                    {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step_time": (now - t_last) / interval,
                        "train/data_time": data_time / interval,
                        "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                )
                logger.scalars(step, scalars)
                logger.text(
                    f"step {step} loss {scalars['loss/total']:.4f} psnr {scalars['metric/psnr']:.2f} "
                    f"motion {scalars['loss/motion']:.4f} nce {scalars['loss/infonce']:.3f} "
                    f"{scalars['train/step_time']:.3f}s/it"
                )
                t_last, data_time, last_log_step = now, 0.0, step
        if step % save_every == 0 or step == total_steps:
            rng_by_rank = gather_rng_states()
            if ctx.is_main:
                save_checkpoint(
                    ckpt_dir / f"step_{step:07d}.pt",
                    step,
                    model.encoder,
                    model.decoder,
                    optimizer,
                    scheduler,
                    cfg,
                    rng_by_rank=rng_by_rank,
                )
            ddp.barrier()
        if step % eval_every == 0 or step == total_steps:
            run_evaluation()
            model.train()
            t_last = time.time()
