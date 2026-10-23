"""Per-frame encoding pilots (2026-10-23; not the frozen method): one shared single-image encoder, the existing Gaussian
decoder for spatial reconstruction of every sampled frame, and (temporal pilot) a pair-conditioned temporal
reconstruction branch. This is pair-conditioned temporal reconstruction, not causal action-conditioned future
prediction: no action, timestep or horizon input reaches the encoder or the branch.

Encoder input: every (sample, training camera, frame) image on its own (encoder ``num_frames`` = 1) with its own uniform
random mask (``model.encoder.mask_mode`` "random", ``mask_ratio`` 0.75 -> 16 of 64 patches), sampled independently
between images and timesteps: no tube masking, no motion prioritisation. The policy state of a history is the
chronological stack [E(I_t-4), E(I_t-2), E(I_t)] of the per-frame states (``vision.frame_features`` stack in
s4d/rl/encoders.py); ``out["slots"]`` (B,V,T*K,Ds) is that stack for every training camera.

Spatial reconstruction (both pilots): the set decoded from the source camera's state of frame t is rendered at time t in
every training camera (and the synthetic views) with the reference's reconstruction terms (RGB L1 + SSIM with the
motion-score pixel weights, expected-depth L1, alpha coverage, bidirectional 3D Chamfer against the training cameras'
fused depth, synthetic-view RGB/depth on covered pixels), over the full image; the three frames weigh equally (each is a
reconstruction of its own input, so no temporal ramp). View invariance: InfoNCE and slot consistency per frame over the
training cameras plus the synthetic views (each synthetic frame encoded on its own with the same random masking,
restricted to covered patches), averaged over the frames.

Temporal pilot (``model.perframe.temporal``; 75 % source / 75 % target masking, user revision 2026-10-23 before any
launch): for every pair (a, b) of ``PAIRS`` = (0,1), (1,2), (0,2) the dynamic Gaussians of the set decoded from frame a
are moved to time b by the decoder's ``PairTemporalBranch``, conditioned on the source camera's encodings of frame a and
frame b, i.e. the SAME masked encoder outputs (16 of 64 patches each, independent uniform masks) that the two frames'
spatial reconstructions use; no further encoding is computed. Cross-time reconstruction: the moved source set rendered at
time b in every training camera against frame b (RGB, depth, coverage, Chamfer as above, mean over the pairs); M3D on
each pair's own source set and displacement (identities of the source set, never matched across sets). Both are ramped
like the reference's temporal terms (``loss.ramp_steps``) and weigh 1 : 1 against the spatial reconstruction. Gradients
reach the shared encoder through both states. Image sampling, masking and encodings are identical to the spatial pilot;
the extra cost is the branch, three renders of the moved sets in the training cameras and M3D. (The superseded first
definition, never run, used a separate 90 %-masked target encoding: commit 5fd623e.)
"""

from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel

from s4d.config import get
from s4d.data.contract import PAIRS, motion_targets
from s4d.losses.chamfer import chamfer_loss, target_clouds
from s4d.losses.depth import abs_rel, depth_l1
from s4d.losses.invariance import multi_positive_info_nce, slot_consistency, state_statistics
from s4d.losses.motion import expected_displacement, motion_loss
from s4d.losses.motion3d import motion3d_loss
from s4d.losses.occlusion import usage_diagnostics
from s4d.losses.rgb import coverage_loss, masked_psnr, masked_ssim, pixel_weights, rgb_loss
from s4d.model.encoder import visible_count
from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet
from s4d.model.render import render_features, render_rgbd
from s4d.train.augment import coverage_visible_tokens, jitter_cameras, synthesize_views

MOVING_SCORE = 0.5
STATIC_SCORE = 0.05
ACTIVE_OPACITY = 0.05
UNSUPPORTED = {  # reference options outside the pilot definition: refused instead of silently ignored
    "loss.depth_grad": 0.0, "loss.depth_hard": 0.0, "loss.visibility": 0.0, "loss.occlusion": 0.0,
    "loss.self_render": 0.0, "loss.depth_align": "none", "loss.teacher_scale_jitter": None,
}


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[3:])


def _per_t(x: torch.Tensor, B: int, T: int, V: int) -> torch.Tensor:
    return x.view(B, T, V).mean(dim=(0, 2))


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    return torch.nanmean(x) if torch.isfinite(x).any() else x.new_tensor(float("nan"))


def _ramp(step: int, ramp_steps: int) -> float:
    return 1.0 if ramp_steps <= 0 else min(1.0, max(0.0, step / ramp_steps))


def index_set(gs: GaussianSet, rows: torch.Tensor) -> GaussianSet:
    """The sets of batch rows ``rows`` (all attributes, displacements included)."""
    return GaussianSet(gs.xyz[rows], gs.scales[rows], gs.quats[rows], gs.opacity[rows], gs.rgb[rows], gs.delta01[rows],
                       gs.delta12[rows], gs.group)


def _float_set(gs: GaussianSet) -> GaussianSet:
    return GaussianSet(gs.xyz.float(), gs.scales.float(), gs.quats.float(), gs.opacity.float(), gs.rgb.float(),
                       gs.delta01.float(), gs.delta12.float(), gs.group)


def _per_time_cams(x: torch.Tensor, S: int) -> torch.Tensor:
    """(B,V,…) cameras -> (B*S,V,…), the same rig for each of S per-time sets of a sample."""
    return x[:, None].expand(x.shape[0], S, *x.shape[1:]).reshape(x.shape[0] * S, *x.shape[1:])


def render_per_time(gs: GaussianSet, xyz: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor, S: int, H: int, W: int,
                    near: float, far: float) -> dict[str, torch.Tensor]:
    """Render the (B*S) sets ``gs`` at centres ``xyz`` (B*S,N,3) in the cameras of their sample (B,V,…) ->
    rgb/depth/alpha (B,S,V,…)."""
    r = render_rgbd(gs, xyz[:, None], _per_time_cams(w2c, S), _per_time_cams(K, S), H, W, near, far)
    B = w2c.shape[0]
    return {k: v.reshape(B, S, *v.shape[2:]) for k, v in r.items()}


def check_perframe_config(cfg: dict) -> dict:
    pf = get(cfg, "model.perframe")
    if not isinstance(pf, dict):
        raise ValueError("model.perframe must be a mapping {temporal}")
    unknown = set(pf) - {"temporal"}
    if unknown:
        raise ValueError(f"unknown model.perframe keys {sorted(unknown)}")
    for key, value in UNSUPPORTED.items():
        if get(cfg, key, value) != value:
            raise ValueError(f"per-frame pilots require {key} = {value!r}")
    aug = get(cfg, "aug", None) or {}
    if aug.get("crop") or aug.get("shift"):
        raise ValueError("per-frame pilots do not define crop/shift encoder augmentation")
    if get(cfg, "data.objective", "reconstruct") != "reconstruct" or get(cfg, "model.single_frame", False):
        raise ValueError("per-frame pilots use the reconstruction objective and never model.single_frame")
    if str(get(cfg, "loss.motion_space", "gaussian")) != "gaussian":
        raise ValueError("loss.motion_space must be 'gaussian' (M3D)")
    return pf


def encode_frames(encoder, frames01: torch.Tensor, mask_ratio: float | None, bf16: bool) -> torch.Tensor:
    """frames01 (R,T,3,H,W) -> per-frame slots (R,T,K,Ds), each image encoded on its own with its own mask."""
    R, T = frames01.shape[:2]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        slots = encoder(frames01.reshape(R * T, 1, *frames01.shape[2:]), None, mask_ratio=mask_ratio)["slots"]
    return slots.view(R, T, *slots.shape[1:])


def forward_losses_perframe(
    model,
    batch: dict,
    cfg: dict,
    step: int,
    *,
    source: torch.Tensor | None = None,
    mask_ratio: float | None = None,
    return_renders: bool = False,
) -> dict:
    """``forward_losses`` of the per-frame pilots (module docstring); the same output keys. ``gs`` holds the B*T
    per-frame sets (row b*T + t: frame t of sample b, decoded from the source camera)."""
    if isinstance(model, DistributedDataParallel):
        raise ValueError("the per-frame pilots run in one process")
    if "history_images" in batch:
        raise ValueError("per-frame pilots use reconstruction batches")
    pf = check_perframe_config(cfg)
    temporal = bool(pf.get("temporal", False))
    loss_cfg = cfg["loss"]
    near, far = float(cfg["render"]["near"]), float(cfg["render"]["far"])
    encoder, decoder = model.encoder, model.decoder
    if encoder.cfg.num_frames != 1:
        raise ValueError("per-frame pilots need a single-image encoder (model.encoder.num_frames = 1)")
    if temporal != (decoder.temporal is not None):
        raise ValueError("model.perframe.temporal must match model.decoder.temporal_branch")
    images = batch["images"]
    B, T, V, _, H, W = images.shape
    if T != 3:
        raise ValueError("per-frame pilots sample three-frame windows")
    device = images.device
    images01 = images.float() / 255.0
    score = batch["motion_score"].float()
    training = model.training
    aug = get(cfg, "aug", None) or {}
    bf16 = bool(get(cfg, "train.bf16", True))
    if source is None:
        source = torch.randint(V, (B,), device=device)
    ar = torch.arange(B, device=device)
    N_tok = encoder.cfg.num_patches
    ratio = encoder.cfg.mask_ratio if mask_ratio is None else mask_ratio

    # ---- per-frame encoding (rows b, v, t) and spatial decoding from the source camera -----------------------
    enc_slots = encode_frames(encoder, images01.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 3, H, W), mask_ratio, bf16)
    Ks, Ds = enc_slots.shape[2:]
    enc_slots = enc_slots.view(B, V, T, Ks, Ds)
    src_slots = enc_slots[ar, source]  # (B,T,K,Ds), the encodings the decoder and the branch see
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        decoded = decoder(src_slots.reshape(B * T, Ks, Ds), return_features=temporal)
    gs, dyn_feats = decoded if temporal else (decoded, None)
    gs = _float_set(gs)
    slots = enc_slots.float()  # (B,V,T,K,Ds)
    N = gs.num_gaussians
    xyz_t = gs.xyz.view(B, T, N, 3)
    w2c, K = batch["w2c"], batch["K"]

    # ---- spatial reconstruction ----------------------------------------------------------------------------
    r = render_per_time(gs, gs.xyz, w2c, K, T, H, W, near, far)  # (B,T,V,…)
    depth_t = batch["depth"].float()
    valid_far = bool(loss_cfg.get("depth_valid_far", False))
    weights = pixel_weights(_rows(score), float(loss_cfg.get("lambda_dyn", 1.0)))
    valid = (_rows(depth_t) > 0) & (_rows(depth_t) < far) if valid_far else _rows(depth_t) > 0
    rgb_rows, tgt_rows, depth_rows, alpha_rows = _rows(r["rgb"]), _rows(images01), _rows(r["depth"]), _rows(r["alpha"])
    ssim_w = float(loss_cfg.get("ssim_weight", 0.2))
    w_rgb, w_cov, w_depth = float(loss_cfg.get("rgb", 1.0)), float(loss_cfg.get("coverage", 0.1)), float(
        loss_cfg.get("depth_l1", 1.0))
    rgb_l = rgb_loss(rgb_rows, tgt_rows, weights, ssim_w)
    cov_l = coverage_loss(alpha_rows, valid, weights)
    dl1 = depth_l1(depth_rows, depth_t.reshape(-1, *depth_t.shape[3:]), valid, weights)
    render_t = w_rgb * _per_t(rgb_l, B, T, V) + w_cov * _per_t(cov_l, B, T, V) + w_depth * _per_t(dl1, B, T, V)
    cham_w = float(loss_cfg.get("chamfer", 0.0))
    cham, cham_stats, clouds, cloud_valid = None, {}, None, None
    if cham_w > 0:
        clouds, cloud_valid, cham_stats = target_clouds(
            batch["depth"][:, :, :, 0], K, batch["c2w"], far, valid_far, float(loss_cfg.get("chamfer_voxel_m", 0.01)),
            int(loss_cfg.get("chamfer_points", 4096)),
        )
        cham = chamfer_loss(xyz_t, clouds, cloud_valid)
        render_t = render_t + cham_w * cham["loss"]
    spatial = render_t.mean()
    extra_losses = {}

    # ---- synthetic views: invariance positives (each frame on its own) and render targets ------------------------
    n_syn = int(aug.get("synth_views", 0)) if training else 0
    syn_slots = None
    if n_syn:
        syn_cams = jitter_cameras(B, n_syn, K[:, 0], H, W, float(aug.get("synth_range_scale", 1.0)))
        syn = synthesize_views(batch, syn_cams, near, far)  # (B,T,n,…)
        syn_in = syn["images"].permute(0, 2, 1, 3, 4, 5).reshape(B * n_syn * T, 1, 3, H, W)
        syn_cov = syn["covered"].permute(0, 2, 1, 3, 4, 5).reshape(B * n_syn * T, 1, 1, H, W)
        syn_visible = coverage_visible_tokens(syn_cov, encoder.cfg.patch_size, visible_count(N_tok, ratio))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            syn_slots = encoder.forward_visible(syn_in, syn_visible)["slots"]
        syn_slots = syn_slots.float().view(B, n_syn, T, Ks, Ds)
        if aug.get("synth_render", False):
            rs = render_per_time(gs, gs.xyz, syn_cams["w2c"], syn_cams["K"], T, H, W, near, far)  # (B,T,n,…)
            cov_rows = _rows(syn["covered"]).float()
            cov_w = cov_rows / cov_rows.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            syn_rgb = rgb_loss(_rows(rs["rgb"]), _rows(syn["images"]), cov_w, ssim_w)
            syn_depth = depth_l1(_rows(rs["depth"]), _rows(syn["depth"]), cov_rows > 0, torch.ones_like(cov_rows))
            extra_losses["synth_render"] = (w_rgb * _per_t(syn_rgb, B, T, n_syn) + w_depth * _per_t(
                syn_depth, B, T, n_syn)).mean()

    # ---- invariance per frame ---------------------------------------------------------------------------------
    nces, conss, pos_cos, neg_cos = [], [], [], []
    for t in range(T):
        inv = slots[:, :, t] if syn_slots is None else torch.cat((slots[:, :, t], syn_slots[:, :, t]), 1)
        nce_t, nce_m = multi_positive_info_nce(inv.mean(2), float(loss_cfg.get("temperature", 0.1)), distributed=False)
        nces.append(nce_t)
        conss.append(slot_consistency(inv))
        pos_cos.append(nce_m["positive_cosine"])
        neg_cos.append(nce_m["negative_cosine"])
    nce, cons = torch.stack(nces).mean(), torch.stack(conss).mean()

    # ---- temporal branch: cross-time reconstruction and M3D (temporal pilot) ---------------------------------
    log_step = not training or step == 0 or (step + 1) % int(get(cfg, "train.log_every", 50)) == 0
    features_needed = log_step or return_renders
    ramp = _ramp(step, int(loss_cfg.get("ramp_steps", 20000))) if temporal else 0.0
    m_loss = images01.new_zeros(())
    m_metrics: dict = {}
    temporal_metrics: dict = {}
    cross_out: dict = {}
    if temporal:
        if motion_targets(batch) != "dense":
            raise ValueError("the per-frame temporal pilot is defined for dense Meta-World motion targets")
        dyn = gs.group == DYNAMIC_GROUP
        dyn_idx = torch.nonzero(dyn, as_tuple=False)[:, 0]
        feats = dyn_feats.view(B, T, dyn_feats.shape[1], dyn_feats.shape[2])
        sources, disps = [], []
        for a, b, _ in PAIRS:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                d = decoder.temporal(feats[:, a], src_slots[:, a], src_slots[:, b])  # the frames' own encodings
            disps.append(torch.zeros(B, N, 3, device=device, dtype=d.dtype).index_copy(1, dyn_idx, d))
            sources.append(xyz_t[:, a])
        P = len(PAIRS)
        src_t, tgt_t = [a for a, _, _ in PAIRS], [b for _, b, _ in PAIRS]
        moved = torch.stack([s + d for s, d in zip(sources, disps)], 1)  # (B,P,N,3)
        pair_rows = (ar[:, None] * T + torch.tensor(src_t, device=device)[None]).reshape(-1)
        gs_pairs = index_set(gs, pair_rows)
        rx = render_per_time(gs_pairs, moved.reshape(B * P, N, 3), w2c, K, P, H, W, near, far)  # (B,P,V,…)
        x_tgt = images01[:, tgt_t]
        x_depth = depth_t[:, tgt_t]
        x_weights = pixel_weights(_rows(score[:, tgt_t]), float(loss_cfg.get("lambda_dyn", 1.0)))
        x_valid = (_rows(x_depth) > 0) & (_rows(x_depth) < far) if valid_far else _rows(x_depth) > 0
        x_rgb_rows = _rows(rx["rgb"])
        x_rgb_l = rgb_loss(x_rgb_rows, _rows(x_tgt), x_weights, ssim_w)
        x_cov_l = coverage_loss(_rows(rx["alpha"]), x_valid, x_weights)
        x_dl1 = depth_l1(_rows(rx["depth"]), _rows(x_depth), x_valid, x_weights)
        cross_t = w_rgb * _per_t(x_rgb_l, B, P, V) + w_cov * _per_t(x_cov_l, B, P, V) + w_depth * _per_t(x_dl1, B, P, V)
        x_cham = None
        if cham_w > 0:
            x_cham = chamfer_loss(moved, clouds[:, tgt_t], cloud_valid[:, tgt_t])
            cross_t = cross_t + cham_w * x_cham["loss"]
        extra_losses["cross_render"] = cross_t.mean()
        motion_weight = batch["motion_weight"]
        if valid_far:
            motion_weight = motion_weight * (batch["depth"][:, src_t] < far).to(motion_weight.dtype)
        m_loss, m3d_metrics = motion3d_loss(None, None, dyn, batch["depth"], K, batch["c2w"], batch["motion3d"],
                                            motion_weight, pair_sources=sources, pair_displacements=disps)
        m_metrics.update(m3d_metrics)
        if features_needed:  # image-space EPE metrics of the rendered displacements, as the reference (no loss)
            is_dyn = dyn.float()[None, :, None].expand(B, -1, 1)
            feat0 = torch.cat((disps[0], disps[2], is_dyn), -1)  # source set 0 at t0: pairs (0,1) and (0,2)
            feat1 = torch.cat((disps[1], torch.zeros_like(disps[1]), is_dyn), -1)  # source set 1 at t1: pair (1,2)
            src_rows = (ar[:, None] * T + torch.arange(2, device=device)[None]).reshape(-1)
            gs_src = index_set(gs, src_rows).detach_geometry()
            fr = render_features(gs_src, gs_src.xyz.detach()[:, None], torch.stack((feat0, feat1), 1).reshape(B * 2, 1, N, 7),
                                 _per_time_cams(w2c, 2), _per_time_cams(K, 2), H, W, near, far)
            fts = fr["features"].reshape(B, 2, V, 7, H, W)
            cv = fr["alpha"].detach().reshape(B, 2, V, 1, H, W)
            pred_feat = torch.stack((fts[:, 0, :, 0:3], fts[:, 1, :, 0:3], fts[:, 0, :, 3:6]), 1)
            coverage = torch.stack((cv[:, 0], cv[:, 1], cv[:, 0]), 1)
            pred_disp = expected_displacement(pred_feat, coverage)
            image_loss, mm = motion_loss(pred_disp, coverage, batch["motion3d"], motion_weight,
                                         float(loss_cfg.get("motion_huber", 0.01)),
                                         normalization=str(loss_cfg.get("motion_norm", "valid")))
            m_metrics.update(mm)
            m_metrics["motion_image"] = image_loss.detach()
            cross_out.update({"pred_disp": pred_disp.detach(), "coverage": coverage.detach()})
        with torch.no_grad():
            x_score = _rows(score[:, tgt_t])
            ones = torch.ones_like(x_score, dtype=torch.bool)
            psnr_x = masked_psnr(x_rgb_rows, _rows(x_tgt), ones).view(B, P, V)
            psnr_xm = masked_psnr(x_rgb_rows, _rows(x_tgt), x_score > MOVING_SCORE).view(B, P, V)
            for p, (a, b, _) in enumerate(PAIRS):
                temporal_metrics[f"psnr_cross_{a}{b}"] = _nanmean(psnr_x[:, p])
                temporal_metrics[f"psnr_cross_moving_{a}{b}"] = _nanmean(psnr_xm[:, p])
            temporal_metrics["translation_dyn_mean"] = torch.stack([d[:, dyn_idx].norm(dim=-1).mean() for d in disps]).mean()
            if not training:  # persistence reference: the source set rendered unmoved at the target time
                rp = render_per_time(gs_pairs, torch.stack(sources, 1).reshape(B * P, N, 3), w2c, K, P, H, W, near, far)
                psnr_p = masked_psnr(_rows(rp["rgb"]), _rows(x_tgt), ones).view(B, P, V)
                psnr_pm = masked_psnr(_rows(rp["rgb"]), _rows(x_tgt), x_score > MOVING_SCORE).view(B, P, V)
                for p, (a, b, _) in enumerate(PAIRS):
                    temporal_metrics[f"psnr_persist_{a}{b}"] = _nanmean(psnr_p[:, p])
                    temporal_metrics[f"psnr_persist_moving_{a}{b}"] = _nanmean(psnr_pm[:, p])
                cross_out["persist_rgb"] = rp["rgb"].detach()
        cross_out["cross_rgb"] = rx["rgb"].detach()
        cross_out["cross_depth"] = rx["depth"].detach()

    total = spatial + float(loss_cfg.get("invariance", 1.0)) * nce + float(loss_cfg.get("consistency", 0.5)) * cons
    if "synth_render" in extra_losses:
        total = total + float(aug.get("synth_render_weight", 1.0)) * extra_losses["synth_render"]
    if temporal:
        total = total + ramp * (extra_losses["cross_render"] + float(loss_cfg.get("motion", 1.0)) * m_loss)

    # ---- metrics ------------------------------------------------------------------------------------------------
    state = slots.reshape(B, V, T * Ks * Ds)  # the policy state: chronological stack of the per-frame states
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
            "opacity_mean": opacity.mean(),
            "opacity_std": opacity.std(unbiased=False),
            "active_fraction": (opacity > ACTIVE_OPACITY).float().mean(),
            "dynamic_score_moving_count": (score > MOVING_SCORE).float().sum(),
            "positive_cosine": torch.stack(pos_cos).mean(),
            "negative_cosine": torch.stack(neg_cos).mean(),
            "encoder_visible_tokens": torch.tensor(float(visible_count(N_tok, ratio) if (training and ratio > 0) else N_tok),
                                                   device=device),
            **temporal_metrics,
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
        used = None
        if log_step:
            metrics.update({k: v.detach() for k, v in cham_stats.items()})
            metrics.update(state_statistics(state))
            first = index_set(gs, ar * T)  # the sets decoded from frame 0, as the reference's usage diagnostics
            usage, used = usage_diagnostics(first, batch["depth"][:, 0], w2c, K, near, far,
                                            float(loss_cfg.get("occlusion_margin", 0.02)))
            metrics.update({k: _nanmean(v) for k, v in usage.items()})
    losses = {
        "total": total.detach(),
        "render": spatial.detach(),
        "motion": m_loss.detach(),
        "infonce": nce.detach(),
        "consistency": cons.detach(),
        **{f"rgb_t{t}": v.detach() for t, v in enumerate(_per_t(rgb_l, B, T, V))},
        **{f"coverage_t{t}": v.detach() for t, v in enumerate(_per_t(cov_l, B, T, V))},
        **{f"depth_l1_t{t}": v.detach() for t, v in enumerate(_per_t(dl1, B, T, V))},
        **({f"chamfer_t{t}": v.detach() for t, v in enumerate(cham["loss"])} if cham is not None else {}),
        **{k: v.detach() for k, v in extra_losses.items()},
    }
    if temporal:
        for p, (a, b, _) in enumerate(PAIRS):
            losses[f"cross_rgb_{a}{b}"] = _per_t(x_rgb_l, B, P, V)[p].detach()
            losses[f"cross_depth_l1_{a}{b}"] = _per_t(x_dl1, B, P, V)[p].detach()
            if x_cham is not None:
                losses[f"cross_chamfer_{a}{b}"] = x_cham["loss"][p].detach()
    out = {"total": total, "losses": losses, "metrics": metrics, "slots": slots.reshape(B, V, T * Ks, Ds),
           "source": source, "gs": gs, "used": used}
    if return_renders:
        out.update({"rgb": r["rgb"].detach(), "depth": r["depth"].detach(), "alpha": r["alpha"].detach(),
                    "pairs": PAIRS, **cross_out})
    return out
