from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.losses import (
    compute_balanced_silhouette_loss,
    compute_global_local_depth_loss,
    compute_reconstruction_loss,
)
from models.splatter import SplatterConfig, render_predicted
from models.gaussians import DirectSplatterToGaussians
from models.train_config import TrainConfig
from models.vae import SplatterVAE
from utils.camera_tensor_utils import gather_camera_rows


def encode_per_view_sequence_batch(
    vae: SplatterVAE,
    images: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Encode every sample/view sequence as an independent contrastive item.

    Returns ``s_inv_by_view`` and ``z_dep_by_view`` with shape ``(B,A,D)`` so
    the caller can define positives by batch/sample id or by camera id.
    """
    if images.dim() == 6:
        bsz, timesteps, num_views, channels, height, width = images.shape
        view_sequences = images.permute(0, 2, 1, 3, 4, 5).reshape(
            bsz * num_views,
            timesteps,
            1,
            channels,
            height,
            width,
        )
    elif images.dim() == 5:
        bsz, num_views, channels, height, width = images.shape
        view_sequences = images.reshape(bsz * num_views, 1, channels, height, width)
    else:
        raise ValueError(f"Expected images as (B,T,A,3,H,W) or legacy (B,A,3,H,W), got {tuple(images.shape)}.")

    latents, inv_loss, dep_loss = vae.encode_sequence(view_sequences)
    s_inv_by_view = latents["s_inv"].view(bsz, num_views, -1).contiguous()
    z_dep_by_view = latents["z_dep_all"][:, 0].view(bsz, num_views, -1).contiguous()
    out = {
        "s_inv_by_view": s_inv_by_view,
        "z_dep_by_view": z_dep_by_view,
        "inv_mask_by_view": latents["inv_mask"].view(bsz, num_views, *latents["inv_mask"].shape[1:]).contiguous(),
        "dep_mask_by_view": latents["dep_mask"].view(bsz, num_views, *latents["dep_mask"].shape[1:]).contiguous(),
    }
    return out, inv_loss, dep_loss


def apply_source_mask_to_gaussians(
    pc: Dict[str, torch.Tensor],
    source_masks: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if source_masks is None:
        return pc
    mask = source_masks.to(device=pc["xyz"].device, dtype=torch.bool)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected source masks as (B,1,H,W), got {tuple(mask.shape)}.")
    flat_mask = mask.flatten(start_dim=2).squeeze(1)
    num_gaussians = pc["xyz"].shape[1]
    if flat_mask.shape[1] != num_gaussians:
        if num_gaussians % flat_mask.shape[1] != 0:
            raise ValueError(
                f"Cannot align {flat_mask.shape[1]} source mask pixels with {num_gaussians} Gaussians."
            )
        repeat = num_gaussians // flat_mask.shape[1]
        flat_mask = flat_mask.repeat_interleave(repeat, dim=1)
    pc = dict(pc)
    valid = pc.get("valid_mask", torch.ones_like(flat_mask, dtype=torch.bool)) & flat_mask
    pc["valid_mask"] = valid.contiguous()
    pc["opacity"] = pc["opacity"] * flat_mask.unsqueeze(-1).to(dtype=pc["opacity"].dtype)
    return pc


def _ensure_temporal_images(images_01: torch.Tensor) -> torch.Tensor:
    if images_01.dim() == 5:
        return images_01[:, None].contiguous()
    if images_01.dim() != 6:
        raise ValueError(f"Expected images_01 as (B,T,A,3,H,W), got {tuple(images_01.shape)}.")
    return images_01


def _ensure_temporal_optional(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.dim() == 5:
        return value[:, None].contiguous()
    if value.dim() != 6:
        raise ValueError(f"Expected optional temporal tensor as (B,T,A,C,H,W), got {tuple(value.shape)}.")
    return value


def _expand_camera_time(camera: torch.Tensor, timesteps: int) -> torch.Tensor:
    if camera.dim() == 4:
        return camera[:, None].expand(-1, timesteps, -1, -1, -1).contiguous()
    if camera.dim() == 5:
        if camera.shape[1] != timesteps:
            raise ValueError(f"Camera tensor has T={camera.shape[1]}, expected {timesteps}.")
        return camera
    raise ValueError(f"Expected camera tensor as (B,A,...) or (B,T,A,...), got {tuple(camera.shape)}.")


def _gather_source_time0(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    if values.dim() >= 5:
        values = values[:, 0]
    return gather_camera_rows(values, source_indices)


def _gather_source_sequence(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    if values.dim() == 5:
        return gather_camera_rows(values, source_indices)
    if values.dim() != 6:
        raise ValueError(f"Expected temporal values as (B,T,A,C,H,W), got {tuple(values.shape)}.")
    batch_ids = torch.arange(values.shape[0], device=values.device)
    return values[batch_ids, :, source_indices].contiguous()


def _farthest_point_sample_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    """Discrete 3D FPS indices; gathered coordinates retain their gradients."""
    num_points = points.shape[0]
    if num_points <= count:
        return torch.arange(num_points, device=points.device, dtype=torch.long)
    detached = points.detach().float()
    centroid = detached.mean(dim=0, keepdim=True)
    first = torch.argmax((detached - centroid).square().sum(dim=-1))
    selected = torch.empty((count,), device=points.device, dtype=torch.long)
    selected[0] = first
    min_distance = (detached - detached[first]).square().sum(dim=-1)
    for slot in range(1, count):
        next_index = torch.argmax(min_distance)
        selected[slot] = next_index
        distance = (detached - detached[next_index]).square().sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
    return selected


def sample_sparse_motion_controls(
    dense_xyz: torch.Tensor,
    dense_valid_mask: torch.Tensor,
    num_controls: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FPS only valid foreground centers, padding unused tensor slots as invalid."""
    max_controls = int(num_controls)
    if max_controls <= 0:
        raise ValueError(f"num_controls must be positive, got {num_controls}.")
    batch = dense_xyz.shape[0]
    control_xyz = dense_xyz.new_zeros((batch, max_controls, 3))
    control_indices = torch.full(
        (batch, max_controls),
        -1,
        device=dense_xyz.device,
        dtype=torch.long,
    )
    control_valid_mask = torch.zeros((batch, max_controls), device=dense_xyz.device, dtype=torch.bool)
    for batch_idx in range(batch):
        valid_indices = torch.nonzero(
            dense_valid_mask[batch_idx].to(dtype=torch.bool)
            & torch.isfinite(dense_xyz[batch_idx]).all(dim=-1),
            as_tuple=False,
        ).flatten()
        if valid_indices.numel() == 0:
            continue
        selected_local = _farthest_point_sample_indices(
            dense_xyz[batch_idx, valid_indices],
            min(max_controls, valid_indices.numel()),
        )
        selected = valid_indices[selected_local]
        selected_count = selected.numel()
        control_xyz[batch_idx, :selected_count] = dense_xyz[batch_idx, selected]
        control_indices[batch_idx, :selected_count] = selected
        control_valid_mask[batch_idx, :selected_count] = True
    return control_xyz, control_indices, control_valid_mask


def normalize_control_coordinates(
    control_xyz: torch.Tensor,
    control_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize each valid t0 control set to a stable [-1,1] coordinate box."""
    mask = control_valid_mask.unsqueeze(-1)
    finite_max = torch.finfo(control_xyz.dtype).max
    minimum = torch.where(mask, control_xyz, finite_max).amin(dim=1, keepdim=True)
    maximum = torch.where(mask, control_xyz, -finite_max).amax(dim=1, keepdim=True)
    has_controls = control_valid_mask.any(dim=1).view(-1, 1, 1)
    minimum = torch.where(has_controls, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_controls, maximum, torch.ones_like(maximum))
    center = (0.5 * (minimum + maximum)).detach()
    half_extent = (0.5 * (maximum - minimum)).clamp_min(1.0e-4).detach()
    normalized = ((control_xyz - center) / half_extent).clamp(-1.0, 1.0)
    return normalized * mask.to(dtype=normalized.dtype)


def compute_dense_control_associations(
    dense_xyz: torch.Tensor,
    dense_valid_mask: torch.Tensor,
    control_xyz: torch.Tensor,
    control_valid_mask: torch.Tensor,
    num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one t0 inverse-distance k-NN table reused by both transitions."""
    neighbors = max(1, min(int(num_neighbors), control_xyz.shape[1]))
    distances = torch.cdist(dense_xyz, control_xyz)
    distances = distances.masked_fill(~control_valid_mask[:, None, :], torch.inf)
    nearest_distances, nearest_indices = torch.topk(
        distances,
        k=neighbors,
        dim=-1,
        largest=False,
        sorted=True,
    )
    batch_indices = torch.arange(dense_xyz.shape[0], device=dense_xyz.device)[:, None, None]
    neighbor_valid = control_valid_mask[batch_indices, nearest_indices]
    inverse_distance = torch.where(
        neighbor_valid & torch.isfinite(nearest_distances),
        nearest_distances.clamp_min(1.0e-6).reciprocal(),
        torch.zeros_like(nearest_distances),
    )
    weights = inverse_distance / inverse_distance.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    weights = weights * dense_valid_mask.unsqueeze(-1).to(dtype=weights.dtype)
    return nearest_indices, weights


def interpolate_control_motion(
    control_motion: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
) -> torch.Tensor:
    batch_indices = torch.arange(control_motion.shape[0], device=control_motion.device)[:, None, None]
    neighbor_motion = control_motion[batch_indices, neighbor_indices]
    return (neighbor_motion * neighbor_weights.unsqueeze(-1)).sum(dim=-2).contiguous()


def translate_gaussians(
    pc: Dict[str, torch.Tensor],
    xyz_delta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if xyz_delta.shape != pc["xyz"].shape:
        raise ValueError(f"Translation shape {tuple(xyz_delta.shape)} does not match {tuple(pc['xyz'].shape)}.")
    out = dict(pc)
    out["xyz"] = (pc["xyz"] + xyz_delta).contiguous()
    return out


def _masked_motion_magnitude_mean(motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    magnitudes = motion.norm(dim=-1)
    weights = valid_mask.to(dtype=magnitudes.dtype)
    return (magnitudes * weights).sum() / weights.sum().clamp_min(1.0)



def _render_sequence(
    pc_sequence: list[Dict[str, torch.Tensor]],
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    bg: torch.Tensor,
    splatter_cfg: SplatterConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_renders = []
    depth_renders = []
    alpha_renders = []
    timesteps = w2c.shape[1]
    for time_idx in range(timesteps):
        pc = pc_sequence[min(time_idx, len(pc_sequence) - 1)]
        out = render_predicted(
            pc=pc,
            world_view_transform=w2c[:, time_idx],
            intrinsics=intrinsics[:, time_idx],
            bg_color=bg,
            cfg=splatter_cfg,
            render_mode="RGB+ED",
        )
        if out["render"] is None or out["depth"] is None:
            raise RuntimeError("RGB+ED rendering must return both RGB and expected depth.")
        rgb_renders.append(out["render"])
        depth_renders.append(out["depth"])
        alpha_renders.append(out["alpha"])
    return (
        torch.stack(rgb_renders, dim=1).contiguous(),
        torch.stack(depth_renders, dim=1).contiguous(),
        torch.stack(alpha_renders, dim=1).contiguous(),
    )


def _exact_mask_sequence(images_01: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
    if masks is None:
        raise ValueError(
            "Sparse control sampling and RGB-silhouette-depth supervision require exact segmentation masks."
        )
    expected_shape = (*images_01.shape[:3], 1, *images_01.shape[-2:])
    if tuple(masks.shape) != expected_shape:
        raise ValueError(f"Expected exact masks with shape {expected_shape}, got {tuple(masks.shape)}.")
    return masks.to(device=images_01.device, dtype=images_01.dtype).clamp(0.0, 1.0)


def _mean_valid_opacity(pc: Dict[str, torch.Tensor]) -> torch.Tensor:
    valid_mask = pc.get(
        "valid_mask",
        torch.ones(pc["xyz"].shape[:2], device=pc["xyz"].device, dtype=torch.bool),
    ).to(dtype=torch.bool)
    opacity = pc["opacity"]
    opacity_mask = valid_mask.unsqueeze(-1).expand_as(opacity)
    if bool(opacity_mask.any()):
        return opacity.masked_select(opacity_mask).mean()
    return opacity.new_zeros(())


def compute_reconstruction_and_renders(
    vae: SplatterVAE,
    splatter_to_gaussians: DirectSplatterToGaussians,
    splatter_cfg: SplatterConfig,
    images_01: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    bg: torch.Tensor,
    cfg_train: TrainConfig,
    depths: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    return_renders: bool = False,
) -> Dict[str, Any]:
    """Encode one sampled source-view sequence, decode it, and render all views."""
    images_01 = _ensure_temporal_images(images_01)
    depths = _ensure_temporal_optional(depths)
    masks = _ensure_temporal_optional(masks)
    bsz, timesteps, num_views = images_01.shape[:3]
    if timesteps != 3:
        raise ValueError(f"Sparse temporal reconstruction requires exactly three RGB frames, got T={timesteps}.")
    if masks is None:
        raise ValueError(
            "Sparse control sampling and RGB-silhouette-depth supervision require exact segmentation masks."
        )
    if num_views < 1:
        raise ValueError("Temporal reconstruction requires at least one camera viewpoint.")
    original_masks = _exact_mask_sequence(images_01, masks)

    intrinsics_t = _expand_camera_time(intrinsics, timesteps)
    c2w_t = _expand_camera_time(c2w, timesteps)
    w2c_t = _expand_camera_time(w2c, timesteps)

    device = images_01.device
    source_indices = torch.randint(low=0, high=num_views, size=(bsz,), device=device)

    source_images_01 = _gather_source_sequence(images_01, source_indices)
    source_images = source_images_01.mul(2.0).sub(1.0).unsqueeze(2).contiguous()
    source_latents, _inv_embed_loss, _dep_embed_loss = vae.encode_sequence(source_images)
    s_inv = source_latents["s_inv"]
    z_dep_source = source_latents["z_dep_all"][:, 0]

    decoded = vae.decode_sequence(s_inv=s_inv, z_dep_source=z_dep_source)
    source_intrinsics = _gather_source_time0(intrinsics_t, source_indices)
    source_c2w = _gather_source_time0(c2w_t, source_indices)
    pc0 = splatter_to_gaussians(
        splatter_map=decoded["base_map"],
        source_cameras_view_to_world=source_c2w,
        intrinsics=source_intrinsics,
        activate_output=True,
    )
    source_masks = _gather_source_time0(original_masks, source_indices)
    pc0 = apply_source_mask_to_gaussians(pc0, source_masks)
    dense_valid_mask = pc0.get(
        "valid_mask",
        torch.ones(pc0["xyz"].shape[:2], device=pc0["xyz"].device, dtype=torch.bool),
    ).to(dtype=torch.bool)

    control_xyz0, control_indices, control_valid_mask = sample_sparse_motion_controls(
        dense_xyz=pc0["xyz"],
        dense_valid_mask=dense_valid_mask,
        num_controls=int(cfg_train.num_motion_controls),
    )
    normalized_controls = normalize_control_coordinates(control_xyz0, control_valid_mask)
    control_delta01, control_delta12 = vae.predict_control_motion(
        s_inv=s_inv,
        normalized_control_xyz=normalized_controls,
        control_valid_mask=control_valid_mask,
        delta_max=float(cfg_train.motion_delta_max),
    )
    neighbor_indices, neighbor_weights = compute_dense_control_associations(
        dense_xyz=pc0["xyz"],
        dense_valid_mask=dense_valid_mask,
        control_xyz=control_xyz0,
        control_valid_mask=control_valid_mask,
        num_neighbors=int(cfg_train.motion_num_neighbors),
    )
    dense_motion01 = interpolate_control_motion(control_delta01, neighbor_indices, neighbor_weights)
    dense_motion12 = interpolate_control_motion(control_delta12, neighbor_indices, neighbor_weights)
    pc1 = translate_gaussians(pc0, dense_motion01)
    pc2 = translate_gaussians(pc1, dense_motion12)
    pc_sequence = [pc0, pc1, pc2]
    control_xyz1 = control_xyz0 + control_delta01
    control_xyz2 = control_xyz1 + control_delta12
    control_xyz_sequence = torch.stack([control_xyz0, control_xyz1, control_xyz2], dim=1)

    rendered, rendered_depth, rendered_alpha = _render_sequence(
        pc_sequence=pc_sequence,
        w2c=w2c_t,
        intrinsics=intrinsics_t,
        bg=bg,
        splatter_cfg=splatter_cfg,
    )
    target_images = images_01 * original_masks

    rec_loss = compute_reconstruction_loss(
        predicted=rendered.reshape(-1, *rendered.shape[3:]),
        ground_truth=target_images.reshape(-1, *target_images.shape[3:]),
        ssim_weight=float(cfg_train.ssim_weight),
        loss_mask=original_masks.reshape(-1, *original_masks.shape[3:]),
    )
    silhouette_foreground_loss, silhouette_background_loss, silhouette_loss = (
        compute_balanced_silhouette_loss(rendered_alpha, original_masks)
    )
    global_depth_loss, local_depth_loss = compute_global_local_depth_loss(
        rendered_depth=rendered_depth,
        target_depth=depths,
        foreground_mask=original_masks,
        patch_size=int(cfg_train.local_depth_patch_size),
        min_valid_pixels=int(cfg_train.local_depth_min_valid_pixels),
    )
    depth_loss = global_depth_loss + float(cfg_train.local_depth_weight) * local_depth_loss
    dense_motion_transitions = torch.stack([dense_motion01, dense_motion12], dim=1)
    dense_motion_mask = dense_valid_mask[:, None].expand(-1, 2, -1)

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rgb_loss": rec_loss,
        "silhouette_foreground_loss": silhouette_foreground_loss,
        "silhouette_background_loss": silhouette_background_loss,
        "silhouette_loss": silhouette_loss,
        "global_depth_loss": global_depth_loss,
        "local_depth_loss": local_depth_loss,
        "depth_loss": depth_loss,
        "control_motion01_mean": _masked_motion_magnitude_mean(control_delta01, control_valid_mask),
        "control_motion12_mean": _masked_motion_magnitude_mean(control_delta12, control_valid_mask),
        "dense_motion_mean": _masked_motion_magnitude_mean(dense_motion_transitions, dense_motion_mask),
        "mean_valid_gaussian_opacity": _mean_valid_opacity(pc0),
    }

    if return_renders:
        out_dict["target_images_self"] = target_images
        out_dict["target_masks_self"] = original_masks
        out_dict["target_depths_self"] = None if depths is None else depths.detach()
        out_dict["rendered_self"] = rendered
        out_dict["rendered_expected_depth_self"] = rendered_depth
        out_dict["rendered_alpha_self"] = rendered_alpha
        out_dict["source_indices"] = source_indices.detach().cpu()
        out_dict["gaussian_pc"] = {k: v.detach() for k, v in pc0.items() if torch.is_tensor(v)}
        out_dict["gaussian_pc_sequence"] = [
            {k: v.detach() for k, v in pc.items() if torch.is_tensor(v)} for pc in pc_sequence
        ]
        out_dict["control_xyz_sequence"] = control_xyz_sequence.detach()
        out_dict["control_valid_mask"] = control_valid_mask.detach()
        out_dict["control_indices"] = control_indices.detach()
        out_dict["control_delta01"] = control_delta01.detach()
        out_dict["control_delta12"] = control_delta12.detach()
        out_dict["source_c2w"] = source_c2w.detach()
        out_dict["source_intrinsics"] = source_intrinsics.detach()
        out_dict["base_map"] = decoded["base_map"].detach()
    return out_dict
