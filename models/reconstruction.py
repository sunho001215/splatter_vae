from __future__ import annotations

from typing import Any, Dict, Optional

import random
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
        "inv_mask_by_view": latents["inv_mask"][:, 0].view(bsz, num_views, -1).contiguous(),
        "dep_mask_by_view": latents["dep_mask"][:, 0].view(bsz, num_views, -1).contiguous(),
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


def _pairwise_squared_distance(
    points: torch.Tensor,
    controls: torch.Tensor,
) -> torch.Tensor:
    """Return batched squared distances without a full-distance square root."""
    point_norm = points.square().sum(dim=-1, keepdim=True)
    control_norm = controls.square().sum(dim=-1).unsqueeze(-2)
    distances = point_norm + control_norm
    distances = distances - 2.0 * torch.matmul(points, controls.transpose(-1, -2))
    return distances.clamp_min(0.0)


def sample_sparse_motion_controls(
    dense_xyz: torch.Tensor,
    dense_valid_mask: torch.Tensor,
    num_controls: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched FPS over valid foreground centers with graph-connected gathers."""
    max_controls = int(num_controls)
    if max_controls <= 0:
        raise ValueError(f"num_controls must be positive, got {num_controls}.")
    if dense_xyz.dim() != 3 or dense_xyz.shape[-1] != 3:
        raise ValueError(f"Expected dense_xyz as (B,N,3), got {tuple(dense_xyz.shape)}.")
    batch, num_points = dense_xyz.shape[:2]
    if num_points < 1:
        raise ValueError("FPS requires at least one dense Gaussian slot.")

    valid = dense_valid_mask.to(dtype=torch.bool) & torch.isfinite(dense_xyz).all(dim=-1)
    valid_counts = valid.sum(dim=-1)
    detached = torch.where(valid.unsqueeze(-1), dense_xyz.detach().float(), 0.0)
    centroid = detached.sum(dim=1) / valid_counts.clamp_min(1).unsqueeze(-1)
    first_distance = (detached - centroid[:, None]).square().sum(dim=-1)
    current = first_distance.masked_fill(~valid, -torch.inf).argmax(dim=-1)

    fps_indices = torch.empty(
        (batch, max_controls), device=dense_xyz.device, dtype=torch.long
    )
    min_squared_distance = torch.full(
        (batch, num_points), torch.inf, device=dense_xyz.device, dtype=detached.dtype
    )
    batch_indices = torch.arange(batch, device=dense_xyz.device)
    for slot in range(max_controls):
        fps_indices[:, slot] = current
        selected_xyz = detached[batch_indices, current]
        squared_distance = (detached - selected_xyz[:, None]).square().sum(dim=-1)
        min_squared_distance = torch.minimum(min_squared_distance, squared_distance)
        current = min_squared_distance.masked_fill(~valid, -torch.inf).argmax(dim=-1)

    # Preserve the former policy of using every valid Gaussian in index order
    # when a sample has no more foreground points than the control budget.
    natural_width = min(max_controls, num_points)
    dense_indices = torch.arange(num_points, device=dense_xyz.device).expand(batch, -1)
    natural_indices = torch.topk(
        dense_indices.masked_fill(~valid, num_points),
        k=natural_width,
        dim=-1,
        largest=False,
        sorted=True,
    ).values
    if natural_width < max_controls:
        padding = torch.full(
            (batch, max_controls - natural_width),
            num_points,
            device=dense_xyz.device,
            dtype=torch.long,
        )
        natural_indices = torch.cat((natural_indices, padding), dim=-1)
    natural_indices = natural_indices.masked_fill(natural_indices == num_points, -1)

    use_fps = (valid_counts > max_controls).unsqueeze(-1)
    selected_indices = torch.where(use_fps, fps_indices, natural_indices)
    control_valid_mask = (
        torch.arange(max_controls, device=dense_xyz.device).unsqueeze(0)
        < valid_counts.clamp_max(max_controls).unsqueeze(-1)
    )
    control_indices = selected_indices.masked_fill(~control_valid_mask, -1)
    safe_indices = control_indices.clamp_min(0)
    control_xyz = dense_xyz[batch_indices[:, None], safe_indices]
    control_xyz = control_xyz * control_valid_mask.unsqueeze(-1).to(dtype=dense_xyz.dtype)
    return control_xyz.contiguous(), control_indices, control_valid_mask


def gather_control_gaussian_attributes(
    pc: Dict[str, torch.Tensor],
    control_indices: torch.Tensor,
    control_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather camera-independent Gaussian attributes at persistent FPS indices."""
    safe_indices = control_indices.clamp_min(0)
    batch_indices = torch.arange(control_indices.shape[0], device=control_indices.device)[:, None]
    scaling = pc["scaling"][batch_indices, safe_indices]
    opacity = pc["opacity"][batch_indices, safe_indices]
    features_dc = pc["features_dc"][batch_indices, safe_indices]
    if features_dc.dim() == 4 and features_dc.shape[-2] == 1:
        features_dc = features_dc.squeeze(-2)
    if features_dc.shape != scaling.shape:
        raise ValueError(
            f"Expected gathered DC features to match scaling shape {tuple(scaling.shape)}, "
            f"got {tuple(features_dc.shape)}."
        )
    mask = control_valid_mask.unsqueeze(-1).to(dtype=scaling.dtype)
    return (
        (scaling * mask).contiguous(),
        (opacity * mask).contiguous(),
        (features_dc * mask).contiguous(),
    )


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
    squared_distances = _pairwise_squared_distance(dense_xyz, control_xyz)
    squared_distances = squared_distances.masked_fill(
        ~control_valid_mask[:, None, :], torch.inf
    )
    nearest_squared, nearest_indices = torch.topk(
        squared_distances,
        k=neighbors,
        dim=-1,
        largest=False,
        sorted=False,
    )
    nearest_distances = nearest_squared.clamp_min(1.0e-12).sqrt()
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


def _two_transition_motion_magnitude_mean(
    motion01: torch.Tensor,
    motion12: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weights = valid_mask.to(dtype=motion01.dtype)
    numerator = ((motion01.norm(dim=-1) + motion12.norm(dim=-1)) * weights).sum()
    return numerator / (2.0 * weights.sum()).clamp_min(1.0)


def _render_sequence(
    pc_sequence: list[Dict[str, torch.Tensor]],
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    bg: torch.Tensor,
    splatter_cfg: SplatterConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the full ``(batch,time,camera)`` grid in one gsplat call."""
    timesteps = w2c.shape[1]
    if len(pc_sequence) != timesteps:
        raise ValueError(
            f"Expected one Gaussian set per timestep ({timesteps}), got {len(pc_sequence)}."
        )
    renderer_keys = (
        "xyz",
        "scaling",
        "rotation",
        "opacity",
        "features_dc",
        "features_rest",
        "valid_mask",
    )
    stacked_pc = {
        key: torch.stack([pc[key] for pc in pc_sequence], dim=1).contiguous()
        for key in renderer_keys
        if key in pc_sequence[0]
    }
    out = render_predicted(
        pc=stacked_pc,
        world_view_transform=w2c,
        intrinsics=intrinsics,
        bg_color=bg,
        cfg=splatter_cfg,
        packed=False,
        render_mode="RGB+ED",
    )
    if out["render"] is None or out["depth"] is None:
        raise RuntimeError("RGB+ED rendering must return both RGB and expected depth.")
    return out["render"], out["depth"], out["alpha"]


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
    ).to(dtype=pc["opacity"].dtype)
    opacity = pc["opacity"].squeeze(-1)
    return (opacity * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def select_local_depth_patch_size(
    cfg_train: TrainConfig,
    training: bool,
) -> int:
    """Choose one shared patch size for the complete reconstruction batch."""
    minimum = int(cfg_train.local_depth_min_patch_size)
    maximum = int(cfg_train.local_depth_max_patch_size)
    if minimum <= 0 or maximum < minimum:
        raise ValueError(
            "Expected 0 < local_depth_min_patch_size <= local_depth_max_patch_size, "
            f"got {minimum} and {maximum}."
        )
    if training:
        return random.randint(minimum, maximum)
    return (minimum + maximum + 1) // 2


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
    s_inv_source: torch.Tensor,
    z_dep_source: torch.Tensor,
    source_indices: torch.Tensor,
    depths: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    return_renders: bool = False,
    compute_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Decode reused source-view features and render all cameras/timesteps."""
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
    source_indices = source_indices.to(device=device, dtype=torch.long)
    if source_indices.shape != (bsz,):
        raise ValueError(f"Expected source_indices shape {(bsz,)}, got {tuple(source_indices.shape)}.")
    if s_inv_source.shape[0] != bsz or z_dep_source.shape[0] != bsz:
        raise ValueError("Source feature batch dimensions must match images_01.")

    decoded = vae.decode_sequence(s_inv=s_inv_source, z_dep_source=z_dep_source)
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
    control_scaling, control_opacity, control_features_dc = gather_control_gaussian_attributes(
        pc=pc0,
        control_indices=control_indices,
        control_valid_mask=control_valid_mask,
    )
    control_delta01, control_delta12 = vae.predict_control_motion(
        s_inv=s_inv_source,
        normalized_control_xyz=normalized_controls,
        control_xyz_world=control_xyz0,
        control_scaling=control_scaling,
        control_opacity=control_opacity,
        control_features_dc=control_features_dc,
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
    local_depth_patch_size = select_local_depth_patch_size(
        cfg_train=cfg_train,
        training=vae.training,
    )
    global_depth_loss, local_depth_loss = compute_global_local_depth_loss(
        rendered_depth=rendered_depth,
        target_depth=depths,
        foreground_mask=original_masks,
        patch_size=local_depth_patch_size,
        min_valid_pixels=int(cfg_train.local_depth_min_valid_pixels),
    )
    depth_loss = global_depth_loss + float(cfg_train.local_depth_weight) * local_depth_loss

    out_dict: Dict[str, Any] = {
        "rec_loss": rec_loss,
        "rgb_loss": rec_loss,
        "silhouette_foreground_loss": silhouette_foreground_loss,
        "silhouette_background_loss": silhouette_background_loss,
        "silhouette_loss": silhouette_loss,
        "global_depth_loss": global_depth_loss,
        "local_depth_loss": local_depth_loss,
        "depth_loss": depth_loss,
    }

    need_diagnostics = bool(compute_diagnostics or return_renders)
    if need_diagnostics:
        with torch.no_grad():
            out_dict.update(
                {
                    "control_motion01_mean": _masked_motion_magnitude_mean(
                        control_delta01.detach(), control_valid_mask
                    ),
                    "control_motion12_mean": _masked_motion_magnitude_mean(
                        control_delta12.detach(), control_valid_mask
                    ),
                    "dense_motion_mean": _two_transition_motion_magnitude_mean(
                        dense_motion01.detach(), dense_motion12.detach(), dense_valid_mask
                    ),
                    "mean_valid_gaussian_opacity": _mean_valid_opacity(pc0).detach(),
                }
            )

    if return_renders:
        with torch.no_grad():
            control_xyz1 = control_xyz0.detach() + control_delta01.detach()
            control_xyz2 = control_xyz1 + control_delta12.detach()
            out_dict["control_xyz_sequence"] = torch.stack(
                [control_xyz0.detach(), control_xyz1, control_xyz2], dim=1
            )
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
        out_dict["control_xyz_sequence"] = out_dict["control_xyz_sequence"].detach()
        out_dict["control_valid_mask"] = control_valid_mask.detach()
        out_dict["control_indices"] = control_indices.detach()
        out_dict["control_delta01"] = control_delta01.detach()
        out_dict["control_delta12"] = control_delta12.detach()
        out_dict["source_c2w"] = source_c2w.detach()
        out_dict["source_intrinsics"] = source_intrinsics.detach()
        out_dict["base_map"] = decoded["base_map"].detach()
    return out_dict
