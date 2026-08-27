from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .camera import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix
from .coverage import source_coverage_from_xlens


@dataclass(frozen=True)
class LagerTargetPoseConfig:
    alpha_min: float = 0.15
    alpha_max: float = 0.85
    prefer_scene_centered_arc: bool = True
    translation_max_baseline_fraction: float = 0.03
    rotation_max_degrees: float = 3.0
    min_source_coverage: float = 0.60
    max_resample_attempts: int = 4
    minimum_geometry_distance_m: float = 0.08
    minimum_optical_axis_cosine: float = 0.35
    coverage_confidence_threshold: float = 0.0
    coverage_sample_stride: int = 2
    coverage_dilation_kernel: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_min <= self.alpha_max <= 1.0:
            raise ValueError("LagerNVS alpha bounds must lie inside [0,1].")
        if not 0.0 <= self.translation_max_baseline_fraction <= 0.05:
            raise ValueError("LagerNVS translation jitter may not exceed 0.05 baseline.")
        if not 0.0 <= self.rotation_max_degrees <= 5.0:
            raise ValueError("LagerNVS rotation jitter may not exceed five degrees.")
        if not 0.0 <= self.min_source_coverage <= 1.0:
            raise ValueError("Source coverage threshold must lie in [0,1].")
        if self.max_resample_attempts <= 0:
            raise ValueError("Target-pose resampling requires at least one attempt.")


def _slerp_quaternion(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    q0 = F.normalize(q0, dim=-1, eps=1.0e-8)
    q1 = F.normalize(q1, dim=-1, eps=1.0e-8)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = F.normalize((1.0 - alpha) * q0 + alpha * q1, dim=-1, eps=1.0e-8)
    spherical = (
        torch.sin((1.0 - alpha) * theta) / sin_theta.clamp_min(1.0e-8) * q0
        + torch.sin(alpha * theta) / sin_theta.clamp_min(1.0e-8) * q1
    )
    return torch.where(sin_theta.abs() < 1.0e-5, linear, spherical)


def _slerp_direction(v0: torch.Tensor, v1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    n0 = F.normalize(v0, dim=-1, eps=1.0e-8)
    n1 = F.normalize(v1, dim=-1, eps=1.0e-8)
    dot = (n0 * n1).sum().clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if float(sin_theta.abs()) < 1.0e-5:
        return F.normalize((1.0 - alpha) * n0 + alpha * n1, dim=-1, eps=1.0e-8)
    return (
        torch.sin((1.0 - alpha) * theta) / sin_theta * n0
        + torch.sin(alpha * theta) / sin_theta * n1
    )


def _base_pose(
    source_c2w: torch.Tensor,
    scene_center: torch.Tensor,
    alpha: torch.Tensor,
    prefer_arc: bool,
) -> tuple[torch.Tensor, bool]:
    center_a, center_b = source_c2w[:, :3, 3]
    vector_a = center_a - scene_center
    vector_b = center_b - scene_center
    radii = torch.stack((vector_a.norm(), vector_b.norm()))
    arc_stable = bool(
        prefer_arc
        and torch.isfinite(radii).all()
        and float(radii.amin()) > 1.0e-4
        and float(
            F.cosine_similarity(vector_a[None], vector_b[None]).abs()
        )
        < 0.9999
    )
    if arc_stable:
        direction = _slerp_direction(vector_a, vector_b, alpha)
        radius = (1.0 - alpha) * radii[0] + alpha * radii[1]
        center = scene_center + radius * direction
    else:
        center = (1.0 - alpha) * center_a + alpha * center_b
    q_a = matrix_to_quaternion_xyzw(source_c2w[0, :3, :3])
    q_b = matrix_to_quaternion_xyzw(source_c2w[1, :3, :3])
    rotation = quaternion_xyzw_to_matrix(
        _slerp_quaternion(q_a, q_b, alpha)
    )
    pose = torch.eye(4, device=source_c2w.device, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    return pose, arc_stable


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, dim=-1, eps=1.0e-8)
    x, y, z = axis
    zero = axis.new_zeros(())
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero)
    ).reshape(3, 3)
    identity = torch.eye(3, device=axis.device)
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def _perturb_pose(
    base: torch.Tensor,
    baseline: torch.Tensor,
    config: LagerTargetPoseConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = base.device
    direction_angle = 2.0 * math.pi * torch.rand((), device=device, generator=generator)
    magnitude = (
        torch.sqrt(torch.rand((), device=device, generator=generator))
        * float(config.translation_max_baseline_fraction)
        * baseline
    )
    local_translation = torch.stack(
        (magnitude * torch.cos(direction_angle), magnitude * torch.sin(direction_angle), magnitude * 0)
    )
    world_translation = base[:3, :3] @ local_translation
    rotation_magnitude = (
        torch.rand((), device=device, generator=generator)
        * math.radians(float(config.rotation_max_degrees))
    )
    rotation_direction = 2.0 * math.pi * torch.rand(
        (), device=device, generator=generator
    )
    # Local x/y axes vary pitch and yaw without introducing roll.
    local_axis = torch.stack(
        (torch.cos(rotation_direction), torch.sin(rotation_direction), rotation_direction * 0)
    )
    candidate = base.clone()
    candidate[:3, :3] = base[:3, :3] @ _axis_angle_matrix(
        local_axis, rotation_magnitude
    )
    candidate[:3, 3] += world_translation
    return candidate, world_translation, torch.rad2deg(rotation_magnitude)


def _candidate_checks(
    candidate: torch.Tensor,
    scene_center: torch.Tensor,
    coverage: dict[str, torch.Tensor],
    config: LagerTargetPoseConfig,
) -> tuple[bool, str]:
    view_to_scene = F.normalize(scene_center - candidate[:3, 3], dim=-1, eps=1.0e-8)
    optical_axis = F.normalize(candidate[:3, 2], dim=-1, eps=1.0e-8)
    if float((view_to_scene * optical_axis).sum()) < float(
        config.minimum_optical_axis_cosine
    ):
        return False, "points_away_from_workspace"
    if float(coverage["minimum_geometry_distance"]) < float(
        config.minimum_geometry_distance_m
    ):
        return False, "too_close_to_source_geometry"
    if float(coverage["coverage_fraction"]) < float(config.min_source_coverage):
        return False, "insufficient_source_coverage"
    return True, "accepted"


@torch.inference_mode()
def sample_safe_target_poses(
    source_c2w: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    depth: torch.Tensor,
    confidence: torch.Tensor,
    validity: torch.Tensor,
    scene_center: torch.Tensor,
    config: LagerTargetPoseConfig,
    *,
    seed: int,
    stage_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    """Interpolate, perturb, validate, and deterministically fall back per sample."""

    if source_c2w.dim() != 4 or source_c2w.shape[1:] != (2, 4, 4):
        raise ValueError("Target sampling expects (B,2,4,4) source c2w poses.")
    batch = source_c2w.shape[0]
    if scene_center.shape == (3,):
        scene_center = scene_center[None].expand(batch, -1)
    if scene_center.shape != (batch, 3):
        raise ValueError("Scene center must be (3,) or (B,3).")
    run_stage = stage_runner or (lambda _name, callable_: callable_())
    generator = torch.Generator(device=source_c2w.device)
    generator.manual_seed(int(seed))
    final_poses = []
    base_poses = []
    support_masks = []
    alphas = []
    baselines = []
    translation_vectors = []
    translation_magnitudes = []
    rotation_magnitudes = []
    coverages = []
    rejected_counts = []
    fallback_flags = []
    arc_flags = []
    distances_a = []
    distances_b = []
    final_reasons: list[str] = []
    rejected_reasons: list[str] = []

    for index in range(batch):
        cameras = source_c2w[index].float()
        baseline = (cameras[1, :3, 3] - cameras[0, :3, 3]).norm()
        accepted = False
        rejected = 0
        chosen = None
        chosen_base = None
        chosen_coverage = None
        chosen_alpha = None
        chosen_translation = cameras.new_zeros(3)
        chosen_rotation = cameras.new_zeros(())
        chosen_arc = False
        reason = "uninitialized"
        for _attempt in range(int(config.max_resample_attempts)):
            alpha = cameras.new_tensor(float(config.alpha_min)) + torch.rand(
                (), device=cameras.device, generator=generator
            ) * float(config.alpha_max - config.alpha_min)
            base_pose, used_arc = run_stage(
                "target_pose_interpolation",
                lambda cameras=cameras, index=index, alpha=alpha: _base_pose(
                    cameras,
                    scene_center[index],
                    alpha,
                    config.prefer_scene_centered_arc,
                ),
            )
            candidate, translation, rotation_deg = run_stage(
                "target_pose_bounded_perturbation",
                lambda base_pose=base_pose, baseline=baseline: _perturb_pose(
                    base_pose, baseline, config, generator
                ),
            )
            coverage = run_stage(
                "target_pose_coverage_validation",
                lambda index=index, cameras=cameras, candidate=candidate: source_coverage_from_xlens(
                    depth[index],
                    confidence[index],
                    validity[index],
                    source_K[index],
                    cameras,
                    target_K[index],
                    candidate,
                    confidence_threshold=config.coverage_confidence_threshold,
                    sample_stride=config.coverage_sample_stride,
                    dilation_kernel=config.coverage_dilation_kernel,
                ),
            )
            accepted, reason = run_stage(
                "target_pose_safety_checks",
                lambda candidate=candidate, index=index, coverage=coverage: _candidate_checks(
                    candidate, scene_center[index], coverage, config
                ),
            )
            if accepted:
                chosen = candidate
                chosen_base = base_pose
                chosen_coverage = coverage
                chosen_alpha = alpha
                chosen_translation = translation
                chosen_rotation = rotation_deg
                chosen_arc = used_arc
                break
            rejected += 1
            rejected_reasons.append(reason)

        fallback = not accepted
        if fallback:
            alpha = cameras.new_tensor(0.5).clamp(config.alpha_min, config.alpha_max)
            base_pose, used_arc = run_stage(
                "target_pose_interpolation",
                lambda cameras=cameras, index=index, alpha=alpha: _base_pose(
                    cameras,
                    scene_center[index],
                    alpha,
                    config.prefer_scene_centered_arc,
                ),
            )
            coverage = run_stage(
                "target_pose_coverage_validation",
                lambda index=index, cameras=cameras, base_pose=base_pose: source_coverage_from_xlens(
                    depth[index],
                    confidence[index],
                    validity[index],
                    source_K[index],
                    cameras,
                    target_K[index],
                    base_pose,
                    confidence_threshold=config.coverage_confidence_threshold,
                    sample_stride=config.coverage_sample_stride,
                    dilation_kernel=config.coverage_dilation_kernel,
                ),
            )
            chosen = chosen_base = base_pose
            chosen_coverage = coverage
            chosen_alpha = alpha
            chosen_arc = used_arc
            reason = "conservative_midpoint_fallback"

        assert chosen is not None and chosen_base is not None
        assert chosen_coverage is not None and chosen_alpha is not None
        final_poses.append(chosen)
        base_poses.append(chosen_base)
        support_masks.append(chosen_coverage["support_mask"])
        alphas.append(chosen_alpha)
        baselines.append(baseline)
        translation_vectors.append(chosen_translation)
        translation_magnitudes.append(chosen_translation.norm())
        rotation_magnitudes.append(chosen_rotation)
        coverages.append(chosen_coverage["coverage_fraction"])
        rejected_counts.append(rejected)
        fallback_flags.append(fallback)
        arc_flags.append(chosen_arc)
        distances_a.append((chosen[:3, 3] - cameras[0, :3, 3]).norm())
        distances_b.append((chosen[:3, 3] - cameras[1, :3, 3]).norm())
        final_reasons.append(reason)

    target_c2w = torch.stack(final_poses)
    return {
        "target_c2w": target_c2w,
        "target_w2c": torch.linalg.inv(target_c2w),
        "base_c2w": torch.stack(base_poses),
        "support_mask": torch.stack(support_masks),
        "alpha": torch.stack(alphas),
        "baseline": torch.stack(baselines),
        "translation_perturbation": torch.stack(translation_vectors),
        "translation_perturbation_magnitude": torch.stack(translation_magnitudes),
        "rotation_perturbation_degrees": torch.stack(rotation_magnitudes),
        "source_coverage": torch.stack(coverages),
        "rejected_candidates": torch.tensor(
            rejected_counts, device=source_c2w.device, dtype=torch.long
        ),
        "fallback_used": torch.tensor(
            fallback_flags, device=source_c2w.device, dtype=torch.bool
        ),
        "scene_centered_arc_used": torch.tensor(
            arc_flags, device=source_c2w.device, dtype=torch.bool
        ),
        "distance_from_camera_a": torch.stack(distances_a),
        "distance_from_camera_b": torch.stack(distances_b),
        "final_reason": final_reasons,
        "rejected_reasons": rejected_reasons,
    }
