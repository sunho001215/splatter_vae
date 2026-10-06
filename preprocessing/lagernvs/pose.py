from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .camera import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix
from .coverage import project_world_support, unproject_metric_depth

ALPHA_BANDS = ((0.15, 0.25), (0.25, 0.35))
TARGETS_PER_TIMESTEP = 4
POSE_SAMPLER_CONTRACT_SCHEMA_VERSION = 1
POSE_SAMPLER_ALGORITHM_REVISION = 5


@dataclass(frozen=True)
class LagerTargetPoseConfig:
    """Bounded, geometry-checked four-view target distribution."""

    alpha_bands: tuple[tuple[float, float], tuple[float, float]] = ALPHA_BANDS
    prefer_scene_centered_arc: bool = True
    translation_max_baseline_fraction: float = 0.03
    exceptional_translation_max_baseline_fraction: float = 0.05
    rotation_max_degrees: float = 3.0
    min_source_coverage: float = 0.60
    exceptional_min_source_coverage: float = 0.40
    max_resample_attempts: int = 4
    alpha_safety_grid_candidates: int = 17
    minimum_geometry_distance_m: float = 0.08
    exceptional_minimum_geometry_distance_m: float = 0.05
    source_clearance_reference_fraction: float = 0.80
    minimum_source_calibrated_clearance_m: float = 0.02
    geometry_clearance_quantile: float = 0.01
    minimum_optical_axis_cosine: float = 0.35
    coverage_sample_stride: int = 2
    coverage_dilation_kernel: int = 7

    def __post_init__(self) -> None:
        if len(self.alpha_bands) != 2:
            raise ValueError("LagerNVS requires exactly two Cam-A alpha bands.")
        previous = 0.0
        for lower, upper in self.alpha_bands:
            if not 0.0 < float(lower) < float(upper) < 0.5:
                raise ValueError(
                    "LagerNVS Cam-A alpha bands must lie strictly below 0.5."
                )
            if float(lower) < previous:
                raise ValueError(
                    "LagerNVS alpha bands must be ordered and non-overlapping."
                )
            previous = float(upper)
        if not 0.0 <= self.translation_max_baseline_fraction <= 0.05:
            raise ValueError(
                "LagerNVS translation jitter may not exceed 0.05 baseline."
            )
        if not (
            self.translation_max_baseline_fraction
            <= self.exceptional_translation_max_baseline_fraction
            <= 0.05
        ):
            raise ValueError(
                "Exceptional LagerNVS safety translation must be between the "
                "ordinary limit and 0.05 baseline."
            )
        if not 0.0 <= self.rotation_max_degrees <= 5.0:
            raise ValueError("LagerNVS rotation jitter may not exceed five degrees.")
        if not 0.0 <= self.min_source_coverage <= 1.0:
            raise ValueError("Source coverage threshold must lie in [0,1].")
        if not 0.0 <= self.exceptional_min_source_coverage <= self.min_source_coverage:
            raise ValueError(
                "Exceptional source coverage must not exceed the ordinary threshold."
            )
        if not (
            0.0
            <= self.exceptional_minimum_geometry_distance_m
            <= self.minimum_geometry_distance_m
        ):
            raise ValueError(
                "Exceptional geometry clearance must not exceed the ordinary threshold."
            )
        if not 0.0 < self.source_clearance_reference_fraction <= 1.0:
            raise ValueError("Source-clearance reference fraction must lie in (0,1].")
        if not (
            0.0
            <= self.minimum_source_calibrated_clearance_m
            <= self.exceptional_minimum_geometry_distance_m
        ):
            raise ValueError(
                "Source-calibrated clearance must be nonnegative and no larger than "
                "the exceptional nominal clearance."
            )
        if self.max_resample_attempts <= 0:
            raise ValueError("Target-pose resampling requires at least one attempt.")
        if self.alpha_safety_grid_candidates < 2:
            raise ValueError("The alpha safety grid must include both band endpoints.")
        if not 0.0 < self.geometry_clearance_quantile < 0.05:
            raise ValueError("Geometry clearance quantile must lie in (0,0.05).")
        if self.coverage_sample_stride <= 0:
            raise ValueError("Coverage sample stride must be positive.")
        if self.coverage_dilation_kernel <= 0 or self.coverage_dilation_kernel % 2 == 0:
            raise ValueError("Coverage dilation must be a positive odd integer.")


def pose_sampler_contract(
    config: LagerTargetPoseConfig | None = None,
) -> dict[str, Any]:
    """Return a stable, signed description of the active pose-safety policy."""

    resolved = config if config is not None else LagerTargetPoseConfig()
    payload = {
        "schema_version": POSE_SAMPLER_CONTRACT_SCHEMA_VERSION,
        "algorithm": "droid_lagernvs_bounded_pose_sampler",
        "algorithm_revision": POSE_SAMPLER_ALGORITHM_REVISION,
        "configuration": asdict(resolved),
        "tiers": {
            "ordinary": {
                "maximum_translation_baseline_fraction": resolved.translation_max_baseline_fraction,
                "maximum_rotation_degrees": resolved.rotation_max_degrees,
                "minimum_source_coverage": resolved.min_source_coverage,
                "minimum_geometry_distance_m": resolved.minimum_geometry_distance_m,
            },
            "exceptional_translation": {
                "maximum_translation_baseline_fraction": resolved.exceptional_translation_max_baseline_fraction,
                "maximum_rotation_degrees": resolved.rotation_max_degrees,
                "minimum_source_coverage": resolved.min_source_coverage,
                "minimum_geometry_distance_m": resolved.minimum_geometry_distance_m,
            },
            "exceptional_safety_thresholds": {
                "maximum_translation_baseline_fraction": resolved.exceptional_translation_max_baseline_fraction,
                "maximum_rotation_degrees": resolved.rotation_max_degrees,
                "minimum_source_coverage": resolved.exceptional_min_source_coverage,
                "nominal_minimum_geometry_distance_m": resolved.exceptional_minimum_geometry_distance_m,
                "source_clearance_reference_fraction": resolved.source_clearance_reference_fraction,
                "hard_minimum_geometry_distance_m": resolved.minimum_source_calibrated_clearance_m,
            },
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "signature": hashlib.sha256(encoded).hexdigest()}


def sample_stratified_alphas(
    batch: int,
    *,
    seed: int,
    device: torch.device | str,
    bands: tuple[tuple[float, float], tuple[float, float]] = ALPHA_BANDS,
) -> torch.Tensor:
    """Return deterministic [a1,a2,1-a2,1-a1] values for every sample."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    left = []
    for lower, upper in bands:
        draw = torch.rand(int(batch), device=device, generator=generator)
        left.append(float(lower) + draw * (float(upper) - float(lower)))
    a1, a2 = left
    return torch.stack((a1, a2, 1.0 - a2, 1.0 - a1), dim=1)


def _slerp_quaternion(
    q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
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


def _slerp_direction(
    v0: torch.Tensor, v1: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
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
        and float(F.cosine_similarity(vector_a[None], vector_b[None]).abs()) < 0.9999
    )
    if arc_stable:
        direction = _slerp_direction(vector_a, vector_b, alpha)
        radius = (1.0 - alpha) * radii[0] + alpha * radii[1]
        center = scene_center + radius * direction
    else:
        center = (1.0 - alpha) * center_a + alpha * center_b
    q_a = matrix_to_quaternion_xyzw(source_c2w[0, :3, :3])
    q_b = matrix_to_quaternion_xyzw(source_c2w[1, :3, :3])
    rotation = quaternion_xyzw_to_matrix(_slerp_quaternion(q_a, q_b, alpha))
    pose = torch.eye(4, device=source_c2w.device, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    return pose, arc_stable


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, dim=-1, eps=1.0e-8)
    x, y, z = axis
    zero = axis.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, device=axis.device)
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def _perturb_pose(
    base: torch.Tensor,
    baseline: torch.Tensor,
    config: LagerTargetPoseConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction_angle = (
        2.0 * math.pi * torch.rand((), device=base.device, generator=generator)
    )
    magnitude = (
        torch.sqrt(torch.rand((), device=base.device, generator=generator))
        * float(config.translation_max_baseline_fraction)
        * baseline
    )
    local_translation = torch.stack(
        (
            magnitude * torch.cos(direction_angle),
            magnitude * torch.sin(direction_angle),
            magnitude * 0,
        )
    )
    world_translation = base[:3, :3] @ local_translation
    rotation_magnitude = torch.rand(
        (), device=base.device, generator=generator
    ) * math.radians(float(config.rotation_max_degrees))
    rotation_direction = (
        2.0 * math.pi * torch.rand((), device=base.device, generator=generator)
    )
    local_axis = torch.stack(
        (
            torch.cos(rotation_direction),
            torch.sin(rotation_direction),
            rotation_direction * 0,
        )
    )
    candidate = base.clone()
    candidate[:3, :3] = base[:3, :3] @ _axis_angle_matrix(
        local_axis, rotation_magnitude
    )
    candidate[:3, 3] += world_translation
    return candidate, world_translation, torch.rad2deg(rotation_magnitude)


def _deterministic_safety_perturbations(
    base: torch.Tensor,
    baseline: torch.Tensor,
    config: LagerTargetPoseConfig,
    rejection_reason: str,
    *,
    translation_max_baseline_fraction: float | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return bounded fallback poses that cover directions random retries can miss.

    The ordinary target distribution remains stochastic.  This small deterministic
    lattice is used only after those retries and the unperturbed pose fail a safety
    check.  Including optical-axis translation is important for real DROID camera
    pairs whose interpolation arc passes close to reconstructed geometry; the
    ordinary perturbation intentionally samples only the target image plane.
    """

    candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    zero_translation = base.new_zeros(3)
    zero_rotation = base.new_zeros(())
    rotation_limit = math.radians(float(config.rotation_max_degrees))
    if rotation_limit > 0.0 and rejection_reason != "too_close_to_source_geometry":
        for axis in (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ):
            local_axis = base.new_tensor(axis)
            for sign in (-1.0, 1.0):
                candidate = base.clone()
                candidate[:3, :3] = base[:3, :3] @ _axis_angle_matrix(
                    local_axis, base.new_tensor(sign * rotation_limit)
                )
                candidates.append(
                    (
                        candidate,
                        zero_translation,
                        base.new_tensor(float(config.rotation_max_degrees)),
                    )
                )

    translation_fraction = (
        float(config.translation_max_baseline_fraction)
        if translation_max_baseline_fraction is None
        else float(translation_max_baseline_fraction)
    )
    if not 0.0 <= translation_fraction <= 0.05:
        raise ValueError("A safety perturbation may not exceed 0.05 baseline.")
    translation_limit = translation_fraction * baseline
    if float(translation_limit) <= 0.0:
        return candidates

    # Try moving radially away from/toward the workspace first, then cover the
    # target image plane and two tilted rings.  Every direction is unit length,
    # so the configured baseline-relative limit is exact.
    local_directions = [base.new_tensor((0.0, 0.0, -1.0))]
    local_directions.append(base.new_tensor((0.0, 0.0, 1.0)))
    for z_component in (0.0, -1.0, 1.0):
        for index in range(16):
            angle = 2.0 * math.pi * float(index) / 16.0
            direction = base.new_tensor((math.cos(angle), math.sin(angle), z_component))
            local_directions.append(F.normalize(direction, dim=0))
    for local_direction in local_directions:
        world_translation = base[:3, :3] @ local_direction * translation_limit
        candidate = base.clone()
        candidate[:3, 3] += world_translation
        candidates.append((candidate, world_translation, zero_rotation))
    return candidates


def _source_calibrated_clearance_threshold(
    source_clearances: torch.Tensor,
    alpha: torch.Tensor,
    config: LagerTargetPoseConfig,
) -> torch.Tensor:
    """Return a safety floor calibrated by two known-physical source cameras."""

    reference = (1.0 - alpha) * source_clearances[0] + alpha * source_clearances[1]
    calibrated = reference * float(config.source_clearance_reference_fraction)
    return calibrated.clamp(
        min=float(config.minimum_source_calibrated_clearance_m),
        max=float(config.exceptional_minimum_geometry_distance_m),
    )


def _coverage_from_points(
    points: torch.Tensor,
    target_K: torch.Tensor,
    candidate: torch.Tensor,
    config: LagerTargetPoseConfig,
) -> dict[str, torch.Tensor]:
    support, zbuffer = project_world_support(
        points,
        candidate,
        target_K,
        dilation_kernel=config.coverage_dilation_kernel,
    )
    center = candidate[:3, 3]
    if points.numel():
        distances = (points - center).norm(dim=-1)
        minimum_distance = distances.amin()
        # A small set of erroneous DA3 points must not classify a camera as
        # colliding with geometry.  The first percentile still requires
        # clearance from a spatial neighborhood (about 288 points at the
        # default sampling rate),
        # while retaining the true nearest distance as an audit diagnostic.
        clearance_distance = torch.quantile(
            distances.float(), float(config.geometry_clearance_quantile)
        )
    else:
        minimum_distance = center.new_tensor(float("inf"))
        clearance_distance = center.new_tensor(float("inf"))
    return {
        "support_mask": support,
        "coverage_fraction": support.float().mean(),
        "target_zbuffer": zbuffer,
        "minimum_geometry_distance": minimum_distance,
        "geometry_clearance_distance": clearance_distance,
    }


def _candidate_checks(
    candidate: torch.Tensor,
    scene_center: torch.Tensor,
    coverage: dict[str, torch.Tensor],
    config: LagerTargetPoseConfig,
    *,
    minimum_source_coverage: float | None = None,
    minimum_geometry_distance_m: float | None = None,
) -> tuple[bool, str]:
    if not torch.isfinite(candidate).all():
        return False, "non_finite_pose"
    view_to_scene = F.normalize(scene_center - candidate[:3, 3], dim=-1, eps=1.0e-8)
    optical_axis = F.normalize(candidate[:3, 2], dim=-1, eps=1.0e-8)
    if float((view_to_scene * optical_axis).sum()) < float(
        config.minimum_optical_axis_cosine
    ):
        return False, "points_away_from_workspace"
    clearance_threshold = (
        float(config.minimum_geometry_distance_m)
        if minimum_geometry_distance_m is None
        else float(minimum_geometry_distance_m)
    )
    coverage_threshold = (
        float(config.min_source_coverage)
        if minimum_source_coverage is None
        else float(minimum_source_coverage)
    )
    if float(coverage["geometry_clearance_distance"]) < clearance_threshold:
        return False, "too_close_to_source_geometry"
    if float(coverage["coverage_fraction"]) < coverage_threshold:
        return False, "insufficient_source_coverage"
    return True, "accepted"


@torch.inference_mode()
def sample_safe_target_poses(
    source_c2w: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    depth: torch.Tensor,
    validity: torch.Tensor,
    scene_center: torch.Tensor,
    config: LagerTargetPoseConfig,
    *,
    seed: int,
    stage_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    """Generate four deterministic, safe targets per synchronized timestamp."""

    if source_c2w.dim() != 4 or source_c2w.shape[1:] != (2, 4, 4):
        raise ValueError("Target sampling expects (B,2,4,4) source c2w poses.")
    batch = int(source_c2w.shape[0])
    if source_K.shape != (batch, 2, 3, 3):
        raise ValueError("Source intrinsics must be (B,2,3,3).")
    if target_K.shape != (batch, 3, 3):
        raise ValueError("Target intrinsics must be (B,3,3).")
    if depth.shape[:3] != (batch, 2, 1) or validity.shape != depth.shape:
        raise ValueError("Depth and validity must be aligned (B,2,1,H,W).")
    if scene_center.shape == (3,):
        scene_center = scene_center[None].expand(batch, -1)
    if scene_center.shape != (batch, 3):
        raise ValueError("Scene center must be (3,) or (B,3).")

    run_stage = stage_runner or (lambda _name, callable_: callable_())
    generator = torch.Generator(device=source_c2w.device)
    generator.manual_seed(int(seed) ^ 0x4C41474552)
    alphas = sample_stratified_alphas(
        batch, seed=int(seed), device=source_c2w.device, bands=config.alpha_bands
    )
    names = (
        "target_c2w",
        "base_c2w",
        "support_mask",
        "baseline",
        "translation_perturbation",
        "translation_perturbation_magnitude",
        "rotation_perturbation_degrees",
        "source_coverage",
        "minimum_geometry_distance",
        "geometry_clearance_distance",
        "rejected_candidates",
        "fallback_used",
        "scene_centered_arc_used",
        "geometry_scene_center_used",
        "safety_translation_limit_escalated",
        "safety_thresholds_relaxed",
        "minimum_source_coverage_threshold",
        "minimum_geometry_distance_threshold",
        "source_clearance_reference",
        "target_scene_center",
        "alpha_resample_count",
        "distance_from_camera_a",
        "distance_from_camera_b",
    )
    field_lists: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    reasons: list[str] = []

    for sample_index in range(batch):
        cameras = source_c2w[sample_index].float()
        baseline = (cameras[1, :3, 3] - cameras[0, :3, 3]).norm()
        points = run_stage(
            "target_pose_depth_unprojection",
            lambda sample_index=sample_index, cameras=cameras: unproject_metric_depth(
                depth[sample_index],
                source_K[sample_index],
                cameras,
                validity[sample_index],
                stride=config.coverage_sample_stride,
            ),
        )
        if not points.numel() or not torch.isfinite(points).all():
            raise RuntimeError("LagerNVS pose sampling requires finite DA3 geometry.")
        geometry_center = points.median(dim=0).values
        source_clearances = torch.stack(
            [
                torch.quantile(
                    (points - camera[:3, 3]).norm(dim=-1).float(),
                    float(config.geometry_clearance_quantile),
                )
                for camera in cameras
            ]
        )
        per_sample: dict[str, list[torch.Tensor]] = {name: [] for name in names}

        def pose_bases(
            alpha_value: torch.Tensor,
            cameras: torch.Tensor = cameras,
            configured_center: torch.Tensor = scene_center[sample_index],
            geometry_center: torch.Tensor = geometry_center,
        ) -> list[tuple[torch.Tensor, bool, torch.Tensor, bool]]:
            configured_base, configured_arc_used = _base_pose(
                cameras,
                configured_center,
                alpha_value,
                config.prefer_scene_centered_arc,
            )
            options = [
                (
                    configured_base,
                    configured_arc_used,
                    configured_center,
                    False,
                )
            ]
            geometry_base, geometry_arc_used = _base_pose(
                cameras,
                geometry_center,
                alpha_value,
                config.prefer_scene_centered_arc,
            )
            if geometry_arc_used:
                options.append((geometry_base, True, geometry_center, True))
            linear_base, _ = _base_pose(cameras, geometry_center, alpha_value, False)
            options.append((linear_base, False, geometry_center, True))
            return options

        def has_safe_bounded_candidate(
            alpha_value: torch.Tensor,
            translation_fraction: float,
            coverage_threshold: float,
            clearance_threshold: float,
            pose_bases=pose_bases,
            points: torch.Tensor = points,
            target_intrinsic: torch.Tensor = target_K[sample_index],
            source_baseline: torch.Tensor = baseline,
        ) -> bool:
            for base, _arc, check_center, _geometry in pose_bases(alpha_value):
                coverage = _coverage_from_points(points, target_intrinsic, base, config)
                accepted, rejection_reason = _candidate_checks(
                    base,
                    check_center,
                    coverage,
                    config,
                    minimum_source_coverage=coverage_threshold,
                    minimum_geometry_distance_m=clearance_threshold,
                )
                if accepted:
                    return True
                for (
                    candidate,
                    _translation,
                    _rotation,
                ) in _deterministic_safety_perturbations(
                    base,
                    source_baseline,
                    config,
                    rejection_reason,
                    translation_max_baseline_fraction=translation_fraction,
                ):
                    candidate_coverage = _coverage_from_points(
                        points, target_intrinsic, candidate, config
                    )
                    if _candidate_checks(
                        candidate,
                        check_center,
                        candidate_coverage,
                        config,
                        minimum_source_coverage=coverage_threshold,
                        minimum_geometry_distance_m=clearance_threshold,
                    )[0]:
                        return True
            return False

        alpha_resamples = torch.zeros(4, device=cameras.device, dtype=torch.int16)
        for left_index, (lower, upper) in enumerate(config.alpha_bands):
            right_index = 3 - left_index
            initial_alpha = alphas[sample_index, left_index]
            candidates = [initial_alpha]
            candidates.extend(
                float(lower)
                + torch.rand((), device=cameras.device, generator=generator)
                * (float(upper) - float(lower))
                for _ in range(int(config.max_resample_attempts))
            )
            grid = torch.linspace(
                float(lower),
                float(upper),
                int(config.alpha_safety_grid_candidates),
                device=cameras.device,
            )
            grid_order = torch.argsort((grid - initial_alpha).abs(), stable=True)
            candidates.extend(grid[index] for index in grid_order)
            accepted_alpha = False
            safety_tiers: list[tuple[float, float, float, bool]] = [
                (
                    float(config.translation_max_baseline_fraction),
                    float(config.min_source_coverage),
                    float(config.minimum_geometry_distance_m),
                    False,
                )
            ]
            if (
                config.exceptional_translation_max_baseline_fraction
                > config.translation_max_baseline_fraction
            ):
                safety_tiers.append(
                    (
                        float(config.exceptional_translation_max_baseline_fraction),
                        float(config.min_source_coverage),
                        float(config.minimum_geometry_distance_m),
                        False,
                    )
                )
            if (
                config.exceptional_min_source_coverage < config.min_source_coverage
                or config.exceptional_minimum_geometry_distance_m
                < config.minimum_geometry_distance_m
            ):
                safety_tiers.append(
                    (
                        float(config.exceptional_translation_max_baseline_fraction),
                        float(config.exceptional_min_source_coverage),
                        float(config.exceptional_minimum_geometry_distance_m),
                        True,
                    )
                )
            for (
                translation_fraction,
                coverage_threshold,
                clearance_threshold,
                source_calibrated,
            ) in safety_tiers:
                for alpha_attempt, candidate_alpha in enumerate(candidates):
                    left_clearance_threshold = (
                        float(
                            _source_calibrated_clearance_threshold(
                                source_clearances,
                                candidate_alpha,
                                config,
                            )
                        )
                        if source_calibrated
                        else clearance_threshold
                    )
                    right_clearance_threshold = (
                        float(
                            _source_calibrated_clearance_threshold(
                                source_clearances,
                                1.0 - candidate_alpha,
                                config,
                            )
                        )
                        if source_calibrated
                        else clearance_threshold
                    )
                    if has_safe_bounded_candidate(
                        candidate_alpha,
                        translation_fraction,
                        coverage_threshold,
                        left_clearance_threshold,
                    ) and has_safe_bounded_candidate(
                        1.0 - candidate_alpha,
                        translation_fraction,
                        coverage_threshold,
                        right_clearance_threshold,
                    ):
                        accepted_alpha = True
                        alphas[sample_index, left_index] = candidate_alpha
                        alphas[sample_index, right_index] = 1.0 - candidate_alpha
                        alpha_resamples[left_index] = alpha_attempt
                        alpha_resamples[right_index] = alpha_attempt
                        break
                if accepted_alpha:
                    break
            if not accepted_alpha:
                raise RuntimeError(
                    "No safe symmetric LagerNVS alpha pair exists in band "
                    f"[{lower}, {upper}] for sample={sample_index}."
                )

        for target_index in range(TARGETS_PER_TIMESTEP):
            alpha = alphas[sample_index, target_index]
            base_options = run_stage(
                "target_pose_interpolation",
                lambda alpha=alpha: pose_bases(alpha),
            )
            configured_base, configured_arc_used, _center, _geometry = base_options[0]
            chosen: torch.Tensor | None = None
            chosen_coverage: dict[str, torch.Tensor] | None = None
            chosen_base = configured_base
            chosen_arc_used = configured_arc_used
            chosen_geometry_center_used = False
            chosen_scene_center = scene_center[sample_index]
            chosen_translation = cameras.new_zeros(3)
            chosen_rotation = cameras.new_zeros(())
            chosen_safety_limit_escalated = False
            chosen_safety_thresholds_relaxed = False
            chosen_coverage_threshold = float(config.min_source_coverage)
            chosen_clearance_threshold = float(config.minimum_geometry_distance_m)
            rejected = 0
            final_reason = "uninitialized"
            fallback = False
            for base, arc_used, check_center, geometry_center_used in base_options:
                for _attempt in range(int(config.max_resample_attempts)):
                    candidate, translation, rotation = run_stage(
                        "target_pose_bounded_perturbation",
                        lambda base=base, baseline=baseline: _perturb_pose(
                            base, baseline, config, generator
                        ),
                    )
                    coverage = run_stage(
                        "target_pose_coverage_validation",
                        lambda points=points, candidate=candidate, sample_index=sample_index: (
                            _coverage_from_points(
                                points, target_K[sample_index], candidate, config
                            )
                        ),
                    )
                    accepted, final_reason = run_stage(
                        "target_pose_safety_checks",
                        lambda candidate=candidate, coverage=coverage, check_center=check_center: (
                            _candidate_checks(candidate, check_center, coverage, config)
                        ),
                    )
                    if accepted:
                        chosen = candidate
                        chosen_coverage = coverage
                        chosen_base = base
                        chosen_arc_used = arc_used
                        chosen_geometry_center_used = geometry_center_used
                        chosen_scene_center = check_center
                        chosen_translation = translation
                        chosen_rotation = rotation
                        break
                    rejected += 1
                if chosen is not None:
                    break
                coverage = _coverage_from_points(
                    points, target_K[sample_index], base, config
                )
                base_coverage = coverage
                accepted, base_rejection_reason = _candidate_checks(
                    base, check_center, coverage, config
                )
                final_reason = base_rejection_reason
                if accepted:
                    chosen = base
                    chosen_coverage = coverage
                    chosen_base = base
                    chosen_arc_used = arc_used
                    chosen_geometry_center_used = geometry_center_used
                    chosen_scene_center = check_center
                    fallback = True
                    final_reason = "accepted_unperturbed_stratified_pose"
                    break
                translation_tiers = (
                    (float(config.translation_max_baseline_fraction), False),
                    (
                        float(config.exceptional_translation_max_baseline_fraction),
                        True,
                    ),
                )
                for translation_fraction, escalated in translation_tiers:
                    if escalated and (
                        translation_fraction
                        <= float(config.translation_max_baseline_fraction)
                    ):
                        continue
                    for (
                        candidate,
                        translation,
                        rotation,
                    ) in _deterministic_safety_perturbations(
                        base,
                        baseline,
                        config,
                        base_rejection_reason,
                        translation_max_baseline_fraction=translation_fraction,
                    ):
                        coverage = run_stage(
                            "target_pose_coverage_validation",
                            lambda points=points, candidate=candidate, sample_index=sample_index: (
                                _coverage_from_points(
                                    points,
                                    target_K[sample_index],
                                    candidate,
                                    config,
                                )
                            ),
                        )
                        accepted, final_reason = run_stage(
                            "target_pose_safety_checks",
                            lambda candidate=candidate, coverage=coverage, check_center=check_center: (
                                _candidate_checks(
                                    candidate, check_center, coverage, config
                                )
                            ),
                        )
                        rejected += 1
                        if accepted:
                            chosen = candidate
                            chosen_coverage = coverage
                            chosen_base = base
                            chosen_arc_used = arc_used
                            chosen_geometry_center_used = geometry_center_used
                            chosen_scene_center = check_center
                            chosen_translation = translation
                            chosen_rotation = rotation
                            chosen_safety_limit_escalated = escalated
                            fallback = True
                            final_reason = (
                                "accepted_exceptional_bounded_translation"
                                if escalated
                                else "accepted_deterministic_bounded_perturbation"
                            )
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
                relaxed_coverage_threshold = float(
                    config.exceptional_min_source_coverage
                )
                relaxed_clearance_threshold = float(
                    _source_calibrated_clearance_threshold(
                        source_clearances,
                        alpha,
                        config,
                    )
                )
                relaxed, relaxed_rejection_reason = _candidate_checks(
                    base,
                    check_center,
                    base_coverage,
                    config,
                    minimum_source_coverage=relaxed_coverage_threshold,
                    minimum_geometry_distance_m=relaxed_clearance_threshold,
                )
                if relaxed:
                    chosen = base
                    chosen_coverage = base_coverage
                    chosen_base = base
                    chosen_arc_used = arc_used
                    chosen_geometry_center_used = geometry_center_used
                    chosen_scene_center = check_center
                    chosen_safety_thresholds_relaxed = True
                    chosen_coverage_threshold = relaxed_coverage_threshold
                    chosen_clearance_threshold = relaxed_clearance_threshold
                    fallback = True
                    final_reason = "accepted_exceptional_safety_thresholds"
                    break
                for (
                    candidate,
                    translation,
                    rotation,
                ) in _deterministic_safety_perturbations(
                    base,
                    baseline,
                    config,
                    relaxed_rejection_reason,
                    translation_max_baseline_fraction=float(
                        config.exceptional_translation_max_baseline_fraction
                    ),
                ):
                    coverage = run_stage(
                        "target_pose_coverage_validation",
                        lambda points=points, candidate=candidate, sample_index=sample_index: (
                            _coverage_from_points(
                                points,
                                target_K[sample_index],
                                candidate,
                                config,
                            )
                        ),
                    )
                    accepted, final_reason = run_stage(
                        "target_pose_safety_checks",
                        lambda candidate=candidate, coverage=coverage, check_center=check_center, coverage_threshold=relaxed_coverage_threshold, clearance_threshold=relaxed_clearance_threshold: (
                            _candidate_checks(
                                candidate,
                                check_center,
                                coverage,
                                config,
                                minimum_source_coverage=coverage_threshold,
                                minimum_geometry_distance_m=clearance_threshold,
                            )
                        ),
                    )
                    rejected += 1
                    if accepted:
                        chosen = candidate
                        chosen_coverage = coverage
                        chosen_base = base
                        chosen_arc_used = arc_used
                        chosen_geometry_center_used = geometry_center_used
                        chosen_scene_center = check_center
                        chosen_translation = translation
                        chosen_rotation = rotation
                        chosen_safety_limit_escalated = bool(
                            float(translation.norm())
                            > float(config.translation_max_baseline_fraction * baseline)
                            + 1.0e-6
                        )
                        chosen_safety_thresholds_relaxed = True
                        chosen_coverage_threshold = relaxed_coverage_threshold
                        chosen_clearance_threshold = relaxed_clearance_threshold
                        fallback = True
                        final_reason = "accepted_exceptional_safety_thresholds"
                        break
                if chosen is not None:
                    break
            if chosen is None:
                coverage = _coverage_from_points(
                    points, target_K[sample_index], configured_base, config
                )
                accepted, final_reason = _candidate_checks(
                    configured_base,
                    scene_center[sample_index],
                    coverage,
                    config,
                )
                raise RuntimeError(
                    "No safe LagerNVS target for "
                    f"sample={sample_index}, target={target_index}, alpha={float(alpha):.6f}: "
                    f"{final_reason}."
                )
            assert chosen is not None and chosen_coverage is not None
            per_sample["target_c2w"].append(chosen)
            per_sample["base_c2w"].append(chosen_base)
            per_sample["support_mask"].append(chosen_coverage["support_mask"])
            per_sample["baseline"].append(baseline)
            per_sample["translation_perturbation"].append(chosen_translation)
            per_sample["translation_perturbation_magnitude"].append(
                chosen_translation.norm()
            )
            per_sample["rotation_perturbation_degrees"].append(chosen_rotation)
            per_sample["source_coverage"].append(chosen_coverage["coverage_fraction"])
            per_sample["minimum_geometry_distance"].append(
                chosen_coverage["minimum_geometry_distance"]
            )
            per_sample["geometry_clearance_distance"].append(
                chosen_coverage["geometry_clearance_distance"]
            )
            per_sample["rejected_candidates"].append(
                torch.tensor(rejected, device=cameras.device, dtype=torch.int16)
            )
            per_sample["fallback_used"].append(
                torch.tensor(fallback, device=cameras.device, dtype=torch.bool)
            )
            per_sample["scene_centered_arc_used"].append(
                torch.tensor(chosen_arc_used, device=cameras.device, dtype=torch.bool)
            )
            per_sample["geometry_scene_center_used"].append(
                torch.tensor(
                    chosen_geometry_center_used,
                    device=cameras.device,
                    dtype=torch.bool,
                )
            )
            per_sample["safety_translation_limit_escalated"].append(
                torch.tensor(
                    chosen_safety_limit_escalated,
                    device=cameras.device,
                    dtype=torch.bool,
                )
            )
            per_sample["safety_thresholds_relaxed"].append(
                torch.tensor(
                    chosen_safety_thresholds_relaxed,
                    device=cameras.device,
                    dtype=torch.bool,
                )
            )
            per_sample["minimum_source_coverage_threshold"].append(
                torch.tensor(
                    chosen_coverage_threshold,
                    device=cameras.device,
                    dtype=torch.float32,
                )
            )
            per_sample["minimum_geometry_distance_threshold"].append(
                torch.tensor(
                    chosen_clearance_threshold,
                    device=cameras.device,
                    dtype=torch.float32,
                )
            )
            per_sample["source_clearance_reference"].append(
                (1.0 - alpha) * source_clearances[0] + alpha * source_clearances[1]
            )
            per_sample["target_scene_center"].append(chosen_scene_center)
            per_sample["alpha_resample_count"].append(alpha_resamples[target_index])
            per_sample["distance_from_camera_a"].append(
                (chosen[:3, 3] - cameras[0, :3, 3]).norm()
            )
            per_sample["distance_from_camera_b"].append(
                (chosen[:3, 3] - cameras[1, :3, 3]).norm()
            )
            reasons.append(final_reason)
        for name, values in per_sample.items():
            field_lists[name].append(torch.stack(values))

    output: dict[str, torch.Tensor | list[str]] = {
        name: torch.stack(values) for name, values in field_lists.items()
    }
    target_c2w = output["target_c2w"]
    assert isinstance(target_c2w, torch.Tensor)
    output["target_w2c"] = torch.linalg.inv(target_c2w)
    output["alpha"] = alphas
    output["final_reason"] = reasons
    return output
