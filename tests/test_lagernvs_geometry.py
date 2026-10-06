from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

import preprocessing.lagernvs.pose as lager_pose
from preprocessing.lagernvs.camera import (
    canonical_intrinsics,
    lager_to_display_plane,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    resample_pinhole_images,
)
from preprocessing.lagernvs.coverage import source_coverage_from_depth
from preprocessing.lagernvs.official import _construct_official_model
from preprocessing.lagernvs.pose import (
    LagerTargetPoseConfig,
    _candidate_checks,
    _coverage_from_points,
    pose_sampler_contract,
    sample_safe_target_poses,
    sample_stratified_alphas,
)


def _look_at(center: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    optical = F.normalize(target - center, dim=0)
    world_up = torch.tensor((0.0, 0.0, 1.0))
    right = F.normalize(torch.cross(world_up, optical, dim=0), dim=0)
    down = torch.cross(optical, right, dim=0)
    pose = torch.eye(4)
    pose[:3, :3] = torch.stack((right, down, optical), dim=1)
    pose[:3, 3] = center
    return pose


def test_quaternion_round_trip_matches_rotation_matrix() -> None:
    axis = F.normalize(torch.tensor((0.3, -0.5, 0.7)), dim=0)
    angle = torch.tensor(math.radians(63.0))
    x, y, z = axis
    skew = torch.tensor(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    rotation = (
        torch.eye(3) + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)
    )
    recovered = quaternion_xyzw_to_matrix(matrix_to_quaternion_xyzw(rotation))
    torch.testing.assert_close(recovered, rotation, atol=1.0e-5, rtol=1.0e-5)


def test_identity_camera_resampling_is_pixel_exact() -> None:
    image = torch.rand(2, 3, 32, 32)
    K = canonical_intrinsics((2,), focal_px=30.0, size=32)
    output, validity = resample_pinhole_images(
        image, K, K, output_height=32, output_width=32
    )
    torch.testing.assert_close(output, image, atol=2.0e-6, rtol=2.0e-6)
    assert validity.all()


def test_lager_display_mapping_letterboxes_without_stretching() -> None:
    image = torch.ones(1, 3, 256, 256)
    K = canonical_intrinsics((1,))
    display, display_K, validity = lager_to_display_plane(image, K)
    assert display.shape == (1, 3, 180, 320)
    assert validity.shape == (1, 1, 180, 320)
    assert display_K[0, 0, 0] == display_K[0, 1, 1]
    columns = validity[0, 0].any(dim=0).nonzero().flatten()
    assert 175 <= columns.numel() <= 181
    assert abs(float(columns.float().mean()) - 159.5) < 1.0


def test_official_model_loader_handles_project_models_package_collision(
    tmp_path,
) -> None:
    repository = tmp_path / "lagernvs"
    upstream_models = repository / "models"
    upstream_models.mkdir(parents=True)
    (upstream_models / "encoder_decoder.py").write_text(
        "import torch\n"
        "class Stub(torch.nn.Module):\n"
        "    upstream_marker = True\n"
        "def EncDec_VitB8(**kwargs):\n"
        "    assert kwargs['pretrained_vggt'] is False\n"
        "    return Stub()\n",
        encoding="utf-8",
    )
    project_models = sys.modules.get("models")
    loaded = _construct_official_model(repository)
    assert loaded.upstream_marker
    assert sys.modules.get("models") is project_models
    assert "models.encoder_decoder" not in sys.modules


def test_source_coverage_projects_real_depth_into_target_domain() -> None:
    source_K = (
        torch.tensor([[[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]]])
        .expand(2, -1, -1)
        .clone()
    )
    source_c2w = torch.eye(4).reshape(1, 4, 4).expand(2, -1, -1).clone()
    depth = torch.ones(2, 1, 180, 320)
    result = source_coverage_from_depth(
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        source_K,
        source_c2w,
        canonical_intrinsics((), focal_px=200.0),
        torch.eye(4),
        sample_stride=2,
        dilation_kernel=3,
    )
    assert int(result["source_point_count"]) == 2 * 90 * 160
    assert 0.50 < float(result["coverage_fraction"]) < 0.80


def test_geometry_clearance_ignores_isolated_depth_outlier() -> None:
    grid = torch.linspace(-0.3, 0.3, 32)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((xx.flatten(), yy.flatten(), torch.ones(1024)), dim=-1)
    points[0] = torch.tensor((0.0, 0.0, 0.01))
    config = LagerTargetPoseConfig(
        min_source_coverage=0.0,
        exceptional_min_source_coverage=0.0,
        minimum_optical_axis_cosine=-1.0,
    )
    pose = torch.eye(4)
    coverage = _coverage_from_points(
        points,
        canonical_intrinsics((), focal_px=200.0),
        pose,
        config,
    )
    assert float(coverage["minimum_geometry_distance"]) < 0.02
    assert float(coverage["geometry_clearance_distance"]) > 0.9
    assert _candidate_checks(pose, torch.tensor((0.0, 0.0, 1.0)), coverage, config)[0]


def test_target_sampler_respects_alpha_and_perturbation_bounds() -> None:
    scene = torch.tensor((0.0, 0.0, 0.5))
    camera_a = _look_at(torch.tensor((-0.6, -0.8, 1.0)), scene)
    camera_b = _look_at(torch.tensor((0.7, -0.7, 0.9)), scene)
    source_c2w = torch.stack((camera_a, camera_b))[None]
    K = (
        torch.tensor([[[210.0, 0.0, 160.0], [0.0, 210.0, 90.0], [0.0, 0.0, 1.0]]])
        .expand(1, 2, -1, -1)
        .clone()
    )
    depth = torch.full((1, 2, 1, 180, 320), 1.0)
    config = LagerTargetPoseConfig(
        min_source_coverage=0.0,
        exceptional_min_source_coverage=0.0,
        minimum_geometry_distance_m=0.0,
        exceptional_minimum_geometry_distance_m=0.0,
        minimum_source_calibrated_clearance_m=0.0,
        minimum_optical_axis_cosine=-1.0,
    )
    profiled_stages = []

    def stage_runner(name, callable_):
        profiled_stages.append(name)
        return callable_()

    result = sample_safe_target_poses(
        source_c2w,
        K,
        canonical_intrinsics((1,)),
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        scene,
        config,
        seed=11,
        stage_runner=stage_runner,
    )
    baseline = result["baseline"][0]
    alpha = result["alpha"][0]
    assert alpha.shape == (4,)
    assert 0.15 <= float(alpha[0]) <= 0.25
    assert 0.25 <= float(alpha[1]) <= 0.35
    torch.testing.assert_close(alpha[2], 1.0 - alpha[1])
    torch.testing.assert_close(alpha[3], 1.0 - alpha[0])
    assert not ((alpha > 0.35) & (alpha < 0.65)).any()
    assert torch.all(
        result["translation_perturbation_magnitude"][0] <= 0.03 * baseline + 1.0e-6
    )
    assert torch.all(result["rotation_perturbation_degrees"][0] <= 3.0 + 1.0e-6)
    assert not result["fallback_used"][0].any()
    rotation = result["target_c2w"][0, 0, :3, :3]
    torch.testing.assert_close(
        rotation.T @ rotation, torch.eye(3), atol=1.0e-5, rtol=1.0e-5
    )
    torch.testing.assert_close(
        torch.det(rotation), torch.tensor(1.0), atol=1.0e-5, rtol=1.0e-5
    )
    assert {
        "target_pose_interpolation",
        "target_pose_bounded_perturbation",
        "target_pose_coverage_validation",
        "target_pose_safety_checks",
    }.issubset(profiled_stages)


def test_target_sampler_uses_bounded_deterministic_safety_fallback(
    monkeypatch,
) -> None:
    source_c2w = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, -1, -1).clone()
    source_c2w[0, 0, 0, 3] = -0.5
    source_c2w[0, 1, 0, 3] = 0.5
    source_K = canonical_intrinsics((1, 2), focal_px=30.0, size=8)
    target_K = canonical_intrinsics((1,), focal_px=30.0, size=8)
    depth = torch.ones(1, 2, 1, 8, 8)

    def synthetic_safety_geometry(points, intrinsic, candidate, config):
        del points, intrinsic, config
        on_camera_a_side = float(candidate[0, 3]) < 0.0
        translated_back = float(candidate[2, 3]) <= -0.029
        exact_bounded_rotation = abs(float(candidate[1, 2])) >= math.sin(
            math.radians(2.99)
        )
        coverage = 1.0 if on_camera_a_side or exact_bounded_rotation else 0.5
        clearance = 0.09 if not on_camera_a_side or translated_back else 0.01
        return {
            "support_mask": torch.ones(1, 8, 8, dtype=torch.bool),
            "coverage_fraction": candidate.new_tensor(coverage),
            "target_zbuffer": torch.ones(1, 8, 8),
            "minimum_geometry_distance": candidate.new_tensor(clearance),
            "geometry_clearance_distance": candidate.new_tensor(clearance),
        }

    monkeypatch.setattr(lager_pose, "_coverage_from_points", synthetic_safety_geometry)
    result = sample_safe_target_poses(
        source_c2w,
        source_K,
        target_K,
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        torch.tensor((0.0, 0.0, 1.0)),
        LagerTargetPoseConfig(
            prefer_scene_centered_arc=False,
            max_resample_attempts=1,
            coverage_sample_stride=1,
        ),
        seed=17,
    )

    assert result["fallback_used"][0].all()
    assert torch.all(result["source_coverage"] >= 0.60)
    assert torch.all(result["geometry_clearance_distance"] >= 0.08)
    assert torch.all(result["translation_perturbation_magnitude"] <= 0.03 + 1e-6)
    assert torch.all(result["rotation_perturbation_degrees"] <= 3.0 + 1e-6)
    assert torch.all(result["translation_perturbation_magnitude"][0, :2] > 0.029)
    assert torch.all(result["rotation_perturbation_degrees"][0, 2:] > 2.99)
    assert result["final_reason"] == ["accepted_deterministic_bounded_perturbation"] * 4


def test_target_sampler_uses_five_percent_translation_only_as_exception(
    monkeypatch,
) -> None:
    source_c2w = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, -1, -1).clone()
    source_c2w[0, 0, 0, 3] = -0.5
    source_c2w[0, 1, 0, 3] = 0.5
    source_K = canonical_intrinsics((1, 2), focal_px=30.0, size=8)
    target_K = canonical_intrinsics((1,), focal_px=30.0, size=8)
    depth = torch.ones(1, 2, 1, 8, 8)

    def geometry_requiring_exception(points, intrinsic, candidate, config):
        del points, intrinsic, config
        clearance = 0.09 if float(candidate[2, 3]) <= -0.049 else 0.01
        return {
            "support_mask": torch.ones(1, 8, 8, dtype=torch.bool),
            "coverage_fraction": candidate.new_tensor(1.0),
            "target_zbuffer": torch.ones(1, 8, 8),
            "minimum_geometry_distance": candidate.new_tensor(clearance),
            "geometry_clearance_distance": candidate.new_tensor(clearance),
        }

    monkeypatch.setattr(
        lager_pose, "_coverage_from_points", geometry_requiring_exception
    )
    result = sample_safe_target_poses(
        source_c2w,
        source_K,
        target_K,
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        torch.tensor((0.0, 0.0, 1.0)),
        LagerTargetPoseConfig(
            prefer_scene_centered_arc=False,
            max_resample_attempts=1,
            alpha_safety_grid_candidates=2,
            coverage_sample_stride=1,
        ),
        seed=23,
    )

    assert result["safety_translation_limit_escalated"][0].all()
    assert torch.all(result["translation_perturbation_magnitude"][0] <= 0.05 + 1e-6)
    assert torch.all(result["translation_perturbation_magnitude"][0] >= 0.049)
    assert torch.all(result["rotation_perturbation_degrees"] == 0.0)
    assert result["final_reason"] == ["accepted_exceptional_bounded_translation"] * 4


def test_target_sampler_flags_last_resort_safety_thresholds(monkeypatch) -> None:
    source_c2w = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, -1, -1).clone()
    source_c2w[0, 0, 0, 3] = -0.5
    source_c2w[0, 1, 0, 3] = 0.5
    source_K = canonical_intrinsics((1, 2), focal_px=30.0, size=8)
    target_K = canonical_intrinsics((1,), focal_px=30.0, size=8)
    depth = torch.ones(1, 2, 1, 8, 8)

    def exceptional_but_usable_geometry(points, intrinsic, candidate, config):
        del points, intrinsic, config
        return {
            "support_mask": torch.ones(1, 8, 8, dtype=torch.bool),
            "coverage_fraction": candidate.new_tensor(0.45),
            "target_zbuffer": torch.ones(1, 8, 8),
            "minimum_geometry_distance": candidate.new_tensor(0.07),
            "geometry_clearance_distance": candidate.new_tensor(0.055),
        }

    monkeypatch.setattr(
        lager_pose, "_coverage_from_points", exceptional_but_usable_geometry
    )
    result = sample_safe_target_poses(
        source_c2w,
        source_K,
        target_K,
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        torch.tensor((0.0, 0.0, 1.0)),
        LagerTargetPoseConfig(
            prefer_scene_centered_arc=False,
            max_resample_attempts=1,
            alpha_safety_grid_candidates=2,
            coverage_sample_stride=1,
        ),
        seed=29,
    )

    assert result["safety_thresholds_relaxed"][0].all()
    assert not result["safety_translation_limit_escalated"][0].any()
    torch.testing.assert_close(
        result["minimum_source_coverage_threshold"][0],
        torch.full((4,), 0.40),
    )
    torch.testing.assert_close(
        result["minimum_geometry_distance_threshold"][0],
        torch.full((4,), 0.05),
    )
    assert result["final_reason"] == ["accepted_exceptional_safety_thresholds"] * 4


def test_pose_sampler_contract_is_stable_and_configuration_sensitive() -> None:
    first = pose_sampler_contract()
    second = pose_sampler_contract(LagerTargetPoseConfig())
    changed = pose_sampler_contract(
        LagerTargetPoseConfig(exceptional_min_source_coverage=0.41)
    )

    assert first == second
    assert len(first["signature"]) == 64
    assert first["signature"] != changed["signature"]
    assert first["tiers"]["ordinary"]["minimum_source_coverage"] == 0.60
    assert (
        first["tiers"]["exceptional_safety_thresholds"]["minimum_source_coverage"]
        == 0.40
    )
    assert (
        first["tiers"]["exceptional_safety_thresholds"][
            "source_clearance_reference_fraction"
        ]
        == 0.80
    )
    assert (
        first["tiers"]["exceptional_safety_thresholds"][
            "hard_minimum_geometry_distance_m"
        ]
        == 0.02
    )


def test_source_calibrated_clearance_is_bounded_and_alpha_aware() -> None:
    config = LagerTargetPoseConfig()
    source_clearances = torch.tensor((0.10, 0.02))
    near_a = lager_pose._source_calibrated_clearance_threshold(
        source_clearances, torch.tensor(0.25), config
    )
    near_b = lager_pose._source_calibrated_clearance_threshold(
        source_clearances, torch.tensor(0.75), config
    )
    hard_floor = lager_pose._source_calibrated_clearance_threshold(
        torch.zeros(2), torch.tensor(0.5), config
    )

    torch.testing.assert_close(near_a, torch.tensor(0.05))
    torch.testing.assert_close(near_b, torch.tensor(0.032))
    torch.testing.assert_close(hard_floor, torch.tensor(0.02))


def test_stratified_alpha_sampling_is_deterministic_symmetric_and_excludes_center() -> (
    None
):
    first = sample_stratified_alphas(8, seed=71, device="cpu")
    second = sample_stratified_alphas(8, seed=71, device="cpu")
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, 2], 1.0 - first[:, 1])
    torch.testing.assert_close(first[:, 3], 1.0 - first[:, 0])
    assert torch.all((first[:, 0] >= 0.15) & (first[:, 0] <= 0.25))
    assert torch.all((first[:, 1] >= 0.25) & (first[:, 1] <= 0.35))
    assert not ((first > 0.35) & (first < 0.65)).any()
