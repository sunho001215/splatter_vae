from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from preprocessing.lagernvs.camera import (
    canonical_intrinsics,
    lager_to_display_plane,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    resample_pinhole_images,
)
from preprocessing.lagernvs.coverage import source_coverage_from_xlens
from preprocessing.lagernvs.official import _construct_official_model
from preprocessing.lagernvs.pose import LagerTargetPoseConfig, sample_safe_target_poses


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
        torch.eye(3)
        + torch.sin(angle) * skew
        + (1 - torch.cos(angle)) * (skew @ skew)
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


def test_official_model_loader_handles_project_models_package_collision(tmp_path) -> None:
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
    source_K = torch.tensor(
        [[[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]]]
    ).expand(2, -1, -1).clone()
    source_c2w = torch.eye(4).reshape(1, 4, 4).expand(2, -1, -1).clone()
    depth = torch.ones(2, 1, 180, 320)
    result = source_coverage_from_xlens(
        depth,
        torch.ones_like(depth),
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


def test_target_sampler_respects_alpha_and_perturbation_bounds() -> None:
    scene = torch.tensor((0.0, 0.0, 0.5))
    camera_a = _look_at(torch.tensor((-0.6, -0.8, 1.0)), scene)
    camera_b = _look_at(torch.tensor((0.7, -0.7, 0.9)), scene)
    source_c2w = torch.stack((camera_a, camera_b))[None]
    K = torch.tensor(
        [[[210.0, 0.0, 160.0], [0.0, 210.0, 90.0], [0.0, 0.0, 1.0]]]
    ).expand(1, 2, -1, -1).clone()
    depth = torch.full((1, 2, 1, 180, 320), 1.0)
    config = LagerTargetPoseConfig(
        min_source_coverage=0.0,
        minimum_geometry_distance_m=0.0,
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
        torch.ones_like(depth),
        torch.ones_like(depth, dtype=torch.bool),
        scene,
        config,
        seed=11,
        stage_runner=stage_runner,
    )
    baseline = float(result["baseline"][0])
    assert 0.15 <= float(result["alpha"][0]) <= 0.85
    assert (
        float(result["translation_perturbation_magnitude"][0])
        <= 0.03 * baseline + 1.0e-6
    )
    assert float(result["rotation_perturbation_degrees"][0]) <= 3.0 + 1.0e-6
    assert not bool(result["fallback_used"][0])
    rotation = result["target_c2w"][0, :3, :3]
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
