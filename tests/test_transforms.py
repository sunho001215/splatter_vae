from __future__ import annotations

import random

import torch

from dataset.droid.sampling import (
    MotionCropConfig,
    build_motion_maps,
    sample_uniform_crop_size,
    select_motion_crop,
    select_motion_crop_from_maps,
)
from dataset.droid.transforms import (
    PAD_BOTTOM,
    PAD_TOP,
    apply_spatial_transform,
    image_validity_mask,
    motion_crop_transform,
    pad_to_square,
    transform_confidence,
    transform_depth,
    transform_flow,
    transform_intrinsics,
    transform_validity,
)


def test_320x180_is_padded_to_320_square_with_70_pixel_borders() -> None:
    transform = motion_crop_transform(320, 160, 160)
    image = torch.ones(3, 180, 320)
    padded = pad_to_square(image, transform, padding_value=(0.5, 0.5, 0.5))
    assert padded.shape == (3, 320, 320)
    assert transform.pad_top == PAD_TOP == 70
    assert transform.pad_bottom == PAD_BOTTOM == 70
    assert torch.all(padded[:, :70] == 0.5)
    assert torch.all(padded[:, 70:250] == 1.0)
    assert torch.all(padded[:, 250:] == 0.5)


def test_crop_size_sampling_is_uniform_on_closed_interval() -> None:
    config = MotionCropConfig()
    rng = random.Random(20260825)
    draws = torch.tensor([sample_uniform_crop_size(config, rng) for _ in range(28_200)])
    assert int(draws.min()) == 180
    assert int(draws.max()) == 320
    counts = torch.bincount(draws - 180, minlength=141).float()
    expected = draws.numel() / 141
    chi_square = (((counts - expected) ** 2) / expected).sum()
    # Deterministic empirical guard against a hidden endpoint/global mixture.
    assert float(chi_square) < 210.0
    assert float(counts.max() / counts.min()) < 1.6


def test_motion_center_is_highest_feasible_flow_after_size_is_known() -> None:
    config = MotionCropConfig(flow_smoothing_kernel=1)
    flows = torch.zeros(3, 2, 180, 320)
    validity = torch.ones(3, 1, 180, 320, dtype=torch.bool)
    # x=310 is outside the feasible center region for S=180; x=220 is inside.
    flows[:, 0, 100, 310] = 100.0
    flows[:, 0, 100, 220] = 10.0
    selected = select_motion_crop(flows, validity, 180, config)
    assert selected.metadata.crop_center_x == 220
    assert selected.metadata.crop_center_y == 170  # raw y=100 plus top padding
    assert selected.transform.crop_x == 130
    assert selected.transform.crop_y == 80
    assert selected.transform.crop_x + 180 <= 320
    assert selected.transform.crop_y + 180 <= 320
    assert not selected.metadata.low_motion_fallback_used


def test_staged_motion_map_and_crop_selection_matches_composed_api() -> None:
    config = MotionCropConfig(flow_smoothing_kernel=9)
    flows = torch.zeros(2, 2, 180, 320)
    flows[:, 0, 40:55, 245:260] = 7.0
    validity = torch.ones(2, 1, 180, 320, dtype=torch.bool)
    aggregate, smoothed = build_motion_maps(flows, validity, config)
    staged = select_motion_crop_from_maps(aggregate, smoothed, 211, config)
    composed = select_motion_crop(flows, validity, 211, config)
    assert staged.metadata == composed.metadata
    assert staged.transform == composed.transform
    torch.testing.assert_close(staged.aggregate_motion_map, aggregate)
    torch.testing.assert_close(staged.smoothed_motion_map, smoothed)


def test_flow_smoothing_prefers_active_region_over_isolated_noisy_pixel() -> None:
    config = MotionCropConfig(flow_smoothing_kernel=15)
    flows = torch.zeros(3, 2, 180, 320)
    validity = torch.ones(3, 1, 180, 320, dtype=torch.bool)
    flows[0, 0, 40, 100] = 100.0
    flows[:, 0, 90:105, 205:220] = 2.0
    selected = select_motion_crop(flows, validity, 180, config)
    assert 205 <= selected.metadata.crop_center_x <= 219
    assert 160 <= selected.metadata.crop_center_y <= 174


def test_full_canvas_crop_has_only_center_and_exact_validity() -> None:
    config = MotionCropConfig(flow_smoothing_kernel=1)
    flows = torch.rand(3, 2, 180, 320)
    selected = select_motion_crop(flows, None, 320, config)
    assert selected.metadata.crop_center_x == 160
    assert selected.metadata.crop_center_y == 160
    assert selected.transform.crop_x == selected.transform.crop_y == 0
    mask = image_validity_mask(selected.transform)
    assert mask.shape == (1, 224, 224)
    assert int(mask.sum()) == 224 * 126
    assert not mask[:, :49].any()
    assert mask[:, 49:175].all()
    assert not mask[:, 175:].any()


def test_zero_flow_uses_deterministic_center_fallback() -> None:
    config = MotionCropConfig(low_motion_threshold=1.0e-4)
    flows = torch.zeros(3, 2, 180, 320)
    first = select_motion_crop(flows, None, 181, config)
    second = select_motion_crop(flows, None, 181, config)
    assert first.metadata.low_motion_fallback_used
    assert first.metadata.crop_center_x == first.metadata.crop_center_y == 160
    assert first.transform == second.transform
    assert first.transform.crop_x == first.transform.crop_y == 70


def test_intrinsics_follow_padding_crop_and_resize_formula_exactly() -> None:
    transform = motion_crop_transform(200, center_x=180, center_y=120)
    assert transform.crop_x == 80
    assert transform.crop_y == 20
    K = torch.tensor(((200.0, 0.0, 160.0), (0.0, 210.0, 90.0), (0.0, 0.0, 1.0)))
    output = transform_intrinsics(K, transform)
    scale = 224.0 / 200.0
    expected = torch.tensor(
        (
            (200.0 * scale, 0.0, (160.0 - 80.0) * scale),
            (0.0, 210.0 * scale, (90.0 + 70.0 - 20.0) * scale),
            (0.0, 0.0, 1.0),
        )
    )
    torch.testing.assert_close(output, expected)


def test_depth_confidence_validity_and_padding_are_consistent() -> None:
    transform = motion_crop_transform(320, 160, 160)
    depth = torch.full((1, 180, 320), 2.0)
    confidence = torch.ones_like(depth)
    validity = torch.ones_like(depth, dtype=torch.bool)
    output_depth = transform_depth(depth, transform)
    output_confidence = transform_confidence(confidence, transform)
    output_validity = transform_validity(validity, transform)
    image_validity = image_validity_mask(transform)
    assert output_depth.shape == output_confidence.shape == (1, 224, 224)
    assert torch.equal(output_validity, image_validity)
    assert torch.count_nonzero(output_depth[~image_validity]) == 0
    assert torch.count_nonzero(output_confidence[~image_validity]) == 0
    torch.testing.assert_close(
        output_depth[image_validity], torch.full((224 * 126,), 2.0)
    )


def test_flow_spatial_resize_and_vector_magnitudes_use_crop_scale() -> None:
    transform = motion_crop_transform(200, 160, 160)
    flow = torch.zeros(2, 180, 320)
    flow[0] = 10.0
    flow[1] = -5.0
    output = transform_flow(flow, transform)
    validity = image_validity_mask(transform)[0]
    scale = 224.0 / 200.0
    torch.testing.assert_close(output[0][validity], torch.full_like(output[0][validity], 10.0 * scale))
    torch.testing.assert_close(output[1][validity], torch.full_like(output[1][validity], -5.0 * scale))
    assert torch.count_nonzero(output[:, ~validity]) == 0


def test_apply_transform_accepts_a_shared_temporal_tensor() -> None:
    transform = motion_crop_transform(231, 160, 160)
    history = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1).expand(3, 3, 180, 320)
    output = apply_spatial_transform(history, transform, mode="bilinear")
    assert output.shape == (3, 3, 224, 224)
    validity = image_validity_mask(transform, leading_shape=(3,))
    for frame in range(3):
        torch.testing.assert_close(
            output[frame][validity[frame].expand_as(output[frame])],
            torch.full_like(output[frame][validity[frame].expand_as(output[frame])], float(frame)),
        )
