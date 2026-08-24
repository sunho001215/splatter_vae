from __future__ import annotations

import torch

from dataset.droid.transforms import (
    GLOBAL_PAD_BOTTOM,
    GLOBAL_PAD_TOP,
    apply_spatial_transform,
    global_transform,
    image_validity_mask,
    local_transform,
    transform_confidence,
    transform_depth,
    transform_flow,
    transform_intrinsics,
    transform_validity,
)


def test_global_320x180_to_224x126_then_padding() -> None:
    transform = global_transform()
    assert transform.scale_x == 0.7
    assert transform.scale_y == 0.7
    assert transform.resized_width == 224
    assert transform.resized_height == 126
    assert transform.pad_top == GLOBAL_PAD_TOP == 49
    assert transform.pad_bottom == GLOBAL_PAD_BOTTOM == 49
    image = torch.ones(3, 180, 320)
    output = apply_spatial_transform(image, transform)
    assert output.shape == (3, 224, 224)
    assert torch.count_nonzero(output[:, :49]) == 0
    assert torch.count_nonzero(output[:, 175:]) == 0
    assert torch.all(output[:, 49:175] == 1)


def test_global_validity_mask() -> None:
    mask = image_validity_mask(global_transform())
    assert mask.shape == (1, 224, 224)
    assert int(mask.sum()) == 224 * 126
    assert not mask[:, :49].any()
    assert mask[:, 49:175].all()
    assert not mask[:, 175:].any()


def test_global_intrinsics_exact_update() -> None:
    K = torch.tensor(((200.0, 0.0, 160.0), (0.0, 210.0, 90.0), (0.0, 0.0, 1.0)))
    output = transform_intrinsics(K, global_transform())
    expected = torch.tensor(((140.0, 0.0, 112.0), (0.0, 147.0, 112.0), (0.0, 0.0, 1.0)))
    torch.testing.assert_close(output, expected)


def test_local_crop_and_intrinsics() -> None:
    transform = local_transform(70)
    assert transform.scale_x == transform.scale_y == 224 / 180
    image = torch.rand(3, 180, 320)
    assert apply_spatial_transform(image, transform).shape == (3, 224, 224)
    K = torch.tensor(((200.0, 0.0, 160.0), (0.0, 200.0, 90.0), (0.0, 0.0, 1.0)))
    output = transform_intrinsics(K, transform)
    assert torch.isclose(output[0, 2], torch.tensor((160.0 - 70.0) * 224 / 180))
    assert torch.isclose(output[1, 2], torch.tensor(90.0 * 224 / 180))


def test_depth_confidence_and_validity_transform_shapes() -> None:
    transform = local_transform(10)
    depth = torch.arange(180 * 320, dtype=torch.float32).view(1, 180, 320)
    confidence = torch.rand_like(depth)
    validity = depth > 10
    assert transform_depth(depth, transform).shape == (1, 224, 224)
    assert transform_confidence(confidence, transform).shape == (1, 224, 224)
    transformed_validity = transform_validity(validity, transform)
    assert transformed_validity.dtype == torch.bool
    assert transformed_validity.shape == (1, 224, 224)


def test_flow_vector_scaling() -> None:
    flow = torch.zeros(2, 180, 320)
    flow[0] = 10.0
    flow[1] = -5.0
    global_flow = transform_flow(flow, global_transform())
    valid = image_validity_mask(global_transform())[0]
    torch.testing.assert_close(global_flow[0][valid], torch.full((224 * 126,), 7.0))
    torch.testing.assert_close(global_flow[1][valid], torch.full((224 * 126,), -3.5))
    local_flow = transform_flow(flow, local_transform(50))
    torch.testing.assert_close(local_flow[0], torch.full((224, 224), 10.0 * 224 / 180))
    torch.testing.assert_close(local_flow[1], torch.full((224, 224), -5.0 * 224 / 180))
