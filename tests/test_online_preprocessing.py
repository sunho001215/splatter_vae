from __future__ import annotations

import torch

from dataset.droid.sampling import MotionCropConfig
from models.training.online_preprocessing import (
    OnlinePreprocessingConfig,
    OnlineTeacherPipeline,
)
from preprocessing.memfof.official import _install_native_droid_corr_fix


class _FakeMEMFOF:
    def __call__(self, histories: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, cameras = histories.shape[:2]
        flow = torch.zeros(batch, cameras, 2, 2, 180, 320)
        flow[:, 0, :, 0, 60:75, 95:110] = 4.0
        flow[:, 1, :, 0, 115:130, 215:230] = 6.0
        validity = torch.ones(batch, cameras, 2, 1, 180, 320, dtype=torch.bool)
        return {
            "flow": flow,
            "confidence": torch.ones_like(validity, dtype=torch.float32),
            "validity": validity,
        }


class _FakeXLens:
    def __call__(
        self, histories: torch.Tensor, intrinsics: torch.Tensor, c2w: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del intrinsics, c2w
        batch, cameras, timesteps = histories.shape[:3]
        shape = (batch, timesteps, cameras, 1, 180, 320)
        depth = torch.full(shape, 2.0)
        return {
            "metric_depth": depth,
            "confidence": torch.ones(shape),
            "validity": torch.ones(shape, dtype=torch.bool),
        }


def _raw_batch() -> dict[str, torch.Tensor]:
    history = torch.zeros(1, 2, 3, 3, 180, 320, dtype=torch.uint8)
    history[:, :, 0] = 32
    history[:, :, 1] = 96
    history[:, :, 2] = 160
    K = torch.tensor(
        [[[200.0, 0.0, 160.0], [0.0, 210.0, 90.0], [0.0, 0.0, 1.0]]]
    ).expand(1, 2, -1, -1).clone()
    c2w = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, -1, -1).clone()
    return {
        "raw_histories": history,
        "raw_K": K,
        "raw_c2w": c2w,
        "raw_w2c": torch.linalg.inv(c2w),
        "sampled_crop_size": torch.tensor([180]),
        "calibration_validity": torch.tensor([True]),
    }


def test_online_teachers_create_temporally_aligned_camera_specific_crops() -> None:
    pipeline = OnlineTeacherPipeline(
        _FakeMEMFOF(),
        _FakeXLens(),
        OnlinePreprocessingConfig(
            motion_crop=MotionCropConfig(
                min_size=180,
                max_size=180,
                flow_smoothing_kernel=15,
            )
        ),
    )
    batch = pipeline(_raw_batch())
    assert batch["representation_histories"].shape == (1, 2, 3, 3, 224, 224)
    assert batch["representation_flows"].shape == (1, 2, 2, 2, 224, 224)
    assert batch["target_rgb"].shape == (1, 3, 2, 3, 224, 224)
    assert batch["target_depth"].shape == (1, 3, 2, 1, 224, 224)
    assert batch["target_flow"].shape == (1, 2, 2, 2, 224, 224)
    assert batch["target_K"].shape == (1, 3, 2, 3, 3)
    centers_x = batch["crop_metadata"]["crop_center_x"][0]
    centers_y = batch["crop_metadata"]["crop_center_y"][0]
    assert 95 <= int(centers_x[0]) <= 109
    assert 215 <= int(centers_x[1]) <= 229
    assert 130 <= int(centers_y[0]) <= 144
    assert 185 <= int(centers_y[1]) <= 199
    assert not batch["crop_metadata"]["low_motion_fallback_used"].any()
    torch.testing.assert_close(batch["target_K"][:, 0], batch["target_K"][:, 1])
    torch.testing.assert_close(batch["target_K"][:, 1], batch["target_K"][:, 2])
    assert torch.equal(
        batch["target_image_validity"][:, 0],
        batch["target_image_validity"][:, 2],
    )
    assert "segmentation" not in batch


def test_online_zero_motion_uses_center_for_both_cameras() -> None:
    class _ZeroMEMFOF(_FakeMEMFOF):
        def __call__(self, histories):
            result = super().__call__(histories)
            result["flow"].zero_()
            return result

    pipeline = OnlineTeacherPipeline(
        _ZeroMEMFOF(),
        _FakeXLens(),
        OnlinePreprocessingConfig(motion_crop=MotionCropConfig()),
    )
    raw = _raw_batch()
    raw["sampled_crop_size"] = torch.tensor([181])
    batch = pipeline(raw)
    assert batch["crop_metadata"]["crop_center_x"].tolist() == [[160, 160]]
    assert batch["crop_metadata"]["crop_center_y"].tolist() == [[160, 160]]
    assert batch["crop_metadata"]["low_motion_fallback_used"].all()


def test_memfof_native_correlation_handles_singleton_pyramid_dimension() -> None:
    import memfof.corr as corr_module

    _install_native_droid_corr_fix()
    image = torch.randn(128, 1, 1, 2)
    coordinates = torch.randn(128, 3, 3, 2)
    sampled = corr_module.bilinear_sampler(image, coordinates)

    assert sampled.shape == (128, 1, 3, 3)
    assert torch.isfinite(sampled).all()
