from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

XLENS_PATCH_SIZE = 14
XLENS_PREPROCESSING_VERSION = "droid-pinhole-symmetric-pad14-crop-v1"


def pad_pinhole_scene_to_patch_multiple(
    images: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    *,
    patch_size: int = XLENS_PATCH_SIZE,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int, int, int]]:
    if not images or len(images) != len(intrinsics):
        raise ValueError("X-Lens requires aligned nonempty image and intrinsic lists.")
    height, width = np.asarray(images[0]).shape[:2]
    target_height = ((height + patch_size - 1) // patch_size) * patch_size
    target_width = ((width + patch_size - 1) // patch_size) * patch_size
    pad_y = target_height - height
    pad_x = target_width - width
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    left, right = pad_x // 2, pad_x - pad_x // 2
    padded_images: list[np.ndarray] = []
    padded_intrinsics: list[np.ndarray] = []
    for image, K in zip(images, intrinsics, strict=True):
        value = np.asarray(image, dtype=np.uint8)
        if value.shape != (height, width, 3):
            raise ValueError(
                "Every synchronized X-Lens image must share one HxWx3 shape."
            )
        padded_images.append(
            np.pad(value, ((top, bottom), (left, right), (0, 0)), mode="edge")
        )
        camera = np.asarray(K, dtype=np.float32).copy()
        if camera.shape != (3, 3):
            raise ValueError("Every X-Lens pinhole intrinsic must be 3x3.")
        camera[0, 2] += left
        camera[1, 2] += top
        padded_intrinsics.append(camera)
    return padded_images, padded_intrinsics, (left, right, top, bottom)


class XLensDROIDTeacher:
    """Thin adapter around the official X-Lens inference implementation."""

    def __init__(
        self,
        repository: str,
        checkpoint: str,
        *,
        architecture_config: str | None = None,
        device: str = "cuda:0",
        amp_dtype: str = "bf16",
    ):
        root = Path(repository).expanduser().resolve()
        if not (root / "xlens" / "inference" / "pipeline.py").is_file():
            raise FileNotFoundError(
                f"Official X-Lens was not found at {root}; clone https://github.com/zhouhengamerica/XLens."
            )
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"X-Lens checkpoint does not exist: {checkpoint_path}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from xlens.inference import XLensInference
        from xlens.inference.preprocess import assemble_batch, pinhole_d_cam

        self._assemble_batch = assemble_batch
        self._pinhole_d_cam = pinhole_d_cam
        self.model = XLensInference(
            str(checkpoint_path),
            device=device,
            amp_dtype=amp_dtype,
            config=architecture_config,
        )

    def predict(
        self,
        images: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray],
        c2w: Sequence[np.ndarray],
    ) -> dict[str, np.ndarray]:
        original_height, original_width = np.asarray(images[0]).shape[:2]
        padded_images, padded_K, padding = pad_pinhole_scene_to_patch_multiple(
            images, intrinsics
        )
        height, width = padded_images[0].shape[:2]
        rays = [self._pinhole_d_cam(K, height, width) for K in padded_K]
        batch = self._assemble_batch(
            padded_images,
            rays,
            [1] * len(padded_images),
            c2w=np.asarray(c2w, dtype=np.float32),
            device=self.model.device,
        )
        output = self.model(batch)
        depth = output["depth_metric"][0].numpy()
        confidence = (
            output["depth_conf"][0].numpy()
            if "depth_conf" in output
            else np.ones_like(depth, dtype=np.float32)
        )
        left, _right, top, _bottom = padding
        depth = depth[:, top : top + original_height, left : left + original_width]
        confidence = confidence[
            :, top : top + original_height, left : left + original_width
        ]
        expected = (len(images), original_height, original_width)
        if depth.shape != expected or confidence.shape != expected:
            raise RuntimeError(
                f"X-Lens cropped output {depth.shape} does not match original grid {expected}."
            )
        validity = (
            np.isfinite(depth)
            & np.isfinite(confidence)
            & (depth > 0.0)
            & (confidence > 0.0)
        )
        return {
            "metric_depth": depth.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "validity": validity,
            "padding_lrtb": np.asarray(padding, dtype=np.int32),
        }
