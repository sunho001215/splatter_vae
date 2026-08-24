from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

RLDS_HEIGHT = 180
RLDS_WIDTH = 320
MODEL_SIZE = 224
GLOBAL_RESIZED_HEIGHT = 126
GLOBAL_PAD_TOP = 49
GLOBAL_PAD_BOTTOM = 49
LOCAL_CROP_SIZE = 180


@dataclass(frozen=True)
class SpatialTransform:
    source_height: int
    source_width: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    output_height: int
    output_width: int
    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = 0
    pad_bottom: int = 0

    def __post_init__(self) -> None:
        values = (
            self.source_height,
            self.source_width,
            self.crop_width,
            self.crop_height,
            self.output_height,
            self.output_width,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("Spatial dimensions must be positive.")
        if self.crop_x < 0 or self.crop_y < 0:
            raise ValueError("Crop origins must be non-negative.")
        if self.crop_x + self.crop_width > self.source_width:
            raise ValueError("Horizontal crop exceeds the source image.")
        if self.crop_y + self.crop_height > self.source_height:
            raise ValueError("Vertical crop exceeds the source image.")
        if min(self.pad_left, self.pad_right, self.pad_top, self.pad_bottom) < 0:
            raise ValueError("Padding must be non-negative.")

    @property
    def resized_height(self) -> int:
        return self.output_height - self.pad_top - self.pad_bottom

    @property
    def resized_width(self) -> int:
        return self.output_width - self.pad_left - self.pad_right

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.crop_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.crop_height


def global_transform() -> SpatialTransform:
    return SpatialTransform(
        source_height=RLDS_HEIGHT,
        source_width=RLDS_WIDTH,
        crop_x=0,
        crop_y=0,
        crop_width=RLDS_WIDTH,
        crop_height=RLDS_HEIGHT,
        output_height=MODEL_SIZE,
        output_width=MODEL_SIZE,
        pad_top=GLOBAL_PAD_TOP,
        pad_bottom=GLOBAL_PAD_BOTTOM,
    )


def local_transform(crop_x: int) -> SpatialTransform:
    return SpatialTransform(
        source_height=RLDS_HEIGHT,
        source_width=RLDS_WIDTH,
        crop_x=int(crop_x),
        crop_y=0,
        crop_width=LOCAL_CROP_SIZE,
        crop_height=LOCAL_CROP_SIZE,
        output_height=MODEL_SIZE,
        output_width=MODEL_SIZE,
    )


def transform_intrinsics(K: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"Expected K (...,3,3), got {tuple(K.shape)}.")
    output = K.clone().to(dtype=torch.float32)
    output[..., 0, 0] *= transform.scale_x
    output[..., 1, 1] *= transform.scale_y
    output[..., 0, 2] = (
        transform.scale_x * (output[..., 0, 2] - transform.crop_x) + transform.pad_left
    )
    output[..., 1, 2] = (
        transform.scale_y * (output[..., 1, 2] - transform.crop_y) + transform.pad_top
    )
    return output


def transform_intrinsics_numpy(
    K: np.ndarray, transform: SpatialTransform
) -> np.ndarray:
    tensor = transform_intrinsics(torch.as_tensor(K, dtype=torch.float32), transform)
    return tensor.cpu().numpy()


def _as_nchw(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if values.dim() < 2:
        raise ValueError("An image-like tensor must have at least H and W dimensions.")
    prefix = values.shape[:-2]
    channels = 1 if len(prefix) == 0 else int(prefix[-1])
    leading = prefix[:-1] if len(prefix) else ()
    if len(prefix) == 0:
        values = values[None, None]
    elif len(prefix) == 1:
        values = values[None]
    else:
        values = values.reshape(-1, channels, *values.shape[-2:])
    return values, tuple(leading)


def apply_spatial_transform(
    values: torch.Tensor,
    transform: SpatialTransform,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """Crop, resize and pad a channel-first image tensor in any leading shape."""
    if tuple(values.shape[-2:]) != (transform.source_height, transform.source_width):
        raise ValueError(
            f"Expected source {(transform.source_height, transform.source_width)}, "
            f"got {tuple(values.shape[-2:])}."
        )
    original_shape = tuple(values.shape)
    original_dtype = values.dtype
    flat, _ = _as_nchw(values)
    cropped = flat[
        ...,
        transform.crop_y : transform.crop_y + transform.crop_height,
        transform.crop_x : transform.crop_x + transform.crop_width,
    ]
    working = cropped.float()
    kwargs = {"mode": mode, "size": (transform.resized_height, transform.resized_width)}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    resized = F.interpolate(working, **kwargs)
    padded = F.pad(
        resized,
        (
            transform.pad_left,
            transform.pad_right,
            transform.pad_top,
            transform.pad_bottom,
        ),
    )
    output_shape = (
        *original_shape[:-2],
        transform.output_height,
        transform.output_width,
    )
    padded = padded.reshape(output_shape)
    if original_dtype == torch.bool:
        return padded > 0.5
    if original_dtype.is_floating_point:
        return padded.to(dtype=original_dtype)
    return padded.round().clamp(0, torch.iinfo(original_dtype).max).to(original_dtype)


def transform_rgb(values: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    return apply_spatial_transform(values, transform, mode="bilinear")


def transform_depth(values: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    return apply_spatial_transform(values, transform, mode="nearest")


def transform_confidence(
    values: torch.Tensor, transform: SpatialTransform
) -> torch.Tensor:
    return apply_spatial_transform(values, transform, mode="bilinear")


def transform_validity(
    values: torch.Tensor, transform: SpatialTransform
) -> torch.Tensor:
    return apply_spatial_transform(values.bool(), transform, mode="nearest")


def transform_flow(flow: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    if flow.shape[-3] != 2:
        raise ValueError(
            f"Expected flow channel dimension of two, got {tuple(flow.shape)}."
        )
    output = apply_spatial_transform(flow, transform, mode="bilinear").float()
    output[..., 0, :, :] *= transform.scale_x
    output[..., 1, :, :] *= transform.scale_y
    return output.to(dtype=flow.dtype)


def image_validity_mask(
    transform: SpatialTransform, leading_shape: tuple[int, ...] = ()
) -> torch.Tensor:
    mask = torch.zeros(
        (*leading_shape, 1, transform.output_height, transform.output_width),
        dtype=torch.bool,
    )
    mask[
        ...,
        transform.pad_top : transform.output_height - transform.pad_bottom,
        transform.pad_left : transform.output_width - transform.pad_right,
    ] = True
    return mask
