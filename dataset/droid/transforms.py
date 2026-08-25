from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

RLDS_HEIGHT = 180
RLDS_WIDTH = 320
PADDED_SIZE = 320
PAD_TOP = 70
PAD_BOTTOM = 70
MODEL_SIZE = 224

# The uint8 values nearest the ImageNet mean. After encoder normalization the
# geometric padding is approximately zero, while its validity remains false.
IMAGENET_NEUTRAL_RGB = (124.0, 116.0, 104.0)


@dataclass(frozen=True)
class SpatialTransform:
    """Pad an RLDS image, take one square crop, then resize it.

    ``crop_x`` and ``crop_y`` are expressed on the padded canvas. The discrete
    center convention uses ``half_left = crop_size // 2`` and therefore
    ``crop_x = center_x - half_left`` for both even and odd crop sizes.
    """

    crop_x: int
    crop_y: int
    crop_size: int
    source_height: int = RLDS_HEIGHT
    source_width: int = RLDS_WIDTH
    padded_height: int = PADDED_SIZE
    padded_width: int = PADDED_SIZE
    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = PAD_TOP
    pad_bottom: int = PAD_BOTTOM
    output_height: int = MODEL_SIZE
    output_width: int = MODEL_SIZE

    def __post_init__(self) -> None:
        dimensions = (
            self.source_height,
            self.source_width,
            self.padded_height,
            self.padded_width,
            self.crop_size,
            self.output_height,
            self.output_width,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Spatial dimensions must be positive.")
        if min(self.pad_left, self.pad_right, self.pad_top, self.pad_bottom) < 0:
            raise ValueError("Padding must be non-negative.")
        if self.source_width + self.pad_left + self.pad_right != self.padded_width:
            raise ValueError("Horizontal padding does not produce the padded width.")
        if self.source_height + self.pad_top + self.pad_bottom != self.padded_height:
            raise ValueError("Vertical padding does not produce the padded height.")
        if self.crop_x < 0 or self.crop_y < 0:
            raise ValueError("Crop origins must be non-negative.")
        if self.crop_x + self.crop_size > self.padded_width:
            raise ValueError("Horizontal crop exceeds the padded image.")
        if self.crop_y + self.crop_size > self.padded_height:
            raise ValueError("Vertical crop exceeds the padded image.")

    @property
    def scale_x(self) -> float:
        return self.output_width / self.crop_size

    @property
    def scale_y(self) -> float:
        return self.output_height / self.crop_size

    @property
    def resize_scale(self) -> float:
        if self.output_height != self.output_width:
            raise ValueError("The DROID motion crop must resize to a square.")
        return self.scale_x

    @property
    def crop_center_x(self) -> int:
        return self.crop_x + self.crop_size // 2

    @property
    def crop_center_y(self) -> int:
        return self.crop_y + self.crop_size // 2


def motion_crop_transform(
    crop_size: int,
    center_x: int,
    center_y: int,
    *,
    padded_size: int = PADDED_SIZE,
    pad_top: int = PAD_TOP,
    pad_bottom: int = PAD_BOTTOM,
    output_size: int = MODEL_SIZE,
) -> SpatialTransform:
    size = int(crop_size)
    half_left = size // 2
    return SpatialTransform(
        crop_x=int(center_x) - half_left,
        crop_y=int(center_y) - half_left,
        crop_size=size,
        padded_height=int(padded_size),
        padded_width=int(padded_size),
        pad_top=int(pad_top),
        pad_bottom=int(pad_bottom),
        output_height=int(output_size),
        output_width=int(output_size),
    )


def transform_intrinsics(K: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    """Apply RLDS -> padded canvas -> crop -> resize to a camera matrix."""
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"Expected K (...,3,3), got {tuple(K.shape)}.")
    output = K.clone().to(dtype=torch.float32)
    output[..., 0, 0] *= transform.scale_x
    output[..., 1, 1] *= transform.scale_y
    output[..., 0, 2] = transform.scale_x * (
        output[..., 0, 2] + transform.pad_left - transform.crop_x
    )
    output[..., 1, 2] = transform.scale_y * (
        output[..., 1, 2] + transform.pad_top - transform.crop_y
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
    original_shape = tuple(values.shape)
    if values.dim() == 2:
        return values[None, None], original_shape
    channels = int(values.shape[-3])
    return values.reshape(-1, channels, *values.shape[-2:]), original_shape


def _padding_tensor(
    values: torch.Tensor,
    channels: int,
    padding_value: float | Sequence[float],
) -> torch.Tensor:
    if isinstance(padding_value, Sequence):
        fill = tuple(float(value) for value in padding_value)
        if len(fill) != channels:
            raise ValueError(
                f"Expected {channels} padding values for the channel dimension, got {len(fill)}."
            )
        return values.new_tensor(fill).view(1, channels, 1, 1)
    return values.new_full((1, channels, 1, 1), float(padding_value))


def pad_to_square(
    values: torch.Tensor,
    transform: SpatialTransform,
    *,
    padding_value: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    """Pad channel-first RLDS tensors without reflection or replication."""
    if tuple(values.shape[-2:]) != (transform.source_height, transform.source_width):
        raise ValueError(
            f"Expected source {(transform.source_height, transform.source_width)}, "
            f"got {tuple(values.shape[-2:])}."
        )
    original_dtype = values.dtype
    flat, original_shape = _as_nchw(values)
    working = flat.float()
    fill = _padding_tensor(working, int(working.shape[1]), padding_value)
    padded = fill.expand(
        int(working.shape[0]),
        int(working.shape[1]),
        transform.padded_height,
        transform.padded_width,
    ).clone()
    padded[
        ...,
        transform.pad_top : transform.pad_top + transform.source_height,
        transform.pad_left : transform.pad_left + transform.source_width,
    ] = working
    output_shape = (*original_shape[:-2], transform.padded_height, transform.padded_width)
    padded = padded.reshape(output_shape)
    if original_dtype == torch.bool:
        return padded > 0.5
    if original_dtype.is_floating_point:
        return padded.to(dtype=original_dtype)
    return padded.round().clamp(0, torch.iinfo(original_dtype).max).to(original_dtype)


def apply_spatial_transform(
    values: torch.Tensor,
    transform: SpatialTransform,
    *,
    mode: Literal["bicubic", "bilinear", "nearest"] = "bilinear",
    padding_value: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    """Pad, crop and resize a channel-first tensor in any leading shape."""
    original_shape = tuple(values.shape)
    original_dtype = values.dtype
    padded = pad_to_square(values, transform, padding_value=padding_value)
    flat, _ = _as_nchw(padded)
    cropped = flat[
        ...,
        transform.crop_y : transform.crop_y + transform.crop_size,
        transform.crop_x : transform.crop_x + transform.crop_size,
    ].float()
    kwargs: dict[str, object] = {
        "mode": mode,
        "size": (transform.output_height, transform.output_width),
    }
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        if mode == "bicubic":
            kwargs["antialias"] = True
    resized = F.interpolate(cropped, **kwargs)
    output_shape = (*original_shape[:-2], transform.output_height, transform.output_width)
    resized = resized.reshape(output_shape)
    if original_dtype == torch.bool:
        return resized > 0.5
    if original_dtype.is_floating_point:
        return resized.to(dtype=original_dtype)
    return resized.round().clamp(0, torch.iinfo(original_dtype).max).to(original_dtype)


def transform_rgb(
    values: torch.Tensor,
    transform: SpatialTransform,
    *,
    padding_value: float | Sequence[float] = IMAGENET_NEUTRAL_RGB,
) -> torch.Tensor:
    return apply_spatial_transform(
        values,
        transform,
        mode="bicubic",
        padding_value=padding_value,
    )


def transform_depth(values: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    # Nearest-neighbor avoids averaging metric depth across discontinuities.
    output = apply_spatial_transform(
        values, transform, mode="nearest", padding_value=0.0
    )
    return output.masked_fill(~image_validity_mask(transform), 0)


def transform_confidence(
    values: torch.Tensor, transform: SpatialTransform
) -> torch.Tensor:
    output = apply_spatial_transform(
        values, transform, mode="bilinear", padding_value=0.0
    )
    return output.masked_fill(~image_validity_mask(transform), 0)


def transform_validity(
    values: torch.Tensor, transform: SpatialTransform
) -> torch.Tensor:
    # A pixel is valid only when the complete bilinear sampling footprint came
    # from real source pixels. This prevents boundary interpolation with padded
    # zeros from leaking into RGB/depth/flow reconstruction losses.
    coverage = apply_spatial_transform(
        values.float(), transform, mode="bilinear", padding_value=0.0
    )
    return coverage >= 1.0 - 1.0e-6


def transform_flow(flow: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    if flow.shape[-3] != 2:
        raise ValueError(
            f"Expected flow channel dimension of two, got {tuple(flow.shape)}."
        )
    output = apply_spatial_transform(
        flow, transform, mode="bilinear", padding_value=0.0
    ).float()
    output[..., 0, :, :] *= transform.scale_x
    output[..., 1, :, :] *= transform.scale_y
    output = output.masked_fill(~image_validity_mask(transform), 0.0)
    return output.to(dtype=flow.dtype)


def image_validity_mask(
    transform: SpatialTransform, leading_shape: tuple[int, ...] = ()
) -> torch.Tensor:
    source = torch.ones(
        (*leading_shape, 1, transform.source_height, transform.source_width),
        dtype=torch.bool,
    )
    return transform_validity(source, transform)
