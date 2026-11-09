"""Torch-free, lossless frame resampling shared by the training store and CPU deployment client."""

import numpy as np

from s4d.data.droid.largescale import crop as CR

IMAGE_SIZE = 128


def nearest_resample(array: np.ndarray, box: CR.CropBox, size: int = IMAGE_SIZE) -> np.ndarray:
    """(H, W) -> (size, size) with the dtype kept: the native pixel containing each output pixel centre (the box must lie
    inside the image; ``crop_depth`` rule)."""
    H, W = array.shape[:2]
    if box.padding(H, W) != (0, 0, 0, 0):
        raise ValueError("the stored crop never pads")
    centres = (np.arange(size, dtype=np.float64) + 0.5) * (box.side / size)
    rows = np.floor(box.y0 + centres).astype(np.int64)
    cols = np.floor(box.x0 + centres).astype(np.int64)
    return np.ascontiguousarray(array[np.ix_(rows, cols)])


def training_depth(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Depth-stage output -> the stored float32 metres: invalid pixels 0, values bitwise; refuses non-finite or negative
    values where valid."""
    d = np.where(valid, np.asarray(depth_m, np.float32), np.float32(0))
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("depth must be finite and >= 0 where valid (0 = invalid)")
    return d.astype(np.float32)

