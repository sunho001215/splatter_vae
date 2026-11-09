"""CPU client preprocessing: numpy + Pillow only; no torch, model imports, or I/O."""
from __future__ import annotations

import numpy as np

from s4d.data.droid.largescale.crop import center_box, crop_intrinsics, crop_rgb
from s4d.realworld.frame_preprocessing import IMAGE_SIZE, nearest_resample, training_depth
from s4d.rwlearn.deploy.contracts import validate_native_correction
from s4d.rwlearn.deploy.errors import ContractViolation


def preprocess_frame(rgb: np.ndarray, *, depth: np.ndarray | None = None, valid: np.ndarray | None = None,
                     K_native: np.ndarray | None = None, out_size: int = IMAGE_SIZE,
                     native_correction: dict | None = None) -> dict:
    """Aligned RGB uint8 + optional metric camera-z on the COLOR grid -> exact training-store frame.

    K_native uses continuous pixel centres. Depth correction, when supplied, must belong to sensor_depth.
    Invalids are zeroed before nearest resampling, exactly as the train_data writer does. No quantisation.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) <= 0:
        raise ContractViolation('rgb must be nonempty H,W,3 uint8 RGB')
    if not isinstance(out_size, int) or out_size <= 0:
        raise ContractViolation('out_size must be a positive integer')
    box = center_box(*rgb.shape[:2])
    out = {'rgb': crop_rgb(rgb, box, out_size)}
    if K_native is not None:
        K = np.asarray(K_native, float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 \
                or not np.allclose(K[2], [0, 0, 1]):
            raise ContractViolation('invalid native colour intrinsics')
        out['K'] = crop_intrinsics(K, box, out_size)
    if (depth is None) != (valid is None):
        raise ContractViolation('metric depth and validity must be supplied together')
    if native_correction is not None and depth is None:
        raise ContractViolation('depth correction without depth')
    if depth is not None:
        depth, valid = np.asarray(depth, np.float32), np.asarray(valid)
        if depth.shape != rgb.shape[:2] or valid.shape != depth.shape or valid.dtype != bool:
            raise ContractViolation('depth and bool validity must match the colour grid')
        if native_correction is not None:
            validate_native_correction(native_correction)
            depth = depth * np.float32(native_correction['factor'])
        out['depth'] = nearest_resample(training_depth(depth, valid), box, out_size)
        out['valid'] = out['depth'] > 0
    return out
