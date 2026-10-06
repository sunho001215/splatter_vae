"""Batch contract shared by the Meta-World and DROID loaders.

images        (B,T=3,V,3,H,W)   uint8
K             (B,V,3,3)         float32, pixel units for H,W (continuous pixel coords, cx~W/2)
w2c, c2w      (B,V,4,4)         float32, OpenCV convention, world = robot base
depth         (B,T,V,1,H,W)     float32 meters, 0 = invalid
motion3d      (B,P=3,V,3,H,W)   float32 world-frame displacement for PAIRS
motion_weight (B,P,V,1,H,W)     float32 in [0,1]; 0 where no target
motion_score  (B,T,V,1,H,W)     float32 in [0,1]
probe_state   (B,T,Dp)          float32 (optional)
eval_images   (B,T,Ve,3,H,W)    uint8   (optional, with eval_K / eval_w2c / eval_depth)
meta          dict
"""

from __future__ import annotations

import torch

T_WINDOW = 3
# (source time, target time, time whose pixel grid the displacement lives on)
PAIRS = ((0, 1, 0), (1, 2, 1), (0, 2, 0))
MOTION_SCORE_SCALE_M = 0.03

REQUIRED = ("images", "K", "w2c", "c2w", "depth", "motion3d", "motion_weight", "motion_score")
EVAL_FIELDS = ("eval_images", "eval_K", "eval_w2c", "eval_depth")


class ContractError(ValueError):
    pass


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ContractError(message)


def _check_tensor(batch: dict, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    _check(name in batch, f"missing field {name!r}")
    value = batch[name]
    _check(torch.is_tensor(value), f"{name} must be a tensor")
    _check(tuple(value.shape) == tuple(shape), f"{name} has shape {tuple(value.shape)}, expected {shape}")
    _check(value.dtype == dtype, f"{name} has dtype {value.dtype}, expected {dtype}")
    if value.is_floating_point():
        _check(bool(torch.isfinite(value).all()), f"{name} contains non-finite values")
    return value


def _check_cameras(K: torch.Tensor, w2c: torch.Tensor, c2w: torch.Tensor | None, H: int, W: int, name: str) -> None:
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    _check(bool((fx > 0).all() and (fy > 0).all()), f"{name}: focal lengths must be positive")
    _check(
        bool((cx > 0).all() and (cx < W).all() and (cy > 0).all() and (cy < H).all()),
        f"{name}: principal point must lie inside the image (pixel units)",
    )
    _check(
        bool(torch.allclose(K[..., 2, :], K.new_tensor([0.0, 0.0, 1.0]), atol=1e-6)),
        f"{name}: last row of K must be [0,0,1]",
    )
    bottom = w2c.new_tensor([0.0, 0.0, 0.0, 1.0])
    _check(bool(torch.allclose(w2c[..., 3, :], bottom, atol=1e-5)), f"{name}: w2c bottom row must be [0,0,0,1]")
    R = w2c[..., :3, :3]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    _check(bool(torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-3)), f"{name}: w2c rotation is not orthonormal")
    _check(
        bool(torch.allclose(torch.linalg.det(R), torch.ones_like(fx), atol=1e-3)),
        f"{name}: rotation must be proper with determinant +1",
    )
    if c2w is not None:
        prod = w2c @ c2w
        eye4 = torch.eye(4, dtype=prod.dtype, device=prod.device)
        _check(bool(torch.allclose(prod, eye4, atol=1e-3)), f"{name}: w2c @ c2w is not identity")


def validate_batch(batch: dict, *, training: bool = True) -> dict:
    """Validate shapes, dtypes, and value ranges. Returns ``(B, T, V, H, W)`` sizes."""
    _check(isinstance(batch, dict), "batch must be a dict")
    images = batch.get("images")
    _check(torch.is_tensor(images) and images.dim() == 6, "images must be a 6-D tensor (B,T,V,3,H,W)")
    B, T, V, C, H, W = images.shape
    _check(T == T_WINDOW and C == 3, f"images must have T={T_WINDOW} and 3 channels, got T={T}, C={C}")
    _check_tensor(batch, "images", (B, T, V, 3, H, W), torch.uint8)
    K = _check_tensor(batch, "K", (B, V, 3, 3), torch.float32)
    w2c = _check_tensor(batch, "w2c", (B, V, 4, 4), torch.float32)
    c2w = _check_tensor(batch, "c2w", (B, V, 4, 4), torch.float32)
    _check_cameras(K, w2c, c2w, H, W, "train cameras")
    depth = _check_tensor(batch, "depth", (B, T, V, 1, H, W), torch.float32)
    _check(bool((depth >= 0).all()), "depth must be >= 0 (0 marks invalid)")
    _check_tensor(batch, "motion3d", (B, len(PAIRS), V, 3, H, W), torch.float32)
    weight = _check_tensor(batch, "motion_weight", (B, len(PAIRS), V, 1, H, W), torch.float32)
    _check(bool((weight >= 0).all() and (weight <= 1).all()), "motion_weight must lie in [0,1]")
    score = _check_tensor(batch, "motion_score", (B, T, V, 1, H, W), torch.float32)
    _check(bool((score >= 0).all() and (score <= 1).all()), "motion_score must lie in [0,1]")
    if "probe_state" in batch:
        probe = batch["probe_state"]
        _check(
            torch.is_tensor(probe)
            and probe.dim() == 3
            and tuple(probe.shape[:2]) == (B, T)
            and probe.dtype == torch.float32,
            "probe_state must be float32 (B,T,Dp)",
        )
        _check(bool(torch.isfinite(probe).all()), "probe_state contains non-finite values")
    present = [name for name in EVAL_FIELDS if name in batch]
    if present:
        _check(len(present) == len(EVAL_FIELDS), f"eval fields must be given together, found {present}")
        Ve = batch["eval_images"].shape[2]
        _check_tensor(batch, "eval_images", (B, T, Ve, 3, H, W), torch.uint8)
        eK = _check_tensor(batch, "eval_K", (B, Ve, 3, 3), torch.float32)
        ew2c = _check_tensor(batch, "eval_w2c", (B, Ve, 4, 4), torch.float32)
        _check_cameras(eK, ew2c, None, H, W, "eval cameras")
        edepth = _check_tensor(batch, "eval_depth", (B, T, Ve, 1, H, W), torch.float32)
        _check(bool((edepth >= 0).all()), "eval_depth must be >= 0")
    # Held-out fields are optional in both regimes, including DROID validation.
    _check("meta" in batch and isinstance(batch["meta"], dict), "meta must be a dict")
    return {"B": B, "T": T, "V": V, "H": H, "W": W}


def collate(samples: list[dict]) -> dict:
    """Stack tensors; gather ``meta`` entries into lists."""
    out: dict = {}
    for key in samples[0]:
        if key == "meta":
            out["meta"] = {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}
        else:
            out[key] = torch.stack([s[key] for s in samples], dim=0)
    validate_batch(out)
    return out
