from __future__ import annotations

from typing import Dict, Tuple, Optional
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


def _get_patchtokens(model, x: torch.Tensor) -> torch.Tensor:
    """
    Returns (B, N, D) patch tokens for DINOv2 hub models.
    """
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]
        # fallback guesses
        for k in ["x_prenorm_patchtokens", "x_patchtokens", "patchtokens"]:
            if k in feats:
                return feats[k]
        raise KeyError(f"forward_features keys: {list(feats.keys())}")
    raise AttributeError("Model has no forward_features; check dinov2 hub model.")

def _preprocess_rgb_batch(
    rgb_batch: np.ndarray,
    *,
    image_size: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized DINO preprocessing for a batch of RGB uint8 images.

    Args:
        rgb_batch: numpy array of shape (B, H, W, 3), uint8
        image_size: target square size
        device: torch device
        mean, std: normalization tensors of shape (1, 3, 1, 1)

    Returns:
        x: torch tensor of shape (B, 3, image_size, image_size), float32
    """
    if rgb_batch.ndim != 4 or rgb_batch.shape[-1] != 3:
        raise ValueError(
            f"Expected rgb_batch with shape (B, H, W, 3), got {tuple(rgb_batch.shape)}"
        )

    # uint8 [0,255] -> float32 [0,1]
    x = torch.from_numpy(rgb_batch).to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous() / 255.0  # (B,3,H,W)

    # Resize in batch on GPU/CPU without PIL.
    if x.shape[-2] != image_size or x.shape[-1] != image_size:
        x = F.interpolate(
            x,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    # ImageNet normalization used by DINOv2 examples.
    x = (x - mean) / std
    return x

def add_dino_features_inplace(
    hdf5_path: str,
    *,
    model_name: str,
    image_size: int,
    batch_size: int,
    device: str,
    out_dtype: str,
    camera_names: Optional[list[str]] = None,
) -> None:
    dev = torch.device(device)
    use_amp = (dev.type == "cuda")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(dev)

    # Optional memory-format hint for faster conv / patch embedding on CUDA.
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    dtype_np = np.float16 if out_dtype == "float16" else np.float32

    mean = torch.tensor((0.485, 0.456, 0.406), device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev, dtype=torch.float32).view(1, 3, 1, 1)

    with h5py.File(hdf5_path, "a") as f:
        data = f["data"]
        demos = sorted(list(data.keys()))

        with torch.inference_mode():
            for demo in tqdm(demos, desc="DINO demos"):
                demo_grp = data[demo]
                obs_grp = demo_grp["obs"]

                cams = camera_names
                if cams is None:
                    cams = list(json_load(demo_grp.attrs["camera_names"]))

                for cam in cams:
                    rgb_ds = obs_grp[f"{cam}_rgb"]
                    Tlen = rgb_ds.shape[0]

                    out_name = f"{cam}_dino"
                    if out_name in obs_grp:
                        continue  # skip if already computed

                    # ----------------------------------------------------------
                    # Infer token grid once from a SINGLE batched-preprocessed image
                    # ----------------------------------------------------------
                    rgb0 = rgb_ds[0:1]  # (1,H,W,3)
                    x0 = _preprocess_rgb_batch(
                        rgb0,
                        image_size=image_size,
                        device=dev,
                        mean=mean,
                        std=std,
                    )

                    if dev.type == "cuda":
                        x0 = x0.to(memory_format=torch.channels_last)

                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        pt0 = _get_patchtokens(model, x0)  # (1, N, D)

                    D = pt0.shape[-1]
                    N = pt0.shape[1]
                    grid = int(round(N ** 0.5))
                    if grid * grid != N:
                        raise RuntimeError(f"Non-square token grid: N={N}")
                    gh = gw = grid

                    out_ds = obs_grp.create_dataset(
                        out_name,
                        shape=(Tlen, gh, gw, D),
                        maxshape=(Tlen, gh, gw, D),
                        dtype=dtype_np,
                        chunks=(min(batch_size, Tlen), gh, gw, D),
                        compression=demo_grp.file["data"].attrs.get("compression", None),
                    )
                    out_ds.attrs["model"] = model_name
                    out_ds.attrs["image_size"] = int(image_size)
                    out_ds.attrs["token_grid_h"] = int(gh)
                    out_ds.attrs["token_grid_w"] = int(gw)

                    # ----------------------------------------------------------
                    # Batch over frames with batched preprocessing
                    # ----------------------------------------------------------
                    idx = 0
                    while idx < Tlen:
                        j = min(Tlen, idx + batch_size)

                        # Read a whole slice from HDF5 at once: (B,H,W,3)
                        rgb_np = rgb_ds[idx:j]

                        x = _preprocess_rgb_batch(
                            rgb_np,
                            image_size=image_size,
                            device=dev,
                            mean=mean,
                            std=std,
                        )

                        if dev.type == "cuda":
                            x = x.to(memory_format=torch.channels_last)

                        with torch.autocast(device_type="cuda", enabled=use_amp):
                            pt = _get_patchtokens(model, x)  # (B, N, D)

                        pt = pt.reshape(-1, gh, gw, D).detach().cpu().numpy().astype(dtype_np, copy=False)
                        out_ds[idx:j] = pt
                        idx = j

def json_load(s: str):
    import json
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return json.loads(s)