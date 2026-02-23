from __future__ import annotations

from typing import Dict, Tuple, Optional
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T


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
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(dev)

    # DINOv2 uses ImageNet-style normalization commonly in examples
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    dtype_np = np.float16 if out_dtype == "float16" else np.float32

    with h5py.File(hdf5_path, "a") as f:
        data = f["data"]
        demos = sorted(list(data.keys()))
        for demo in tqdm(demos, desc="DINO demos"):
            demo_grp = data[demo]
            obs_grp = demo_grp["obs"]

            cams = camera_names
            if cams is None:
                cams = list(json_load(demo_grp.attrs["camera_names"]))

            for cam in cams:
                rgb_ds = obs_grp[f"{cam}_rgb"]
                Tlen = rgb_ds.shape[0]

                # create output ds if missing
                out_name = f"{cam}_dino"
                if out_name in obs_grp:
                    continue  # skip if already computed

                # infer token grid from a dummy forward
                dummy = tfm(rgb_ds[0]).unsqueeze(0).to(dev)
                with torch.no_grad():
                    pt = _get_patchtokens(model, dummy)  # (1, N, D)
                D = pt.shape[-1]
                N = pt.shape[1]
                # For ViT patch models: N = (H/ps)*(W/ps). Assume square grid.
                grid = int(round(N ** 0.5))
                if grid * grid != N:
                    raise RuntimeError(f"Non-square token grid: N={N}")
                gh = gw = grid

                out_ds = obs_grp.create_dataset(
                    out_name,
                    shape=(Tlen, gh, gw, D),
                    maxshape=(Tlen, gh, gw, D),
                    dtype=dtype_np,
                    chunks=True,
                    compression=demo_grp.file["data"].attrs.get("compression", None),
                )
                out_ds.attrs["model"] = model_name
                out_ds.attrs["image_size"] = int(image_size)
                out_ds.attrs["token_grid_h"] = int(gh)
                out_ds.attrs["token_grid_w"] = int(gw)

                # batch over frames
                idx = 0
                while idx < Tlen:
                    j = min(Tlen, idx + batch_size)
                    imgs = []
                    for k in range(idx, j):
                        imgs.append(tfm(rgb_ds[k]))
                    x = torch.stack(imgs, dim=0).to(dev)

                    with torch.no_grad():
                        pt = _get_patchtokens(model, x)          # (B, N, D)
                        pt = pt.reshape(-1, gh, gw, D).cpu().numpy()

                    out_ds[idx:j] = pt.astype(dtype_np)
                    idx = j


def json_load(s: str):
    import json
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return json.loads(s)