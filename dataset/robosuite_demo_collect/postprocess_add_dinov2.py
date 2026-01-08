from __future__ import annotations

import argparse
import json
import warnings
from typing import List

import h5py
import numpy as np

from demo_collector.dino_dense import DINOv2DenseExtractor


def _get_demo_names(data_grp: h5py.Group) -> List[str]:
    """
    Demos are stored as /data/demo1, /data/demo2, ...
    Return sorted by numeric index if possible.
    """
    names = [k for k in data_grp.keys() if k.startswith("demo")]
    def _key(x: str):
        try:
            return int(x.replace("demo", ""))
        except Exception:
            return 10**9
    return sorted(names, key=_key)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hdf5", type=str, required=True, help="Existing collection output .hdf5 (modified in-place)")

    # DINO settings
    parser.add_argument("--dino_source", type=str, default="torchhub", choices=["torchhub", "hf"])
    parser.add_argument("--dino_mode", type=str, default="patch", choices=["patch", "pixel"])
    parser.add_argument("--dino_device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP even on CUDA")

    # Performance / safety knobs
    parser.add_argument("--batch_steps", type=int, default=32, help="How many timesteps per chunk (per camera) to process")
    parser.add_argument("--overwrite", action="store_true", help="If set, delete existing *_dinov2 datasets and recompute")
    parser.add_argument("--skip_if_exists", action="store_true", help="If set, skip demos that already have *_dinov2")

    args = parser.parse_args()

    # Optional: silence the common DINOv2 torchhub warnings about xFormers.
    # (This does NOT enable xFormers; it only hides warnings.)
    warnings.filterwarnings("ignore", message=r"xFormers is not available.*", category=UserWarning)

    # Create extractor once (reused for all demos)
    extractor = DINOv2DenseExtractor(
        device=args.dino_device,
        model_source=args.dino_source,
        mode=args.dino_mode,
        amp=(not args.no_amp),
    )

    # Modify the SAME file (in-place)
    with h5py.File(args.hdf5, "r+") as f:
        if "data" not in f:
            raise RuntimeError("Invalid file: missing /data group")

        data_grp = f["data"]
        demo_names = _get_demo_names(data_grp)

        for demo_name in demo_names:
            demo_grp = data_grp[demo_name]
            obs_grp = demo_grp.get("obs", None)
            if obs_grp is None:
                print(f"[skip] {demo_name}: missing /obs")
                continue

            # Camera list is stored as JSON in demo attrs (same as your writer)
            cam_names = json.loads(demo_grp.attrs["camera_names"])
            if not cam_names:
                print(f"[skip] {demo_name}: empty camera_names")
                continue

            # Determine T,H,W from the first camera RGB dataset
            first_cam = cam_names[0]
            rgb_key = f"{first_cam}_rgb"
            if rgb_key not in obs_grp:
                print(f"[skip] {demo_name}: missing {rgb_key}")
                continue

            rgb0 = obs_grp[rgb_key]
            T, H, W, C = rgb0.shape
            assert C == 3

            # If DINO already exists, optionally skip
            any_dino_key = f"{first_cam}_dinov2"
            if any_dino_key in obs_grp and args.skip_if_exists:
                print(f"[skip] {demo_name}: DINO already exists (skip_if_exists)")
                continue

            # If overwrite, delete old datasets
            if args.overwrite:
                for cam in cam_names:
                    k = f"{cam}_dinov2"
                    if k in obs_grp:
                        del obs_grp[k]

            # If not overwrite and already exists, we either skip or continue safely
            if (any_dino_key in obs_grp) and (not args.overwrite):
                print(f"[skip] {demo_name}: DINO already exists (use --overwrite to recompute)")
                continue

            # Compute one tiny batch to infer output spec (grid_h, grid_w, feat_dim, patch_size)
            # Using 1 image from cam0
            sample_img = obs_grp[f"{first_cam}_rgb"][0:1]  # (1,H,W,3)
            feats1, spec = extractor.extract(np.asarray(sample_img, dtype=np.uint8))

            # Create output datasets: same naming convention as your old online writer
            # Use compression matching your existing images if possible.
            compression = rgb0.compression or "lzf"

            for cam in cam_names:
                dino_key = f"{cam}_dinov2"
                if spec.mode == "patch":
                    tail = (spec.grid_h, spec.grid_w, spec.feat_dim)
                else:
                    # pixel mode output is (H,W,D) and can be huge
                    tail = (spec.grid_h, spec.grid_w, spec.feat_dim)

                obs_grp.create_dataset(
                    dino_key,
                    shape=(T, *tail),
                    maxshape=(T, *tail),          # fixed-length since T is known now
                    dtype=np.float16,
                    chunks=(1, *tail),            # chunk per timestep
                    compression=compression,
                )

            # Store metadata on obs group (same keys as before)
            obs_grp.attrs["dinov2_patch_size"] = int(spec.patch_size)
            obs_grp.attrs["dinov2_feat_dim"] = int(spec.feat_dim)
            obs_grp.attrs["dinov2_mode"] = str(spec.mode)

            print(f"[dino] {demo_name}: T={T}, HxW={H}x{W}, mode={spec.mode}, grid={spec.grid_h}x{spec.grid_w}, D={spec.feat_dim}")

            # Process in time chunks to control RAM
            bs = int(args.batch_steps)
            for t0 in range(0, T, bs):
                t1 = min(T, t0 + bs)
                cur_bs = t1 - t0

                # Build a single big batch: (num_cams*cur_bs, H, W, 3)
                # Ordering: [cam0 frames..., cam1 frames..., ...]
                img_batches = []
                for cam in cam_names:
                    rgb_ds = obs_grp[f"{cam}_rgb"]
                    img_batches.append(np.asarray(rgb_ds[t0:t1], dtype=np.uint8))
                imgs = np.concatenate(img_batches, axis=0)

                feats, _ = extractor.extract(imgs)  # (num_cams*cur_bs, gh, gw, D) or (.., H, W, D)

                # Split back per-camera and write to disk
                for cam_i, cam in enumerate(cam_names):
                    dino_ds = obs_grp[f"{cam}_dinov2"]
                    start = cam_i * cur_bs
                    end = (cam_i + 1) * cur_bs
                    dino_ds[t0:t1] = feats[start:end].astype(np.float16)

            # Mark file-level root attr if you want (optional)
            data_grp.attrs["dinov2_status"] = "computed_offline"

            # Ensure changes are persisted
            f.flush()

        print(f"[DONE] Added DINOv2 features into: {args.hdf5}")


if __name__ == "__main__":
    main()
