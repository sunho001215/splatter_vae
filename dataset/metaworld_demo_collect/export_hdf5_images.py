from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import h5py
import numpy as np
import yaml


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_camera_names(demo_grp: h5py.Group) -> list[str]:
    if "camera_names" in demo_grp.attrs:
        raw = _decode_attr(demo_grp.attrs["camera_names"])
        return list(json.loads(raw))

    if "obs" not in demo_grp:
        return []
    cams = []
    for key in demo_grp["obs"].keys():
        if key.endswith("_rgb"):
            cams.append(key[: -len("_rgb")])
    return sorted(cams)


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo"):
        try:
            return int(name[len("demo") :]), name
        except ValueError:
            pass
    return 10**18, name


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image as (H,W,3), got {tuple(rgb.shape)}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def _write_seg_png(path: Path, seg: np.ndarray) -> None:
    seg = np.asarray(seg)
    if seg.ndim != 2:
        raise ValueError(f"Expected segmentation image as (H,W), got {tuple(seg.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), seg.astype(np.uint16 if seg.max(initial=0) > 255 else np.uint8))
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def _selected(items: Sequence[str], requested: Iterable[str] | None) -> list[str]:
    if requested is None:
        return list(items)
    requested_set = set(requested)
    missing = sorted(requested_set.difference(items))
    if missing:
        raise ValueError(f"Requested cameras not found: {missing}; available={list(items)}")
    return [item for item in items if item in requested_set]


def export_metaworld_hdf5_images(
    hdf5_path: str | Path,
    output_dir: str | Path,
    *,
    cameras: Sequence[str] | None = None,
    max_frames_per_demo: int | None = None,
    include_seg: bool = False,
) -> int:
    hdf5_path = Path(hdf5_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")

    saved = 0
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError(f'Invalid MetaWorld HDF5 file "{hdf5_path}": missing /data group.')

        data_grp = f["data"]
        demo_names = sorted(data_grp.keys(), key=_demo_sort_key)
        for demo_name in demo_names:
            demo_grp = data_grp[demo_name]
            if "obs" not in demo_grp:
                print(f"[skip] /data/{demo_name} has no obs group")
                continue

            obs_grp = demo_grp["obs"]
            camera_names = _selected(_load_camera_names(demo_grp), cameras)
            for cam in camera_names:
                rgb_key = f"{cam}_rgb"
                if rgb_key not in obs_grp:
                    print(f"[skip] /data/{demo_name}/obs/{rgb_key} not found")
                    continue

                rgb_ds = obs_grp[rgb_key]
                num_frames = int(rgb_ds.shape[0])
                if max_frames_per_demo is not None:
                    num_frames = min(num_frames, int(max_frames_per_demo))

                for t in range(num_frames):
                    out_path = output_dir / demo_name / cam / f"t_{t:06d}.png"
                    _write_rgb_png(out_path, np.asarray(rgb_ds[t]))
                    saved += 1

                if include_seg:
                    for suffix in ("seg", "seg_type"):
                        seg_key = f"{cam}_{suffix}"
                        if seg_key not in obs_grp:
                            continue
                        seg_ds = obs_grp[seg_key]
                        seg_frames = min(num_frames, int(seg_ds.shape[0]))
                        for t in range(seg_frames):
                            out_path = output_dir / demo_name / f"{cam}_{suffix}" / f"t_{t:06d}.png"
                            _write_seg_png(out_path, np.asarray(seg_ds[t]))
                            saved += 1

            print(f"[done] {demo_name}: exported cameras={camera_names}")

    return saved


def _hdf5_path_from_config(config_path: str | Path) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        return Path(cfg["output"]["path"])
    except Exception as exc:
        raise ValueError(f'Config file "{config_path}" must contain output.path') from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all RGB images from a MetaWorld demonstration HDF5 file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hdf5-path", type=Path, help="Path to the input MetaWorld HDF5 file.")
    source.add_argument(
        "--config",
        type=Path,
        help="MetaWorld demo collection config YAML; uses output.path as the input HDF5 path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where exported PNG files will be written.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Optional camera names to export, e.g. --cameras cam0 cam1. Defaults to all cameras.",
    )
    parser.add_argument(
        "--max-frames-per-demo",
        type=int,
        default=None,
        help="Optional cap on exported frames per demo/camera.",
    )
    parser.add_argument(
        "--include-seg",
        action="store_true",
        help="Also export {cam}_seg and {cam}_seg_type datasets when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = args.hdf5_path if args.hdf5_path is not None else _hdf5_path_from_config(args.config)
    saved = export_metaworld_hdf5_images(
        hdf5_path=hdf5_path,
        output_dir=args.output_dir,
        cameras=args.cameras,
        max_frames_per_demo=args.max_frames_per_demo,
        include_seg=bool(args.include_seg),
    )
    print(f"Saved {saved} images to {args.output_dir}")


if __name__ == "__main__":
    main()
