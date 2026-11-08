"""Visualization geometry store (separate from training data; data-format policy of 2026-11-06).

The viewer reads high-resolution RGB/IR from the original recording (or the normalized frames). The only expensive
DERIVED geometry it needs is the native-resolution metric depth of the depth stage (FoundationStereo on the device IR
pair: a GPU model call per frame), kept once here in a compact shared form: per camera the colour-grid depth as uint16
0.1 mm steps (0 = invalid; display precision, error <= 0.05 mm, NOT a training input), gzip + shuffle, one frame per
chunk, with the native K, the accepted T_cam_base and the frame timestamps. Point clouds are reconstructed from
depth + K + pose on demand (no XYZ stored); a viewer keeps only bounded, regenerable caches. Stage ``vis_geometry``
(optional; ``<output_root>/vis_geometry/<episode>/geometry.h5``).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio
from s4d.realworld.train_store import GZIP

FORMAT = "s4d.realworld.vis_geometry"
SCHEMA_VERSION = "1.0.0"
STAGE = "vis_geometry"
FILE = "geometry.h5"
DEPTH_UNIT_M = 1.0e-4  # uint16 0.1 mm steps up to 6.5535 m (as the Meta-World collector); 0 = invalid


def quantize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Metres (0 invalid) -> uint16 ``DEPTH_UNIT_M`` steps (display only; error <= 0.05 mm); a valid depth never
    becomes 0 and nothing saturates."""
    d = np.asarray(depth_m, np.float64)
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("depth must be finite and >= 0 (0 = invalid)")
    q = np.rint(d / DEPTH_UNIT_M)
    if q.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError(f"depth {d.max():.3f} m exceeds the uint16 range ({np.iinfo(np.uint16).max * DEPTH_UNIT_M} m)")
    return np.where(d > 0, np.maximum(q, 1), 0).astype(np.uint16)


def write_native_geometry(root: Path, episode_id: str, out_dir: Path, *, depth_stage: str | None = None,
                          chunk: int = 16, force: bool = False) -> dict:
    from s4d.realworld.depth_stages import load_view_depth
    from s4d.realworld.pipeline import (
        accepted_calibration,
        calibrated_geometry_config,
        calibrated_version,
        calibration_depth_scale,
        camera_views,
        default_depth_stage,
        load_meta,
    )

    root = Path(root)
    meta = load_meta(root, episode_id)
    depth_stage = depth_stage or default_depth_stage(meta, "lift")
    rec = rio.read_record(rio.stage_dir(root, episode_id, depth_stage))
    cal = rio.read_record(rio.stage_dir(root, episode_id, "calibrate"))
    if rec is None:
        raise RuntimeError(f"{STAGE} needs a complete {depth_stage} stage")
    views = [v for v in camera_views(meta) if v.stream_id in rec["summary"] and "skipped" not in rec["summary"][v.stream_id]]
    norm = rio.read_record(rio.episode_dir(root, episode_id) / "normalized")
    inputs = {"ingest": rio.build_id(norm), depth_stage: rio.build_id(rec),
              **({"calibrate": rio.build_id(cal)} if cal else {})}
    # depth stored with the accepted calibration's per-camera scale (point clouds of the cameras then overlay)
    cal_extra = calibrated_geometry_config(root, episode_id, meta, depth_stage) if cal else {}
    cal_extra.pop("robot_model", None)  # no robot geometry in this store
    config = {"depth_stage": depth_stage, "schema": SCHEMA_VERSION, **cal_extra}

    def build(tmp: Path) -> dict:
        with h5py.File(tmp / FILE, "x") as f:
            f.attrs.update(format=FORMAT, schema_version=SCHEMA_VERSION, episode_id=episode_id,
                           depth_stage=depth_stage, depth_unit_m=DEPTH_UNIT_M,
                           cameras=json.dumps([v.rig_id for v in views]))
            for view in views:
                cam = view.stream_id
                stamps = np.asarray(meta.stream_timestamps_ns[cam], np.int64)
                _, _, K = load_view_depth(root, episode_id, cam, [0], depth_stage)
                s = meta.stream(cam)
                H, W = s.intrinsics.height, s.intrinsics.width
                g = f.create_group(f"cameras/{view.rig_id}")
                g.attrs.update(stream_id=cam, native_hw=[H, W])
                g.create_dataset("timestamps_ns", data=stamps)
                g.create_dataset("K", data=np.asarray(K, np.float64))
                w2c = accepted_calibration(root, episode_id, cam)
                if w2c is not None:
                    g.create_dataset("w2c", data=np.asarray(w2c, np.float64))
                ds = g.create_dataset("depth", (len(stamps), H, W), np.uint16, chunks=(1, H, W), shuffle=True, **GZIP)
                scale = calibration_depth_scale(root, episode_id, cam, depth_stage)
                if scale != 1.0:
                    g.attrs["depth_scale"] = scale
                for start in range(0, len(stamps), chunk):
                    idx = list(range(start, min(len(stamps), start + chunk)))
                    depth, valid, _ = load_view_depth(root, episode_id, cam, idx, depth_stage)
                    depth = depth if scale == 1.0 else depth * np.float32(scale)
                    ds[start:start + len(idx)] = quantize_depth(np.where(valid, depth, 0.0))
        return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "cameras": [v.rig_id for v in views],
                "file_bytes": (tmp / FILE).stat().st_size}

    return rio.run_stage(Path(out_dir), stage=STAGE, version=calibrated_version("1", cal_extra), config=config,
                         inputs=inputs, build=build, force=force)


def native_depth(path: Path, camera: str, idx) -> tuple[np.ndarray, np.ndarray]:
    """(depth (n,H,W) float32 metres with 0 invalid, K (3,3)) of ``camera`` (rig serial) from a ``geometry.h5``."""
    path = Path(path)
    with h5py.File(path / FILE if path.is_dir() else path, "r") as f:
        g = f[f"cameras/{camera}"]
        idx = np.atleast_1d(np.asarray(idx, np.int64))
        d = np.stack([g["depth"][int(i)] for i in idx]).astype(np.float32) * np.float32(f.attrs["depth_unit_m"])
        return d.astype(np.float32), g["K"][:]
