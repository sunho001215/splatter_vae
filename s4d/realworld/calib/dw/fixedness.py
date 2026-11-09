"""Camera fixedness of one task (``rig_dw``): every episode's cameras registered to the task's reference episode
(the first in recording order) with ``calibration_rig.episode_motion`` (static textured patches, every temporal
segment). Writes ``<out>.json`` (per episode and camera) and ``<out>.reference.npz`` (reference grey image + dynamic
mask per camera, the arrays a ``calibration_rig`` document references).

    python -m s4d.realworld.rig_dw_fixedness --task-root processed/<release>/<task> --out OUT"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np


def appearance(root, ep, cam, frames):
    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import load_meta, read_frames, to_model_rgb

    meta = load_meta(root, ep)
    n = len(meta.stream_timestamps_ns[cam])
    pick = np.linspace(0, n - 1, min(frames, n)).round().astype(int)
    rgb = to_model_rgb(read_frames(root, ep, cam, pick), meta.stream(cam))
    with h5py.File(rio.stage_dir(root, ep, "sam3_masks") / f"{rio.safe_name(cam)}.h5", "r") as f:
        cls = json.loads(f.attrs.get("class_names", '["background", "robot", "object", "fixture"]'))
        sem = f["semantic"][()]
    K = meta.stream(cam).intrinsics.K()
    return rgb, np.isin(sem, [cls.index(c) for c in ("robot", "object") if c in cls]), K


def main(argv=None):
    from s4d.realworld.calib.rig import FixednessConfig, episode_motion, static_image
    from s4d.realworld.pipeline import camera_views, load_meta

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segments", type=int, default=None, help="temporal segments per episode (default: FixednessConfig)")
    a = ap.parse_args(argv)
    task_root = Path(a.task_root)
    eps = sorted((p.name for p in task_root.iterdir() if p.is_dir()), key=lambda e: int(e.rsplit("-", 1)[1]))
    cfg = FixednessConfig() if a.segments is None else FixednessConfig(segments=a.segments)
    ref_ep = eps[0]
    cams = [v.stream_id for v in camera_views(load_meta(task_root / ref_ep, ref_ep))]
    ref, ref_arrays = {}, {}
    for cam in cams:
        rgb, dyn, K = appearance(task_root / ref_ep, ref_ep, cam, cfg.frames)
        g, d = static_image(rgb, dyn, cfg)
        ref[cam] = (g, d)
        key = cam.replace("/", "__")
        ref_arrays[f"{key}__grey"], ref_arrays[f"{key}__dynamic"] = g, d
    out = {"reference_episode": ref_ep, "config": asdict(cfg), "episodes": {}}
    for ep in eps:
        res = {}
        for cam in cams:
            rgb, dyn, K = appearance(task_root / ep, ep, cam, cfg.frames)
            res[cam] = episode_motion(ref[cam][0], ref[cam][1], rgb, dyn, K, cfg)
        out["episodes"][ep] = res
        print(json.dumps({"episode": ep, **{c: [r["fixed"], r.get("max_shift_px")] for c, r in res.items()}}),
              flush=True)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p.with_suffix(".reference.npz"), **ref_arrays)
    p.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
