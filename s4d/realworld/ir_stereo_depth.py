"""Stereo depth from an RGB-D device's own infrared pair: FoundationStereo on each RealSense left/right IR pair ->
metric depth on the colour grid. The standard own-rig depth for lifting and export ("B1", selected by the 2026-11-05
depth comparison; docs/realworld_preprocessing.md section 9). The model receives the IR pair only.

Conventions (``s4d.realworld.stereo``; device facts checked by the adapter and ``validate_episode``):

* The IR pair is already rectified by the device (identical K, zero distortion, R = I, baseline along x): it is
  used as recorded (never rectified a second time) at its native 640x480 (no resize; FoundationStereo pads to a
  multiple of 32 internally and unpads). Grayscale IR is replicated to 3 channels as MODEL INPUT ONLY.
* Disparity is left-referenced, ``d = x_left - x_right`` > 0, on the left-IR grid; the right-referenced disparity
  for the left-right check comes from flip-and-swap. Depth ``z = fx_ir * B / d`` (``doffs = 0``: both IR streams
  share K) with ``B`` the device's metric baseline; valid where ``d`` exceeds ``min_disparity``, the match lies
  inside the right image and ``z`` is within [min_depth_m, max_depth_m].
* The left-IR frame is the native depth frame (the adapter checks ``depth_from_infrared_1`` = I): the IR depth is
  registered onto the colour grid with exactly the native-depth registration (``geometry.reproject_depth``,
  z-buffer forward warp with the same ``splat``, holes stay invalid), so B0 and B1 differ only in their source.
* A failed or missing model output is an error; native depth is never substituted.

Stage ``foundation_stereo_depth`` (B1) writes per colour stream ``depth``/``valid`` (colour grid) and the stage
summary (per-stream IR-valid / registered-valid fractions and the median left-right error from the kernel's per-frame
statistics, model identity, runner). Output layout ``OUTPUT_LAYOUT`` "2" (2026-11-06, user decision item 4): the
per-pixel diagnostics of layout "1" (``source_index`` colour grid; ``disparity_left``, ``lr_error``, ``ir_valid``
left-IR grid; ~9 MB per three-camera frame) are no longer persisted: no retained consumer reads them (every reader
goes through ``depth_stages.load_view_depth``: depth, valid, attrs). ``STAGE_VERSION`` is the COMPUTATION version and
the fingerprint key: depth and valid are computed exactly as before, so completed layout-1 outputs stay valid inputs
(downstream stages fingerprint the build id only) and are compacted in place by the operational repack, which records
``output_layout`` "2" and its provenance in the completion record.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio

STAGE_VERSION = "2"  # computation version (fingerprint key); unchanged by the layout change
OUTPUT_LAYOUT = "2"  # persisted datasets: depth, valid (layout "1" also stored the per-pixel diagnostics)
PERSISTED = ("depth", "valid")


def ir_rigs(meta) -> list:
    return [g for g in meta.ir_stereo if g.status == "metric" and any(r.rig_id == g.rig_id for r in meta.rgbd)]


def ir_stereo_depth(root: Path, episode_id: str, spec, runner, *, chunk: int = 8, valid_iters: int = 32,
                    hiera: int = 0, min_disparity: float = 0.5, min_depth_m: float = 0.1, max_depth_m: float = 6.0,
                    splat: int = 2, output_stage: str = "foundation_stereo_depth", force: bool = False) -> dict:
    """B1: FoundationStereo on every device IR pair (remote worker ``foundation_stereo`` task ``ir_pair_disparity``),
    disparity -> metric depth -> colour grid (``ir_disparity_to_color`` kernel)."""
    from s4d.realworld.baseline_stages import worker_identity
    from s4d.realworld.depth_stages import view_K
    from s4d.realworld.pipeline import _record_fp, load_meta, read_frames

    meta = load_meta(root, episode_id)
    rigs = ir_rigs(meta)
    if not rigs:
        raise RuntimeError("no metric device infrared pair in this episode (adapter include_ir and a recorder that "
                           "stores stereo/left|right are required); native depth is never substituted")
    config = {"chunk": chunk, "valid_iters": valid_iters, "hiera": hiera, "min_disparity": min_disparity,
              "min_depth_m": min_depth_m, "max_depth_m": max_depth_m, "splat": splat,
              "worker_identity": worker_identity(spec), "kernel": "ir_disparity_to_color"}
    inputs = {"ingest": _record_fp(root, episode_id, "ingest")}
    directory = rio.stage_dir(root, episode_id, output_stage)
    store = rio.ChunkStore(directory, rio.stage_fingerprint(output_stage, STAGE_VERSION, config, inputs))
    if force:
        store.clear()

    def build(tmp: Path) -> dict:
        summary, identity = {}, None
        for g in rigs:
            rg = meta.rig_rgbd(g.rig_id)
            cam = rg.color_stream
            cs, ls = meta.stream(cam), meta.stream(g.left_stream)
            K_c, K_ir = view_K(meta, cam), view_K(meta, g.left_stream)
            hw = (cs.intrinsics.height, cs.intrinsics.width)
            ir_hw = (ls.intrinsics.height, ls.intrinsics.width)
            n = len(meta.stream_timestamps_ns[g.left_stream])
            if n != len(meta.stream_timestamps_ns[cam]):
                raise ValueError(f"{g.rig_id}: {n} IR pairs vs {len(meta.stream_timestamps_ns[cam])} colour frames")
            stats = []
            with h5py.File(tmp / f"{rio.safe_name(cam)}.h5", "w") as f:
                f.attrs.update(K=K_c, K_ir=K_ir, stream_id=cam, rig_id=g.rig_id, baseline_m=g.baseline_m,
                               T_color_left=g.T_color(), grid="colour sensor pinhole grid (zero distortion)",
                               depth_convention="metric camera-z of the colour camera, 0 invalid",
                               source="FoundationStereo on the device IR pair (left reference), z = fx*B/d, "
                                      "registered by z-buffer forward warp exactly as the native depth",
                               disparity_convention="left-IR grid, d = x_left - x_right (pixels), right reference by "
                                                    "flip-and-swap; lr_error = |d_L(x) - d_R(x - d_L)|",
                               output_layout=OUTPUT_LAYOUT, persisted_datasets=json.dumps(list(PERSISTED)),
                               diagnostics="IR-grid disparity / left-right error / IR validity and the colour-grid "
                                           "source pixel are summarised per stream in the stage summary only",
                               ir_hw=list(ir_hw))
                dd = f.create_dataset("depth", (n, *hw), np.float32, chunks=(1, *hw), compression="lzf")
                vv = f.create_dataset("valid", (n, *hw), bool, chunks=(1, *hw), compression="lzf")
                for start in range(0, n, chunk):
                    idx = list(range(start, min(n, start + chunk)))
                    cid = f"{cam}/{start:05d}-{idx[-1]:05d}"
                    # layout-2 chunks carry depth only, under their own id: a layout-1 writer resuming the same build
                    # (same fingerprint) never reads them, and vice versa; both reuse the unchanged model-output chunk
                    got = store.get(f"{cid}/depth")
                    if got is None:
                        dgot = store.get(f"{cid}/disparity")
                        if dgot is None:  # model output saved durably before its ack (resumable)
                            left = read_frames(root, episode_id, g.left_stream, idx)
                            right = read_frames(root, episode_id, g.right_stream, idx)
                            resp, disp, handle = runner.worker(spec, "ir_pair_disparity",
                                                               {"valid_iters": valid_iters, "hiera": hiera},
                                                               {"left": left, "right": right}, chunk_id=f"fs/{cid}")
                            store.put(f"{cid}/disparity", disp, {"model": resp["model"], "meta": resp["meta"]})
                            runner.ack(handle)
                            dgot = store.get(f"{cid}/disparity")
                        disp, dmeta = dgot
                        resp = {"model": dmeta["model"], "meta": dmeta["meta"]}
                        out, kmeta, handle2 = runner.kernel(
                            "ir_disparity_to_color",
                            {"color_hw": list(hw), "min_disparity": min_disparity, "min_depth_m": min_depth_m,
                             "max_depth_m": max_depth_m, "splat": splat, "baseline_m": float(g.baseline_m)},
                            {"disparity_left": disp["disparity_left"], "disparity_right": disp["disparity_right"],
                             "K_ir": K_ir, "K_color": K_c, "T_color_left": g.T_color()}, chunk_id=f"fs2c/{cid}")
                        out = {"depth": out["depth"]}  # only what is persisted (the stats are in kmeta)
                        store.put(f"{cid}/depth", out, {**kmeta, "model": resp["model"], "worker_meta": resp["meta"]})
                        runner.ack(handle2)
                        got = (out, {**kmeta, "model": resp["model"], "worker_meta": resp["meta"]})
                    out, kmeta = got
                    sl = slice(idx[0], idx[-1] + 1)
                    dd[sl], vv[sl] = out["depth"], out["depth"] > 0
                    stats += kmeta["frames"]
                    identity = kmeta["model"]
            summary[cam] = {"frames": n, "ir_valid_fraction_mean": float(np.mean([s["ir_valid_fraction"] for s in stats])),
                            "registered_valid_fraction_mean": float(np.mean([s["registered_valid_fraction"]
                                                                             for s in stats])),
                            "lr_error_median_px": float(np.median([s["lr_error_median_px"] for s in stats])),
                            "ir_rig": g.rig_id, "baseline_m": g.baseline_m}
        summary["model"] = identity
        summary["runner"] = runner.describe()
        summary["output_layout"], summary["persisted_datasets"] = OUTPUT_LAYOUT, list(PERSISTED)
        return summary

    rec = rio.run_stage(directory, stage=output_stage, version=STAGE_VERSION, config=config, inputs=inputs,
                        build=build, force=force)
    store.cleanup()
    return rec

