"""Raw RealSense vs MapAnything depth, v2 (2026-11-05): common-support comparison, per region, with plain-language
metric definitions. Replaces nothing: the v1 ``depth_report.py`` output stays as historical evidence.

Both sources are evaluated with the same accepted calibration, the same synchronized sets, the same cameras and the
same settings. Direct comparisons use only pixels (and reprojected points) valid in BOTH sources; coverage and hole
filling are reported separately, and MapAnything-filled pixels are checked against the other cameras' raw
measurements. The per-set kernel (``depth_compare_v2``) runs in the inference container through the RAM-only stream
(chunk saved locally before ack); aggregation runs locally.

    .venv/bin/python -I ops/inference_server/depth_compare.py --pipeline configs/realworld/mug_hanging_rgbd.yaml \\
        --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000000

Writes ``<output_root>/reports/<episode>/depth_v2/{depth_compare.json, depth_compare.md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ops.inference_server.command_support import pool_config  # noqa: E402
from ops.inference_server.run_episode import local_head, spec  # noqa: E402
from ops.inference_server.session import RemoteRunner, Server  # noqa: E402
from ops.inference_server.stream_client import RemoteWorkerPool  # noqa: E402

REGIONS = ("scene", "mug", "fixture", "robot", "boundary")

GLOSSARY = {
    "coverage": ("Share of the region's pixels with a valid depth value (percent; higher = more pixels have depth). "
                 "Says nothing about whether the values are right: a model can fill every pixel with a wrong value."),
    "filled": ("Share of the region's pixels where the RealSense had no depth but MapAnything gives one (percent). "
               "These values are model estimates, not measurements; their consistency is checked separately below."),
    "ma_minus_raw": ("|MapAnything - RealSense| at pixels where both have depth (mm; lower = MapAnything stays closer "
                     "to the measurement). MapAnything received the RealSense depth as input, so small values are "
                     "partly by construction; it is a 'how much does the model change the measurement' number, not "
                     "an error of either source."),
    "crossview": ("Cross-view consistency (mm; lower is better): a pixel's 3D point from one camera, moved into a "
                  "second camera with the calibration, is compared with the depth that camera reports at the same "
                  "place (median |difference| over points; points hidden behind the other camera's surface by more "
                  "than 5 cm are excluded as occluded). Both sources are scored on the SAME points, valid in both "
                  "sources in both cameras. Limits: includes calibration error and the up-to-15 ms camera offset "
                  "for moving parts; agreement between cameras can also come from errors shared by both."),
    "crossview_within_10mm": "Share of those cross-view points within 1 cm (percent; higher is better).",
    "filled_vs_other_raw": ("For MapAnything-filled pixels only: the filled 3D point moved into another camera and "
                            "compared with that camera's RealSense MEASUREMENT (mm; lower is better). The reference "
                            "line gives the same number for measured RealSense points, i.e. what measured depth "
                            "achieves on the same camera pairs."),
    "robot": ("Robot-geometry agreement (mm; lower is better): measured depth minus the depth of the calibrated "
              "Panda + Robotiq model at the recorded joints, at pixels where the robot model is the front surface "
              "and both sources have depth. 'Independent' frames were not used to fit the calibration; 'fit' frames "
              "were (they favour the RealSense depth, which the calibration fitted). Limited by the model/mount "
              "accuracy (nominal, not measured) and calibration error."),
    "edge_to_rgb_px": ("Distance from depth discontinuities to the nearest colour edge (pixels; lower = depth edges sit "
                       "on image edges). Limits: not every colour edge is a depth edge (texture), and missing depth "
                       "near edges hides errors."),
    "edge_shift": ("Registration check: the pixel shift of the depth map that best aligns its edges with colour edges "
                   "(0,0 = no detectable RGB/depth misregistration)."),
    "temporal_std": ("Temporal spread on static background (mm; per-pixel standard deviation over the episode at grid "
                     "pixels away from the robot and the mug, valid in both sources in every frame). Lower = steadier, "
                     "but over-smoothing also lowers it: steadier is not more accurate."),
    "ma_over_raw": "Median MapAnything/RealSense depth ratio on common pixels (unit/scale check; 1.000 = same scale).",
}


def _med(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _pct(x):
    return None if x is None else round(100.0 * x, 1)


def _mm(x):
    return None if x is None else round(1000.0 * x, 1)


def aggregate(rows, grids, streams) -> dict:
    V = len(streams)
    views = [vw for r in rows for vw in r["views"]]
    cov = {reg: {k: _med([vw["coverage"][reg][k] for vw in views if vw["coverage"].get(reg)])
                 for k in ("raw", "ma", "both", "ma_only_filled", "raw_only_dropped_by_ma")} for reg in REGIONS}
    agree = {reg: {"median_abs_m": _med([vw["ma_minus_raw"][reg].get("median_abs") for vw in views]),
                   "mean_signed_m": _med([vw["ma_minus_raw"][reg].get("mean_signed") for vw in views])}
             for reg in REGIONS}
    cross = {}
    for reg in REGIONS:
        entries = [r["pairs"][p][reg] for r in rows for p in r["pairs"] if reg in r["pairs"][p]
                   and r["pairs"][p][reg]["raw"].get("n")]
        cross[reg] = {src: {"median_abs_m": _med([e[src]["median_abs"] for e in entries]),
                            "within_10mm": _med([e[src]["within_10mm"] for e in entries])} for src in ("raw", "ma")}
        cross[reg]["points_kept"] = int(sum(e["kept"] for e in entries))
        cross[reg]["pair_sets"] = len(entries)
    filled = [r["pairs"][p]["filled"] for r in rows for p in r["pairs"] if "filled" in r["pairs"][p]]
    ref = [r["pairs"][p]["raw_all_vs_other_raw"] for r in rows for p in r["pairs"]
           if r["pairs"][p].get("raw_all_vs_other_raw", {}).get("n")]
    fill = {"filled_vs_other_raw_median_abs_m": _med([f["vs_other_raw"].get("median_abs") for f in filled]),
            "filled_vs_other_raw_within_20mm": _med([f["vs_other_raw"].get("within_20mm") for f in filled]),
            "filled_vs_other_raw_points": int(sum(f["vs_other_raw"].get("n", 0) for f in filled)),
            "filled_vs_other_ma_median_abs_m": _med([f["vs_other_ma"].get("median_abs") for f in filled]),
            "reference_raw_vs_other_raw_median_abs_m": _med([x["median_abs"] for x in ref]),
            "reference_raw_vs_other_raw_within_20mm": _med([x["within_20mm"] for x in ref])}
    robot = {}
    for used, key in ((False, "independent_frames"), (True, "fit_frames")):
        sel = [vw for vw in views if vw["calib_used"] == used and vw["robot_common"]["raw"].get("n")]
        robot[key] = {src: _med([vw["robot_common"][src]["median_abs"] for vw in sel]) for src in ("raw", "ma")}
        robot[key]["view_frames"] = len(sel)
    edges = {src: _med([vw[src]["edge_to_rgb_px"].get("median_abs") for vw in views]) for src in ("raw", "ma")}
    flying = {src: _med([vw[src]["flying_pixel_fraction"] for vw in views]) for src in ("raw", "ma")}
    scans = {src: [tuple(vw[src]["edge_shift_scan"]["best_shift_px"]) for vw in views
                   if vw[src].get("edge_shift_scan", {}).get("best_shift_px") is not None] for src in ("raw", "ma")}
    shift = {src: {"frames": len(v), "best_shift_counts": {f"{k[0]},{k[1]}": v.count(k) for k in sorted(set(v))}}
             for src, v in scans.items()}
    conf = {"ma_minus_raw_high_conf_median_abs_m": _med([vw["conf"]["ma_minus_raw_high_conf"].get("median_abs")
                                                         for vw in views if "conf" in vw]),
            "ma_minus_raw_low_conf_median_abs_m": _med([vw["conf"]["ma_minus_raw_low_conf"].get("median_abs")
                                                        for vw in views if "conf" in vw]),
            "filled_median_conf": _med([vw["conf"].get("filled_median_conf") for vw in views if "conf" in vw]),
            "common_median_conf": _med([vw["conf"]["median_on_common"] for vw in views if "conf" in vw])}
    ratio = _med([vw["ma_over_raw_median"] for vw in views])
    stab = {}
    for v in range(V):
        raw, ma, static = grids["raw"][:, v], grids["ma"][:, v], grids["static"][:, v]
        keep = static.all(0) & np.isfinite(raw).all(0) & np.isfinite(ma).all(0)
        stab[streams[v]] = {"pixels": int(keep.sum()),
                            "raw_temporal_std_median_m": float(np.median(raw[:, keep].std(0))) if keep.any() else None,
                            "ma_temporal_std_median_m": float(np.median(ma[:, keep].std(0))) if keep.any() else None}
    return {"coverage": cov, "ma_minus_raw_common": agree, "crossview_common": cross, "hole_filling": fill,
            "robot_common": robot, "edge_to_rgb_median_px": edges, "flying_pixel_fraction": flying,
            "edge_shift_scan": shift, "confidence": conf, "ma_over_raw_median": ratio, "static_stability": stab}


def markdown(rep: dict) -> str:
    a = rep["aggregate"]
    L = ["# Raw RealSense vs MapAnything depth (v2: common support)", "",
         f"Episode `{rep['episode']}`, {rep['sets']} synchronized sets x {len(rep['views'])} cameras, calibration "
         f"`{rep['inputs']['calibrate'][:12]}`. No ground-truth depth exists: nothing here measures absolute accuracy.",
         "", "## Coverage (median over camera-frames, % of the region's pixels)", "",
         "| region | RealSense valid | MapAnything valid | both | filled by MA | dropped by MA |", "|---|---|---|---|---|---|"]
    for reg, c in a["coverage"].items():
        L.append(f"| {reg} | {_pct(c['raw'])} | {_pct(c['ma'])} | {_pct(c['both'])} | {_pct(c['ma_only_filled'])} | "
                 f"{_pct(c['raw_only_dropped_by_ma'])} |")
    L += ["", "## Cross-view consistency on common support (mm, median |difference|; lower is better)", "",
          "| region | RealSense | MapAnything | RealSense within 1 cm | MapAnything within 1 cm | points |",
          "|---|---|---|---|---|---|"]
    for reg, c in a["crossview_common"].items():
        L.append(f"| {reg} | {_mm(c['raw']['median_abs_m'])} | {_mm(c['ma']['median_abs_m'])} | "
                 f"{_pct(c['raw']['within_10mm'])} % | {_pct(c['ma']['within_10mm'])} % | {c['points_kept']} |")
    f = a["hole_filling"]
    L += ["", "## Hole filling, checked against the other cameras' RealSense measurements", "",
          f"- MapAnything-filled points vs another camera's measurement: median {_mm(f['filled_vs_other_raw_median_abs_m'])}"
          f" mm, {_pct(f['filled_vs_other_raw_within_20mm'])} % within 2 cm ({f['filled_vs_other_raw_points']} points).",
          f"- Reference, measured RealSense points vs another camera's measurement: median "
          f"{_mm(f['reference_raw_vs_other_raw_median_abs_m'])} mm, {_pct(f['reference_raw_vs_other_raw_within_20mm'])} %"
          " within 2 cm.", "",
          "## Robot geometry on common support (mm, median |measured - model|; lower is better)", "",
          "| frames | RealSense | MapAnything | camera-frames |", "|---|---|---|---|"]
    for key, r in a["robot_common"].items():
        L.append(f"| {key.replace('_', ' ')} | {_mm(r['raw'])} | {_mm(r['ma'])} | {r['view_frames']} |")
    L += ["", "## MapAnything vs RealSense on common pixels (mm)", "", "| region | median |MA - raw| | median signed |",
          "|---|---|---|"]
    for reg, c in a["ma_minus_raw_common"].items():
        L.append(f"| {reg} | {_mm(c['median_abs_m'])} | {_mm(c['mean_signed_m'])} |")
    st = a["static_stability"]
    L += ["", "## Other checks", "",
          f"- Depth edge to colour edge (px): RealSense {a['edge_to_rgb_median_px']['raw']}, MapAnything "
          f"{a['edge_to_rgb_median_px']['ma']}.",
          f"- Best registration shift (frames per shift): RealSense {a['edge_shift_scan']['raw']['best_shift_counts']}, "
          f"MapAnything {a['edge_shift_scan']['ma']['best_shift_counts']}.",
          f"- Flying-pixel fraction: RealSense {a['flying_pixel_fraction']['raw']}, MapAnything "
          f"{a['flying_pixel_fraction']['ma']}.",
          f"- Scale: median MA/raw ratio {a['ma_over_raw_median']}.",
          f"- Confidence: |MA - raw| high-confidence half {_mm(a['confidence']['ma_minus_raw_high_conf_median_abs_m'])}"
          f" mm, low-confidence half {_mm(a['confidence']['ma_minus_raw_low_conf_median_abs_m'])} mm; filled pixels' "
          f"median confidence {a['confidence']['filled_median_conf']} vs {a['confidence']['common_median_conf']} on "
          "common pixels.",
          "- Static temporal spread (mm, RealSense / MapAnything): " + ", ".join(
              f"{k[:12]} {_mm(v['raw_temporal_std_median_m'])} / {_mm(v['ma_temporal_std_median_m'])}"
              for k, v in st.items()), "", "## Convention checks", ""]
    for k, v in rep["convention_checks"].items():
        L.append(f"- {k}: {json.dumps(v)}")
    L += ["", "## What each number means", ""]
    for k, v in GLOSSARY.items():
        L.append(f"- **{k}**: {v}")
    return "\n".join(L) + "\n"


def convention_checks(root, ep, meta, streams, sets, T, geom) -> dict:
    """Registration/unit/convention facts from the stored outputs (local, no model calls)."""
    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.depth_stages import depth_file, set_robot_motion
    from s4d.realworld.geometry import rotation_angle_deg

    out = {}
    for v, s in enumerate(streams):
        intr = meta.stream(s).intrinsics
        with h5py.File(depth_file(root, ep, "mapanything_depth", s), "r") as f:
            K = np.asarray(f.attrs["K"], float)
            P = f["cam2world_pred"][:].astype(np.float64)
            scale = f["metric_scale"][:]
            hw = f["depth"].shape[1:]
        with h5py.File(depth_file(root, ep, "sensor_depth", s), "r") as f:
            hw_raw = f["depth"].shape[1:]
            K_raw = np.asarray(f.attrs["K"], float)
        c2w = np.linalg.inv(T[v])
        rot = [rotation_angle_deg(c2w[:3, :3], p[:3, :3]) for p in P]
        tr = [float(np.linalg.norm(c2w[:3, 3] - p[:3, 3])) for p in P]
        out[s] = {"grid_ma_raw_colour": [list(hw), list(hw_raw), [intr.height, intr.width]],
                  "K_ma_minus_colour_max": float(np.abs(K - intr.K()).max()),
                  "K_raw_minus_colour_max": float(np.abs(K_raw - intr.K()).max()),
                  "ma_pred_pose_vs_supplied_rot_deg_max": float(np.max(rot)),
                  "ma_pred_pose_vs_supplied_trans_m_max": float(np.max(tr)),
                  "metric_scale_range": [float(scale.min()), float(scale.max())]}
    spread = sets["spread_ns"] / 1e6
    motion = set_robot_motion(meta, sets["stamps_ns"], geom)
    finite = motion[np.isfinite(motion)]
    out["sync"] = {"set_time_spread_ms_median_max": [float(np.median(spread)), float(spread.max())],
                   "robot_motion_within_set_m_median_max": [float(np.median(finite)), float(finite.max())]
                   if len(finite) else None,
                   "unknown_motion_sets": int((~np.isfinite(motion)).sum())}
    rec = rio.read_record(rio.stage_dir(root, ep, "mapanything_depth"))
    out["mapanything_inputs"] = {k: rec["config"].get(k) for k in ("max_set_motion_m", "input_depth_stage")
                                 if k in rec["config"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--output-root")
    ap.add_argument("--chunk", type=int, default=6)
    args = ap.parse_args(argv)

    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.depth_stages import depth_file, frame_sets, load_view_depth
    from s4d.realworld.pipeline import RobotGeometry, accepted_calibration, load_meta, read_frames, robot_state_at

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    if args.output_root:
        cfg["output_root"] = args.output_root
    server = Server.load(Path(args.server))
    root, ep = Path(cfg["output_root"]), args.episode
    out_dir = root / "reports" / ep / "depth_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    head = local_head()
    server.verify_container()
    server.verify_source(head)
    audit = server.audit_start()
    meta = load_meta(root, ep)
    streams = [v.stream_id for v in meta.views()]
    sets = frame_sets(meta, streams)
    T = np.stack([accepted_calibration(root, ep, s) for s in streams])
    geom = RobotGeometry(meta)
    calib_frames = {}
    for s in streams:
        doc = json.loads((rio.stage_dir(root, ep, "calibrate") / f"{rio.safe_name(s)}.json").read_text())
        calib_frames[s] = {r["t_index"] for r in doc["diagnostics"].get("per_frame_fit", [])}
    sam = {s: rio.stage_dir(root, ep, "sam3_masks") / f"{rio.safe_name(s)}.h5" for s in streams}
    has_sam = all(p.is_file() for p in sam.values())
    pool = RemoteWorkerPool(pool_config(server.config, head), log_dir=out_dir / "worker_logs")
    runner = RemoteRunner(pool, spec(cfg, "s4d_kernels", expected_commit=head))
    config = {"chunk": args.chunk, "kernel": "depth_compare_v2", "occlusion_m": 0.05, "grid_step": 8,
              "boundary_px": 3, "front_tol_m": 0.05, "max_points": 20000, "scan_every": 8}
    inputs = {k: rio.build_id(rio.read_record(rio.stage_dir(root, ep, k))) for k in
              ("sensor_depth", "mapanything_depth", "calibrate") + (("sam3_masks",) if has_sam else ())}
    store = rio.ChunkStore(out_dir / "sets", rio.stage_fingerprint("depth_compare_v2", "1", config, inputs))
    n = len(sets["index"])
    rows, grids = [], {"raw": [], "ma": [], "static": []}
    try:
        for start in range(0, n, args.chunk):
            ks = list(range(start, min(n, start + args.chunk)))
            cid = f"sets/{start:05d}-{ks[-1]:05d}"
            got = store.get(cid)
            if got is None:
                idx = sets["index"][ks]
                arrays = {"images": np.stack([np.stack([read_frames(root, ep, s, [int(idx[j, v])])[0]
                                                        for v, s in enumerate(streams)]) for j in range(len(ks))]),
                          "K": np.stack([load_view_depth(root, ep, s, [0], "mapanything_depth")[2] for s in streams]),
                          "T_cam_base": T}
                for src, stage in (("raw", "sensor_depth"), ("ma", "mapanything_depth")):
                    d = [[load_view_depth(root, ep, s, [int(idx[j, v])], stage) for v, s in enumerate(streams)]
                         for j in range(len(ks))]
                    arrays[f"depth_{src}"] = np.stack([np.stack([x[0][0] for x in r]) for r in d]).astype(np.float32)
                    arrays[f"valid_{src}"] = np.stack([np.stack([x[1][0] for x in r]) for r in d])
                conf = []
                for j in range(len(ks)):
                    row = []
                    for v, s in enumerate(streams):
                        with h5py.File(depth_file(root, ep, "mapanything_depth", s), "r") as f:
                            row.append(f["conf"][int(idx[j, v])])
                    conf.append(np.stack(row))
                arrays["conf_ma"] = np.stack(conf).astype(np.float16)
                surf = np.zeros((len(ks), len(streams)) + geom.surface.points(geom.q(np.zeros(7), 0.0))[0].shape,
                                np.float32)
                for v, s in enumerate(streams):
                    stamps = meta.stream_timestamps_ns[s][idx[:, v]]
                    q, g, _, _ = robot_state_at(meta, stamps.astype(np.float64))
                    for j in range(len(ks)):
                        surf[j, v] = geom.surface.points(geom.q(q[j], g[j]))[0]
                arrays["surface_base"] = surf
                arrays["calib_used"] = np.array([[int(idx[j, v]) in calib_frames[s] for v, s in enumerate(streams)]
                                                 for j in range(len(ks))])
                if has_sam:
                    sem = []
                    for j in range(len(ks)):
                        row = []
                        for v, s in enumerate(streams):
                            with h5py.File(sam[s], "r") as f:
                                row.append(f["semantic"][int(idx[j, v])])
                        sem.append(np.stack(row))
                    arrays["semantic"] = np.stack(sem)
                out, kmeta, handle = runner.kernel("depth_compare_v2", {**config, "first_set": start}, arrays,
                                                   chunk_id=cid)
                store.put(cid, out, kmeta)
                runner.ack(handle)
                got = store.get(cid)
            out, kmeta = got
            rows += kmeta["sets"]
            for k in grids:
                grids[k].append(out[f"grid_{k}"])
            print(f"{cid} done", flush=True)
    finally:
        pool.close()
        audit_end = server.audit_end(audit)
    agg = aggregate(rows, {k: np.concatenate(v, 0) for k, v in grids.items()}, streams)
    rep = {"episode": ep, "views": streams, "sets": int(n), "source_commit": head, "inputs": inputs, "config": config,
           "aggregate": agg, "convention_checks": convention_checks(root, ep, meta, streams, sets, T, geom),
           "glossary": GLOSSARY, "persistence_audit": audit_end, "per_set": rows,
           "caveats": ["no ground-truth depth: consistency evidence only, never absolute accuracy",
                       "MapAnything received the RealSense depth as metric input: agreement with it is partly by "
                       "construction",
                       "robot agreement on calibration-fit frames favours RealSense (fitted); use independent frames",
                       "steadier/smoother depth is not evidence of accuracy",
                       "regions mug/fixture come from SAM 3 masks, robot from the calibrated URDF render"]}
    rio.atomic_write_json(out_dir / "depth_compare.json", json.loads(json.dumps(rep, default=str)))
    (out_dir / "depth_compare.md").write_text(markdown(rep))
    store.cleanup()
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
