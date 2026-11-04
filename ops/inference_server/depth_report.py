"""Raw RealSense depth versus MapAnything depth on one processed RGB-D episode (consistency evidence, no ground truth).

Streams chunks of synchronized frame sets to the ``depth_set_diagnostics`` kernel in the inference container (same
runner/transport/ack discipline as the pipeline), then aggregates locally:
* cross-view reprojection consistency per camera pair (all pixels, SAM 3 mug and mug-tree regions);
* robot-geometry alignment against the calibrated URDF surface, split by frames the calibration fitted (favours raw
  depth by construction) versus frames it did not use;
* static-surface temporal spread on a fixed pixel grid (pixels static and valid in BOTH sources in every frame);
* flying pixels and depth-edge to RGB-edge distance;
* coverage (valid fraction, raw holes MapAnything fills);
* MapAnything WITHOUT pose input on a subset of sets: its relative camera poses versus the robot-grounded calibration
  (an independent check of the calibration; the calibration itself never uses MapAnything poses).

    .venv/bin/python -I ops/inference_server/depth_report.py --pipeline configs/realworld/mug_hanging_rgbd.yaml \\
        --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000000
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

from ops.inference_server.run_episode import local_head, spec  # noqa: E402
from ops.inference_server.session import RemoteRunner, Server  # noqa: E402
from ops.inference_server.stream_client import RemoteWorkerPool  # noqa: E402


def rel_pose_error(T_a, T_b, P_a, P_b) -> dict:
    """Relative pose a->b from the calibration (T = T_cam_base) vs predicted cam2world (P): rotation deg, translation."""
    from s4d.realworld.geometry import rotation_angle_deg

    calib = T_b @ np.linalg.inv(T_a)  # a-camera points -> b-camera points
    pred = np.linalg.inv(P_b) @ P_a
    return {"rot_deg": float(rotation_angle_deg(calib[:3, :3], pred[:3, :3])),
            "trans_m": float(np.linalg.norm(calib[:3, 3] - pred[:3, 3])),
            "baseline_calib_m": float(np.linalg.norm(calib[:3, 3]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--posefree-sets", type=int, default=12)
    args = ap.parse_args(argv)

    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.depth_stages import frame_sets, load_view_depth
    from s4d.realworld.pipeline import RobotGeometry, accepted_calibration, load_meta, read_frames, robot_state_at

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    server = Server.load(Path(args.server))
    root, ep = Path(cfg["output_root"]), args.episode
    out_dir = root / "reports" / ep / "depth"
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
    sam = {}
    for s in streams:
        p = rio.stage_dir(root, ep, "sam3_masks") / f"{rio.safe_name(s)}.h5"
        sam[s] = p if p.is_file() else None
    pool_cfg = {k: server.config[k] for k in ("ssh", "container", "workdir", "source_root", "retries", "backoff_s",
                                              "call_timeout_s", "hello_timeout_s", "workers", "env")}
    pool_cfg["expected_source_commit"] = head
    pool = RemoteWorkerPool(pool_cfg, log_dir=out_dir / "worker_logs")
    runner = RemoteRunner(pool, spec(cfg, "s4d_kernels", expected_commit=head))
    config = {"chunk": args.chunk, "kernel": "depth_set_diagnostics", "occlusion_m": 0.05, "grid_step": 8}
    inputs = {k: rio.build_id(rio.read_record(rio.stage_dir(root, ep, k))) for k in
              ("sensor_depth", "mapanything_depth", "calibrate") + (("sam3_masks",) if all(sam.values()) else ())}
    store = rio.ChunkStore(out_dir / "sets", rio.stage_fingerprint("depth_report", "1", config, inputs))
    n = len(sets["index"])
    rows, grids = [], {"raw": [], "ma": [], "static": []}
    try:
        for start in range(0, n, args.chunk):
            ks = list(range(start, min(n, start + args.chunk)))
            cid = f"sets/{start:05d}-{ks[-1]:05d}"
            got = store.get(cid)
            if got is None:
                idx = sets["index"][ks]  # (k, V)
                arrays = {"images": np.stack([np.stack([read_frames(root, ep, s, [int(idx[j, v])])[0]
                                                        for v, s in enumerate(streams)]) for j in range(len(ks))]),
                          "K": np.stack([load_view_depth(root, ep, s, [0], "mapanything_depth")[2] for s in streams]),
                          "T_cam_base": T}
                for src, stage in (("raw", "sensor_depth"), ("ma", "mapanything_depth")):
                    d = [[load_view_depth(root, ep, s, [int(idx[j, v])], stage) for v, s in enumerate(streams)]
                         for j in range(len(ks))]
                    arrays[f"depth_{src}"] = np.stack([np.stack([x[0][0] for x in r]) for r in d]).astype(np.float32)
                    arrays[f"valid_{src}"] = np.stack([np.stack([x[1][0] for x in r]) for r in d])
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
                if all(sam.values()):
                    sem = []
                    for j in range(len(ks)):
                        row = []
                        for v, s in enumerate(streams):
                            with h5py.File(sam[s], "r") as f:
                                row.append(f["semantic"][int(idx[j, v])])
                        sem.append(np.stack(row))
                    arrays["semantic"] = np.stack(sem)
                out, kmeta, handle = runner.kernel("depth_set_diagnostics", config, arrays, chunk_id=cid)
                store.put(cid, out, kmeta)
                runner.ack(handle)
                got = store.get(cid)
            out, kmeta = got
            rows += kmeta["sets"]
            for k in grids:
                grids[k].append(out[f"grid_{k}"])
        # MapAnything without pose input: independent relative poses vs the robot-grounded calibration
        posefree = []
        pick = np.linspace(0, n - 1, min(args.posefree_sets, n)).round().astype(int)
        mw = cfg["workers"]["mapanything"]
        ma_spec = spec(cfg, "mapanything")
        for k in pick:
            idx = sets["index"][k]
            images = np.stack([read_frames(root, ep, s, [int(idx[v])])[0] for v, s in enumerate(streams)])[None]
            depth = np.stack([load_view_depth(root, ep, s, [int(idx[v])], "sensor_depth")[0][0]
                              for v, s in enumerate(streams)])[None]
            Ks = np.stack([load_view_depth(root, ep, s, [0], "mapanything_depth")[2] for s in streams])
            resp, outp, handle = runner.worker(ma_spec, "infer_views", {**mw["params"], "use_pose": False},
                                               {"images": images, "K": Ks, "cam2world": np.stack([np.eye(4)] * len(streams)),
                                                "depth": depth.astype(np.float32)}, chunk_id=f"posefree/{k}")
            runner.ack(handle)
            P = outp["cam2world_pred"][0].astype(np.float64)
            posefree.append({"set": int(k), "pairs": {f"{a}->{b}": rel_pose_error(T[a], T[b], P[a], P[b])
                                                      for a in range(len(streams)) for b in range(len(streams)) if a < b}})
    finally:
        pool.close()
        audit_end = server.audit_end(audit)
    report = aggregate(rows, {k: np.concatenate(v, 0) for k, v in grids.items()}, streams, posefree)
    report.update({"episode": ep, "views": streams, "sets": int(n), "source_commit": head, "persistence_audit": audit_end,
                   "inputs": inputs, "remote_calls": len(runner.calls),
                   "caveats": [
                       "no ground-truth depth: every number is consistency evidence, not accuracy",
                       "MapAnything received the raw depth as metric input, so agreement with raw depth is partly by "
                       "construction; differences concentrate where raw depth is missing or noisy",
                       "robot alignment on calibration-fit frames favours raw depth (the calibration fitted it); the "
                       "not-fitted split is the fairer comparison, still bounded by URDF/mount accuracy",
                       "smoother depth is not evidence of accuracy (temporal spread also shrinks with over-smoothing)"]})
    rio.atomic_write_json(out_dir / "depth_report.json", report)
    contact_sheet(root, ep, streams, sets, out_dir / "depth_contact_sheet.png")
    store.cleanup()
    print(json.dumps(report["headline"], indent=1))
    return 0


def contact_sheet(root, ep, streams, sets, path, picks=(0.1, 0.5, 0.9), lo=0.4, hi=2.0) -> None:
    """Per camera and picked set: RGB | raw depth | MapAnything depth | |MA - raw| (0..5 cm) | MA confidence;
    depth colours on one fixed [lo, hi] m scale, invalid black (image composition only, local)."""
    import h5py
    from matplotlib import colormaps
    from PIL import Image

    from s4d.realworld.depth_stages import depth_file, load_view_depth
    from s4d.realworld.pipeline import read_frames

    turbo, mag = colormaps["turbo"], colormaps["magma"]

    def colour(x, valid, a, b, cmap):
        rgb = (cmap(np.clip((x - a) / (b - a), 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 0
        return rgb

    n = len(sets["index"])
    rows = []
    for frac in picks:
        k = int(round(frac * (n - 1)))
        for v, s in enumerate(streams):
            i = int(sets["index"][k, v])
            rgb = read_frames(root, ep, s, [i])[0]
            dr, vr, _ = load_view_depth(root, ep, s, [i], "sensor_depth")
            dm, vm, _ = load_view_depth(root, ep, s, [i], "mapanything_depth")
            with h5py.File(depth_file(root, ep, "mapanything_depth", s), "r") as f:
                conf = f["conf"][i].astype(np.float32)
            both = vr[0] & vm[0]
            tiles = [rgb, colour(dr[0], vr[0], lo, hi, turbo), colour(dm[0], vm[0], lo, hi, turbo),
                     colour(np.abs(dm[0] - dr[0]), both, 0.0, 0.05, mag),
                     colour(np.log(np.maximum(conf, 1e-6)), vm[0], 0.0, float(np.log(max(conf.max(), 1.0001))), mag)]
            rows.append(np.concatenate([t[::2, ::2] for t in tiles], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def _med(values):
    v = [x for x in values if x is not None and not isinstance(x, str)]
    return float(np.median(v)) if v else None


def aggregate(rows, grids, streams, posefree) -> dict:
    V = len(streams)
    head = {}
    for src in ("raw", "ma"):
        pairs = [r["pairs"][p][src] for r in rows for p in r["pairs"] if r["pairs"][p][src].get("n")]
        head[f"{src}_crossview_median_abs_m"] = _med([p["median_abs"] for p in pairs])
        head[f"{src}_crossview_within_20mm"] = _med([p["within_20mm"] for p in pairs])
        for region in ("object", "fixture"):
            rp = [r["pairs"][p][f"{src}_{region}"] for r in rows for p in r["pairs"]
                  if f"{src}_{region}" in r["pairs"][p] and r["pairs"][p][f"{src}_{region}"].get("n")]
            head[f"{src}_{region}_crossview_median_abs_m"] = _med([p["median_abs"] for p in rp])
        for used in (True, False):
            ra = [vw[src]["robot_alignment"] for r in rows for vw in r["views"]
                  if vw["calib_used"] == used and vw[src]["robot_alignment"].get("n")]
            head[f"{src}_robot_median_abs_m_{'calibfit' if used else 'notfit'}_frames"] = _med(
                [x["median_abs"] for x in ra])
        for key in ("valid_fraction", "flying_pixel_fraction", "object_valid_fraction", "fixture_valid_fraction"):
            head[f"{src}_{key}"] = _med([vw[src][key] for r in rows for vw in r["views"]])
        head[f"{src}_edge_to_rgb_median_px"] = _med([vw[src]["edge_to_rgb_px"].get("median_abs") for r in rows
                                                     for vw in r["views"]])
    head["raw_invalid_filled_by_ma"] = _med([vw["raw_invalid_filled_by_ma"] for r in rows for vw in r["views"]])
    # temporal spread on grid pixels static + valid in both sources in every frame
    stab = {}
    for v in range(V):
        raw, ma, static = grids["raw"][:, v], grids["ma"][:, v], grids["static"][:, v]
        keep = static.all(0) & np.isfinite(raw).all(0) & np.isfinite(ma).all(0)
        stab[streams[v]] = {"pixels": int(keep.sum()),
                            "raw_temporal_std_median_m": float(np.median(raw[:, keep].std(0))) if keep.any() else None,
                            "ma_temporal_std_median_m": float(np.median(ma[:, keep].std(0))) if keep.any() else None,
                            "ma_minus_raw_mean_median_m": float(np.median((ma[:, keep] - raw[:, keep]).mean(0)))
                            if keep.any() else None}
    head["static_temporal_std_median_m"] = {s: (x["raw_temporal_std_median_m"], x["ma_temporal_std_median_m"])
                                            for s, x in stab.items()}
    pf = [p for s in posefree for p in s["pairs"].values()]
    head["posefree_vs_calibration"] = {"rot_deg_median": _med([p["rot_deg"] for p in pf]),
                                       "trans_m_median": _med([p["trans_m"] for p in pf])}
    return {"headline": head, "static_stability": stab, "per_set": rows, "posefree": posefree}


if __name__ == "__main__":
    raise SystemExit(main())
