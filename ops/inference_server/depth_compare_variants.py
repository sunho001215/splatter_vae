"""Five-variant depth comparison (B0, B1, E1, E2, E3) of one episode with a frozen calibration.

No independent ground-truth depth exists for these recordings: every number is geometric CONSISTENCY or DIAGNOSTIC
evidence, never accuracy. Native depth, MapAnything output and the fitted robot model are never treated as ground
truth. The calibration was fitted with the native depth (B0) and SAM 3 robot masks, which favours B0 in the robot and
cross-view metrics; frames used by the calibration (fit + initialisation) are reported separately from independent
frames. Direct comparisons use pairwise common valid support (and a joint table on the support valid in all
sources); completeness is reported separately; pixels a method fills where native depth is missing are checked only
against the OTHER cameras' native measurements (an independent sensor view of the same surface, itself noisy).

    .venv/bin/python -I ops/inference_server/depth_compare_variants.py --pipeline configs/realworld/mug_hanging_ep1.yaml \\
        --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000001

Writes ``<base root>/reports/<episode>/depth_variants/{depth_variants.json, depth_variants.md}``.
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

SOURCES = {  # name -> (root relative to the base root's parent ("" = base), depth stage)
    "B0": ("base", "sensor_depth"), "B1": ("base", "foundation_stereo_depth"),
    "B1f": ("base", "foundation_stereo_filtered"),
    "E1": ("E1", "mapanything_depth"), "E2": ("E2", "mapanything_depth"), "E3": ("E3", "mapanything_depth")}
PAIRS = (("B0", "B1"), ("B0", "E1"), ("B1", "E2"), ("E1", "E2"), ("E2", "E3"), ("B1", "B1f"))
REGIONS = ("scene", "mug", "fixture", "robot", "gripper", "boundary")
EXPLAIN = {
    "coverage": "share of the region's pixels with depth (higher = more complete; says nothing about correctness)",
    "crossview": ("median |depth difference| when a point seen by one camera is moved with the frozen calibration into "
                  "another camera and compared with that camera's depth from the SAME method (mm, lower = more "
                  "consistent; on points valid for both compared methods in both cameras; occluded points excluded; "
                  "includes calibration error and the up-to-23 ms inter-camera offset for moving parts)"),
    "robot": ("median |measured - model| where the calibrated Panda + Robotiq model is the front surface (mm, lower = "
              "closer to the robot model; independent frames only; bounded by model/mount accuracy; favours B0, "
              "whose depth fitted the calibration)"),
    "filled": ("pixels a method has where native depth is missing, moved into another camera and compared with that "
               "camera's NATIVE measurement (mm; reference line: native vs native)"),
    "edges": "distance from depth discontinuities to the nearest colour edge (px, lower = sharper/better registered)",
    "flying": "share of mixed-depth pixels between neighbours (lower = cleaner boundaries), on the joint support",
    "temporal": ("per-pixel std over time on static background (mm); low = steady, but smoothing also lowers it: "
                 "steadiness is not accuracy"),
}


def _med(xs):
    v = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def verify_calibration_inputs(base, episode, roots, stages):
    """Reject saved outputs made with a previous calibration, even if current symlinked matrices agree.

    MapAnything depth and every compared variant's lift3d must directly name the frozen build. Sensor/IR stereo
    depth may be calibration-independent; follow their recorded dependencies to ensure that exception is true.
    """
    from s4d.realworld import io as rio

    base_record = rio.read_record(rio.stage_dir(base, episode, "calibrate"))
    if base_record is None:
        raise SystemExit("base calibrate record is missing")
    frozen = rio.build_id(base_record)

    def require(root, stage, independent=False, seen=None):
        seen = set() if seen is None else seen
        if stage in seen:
            return
        seen.add(stage)
        directory = (rio.episode_dir(root, episode) / "normalized" if stage == "ingest" else
                     rio.stage_dir(root, episode, stage))
        rec = rio.read_record(directory)
        if rec is None:
            raise SystemExit(f"{root}: missing {stage} record; rebuild before comparing")
        inputs = rec.get("inputs", {})
        if "calibrate" in inputs:
            if inputs["calibrate"] != frozen:
                raise SystemExit(f"{root}: stale {stage}: calibrate input is not base build {frozen}; rebuild")
        elif not independent:
            raise SystemExit(f"{root}: stale {stage}: missing base calibrate build {frozen} in inputs; rebuild")
        for parent, build in inputs.items():
            if parent == "calibrate":
                continue
            parent_dir = (rio.episode_dir(root, episode) / "normalized" if parent == "ingest" else
                          rio.stage_dir(root, episode, parent))
            parent_rec = rio.read_record(parent_dir)
            if parent_rec is not None:
                if rio.build_id(parent_rec) != build:
                    raise SystemExit(f"{root}: stale {stage}: input {parent} no longer matches its record; rebuild")
                require(root, parent, independent=True, seen=seen)

    for name, root in roots.items():
        require(root, stages[name], independent=stages[name] in {
            "sensor_depth", "foundation_stereo_depth", "foundation_stereo_filtered"})
        # B0/B1 depths live in the base; their corresponding tracker outputs live in B0/B1 variant roots.
        variant_root = base.parent / name if SOURCES[name][0] == "base" and name in ("B0", "B1") else root
        if variant_root != base:
            require(variant_root, "lift3d")
    return frozen


def aggregate(rows, grids, names, pairs) -> dict:
    views = [vw for r in rows for vw in r["views"]]
    cov = {reg: {n: _med([vw["coverage"][reg][n] for vw in views if reg in vw["coverage"]]) for n in names + ["joint"]}
           for reg in REGIONS}
    cross = {}
    for key in [f"{a}|{b}" for a, b in pairs] + ["joint"]:
        members = key.split("|") if key != "joint" else names
        cross[key] = {}
        for reg in REGIONS:
            ents = [r["pairs"][p][f"{key}|{reg}"] for r in rows for p in r["pairs"] if f"{key}|{reg}" in r["pairs"][p]]
            ents = [e for e in ents if e["kept"] > 20]
            cross[key][reg] = {"kept": int(sum(e["kept"] for e in ents)), **{
                n: {"median_abs_mm": None if _med([e[n].get("median_abs") for e in ents]) is None else
                    round(1000 * _med([e[n].get("median_abs") for e in ents]), 2),
                    "within_10mm": _med([e[n].get("within_10mm") for e in ents])} for n in members}}
    robot = {}
    for a, b in pairs:
        for rname in ("robot", "gripper"):
            key = f"{a}|{b}|{rname}"
            for split, used in (("independent", False), ("calibration_frames", True)):
                sel = [vw["robot"][key] for vw in views if vw["calib_used"] == used and key in vw["robot"]
                       and vw["robot"][key][a].get("n", 0) > 20]
                robot[f"{key}|{split}"] = {n: _med([x[n]["median_abs"] for x in sel]) for n in (a, b)}
                robot[f"{key}|{split}"]["view_frames"] = len(sel)
    filled = {}
    for n in names:
        ents = [r["pairs"][p][f"filled|{n}"] for r in rows for p in r["pairs"] if f"filled|{n}" in r["pairs"][p]]
        filled[n] = {"median_abs_mm": None if _med([e["vs_other_reference"].get("median_abs") for e in ents]) is None
                     else round(1000 * _med([e["vs_other_reference"].get("median_abs") for e in ents]), 2),
                     "within_20mm": _med([e["vs_other_reference"].get("within_20mm") for e in ents]),
                     "points": int(sum(e["vs_other_reference"].get("n", 0) for e in ents))}
    refself = [r["pairs"][p][k] for r in rows for p in r["pairs"] for k in r["pairs"][p] if k.startswith("reference_self")]
    filled["reference_native_vs_native_mm"] = (None if not refself else
                                               round(1000 * _med([x["median_abs"] for x in refself]), 2))
    edges = {n: _med([vw["edges"][n].get("median_abs") for vw in views]) for n in names}
    flying = {n: _med([vw["flying_common"][n] for vw in views]) for n in names}
    stab = {}
    V = grids["static"].shape[1]
    for v in range(V):
        keep = grids["static"][:, v].all(0) & np.logical_and.reduce([np.isfinite(grids[n][:, v]).all(0) for n in names])
        stab[f"view{v}"] = {"pixels": int(keep.sum()),
                            **{n: (round(1000 * float(np.median(grids[n][:, v][:, keep].std(0))), 2) if keep.any()
                                   else None) for n in names}}
    return {"coverage": cov, "crossview": cross, "robot": robot, "filled": filled, "edges_px": edges,
            "flying_common": flying, "temporal_static_std_mm": stab}


def markdown(rep: dict) -> str:
    a, names = rep["aggregate"], rep["sources"]
    L = [f"# Depth variants of {rep['episode']} (consistency/diagnostic evidence; no ground truth)", "",
         "## Coverage (median over camera-frames, % of region pixels)", "",
         "| region | " + " | ".join(names) + " | all |", "|---" * (len(names) + 2) + "|"]
    for reg, c in a["coverage"].items():
        L.append(f"| {reg} | " + " | ".join("-" if c[n] is None else f"{100 * c[n]:.1f}" for n in names + ["joint"])
                 + " |")
    L += ["", "## Pairwise cross-view consistency on common support (median |d| mm; lower = more consistent)", "",
          "| pair | " + " | ".join(REGIONS) + " |", "|---" * (len(REGIONS) + 1) + "|"]
    for key, regs in a["crossview"].items():
        if key == "joint":
            continue
        x, y = key.split("|")
        L.append(f"| {x} vs {y} | " + " | ".join(
            "-" if regs[r][x]["median_abs_mm"] is None else f"{regs[r][x]['median_abs_mm']} / {regs[r][y]['median_abs_mm']}"
            for r in REGIONS) + " |")
    L += ["", "## Joint common support (all sources valid; median |d| mm)", "",
          "| region | " + " | ".join(names) + " | points |", "|---" * (len(names) + 2) + "|"]
    for reg, c in a["crossview"]["joint"].items():
        L.append(f"| {reg} | " + " | ".join("-" if c[n]["median_abs_mm"] is None else str(c[n]["median_abs_mm"])
                                             for n in names) + f" | {c['kept']} |")
    L += ["", "## Robot / gripper model agreement, independent frames (median |d| mm)", ""]
    for key, v in a["robot"].items():
        if key.endswith("independent"):
            L.append(f"- {key.replace('|', ' ')}: " + ", ".join(f"{k} {'-' if v[k] is None else round(1000 * v[k], 1)}"
                                                                  for k in v if k != "view_frames")
                     + f" ({v['view_frames']} camera-frames)")
    L += ["", "## Filled pixels vs the other cameras' native depth", ""]
    for n, v in a["filled"].items():
        L.append(f"- {n}: {v}")
    L += ["", f"Edge-to-RGB median px: {a['edges_px']}", f"Flying pixels (joint support): {a['flying_common']}",
          f"Static temporal std (mm): {a['temporal_static_std_mm']}", "", "## Definitions", ""]
    L += [f"- **{k}**: {v}" for k, v in EXPLAIN.items()]
    L += ["", "Caveats: " + "; ".join(rep["caveats"])]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args(argv)

    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.calibration import evidence_frames
    from s4d.realworld.depth_stages import frame_sets, load_view_depth, sets_policy
    from s4d.realworld.pipeline import RobotGeometry, accepted_calibration, load_meta, read_frames, robot_state_at

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    base = Path(cfg["output_root"])
    ep = args.episode
    names = args.sources.split(",")
    roots = {n: (base if SOURCES[n][0] == "base" else base.parent / SOURCES[n][0]) for n in names}
    stages = {n: SOURCES[n][1] for n in names}
    pairs = [p for p in PAIRS if p[0] in names and p[1] in names]
    frozen = verify_calibration_inputs(base, ep, roots, stages)
    server = Server.load(Path(args.server))
    out_dir = base / "reports" / ep / "depth_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    head = local_head()
    server.verify_container()
    server.verify_source(head)
    audit = server.audit_start()
    meta = load_meta(base, ep)
    streams = [v.stream_id for v in meta.views()]
    ma_root = next((roots[n] for n in names if stages[n] == "mapanything_depth"), None)
    sets = frame_sets(meta, streams, **(sets_policy(ma_root, ep) if ma_root else {}))
    T = np.stack([accepted_calibration(base, ep, s) for s in streams])
    for n in names:  # every variant must use the identical frozen calibration
        for v, s in enumerate(streams):
            if not np.array_equal(accepted_calibration(roots[n], ep, s), T[v]):
                raise SystemExit(f"{n}: calibration of {s} differs from the base (not frozen)")
    geom = RobotGeometry(meta)
    links = sorted(geom.surface.link_points)
    gl = set(geom.gripper_links())
    gripper_points = np.concatenate([np.full(len(geom.surface.link_points[k]), k in gl) for k in links])
    calib = {s: evidence_frames(json.loads((rio.stage_dir(base, ep, "calibrate") / f"{rio.safe_name(s)}.json")
                                           .read_text())) for s in streams}
    sam = {s: rio.stage_dir(base, ep, "sam3_masks") / f"{rio.safe_name(s)}.h5" for s in streams}
    has_sam = all(p.is_file() for p in sam.values())
    pool = RemoteWorkerPool(pool_config(server.config, head), log_dir=out_dir / "worker_logs")
    runner = RemoteRunner(pool, spec(cfg, "s4d_kernels", expected_commit=head))
    config = {"kernel": "depth_compare_multi", "sources": names, "pairs": [list(p) for p in pairs],
              "reference_source": "B0" if "B0" in names else None, "occlusion_m": 0.05, "boundary_px": 3,
              "front_tol_m": 0.05, "max_points": 15000, "grid_step": 8}
    inputs = {n: rio.build_id(rio.read_record(rio.stage_dir(roots[n], ep, stages[n]))) for n in names}
    inputs["calibrate"] = frozen
    store = rio.ChunkStore(out_dir / "sets", rio.stage_fingerprint("depth_compare_multi", "1", config, inputs))
    n_sets = len(sets["index"])
    rows, grids = [], {k: [] for k in names + ["static"]}
    try:
        for start in range(0, n_sets, args.chunk):
            ks = list(range(start, min(n_sets, start + args.chunk)))
            cid = f"sets/{start:05d}-{ks[-1]:05d}"
            got = store.get(cid)
            if got is None:
                idx = sets["index"][ks]
                arrays = {"images": np.stack([np.stack([read_frames(base, ep, s, [int(idx[j, v])])[0]
                                                        for v, s in enumerate(streams)]) for j in range(len(ks))]),
                          "K": np.stack([load_view_depth(base, ep, s, [0], "sensor_depth")[2] for s in streams]),
                          "T_cam_base": T, "gripper_points": gripper_points}
                for n in names:
                    d = [[load_view_depth(roots[n], ep, s, [int(idx[j, v])], stages[n]) for v, s in enumerate(streams)]
                         for j in range(len(ks))]
                    arrays[f"depth_{n}"] = np.stack([np.stack([x[0][0] for x in r]) for r in d]).astype(np.float32)
                    arrays[f"valid_{n}"] = np.stack([np.stack([x[1][0] for x in r]) for r in d])
                surf = []
                for j in range(len(ks)):
                    row = []
                    for v, s in enumerate(streams):
                        q, g, _, _ = robot_state_at(meta, meta.stream_timestamps_ns[s][[int(idx[j, v])]].astype(float))
                        row.append(geom.surface.points(geom.q(q[0], g[0]))[0])
                    surf.append(np.stack(row))
                arrays["surface_base"] = np.stack(surf).astype(np.float32)
                arrays["calib_used"] = np.array([[int(idx[j, v]) in calib[s][0] for v, s in enumerate(streams)]
                                                 for j in range(len(ks))])
                if has_sam:
                    sem = {}
                    for v, s in enumerate(streams):
                        with h5py.File(sam[s], "r") as f:
                            sem[v] = [f["semantic"][int(idx[j, v])] for j in range(len(ks))]
                    arrays["semantic"] = np.stack([np.stack([sem[v][j] for v in range(len(streams))])
                                                   for j in range(len(ks))])
                out, kmeta, handle = runner.kernel("depth_compare_multi", {**config, "first_set": start}, arrays,
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
    agg = aggregate(rows, {k: np.concatenate(v, 0) for k, v in grids.items()}, names, pairs)
    rep = {"episode": ep, "sources": names, "source_roots": {n: [str(roots[n]), stages[n]] for n in names},
           "pairs": pairs, "sets": int(n_sets), "unmatched_frames": sets.get("unmatched"), "source_commit": head,
           "inputs": inputs, "config": config, "aggregate": agg, "explain": EXPLAIN,
           "calibration_evidence_frames": {s: {"count": len(v[0]), "how": v[1]} for s, v in calib.items()},
           "persistence_audit": audit_end, "per_set": rows,
           "caveats": ["no ground-truth depth: consistency/diagnostic evidence only",
                       "the frozen calibration was fitted with native depth (B0) + SAM 3 masks: robot and cross-view "
                       "metrics favour B0; independent frames reported separately",
                       "MapAnything (E1-E3) receives the depth it is compared with as input",
                       "steadier or smoother depth is not evidence of accuracy",
                       "the E3 left-right threshold (1 px) was fixed a priori (no tuning on this episode)"]}
    rio.atomic_write_json(out_dir / "depth_variants.json", json.loads(json.dumps(rep, default=str)))
    (out_dir / "depth_variants.md").write_text(markdown(rep))
    store.cleanup()
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
