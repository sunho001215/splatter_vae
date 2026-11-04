"""Command line for the real-world preprocessing pipeline.

    python -m s4d.realworld.cli validate --config configs/realworld/pipeline_example.yaml
    python -m s4d.realworld.cli run --config CONFIG --episode ID --stages ingest,stereo_rectify,mapanything_rig_depth
    python -m s4d.realworld.cli run --config CONFIG --episode ID --stages calibrate,mapanything_depth,track2d,lift3d,export
    python -m s4d.realworld.cli status --root OUTPUT_ROOT --episode ID
    optional baseline stages (not needed by our method): --stages sam3_masks,dino_snerl (after stereo_rectify / export)

Stages are idempotent: re-running skips complete stages whose fingerprint matches (``--force`` rebuilds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from s4d.realworld import io as rio

STAGES = ("ingest", "sensor_depth", "stereo_rectify", "mapanything_rig_depth", "calibrate",
          "mapanything_depth", "track2d", "lift3d", "export")
BASELINE_STAGES = ("sam3_masks", "dino_snerl")


class AnnotationLandmarkSource:
    """2D robot-landmark observations from a JSON file (manual clicks, marker detector or render-and-match output):
    {camera_id: {frame_index: {"uv": [[u, v] | null, ...], "conf": [...]}}} in continuous pixel coordinates of the
    rectified left image, landmarks ordered as ``landmark_links``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        self.identity = {"kind": "annotation_file", "path": str(self.path), "sha256": rio.sha256_file(self.path)}

    def __call__(self, camera_id, frame_index, links):
        entry = self.data.get(camera_id, {}).get(str(frame_index))
        if entry is None:
            return np.full((len(links), 2), np.nan), np.zeros(len(links))
        uv = np.array([[np.nan, np.nan] if p is None else p for p in entry["uv"]], float)
        return uv, np.asarray(entry.get("conf", np.ones(len(uv))), float)


def _spec(cfg: dict, name: str):
    from s4d.realworld.workers import WorkerSpec

    w = cfg["workers"][name]
    return WorkerSpec(name, w["python"], repo_dir=w.get("repo_dir"), expected_commit=w.get("expected_commit"),
                      checkpoint=w.get("checkpoint"), env=w.get("env", {}), timeout_s=float(w.get("timeout_s", 3600)))


def adapter_config(cfg: dict) -> dict:
    """Adapter config plus ``calibration_source`` = ``stages.calibrate.source``, so an adapter can tie robot geometry to
    the camera calibration actually used (e.g. DROID-3D joint offsets only with DROID-3D extrinsics). An explicit
    adapter value must agree with the stage."""
    acfg = dict(cfg["adapter"].get("config", {}) or {})
    stage = ((cfg.get("stages") or {}).get("calibrate") or {}).get("source")
    if "calibration_source" in acfg and acfg["calibration_source"] != stage:
        raise ValueError(f"adapter calibration_source {acfg['calibration_source']!r} != stages.calibrate.source {stage!r}")
    if stage is not None:
        acfg["calibration_source"] = stage
    return acfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s4d.realworld")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--config", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--episode", required=True)
    r.add_argument("--stages", default=",".join(STAGES))
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--root", required=True)
    s.add_argument("--episode", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        for stage in ("normalized",) + STAGES[1:-1] + BASELINE_STAGES:
            d = (rio.episode_dir(args.root, args.episode) / "normalized" if stage == "normalized"
                 else rio.stage_dir(args.root, args.episode, stage))
            rec = rio.read_record(d)
            print(f"{stage:13s} {'complete ' + rec['fingerprint'][:12] if rec else 'missing'}")
        return 0

    cfg = yaml.safe_load(Path(args.config).read_text())
    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.contract import validate_episode

    adapter = get_adapter(cfg["adapter"]["name"], adapter_config(cfg))
    if args.cmd == "validate":
        ok = True
        for ep in adapter.list_episodes():
            rep = validate_episode(adapter.read_meta(ep))
            ok &= rep.ok
            print(json.dumps({"episode": ep, "errors": rep.errors, "warnings": rep.warnings}, indent=1))
        return 0 if ok else 1

    from s4d.realworld.runners import LocalRunner

    root = Path(cfg["output_root"])
    runner = LocalRunner(root / "_worker_tmp" / args.episode)
    wanted = args.stages.split(",")
    unknown = set(wanted) - set(STAGES + BASELINE_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    def run(name, fn):
        if name in wanted:
            print(json.dumps({name: fn(args.force)["summary"]}, indent=1, default=str))

    run_stages(cfg, adapter, args.episode, runner, run, wanted)
    return 0


def run_stages(cfg, adapter, ep, runner, run, wanted, spec_factory=None):
    """Identical local/remote stage dispatch. ``run`` persists the caller's stage report and applies force policy."""
    from s4d.realworld import pipeline as P
    from s4d.realworld.baseline_stages import dino_snerl, sam3_masks
    from s4d.realworld.calibration import CalibrationConfig
    from s4d.realworld.depth_stages import MapAnythingBackend, mapanything_depth, mapanything_rig_depth, sensor_depth
    from s4d.realworld.export import export

    root, st = Path(cfg["output_root"]), cfg.get("stages", {})
    spec_factory = spec_factory or (lambda name: _spec(cfg, name))
    run("ingest", lambda f: P.ingest(adapter, ep, root, force=f))
    run("sensor_depth", lambda f: sensor_depth(root, ep, runner, force=f, **st.get("sensor_depth", {})))
    has_stereo = (bool(P.load_meta(root, ep).stereo)
                  if any(s in wanted for s in ("stereo_rectify", "mapanything_rig_depth")) else False)
    if has_stereo:
        run("stereo_rectify", lambda f: P.stereo_rectify(root, ep, runner=runner, force=f,
                                                        **st.get("stereo_rectify", {})))
    if any(stage in wanted for stage in ("mapanything_rig_depth", "mapanything_depth")):
        ma = MapAnythingBackend(spec_factory("mapanything"), runner, cfg["workers"]["mapanything"].get("params", {}))
        if has_stereo:
            run("mapanything_rig_depth", lambda f: mapanything_rig_depth(root, ep, ma, force=f,
                                                                        **st.get("mapanything_rig_depth", {})))
    cc = dict(st.get("calibrate", {}))

    def calibration(force):
        if cc.get("source") == "adapter_provided":
            return P.calibrate_provided(root, ep, adapter.provided_extrinsics(ep), force=force)
        options = {k: v for k, v in cc.items() if k not in ("source", "evidence", "landmark_source", "cfg")}
        options.setdefault("landmark_links", [])
        source = AnnotationLandmarkSource(cc["landmark_source"]) if cc.get("landmark_source") else None
        return P.calibrate(root, ep, source, cfg=CalibrationConfig(**{**cc.get("cfg", {}),
                                                                     "evidence": cc.get("evidence", "landmarks")}),
                           force=force, runner=runner, **options)

    s3 = st.get("sam3_masks", {})
    masks_first = cc.get("robot_mask_stage") == "sam3_masks"  # segmentation is calibration evidence: run it first

    def segment():
        meta = P.load_meta(root, ep)
        run("sam3_masks", lambda f: sam3_masks(root, ep, spec_factory("sam3"),
                                               cameras=s3.get("cameras") or [v.stream_id for v in meta.views()],
                                               prompts=s3["prompts"], mode=s3.get("mode", "segment_images"),
                                               chunk=s3.get("chunk", 16), score_threshold=s3.get("score_threshold", 0.5),
                                               force=f, runner=runner))

    if masks_first and "sam3_masks" in wanted:
        segment()
    run("calibrate", calibration)
    if "mapanything_depth" in wanted:
        run("mapanything_depth", lambda f: mapanything_depth(root, ep, ma, force=f, **st.get("mapanything_depth", {})))
    if "track2d" in wanted:
        tc = st.get("track2d", {})
        tracker = P.WorkerTrackerBackend(spec_factory("alltracker"), root / "_unused",
                                         cfg["workers"]["alltracker"].get("params"), runner=runner)
        run("track2d", lambda f: P.track2d(root, ep, tracker, window=tc.get("window", 16), step=tc.get("step", 8),
                                           grid_stride=tc.get("grid_stride", 8),
                                           tracker_hw=tuple(tc["tracker_hw"]) if tc.get("tracker_hw") else None,
                                           force=f))
    run("lift3d", lambda f: P.lift3d(root, ep, **st.get("lift3d", {}), force=f, runner=runner))
    ec = st.get("export", {})
    run("export", lambda f: export(root, ep, root / "export" / ep, strides=tuple(ec.get("strides", (1, 2))),
                                   allow_infrared_input=ec.get("allow_infrared_input", False),
                                   validation_tail_frames=ec.get("validation_tail_frames", 0),
                                   max_sync_ns=ec.get("max_sync_ns"), max_gap_factor=ec.get("max_gap_factor"),
                                   aspect=ec.get("aspect", "stretch"), force=f))
    if "sam3_masks" in wanted and not masks_first:
        segment()
    if "dino_snerl" in wanted:
        dc = st.get("dino_snerl", {})
        run("dino_snerl", lambda f: dino_snerl(root / "export" / ep, rio.stage_dir(root, ep, "dino_snerl"),
                                               spec_factory("dino_snerl"), backbone=dc.get("backbone", "scratch"),
                                               force=f, runner=runner))


if __name__ == "__main__":
    raise SystemExit(main())
