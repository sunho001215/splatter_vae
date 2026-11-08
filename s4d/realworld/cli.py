"""Command line for the real-world preprocessing pipeline (own rig: Franka Panda + Robotiq + RealSense RGB-D rigs).

    python -m s4d.realworld.cli validate --config configs/realworld/mug_hanging.yaml
    python -m s4d.realworld.cli run --config CONFIG --episode ID [--stages a,b]
    python -m s4d.realworld.cli status --root OUTPUT_ROOT --episode ID

Standard stage graph (configs/realworld/mug_hanging.yaml, dispatched by ``run_stages``): ingest, sensor_depth
(calibration evidence), foundation_stereo_depth (device IR pair -> colour grid), sam3_masks, calibrate,
track_everything (per camera, writes the track2d + lift3d records), track_motion, train_data (the compact training
store, ``s4d.realworld.train_store``; output ``<output_root>/train_data/<episode>``). The former window export
(``export``) runs only when configured; the ``dino_snerl`` cache stage is retired (SNeRL DINO targets are online).

The depth model and the tracker run only when configured under ``stages`` (no default depth model or tracker; a
failure is an error, never a switch to another backend). Stages are idempotent: re-running skips complete stages whose
fingerprint matches (``--force`` rebuilds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from s4d.realworld import io as rio

STAGES = ("ingest", "sensor_depth", "calibrate", "track_motion", "train_data", "vis_geometry", "export",
          "foundation_stereo_depth", "s2m2_depth",
          "track_everything")  # optional stages last (dispatch order is in code)
BASELINE_STAGES = ("sam3_masks",)
# run only when configured under ``stages``: the depth model, the tracker (no default backend), the motion filter, the
# training store (it needs the task name), the viewer's native depth store and the former window export (legacy,
# window-level copies)
OPTIONAL_STAGES = ("track_motion", "train_data", "vis_geometry", "export", "foundation_stereo_depth",
                   "s2m2_depth", "track_everything")
# stage directories with a completion record (track_everything writes the track2d and lift3d records)
RECORD_STAGES = ("sensor_depth", "foundation_stereo_depth", "s2m2_depth", "calibrate", "track2d", "lift3d", "track_motion")
# model worker each stage calls (weights are staged only for the workers of the requested stages)
STAGE_WORKERS = {"foundation_stereo_depth": "foundation_stereo", "s2m2_depth": "s2m2",
                 "track_everything": "trackeverything",
                 "sam3_masks": "sam3"}


def stage_workers(stages) -> set:
    return {STAGE_WORKERS[s] for s in stages if s in STAGE_WORKERS}


def save_root_config(cfg: dict, source: str) -> Path:
    """The run's pipeline config as ``<output_root>/pipeline.yaml`` (frame-set policy for viewers and contract checks,
    ``depth_stages.configured_sets_policy``); the latest run's config is kept."""
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline.yaml"
    rio.atomic_write_bytes(path, (f"# resolved pipeline config of the latest run (source: {source})\n"
                                  + yaml.safe_dump(cfg, sort_keys=False)).encode())
    return path


def default_stages(cfg: dict) -> list[str]:
    """Stages a config runs by default: every pipeline stage (optional ones only when configured) plus exactly the
    baseline stages its declared baselines / calibration need (``required_baseline_stages``)."""
    from s4d.realworld.baseline_stages import required_baseline_stages

    st = cfg.get("stages") or {}
    return [s for s in STAGES if s not in OPTIONAL_STAGES or s in st] + list(required_baseline_stages(cfg))


def check_stage_scope(cfg: dict, wanted) -> None:
    """Refuse configured stages this pipeline does not have, baseline stages the config does not need and optional
    stages without a configuration."""
    from s4d.realworld.baseline_stages import required_baseline_stages

    unknown = sorted(set(cfg.get("stages") or {}) - set(STAGES + BASELINE_STAGES))
    if unknown:
        raise ValueError(f"configured stages {unknown} are not stages of this pipeline (known: "
                         f"{list(STAGES + BASELINE_STAGES)})")
    if "source" in ((cfg.get("stages") or {}).get("calibrate") or {}):
        raise ValueError("stages.calibrate.source is not supported: the calibration is robot-grounded (externally "
                         "provided extrinsics are not a calibration route of this pipeline)")
    extra = sorted(set(wanted) & set(BASELINE_STAGES) - set(required_baseline_stages(cfg)))
    if extra:
        raise ValueError(f"baseline stages {extra} are not needed by this config's declared baselines "
                         f"{cfg.get('baselines') or []} (and not by its calibration); declare the baseline first")
    missing = sorted(s for s in set(wanted) & set(OPTIONAL_STAGES + BASELINE_STAGES) if s not in (cfg.get("stages") or {}))
    if missing:
        raise ValueError(f"stages {missing} are requested but not configured under 'stages'")


class AnnotationLandmarkSource:
    """2D robot-landmark observations from a JSON file (manual clicks, marker detector or render-and-match output):
    {camera_id: {frame_index: {"uv": [[u, v] | null, ...], "conf": [...]}}} in continuous pixel coordinates of the
    camera view's grid, landmarks ordered as ``landmark_links``."""

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
    """The adapter's config section (a copy)."""
    return dict(cfg["adapter"].get("config", {}) or {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s4d.realworld")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--config", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--episode", required=True)
    r.add_argument("--stages", help="comma-separated; default: default_stages(config)")
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--root", required=True)
    s.add_argument("--episode", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        for stage in ("normalized",) + RECORD_STAGES + ("train_data", "vis_geometry", "export") + BASELINE_STAGES:
            d = (rio.episode_dir(args.root, args.episode) / "normalized" if stage == "normalized"
                 else Path(args.root) / stage / args.episode if stage in ("train_data", "vis_geometry", "export")
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
    wanted = args.stages.split(",") if args.stages else default_stages(cfg)
    unknown = set(wanted) - set(STAGES + BASELINE_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")
    try:
        check_stage_scope(cfg, wanted)
    except ValueError as exc:
        ap.error(str(exc))
    runner = LocalRunner(root / "_worker_tmp" / args.episode)
    save_root_config(cfg, args.config)

    def run(name, fn):
        if name in wanted:
            print(json.dumps({name: fn(args.force)["summary"]}, indent=1, default=str))

    run_stages(cfg, adapter, args.episode, runner, run, wanted)
    return 0


def run_stages(cfg, adapter, ep, runner, run, wanted, spec_factory=None):
    """Identical local/remote stage dispatch. ``run`` persists the caller's stage report and applies force policy."""
    from s4d.realworld import pipeline as P
    from s4d.realworld.baseline_stages import sam3_masks
    from s4d.realworld.calibration import CalibrationConfig
    from s4d.realworld.depth_stages import sensor_depth
    from s4d.realworld.export import export

    root, st = Path(cfg["output_root"]), cfg.get("stages", {})
    spec_factory = spec_factory or (lambda name: _spec(cfg, name))
    run("ingest", lambda f: P.ingest(adapter, ep, root, force=f))
    run("sensor_depth", lambda f: sensor_depth(root, ep, runner, force=f, **st.get("sensor_depth", {})))
    if "foundation_stereo_depth" in wanted:  # device-IR stereo depth (B1): the own-rig lifting/export depth
        from s4d.realworld.ir_stereo_depth import ir_stereo_depth

        run("foundation_stereo_depth", lambda f: ir_stereo_depth(root, ep, spec_factory("foundation_stereo"), runner,
                                                                 force=f, **st.get("foundation_stereo_depth", {})))
    if "s2m2_depth" in wanted:  # candidate IR stereo depth (S2M2; study 2026-11-08), same contract as B1
        from s4d.realworld.ir_stereo_depth import s2m2_ir_depth

        run("s2m2_depth", lambda f: s2m2_ir_depth(root, ep, spec_factory("s2m2"), runner, force=f,
                                                  **st.get("s2m2_depth", {})))
    cc = dict(st.get("calibrate", {}))

    def calibration(force):
        options = {k: v for k, v in cc.items() if k not in ("evidence", "landmark_source", "cfg")}
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
                                               cameras=s3.get("cameras") or [v.stream_id for v in P.camera_views(meta)],
                                               prompts=s3["prompts"], mode=s3.get("mode", "segment_images"),
                                               chunk=s3.get("chunk", 16), score_threshold=s3.get("score_threshold", 0.5),
                                               force=f, runner=runner))

    if masks_first and "sam3_masks" in wanted:
        segment()
    run("calibrate", calibration)
    if "track_everything" in wanted:  # single-view TrackEverything per camera writes the track2d + lift3d records
        from s4d.realworld.trackeverything_stage import track_everything

        tc = dict(st.get("track_everything") or {})
        if "model_hw" in tc:
            tc["model_hw"] = tuple(tc["model_hw"])

        def te(force):
            recs = track_everything(root, ep, spec_factory("trackeverything"), runner, force=force, **tc)
            return {"skipped": all(r.get("skipped") for r in recs.values()),
                    "summary": {k: r.get("summary") for k, r in recs.items()}}

        run("track_everything", te)
    if "track_motion" in wanted:
        from s4d.realworld.track_motion import MotionFilterConfig, track_motion

        mc = dict(st.get("track_motion") or {})
        if "depth_bins_m" in mc:
            mc["depth_bins_m"] = tuple(mc["depth_bins_m"])
        run("track_motion", lambda f: track_motion(root, ep, cfg=MotionFilterConfig(**mc), force=f))
    if "sam3_masks" in wanted and not masks_first:  # the training store carries the semantics: segment first
        segment()
    if "train_data" in wanted:
        from s4d.realworld.train_store import write_episode

        sc = dict(st.get("train_data") or {})
        task = sc.pop("task", None) or cfg.get("task")  # default: the recording's dataset folder (adapter identity)
        sc.setdefault("semantic", "snerl" in (cfg.get("baselines") or []))  # SNeRL's semantic targets
        run("train_data", lambda f: write_episode(root, ep, root / "train_data" / ep, task=task, force=f, **sc))
    if "vis_geometry" in wanted:  # viewer: native-resolution depth, compact (s4d.realworld.vis_store)
        from s4d.realworld.vis_store import write_native_geometry

        run("vis_geometry", lambda f: write_native_geometry(root, ep, root / "vis_geometry" / ep, force=f,
                                                            **(st.get("vis_geometry") or {})))
    ec = st.get("export", {})
    run("export", lambda f: export(root, ep, root / "export" / ep, strides=tuple(ec.get("strides", (1, 2))),
                                   allow_infrared_input=ec.get("allow_infrared_input", False),
                                   validation_tail_frames=ec.get("validation_tail_frames", 0),
                                   max_sync_ns=ec.get("max_sync_ns"), max_gap_factor=ec.get("max_gap_factor"),
                                   aspect=ec.get("aspect", "stretch"),
                                   track_source=ec.get("track_source", "lift3d"),
                                   depth_stage=ec.get("depth_stage"), force=f))


if __name__ == "__main__":
    raise SystemExit(main())
