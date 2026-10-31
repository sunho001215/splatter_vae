"""Command line for the real-world preprocessing pipeline.

    python -m s4d.realworld.cli validate --config configs/realworld/pipeline_example.yaml
    python -m s4d.realworld.cli run --config CONFIG --episode ID --stages ingest,stereo_depth,calibrate,track2d,lift3d,export
    python -m s4d.realworld.cli status --root OUTPUT_ROOT --episode ID

Stages are idempotent: re-running skips complete stages whose fingerprint matches (``--force`` rebuilds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from s4d.realworld import io as rio

STAGES = ("ingest", "stereo_depth", "calibrate", "track2d", "lift3d", "export")


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
        for stage in ("normalized",) + STAGES[1:5]:
            d = (rio.episode_dir(args.root, args.episode) / "normalized" if stage == "normalized"
                 else rio.stage_dir(args.root, args.episode, stage))
            rec = rio.read_record(d)
            print(f"{stage:13s} {'complete ' + rec['fingerprint'][:12] if rec else 'missing'}")
        return 0

    cfg = yaml.safe_load(Path(args.config).read_text())
    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.contract import validate_episode

    adapter = get_adapter(cfg["adapter"]["name"], cfg["adapter"].get("config", {}))
    if args.cmd == "validate":
        ok = True
        for ep in adapter.list_episodes():
            rep = validate_episode(adapter.read_meta(ep))
            ok &= rep.ok
            print(json.dumps({"episode": ep, "errors": rep.errors, "warnings": rep.warnings}, indent=1))
        return 0 if ok else 1

    from s4d.realworld import pipeline as P
    from s4d.realworld.export import export

    root = Path(cfg["output_root"])
    st = cfg.get("stages", {})
    wanted = args.stages.split(",")
    work = root / "_worker_tmp" / args.episode
    if "ingest" in wanted:
        print(json.dumps(P.ingest(adapter, args.episode, root, force=args.force)["summary"], indent=1, default=str))
    if "stereo_depth" in wanted:
        sc = st.get("stereo_depth", {})
        backend = P.WorkerStereoBackend(_spec(cfg, "foundation_stereo"), work / "stereo",
                                        cfg["workers"]["foundation_stereo"].get("params"))
        P.stereo_depth(root, args.episode, backend, chunk=sc.get("chunk", 16),
                       references=tuple(sc.get("references", ["left"])), force=args.force)
    if "calibrate" in wanted:
        cc = st["calibrate"]
        P.calibrate(root, args.episode, AnnotationLandmarkSource(cc["landmark_source"]),
                    landmark_links=cc["landmark_links"], every=cc.get("every", 1), force=args.force)
    if "track2d" in wanted:
        tc = st.get("track2d", {})
        backend = P.WorkerTrackerBackend(_spec(cfg, "alltracker"), work / "track",
                                         cfg["workers"]["alltracker"].get("params"))
        P.track2d(root, args.episode, backend, window=tc.get("window", 16), step=tc.get("step", 8),
                  grid_stride=tc.get("grid_stride", 8),
                  tracker_hw=tuple(tc["tracker_hw"]) if tc.get("tracker_hw") else None, force=args.force)
    if "lift3d" in wanted:
        P.lift3d(root, args.episode, **st.get("lift3d", {}), force=args.force)
    if "export" in wanted:
        ec = st.get("export", {})
        export(root, args.episode, root / "export" / args.episode, strides=tuple(ec.get("strides", (1, 2))),
               allow_infrared_input=ec.get("allow_infrared_input", False),
               validation_tail_windows=ec.get("validation_tail_windows", 0), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
