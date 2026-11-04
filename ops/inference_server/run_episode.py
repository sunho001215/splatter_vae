"""Run the real-world pipeline for one episode with every compute stage in the inference container.

The local host only reads the raw episode, streams bounded chunks over the authenticated SSH channel, saves each
returned chunk durably (``io.ChunkStore``) and acknowledges it, and assembles the export cache. Stages are the same
functions as ``python -m s4d.realworld.cli`` with a ``RemoteRunner``; completed stages are skipped and an interrupted
stage resumes at its first unsaved chunk (re-running this command is the resume procedure).

    .venv/bin/python -I ops/inference_server/run_episode.py --pipeline configs/realworld/mug_hanging_rgbd.yaml \\
        --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000000 [--stages a,b] [--force-stage s]

Writes ``<output_root>/runs/<utc>.json``: container/source checks, weight staging, per-call timings and resources,
the persistence audit (nothing below the host root changed; empty container diff) and each stage's record id.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from ops.inference_server.session import RemoteRunner, Server  # noqa: E402
from ops.inference_server.stream_client import RemoteWorkerPool  # noqa: E402

STAGES = ("ingest", "sensor_depth", "sam3_masks", "calibrate", "mapanything_depth", "track2d", "lift3d", "export",
          "dino_snerl")


def spec(cfg: dict, name: str, **override):
    from s4d.realworld.workers import WorkerSpec

    w = {**cfg["workers"][name], **override}
    return WorkerSpec(name, w["python"], repo_dir=w.get("repo_dir"), expected_commit=w.get("expected_commit"),
                      checkpoint=w.get("checkpoint"), timeout_s=float(w.get("timeout_s", 3600)))


def local_head() -> str:
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"local checkout {REPO} has uncommitted changes; commit before a remote run")
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force-stage", action="append", default=[])
    ap.add_argument("--skip-weights", action="store_true", help="weights already staged (still sha-checked)")
    args = ap.parse_args(argv)

    from s4d.realworld import io as rio
    from s4d.realworld import pipeline as P
    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.baseline_stages import dino_snerl, sam3_masks
    from s4d.realworld.calibration import CalibrationConfig
    from s4d.realworld.depth_stages import MapAnythingBackend, mapanything_depth, sensor_depth
    from s4d.realworld.export import export

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    server = Server.load(Path(args.server))
    root = Path(cfg["output_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"started_utc": stamp, "episode": args.episode, "pipeline": args.pipeline, "server": args.server,
                    "stages": {}}

    def save():
        rio.atomic_write_json(run_dir / "run.json", json.loads(json.dumps(record, default=str)))

    head = local_head()
    record["source_commit"] = head
    record["container"] = server.verify_container()
    record["server_source"] = server.verify_source(head)
    record["weights"] = server.stage_weights()
    record["audit_start"] = server.audit_start()
    save()
    pool_cfg = {k: server.config[k] for k in ("ssh", "container", "workdir", "source_root", "retries", "backoff_s",
                                              "call_timeout_s", "hello_timeout_s", "workers", "env")}
    pool_cfg["expected_source_commit"] = head
    pool = RemoteWorkerPool(pool_cfg, log_dir=run_dir / "worker_logs")
    kernels = spec(cfg, "s4d_kernels", expected_commit=head)
    runner = RemoteRunner(pool, kernels, describe={"server": server.config["ssh"][-1], "container": server.container,
                                                   "source_commit": head})
    adapter = get_adapter(cfg["adapter"]["name"], cfg["adapter"]["config"])
    st = cfg.get("stages", {})
    wanted = args.stages.split(",")
    ep = args.episode

    def run(name, fn):
        if name not in wanted or name in record["stages"]:
            return
        t0 = time.time()
        n0 = len(runner.calls)
        rec = fn(name in args.force_stage)
        calls = runner.calls[n0:]
        record["stages"][name] = {"seconds": round(time.time() - t0, 1), "skipped": rec.get("skipped"),
                                  "build_id": rec.get("build_id"), "remote_calls": len(calls),
                                  "remote_seconds": round(sum(c["elapsed_s"] or 0 for c in calls), 1),
                                  "peak_container_memory_bytes": max(
                                      ((c["resources"].get("cgroup") or {}).get("memory.peak") or 0 for c in calls),
                                      default=None),
                                  "peak_worker_rss_bytes": max((c["resources"].get("ru_maxrss") or 0 for c in calls),
                                                               default=None),
                                  "peak_cuda_reserved_bytes": max(
                                      ((c["resources"].get("cuda") or {}).get("max_memory_reserved") or 0
                                       for c in calls), default=None),
                                  "summary": rec.get("summary")}
        save()
        print(f"[{name}] {record['stages'][name]['seconds']} s, {len(calls)} remote calls"
              f"{' (skipped: complete)' if rec.get('skipped') else ''}", flush=True)

    try:
        run("ingest", lambda f: P.ingest(adapter, ep, root, force=f))
        sc = st.get("sensor_depth", {})
        run("sensor_depth", lambda f: sensor_depth(root, ep, runner, force=f, **sc))
        cc = dict(st.get("calibrate", {}))
        ccfg = CalibrationConfig(evidence=cc.get("evidence", "landmarks"), **cc.get("cfg", {}))
        s3 = st.get("sam3_masks", {})
        meta = P.load_meta(root, ep)

        def run_sam3():
            run("sam3_masks", lambda f: sam3_masks(root, ep, spec(cfg, "sam3"),
                                                   cameras=s3.get("cameras") or [v.stream_id for v in meta.views()],
                                                   prompts=s3["prompts"], mode=s3.get("mode", "segment_images"),
                                                   chunk=s3.get("chunk", 16),
                                                   score_threshold=s3.get("score_threshold", 0.5), force=f,
                                                   runner=runner))

        if cc.get("robot_mask_stage") == "sam3_masks":  # segmentation is calibration evidence: run it first
            run_sam3()
        run("calibrate", lambda f: P.calibrate(root, ep, None, landmark_links=cc.get("landmark_links", []),
                                               every=cc.get("every", 1), cfg=ccfg, overlays=cc.get("overlays", 3),
                                               force=f, runner=runner, robot_mask_stage=cc.get("robot_mask_stage")))
        mc = st.get("mapanything_depth", {})
        ma = MapAnythingBackend(spec(cfg, "mapanything"), runner, cfg["workers"]["mapanything"].get("params", {}))
        run("mapanything_depth", lambda f: mapanything_depth(root, ep, ma, force=f, **mc))
        tc = st.get("track2d", {})
        tracker = P.WorkerTrackerBackend(spec(cfg, "alltracker"), root / "_unused", cfg["workers"]["alltracker"].get(
            "params"), runner=runner)
        run("track2d", lambda f: P.track2d(root, ep, tracker, window=tc.get("window", 16), step=tc.get("step", 8),
                                           grid_stride=tc.get("grid_stride", 8),
                                           tracker_hw=tuple(tc["tracker_hw"]) if tc.get("tracker_hw") else None,
                                           force=f))
        run("lift3d", lambda f: P.lift3d(root, ep, **st.get("lift3d", {}), force=f, runner=runner))
        ec = st.get("export", {})
        run("export", lambda f: export(root, ep, root / "export" / ep, strides=tuple(ec.get("strides", (1, 2))),
                                       validation_tail_frames=ec.get("validation_tail_frames", 0),
                                       max_sync_ns=ec.get("max_sync_ns"), max_gap_factor=ec.get("max_gap_factor"),
                                       aspect=ec.get("aspect", "stretch"), force=f))
        if cc.get("robot_mask_stage") != "sam3_masks":
            run_sam3()
        dc = st.get("dino_snerl", {})
        run("dino_snerl", lambda f: dino_snerl(root / "export" / ep, rio.stage_dir(root, ep, "dino_snerl"),
                                               spec(cfg, "dino_snerl"), backbone=dc.get("backbone", "scratch"),
                                               force=f, runner=runner))
    finally:
        record["sessions"] = {name: pool.session_info(name) for name in server.config["workers"]
                              if name in pool._info}
        pool.close()
        record["resources_end"] = server.resources()
        record["audit_end"] = server.audit_end(record["audit_start"])
        record["remote_calls"] = runner.calls
        record["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        save()
        print(json.dumps({"audit_ok": record["audit_end"]["ok"], "run": str(run_dir / "run.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
