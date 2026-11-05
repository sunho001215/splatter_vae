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

from ops.inference_server import command_support as _command_support  # noqa: E402
from ops.inference_server.session import RemoteRunner, Server  # noqa: E402
from ops.inference_server.stream_client import RemoteWorkerPool  # noqa: E402
from s4d.realworld.cli import BASELINE_STAGES  # noqa: E402
from s4d.realworld.cli import STAGES as PIPELINE_STAGES  # noqa: E402

STAGES = PIPELINE_STAGES + BASELINE_STAGES


def spec(cfg: dict, name: str, **override):
    return _command_support.spec(cfg, name, **override)


def remote_pipeline_config(pipeline: dict, server: dict) -> dict:
    """Resolve worker locations in the container while leaving adapter and output paths on the host.

    Server worker fields override host-specific worker paths; model parameters merge so inference settings from
    the episode config survive. This also supplies the kernel interpreter to older stereo episode configs.
    """
    return _command_support.remote_pipeline_config(pipeline, server)


def local_head() -> str:
    return _command_support.local_head(repo=REPO, subprocess_module=subprocess)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--output-root", help="override the pipeline config's output_root (e.g. a fresh baseline run)")
    ap.add_argument("--stages", help="comma-separated; default: the config's stages (s4d.realworld.cli.default_stages)")
    ap.add_argument("--force-stage", action="append", default=[])
    ap.add_argument("--skip-weights", action="store_true", help="weights already staged (still sha-checked)")
    args = ap.parse_args(argv)
    wanted = args.stages.split(",") if args.stages else None
    unknown = (set(wanted or ()) | set(args.force_stage)) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    from s4d.realworld import io as rio
    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.cli import adapter_config, check_stage_scope, default_stages, run_stages

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    wanted = wanted or default_stages(cfg)
    try:
        check_stage_scope(cfg, wanted)
    except ValueError as exc:
        ap.error(str(exc))
    if args.output_root:
        cfg["output_root"] = args.output_root
    server = Server.load(Path(args.server))
    cfg = remote_pipeline_config(cfg, server.config)
    root = Path(cfg["output_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"started_utc": stamp, "episode": args.episode, "pipeline": args.pipeline, "server": args.server,
                    "requested_stages": wanted, "baselines": cfg.get("baselines") or [], "stages": {}}

    def save():
        rio.atomic_write_json(run_dir / "run.json", json.loads(json.dumps(record, default=str)))

    head = local_head()
    record["source_commit"] = head
    record["container"] = server.verify_container()
    record["server_source"] = server.verify_source(head)
    record["weights"] = server.stage_weights(fetch=not args.skip_weights)
    record["audit_start"] = server.audit_start()
    save()
    pool_cfg = _command_support.pool_config(server.config, head)
    pool = RemoteWorkerPool(pool_cfg, log_dir=run_dir / "worker_logs")
    kernels = spec(cfg, "s4d_kernels", expected_commit=head)
    runner = RemoteRunner(pool, kernels, describe={"server": server.config["ssh"][-1], "container": server.container,
                                                   "source_commit": head})
    adapter = get_adapter(cfg["adapter"]["name"], adapter_config(cfg))
    ep = args.episode

    def run(name, fn):
        if name not in wanted:
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
        run_stages(cfg, adapter, ep, runner, run, wanted, spec_factory=lambda name: spec(cfg, name))
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
