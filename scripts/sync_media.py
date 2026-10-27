"""Bulk synchronization of remote runs (media, deferred checkpoints), kept apart from job admission (2026-10-27).

The scheduler (scripts/jobs.py) syncs only what advances dependencies: result scope "essential" (status, completion
records, logs, metrics, configs, arrays, checkpoints, exports). This worker copies the rest, the evaluation imagery and
video (scope "media", s4d.remote_access.MEDIA_NAMES), with one batched, checksum-verified rsync per run, so admission
never waits for it. It reads the registry and never writes it; it launches, stops and admits nothing.

Runs, completed first:
- completed remote campaign jobs whose final scheduler sync deferred media (registry sync scope "essential"): a final
  media sync (the remote manifest must be stable across the copy), recorded in runs/remote/results-media/<run>.json;
- active remote jobs: a live media copy (files that change during the copy are retried on the next pass).
- last, completed jobs whose final scheduler sync copied only the required checkpoints (registry sync
  "checkpoints": "required"): their remaining weights (scope "checkpoints"), recorded in
  runs/remote/results-checkpoints/<run>.json.
All transfers run as "bulk" inside the shared download budget (s4d.remote_access.download_slot: one transfer at a
time, rsync --bwlimit, chunked so that dependency transfers of the scheduler go first). One worker at a time
(non-blocking lock runs/remote/access/media-worker.lock); per-run media locks are separate from the scheduler's
per-run result locks. --min-interval skips a pass that starts too soon after the previous one (the manifests of large
runs are themselves downloads).

    .venv/bin/python -I scripts/sync_media.py [--budget-seconds 1500] [--min-interval 600] [--run <id> ...]
"""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobs  # noqa: E402

from s4d.remote_access import RemoteClient, RemoteCommandError, RemoteUnreachable  # noqa: E402


def scope_done(client: RemoteClient, run_id: str, scope: str = "media") -> bool:
    path = client.network_dir / ("results-" + scope) / (run_id + ".json")
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return record.get("verified") is True and record.get("final") is True and record.get("scope") == scope


def media_done(client: RemoteClient, run_id: str) -> bool:
    return scope_done(client, run_id, "media")


def candidates(state: dict, client: RemoteClient) -> tuple[list[str], list[str]]:
    """(completed runs with deferred media, active runs), each in registry order."""
    final, live = [], []
    for job_id, current in state.items():
        if (current.get("host") != "remote" or current.get("verification_purpose") is not None
                or current.get("campaign_eligible") is False):
            continue
        if current.get("status") == jobs.DONE and (current.get("sync") or {}).get("scope") == jobs.ESSENTIAL:
            if not media_done(client, job_id):
                final.append(job_id)
        elif current.get("status") in jobs.ACTIVE:
            live.append(job_id)
    return final, live


def checkpoint_candidates(state: dict, client: RemoteClient) -> list[str]:
    """Completed runs whose final scheduler sync copied only the required checkpoints, not yet completed here."""
    return [job_id for job_id, current in state.items()
            if current.get("host") == "remote" and current.get("verification_purpose") is None
            and current.get("campaign_eligible") is not False and current.get("status") == jobs.DONE
            and (current.get("sync") or {}).get("checkpoints") == "required"
            and not scope_done(client, job_id, "checkpoints")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=float, default=1500.0,
                        help="start no new run after this many seconds (a run in progress finishes)")
    parser.add_argument("--run", action="append", default=[], help="restrict to these run ids")
    parser.add_argument("--min-interval", type=float, default=600.0,
                        help="skip this pass if the previous one finished less than this many seconds ago")
    args = parser.parse_args()
    client = RemoteClient()
    with (client.state_dir / "media-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"skipped": "another media worker holds the lock"}), flush=True)
            return 0
        stamp = client.state_dir / "media-worker.last"
        try:
            since = time.time() - float(stamp.read_text())
        except (OSError, ValueError):
            since = float("inf")
        if since < args.min_interval:
            print(json.dumps({"skipped": f"previous pass ended {since:.0f} s ago"}), flush=True)
            return 0
        start = time.time()
        state = jobs.read_registry()
        final, live = candidates(state, client)
        weights = checkpoint_candidates(state, client)
        if args.run:
            final, live = [r for r in final if r in args.run], [r for r in live if r in args.run]
            weights = [r for r in weights if r in args.run]
        print(json.dumps({"start": time.strftime("%F %T"), "final": len(final), "live": len(live),
                          "checkpoints": len(weights)}), flush=True)
        work = [(r, True, "media") for r in final] + [(r, False, "media") for r in live] + [(r, True, "checkpoints")
                                                                                          for r in weights]
        for run_id, is_final, scope in work:
            if time.time() - start > args.budget_seconds:
                print(json.dumps({"stopped": "budget", "seconds": round(time.time() - start, 1)}), flush=True)
                break
            t0 = time.time()
            entry = {"run": run_id, "final": is_final, "scope": scope}
            try:
                result = client.sync_results(run_id, final=is_final, wait=False, scope=scope)
                entry.update(ok=True, files=len(result["files"]))
            except RemoteUnreachable:
                print(json.dumps({**entry, "ok": False, "error": "remote unreachable; pass ends"}), flush=True)
                return 0
            except (RemoteCommandError, OSError, ValueError) as exc:
                entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            entry["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(entry), flush=True)
        stamp.write_text(str(time.time()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
