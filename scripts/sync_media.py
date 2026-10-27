"""Bulk media synchronization of remote runs, kept apart from job admission (2026-10-27).

The scheduler (scripts/jobs.py) syncs only what advances dependencies: result scope "essential" (status, completion
records, logs, metrics, configs, arrays, checkpoints, exports). This worker copies the rest, the evaluation imagery and
video (scope "media", s4d.remote_access.MEDIA_NAMES), with one batched, checksum-verified rsync per run, so admission
never waits for it. It reads the registry and never writes it; it launches, stops and admits nothing.

Runs, completed first:
- completed remote campaign jobs whose final scheduler sync deferred media (registry sync scope "essential"): a final
  media sync (the remote manifest must be stable across the copy), recorded in runs/remote/results-media/<run>.json;
- active remote jobs: a live media copy (files that change during the copy are retried on the next pass).
One worker at a time (non-blocking lock runs/remote/access/media-worker.lock); per-run media locks are separate from the
scheduler's per-run result locks, and the two scopes copy disjoint file sets.

    .venv/bin/python -I scripts/sync_media.py [--budget-seconds 1500] [--run <id> ...]
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


def media_done(client: RemoteClient, run_id: str) -> bool:
    path = client.network_dir / "results-media" / (run_id + ".json")
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return record.get("verified") is True and record.get("final") is True and record.get("scope") == "media"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=float, default=1500.0,
                        help="start no new run after this many seconds (a run in progress finishes)")
    parser.add_argument("--run", action="append", default=[], help="restrict to these run ids")
    args = parser.parse_args()
    client = RemoteClient()
    with (client.state_dir / "media-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"skipped": "another media worker holds the lock"}), flush=True)
            return 0
        start = time.time()
        final, live = candidates(jobs.read_registry(), client)
        if args.run:
            final, live = [r for r in final if r in args.run], [r for r in live if r in args.run]
        print(json.dumps({"start": time.strftime("%F %T"), "final": len(final), "live": len(live)}), flush=True)
        for run_id, is_final in [(r, True) for r in final] + [(r, False) for r in live]:
            if time.time() - start > args.budget_seconds:
                print(json.dumps({"stopped": "budget", "seconds": round(time.time() - start, 1)}), flush=True)
                break
            t0 = time.time()
            entry = {"run": run_id, "final": is_final}
            try:
                result = client.sync_results(run_id, final=is_final, wait=False, scope="media")
                entry.update(ok=True, files=len(result["files"]))
            except RemoteUnreachable:
                print(json.dumps({**entry, "ok": False, "error": "remote unreachable; pass ends"}), flush=True)
                return 0
            except (RemoteCommandError, OSError, ValueError) as exc:
                entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            entry["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(entry), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
