"""Local backup of remote runs, kept apart from reconciliation and job admission (2026-10-27).

The scheduler (scripts/jobs.py) downloads nothing for campaign jobs: it verifies a finished remote run in place (one
small inspection report), which releases same-host dependents, and marks the run done once this worker's
checksum-verified final local backup exists. This worker makes every campaign copy, one batched, checksum-verified
rsync per run and scope. It reads the registry and the queue and never writes them; it launches, stops and admits
nothing.

Runs, in this order:
- backups: remote campaign jobs whose compute is complete (registry status sync_pending after a zero exit) and whose
  final essential copy (status, completion records, logs, metrics, configs, arrays, exports and the required
  checkpoints) is not yet verified here, recorded in runs/remote/results/<run>.json; runs that a local queued job
  depends on (a cross-host dependency) first. A pass with backups to make ignores --min-interval;
- live essential copies of active remote jobs (progress monitoring only; nothing waits for them);
- completed remote campaign jobs whose final copy deferred media (registry sync scope "essential"): a final
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


def backup_done(client: RemoteClient, run_id: str, attempt: int) -> bool:
    """A verified final essential record of this attempt (its exit.json names the attempt)."""
    try:
        record = json.loads((client.network_dir / "results" / (run_id + ".json")).read_text())
        exit_report = json.loads((client.main_repo / "runs" / run_id / "exit.json").read_text())
    except (OSError, ValueError):
        return False
    return (record.get("verified") is True and record.get("final") is True and record.get("scope") == jobs.ESSENTIAL
            and exit_report.get("attempt") == attempt)


def backup_candidates(state: dict, client: RemoteClient, queue_jobs: list[dict]) -> list[str]:
    """Compute-complete remote campaign runs without a verified final local backup; cross-host dependencies first."""
    needed_locally = {dep for job in queue_jobs if job.get("host", "local") == "local" for dep in job.get("deps", [])
                      if state.get(job["id"], {}).get("status", jobs.PENDING) == jobs.PENDING}
    found = [job_id for job_id, current in state.items()
             if current.get("host") == "remote" and current.get("verification_purpose") is None
             and current.get("campaign_eligible") is not False and current.get("status") == "sync_pending"
             and current.get("code") == 0 and not backup_done(client, job_id, current.get("attempts"))]
    return sorted(found, key=lambda job_id: job_id not in needed_locally)


def required_checkpoints(current: dict) -> list[str]:
    return [name for name in current.get("required_results", []) if Path(name).parts[0] == "checkpoints"]


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


def running(state: dict) -> list[str]:
    """Executing remote campaign runs (live essential copies)."""
    return [job_id for job_id, current in state.items()
            if current.get("host") == "remote" and current.get("verification_purpose") is None
            and current.get("campaign_eligible") is not False and current.get("status") in jobs.RESERVING]


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
        state = jobs.read_registry()
        backups = backup_candidates(state, client, jobs.load_queue().get("jobs", []))
        if args.run:
            backups = [r for r in backups if r in args.run]
        if since < args.min_interval and not backups:
            print(json.dumps({"skipped": f"previous pass ended {since:.0f} s ago"}), flush=True)
            return 0
        start = time.time()
        final, live = candidates(state, client)
        progress = running(state)
        weights = checkpoint_candidates(state, client)
        if args.run:
            final, live = [r for r in final if r in args.run], [r for r in live if r in args.run]
            weights, progress = [r for r in weights if r in args.run], [r for r in progress if r in args.run]
        print(json.dumps({"start": time.strftime("%F %T"), "backups": len(backups), "progress": len(progress),
                          "final": len(final), "live": len(live), "checkpoints": len(weights)}), flush=True)
        work = ([(r, True, "backup") for r in backups] + [(r, False, jobs.ESSENTIAL) for r in progress]
                + [(r, True, "media") for r in final] + [(r, False, "media") for r in live]
                + [(r, True, "checkpoints") for r in weights])
        for run_id, is_final, scope in work:
            if time.time() - start > args.budget_seconds:
                print(json.dumps({"stopped": "budget", "seconds": round(time.time() - start, 1)}), flush=True)
                break
            t0 = time.time()
            entry = {"run": run_id, "final": is_final, "scope": scope}
            try:
                if scope == "backup":  # the final essential copy with only the required weights (the rest: "checkpoints")
                    weights_needed = required_checkpoints(state[run_id])
                    result = client.sync_results(run_id, checkpoints=bool(weights_needed), final=True, wait=False,
                                                 scope=jobs.ESSENTIAL, only_checkpoints=weights_needed or None)
                else:
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
