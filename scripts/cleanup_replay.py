"""Automatic, idempotent replay cleanup of completed RL runs on one host (user directive 2026-10-28).

Decides eligibility from the authoritative queue, registry and admission pauses of the main repository
(s4d/lifecycle_cleanup.py: budget complete, record complete, no pause, no continuation), then re-checks every on-host
condition right before deleting (s4d/replay_cleanup_host.py: real directories, final.json at the budget, final policy
intact and loadable, no writer) and deletes only the allowlisted replay files. Local runs are handled in-process
(including a CPU ``torch.load(weights_only=True)`` of the final policy); remote runs in one SSH command per batch,
checked on the remote host with the standard library (zip integrity and an inert pickle read of the final policy,
Docker state of the run's containers). Every completed deletion is appended once to the ledger
(runs/setup/replay-cleanup-ledger.jsonl: run, host, files with sizes, time, host and controller evidence); a run with
no remaining replay files is a no-op. ``--dry-run`` performs every check and prints what would be deleted, deleting and
recording nothing (it writes no file at all). Reads the queue and registry and never writes them; launches, stops and
signals nothing. One deleting pass per host at a time (non-blocking lock experiments/replay-cleanup-<host>.lock). Not
scheduled by any loop here: the user schedules it.

    .venv/bin/python -I scripts/cleanup_replay.py --host local|remote [--dry-run] [--run <id> ...] [--verbose]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import jobs  # noqa: E402  (scripts/jobs.py: queue validation and registry replay)

from s4d import lifecycle_cleanup as lifecycle  # noqa: E402
from s4d import replay_cleanup_host as host_check  # noqa: E402
from s4d.remote_access import RemoteClient, RemoteCommandError, RemoteUnreachable, main_repository  # noqa: E402


def summarize(report: dict) -> dict:
    files = report.get("files") or []
    line = {"run": report.get("run_id"), "status": report.get("status")}
    if report.get("reason"):
        line["reason"] = report["reason"]
    if report.get("status") == "would_delete":
        line.update(files=[item["name"] for item in files], bytes=sum(item.get("allocated_bytes", 0) for item in files))
    elif report.get("status") == "deleted":
        removed = [item for item in files if item.get("removed") is True]
        line.update(files=[item["name"] for item in removed], bytes=sum(item.get("allocated_bytes", 0) for item in removed))
    return line


def main() -> int:
    main_repo = main_repository(REPO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", choices=("local", "remote"), required=True)
    parser.add_argument("--dry-run", action="store_true", help="check everything, print what would be deleted")
    parser.add_argument("--run", action="append", default=[], help="restrict to these run ids")
    parser.add_argument("--batch", type=int, default=40, help="remote runs per SSH command")
    parser.add_argument("--verbose", action="store_true", help="also list every ineligible run with its reason")
    parser.add_argument("--queue", type=Path, default=main_repo / "experiments/queue.yaml")
    parser.add_argument("--registry", type=Path, default=main_repo / "experiments/registry.jsonl")
    parser.add_argument("--runs", type=Path, default=main_repo / "runs")
    parser.add_argument("--pauses", type=Path, default=main_repo / "runs/setup/admission-pauses.json")
    parser.add_argument("--ledger", type=Path, default=main_repo / "runs/setup/replay-cleanup-ledger.jsonl")
    args = parser.parse_args()
    if not 1 <= args.batch <= 100:
        parser.error("--batch must be in [1, 100]")
    if args.dry_run:  # writes nothing at all, not even the pass lock
        return run(args)
    lock_path = args.queue.parent / f"replay-cleanup-{args.host}.lock"
    with open(lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"skipped": f"another {args.host} replay cleanup holds {lock_path.name}"}), flush=True)
            return 0
        return run(args)


def run(args) -> int:
    pauses = lifecycle.load_pauses(args.pauses)  # fails closed: an unreadable pause file stops the pass
    rows = jobs.load_queue(args.queue).get("jobs", [])
    state = jobs.read_registry(args.registry)
    decisions = lifecycle.plan(rows, state, pauses, args.host, args.runs.resolve())
    if args.run:
        decisions = [decision for decision in decisions if decision.run_id in args.run]
    eligible = [decision for decision in decisions if decision.eligible]
    refused = Counter(decision.reason.split(":")[0] for decision in decisions if not decision.eligible)
    if args.verbose:
        for decision in decisions:
            if not decision.eligible:
                print(json.dumps({"run": decision.run_id, "status": "ineligible", "reason": decision.reason}), flush=True)
    if args.host == "local":  # runs without a replay directory need no check at all
        eligible = [decision for decision in eligible if (args.runs / decision.run_id / "replay").is_dir()]
    recorded = lifecycle.recorded_receipts(args.ledger)
    totals = Counter()
    reports = []
    def still_eligible(batch):
        """Re-plan from the current queue, registry and pauses right before deleting: a run whose state changed since
        the plan (reset, resume, new continuation, pause) is left alone."""
        fresh = {decision.run_id: decision for decision in lifecycle.plan(
            jobs.load_queue(args.queue).get("jobs", []), jobs.read_registry(args.registry),
            lifecycle.load_pauses(args.pauses), args.host, args.runs.resolve())}
        return [decision for decision in batch if fresh.get(decision.run_id) is not None
                and fresh[decision.run_id].eligible and fresh[decision.run_id].spec == decision.spec]

    if args.host == "local":
        for decision in eligible:
            if not args.dry_run and not still_eligible([decision]):
                print(json.dumps({"run": decision.run_id, "status": "skipped", "reason": "state changed"}), flush=True)
                continue
            report = host_check.clean_run(
                str(args.runs), decision.spec, dry_run=args.dry_run,
                writers=lambda run_id, run_dir, session=decision.session: lifecycle.local_writers(
                    Path(run_dir), run_id, session),
                verify_policy=lifecycle.torch_policy_check)
            reports.append((decision, report))
    else:
        client = RemoteClient(repo=REPO)
        for start in range(0, len(eligible), args.batch):
            batch = eligible[start:start + args.batch]
            if not args.dry_run:
                batch = still_eligible(batch)
                if not batch:
                    continue
            specs = [decision.spec for decision in batch]
            command = lifecycle.remote_command(client.root, specs, args.dry_run)
            try:
                response = client.checked_ssh(command, wait=False, idempotent=True, timeout=900)
                found = lifecycle.parse_remote(response.stdout, specs, args.dry_run)
            except RemoteUnreachable as exc:
                print(json.dumps({"error": f"remote unreachable ({exc}); pass ends"}), flush=True)
                return RemoteUnreachable.exit_code
            except (RemoteCommandError, ValueError) as exc:
                print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:300]}; pass ends"}), flush=True)
                return 1
            reports += list(zip(batch, found, strict=True))
    for decision, report in reports:
        print(json.dumps(summarize(report)), flush=True)
        totals[report["status"]] += 1
        if args.dry_run:
            totals["would_free_bytes"] += sum(item.get("allocated_bytes", 0) for item in report.get("files") or [])
            continue
        entries = lifecycle.ledger_entries(report, args.host, decision, recorded, time.time())
        lifecycle.append_ledger(args.ledger, entries)
        totals["ledger_entries"] += len(entries)
        totals["freed_bytes"] += sum(entry["freed_bytes"] for entry in entries)
    print(json.dumps({"host": args.host, "dry_run": args.dry_run, "considered": len(decisions),
                      "eligible": len(eligible), "ineligible_by_reason": dict(refused), **totals}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
