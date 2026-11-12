"""The remote allowlist switch to MIG instances: preflight and the queue-limits rewrite (Codex #5; ARCH Phase 0).

The switch is ONE step in the local main checkout, taken only after the user's explicit resume authorization:
configs/hosts/remote-mig.yaml + configs/hosts/remote.yaml (ops/resources/mig_inventory.py render) together with
``hosts.remote.limits`` of experiments/queue.yaml rewritten to the instances (``rewrite-queue``). Before it, the
``preflight`` must pass: no remote attempt may still be live or unknown on a physical GPU (they are closed first:
checkpoint pause / exit recorded and reconciled), and no queued row may name a physical GPU explicitly. After the
switch, compute-complete attempts of physical GPUs still reconcile and clean up (remote_backend.validate_container
accepts the MIG parents for EXISTING attempts only; nothing can be launched on them).

    python -I ops/jobs/mig_switch.py preflight --allowlist <remote-mig.yaml> [--queue experiments/queue.yaml]
    python -I ops/jobs/mig_switch.py rewrite-queue --allowlist <remote-mig.yaml> [--queue ...] [--max-jobs 4] [--max-mem-gb 21.5] [--write]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.resources.mig_inventory import REMOTE_PHYSICAL_GPUS, load_allowlist, queue_limits  # noqa: E402

LIVE = {"running", "launching"}


def preflight(state: dict, queues: dict[str, dict], instances) -> dict:
    """{"blockers": [...], "notes": [...]} for the switch. ``state``: read_registry(); ``queues``: {path: parsed queue}."""
    blockers, notes = [], []
    allowed = {item["uuid"] for item in instances}
    for job_id, current in sorted(state.items()):
        if current.get("host") != "remote":
            continue
        gpu = current.get("gpu")
        if current.get("status") in LIVE or current.get("unknown"):
            blockers.append(f"{job_id}: attempt {current.get('attempt')} {current.get('status')}"
                            f"{' (unknown)' if current.get('unknown') else ''} on {gpu}: close it first "
                            "(checkpoint pause or recorded exit, reconciled)")
        elif current.get("status") == "sync_pending" and gpu in REMOTE_PHYSICAL_GPUS:
            notes.append(f"{job_id}: compute-complete on {gpu}; completes from its backup after the switch")
    for path, queue in queues.items():
        remote = (queue.get("hosts") or {}).get("remote") or {}
        stale = [uuid for uuid in remote.get("limits", {}) if uuid not in allowed]
        if stale:
            notes.append(f"{path}: hosts.remote.limits names {len(stale)} non-instance devices (rewrite-queue)")
        for job in queue.get("jobs", []):
            if job.get("host") == "remote" and job.get("gpu", "any") not in ("any", *allowed):
                if state.get(job["id"], {}).get("status") in (None, "pending", "crashed"):
                    blockers.append(f"{path}: queued remote row {job['id']} pins device {job['gpu']}")
    return {"blockers": blockers, "notes": notes}


def _child(node, key):
    return next((value for k, value in node.value if k.value == key), None)


def rewrite_hosts(text: str, instances, max_jobs: int = 4, max_mem_gb: float = 21.5) -> str:
    """The queue text with ONLY the ``hosts.remote.limits`` mapping replaced by one entry per instance; every other
    byte (comments, e.g. the ``hold: true`` annotation, and the jobs) is kept."""
    limits = _child(_child(_child(yaml.compose(text), "hosts"), "remote"), "limits")
    if limits is None or not isinstance(limits, yaml.MappingNode) or not limits.value:
        raise ValueError("queue has no block mapping hosts.remote.limits to replace")
    indent = " " * limits.start_mark.column
    dumped = yaml.safe_dump(queue_limits(instances, max_jobs=max_jobs, max_mem_gb=max_mem_gb), sort_keys=False)
    block = "".join(indent + line if line.strip() else line for line in dumped.splitlines(True))
    start = text.rfind("\n", 0, limits.start_mark.index) + 1
    end = limits.end_mark.index
    line_start = text.rfind("\n", 0, end) + 1
    if not text[line_start:end].strip():  # the mapping ends at the indentation of the next key: keep that indentation
        end = line_start
    if text[start:limits.start_mark.index].strip():
        raise ValueError("unexpected layout before hosts.remote.limits")
    return text[:start] + block + text[end:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("preflight", "rewrite-queue"))
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=REPO / "experiments/queue.yaml")
    parser.add_argument("--extra-queue", type=Path, action="append", default=[],
                        help="further queues to check, e.g. runs/remote/verification/*-queue.yaml")
    parser.add_argument("--max-jobs", type=int, default=4)
    parser.add_argument("--max-mem-gb", type=float, default=21.5)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    instances = load_allowlist(args.allowlist)
    if not instances:
        raise SystemExit("no valid MIG allowlist")
    if args.command == "preflight":
        from ops.jobs import scheduler as jobs  # noqa: PLC0415  (read-only registry view)

        queues = {str(p): yaml.safe_load(p.read_text()) or {} for p in [args.queue, *args.extra_queue]}
        report = preflight(jobs.read_registry(), queues, instances)
        print(json.dumps(report, indent=1))
        return 3 if report["blockers"] else 0
    text = args.queue.read_text()
    new = rewrite_hosts(text, instances, args.max_jobs, args.max_mem_gb)
    before, after = yaml.safe_load(text), yaml.safe_load(new)
    if before["jobs"] != after["jobs"] or {k: v for k, v in before["hosts"].items() if k != "remote"} != {
            k: v for k, v in after["hosts"].items() if k != "remote"}:
        raise SystemExit("rewrite changed more than hosts.remote.limits; refused")
    print(yaml.safe_dump({"hosts": {"remote": {"limits": after["hosts"]["remote"]["limits"]}}}, sort_keys=False))
    if args.write:
        backup = args.queue.with_name(args.queue.name + f".before-mig-{time.strftime('%Y%m%d-%H%M%S')}")
        backup.write_text(text)
        staged = args.queue.with_name(args.queue.name + ".mig.tmp")
        staged.write_text(new)
        os.replace(staged, args.queue)
        print(f"written; backup {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
