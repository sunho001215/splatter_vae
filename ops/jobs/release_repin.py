"""Same-host release re-pin of a stopped remote run (remote MIG migration, user directive 2026-11-11 ~19:00).

A retry always keeps its first launch's (commit, image) (remote_campaign_driver.job_release; the scheduler holds a retry
whose release differs). After the remote GPUs become MIG instances, every release older than the MIG allowlist refuses
the instances in its in-container GPU guard, so an interrupted run could never resume. This tool moves a STOPPED run to
an accepted release whose execution scope is identical to its pinned release except the GPU-allowlist files
(REPIN_OPERATIONAL): same job id, arguments, fingerprint, data identity and run directory; the trainer resumes from its
own checkpoints/latest.pt under the next attempt. It only appends one ``release_repinned`` registry event (read by
ops/jobs/scheduler.py read_registry); it never launches, stops or deletes anything.

USE ONLY AFTER THE USER'S EXPLICIT RESUME AUTHORIZATION (the event records it).

    python -I ops/jobs/release_repin.py plan  --to-commit C --to-image I <job id>
    python -I ops/jobs/release_repin.py apply --to-commit C --to-image I --authorization TEXT <job id>
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.jobs.migration import commit_reader, import_closure  # noqa: E402

# Files allowed to differ between the pinned and the destination release: the GPU allowlist and its consumers inside
# the container (recorded in the event, not compared). Everything else in the scripts' import closure, the task / model
# configuration trees and the environment pins must be byte-identical.
REPIN_OPERATIONAL = frozenset({"ops/resources/gpu_guard.py", "ops/resources/mig_inventory.py", "mujoco_mig_setup.py",
                               "configs/hosts/remote.yaml", "configs/hosts/remote-mig.yaml"})
REPIN_CONFIG_SCOPE = ("configs/realworld/", "configs/metaworld/", "configs/rl/", "configs/baselines/")
REPIN_ENVIRONMENT = ("uv.lock", "pyproject.toml")
STOPPED = {"crashed", "pending", "failed"}


class RepinError(RuntimeError):
    pass


def scope_comparison(scripts, before_read, before_listed, after_read, after_listed) -> dict:
    """{"identical", "differences", "operational_differences", "files"} for two release trees given as readers."""
    paths = import_closure(scripts, before_read) | import_closure(scripts, after_read)
    paths |= {p for p in list(before_listed) + list(after_listed) if p.startswith(REPIN_CONFIG_SCOPE)}
    paths |= set(REPIN_ENVIRONMENT)
    files, differences, operational = {}, [], []
    for relative in sorted(paths):
        before, after = before_read(relative), after_read(relative)
        if relative in REPIN_OPERATIONAL:
            operational += [relative] if before != after else []
            continue
        if before != after:
            differences.append(relative)
        if after is not None:
            files[relative] = hashlib.sha256(after).hexdigest()
    return {"identical": not differences, "differences": differences, "operational_differences": operational,
            "files": files}


def check_state(job_id: str, current: dict, to_commit: str, to_image: str) -> None:
    """The run must be a stopped remote run pinned to another release (refuses active / unknown / never launched)."""
    if current.get("host") != "remote" or not current.get("commit") or not current.get("image_digest"):
        raise RepinError(f"{job_id}: not a launched remote run")
    if current.get("status") not in STOPPED or current.get("unknown"):
        raise RepinError(f"{job_id}: status {current.get('status')} (unknown={current.get('unknown')}); only a "
                         "stopped run whose container exit is recorded can be re-pinned")
    if (current["commit"], current["image_digest"]) == (to_commit, to_image):
        raise RepinError(f"{job_id}: already on {to_commit[:7]}")


def repin_event(job_id: str, current: dict, to_commit: str, to_image: str, scope: dict, scripts, authorization: str) -> dict:
    if not scope["identical"]:
        raise RepinError(f"{job_id}: execution scope differs: {scope['differences'][:8]}")
    if not authorization.strip():
        raise RepinError("the user's explicit resume authorization must be recorded")
    return {"event": "release_repinned", "id": job_id, "host": "remote", "attempt": current.get("attempt"),
            "from_commit": current["commit"], "from_image_digest": current["image_digest"],
            "commit": to_commit, "image_digest": to_image, "sources_identical": True,
            "operational_differences": scope["operational_differences"], "scripts": list(scripts),
            "scope_sha256": hashlib.sha256(json.dumps(scope["files"], sort_keys=True).encode()).hexdigest(),
            "authorization": authorization.strip()}


def main(argv=None) -> int:
    from ops.jobs import scheduler as jobs  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "apply"))
    parser.add_argument("--to-commit", required=True)
    parser.add_argument("--to-image", required=True)
    parser.add_argument("--authorization", default="")
    parser.add_argument("job_id")
    args = parser.parse_args(argv)
    queue = {job["id"]: job for job in jobs.load_queue()["jobs"]}
    job = queue[args.job_id]
    scripts = [job["script"]]
    with jobs.REGISTRY.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = jobs.read_registry().get(args.job_id, {})
        check_state(args.job_id, current, args.to_commit, args.to_image)
        before_read, before_listed = commit_reader(REPO, current["commit"])
        after_read, after_listed = commit_reader(REPO, args.to_commit)
        scope = scope_comparison(scripts, before_read, before_listed, after_read, after_listed)
        if args.command == "plan":
            print(json.dumps({k: scope[k] for k in ("identical", "differences", "operational_differences")}, indent=1))
            return 0 if scope["identical"] else 3
        event = repin_event(args.job_id, current, args.to_commit, args.to_image, scope, scripts, args.authorization)
        with jobs.REGISTRY.open("a") as stream:
            stream.write(json.dumps({**event, "time": time.time()}) + "\n")
    print(json.dumps(event, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
