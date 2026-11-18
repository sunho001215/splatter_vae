"""pick-and-place-v2 pipeline entry point: status | check-data | register.

  status      read-only: has the dataset arrived (complete marker or stable inventory)? which upstream stages are
              done (ingest, Phase A, rig, Phase B, RGB/geometry publications, scene)? which queue waves would be
              registered, with how many rows, and the exact next command per stage.
  check-data  read-only: rehash and validate every arrived recording (count, integrity vs the Robot-PC SHA list,
              recorder completion, cameras/timing/actions, success-only labels, seed-0 90/10 split).
  register    DRY RUN by default. Refuses unless check-data passes on the actual arrived recordings and the wave's
              upstream evidence exists. --execute (adopted main only) appends the rows under the queue/registry
              locks behind the exact-ID placement pause pnpv2-await-placement-20261117. Never launches anything.

Exit codes: 0 ok, 1 check-data failed, 2 refused. Never reads or writes the Robot PC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLACEHOLDER_RELEASE = {"commit": "0" * 40, "image": "sha256:" + "0" * 64, "source": "local"}
PLACEHOLDER_SCENE = {"lookat": [0.0, 0.0, 0.0], "pose_scale": 1.0, "bounds": [0, 0, 0, 1, 1, 1]}


def _spec(a):
    from s4d.rwlearn.pnpv2_spec import load_spec, read_arrival

    spec = load_spec(a.task_spec)
    if a.raw_root:
        spec["roots"]["raw"] = str(Path(a.raw_root).resolve())
    doc = read_arrival(spec)
    return (load_spec(a.task_spec, overlay=doc) | {"roots": spec["roots"]}) if doc is not None else spec


def _summary(report):
    keep = ("schema", "task", "passed", "problems", "episodes", "split", "manifest_sha256", "writes", "robot_pc_io",
            "notes")
    return {k: report[k] for k in keep if k in report} | {"arrival": report["arrival"]}


def _registered(repo):
    import yaml

    from s4d.rwlearn.pnpv2_spec import STEM

    queue = Path(repo) / "experiments/queue.yaml"
    if not queue.is_file():
        return []
    return sorted(r["id"] for r in yaml.safe_load(queue.read_text())["jobs"] if r["id"].startswith(STEM + "-"))


def status(a):
    from ops.jobs.pnpv2_plan import build_rows, counts
    from ops.jobs.pnpv2_registration import evidence, next_steps, ready_waves
    from s4d.rwlearn.pnpv2_arrival import arrival

    spec = _spec(a)
    state = arrival(spec, settle_seconds=a.settle_seconds)
    ev = evidence(spec, None, arrival_state=state)
    # Waves that would register once check-data passes (status never hashes the recordings).
    assumed = {**ev, "check-data": {"status": "done"}}
    rows = build_rows(spec, PLACEHOLDER_RELEASE, scene=PLACEHOLDER_SCENE)
    registered = _registered(a.repo)
    out = dict(schema="pick-and-place-v2-status/1", arrived=state["state"] == "complete", arrival=state,
               stages={k: {"status": v["status"], "detail": v["detail"], **({"acceptance_override": v["acceptance_override"]}
                       if v.get("acceptance_override") else {})} for k, v in ev.items()},
               waves={w: dict(rows=sum(r["pnpv2_wave"] == w for r in rows),
                              ready_after_check_data=w in ready_waves(assumed),
                              registered=sum(r["id"] in registered for r in rows if r["pnpv2_wave"] == w))
                      for w in ("rgb", "geometry")},
               would_register=counts([r for r in rows if r["pnpv2_wave"] in ready_waves(assumed)
                                      and r["id"] not in registered]),
               next=next_steps(spec, ev), writes=False)
    print(json.dumps(out, indent=2))
    return 0


def check_data(a):
    from s4d.rwlearn.pnpv2_arrival import check_data as check

    report = check(_spec(a), settle_seconds=a.settle_seconds)
    print(json.dumps(report if a.full else _summary(report), indent=2))
    return 0 if report["passed"] else 1


def register(a):
    from ops.jobs import pnpv2_registration as reg
    from s4d.rwlearn.pnpv2_arrival import check_data as check

    release = {"commit": a.commit, "image": a.image} | ({"source": "local"} if a.release_source == "local" else {})
    spec = _spec(a)
    record = reg.prepare(Path(a.repo), spec, check(spec, settle_seconds=a.settle_seconds), release,
                         waves=a.wave or None)
    if a.execute:
        record = reg.execute(Path(a.repo), record, recheck=lambda: check(spec, settle_seconds=a.settle_seconds))
    print(json.dumps(record, indent=2))
    return 0


def main(argv=None):
    from s4d.rwlearn.pnpv2_spec import SPEC

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-spec", type=Path, default=ROOT / SPEC)
    parser.add_argument("--raw-root", type=Path, help="override <roots.raw> (layout <raw>/pick-and-place-v2/success)")
    parser.add_argument("--settle-seconds", type=int, help="stable-inventory window without a VERIFIED marker")
    parser.add_argument("--repo", type=Path, default=Path("/home/ws/ws/splatter4d"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    c = sub.add_parser("check-data")
    c.add_argument("--full", action="store_true", help="include the canonical manifest")
    r = sub.add_parser("register")
    r.add_argument("--commit", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--release-source", choices=("local", "role"), default="local")
    r.add_argument("--wave", action="append", choices=("rgb", "geometry"))
    r.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv)
    from ops.jobs.pnpv2_registration import Refused

    try:
        return {"status": status, "check-data": check_data, "register": register}[a.command](a)
    except (Refused, ValueError, KeyError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
