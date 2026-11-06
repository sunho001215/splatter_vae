"""Real-world learning tooling (docs/realworld_baselines.md, docs/realworld_fql.md). Nothing here trains, launches,
uploads or registers anything, and nothing touches the robot.

    python -I scripts/realworld_learning.py configs                 # (re)write configs/realworld/baselines/*/*.yaml
    python -I scripts/realworld_learning.py gate [--manifest M] [--out report.json]
           # exit 0 iff every task passed the 100-episode gate; the report also carries the calibration hold
           # (geometry-dependent consumers) per task
    python -I scripts/realworld_learning.py plan --tasks drawer-open mug-hanging [--stage all|pretrain|fql]
           [--commit <release commit>] [--out plan.yaml]       # rows only for consumers whose gate level passed
    python -I scripts/realworld_learning.py plan --preview [--out plan.yaml]   # full plan, NOT registrable
    python -I scripts/realworld_learning.py outcomes --raw-root /home/ws/data/real_world --out ending_report.json
           # ending (reset / reversal) check of every raw recording (read-only); outcome success for all
    python -I scripts/realworld_learning.py inputs --store <store.yaml> --gate-report <report> --task <task>
           --out-dir <dir> [--ending-report <report>]      # split manifest, outcomes, gate report + store copy
    python -I scripts/realworld_learning.py scene-stats --store <store.yaml> [--episodes ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from s4d.rwlearn import configs, gate, launch, outcomes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    g = sub.add_parser("gate")
    g.add_argument("--manifest", type=Path, default=gate.MANIFEST)
    g.add_argument("--tasks", nargs="+", default=list(launch.TASKS))
    g.add_argument("--required-stages", nargs="+", default=list(gate.REQUIRED_STAGES))
    g.add_argument("--out", type=Path)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", type=Path, default=gate.MANIFEST)
    p.add_argument("--tasks", nargs="+", default=list(launch.TASKS), choices=launch.TASKS)
    p.add_argument("--stage", choices=("all", "pretrain", "fql"), default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=list(launch.SEEDS))
    p.add_argument("--required-stages", nargs="+", default=list(gate.REQUIRED_STAGES))
    p.add_argument("--commit", help="the release commit the coordinator pins (default: this checkout's HEAD)")
    p.add_argument("--preview", action="store_true", help="every row, without gate evidence (not registrable)")
    p.add_argument("--out", type=Path)
    o = sub.add_parser("outcomes")
    o.add_argument("--raw-root", type=Path, default=Path("/home/ws/data/real_world"))
    o.add_argument("--tasks", nargs="+", default=list(launch.TASKS))
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--summary-out", type=Path, help="compact per-episode summary (committed as docs evidence)")
    i = sub.add_parser("inputs")
    i.add_argument("--store", required=True)
    i.add_argument("--gate-report", required=True, type=Path)
    i.add_argument("--task", required=True, choices=launch.TASKS)
    i.add_argument("--out-dir", required=True, type=Path)
    i.add_argument("--ending-report", type=Path, help="rw-ending-flags report (flags are carried, never change labels)")
    s = sub.add_parser("scene-stats")
    s.add_argument("--store", required=True)
    s.add_argument("--episodes", nargs="*")
    args = ap.parse_args()

    if args.cmd == "configs":
        for path in configs.write_all():
            print(path.relative_to(REPO))
    elif args.cmd == "gate":
        report = gate.evaluate_file(args.manifest, args.tasks, required_stages=tuple(args.required_stages))
        text = json.dumps(report, indent=1)
        if args.out:
            args.out.write_text(text + "\n")
        print(json.dumps({"passed": report["passed"], "geometry_passed": report["geometry_passed"],
                          **{t: {**{k: v[k] for k in ("passed", "valid_and_preprocessed", "problem_count")},
                                 "calibration_hold": {k: report["calibration_hold"][t][k]
                                                      for k in ("passed", "validated", "problem_count")}}
                             for t, v in report["tasks"].items()}}, indent=1))
        sys.exit(0 if report["passed"] else 1)
    elif args.cmd == "plan":
        commit = args.commit or subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                               text=True, check=True).stdout.strip()
        if args.preview:
            result = launch.plan_preview(args.commit or launch.PLACEHOLDER_COMMIT, tuple(args.seeds))
            result["table"] = launch.plan_table(result)
        else:
            report = gate.evaluate_file(args.manifest, args.tasks, required_stages=tuple(args.required_stages))
            result = launch.plan(report, args.tasks, commit, stage=args.stage, seeds=tuple(args.seeds))
        if args.out and result["rows"]:
            args.out.write_text(yaml.safe_dump(result, sort_keys=False, width=120))
        print(json.dumps({"rows": len(result["rows"]), "refused": result.get("refused", {})}, indent=1))
        sys.exit(0 if result["rows"] and not result.get("refused") else 1)
    elif args.cmd == "outcomes":
        report = outcomes.scan(args.raw_root, tuple(args.tasks))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        if args.summary_out:
            args.summary_out.write_text(json.dumps(outcomes.summary(report), indent=1) + "\n")
        print(json.dumps({"counts": report["counts"], "flagged": outcomes.flagged(report)}, indent=1))
    elif args.cmd == "inputs":
        from s4d.baselines import data as _  # noqa: F401  (fail early if the training environment is incomplete)
        from s4d.rwlearn.baseline_data import check_gate
        from s4d.rwlearn.store import open_store, split_episodes

        store = open_store(args.store)
        report = json.loads(args.gate_report.read_text())
        # store episodes = gate's; tasks = the task's (the 100-episode level: every consumer re-checks its own level)
        check_gate(store, report, steps=None, target=args.task, consumer="e2e_cnn")
        ending = json.loads(args.ending_report.read_text()) if args.ending_report else None
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "split_seed0.json").write_text(json.dumps(split_episodes(store), indent=1) + "\n")
        (args.out_dir / "outcomes.json").write_text(
            json.dumps(outcomes.outcome_labels(store.episodes(), ending_report=ending), indent=1) + "\n")
        (args.out_dir / "gate_report.json").write_text(json.dumps(report, indent=1) + "\n")
        locator = Path(args.store)
        (args.out_dir / "store.yaml").write_text(locator.read_text())
        print(f"wrote {args.out_dir}/{{store.yaml,gate_report.json,split_seed0.json,outcomes.json}}")
    elif args.cmd == "scene-stats":
        from s4d.rwlearn.configs import MVP3D_BOUNDS
        from s4d.rwlearn.store import open_store, scene_statistics

        print(json.dumps(scene_statistics(open_store(args.store), args.episodes or None, bounds=MVP3D_BOUNDS),
                         indent=1))


if __name__ == "__main__":
    main()
