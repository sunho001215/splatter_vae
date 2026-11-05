"""Real-world learning tooling (docs/realworld_baselines.md, docs/realworld_fql.md). Nothing here trains, launches,
uploads or registers anything, and nothing touches the robot.

    python -I scripts/realworld_learning.py configs                 # (re)write configs/realworld/baselines/*/*.yaml
    python -I scripts/realworld_learning.py gate [--manifest M] [--out report.json]   # exit 1 unless every task passed
    python -I scripts/realworld_learning.py plan --targets drawer-open mug-hanging joint [--stage all|pretrain|fql]
           [--out plan.yaml]                                     # refuses targets whose tasks did not pass the gate
    python -I scripts/realworld_learning.py inputs --store <store.yaml> --gate-report <report> --out-dir <dir>
                                                                 # split manifest + store copy for one target
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

from s4d.rwlearn import configs, gate, launch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    g = sub.add_parser("gate")
    g.add_argument("--manifest", type=Path, default=gate.MANIFEST)
    g.add_argument("--tasks", nargs="+", default=["drawer-open", "mug-hanging"])
    g.add_argument("--required-stages", nargs="+", default=list(gate.REQUIRED_STAGES))
    g.add_argument("--out", type=Path)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", type=Path, default=gate.MANIFEST)
    p.add_argument("--targets", nargs="+", default=["drawer-open", "mug-hanging", "joint"],
                   choices=sorted(launch.TARGET_TASKS))
    p.add_argument("--stage", choices=("all", "pretrain", "fql"), default="all")
    p.add_argument("--fql-encoder", choices=("task", "joint"), default="task")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--required-stages", nargs="+", default=list(gate.REQUIRED_STAGES))
    p.add_argument("--out", type=Path)
    i = sub.add_parser("inputs")
    i.add_argument("--store", required=True)
    i.add_argument("--gate-report", required=True, type=Path)
    i.add_argument("--target", required=True, choices=sorted(launch.TARGET_TASKS))
    i.add_argument("--out-dir", required=True, type=Path)
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
        print(json.dumps({"passed": report["passed"], **{t: {k: v[k] for k in ("passed", "valid_and_preprocessed",
                                                                               "problem_count")}
                                                        for t, v in report["tasks"].items()}}, indent=1))
        sys.exit(0 if report["passed"] else 1)
    elif args.cmd == "plan":
        tasks = sorted({t for target in args.targets for t in launch.TARGET_TASKS[target]})
        report = gate.evaluate_file(args.manifest, tasks, required_stages=tuple(args.required_stages))
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        result = launch.plan(report, args.targets, commit, stage=args.stage, seeds=tuple(args.seeds),
                             fql_encoder=args.fql_encoder)
        if args.out and result["rows"]:
            args.out.write_text(yaml.safe_dump(result, sort_keys=False, width=120))
        print(json.dumps({"rows": len(result["rows"]), "refused": result["refused"]}, indent=1))
        sys.exit(0 if result["rows"] and not result["refused"] else 1)
    elif args.cmd == "inputs":
        from s4d.baselines import data as _  # noqa: F401  (fail early if the training environment is incomplete)
        from s4d.rwlearn.baseline_data import check_gate
        from s4d.rwlearn.store import open_store, split_episodes

        store = open_store(args.store)
        report = json.loads(args.gate_report.read_text())
        check_gate(store, report, steps=None, target=args.target)  # store episodes = gate's; tasks = target's
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "split_seed0.json").write_text(json.dumps(split_episodes(store), indent=1) + "\n")
        (args.out_dir / "gate_report.json").write_text(json.dumps(report, indent=1) + "\n")
        locator = Path(args.store)
        (args.out_dir / "store.yaml").write_text(locator.read_text())
        print(f"wrote {args.out_dir}/{{store.yaml,gate_report.json,split_seed0.json}}")
    elif args.cmd == "scene-stats":
        from s4d.rwlearn.configs import MVP3D_BOUNDS
        from s4d.rwlearn.store import open_store, scene_statistics

        print(json.dumps(scene_statistics(open_store(args.store), args.episodes or None, bounds=MVP3D_BOUNDS),
                         indent=1))


if __name__ == "__main__":
    main()
