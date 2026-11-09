"""Launch PLANS for real-world baseline pretraining and offline FQL on the remote role host 192.168.10.22
(docs/realworld_fql.md, "Training plan"). Nothing here registers, launches or uploads: ``plan`` returns queue rows
(``experiments/queue.yaml`` schema plus an explicit ``role:``, as configs/hosts/roles.yaml asks of new real-world rows)
that the coordinator registers through the normal release path, with the release commit it pins.

User decisions 2026-11-06 ~11:20 KST that shape the plan: task-specific encoders only (no joint-task rows); FQL
``alpha`` = 10 with the reference configuration, no alpha sweep (later tuning follows a predefined real-robot
evaluation protocol, not offline action error); one end-to-end CNN + FQL condition (``e2e_cnn``); all episodes are
successful demonstrations (``outcomes.json``). Order: representation baselines first, then FQL (every FQL row depends
on the pretraining rows it is compared with); our own real-world pretraining stays deferred until the architecture is
final.

Gating per row: the 100-episode gate of its task, plus the calibration hold for GEOMETRY-DEPENDENT consumers
(``s4d.rwlearn.gate.CONSUMERS``). A plan under the hold emits the non-geometric rows and lists the held ones under
``refused`` with the gate's reasons.
"""

from __future__ import annotations

import hashlib
import json

from s4d.rwlearn.configs import BASELINES, REMOTE_REALWORLD_ROOT, run_inputs_dir
from s4d.rwlearn.configs import TARGET_TASKS as TARGET_TASKS
from s4d.rwlearn.gate import FQL_CONSUMER, consumer_info, task_passed, task_problems

REMOTE_HOST = "192.168.10.22"
TASKS = ("drawer-open", "mug-hanging")
PLACEHOLDER_COMMIT = "<release commit pinned by the coordinator>"
SEEDS = (0, 1, 2)
# (GPU GB, host RAM GB): pretraining from the campaign's Meta-World rows of the same trainers (real-world images are the
# same 128 px, three instead of six cameras); FQL measured on the synthetic pipeline tests, scaled to the full stores
RESOURCES = {"snerl": (16, 20), "sincro": (12, 24), "reviwo": (4, 16), "dynamo": (12, 32), "mvp3d": (12, 24)}
FQL_RESOURCES = {"frozen_data": (8, 16), "frozen_run": (6, 12), "e2e_data": (1, 8), "e2e_run": (12, 24)}
FQL_ENCODER = {"dynamo": "dynamo", "reviwo": "reviwo", "sincro": "sincro", "snerl": "snerl",
               "mvp3d_multiview": "mvp3d", "mvp3d_single": "mvp3d", "e2e_cnn": None}
FQL_CONDITIONS = tuple(FQL_ENCODER)
RESUME_ARGS = {"snerl": ["--resume", "auto"], "dynamo": ["--resume", "auto"], "mvp3d": ["--resume", "auto"],
               "sincro": [], "reviwo": []}  # SinCro / ReViWo always resume from latest.pt under the same --name
PRETRAIN_PRIORITY, FQL_PRIORITY = 1, 2  # lower runs first: baselines before FQL
FQL_BUDGET = {"offline_steps": 1_000_000, "batch_size": 256, "alpha": 10.0, "log_interval": 5000,
              "eval_interval": 100_000}


def pretrain_row_id(baseline: str, task: str) -> str:
    return f"rw-{baseline}-{task}-pre"


def _gate_fields(consumer: str, gate_sha: str, commit: str) -> dict:
    c = consumer_info(consumer)
    return {"realworld_consumer": consumer, "calibration_dependent": c.geometry,
            "gate_level": "geometry (100-episode gate + calibration_validated)" if c.geometry else
            "image (100-episode gate)", "realworld_gate_sha256": gate_sha, "planned_commit": commit,
            "remote_host": REMOTE_HOST}


def pretrain_rows(task: str, commit: str, gate_sha: str, baselines=BASELINES) -> list[dict]:
    rows = []
    inputs = run_inputs_dir(task)
    for b in baselines:
        mem, ram = RESOURCES[b]
        rid = pretrain_row_id(b, task)
        rows.append({
            "id": rid, "script": f"scripts/train_{b}.py",
            "args": ["--config", f"configs/realworld/baselines/{b}/{task}.yaml", "--name", rid,
                     "--output-root", "runs/pretrain", *RESUME_ARGS[b]],
            "role": "realworld_pretrain", "host": "remote", "gpu": "any", "mem_gb": mem, "ram_gb": ram,
            "priority": PRETRAIN_PRIORITY, "deps": [], "max_restarts": 2, "oom_score_adj": 0, "baseline": b,
            "realworld_target": task, "pretraining": "task-specific, documented protocol (configs/realworld/baselines)",
            **_gate_fields(b, gate_sha, commit),
            "inputs": [f"{inputs}/store.yaml", f"{inputs}/gate_report.json", f"{inputs}/split_seed0.json"]})
    return rows


def fql_rows(task: str, commit: str, gate_sha: str, seeds=SEEDS, conditions=FQL_CONDITIONS,
             e2e_after=("dynamo", "reviwo")) -> list[dict]:
    """Per condition: one dataset build and one FQL run per seed. Frozen-encoder conditions depend on their encoder's
    pretraining row; ``e2e_cnn`` (no pretraining) waits for the task's non-geometric baseline pretrainings
    ``e2e_after`` (baselines first; it is not held by the calibration of the geometric ones)."""
    rows = []
    inputs = run_inputs_dir(task)
    for cond in conditions:
        enc = FQL_ENCODER[cond]
        consumer = FQL_CONSUMER[cond]
        data_id = f"rw-fqldata-{cond}-{task}"
        data_path = f"runs/realworld_fql/data/{cond}-{task}.h5"
        common_in = ["--store", f"{inputs}/store.yaml", "--condition", cond, "--out", data_path,
                     "--gate-report", f"{inputs}/gate_report.json", "--outcomes", f"{inputs}/outcomes.json"]
        if enc is None:
            mem, ram = FQL_RESOURCES["e2e_data"]
            args, deps = [*common_in, "--device", "cpu"], [pretrain_row_id(b, task) for b in e2e_after]
        else:
            mem, ram = FQL_RESOURCES["frozen_data"]
            args = [*common_in, "--export", f"runs/pretrain/{pretrain_row_id(enc, task)}/encoder.pt", "--device",
                    "cuda"]
            deps = [pretrain_row_id(enc, task)]
        rows.append({"id": data_id, "script": "scripts/realworld_fql_dataset.py", "args": ["build", *args],
                     "role": "realworld_fql", "host": "remote", "gpu": "any", "mem_gb": mem, "ram_gb": ram,
                     "priority": FQL_PRIORITY, "deps": deps, "max_restarts": 1, "oom_score_adj": 0,
                     "realworld_target": task, "fql_condition": cond, **_gate_fields(consumer, gate_sha, commit)})
        mem, ram = FQL_RESOURCES["e2e_run" if enc is None else "frozen_run"]
        for seed in seeds:
            rid = f"rw-fql-{cond}-{task}-s{seed}"
            sets = [f"condition={cond}", f"data.dataset={data_path}", f"data.split_manifest={inputs}/split_seed0.json",
                    f"data.gate_report={inputs}/gate_report.json", f"train.seed={seed}"]
            if enc is None:
                sets.append(f"data.store={inputs}/store.yaml")
            rows.append({
                "id": rid, "script": "scripts/train_realworld_fql.py",
                "args": ["--config", "configs/realworld/fql/base.yaml", "--name", rid,
                         "--output-root", "runs/realworld_fql", "--set", *sets],
                "role": "realworld_fql", "host": "remote", "gpu": "any", "mem_gb": mem, "ram_gb": ram,
                "priority": FQL_PRIORITY, "deps": [data_id], "max_restarts": 2, "oom_score_adj": 0,
                "realworld_target": task, "fql_condition": cond, "fql_seed": seed, "fql_budget": dict(FQL_BUDGET),
                **_gate_fields(consumer, gate_sha, commit)})
    return rows


def plan(report: dict, tasks, commit: str, *, stage: str = "all", seeds=SEEDS) -> dict:
    """Rows for every requested task and consumer that passed its gate level; the others are listed under
    ``refused`` with the gate's reasons (no row of a refused consumer, and no FQL row whose pretraining is refused)."""
    gate_sha = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    rows, refused = [], {}
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r} (task-specific targets only: {TASKS})")
        ok_b = [b for b in BASELINES if task_passed(report, task, b)]
        for b in BASELINES:
            if b not in ok_b:
                refused[pretrain_row_id(b, task)] = task_problems(report, task, b)
        if stage in ("all", "pretrain"):
            rows += pretrain_rows(task, commit, gate_sha, baselines=ok_b)
        if stage in ("all", "fql"):
            conds = []
            for cond in FQL_CONDITIONS:
                enc = FQL_ENCODER[cond]
                need = [enc] if enc else ["dynamo", "reviwo"]
                if task_passed(report, task, FQL_CONSUMER[cond]) and all(n in ok_b for n in need):
                    conds.append(cond)
                else:
                    refused[f"rw-fql-{cond}-{task}"] = task_problems(report, task, FQL_CONSUMER[cond])
            rows += fql_rows(task, commit, gate_sha, seeds, tuple(conds))
    return {"rows": rows, "refused": refused, "gate_sha256": gate_sha, "remote_root": REMOTE_REALWORLD_ROOT,
            "remote_host": REMOTE_HOST, "registered": False,
            "note": "plan only: register through the release path with the pinned release commit; nothing launched"}


def plan_preview(commit: str = PLACEHOLDER_COMMIT, seeds=SEEDS) -> dict:
    """The full plan as it will be emitted once both tasks pass every gate level (documentation and tests; NOT
    registrable: no gate evidence)."""
    rows = []
    for task in TASKS:
        rows += pretrain_rows(task, commit, "<gate report sha256 at registration>")
        rows += fql_rows(task, commit, "<gate report sha256 at registration>", seeds)
    return {"rows": rows, "registrable": False, "remote_host": REMOTE_HOST, "commit": commit}


def plan_table(preview: dict | None = None) -> list[dict]:
    """One line per (task, baseline / condition): the documented training-plan table."""
    preview = preview or plan_preview()
    out = []
    by_id = {r["id"]: r for r in preview["rows"]}
    for task in TASKS:
        for cond in FQL_CONDITIONS:
            enc = FQL_ENCODER[cond]
            runs = [r for r in preview["rows"] if r.get("fql_condition") == cond and r["realworld_target"] == task
                    and r["script"] == "scripts/train_realworld_fql.py"]
            pre = by_id.get(pretrain_row_id(enc, task)) if enc else None
            out.append({"task": task, "condition": cond, "baseline": enc or "none (end-to-end)",
                        "pretraining": f"{pre['id']} (task-specific, seed from config)" if pre else
                        "none: random-init CNN trained jointly with FQL",
                        "fql_seeds": [r["fql_seed"] for r in runs], "fql_steps": FQL_BUDGET["offline_steps"],
                        "calibration_dependent": runs[0]["calibration_dependent"],
                        "resources_gpu_ram_gb": [runs[0]["mem_gb"], runs[0]["ram_gb"]]})
    return out
