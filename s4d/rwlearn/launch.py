"""Launch PLANS for real-world baseline pretraining and offline FQL on the remote role host (docs/realworld_fql.md,
"Launch tooling"). Nothing here registers, launches or uploads: ``plan`` returns queue rows (``experiments/queue.yaml``
schema plus an explicit ``role:``, as configs/hosts/roles.yaml asks of new real-world rows) that the coordinator
registers through the normal release path. The full-data gate is evaluated first and a plan is refused for any target
whose task(s) did not pass; joint targets need both tasks.
"""

from __future__ import annotations

import hashlib
import json

from s4d.rwlearn.configs import BASELINES, REMOTE_REALWORLD_ROOT, run_inputs_dir

TARGET_TASKS = {"drawer-open": ("drawer-open",), "mug-hanging": ("mug-hanging",),
                "joint": ("drawer-open", "mug-hanging")}
# per-baseline resources (from the campaign's Meta-World rows of the same trainers; real-world images are the same
# 128 px, three instead of six cameras)
RESOURCES = {"snerl": (16, 20), "sincro": (12, 24), "reviwo": (4, 16), "dynamo": (12, 32), "mvp3d": (12, 24)}
FQL_CONDITIONS = {"dynamo": "dynamo", "reviwo": "reviwo", "sincro": "sincro", "snerl": "snerl",
                  "mvp3d_multiview": "mvp3d", "mvp3d_single": "mvp3d"}
RESUME_ARGS = {"snerl": ["--resume", "auto"], "dynamo": ["--resume", "auto"], "mvp3d": ["--resume", "auto"],
               "sincro": [], "reviwo": []}  # SinCro / ReViWo always resume from latest.pt under the same --name


def gate_passed_for(report: dict, target: str) -> bool:
    return bool(report.get("passed") or all((report.get("tasks") or {}).get(t, {}).get("passed")
                                            for t in TARGET_TASKS[target])) and all(
        (report.get("tasks") or {}).get(t, {}).get("passed") for t in TARGET_TASKS[target])


def pretrain_rows(target: str, commit: str, gate_sha: str) -> list[dict]:
    rows = []
    for b in BASELINES:
        mem, ram = RESOURCES[b]
        rid = f"rw-{b}-{target}-pre"
        rows.append({
            "id": rid, "script": f"scripts/train_{b}.py",
            "args": ["--config", f"configs/realworld/baselines/{b}/{target}.yaml", "--name", rid,
                     "--output-root", "runs/pretrain", *RESUME_ARGS[b]],
            "role": "realworld_pretrain", "host": "remote", "gpu": "any", "mem_gb": mem, "ram_gb": ram,
            "priority": 1, "deps": [], "max_restarts": 2, "oom_score_adj": 0, "baseline": b,
            "realworld_target": target, "realworld_gate_sha256": gate_sha, "planned_commit": commit,
            "inputs": [f"{run_inputs_dir(target)}/store.yaml", f"{run_inputs_dir(target)}/gate_report.json",
                       f"{run_inputs_dir(target)}/split_seed0.json"]})
    return rows


def fql_rows(task: str, encoder_target: str, commit: str, gate_sha: str, seeds=(0, 1, 2),
             conditions=tuple(FQL_CONDITIONS)) -> list[dict]:
    """Per condition: one feature-dataset build (depends on the encoder's pretraining row) and one FQL run per seed.
    ``encoder_target`` is the pretraining target whose encoder is used (the task itself by default; ``joint`` is the
    open decision of training FQL per task on the jointly pretrained encoder)."""
    rows = []
    inputs = run_inputs_dir(task)
    for cond in conditions:
        b = FQL_CONDITIONS[cond]
        enc_row = f"rw-{b}-{encoder_target}-pre"
        data_id = f"rw-fqldata-{cond}-{task}-enc{encoder_target}"
        data_path = f"runs/realworld_fql/data/{cond}-{task}-enc{encoder_target}.h5"
        rows.append({
            "id": data_id, "script": "scripts/realworld_fql_dataset.py",
            "args": ["build", "--store", f"{inputs}/store.yaml", "--condition", cond,
                     "--export", f"runs/pretrain/{enc_row}/encoder.pt", "--out", data_path,
                     "--gate-report", f"{inputs}/gate_report.json", "--device", "cuda"],
            "role": "realworld_fql", "host": "remote", "gpu": "any", "mem_gb": 8, "ram_gb": 16, "priority": 1,
            "deps": [enc_row], "max_restarts": 1, "oom_score_adj": 0, "realworld_target": task,
            "fql_condition": cond, "realworld_gate_sha256": gate_sha, "planned_commit": commit})
        for seed in seeds:
            rid = f"rw-fql-{cond}-{task}-enc{encoder_target}-s{seed}"
            rows.append({
                "id": rid, "script": "scripts/train_realworld_fql.py",
                "args": ["--config", "configs/realworld/fql/base.yaml", "--name", rid,
                         "--output-root", "runs/realworld_fql", "--set", f"condition={cond}",
                         f"data.dataset={data_path}", f"data.split_manifest={inputs}/split_seed0.json",
                         f"data.gate_report={inputs}/gate_report.json", f"train.seed={seed}"],
                "role": "realworld_fql", "host": "remote", "gpu": "any", "mem_gb": 6, "ram_gb": 12, "priority": 1,
                "deps": [data_id], "max_restarts": 2, "oom_score_adj": 0, "realworld_target": task,
                "fql_condition": cond, "fql_seed": seed, "realworld_gate_sha256": gate_sha, "planned_commit": commit})
    return rows


def plan(report: dict, targets, commit: str, *, stage: str = "all", seeds=(0, 1, 2), fql_encoder: str = "task") -> dict:
    """Rows for every requested target that passed the gate; refused targets are listed with the gate's reasons."""
    gate_sha = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    rows, refused = [], {}
    for target in targets:
        tasks = TARGET_TASKS[target]
        failing = [t for t in tasks if not (report.get("tasks") or {}).get(t, {}).get("passed")]
        if failing:
            refused[target] = {t: (report.get("tasks") or {}).get(t, {}).get("problems", ["not evaluated"])[:5]
                               for t in failing}
            continue
        if stage in ("all", "pretrain"):
            rows += pretrain_rows(target, commit, gate_sha)
        if stage in ("all", "fql") and target != "joint":
            enc = target if fql_encoder == "task" else "joint"
            if enc == "joint" and not all((report.get("tasks") or {}).get(t, {}).get("passed")
                                          for t in TARGET_TASKS["joint"]):
                refused[f"fql-{target}-encjoint"] = "joint encoder needs both tasks through the gate"
                continue
            rows += fql_rows(target, enc, commit, gate_sha, seeds)
    return {"rows": rows, "refused": refused, "gate_sha256": gate_sha, "remote_root": REMOTE_REALWORLD_ROOT,
            "registered": False, "note": "plan only: register through the release path; nothing was launched"}
