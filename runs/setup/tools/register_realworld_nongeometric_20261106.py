"""Gate real-world baseline/FQL registration per task and consumer. Default: DRY RUN.

    nice -n 15 /home/ws/ws/splatter4d/.venv/bin/python -I -B \
        runs/setup/tools/register_realworld_nongeometric_20261106.py [--tasks TASK ...]

--publish is the explicit, FUTURE queue mutation path (not used to prepare this tool).
It takes stage2-driver.lock, then registry.lock, evaluates the live manifest under
both locks, stages the queue in runs/setup, journals the exact gate/rows, and uses
os.replace before writing the proof. --recover verifies the saved transaction and
current gates before completing its proof; it never reconstructs evidence from a
new plan. Neither path appends registry events or activates role-release.json.
The proposed release is written only to role-release.proposed.json.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# -I does not suppress bytecode writes to imported sources outside runs/setup.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "runs/setup"
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from ops.jobs import host_roles, scheduler as jobs  # noqa: E402
from s4d.rwlearn import gate, launch  # noqa: E402

COMMIT = "782f14080a25d0d62fbd8594662cdfdb512b5a7a"
IMAGE = "sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a"
RELEASE = {"commit": COMMIT, "image": IMAGE}
PREFIX = "rwng-20261107-"  # 2026-11-07: rows pinned to 782f140 (raw-provenance episode joins); rwng-20261106-* superseded (paused)
STEM = "rw-registration-nongeometric-20261106"
PLAN = ROOT / "docs/realworld_training_plan.yaml"
ACCEPTANCE = ROOT / f"runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json"
PROPOSED = SETUP / "role-release.proposed.json"
ACTIVE_RELEASE = SETUP / "role-release.json"
DRY_PROOF = SETUP / "rwreg-dry-run-20261106.json"
PENDING = SETUP / f"{STEM}.pending.json"
CANDIDATE = SETUP / f"{STEM}.queue-candidate.yaml"
QUEUE = jobs.QUEUE
REGISTRY = jobs.REGISTRY
ADMISSION_LOCK = host_roles.ADMISSION_LOCK
BASELINES = ("reviwo", "dynamo")
CONDITIONS = ("dynamo", "reviwo", "e2e_cnn")
# The plan specifies GPU/RAM, but no disk. Pretraining: existing remote
# ReViWo/DynaMo reservation (20 GB). Dataset/FQL: conservative output-growth
# allowances (20/10 GB); existing shared inputs are not charged again.
DISK_GB = {"pretrain": 20, "dataset": 20, "fql": 10}
SOURCE_PATHS = (
    "s4d/rwlearn/gate.py", "s4d/rwlearn/identity.py", "s4d/rwlearn/launch.py", "s4d/rwlearn/configs.py",
    "s4d/rwlearn/actions.py", "s4d/rwlearn/fql_data.py",
    "scripts/train_reviwo.py", "scripts/train_dynamo.py",
    "scripts/train_realworld_fql.py", "scripts/realworld_fql_dataset.py",
    "configs/realworld/fql/base.yaml",
    *(f"configs/realworld/baselines/{b}/{t}.yaml" for t in launch.TASKS for b in BASELINES),
)


class Refused(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise Refused(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def now():
    return datetime.now(timezone.utc).isoformat()


def kind(row):
    return ("pretrain" if row["role"] == "realworld_pretrain" else
            "dataset" if row["script"] == "scripts/realworld_fql_dataset.py" else "fql")


def atomic_json(path, value):
    require(path.resolve().is_relative_to(SETUP.resolve()), "JSON output must stay inside runs/setup")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def release_evidence():
    raw = ACCEPTANCE.read_bytes()
    manifest = json.loads(raw)
    require(manifest.get("commit") == COMMIT and manifest.get("image_digest") == IMAGE,
            "acceptance manifest does not match the pinned commit/image")
    for section in ("cross_host", "data", "full_suite"):
        rec = manifest.get(section) or {}
        path = ROOT / rec.get("path", "")
        require(path.resolve().is_relative_to(ACCEPTANCE.parent.resolve()) and path.is_file(),
                f"missing acceptance evidence: {section}")
        require(sha(path.read_bytes()) == rec.get("sha256"), f"acceptance hash mismatch: {section}")
    isolation = manifest.get("gpu_isolation") or {}
    require(set(isolation) == set(jobs.APPROVED_HOST_GPUS["remote"]), "acceptance GPU coverage differs")
    for gpu, rec in isolation.items():
        path = ROOT / rec.get("path", "")
        require(path.resolve().is_relative_to(ACCEPTANCE.parent.resolve()) and path.is_file()
                and sha(path.read_bytes()) == rec.get("sha256"), f"acceptance GPU hash mismatch: {gpu}")
    source = {}
    for name in SOURCE_PATHS:
        raw_source = (ROOT / name).read_bytes()
        pinned = subprocess.run(["git", "-C", str(ROOT), "show", f"{COMMIT}:{name}"],
                                check=True, capture_output=True).stdout
        require(raw_source == pinned, f"working source differs from the pinned release: {name}")
        source[name] = sha(raw_source)
    active = json.loads(ACTIVE_RELEASE.read_text()) if ACTIVE_RELEASE.is_file() else None
    return {"commit": COMMIT, "image": IMAGE, "acceptance_manifest": str(ACCEPTANCE.relative_to(ROOT)),
            "acceptance_manifest_sha256": sha(raw), "source_sha256": source,
            "active_role_release": active, "activation_required": active != RELEASE}


def template_rows(tasks):
    """Compare the documented plan with the release's plan builder, then select an explicit allowlist."""
    documented = yaml.safe_load(PLAN.read_text())
    expected = launch.plan_preview()["rows"]
    normalized = copy.deepcopy(documented.get("rows"))
    require(isinstance(normalized, list), "documented plan has no rows")
    # The saved YAML predates the FQL-specific QA family aliases. Only this
    # documented stale field may differ; current launch.py supplies the semantics.
    for row in normalized:
        cond = row.get("fql_condition")
        if cond is not None:
            canonical = launch.FQL_CONSUMER[cond]
            require(row.get("realworld_consumer") in {canonical, launch.FQL_ENCODER[cond] or cond},
                    f"unexpected consumer in documented plan: {row['id']}")
            row["realworld_consumer"] = canonical
    require(normalized == expected, "documented plan drifted from launch.plan_preview() beyond FQL family aliases")
    out = []
    for row in expected:
        if row["realworld_target"] not in tasks:
            continue
        if row.get("baseline") in BASELINES or row.get("fql_condition") in CONDITIONS:
            require(not row["calibration_dependent"] and not gate.consumer_info(row["realworld_consumer"]).geometry,
                    f"geometry consumer excluded: {row['id']}")
            out.append(copy.deepcopy(row))
    require(len(out) == 14 * len(tasks), "expected exactly 14 non-geometric rows per task")
    return out


def adapt_rows(templates, report):
    """Use the scheduler's runs/<id>/ artifact layout; retain all plan compute resources/budgets."""
    mapping = {r["id"]: PREFIX + r["id"][3:] for r in templates}
    paths = {}
    for row in templates:
        old, new = row["id"], mapping[row["id"]]
        if kind(row) == "pretrain":
            paths[f"runs/pretrain/{old}/encoder.pt"] = f"runs/{new}/encoder.pt"
        elif kind(row) == "dataset":
            a = row["args"]
            paths[a[a.index("--out") + 1]] = f"runs/{new}/dataset.h5"
    gate_sha = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    out = []
    for template in templates:
        row = copy.deepcopy(template)
        row["id"] = mapping[template["id"]]
        row["deps"] = [mapping[d] for d in row["deps"]]
        a = row["args"]
        for i, value in enumerate(a):
            if value in mapping:
                a[i] = mapping[value]
            elif value in paths:
                a[i] = paths[value]
            elif value.startswith("data.dataset="):
                a[i] = "data.dataset=" + paths[value.split("=", 1)[1]]
        if "--output-root" in a:
            a[a.index("--output-root") + 1] = "runs"
        k = kind(row)
        row.update(pin_host="remote", planned_commit=COMMIT, planned_image=IMAGE,
                   evidence_manifest=str(ACCEPTANCE.relative_to(ROOT)), disk_gb=DISK_GB[k],
                   realworld_gate_sha256=gate_sha, realworld_manifest_sha256=report["manifest"]["sha256"],
                   registration_tool=f"runs/setup/tools/{Path(__file__).name}",
                   plan_row_id=template["id"],
                   required_results=(["encoder.pt", "metrics.jsonl", "completion.json"] if k == "pretrain" else
                                     ["dataset.h5"] if k == "dataset" else
                                     ["policy.pt", "metrics.jsonl", "completion.json"]))
        # Materialise the decision's budget in the invocation, as well as metadata.
        if k == "fql":
            a += ["agent.alpha=10.0", "agent.batch_size=256", "train.offline_steps=1000000",
                  "train.log_interval=5000", "train.eval_interval=100000"]
        require((row["mem_gb"], row["ram_gb"], row["gpu"], row["priority"], row["max_restarts"]) ==
                (template["mem_gb"], template["ram_gb"], template["gpu"], template["priority"], template["max_restarts"]),
                "plan compute resources or scheduling changed")
        out.append(row)
    # Global baseline-first priority order, preserving task order within each kind.
    return sorted(out, key=lambda r: r["priority"])


def structural(row):
    return {k: v for k, v in row.items() if k not in {"realworld_gate_sha256", "realworld_manifest_sha256"}}


def published_rows():
    known = {}
    for path in SETUP.glob(f"{STEM}-*.json"):
        proof = json.loads(path.read_text())
        require(proof.get("status") == "published" and proof.get("schema") == "rw-registration/v1",
                f"invalid published proof: {path.name}")
        require(json_sha(proof["jobs"]) == proof.get("jobs_sha256"), f"proof rows changed: {path.name}")
        for row in proof["jobs"]:
            require(row["id"] not in known, f"duplicate published proof: {row['id']}")
            known[row["id"]] = row
    return known


def select_rows(potential, report, current, known):
    by_id = {r["id"]: r for r in current["jobs"]}
    for rid, row in known.items():
        require(by_id.get(rid) == row, f"published row missing or changed in queue: {rid}")
    for row in current["jobs"]:
        if row["id"].startswith(PREFIX):
            require(row["id"] in known, f"unproved ID collision: {row['id']} (use --recover for a pending transaction)")
    accepted, refused = [], {}
    available = set(known)
    for row in potential:
        rid, task, consumer = row["id"], row["realworld_target"], row["realworld_consumer"]
        if rid in by_id:
            require(structural(by_id[rid]) == structural(row), f"existing row differs from the plan: {rid}")
            continue
        reasons = gate.task_problems(report, task, consumer) if not gate.task_passed(report, task, consumer) else []
        if not gate.task_passed(report, task, consumer) and not reasons:
            reasons = [report.get("error", "consumer gate refused")]
        missing = [d for d in row["deps"] if d not in available]
        if missing:
            reasons.append("dependencies not gated/registered: " + ", ".join(missing))
        if reasons:
            refused[rid] = reasons
        else:
            accepted.append(row)
            available.add(rid)
    return accepted, refused


def check_rows(rows, report, current):
    roles = host_roles.load_roles()
    by_id = {r["id"]: r for r in current["jobs"] + rows}
    for row in rows:
        require(not gate.consumer_info(row["realworld_consumer"]).geometry and not row["calibration_dependent"],
                f"geometry consumer forbidden: {row['id']}")
        require(gate.task_passed(report, row["realworld_target"], row["realworld_consumer"]),
                f"gate refused: {row['id']}")
        require(host_roles.role_host(row, roles, by_id) == "remote", f"not a remote role: {row['id']}")
        require(host_roles.refusal(row, "remote", roles, {}, by_id) is None, f"remote role refused: {row['id']}")
        require(row["host"] == row["pin_host"] == "remote" and row["planned_commit"] == COMMIT
                and row["planned_image"] == IMAGE, f"row release/placement mismatch: {row['id']}")
    return roles["sha256"]


def stage_queue(before, rows):
    parsed = yaml.safe_load(before)
    node = next(v for k, v in yaml.compose(before.decode()).value if k.value == "jobs")
    require(isinstance(node, yaml.SequenceNode), "queue jobs must be a sequence")
    addition = yaml.safe_dump(rows, sort_keys=False)
    text = before.decode()
    if node.flow_style or not node.value:
        replacement = "\n" + yaml.safe_dump(parsed["jobs"] + rows, sort_keys=False)
        candidate = text[:node.start_mark.index] + replacement + text[node.end_mark.index:]
    else:
        candidate = text[:node.end_mark.index].rstrip() + "\n" + addition + "\n" + text[node.end_mark.index:]
    require(yaml.safe_load(candidate) == {**parsed, "jobs": parsed["jobs"] + rows}, "staging changed existing queue data")
    with CANDIDATE.open("w") as stream:
        stream.write(candidate)
        stream.flush()
        os.fsync(stream.fileno())
    require(jobs.load_queue(CANDIDATE) == {**jobs.load_queue(QUEUE), "jobs": jobs.load_queue(QUEUE)["jobs"] + rows},
            "candidate does not meet the scheduler schema")
    return sha(candidate.encode())


def counts(rows):
    return dict(Counter(kind(r) for r in rows))


def prepare(tasks):
    evidence = release_evidence()
    templates = template_rows(tasks)
    # Only the authoritative, read-only manifest; no target/stage/manifest override.
    report = gate.evaluate_file(gate.MANIFEST, tasks)
    potential = adapt_rows(templates, report)
    current = jobs.load_queue(QUEUE)
    selected, refused = select_rows(potential, report, current, published_rows())
    roles_sha = check_rows(selected, report, current)
    state = jobs.read_registry(REGISTRY)
    for row in selected:
        require(row["id"] not in state, f"ID already exists in registry: {row['id']}")
        require(not (ROOT / "runs" / row["id"]).exists(), f"run directory exists: {row['id']}")
    return {"schema": "rw-registration/v1", "status": "dry-run", "prepared_at": now(), "tasks": list(tasks),
            "release": evidence, "plan_sha256": sha(PLAN.read_bytes()), "roles_sha256": roles_sha,
            "queue_sha256_at_evaluation": sha(QUEUE.read_bytes()),
            "registry_sha256_at_evaluation": sha(REGISTRY.read_bytes()),
            "tool_sha256": sha(Path(__file__).read_bytes()), "gate_evaluation": report,
            "potential_counts_per_task": {t: counts([r for r in potential if r["realworld_target"] == t]) for t in tasks},
            "eligible_counts_per_task": {t: counts([r for r in selected if r["realworld_target"] == t]) for t in tasks},
            "refused": refused, "jobs": selected, "jobs_sha256": json_sha(selected),
            "potential_jobs_not_registrable": potential,
            "plan_adjustment": "FQL consumer aliases from current launch.py supersede stale preview labels",
            "disk_reservations_gb": DISK_GB, "disk_note": "plan GPU/RAM preserved; added output-growth allowances",
            "decisions": {"alpha": 10, "gripper": "continuous", "encoders": "task-specific",
                          "fql_seeds": list(launch.SEEDS), "e2e": "random-init CNN trained jointly with FQL",
                          "pretraining_budgets": "unchanged release configs (seed 42)", "fql_budget": launch.FQL_BUDGET}}


def proof_path(record):
    return SETUP / f"{STEM}-{json_sha([r['id'] for r in record['jobs']])[:12]}.json"


@contextmanager
def locks():
    # Same lock order as the established registration tools. Only --publish/--recover enter here.
    with ADMISSION_LOCK.open("a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with REGISTRY.with_suffix(".lock").open("a") as registry_lock:
            fcntl.flock(registry_lock, fcntl.LOCK_EX)
            yield


def publish(tasks):
    with locks():
        require(not PENDING.exists(), "pending transaction exists; use --recover first")
        record = prepare(tasks)  # gate + collision checks under both locks
        summarize(record, "PUBLISH CHECK")
        active = record["release"]["active_role_release"]
        require(active is None or active == RELEASE,
                "active role-release.json differs from the proposed pin; role rows would launch the wrong release")
        if not record["jobs"]:
            return 2 if record["refused"] else 0
        proof = proof_path(record)
        require(not proof.exists(), f"proof already exists: {proof.name}")
        before = QUEUE.read_bytes()
        record.update(status="pending", queue_sha256_before=sha(before), proof_file=proof.name)
        record["queue_sha256_after"] = stage_queue(before, record["jobs"])
        atomic_json(PENDING, record)  # durable original gate snapshot, before queue mutation
        require(sha(QUEUE.read_bytes()) == sha(before), "queue changed while registration locks held")
        require(sha(gate.MANIFEST.read_bytes()) == record["gate_evaluation"]["manifest"]["sha256"],
                "ingest manifest changed during staging; rerun --recover, then --publish")
        # Same-filesystem rename: no queue/registry writes occur in the default dry run.
        os.replace(CANDIDATE, QUEUE)
        sync_directory(QUEUE.parent)
        record.update(status="published", registered_at=now())
        atomic_json(proof, record)
        PENDING.unlink()
        sync_directory(SETUP)
        print(f"REGISTERED {len(record['jobs'])} rows; proof {proof}")
        return 0


def recover():
    with locks():
        require(PENDING.is_file(), "no pending transaction to recover")
        record = json.loads(PENDING.read_text())
        require(record.get("schema") == "rw-registration/v1" and record.get("status") == "pending",
                "invalid recovery journal")
        rows = record["jobs"]
        require(rows and json_sha(rows) == record.get("jobs_sha256"), "recovery journal rows changed")
        require(record.get("release", {}).get("commit") == COMMIT and record["release"]["image"] == IMAGE,
                "recovery release differs")
        proof = proof_path(record)
        require(record.get("proof_file") == proof.name, "recovery proof path differs")
        before = QUEUE.read_bytes()
        current = jobs.load_queue(QUEUE)
        ids = {r["id"] for r in rows}
        if not ids.intersection(r["id"] for r in current["jobs"]):
            require(sha(before) == record["queue_sha256_before"], "queue changed; journal cannot be safely aborted")
            CANDIDATE.unlink(missing_ok=True)
            PENDING.unlink()
            sync_directory(SETUP)
            print("RECOVERED aborted staging: queue was never replaced; no registration")
            return 0
        require(current["jobs"][-len(rows):] == rows, "queue tail is not exactly the saved registration")
        # Current gates must still pass; original gate evidence is retained in the proof.
        release_evidence()
        original = record["gate_evaluation"]
        expected = {r["id"]: r for r in adapt_rows(template_rows(record["tasks"]), original)}
        require(all(expected.get(row["id"]) == row for row in rows),
                "saved rows differ from this registrar's pinned plan")
        for row in rows:
            require(row["realworld_gate_sha256"] == hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
                    and row["realworld_manifest_sha256"] == original["manifest"]["sha256"], "saved gate hashes differ")
        check_rows(rows, original, current)
        report = gate.evaluate_file(gate.MANIFEST, record["tasks"])
        check_rows(rows, report, current)
        for task in {row["realworld_target"] for row in rows}:
            identities = report["tasks"][task].get("episode_identities")
            require(identities and identities == original["tasks"][task].get("episode_identities"),
                    f"recovery task raw episode identities changed or missing: {task}")
        require(sha(QUEUE.read_bytes()) == sha(before), "queue changed during recovery")
        if proof.exists():
            saved = json.loads(proof.read_text())
            require(saved.get("jobs") == rows and saved.get("queue_sha256_before") == record["queue_sha256_before"]
                    and saved.get("status") == "published", "existing proof differs from pending transaction")
        else:
            record.update(status="published", registered_at="recovered " + now(),
                          recovered_queue_sha256=sha(before), recovery_gate_evaluation=report)
            atomic_json(proof, record)
        CANDIDATE.unlink(missing_ok=True)
        PENDING.unlink()
        sync_directory(SETUP)
        print(f"RECOVERED proof for {len(rows)} rows: {proof}")
        return 0


def summarize(record, mode):
    print("Evaluated " + datetime.now(ZoneInfo("Asia/Seoul")).isoformat())
    print(f"{mode}: {len(record['jobs'])} new rows; {len(record['refused'])} refused rows")
    print(f"Pinned commit {COMMIT}; image {IMAGE}")
    report = record["gate_evaluation"]
    print("Manifest sha256 " + report["manifest"]["sha256"])
    if report.get("error"):
        print("Gate REFUSED: " + report["error"])
    for task in record["tasks"]:
        t = report["tasks"].get(task, {})
        families = report["families"].get(task, {})
        print(f"{task}: {t.get('valid_and_preprocessed', 0)}/100 ready; "
              + ", ".join(f"{f}={'PASS' if s['passed'] else 'REFUSED'}" for f, s in families.items())
              + f"; calibration_validated={report['calibration_hold'].get(task, {}).get('validated', 0)}")
        for consumer in (*BASELINES, "fql_reviwo", "e2e_cnn"):
            ok = gate.task_passed(report, task, consumer)
            reasons = gate.task_problems(report, task, consumer, limit=3)
            print(f"  {consumer}: " + ("PASS" if ok else "REFUSED: " + "; ".join(reasons or [report.get('error', 'gate refused')])))
        print("  New rows: " + json.dumps(record["eligible_counts_per_task"][task], sort_keys=True)
              + "; once all non-geometric gates pass: "
              + json.dumps(record["potential_counts_per_task"][task], sort_keys=True))
    print("Geometry consumers are excluded from this registrar regardless of calibration status.")
    print("Proposed release: " + str(PROPOSED) + "; activation_required="
          + str(record["release"]["activation_required"]).lower())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true", help="explicitly replace the real queue (future operator action)")
    mode.add_argument("--recover", action="store_true", help="recover the exact journaled transaction under both locks")
    mode.add_argument("--dry-run", action="store_true", help="default; write only proposal/evidence inside runs/setup")
    ap.add_argument("--tasks", nargs="+", choices=launch.TASKS, default=list(launch.TASKS))
    args = ap.parse_args()
    require(len(args.tasks) == len(set(args.tasks)), "duplicate task selection")
    if args.recover:
        return recover()
    if args.publish:
        return publish(args.tasks)
    require(not PENDING.exists(), "pending transaction exists; use --recover before another registration")
    record = prepare(args.tasks)
    atomic_json(PROPOSED, RELEASE)
    atomic_json(DRY_PROOF, record)
    # Validate even the unregistrable full preview against the remote scheduler schema.
    preview = SETUP / "rwreg-plan-preview-20261106.yaml"
    atomic_preview = preview.with_suffix(".yaml.tmp")
    atomic_preview.write_text(yaml.safe_dump({"registrable": False,
        "note": "Ungated preview only; publish always reevaluates the authoritative manifest",
        "hosts": {"local": {}, "remote": {}}, "jobs": record["potential_jobs_not_registrable"]}, sort_keys=False))
    jobs.load_queue(atomic_preview)
    os.replace(atomic_preview, preview)
    summarize(record, "DRY-RUN")
    print(f"DRY-RUN proof: {DRY_PROOF}; ungated preview: {preview}")
    print("Queue/registry unchanged. role-release.json was not created or modified.")
    return 2 if not record["jobs"] and record["refused"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Refused, OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
