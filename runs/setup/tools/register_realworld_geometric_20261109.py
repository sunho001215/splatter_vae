"""Register the real-world rows released by the calibration release (agent OPS, 2026-11-09). Default: DRY RUN.

    nice -n 15 /home/ws/ws/splatter4d/.venv/bin/python -I -B \
        runs/setup/tools/register_realworld_geometric_20261109.py [--tasks TASK ...] [--publish | --recover]

Rows (plan rows of ``s4d.rwlearn.launch.plan`` on the LIVE gate report, ids ``rwcal-20261109-<plan id without rw->``):
* GEOMETRY-DEPENDENT, per task: SNeRL (3 input views), 3D-MVP, SinCro pretraining + offline FQL conditions sincro,
  snerl, mvp3d_multiview, mvp3d_single (dataset build + seeds 0, 1, 2; alpha 10, continuous gripper, task-specific
  encoders, single view = random camera per sample fixed through history + paired next obs: the recorded protocol
  in launch.py / fql_data.py / configs/realworld/fql/base.yaml); our own method has no plan row (deferred until the
  architecture is final).
* WAITING ON CALIBRATION ACCEPTANCE (non-geometric, mug-hanging only): ReViWo pretraining (needs the multiview family,
  failed on mug 62 / 75 / 110 while their calibration was rejected) + FQL reviwo + e2e_cnn (whose dataset row waits on
  the task's DynaMo and ReViWo pretrainings; DynaMo = the DONE rwng-20261107-dynamo-mug-hanging-pre).
Everything already registered (rwng-20261107-* / rwng-20261107b-*) is never registered again.

Inputs: the calibration-release learning inputs ``learning_inputs/calrel-6c3b9ec/<task>/`` (store.yaml of the +cal1
stores, the bound gate report with the calibration hold released, split_seed0.json, outcomes.json), written by
``scripts/realworld_learning.py inputs``; the generated configs keep pointing at ``learning_inputs/<task>/`` (used by the
done rows), so pretraining rows override exactly dataset.hdf5_path / data_interface.split_manifest /
data_interface.gate_report with ``--set`` and dataset / FQL rows get the new paths as arguments. A local copy of the
remote inputs (``--inputs-copy``) is verified against the live manifest (store paths = the manifest's train_data outputs,
gate report bound + calibration hold passed for geometry rows) and hashed into the proof.

Release pin: 6c3b9ec (accepted runs/remote/acceptance/6c3b9ec-cc, image 59082834); role rows launch on
runs/setup/role-release.json, which must equal the pin at publication. Same lock order, staging, journal and proof
conventions as register_realworld_nongeometric_20261106.py; no registry events are appended by this tool.
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "runs/setup"
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from ops.jobs import host_roles, scheduler as jobs  # noqa: E402
from s4d.rwlearn import configs as rwconfigs, gate, launch  # noqa: E402

COMMIT = "6c3b9ec0077f50cce7133f54a72910b38b942a23"
IMAGE = "sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a"
ACCEPTANCE = ROOT / f"runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json"
PREFIX = "rwcal-20261109-"
STEM = "rw-registration-calrel-20261109"
ACTIVE_RELEASE = SETUP / "role-release.json"
PENDING = SETUP / f"{STEM}.pending.json"
CANDIDATE = SETUP / f"{STEM}.queue-candidate.yaml"
DRY_PROOF = SETUP / f"{STEM}.dry-run.json"
QUEUE, REGISTRY, ADMISSION_LOCK = jobs.QUEUE, jobs.REGISTRY, host_roles.ADMISSION_LOCK
OLD_INPUTS = {t: rwconfigs.run_inputs_dir(t) for t in launch.TASKS}
NEW_INPUTS = {t: f"{rwconfigs.REMOTE_REALWORLD_ROOT}/learning_inputs/calrel-6c3b9ec/{t}" for t in launch.TASKS}
INPUTS_COPY = Path("/home/ws/ws/realworld-ops/calib-loma/ops/results/learning_inputs")
GEOMETRY_BASELINES = ("snerl", "mvp3d", "sincro")
GEOMETRY_CONDITIONS = ("sincro", "snerl", "mvp3d_multiview", "mvp3d_single")
# rows that waited on the calibration acceptance (mug-hanging multiview family) - non-geometric consumers
WAITING = {"mug-hanging": {"baselines": ("reviwo",), "conditions": ("reviwo", "e2e_cnn")}}
# already DONE pretraining rows of earlier registrations that plan dependencies map to (never re-registered)
EXISTING_PRETRAIN = {"rw-dynamo-mug-hanging-pre": "rwng-20261107-dynamo-mug-hanging-pre"}
DISK_GB = {"pretrain": 20, "dataset": 20, "fql": 10}  # as the non-geometric registration
SOURCE_PATHS = (
    "s4d/rwlearn/gate.py", "s4d/rwlearn/identity.py", "s4d/rwlearn/launch.py", "s4d/rwlearn/configs.py",
    "s4d/rwlearn/actions.py", "s4d/rwlearn/fql_data.py", "s4d/rwlearn/store.py", "s4d/rwlearn/baseline_data.py",
    "scripts/train_snerl.py", "scripts/train_mvp3d.py", "scripts/train_sincro.py", "scripts/train_reviwo.py",
    "scripts/train_realworld_fql.py", "scripts/realworld_fql_dataset.py", "configs/realworld/fql/base.yaml",
    *(f"configs/realworld/baselines/{b}/{t}.yaml" for t in launch.TASKS for b in (*GEOMETRY_BASELINES, "reviwo")),
)
OVERRIDE_KEYS = {"dataset.hdf5_path": "store.yaml", "data_interface.split_manifest": "split_seed0.json",
                 "data_interface.gate_report": "gate_report.json"}


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
        require(path.resolve().is_relative_to(ACCEPTANCE.parent.resolve()) and path.is_file()
                and sha(path.read_bytes()) == rec.get("sha256"), f"acceptance evidence missing or changed: {section}")
    isolation = manifest.get("gpu_isolation") or {}
    require(set(isolation) == set(jobs.APPROVED_HOST_GPUS["remote"]), "acceptance GPU coverage differs")
    for gpu, rec in isolation.items():
        path = ROOT / rec.get("path", "")
        require(path.is_file() and sha(path.read_bytes()) == rec.get("sha256"), f"acceptance GPU hash mismatch: {gpu}")
    source = {}
    for name in SOURCE_PATHS:
        raw_source = (ROOT / name).read_bytes()
        pinned = subprocess.run(["git", "-C", str(ROOT), "show", f"{COMMIT}:{name}"], check=True,
                                capture_output=True).stdout
        require(raw_source == pinned, f"working source differs from the pinned release: {name}")
        source[name] = sha(raw_source)
    active = json.loads(ACTIVE_RELEASE.read_text()) if ACTIVE_RELEASE.is_file() else None
    return {"commit": COMMIT, "image": IMAGE, "acceptance_manifest": str(ACCEPTANCE.relative_to(ROOT)),
            "acceptance_manifest_sha256": sha(raw), "source_sha256": source, "active_role_release": active,
            "activation_required": active != {"commit": COMMIT, "image": IMAGE}}


def inputs_evidence(report, tasks):
    """The local copy of the remote learning inputs must describe exactly the live manifest's +cal1 stores."""
    manifest = json.loads(gate.MANIFEST.read_bytes())
    out = {}
    for task in tasks:
        d = INPUTS_COPY / task
        files = {n: d / n for n in ("store.yaml", "gate_report.json", "split_seed0.json", "outcomes.json")}
        require(all(p.is_file() for p in files.values()), f"{task}: local copy of the learning inputs incomplete ({d})")
        store = yaml.safe_load(files["store.yaml"].read_text())
        want = sorted(e["preprocessing"]["stages"]["train_data"]["outputs"] for e in manifest["episodes"].values()
                      if e["task"] == task and (e.get("validation") or {}).get("status") == "valid")
        require(store.get("kind") == "compact" and sorted(store.get("paths") or []) == want and len(want) == 100,
                f"{task}: store.yaml does not list exactly the manifest's 100 train_data outputs")
        require(all("/processed/calrel-6c3b9ec/" in p for p in want), f"{task}: train_data not from calrel-6c3b9ec")
        rep = json.loads(files["gate_report.json"].read_text())
        t = rep["tasks"][task]
        require(t.get("passed") and len(t.get("store_episodes") or []) == 100, f"{task}: inputs gate report unbound")
        hold = rep["calibration_hold"][task]
        require(hold.get("passed") and len(hold.get("episodes") or {}) == 100,
                f"{task}: inputs gate report has the calibration hold closed")
        live = report["calibration_hold"][task]["episodes"]
        bound = {v["calibrate_build_id"] for v in hold["episodes"].values()}
        require(bound == {v["calibrate_build_id"] for v in live.values()},
                f"{task}: inputs gate report calibrate builds differ from the live manifest's validated builds")
        split = json.loads(files["split_seed0.json"].read_text())
        require(sorted(split["train"] + split["validation"]) == sorted(t["store_episodes"]),
                f"{task}: split is not a partition of the store episodes")
        out[task] = {n: sha(p.read_bytes()) for n, p in files.items()}
        out[task]["remote_dir"] = NEW_INPUTS[task]
    return out


def selected_templates(report, tasks):
    out, refused = [], {}
    for task in tasks:
        result = launch.plan(report, [task], COMMIT, stage="all")
        refused.update(result["refused"])
        wait = WAITING.get(task, {})
        for row in result["rows"]:
            geo = bool(row["calibration_dependent"])
            want = (row.get("baseline") in GEOMETRY_BASELINES or row.get("fql_condition") in GEOMETRY_CONDITIONS
                    if geo else row.get("baseline") in wait.get("baselines", ())
                    or row.get("fql_condition") in wait.get("conditions", ()))
            if want:
                require(geo == gate.consumer_info(row["realworld_consumer"]).geometry, f"class mismatch: {row['id']}")
                out.append(copy.deepcopy(row))
    return out, refused


def adapt_rows(templates, report):
    mapping = {r["id"]: PREFIX + r["id"][3:] for r in templates}
    mapping.update(EXISTING_PRETRAIN)
    paths = {}
    for row in templates:
        if kind(row) == "pretrain":
            paths[f"runs/pretrain/{row['id']}/encoder.pt"] = f"runs/{mapping[row['id']]}/encoder.pt"
        elif kind(row) == "dataset":
            a = row["args"]
            paths[a[a.index("--out") + 1]] = f"runs/{mapping[row['id']]}/dataset.h5"
    for old, new in EXISTING_PRETRAIN.items():
        paths[f"runs/pretrain/{old}/encoder.pt"] = f"runs/{new}/encoder.pt"
    gate_sha = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    out = []
    for template in templates:
        row = copy.deepcopy(template)
        task = row["realworld_target"]
        old_in, new_in = OLD_INPUTS[task], NEW_INPUTS[task]
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
            if old_in + "/" in a[i]:
                a[i] = a[i].replace(old_in + "/", new_in + "/")
        if "--output-root" in a:
            a[a.index("--output-root") + 1] = "runs"
        k = kind(row)
        if k == "pretrain":  # the generated config points at learning_inputs/<task>/ (done rows): override 3 keys
            a += ["--set", *(f"{key}={new_in}/{name}" for key, name in OVERRIDE_KEYS.items())]
        if row.get("inputs"):
            row["inputs"] = [p.replace(old_in + "/", new_in + "/") for p in row["inputs"]]
        if k == "fql":
            a += ["agent.alpha=10.0", "agent.batch_size=256", "train.offline_steps=1000000",
                  "train.log_interval=5000", "train.eval_interval=100000"]
        require(not any(old_in + "/" in str(x) for x in a + row.get("inputs", [])), f"old inputs left: {row['id']}")
        row.update(pin_host="remote", planned_commit=COMMIT, planned_image=IMAGE,
                   evidence_manifest=str(ACCEPTANCE.relative_to(ROOT)), disk_gb=DISK_GB[k],
                   realworld_gate_sha256=gate_sha, realworld_manifest_sha256=report["manifest"]["sha256"],
                   realworld_inputs=new_in, calibration_release="rig_loma a935aef (calrel-6c3b9ec)",
                   registration_tool=f"runs/setup/tools/{Path(__file__).name}", plan_row_id=template["id"],
                   required_results=(["encoder.pt", "metrics.jsonl", "completion.json"] if k == "pretrain" else
                                     ["dataset.h5"] if k == "dataset" else
                                     ["policy.pt", "metrics.jsonl", "completion.json"]))
        require((row["mem_gb"], row["ram_gb"], row["gpu"], row["priority"], row["max_restarts"]) ==
                (template["mem_gb"], template["ram_gb"], template["gpu"], template["priority"],
                 template["max_restarts"]), "plan compute resources or scheduling changed")
        out.append(row)
    return sorted(out, key=lambda r: r["priority"])


def check_rows(rows, report, current, state):
    roles = host_roles.load_roles()
    by_id = {r["id"]: r for r in current["jobs"] + rows}
    for row in rows:
        require(row["id"].startswith(PREFIX), f"unexpected id {row['id']}")
        require(row["id"] not in {r["id"] for r in current["jobs"]}, f"ID already in queue: {row['id']}")
        require(row["id"] not in state, f"ID already exists in registry: {row['id']}")
        require(not (ROOT / "runs" / row["id"]).exists(), f"run directory exists: {row['id']}")
        require(gate.task_passed(report, row["realworld_target"], row["realworld_consumer"]),
                f"gate refused: {row['id']}")
        if row["calibration_dependent"]:
            require(report["calibration_hold"][row["realworld_target"]]["passed"], f"calibration hold: {row['id']}")
        require(host_roles.role_host(row, roles, by_id) == "remote", f"not a remote role: {row['id']}")
        require(host_roles.refusal(row, "remote", roles, {}, by_id) is None, f"remote role refused: {row['id']}")
        for dep in row["deps"]:
            require(dep in by_id, f"unknown dependency {dep} of {row['id']}")
            if not dep.startswith(PREFIX):
                require(state.get(dep, {}).get("status") == jobs.DONE, f"reused dependency not DONE: {dep}")
    return roles["sha256"]


def stage_queue(before, rows):
    parsed = yaml.safe_load(before)
    node = next(v for k, v in yaml.compose(before.decode()).value if k.value == "jobs")
    require(isinstance(node, yaml.SequenceNode) and not node.flow_style and node.value, "unexpected queue layout")
    text = before.decode()
    candidate = text[:node.end_mark.index].rstrip() + "\n" + yaml.safe_dump(rows, sort_keys=False) + "\n" \
        + text[node.end_mark.index:]
    require(yaml.safe_load(candidate) == {**parsed, "jobs": parsed["jobs"] + rows}, "staging changed existing queue data")
    with CANDIDATE.open("w") as stream:
        stream.write(candidate)
        stream.flush()
        os.fsync(stream.fileno())
    require(jobs.load_queue(CANDIDATE) == {**jobs.load_queue(QUEUE), "jobs": jobs.load_queue(QUEUE)["jobs"] + rows},
            "candidate does not meet the scheduler schema")
    return sha(candidate.encode())


def counts(rows):
    return dict(Counter(f"{'geometry' if r['calibration_dependent'] else 'image'}-{kind(r)}" for r in rows))


def prepare(tasks):
    evidence = release_evidence()
    report = gate.evaluate_file(gate.MANIFEST, tasks)
    for task in tasks:
        require(report["tasks"][task]["passed"], f"{task}: 100-episode gate refused: {report['tasks'][task]['problems'][:3]}")
        require(report["calibration_hold"][task]["passed"],
                f"{task}: calibration hold closed: {report['calibration_hold'][task]['problems'][:3]}")
    templates, refused = selected_templates(report, tasks)
    rows = adapt_rows(templates, report)
    current = jobs.load_queue(QUEUE)
    state = jobs.read_registry(REGISTRY)
    roles_sha = check_rows(rows, report, current, state)
    inputs = inputs_evidence(report, tasks)
    return {"schema": "rw-registration/v1", "status": "dry-run", "prepared_at": now(), "tasks": list(tasks),
            "registration_mode": "calibration-release", "release": evidence, "roles_sha256": roles_sha,
            "queue_sha256_at_evaluation": sha(QUEUE.read_bytes()),
            "registry_sha256_at_evaluation": sha(REGISTRY.read_bytes()),
            "tool_sha256": sha(Path(__file__).read_bytes()), "gate_evaluation": report, "learning_inputs": inputs,
            "plan_refused": refused, "counts_per_task": {t: counts([r for r in rows if r["realworld_target"] == t])
                                                         for t in tasks},
            "jobs": rows, "jobs_sha256": json_sha(rows), "disk_reservations_gb": DISK_GB,
            "not_registered": {"ours": "no plan row: our own real-world pretraining is deferred until the architecture "
                                       "is final (launch.py, docs/realworld_fql.md) - user decision"},
            "decisions": {"alpha": 10, "gripper": "continuous", "encoders": "task-specific", "snerl_inputs": 3,
                          "single_view": "random camera per sample, fixed through history + paired next obs",
                          "fql_seeds": list(launch.SEEDS), "fql_budget": launch.FQL_BUDGET,
                          "priorities": "plan priorities kept (pretraining 1, FQL 2); remote had no other admissible "
                                        "work at registration"}}


def proof_path(record):
    return SETUP / f"{STEM}-{json_sha([r['id'] for r in record['jobs']])[:12]}.json"


@contextmanager
def locks():
    with ADMISSION_LOCK.open("a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with REGISTRY.with_suffix(".lock").open("a") as registry_lock:
            fcntl.flock(registry_lock, fcntl.LOCK_EX)
            yield


def publish(tasks):
    with locks():
        require(not PENDING.exists(), "pending transaction exists; use --recover first")
        record = prepare(tasks)
        summarize(record, "PUBLISH CHECK")
        require(not record["release"]["activation_required"],
                "role-release.json differs from the pin; role rows would launch the wrong release")
        require(record["jobs"], "no rows")
        proof = proof_path(record)
        require(not proof.exists(), f"proof already exists: {proof.name}")
        before = QUEUE.read_bytes()
        record.update(status="pending", queue_sha256_before=sha(before), proof_file=proof.name)
        record["queue_sha256_after"] = stage_queue(before, record["jobs"])
        atomic_json(PENDING, record)
        require(sha(QUEUE.read_bytes()) == sha(before), "queue changed while registration locks held")
        require(sha(gate.MANIFEST.read_bytes()) == record["gate_evaluation"]["manifest"]["sha256"],
                "ingest manifest changed during staging; rerun --recover, then --publish")
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
        rows = record["jobs"]
        require(rows and json_sha(rows) == record.get("jobs_sha256"), "recovery journal rows changed")
        before = QUEUE.read_bytes()
        current = jobs.load_queue(QUEUE)
        if not {r["id"] for r in rows}.intersection(r["id"] for r in current["jobs"]):
            require(sha(before) == record["queue_sha256_before"], "queue changed; journal cannot be safely aborted")
            CANDIDATE.unlink(missing_ok=True)
            PENDING.unlink()
            print("RECOVERED aborted staging: queue was never replaced; no registration")
            return 0
        require(current["jobs"][-len(rows):] == rows, "queue tail is not exactly the saved registration")
        proof = SETUP / record["proof_file"]
        if not proof.exists():
            record.update(status="published", registered_at="recovered " + now(), recovered_queue_sha256=sha(before))
            atomic_json(proof, record)
        CANDIDATE.unlink(missing_ok=True)
        PENDING.unlink()
        print(f"RECOVERED proof for {len(rows)} rows: {proof}")
        return 0


def summarize(record, mode):
    print(f"{mode}: {len(record['jobs'])} rows; pinned {record['release']['commit'][:7]} image "
          f"{record['release']['image'][:19]}; role-release activation_required="
          f"{str(record['release']['activation_required']).lower()}")
    rep = record["gate_evaluation"]
    print("Manifest sha256 " + rep["manifest"]["sha256"])
    for task in record["tasks"]:
        fam = rep["families"][task]
        print(f"{task}: gate {'PASS' if rep['tasks'][task]['passed'] else 'REFUSED'}; families "
              + ", ".join(f"{f}={'PASS' if v['passed'] else 'FAIL'}" for f, v in fam.items())
              + f"; calibration_hold {'PASS' if rep['calibration_hold'][task]['passed'] else 'CLOSED'} "
              f"({rep['calibration_hold'][task]['validated']}/100); rows {record['counts_per_task'][task]}")
    for row in record["jobs"]:
        print(f"  {row['id']:52s} prio {row['priority']} deps {row['deps']}")
    print("Not registered: " + json.dumps(record["not_registered"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--recover", action="store_true")
    ap.add_argument("--tasks", nargs="+", choices=launch.TASKS, default=list(launch.TASKS))
    args = ap.parse_args()
    if args.recover:
        return recover()
    if args.publish:
        return publish(args.tasks)
    require(not PENDING.exists(), "pending transaction exists; use --recover first")
    record = prepare(args.tasks)
    atomic_json(DRY_PROOF, record)
    preview = SETUP / f"{STEM}.preview.yaml"  # the rows against the scheduler schema (not registrable)
    tmp = preview.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump({"registrable": False, "hosts": {"local": {}, "remote": {}},
                                   "jobs": record["jobs"]}, sort_keys=False))
    jobs.load_queue(tmp)
    os.replace(tmp, preview)
    summarize(record, "DRY-RUN")
    print(f"DRY-RUN proof: {DRY_PROOF}; queue/registry unchanged; role-release.json not modified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Refused, OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
