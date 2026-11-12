"""Persistent server roles (user directive 2026-11-06): role declaration, job classes, per-host admission, ownership
records, the scheduler's role gate, role re-placement of never-launched remote Meta-World rows and the explicit
remote -> local migration of a checkpoint-paused run. Synthetic queues and registries only (never the live ones)."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from ops.jobs import host_roles as roles_mod
from ops.jobs import migration
from ops.jobs import scheduler as jobs

REPO = Path(__file__).resolve().parents[1]
ROLES = roles_mod.load_roles()
# Keep regression coverage for the retained exclusive-role replacement/retry/drain machinery using a synthetic policy.
EXCLUSIVE_ROLES = copy.deepcopy(ROLES)
if "metaworld" in EXCLUSIVE_ROLES["hosts"]["remote"]["admit"]:  # shipped without it since 2026-11-12
    EXCLUSIVE_ROLES["hosts"]["remote"]["admit"].remove("metaworld")
EXCLUSIVE_ROLES["hosts"]["remote"]["unknown"] = "allow"
LOCAL_GPU = jobs.APPROVED_HOST_GPUS["local"][0]
REMOTE_GPU = jobs.APPROVED_HOST_GPUS["remote"][0]


def row(job_id, script, *args, host="local", **extra):
    return {"id": job_id, "host": host, "script": script, "args": list(args), "gpu": "any", "mem_gb": 1, "ram_gb": 1,
            "priority": 1, **extra}


DRM = row("cst75-drm-hammer-s2000", "scripts/train_rl.py", "--task", "hammer", "--encoder", "splatter4d", "--seed",
          "2000", "--run-dir", "runs/cst75-drm-hammer-s2000", "--set",
          "vision.export_path=runs/cst75-export-seed0-hammer-200k/encoder.pt", deps=["cst75-export-seed0-hammer-200k"])
DRM_EVAL = row("cst75-drm-hammer-s2000-eval", "scripts/eval_rl.py", "--run-dir", "runs/cst75-drm-hammer-s2000",
               "--train-job", "cst75-drm-hammer-s2000", "--output-dir", "runs/cst75-drm-hammer-s2000-eval")
PRETRAIN = row("cst75-seed0-hammer-200k", "scripts/train.py", "--config", "configs/metaworld/base.yaml",
               "configs/metaworld/tasks/hammer.yaml", "--name", "cst75-seed0-hammer-200k", "--output-root", "runs")
EXPORT = row("cst75-export-seed0-hammer-200k", "scripts/export_encoder.py", "--run", "runs/cst75-seed0-hammer-200k",
             "--checkpoint", "runs/cst75-seed0-hammer-200k/checkpoints/step_0200000.pt", "--out",
             "runs/cst75-export-seed0-hammer-200k/encoder.pt", deps=["cst75-seed0-hammer-200k"])
SNERL = row("s4-snerl-hammer", "scripts/train_snerl.py", "--config", "configs/baselines/snerl/hammer.yaml")
COLLECT = row("collect-hammer", "scripts/collect_metaworld.py", "--task", "hammer")
ANALYSIS = row("ana-render-hammer", "reports/x/run_render.py", "--task", "hammer")
DROID_PRETRAIN = row("droid-csw4-a", "scripts/train.py", "--config", "configs/droid/csw4_largescale.yaml", "--name",
                     "droid-csw4-a")
DROID_EXPORT = row("droid-csw4-a-export", "scripts/export_encoder.py", "--run", "runs/droid-csw4-a", deps=["droid-csw4-a"])
DROID_PRE = row("droid-ls-manifest", "scripts/droid_ls_build_manifest.py", "--out", "runs/x")
RW_PRE = row("rw-preprocess-drawer-open-ep1", "scripts/realworld_preprocess.py", "--episode", "1")
RW_FQL = row("rw-fql-drawer-open-s0", "scripts/train_fql.py", "--dataset", "/home/ws/data/real_world/drawer-open")
RW_FQL_TASK = row("rw-fql-task", "scripts/train_fql.py", "--task", "drawer-open", "--dataset",
                  "/home/ws/data/real_world/drawer-open")
VIEWER = row("rw-viewer", "scripts/realworld_viewer.py", "--episode", "1")
PLAIN = row("plain", "tests/_job_worker.py", "0")
DRM_DROID_ENCODER = row("drm-droid-transfer", "scripts/train_rl.py", "--task", "hammer", "--set",
                        "vision.export_path=runs/droid-csw4-a-export/encoder.pt")
CONTEXT = {r["id"]: r for r in (DRM, DRM_EVAL, PRETRAIN, EXPORT, DROID_PRETRAIN, DROID_EXPORT)}


# ---------------------------------------------------------------------------------------------------- declaration
def test_shipped_role_declaration():
    hosts = ROLES["hosts"]
    assert hosts["local"]["admit"] == ["metaworld", "realworld_pretrain", "droid_pretrain", "realworld_fql"]
    assert hosts["local"]["unknown"] == "allow"
    assert ROLES["version"] == 2 and "scripts" not in hosts["local"]
    assert set(ROLES["data_locality"]["remote_resident"]) == set(roles_mod.CLASSES) - {"metaworld", "verification"}
    # user 2026-11-12: the Meta-World development data was removed from the remote; Meta-World rows run locally
    assert set(hosts["remote"]["admit"]) == {"realworld_preprocess", "droid_preprocess", "realworld_pretrain",
                                             "droid_pretrain", "realworld_fql", "verification"}
    assert hosts["remote"]["unknown"] == "allow" and hosts["remote"]["drain_evaluators"] is True
    assert "drain_evaluators" not in hosts["local"]
    assert hosts["inference"]["scheduler"] is False and hosts["inference"]["admit"] == []
    assert roles_mod.ROLES_PATH == REPO / "configs/hosts/roles.yaml"


@pytest.mark.parametrize("edit", [
    lambda r: r["hosts"]["remote"]["admit"].remove("verification"),  # every class has a host
    lambda r: r["hosts"]["local"]["admit"].append("verification"),
    lambda r: r["hosts"]["local"]["admit"].append("realworld_preprocess"),
    lambda r: r["hosts"]["local"]["admit"].append("droid_preprocess"),
    lambda r: r["data_locality"]["remote_resident"].append("made_up"),
    lambda r: r["hosts"]["local"].update(scripts={"realworld_fql": []}),
    lambda r: r["hosts"]["local"].update(scripts={"realworld_fql": "scripts/train_realworld_fql.py"}),
    lambda r: r["hosts"]["local"].update(scripts={"realworld_fql": ["../train_realworld_fql.py"]}),
    lambda r: r["hosts"]["local"].update(scripts=None),
    lambda r: r["hosts"]["local"].update(scripts={"realworld_fql": [7]}),
    lambda r: r["hosts"]["inference"].update(scripts={}),
    lambda r: r["hosts"]["inference"].update(scheduler=True),
    lambda r: r["hosts"]["inference"]["admit"].append("realworld_preprocess"),
    lambda r: r["hosts"]["remote"].update(unknown="maybe"),
    lambda r: r["hosts"]["local"]["admit"].append("sim"),
    lambda r: r["classes"].pop("realworld_fql"),
    lambda r: r["hosts"].pop("inference"),
    lambda r: r.update(version=1),
    lambda r: r.update(data_locality=None),
    lambda r: r["data_locality"].update(remote_resident=[]),
    lambda r: r["hosts"]["local"].update(drain_evaluators=True),
    lambda r: r["hosts"]["remote"].update(drain_evaluators="yes"),
])
def test_inconsistent_role_declarations_fail_closed(tmp_path, edit):
    declared = yaml.safe_load(roles_mod.ROLES_PATH.read_text())
    edit(declared)
    path = tmp_path / "roles.yaml"
    path.write_text(yaml.safe_dump(declared, sort_keys=False))
    with pytest.raises(ValueError):
        roles_mod.load_roles(path)


@pytest.mark.parametrize('version', [0, 1, 3, None, 2.0, '2', True])
def test_only_integer_version_two_is_accepted(tmp_path, version):
    declared = yaml.safe_load(roles_mod.ROLES_PATH.read_text())
    declared['version'] = version
    path = tmp_path / 'roles.yaml'
    path.write_text(yaml.safe_dump(declared, sort_keys=False))
    with pytest.raises(ValueError, match='unsupported role declaration'):
        roles_mod.load_roles(path)


# ---------------------------------------------------------------------------------------------------- classes
@pytest.mark.parametrize("job,cls,domain", [
    (DRM, "metaworld", "metaworld"), (DRM_EVAL, "metaworld", "metaworld"), (PRETRAIN, "metaworld", "metaworld"),
    (EXPORT, "metaworld", "metaworld"), (SNERL, "metaworld", "metaworld"), (COLLECT, "metaworld", "metaworld"),
    (ANALYSIS, "metaworld", "metaworld"), (DRM_DROID_ENCODER, "metaworld", "metaworld"),
    (DROID_PRETRAIN, "droid_pretrain", "droid"), (DROID_EXPORT, "droid_pretrain", "droid"),
    (DROID_PRE, "droid_preprocess", "droid"), (RW_PRE, "realworld_preprocess", "realworld"),
    (RW_FQL, "realworld_fql", "realworld"), (VIEWER, "unknown", "realworld"), (PLAIN, "unknown", None),
    (RW_FQL_TASK, "ambiguous", None),  # Meta-World and real-world share task names: declare role:
    ({**RW_FQL_TASK, "role": "realworld_fql"}, "realworld_fql", "realworld"),
    ({**PLAIN, "role": "droid_preprocess"}, "droid_preprocess", "droid"),
    ({**DRM, "role": "realworld_pretrain"}, "invalid", None),  # a declared role cannot contradict the script
    ({**PRETRAIN, "role": "droid_pretrain"}, "invalid", None),  # ... or the configuration
    ({**PLAIN, "role": "sim"}, "invalid", None),
    ({**PLAIN, "role": "verification"}, "invalid", None),
    ({**PLAIN, "verification_purpose": "native_suite"}, "verification", None),
])
def test_job_class(job, cls, domain):
    found = roles_mod.job_class(job, CONTEXT)
    assert (found.cls, found.domain) == (cls, domain), found


def test_exports_without_their_producer_in_context_are_unclassified():
    assert roles_mod.job_class(EXPORT).cls == "unknown"  # a dispatch queue holds only the admitted rows
    assert roles_mod.job_class({**EXPORT, "deps": ["missing"]}, CONTEXT).cls == "unknown"


# ---------------------------------------------------------------------------------------------------- admission
@pytest.mark.parametrize("job,local,remote", [
    # 2026-11-12 (user): no Meta-World development data on the remote -> Meta-World rows run locally only
    (DRM, True, False), (PRETRAIN, True, False), (EXPORT, True, False), (SNERL, True, False),
    (DROID_PRETRAIN, False, True), (DROID_PRE, False, True), (RW_PRE, False, True), (RW_FQL, False, True),
    (VIEWER, False, False), (PLAIN, True, True), (RW_FQL_TASK, False, False),
    ({**DRM, "role": "realworld_pretrain"}, False, False),
])
def test_host_admission_matrix(job, local, remote):
    assert (roles_mod.refusal(job, "local", ROLES, {}, CONTEXT) is None) is local
    assert (roles_mod.refusal(job, "remote", ROLES, {}, CONTEXT) is None) is remote
    assert roles_mod.refusal(job, "inference", ROLES, {}, CONTEXT) == \
        "inference is not a scheduler host (configs/hosts/roles.yaml)"
    assert roles_mod.role_host(job, ROLES, CONTEXT) == ("local" if local else
                                                         "remote" if remote else None)


def ownership(job_id, state, owner, **extra):
    return {job_id: {"job_id": job_id, "state": state, "owner_host": owner, "from_host": "remote", "to_host": "local",
                     "kind": "migration", **extra}}


def test_ownership_records_gate_admission_on_both_hosts():
    migrating = ownership(DRM["id"], "migrating", "remote")
    assert "in progress" in roles_mod.refusal(DRM, "local", ROLES, migrating, CONTEXT)
    assert roles_mod.ownership_refusal(DRM["id"], "remote", migrating).startswith("migration remote -> local")
    switched = ownership(DRM["id"], "switched", "local")
    assert roles_mod.refusal(DRM, "local", ROLES, switched, CONTEXT) is None
    assert roles_mod.ownership_refusal(DRM["id"], "remote", switched).startswith("owned by local")
    aborted = ownership(DRM["id"], "aborted", "remote")
    assert roles_mod.refusal(DRM, "local", ROLES, aborted, CONTEXT).startswith("owned by remote")


def test_explicit_retry_in_place_exceptions(tmp_path):
    path = tmp_path / "role-exceptions.json"
    assert roles_mod.load_exceptions(path) == {}
    grant = {"id": DRM["id"], "host": "remote", "scope": "retry_in_place", "reason": "remote outage at 990k",
             "granted": "2026-11-06T03:00:00+09:00", "attempt": 1, "job_sha256": "1" * 64}
    path.write_text(json.dumps({"exceptions": [grant, {**grant, "id": "old", "active": False}]}))
    exceptions = roles_mod.load_exceptions(path)
    assert set(exceptions) == {DRM["id"]}
    started = {"host": "remote", "attempt": 1, "attempts": 1, "status": "crashed", "job_sha256": "1" * 64,
               "attempt_containers": {"1": {"container": "c", "container_id": "f" * 64}}}
    assert roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, started, exceptions) is None
    assert roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, None, exceptions) is not None  # never launched
    assert roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, {}, CONTEXT,
                             {**started, "host": "local"}, exceptions) is not None
    refused_launch = {**started, "attempt_containers": {}}  # reserved but never executed (no container recorded)
    assert roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, refused_launch, exceptions) is not None
    later = {**started, "attempt": 2, "attempts": 2, "attempt_containers": {"2": {"container_id": "e" * 64}}}
    # The grant covers one attempt.
    assert roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, later, exceptions) is not None
    changed = {**started, "job_sha256": "2" * 64}
    assert roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, changed, exceptions) is not None
    assert roles_mod.refusal(EXPORT, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, started, exceptions) is not None  # not listed
    migrating = ownership(DRM["id"], "migrating", "remote")
    assert "in progress" in roles_mod.refusal(DRM, "remote", EXCLUSIVE_ROLES, migrating, CONTEXT, started, exceptions)
    for bad in ({**grant, "scope": "admit"}, {**grant, "host": "inference"}, {**grant, "reason": ""},
                {k: v for k, v in grant.items() if k != "granted"}, {**grant, "attempt": 0},
                {k: v for k, v in grant.items() if k != "job_sha256"}):
        path.write_text(json.dumps({"exceptions": [bad]}))
        with pytest.raises(ValueError):
            roles_mod.load_exceptions(path)


def test_ownership_records_fail_closed(tmp_path):
    roles_mod.write_ownership({"job_id": "a", "state": "placed", "owner_host": "local"}, tmp_path)
    assert set(roles_mod.read_ownership(tmp_path)) == {"a"}
    (tmp_path / "b.json").write_text(json.dumps({"job_id": "c", "state": "placed", "owner_host": "local"}))
    with pytest.raises(ValueError):
        roles_mod.read_ownership(tmp_path)
    with pytest.raises(ValueError):
        roles_mod.write_ownership({"job_id": "a", "state": "moving", "owner_host": "local"}, tmp_path)


def test_migrated_runs_resume_only_on_the_recorded_execution_scope(tmp_path):
    (tmp_path / "s4d").mkdir()
    (tmp_path / "s4d/x.py").write_text("A = 1\n")
    digest = hashlib.sha256(b"A = 1\n").hexdigest()
    record = ownership("r", "switched", "local", destination={"scope_files": {"s4d/x.py": digest},
                                                               "scope_third_party": {"third_party/a@commit": "c1"}})
    current = {"third_party/a@commit": "c1"}
    assert roles_mod.migrated_source_refusal("r", record, current, tmp_path) is None
    assert roles_mod.migrated_source_refusal("r", record, {**current, "s4d/x.py": "0" * 64}, tmp_path) is not None
    assert roles_mod.migrated_source_refusal("r", record, {"third_party/a@commit": "c2"}, tmp_path) is not None
    (tmp_path / "s4d/x.py").write_text("A = 2\n")
    assert "s4d/x.py differs" in roles_mod.migrated_source_refusal("r", record, current, tmp_path)
    assert roles_mod.migrated_source_refusal("r", ownership("r", "placed", "local"), {}, tmp_path) is None
    assert roles_mod.migrated_source_refusal("r", ownership("r", "switched", "local"), {}, tmp_path) is not None


# ---------------------------------------------------------------------------------------------------- scheduler gate
def scheduler_queue(tmp_path, rows):
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump({"hosts": {"local": {"limits": {LOCAL_GPU: {"max_jobs": 8, "max_mem_gb": 40}}}},
                                    "jobs": rows}))
    return path


def local_tick(tmp_path, monkeypatch, rows, identity=None):
    launched = []
    monkeypatch.setattr(jobs, "launch", lambda job, gpu, attempt, runs: launched.append(job["id"]) or 2**30)
    monkeypatch.setattr(jobs, "source_identity", lambda: identity or {"commit": "c", "source_sha256": {}})
    monkeypatch.setattr(jobs, "disk_usage", lambda _: (2000 * 2**30, 1000 * 2**30))
    messages = jobs.tick(scheduler_queue(tmp_path, rows), tmp_path / "registry.jsonl", tmp_path / "runs", usage={},
                         available_gb=500)
    return launched, messages


def test_scheduler_local_role_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(roles_mod, "OWNERSHIP_DIR", tmp_path / "ownership")
    rows = [{**r, "deps": []} for r in (DRM, PRETRAIN, PLAIN, DROID_PRETRAIN, RW_FQL, VIEWER, RW_FQL_TASK)]
    launched, messages = local_tick(tmp_path, monkeypatch, rows)
    assert launched == [DRM["id"], PRETRAIN["id"], PLAIN["id"]]
    refused = {m.split(":")[0] for m in messages if "not admitted on local" in m}
    assert refused == {DROID_PRETRAIN["id"], RW_FQL["id"], VIEWER["id"], RW_FQL_TASK["id"]}


def test_scheduler_refuses_runs_under_migration_and_checks_the_migrated_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(roles_mod, "OWNERSHIP_DIR", tmp_path / "ownership")
    drm = {**DRM, "deps": []}
    roles_mod.write_ownership({**ownership(drm["id"], "migrating", "remote")[drm["id"]], "time": "t"})
    launched, messages = local_tick(tmp_path, monkeypatch, [drm])
    assert launched == [] and any("migration remote -> local in progress" in m for m in messages)
    roles_mod.write_ownership({**ownership(drm["id"], "switched", "local")[drm["id"]], "time": "t",
                               "destination": {"scope_files": {"s4d/model/encoder.py": "0" * 64},
                                               "scope_third_party": {}}})
    launched, messages = local_tick(tmp_path, monkeypatch, [drm])
    assert launched == [] and any("s4d/model/encoder.py differs" in m for m in messages)
    identity = {"commit": "c", "source_sha256": {"s4d/model/encoder.py": "0" * 64}}
    launched, _ = local_tick(tmp_path, monkeypatch, [drm], identity)
    assert launched == [drm["id"]]


def test_scheduler_admits_nothing_with_an_unreadable_exception_file(tmp_path, monkeypatch):
    monkeypatch.setattr(roles_mod, "EXCEPTIONS_PATH", tmp_path / "role-exceptions.json")
    (tmp_path / "role-exceptions.json").write_text("{not json")
    launched, messages = local_tick(tmp_path, monkeypatch, [{**DRM, "deps": []}])
    assert launched == [] and any("server roles / ownership records unreadable" in m for m in messages)


def test_scheduler_admits_nothing_without_a_readable_role_declaration(tmp_path, monkeypatch):
    monkeypatch.setattr(roles_mod, "ROLES_PATH", tmp_path / "missing.yaml")
    launched, messages = local_tick(tmp_path, monkeypatch, [{**DRM, "deps": []}])
    assert launched == [] and any("server roles / ownership records unreadable" in m for m in messages)


# ---------------------------------------------------------------------------------------------------- re-placement
def sandbox(tmp_path, rows, events=()):
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    queue = tmp_path / "experiments/queue.yaml"
    remote = {"enabled": False, "limits": {}, "host_ram_reserve_gb": 120}
    queue.write_text("# header kept byte for byte\n" + yaml.safe_dump({"hosts": {"remote": remote}, "jobs": rows},
                                                                       sort_keys=False))
    registry = tmp_path / "experiments/registry.jsonl"
    registry.touch()
    for event in events:
        jobs.record(event, registry)
    return queue, registry


def remote_row(job, **extra):
    return {**job, "host": "remote", "disk_gb": 10, "required_results": ["final.json"], **extra}


def replace(tmp_path, paused=lambda job_id: False, **kwargs):
    policy = tmp_path / "exclusive-roles.yaml"
    policy.write_text(yaml.safe_dump({k: v for k, v in EXCLUSIVE_ROLES.items() if k not in ("path", "sha256")},
                                     sort_keys=False))
    return roles_mod.replace_to_local(
        paused, queue_path=tmp_path / "experiments/queue.yaml", registry_path=tmp_path / "experiments/registry.jsonl",
        manifest=tmp_path / "pool-placement.json", ownership_dir=tmp_path / "ownership",
        admission_lock=tmp_path / "admission.lock", root=tmp_path, roles_path=policy, **kwargs)


def test_never_launched_remote_metaworld_rows_are_replaced_to_local(tmp_path):
    rows = [remote_row(DRM, pin_host="remote"), remote_row(DRM_EVAL, pin_host="remote"), remote_row(PRETRAIN),
            remote_row(EXPORT), remote_row(DROID_PRETRAIN), remote_row(PLAIN), {**COLLECT}]
    queue, registry = sandbox(tmp_path, rows)
    before = jobs.load_queue(queue)["jobs"]
    dry = replace(tmp_path, dry_run=True)
    assert dry["moved"] == [] and sorted(dry["would_move"]) == sorted([DRM["id"], DRM_EVAL["id"], PRETRAIN["id"],
                                                                        EXPORT["id"]])
    assert jobs.load_queue(queue)["jobs"] == before
    result = replace(tmp_path)
    assert sorted(result["moved"]) == sorted(dry["would_move"])
    after = {j["id"]: j for j in jobs.load_queue(queue)["jobs"]}
    assert queue.read_text().startswith("# header kept byte for byte\n")
    for job in before:
        moved = job["id"] in result["moved"]
        assert after[job["id"]]["host"] == ("local" if moved else job.get("host", "local"))
        assert jobs.job_fingerprint(after[job["id"]]) == jobs.job_fingerprint(job)
        assert {k: v for k, v in after[job["id"]].items() if k not in roles_mod.HOST_FIELDS} == \
            {k: v for k, v in job.items() if k not in roles_mod.HOST_FIELDS}
    assert after[DRM["id"]]["pin_host"] == "local" and after[DRM_EVAL["id"]]["pin_host"] == "local"
    assert after[DROID_PRETRAIN["id"]]["host"] == "remote" and after[PLAIN["id"]]["host"] == "remote"
    pool = json.loads((tmp_path / "pool-placement.json").read_text())
    assert pool["moves"][-1]["to"] == "local" and sorted(pool["moves"][-1]["ids"]) == sorted(result["moved"])
    assert pool["effective_jobs"] == {i: after[i] for i in result["moved"]}
    assert pool["moves"][-1]["queue_sha256_after"] == hashlib.sha256(queue.read_bytes()).hexdigest()
    records = roles_mod.read_ownership(tmp_path / "ownership")
    assert set(records) == set(result["moved"])
    assert all(r["state"] == "placed" and r["owner_host"] == "local" and r["from_host"] == "remote"
               and r["job_class"] == "metaworld" for r in records.values())
    assert replace(tmp_path)["moved"] == []  # idempotent


def test_started_paused_or_half_started_rows_are_never_replaced(tmp_path):
    rows = [remote_row(DRM), remote_row(DRM_EVAL), remote_row(PRETRAIN), remote_row(EXPORT),
            remote_row(SNERL, stage=2)]
    events = [{"event": "launched", "id": DRM_EVAL["id"], "host": "remote", "gpu": REMOTE_GPU, "attempt": 1}]
    sandbox(tmp_path, rows, events)
    result = replace(tmp_path, paused=lambda job_id: job_id == PRETRAIN["id"])
    assert result["moved"] == [EXPORT["id"]]
    assert "registry state" in result["kept"][DRM_EVAL["id"]] and "registry state" in result["kept"][DRM["id"]]
    assert result["kept"][PRETRAIN["id"]] == "intentional admission pause"
    assert "Stage 2" in result["kept"][SNERL["id"]]
    rows = [remote_row({**PRETRAIN, "id": "x"})]
    sandbox(tmp_path / "other", rows)
    (tmp_path / "other/runs/x").mkdir(parents=True)
    assert "local output directory" in roles_mod.replaceable(
        rows[0], {}, {"x": rows[0]}, EXCLUSIVE_ROLES, tmp_path / "other")


def test_replacement_waits_for_the_admission_lock(tmp_path):
    import fcntl

    sandbox(tmp_path, [remote_row(PRETRAIN)])
    with open(tmp_path / "admission.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert replace(tmp_path) == {"moved": [], "busy": True}
    assert replace(tmp_path)["moved"] == [PRETRAIN["id"]]


# ---------------------------------------------------------------------------------------------------- migration
STOP = {"event": "checkpoint_paused", "host": "remote", "attempt": 1, "intentional_stop": True,
        "checkpoint_step": 300000, "reason": "server roles: migrate to local"}


def migration_sandbox(tmp_path, *, evaluator_launched=True, container_removed=True):
    import torch

    rows = [remote_row(DRM, pin_host="remote"), remote_row(DRM_EVAL, pin_host="remote"), {**EXPORT}]
    run, out = tmp_path / "runs/cst75-drm-hammer-s2000", tmp_path / "runs/cst75-drm-hammer-s2000-eval"
    (run / "checkpoints").mkdir(parents=True)
    (run / "replay").mkdir()
    out.mkdir(parents=True)
    agent = {name: {"w": torch.zeros(2)} for name in migration.COMPONENTS["train_rl.py"]["agent"]}
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(layer.parameters())
    layer(torch.ones(1, 2)).sum().backward()
    optimizer.step()  # a real optimizer state: 0-dim "step" tensors
    agent.update(dormant_ratio=0.1, awaken_step=None, actor_opt=optimizer.state_dict())
    state = {key: 0 for key in migration.COMPONENTS["train_rl.py"]["top"]}
    state.update(step=300000, agent=agent, replay_meta=[1, 2, 3], torch_rng=torch.zeros(4, dtype=torch.uint8))
    torch.save(state, run / "checkpoints/latest.pt")
    (run / "replay/ram_snapshot.npz").write_bytes(b"replay")
    (run / "train.jsonl").write_text(json.dumps({"step": 300000}) + "\n")
    (out / "eval.jsonl").write_text(json.dumps({"step": 300000}) + "\n")
    for folder, job_id in ((run, DRM["id"]), (out, DRM_EVAL["id"])):  # the source host's campaign provenance
        (folder / "provenance.json").write_text(json.dumps({"host": "remote", "job_id": job_id,
                                                            "campaign_eligible": True}))
    (out / "eval_provenance").mkdir()  # scripts/eval_rl.py records its own provenance in a sub-folder
    (out / "eval_provenance/provenance.json").write_text(json.dumps({"host": "remote", "campaign_eligible": True}))
    export = tmp_path / "runs/cst75-export-seed0-hammer-200k/encoder.pt"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"encoder")
    digest = migration.sha256(run / "checkpoints/latest.pt")
    events = []
    for job_id, cid in ((DRM["id"], "a" * 64), (DRM_EVAL["id"], "b" * 64)):
        if job_id == DRM_EVAL["id"] and not evaluator_launched:
            continue
        events += [{"event": "launched", "id": job_id, "host": "remote", "gpu": REMOTE_GPU, "attempt": 1,
                    "container": f"s4d-{job_id}-a1", "container_id": cid, "commit": "c" * 40,
                    "image_digest": "sha256:" + "e" * 64,
                    "job_sha256": jobs.job_fingerprint(remote_row(DRM if job_id == DRM["id"] else DRM_EVAL))},
                   {**STOP, "id": job_id, "checkpoint": f"runs/{job_id}/checkpoints/latest.pt",
                    "checkpoint_sha256": digest if job_id == DRM["id"] else None},
                   {"event": "exited", "id": job_id, "host": "remote", "attempt": 1, "code": 143,
                    "container_id": cid}]
        if container_removed:
            events.append({"event": "container_removed", "id": job_id, "host": "remote", "attempt": 1,
                           "container_id": cid})
    events += [{"event": "launched", "id": EXPORT["id"], "pid": 1, "gpu": LOCAL_GPU, "attempt": 1},
               {"event": "results_synced", "id": EXPORT["id"], "sync": {"files": {"encoder.pt": migration.sha256(export)}}},
               {"event": "exited", "id": EXPORT["id"], "attempt": 1, "code": 0}]
    sandbox(tmp_path, rows, events)
    paths = migration.Paths(root=tmp_path, ownership=tmp_path / "ownership", pool_manifest=tmp_path / "pool.json",
                            admission_lock=tmp_path / "admission.lock")
    listing = {i: {str(p.relative_to(folder)): migration.sha256(p) for p in folder.rglob("*") if p.is_file()}
               for i, folder in ((DRM["id"], run), (DRM_EVAL["id"], out))}
    return paths, listing


def identical_scope(scripts, commit, root):
    return {"identical": True, "files": {"s4d/model/encoder.py": "0" * 64}, "third_party": {}, "differences": []}


def test_remote_to_local_migration_of_a_paused_run(tmp_path, monkeypatch):
    paths, listing = migration_sandbox(tmp_path)
    with pytest.raises(migration.MigrationError, match="invalid destination"):
        migration.begin(DRM["id"], "remote", "x", paths)
    records = migration.begin(DRM["id"], "local", "server roles", paths)
    assert [r["state"] for r in records] == ["migrating", "migrating"] and records[0]["owner_host"] == "remote"
    owned = roles_mod.read_ownership(paths.ownership)
    for host in ("local", "remote"):  # neither host admits the unit while the migration is in progress
        assert "in progress" in roles_mod.ownership_refusal(DRM["id"], host, owned)
        assert "in progress" in roles_mod.ownership_refusal(DRM_EVAL["id"], host, owned)
    with pytest.raises(migration.MigrationError, match="a migration already owns it"):
        migration.begin(DRM["id"], "local", "again", paths)
    with pytest.raises(migration.MigrationError, match="transferred state not verified"):
        migration.switch(DRM["id"], paths, identical_scope)
    bad = json.loads(json.dumps(listing))
    bad[DRM["id"]]["train.jsonl"] = "0" * 64
    with pytest.raises(migration.MigrationError, match="differs from the source checksum"):
        migration.verify(DRM["id"], bad, paths)
    records = migration.verify(DRM["id"], listing, paths)
    manifest = records[0]["manifest"]
    assert manifest["components"]["step"] == 300000
    groups = manifest["components"]["groups"]
    assert groups["optimizer"] == ["agent.actor_opt", "agent.critic_opt", "agent.predictor_opt"]
    assert manifest["files"]["runs/cst75-drm-hammer-s2000/checkpoints/latest.pt"]["mtime_ns"] > 0
    assert groups["scheduler"] == ["agent.awaken_step", "agent.dormant_ratio"]
    assert groups["rng"] == ["camera_rng", "cuda_rng", "numpy_rng", "python_rng", "torch_rng"]
    replay = listing[DRM["id"]]["replay/ram_snapshot.npz"]
    assert manifest["components"]["replay_files"] == {"replay/ram_snapshot.npz": replay}
    assert manifest["inputs"] == {"runs/cst75-export-seed0-hammer-200k/encoder.pt": migration.sha256(
        tmp_path / "runs/cst75-export-seed0-hammer-200k/encoder.pt")}
    with pytest.raises(migration.MigrationError, match="execution scope differs"):
        migration.switch(DRM["id"], paths, lambda *a: {**identical_scope(*a), "identical": False,
                                                       "differences": ["s4d/rl/agent.py"]})
    records = migration.switch(DRM["id"], paths, identical_scope, lambda root: ("d" * 40, "local-venv:x"))
    assert [r["state"] for r in records] == ["switched", "switched"] and records[0]["owner_host"] == "local"
    run, out = tmp_path / "runs/cst75-drm-hammer-s2000", tmp_path / "runs/cst75-drm-hammer-s2000-eval"
    for folder in (run, out, out / "eval_provenance"):  # source provenance preserved; the local attempt writes its own
        assert not (folder / "provenance.json").exists() and (folder / "provenance.remote-a1.json").is_file()
    assert records[0]["provenance_transition"]["renamed"] == {
        "runs/cst75-drm-hammer-s2000/provenance.json": "runs/cst75-drm-hammer-s2000/provenance.remote-a1.json",
        "runs/cst75-drm-hammer-s2000-eval/provenance.json":
            "runs/cst75-drm-hammer-s2000-eval/provenance.remote-a1.json",
        "runs/cst75-drm-hammer-s2000-eval/eval_provenance/provenance.json":
            "runs/cst75-drm-hammer-s2000-eval/eval_provenance/provenance.remote-a1.json"}
    assert migration.switch(DRM["id"], paths, identical_scope) == records  # idempotent
    state = jobs.read_registry(paths.registry)
    for job_id in (DRM["id"], DRM_EVAL["id"]):
        assert state[job_id]["host"] == "local" and state[job_id]["status"] == "pending"
        assert state[job_id]["migrated_from"] == {"host": "remote", "attempt": 1, "checkpoint_step": 300000}
        assert state[job_id]["attempts"] == 1
    queue = {j["id"]: j for j in jobs.load_queue(paths.queue)["jobs"]}
    assert queue[DRM["id"]]["host"] == "local" and queue[DRM["id"]]["pin_host"] == "local"
    assert jobs.job_fingerprint(queue[DRM["id"]]) == jobs.job_fingerprint(remote_row(DRM))
    pool = json.loads(paths.pool_manifest.read_text())
    assert pool["moves"][-1]["to"] == "local" and set(pool["effective_jobs"]) == {DRM["id"], DRM_EVAL["id"]}
    owned = roles_mod.read_ownership(paths.ownership)
    assert roles_mod.ownership_refusal(DRM["id"], "local", owned) is None
    assert roles_mod.ownership_refusal(DRM["id"], "remote", owned).startswith("owned by local")
    with pytest.raises(migration.MigrationError, match="only a migration before its switch"):
        migration.abort(DRM["id"], "late", paths)
    with pytest.raises(migration.MigrationError, match="no running or finished destination attempt"):
        migration.confirm(DRM["id"], paths)
    jobs.record({"event": "launched", "id": DRM["id"], "host": "local", "pid": 2, "gpu": LOCAL_GPU, "attempt": 2},
                paths.registry)
    with pytest.raises(migration.MigrationError, match="no running or finished destination attempt"):
        migration.confirm(DRM["id"], paths)  # the evaluator has not started on local yet
    jobs.record({"event": "launched", "id": DRM_EVAL["id"], "host": "local", "pid": 3, "gpu": LOCAL_GPU, "attempt": 2},
                paths.registry)
    with pytest.raises(migration.MigrationError, match="no verified resume"):
        migration.confirm(DRM["id"], paths)
    (run / "events.jsonl").write_text(json.dumps({"event": "resumed", "step": 300000, "time": 1.0}) + "\n")
    with open(run / "train.jsonl", "a") as stream:
        stream.write(json.dumps({"step": 301000}) + "\n")
    with pytest.raises(migration.MigrationError, match="no verified resume"):
        migration.confirm(DRM["id"], paths)  # a resume record older than the destination attempt
    (run / "events.jsonl").write_text(json.dumps({"event": "resumed", "step": 300000, "time": 4e9}) + "\n")
    records = migration.confirm(DRM["id"], paths)
    assert [r["state"] for r in records] == ["verified", "verified"] and records[0]["resumed_attempt"] == 2
    assert [h["state"] for h in records[0]["history"]] == ["migrating", "migrating", "switched"]


def test_migration_preconditions(tmp_path):
    paths, listing = migration_sandbox(tmp_path, container_removed=False)
    migration.begin(DRM["id"], "local", "x", paths)
    migration.verify(DRM["id"], listing, paths)
    with pytest.raises(migration.MigrationError, match="paused remote container is not removed"):
        migration.switch(DRM["id"], paths, identical_scope)
    assert [r["state"] for r in migration.abort(DRM["id"], "not now", paths)] == ["aborted", "aborted"]
    owned = roles_mod.read_ownership(paths.ownership)
    assert owned[DRM["id"]]["owner_host"] == "remote"
    assert roles_mod.ownership_refusal(DRM["id"], "local", owned).startswith("owned by remote")

    other = tmp_path / "running"
    paths, _ = migration_sandbox(other)
    jobs.record({"event": "launched", "id": DRM["id"], "host": "remote", "gpu": REMOTE_GPU, "attempt": 2,
                 "container": "c", "container_id": "f" * 64, "commit": "c" * 40}, paths.registry)
    with pytest.raises(migration.MigrationError, match="still active"):
        migration.begin(DRM["id"], "local", "x", paths)
    jobs.record({"event": "exited", "id": DRM["id"], "host": "remote", "attempt": 2, "code": 1}, paths.registry)
    with pytest.raises(migration.MigrationError, match="no recorded intentional checkpoint pause"):
        migration.begin(DRM["id"], "local", "x", paths)

    fresh = tmp_path / "fresh"
    paths, _ = migration_sandbox(fresh, evaluator_launched=False)
    records = migration.begin(DRM["id"], "local", "x", paths)
    assert records[1]["member"] == {"launched": False}  # the evaluator moves with the queue rows only

    unstarted = tmp_path / "unstarted"
    sandbox(unstarted, [remote_row(PRETRAIN)])
    with pytest.raises(migration.MigrationError, match="never launched"):
        migration.begin(PRETRAIN["id"], "local", "x", migration.Paths(root=unstarted, ownership=unstarted / "own",
                                                                      admission_lock=unstarted / "lock"))


# ---------------------------------------------------------------------------------------------------- source scope
def test_import_closure_follows_function_level_relative_and_string_imports(tmp_path):
    files = {
        "scripts/run.py": "import _helper\nfrom s4d.a import b\n",
        "scripts/_helper.py": "def f():\n    import s4d.lazy\n",
        "s4d/__init__.py": "", "s4d/a/__init__.py": "", "s4d/a/b.py": "from . import c\nX = 's4d.dyn'\n",
        "s4d/a/c.py": "", "s4d/lazy.py": "", "s4d/dyn.py": "", "s4d/unused.py": "",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    read, _ = migration.tree_reader(tmp_path)
    assert migration.import_closure(["scripts/run.py"], read) == set(files) - {"s4d/unused.py"}


def test_execution_scope_compares_a_release_commit_with_the_working_tree(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    files = {"scripts/train_rl.py": "import _bootstrap\nimport s4d.rl\n", "scripts/_bootstrap.py": "import mujoco_setup\n",
             "mujoco_setup.py": "", "s4d/__init__.py": "", "s4d/rl.py": "import ops.x\nimport ops.jobs.scheduler\nA = 1\n",
             "ops/__init__.py": "", "ops/x.py": "", "ops/jobs/__init__.py": "", "ops/jobs/scheduler.py": "",
             "configs/rl/base.yaml": "a: 1\n", "configs/droid/x.yaml": "", "uv.lock": "lock\n",
             "pyproject.toml": "[project]\n"}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "release")
    commit = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    scope = migration.execution_scope(["scripts/train_rl.py"], commit, tmp_path)
    assert scope["identical"] and set(scope["files"]) == {
        "scripts/train_rl.py", "scripts/_bootstrap.py", "mujoco_setup.py", "s4d/__init__.py", "s4d/rl.py",
        "ops/__init__.py", "ops/x.py", "ops/jobs/__init__.py", "configs/rl/base.yaml", "uv.lock", "pyproject.toml"}
    (tmp_path / "ops/jobs/scheduler.py").write_text("ADMIT = 1\n")  # a listed operational file: recorded only
    (tmp_path / "configs/droid/x.yaml").write_text("b: 2\n")  # outside the scope
    scope = migration.execution_scope(["scripts/train_rl.py"], commit, tmp_path)
    assert scope["identical"] and scope["operational_differences"] == ["ops/jobs/scheduler.py"]
    (tmp_path / "ops/x.py").write_text("HOST = 'local'\n")  # any other imported operational file is compared
    (tmp_path / "mujoco_setup.py").write_text("X = 1\n")  # a repository-root module imported by _bootstrap
    scope = migration.execution_scope(["scripts/train_rl.py"], commit, tmp_path)
    assert not scope["identical"] and scope["differences"] == ["mujoco_setup.py", "ops/x.py"]
    git("checkout", "--", "ops/x.py", "mujoco_setup.py")
    (tmp_path / "configs/rl/base.yaml").write_text("a: 2\n")
    (tmp_path / "s4d/new.py").write_text("")
    (tmp_path / "s4d/rl.py").write_text("import ops.x\nimport s4d.new\nA = 1\n")
    scope = migration.execution_scope(["scripts/train_rl.py"], commit, tmp_path)
    assert not scope["identical"] and scope["differences"] == ["configs/rl/base.yaml", "s4d/new.py", "s4d/rl.py"]


def test_component_digests_are_canonical():
    import numpy as np
    import torch

    a = {"x": torch.arange(3), "y": [np.zeros(2), 1, None]}
    b = {"y": [np.zeros(2), 1, None], "x": torch.arange(3)}
    assert migration.canonical_digest(a) == migration.canonical_digest(b)
    assert migration.canonical_digest(a) != migration.canonical_digest({**a, "x": torch.arange(3).float()})


def test_remote_listing_is_one_read_only_checksum_command():
    class Client:
        def checked_ssh(self, command, **kwargs):
            self.command, self.kwargs = command, kwargs
            return subprocess.CompletedProcess(command, 0, "aa  ./checkpoints/latest.pt\nbb  ./train.jsonl\n", "")

    client = Client()
    assert migration.remote_listing(client, "runs/r") == {"checkpoints/latest.pt": "aa", "train.jsonl": "bb"}
    assert client.command.startswith("cd /home/compu/kaist/sunho/runs/r && find . -type f")
    assert client.kwargs["idempotent"] is True
    with pytest.raises(migration.MigrationError):
        migration.remote_listing(client, "runs/../x")


# ---------------------------------------------------------------------------------------------------- drivers
def driver_source(name):
    return (REPO / "ops/campaign" / name).read_text()


def test_remote_driver_offers_and_pool_moves_only_role_admitted_rows():
    """Read as text (importing an ops.campaign module runs it): the role check precedes ready() (no export upload
    for a row the remote may not run), guards every pool unit and the re-placement runs each non-dry pass."""
    source = driver_source("remote_campaign_driver.py")
    assert "candidates = [j for j in jobs.eligible(remote, retry_view) if role_admits(j) and ready(j)]" in source
    assert "replaced = host_roles.replace_to_local(paused)" in source
    # 2026-11-06: no pool move either while the co-tenant evidence is unreadable (COTENANT None)
    assert "if POOL_READY and not DRY and not held and ROLES is not None and COTENANT is not None:" in source
    tree = ast.parse(source)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "host_roles"}
    assert calls == {"load_roles", "read_ownership", "load_exceptions", "replace_to_local", "refusal", "role_host",
                     "job_class", "placement_repair_needed"}
    assert source.count("chosen.append({**with_role(") == 2  # dispatch copies carry their class
    assert "if not all(role_admits(u, drain=False) for u in members):" in source  # the pool never drains
    assert "elif (not DRY and not (REPO / 'runs/setup/remote-hold-admissions').exists()" in source  # hold: no moves
    assert "candidates = []" in source.split("if (REPO / 'runs/setup/remote-hold-admissions').exists():", 1)[1][:200]


@pytest.mark.parametrize("name", ["stage2_local_driver.py", "baseline_local_driver.py"])
def test_local_drivers_filter_candidates_by_the_local_role(name):
    assert "host_roles.refusal(j, 'local', _roles, _ownership, _context) is None" in driver_source(name)


def test_an_interrupted_switch_completes_when_rerun(tmp_path, monkeypatch):
    paths, listing = migration_sandbox(tmp_path)
    migration.begin(DRM["id"], "local", "server roles", paths)
    migration.verify(DRM["id"], listing, paths)
    real = jobs.record

    def killed(event, path=jobs.REGISTRY):
        if event.get("event") == "migrated" and Path(path) == paths.registry:
            raise OSError("process killed after the queue rewrite")
        return real(event, path)

    monkeypatch.setattr(jobs, "record", killed)
    destination = lambda root: ("d" * 40, "local-venv:x")  # noqa: E731
    with pytest.raises(OSError):
        migration.switch(DRM["id"], paths, identical_scope, destination)
    assert {j["id"]: j["host"] for j in jobs.load_queue(paths.queue)["jobs"]}[DRM["id"]] == "local"
    owned = roles_mod.read_ownership(paths.ownership)
    assert all("in progress" in roles_mod.ownership_refusal(i, h, owned) for i in owned for h in ("local", "remote"))
    monkeypatch.setattr(jobs, "record", real)
    records = migration.switch(DRM["id"], paths, identical_scope, destination)
    assert [r["state"] for r in records] == ["switched", "switched"]
    events = [json.loads(line) for line in paths.registry.read_text().splitlines()]
    assert sorted(e["id"] for e in events if e["event"] == "migrated") == sorted([DRM["id"], DRM_EVAL["id"]])
    assert len(json.loads(paths.pool_manifest.read_text())["moves"]) == 1
    assert jobs.read_registry(paths.registry)[DRM["id"]]["host"] == "local"


# ---------------------------------------------------------------------------------------------------- review round 1
@pytest.mark.parametrize("job,cls", [
    (row("s", "scripts/compute_workspace_stats.py", "--regime=metaworld", role="droid_preprocess"), "invalid"),
    (row("s", "scripts/compute_workspace_stats.py", "--regime=droid_largescale", role="droid_preprocess"),
     "droid_preprocess"),
    (row("f", "scripts/train_fql.py", "--task", "drawer-open", "--dataset", "/datasets/real_world/drawer-open"),
     "ambiguous"),
    (row("f", "scripts/train_fql.py", "--task=drawer-open"), "ambiguous"),  # shared task name alone
    (row("f", "scripts/train_fql.py", "--task", "mug-hanging"), "realworld_fql"),
    (row("a", "scripts/encp_repr_diagnostics.py", "--task", "hammer", "--arm", "d=runs/droid-x/encoder.pt"),
     "ambiguous"),
    (row("a", "scripts/encp_repr_diagnostics.py", "--task", "hammer", "--arm", "d=runs/droid-x/encoder.pt",
         role="metaworld"), "metaworld"),  # a declared role resolves an ambiguity it belongs to
    (row("a", "scripts/encp_repr_diagnostics.py", "--task", "hammer", role="realworld_preprocess"), "invalid"),
    ({**EXPORT, "role": "droid_pretrain"}, "invalid"),  # contradicts its producer's class
])
def test_argument_forms_shared_task_names_and_declared_roles(job, cls):
    assert roles_mod.job_class(job, CONTEXT).cls == cls


def test_ownership_location_must_be_a_directory(tmp_path):
    (tmp_path / "ownership").write_text("")
    with pytest.raises(ValueError):
        roles_mod.read_ownership(tmp_path / "ownership")


def test_replacement_translates_remote_gpu_selections_and_repairs_interrupted_records(tmp_path, monkeypatch):
    rows = [remote_row(PRETRAIN, gpu=REMOTE_GPU)]
    queue, _ = sandbox(tmp_path, rows)
    real = roles_mod.append_pool_record

    def killed(*args, **kwargs):
        raise OSError("killed after the queue rewrite")

    monkeypatch.setattr(roles_mod, "append_pool_record", killed)
    with pytest.raises(OSError):
        replace(tmp_path)
    moved = jobs.load_queue(queue)["jobs"][0]
    assert moved["host"] == "local" and moved["gpu"] == "any" and not (tmp_path / "pool-placement.json").exists()
    assert roles_mod.placement_repair_needed([moved], tmp_path / "pool-placement.json")
    monkeypatch.setattr(roles_mod, "append_pool_record", real)
    result = replace(tmp_path)
    assert result["moved"] == [] and result["repaired"] == [PRETRAIN["id"]]
    pool = json.loads((tmp_path / "pool-placement.json").read_text())
    assert pool["effective_jobs"] == {PRETRAIN["id"]: moved}
    assert roles_mod.read_ownership(tmp_path / "ownership")[PRETRAIN["id"]]["state"] == "placed"
    assert not roles_mod.placement_repair_needed([moved], tmp_path / "pool-placement.json")


def test_migration_refuses_changed_rows_stage2_rows_and_changed_state(tmp_path):
    paths, listing = migration_sandbox(tmp_path)
    text = paths.queue.read_text()
    paths.queue.write_text(text.replace("--seed\n  - '2000'", "--seed\n  - '2001'"))
    with pytest.raises(migration.MigrationError, match="fingerprint"):
        migration.begin(DRM["id"], "local", "x", paths)
    paths.queue.write_text(text)
    migration.begin(DRM["id"], "local", "x", paths)
    migration.verify(DRM["id"], listing, paths)
    replay = tmp_path / "runs/cst75-drm-hammer-s2000/replay/ram_snapshot.npz"
    replay.write_bytes(b"changed")
    with pytest.raises(migration.MigrationError, match="changed after its verification"):
        migration.switch(DRM["id"], paths, identical_scope)

    stage2 = tmp_path / "stage2"
    paths, _ = migration_sandbox(stage2)
    text = paths.queue.read_text()
    paths.queue.write_text(text.replace("pin_host: remote", "pin_host: remote\n  stage: 2", 1))
    with pytest.raises(migration.MigrationError, match="Stage 2"):
        migration.begin(DRM["id"], "local", "x", paths)


def test_a_switched_run_launches_only_with_its_verified_state_and_row(tmp_path):
    paths, listing = migration_sandbox(tmp_path)
    migration.begin(DRM["id"], "local", "x", paths)
    migration.verify(DRM["id"], listing, paths)
    migration.switch(DRM["id"], paths, identical_scope, lambda root: ("d" * 40, "local-venv:x"))
    owned = roles_mod.read_ownership(paths.ownership)
    queue = {j["id"]: j for j in jobs.load_queue(paths.queue)["jobs"]}
    state = jobs.read_registry(paths.registry)

    def check(job_id, fingerprint=None):
        return roles_mod.migrated_state_refusal(queue[job_id], owned, state[job_id],
                                                fingerprint or jobs.job_fingerprint(queue[job_id]), tmp_path)

    assert check(DRM["id"]) is None and check(DRM_EVAL["id"]) is None
    assert "job fingerprint" in check(DRM["id"], "0" * 64)
    latest = tmp_path / "runs/cst75-drm-hammer-s2000/checkpoints/latest.pt"
    data = latest.read_bytes()
    latest.unlink()
    assert "latest.pt is missing" in check(DRM["id"])
    latest.write_bytes(data)
    assert "changed after its verification" in check(DRM["id"])  # same bytes, new modification time
    eval_jsonl = tmp_path / "runs/cst75-drm-hammer-s2000-eval/eval.jsonl"
    eval_jsonl.write_text(eval_jsonl.read_text() + "{}\n")
    assert "eval.jsonl changed" in check(DRM_EVAL["id"])
    jobs.record({"event": "launched", "id": DRM["id"], "host": "local", "pid": 2, "gpu": LOCAL_GPU, "attempt": 2},
                paths.registry)
    state = jobs.read_registry(paths.registry)
    assert check(DRM["id"]) is None  # after the first destination attempt the run owns its files again


def test_pool_mover_respects_roles_ownership_and_pauses(tmp_path, monkeypatch):
    from ops.jobs import pool_placement as pool

    declared = {"required_results": ["completion.json"], "disk_gb": 30}
    rows = [{**PRETRAIN, **declared}, {**DROID_PRETRAIN, **declared, "id": "droid-pre-a"},
            {**DROID_PRETRAIN, **declared, "id": "droid-pre-b"}]
    queue, registry = sandbox(tmp_path, rows)
    real_load, real_read = jobs.load_queue, jobs.read_registry
    monkeypatch.setattr(jobs, "QUEUE", queue)
    monkeypatch.setattr(jobs, "REGISTRY", registry)
    monkeypatch.setattr(jobs, "load_queue", lambda path=queue: real_load(path))  # defaults bound at definition
    monkeypatch.setattr(jobs, "read_registry", lambda path=registry: real_read(path))
    monkeypatch.setattr(pool, "SETUP", tmp_path)
    monkeypatch.setattr(pool, "ADMISSION_LOCK", tmp_path / "admission.lock")
    monkeypatch.setattr(pool, "MANIFEST", tmp_path / "pool-placement.json")
    monkeypatch.setattr(pool, "ROOT", tmp_path)
    monkeypatch.setattr(roles_mod, "OWNERSHIP_DIR", tmp_path / "ownership")
    roles_mod.write_ownership({"job_id": "droid-pre-b", "state": "placed", "owner_host": "local"})
    moved = pool.move([PRETRAIN["id"], "droid-pre-a", "droid-pre-b"], "remote", "test", blocking=True)
    assert moved == ["droid-pre-a"]  # Meta-World rows are not a remote role since 2026-11-12
    hosts = {j["id"]: j.get("host", "local") for j in jobs.load_queue(queue)["jobs"]}
    assert hosts == {PRETRAIN["id"]: "local", "droid-pre-a": "remote", "droid-pre-b": "local"}


# ---------------------------------------------------------------------------------------------------- drain exception
def test_drain_exception_admits_only_evaluators_of_trainers_that_ran_on_the_remote():
    """Coordinator rule 2026-11-06: while remote Meta-World work drains, the evaluator companion of a trainer running
    or completed on the remote may still be admitted there (it reads that trainer's snapshots); nothing else, and the
    pool never moves rows by it (no trainer state is passed for placements)."""
    evaluator = remote_row(DRM_EVAL)
    running = {"host": "remote", "status": "running", "attempt": 1,
               "attempt_containers": {"1": {"container": "c", "container_id": "f" * 64}}}
    for trainer in (running, {"host": "remote", "status": "sync_pending"}, {"host": "remote", "status": "done"}):
        assert roles_mod.refusal(evaluator, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, None, None, trainer) is None
    for trainer in ({"host": "remote", "status": "crashed"}, {"host": "remote", "status": "pending"},
                    {"host": "remote", "status": "launching"},  # never confirmed to execute
                    {**running, "attempt_containers": {}}, {**running, "unknown": True},
                    {**running, "host": "local"}, None):
        assert roles_mod.refusal(evaluator, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, None, None, trainer) is not None
    for args in (["--run-dir", "runs/other", "--train-job", DRM["id"]],  # not the trainer's companion
                 ["--run-dir", f"runs/{DRM['id']}", "--train-job", "other"],
                 ["--run-dir", f"runs/{DRM['id']}", "--train-job", DRM["id"], "--snapshots", "final"],
                 ["--run-dir", f"runs/{DRM['id']}", "--train-job", DRM["id"], "--suites=v2"]):
        assert roles_mod.refusal({**evaluator, "args": args}, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, None, None,
                                 running) is not None
    assert roles_mod.refusal(remote_row(DRM), "remote", EXCLUSIVE_ROLES, {}, CONTEXT, None, None, running) is not None
    assert roles_mod.refusal({**evaluator, "reevaluates": DRM["id"]}, "remote", EXCLUSIVE_ROLES, {}, CONTEXT, None, None,
                             running) is not None  # a re-evaluation row is not a companion
    assert roles_mod.refusal(evaluator, "remote", EXCLUSIVE_ROLES, {}, CONTEXT) is not None  # placement path: no drain
    migrating = ownership(DRM_EVAL["id"], "migrating", "remote")
    assert "in progress" in roles_mod.refusal(evaluator, "remote", EXCLUSIVE_ROLES, migrating, CONTEXT, None, None, running)
    closed = copy.deepcopy(EXCLUSIVE_ROLES)
    closed["hosts"]["remote"]["drain_evaluators"] = False
    assert roles_mod.refusal(evaluator, "remote", closed, {}, CONTEXT, None, None, running) is not None


def test_a_switch_interrupted_between_ownership_records_holds_the_whole_unit(tmp_path, monkeypatch):
    paths, listing = migration_sandbox(tmp_path)
    migration.begin(DRM["id"], "local", "x", paths)
    migration.verify(DRM["id"], listing, paths)
    real = roles_mod.write_ownership

    def killed(record, directory=None):
        if record["job_id"] == DRM["id"] and record["state"] == "switched":
            raise OSError("killed between the evaluator's and the run's record")
        return real(record, directory)

    monkeypatch.setattr(roles_mod, "write_ownership", killed)
    destination = lambda root: ("d" * 40, "local-venv:x")  # noqa: E731
    with pytest.raises(OSError):
        migration.switch(DRM["id"], paths, identical_scope, destination)
    owned = roles_mod.read_ownership(paths.ownership)
    assert owned[DRM_EVAL["id"]]["state"] == "switched" and owned[DRM["id"]]["state"] == "migrating"
    for host in ("local", "remote"):  # the evaluator stays held while its unit is not switched
        assert "in progress" in roles_mod.ownership_refusal(DRM_EVAL["id"], host, owned)
    monkeypatch.setattr(roles_mod, "write_ownership", real)
    assert [r["state"] for r in migration.switch(DRM["id"], paths, identical_scope, destination)] == ["switched"] * 2
    assert roles_mod.ownership_refusal(DRM_EVAL["id"], "local", roles_mod.read_ownership(paths.ownership)) is None


def test_never_launched_remote_metaworld_rows_are_replaced_to_local(tmp_path):
    """2026-11-12 (user): the Meta-World development data were removed from the remote; Meta-World rows queued for the
    remote that never launched are replaced to local by the existing exclusive-role machinery."""
    rows = [remote_row(DRM), remote_row(DRM_EVAL), remote_row(PRETRAIN)]  # a trainer moves with its evaluator
    queue, registry = sandbox(tmp_path, rows)
    result = roles_mod.replace_to_local(lambda _: False, queue_path=queue, registry_path=registry,
                                        manifest=tmp_path / 'pool.json', ownership_dir=tmp_path / 'ownership',
                                        admission_lock=tmp_path / 'admission.lock', root=tmp_path)
    assert sorted(result['moved']) == sorted([DRM['id'], DRM_EVAL['id'], PRETRAIN['id']])
    assert all(roles_mod.role_host(j, ROLES, {r['id']: r for r in rows}) == 'local' for j in rows)


def test_shared_classes_never_trigger_generic_replacement(tmp_path):
    rows = [remote_row(DROID_PRETRAIN), remote_row(DROID_EXPORT)]  # classes both hosts admit
    queue, registry = sandbox(tmp_path, rows)
    before = queue.read_bytes()
    result = roles_mod.replace_to_local(lambda _: False, queue_path=queue, registry_path=registry,
                                        manifest=tmp_path / 'pool.json', ownership_dir=tmp_path / 'ownership',
                                        admission_lock=tmp_path / 'admission.lock', root=tmp_path)
    assert result['moved'] == [] and queue.read_bytes() == before
    assert all(roles_mod.role_host(j, ROLES, {r['id']: r for r in rows}) == 'remote' for j in rows)


@pytest.mark.parametrize('remote_spec', [{'enabled': True, 'hold': True}, {'enabled': False}])
def test_remote_host_hold_or_disable_never_blocks_local(tmp_path, monkeypatch, remote_spec):
    class NoRemoteCalls:
        def snapshot(self):
            pytest.fail('held or disabled remote must not be queried for ready rows')
    queue, registry = sandbox(tmp_path, [{**DRM, "deps": []}, remote_row(PRETRAIN)])
    spec = yaml.safe_load(queue.read_text())
    spec['hosts']['local'] = {'limits': {LOCAL_GPU: {'max_jobs': 3, 'max_mem_gb': 22}}}
    spec['hosts']['remote'].update(remote_spec)
    queue.write_text(yaml.safe_dump(spec))
    launched = []
    monkeypatch.setattr(jobs, 'launch', lambda job, *args: launched.append(job['id']) or 2**30)
    monkeypatch.setattr(jobs, 'source_identity', lambda: {'commit': 'c', 'source_sha256': {}})
    monkeypatch.setattr(jobs, 'gpu_usage', lambda: {})
    monkeypatch.setattr(jobs, 'host_available_gb', lambda: 500)
    monkeypatch.setattr(roles_mod, 'OWNERSHIP_DIR', tmp_path / 'ownership')
    monkeypatch.setattr(roles_mod, 'EXCEPTIONS_PATH', tmp_path / 'exceptions.json')
    jobs.tick(queue, registry, tmp_path / 'runs', disk_bytes=(2000 * 2**30, 1000 * 2**30),
              remote_backend=NoRemoteCalls())
    assert launched == [DRM['id']]


def test_only_hold_next_to_dispatch_queue_holds_it(tmp_path, monkeypatch):
    setup = tmp_path / 'runs/setup'
    verification = tmp_path / 'runs/remote/verification'
    setup.mkdir(parents=True)
    verification.mkdir(parents=True)
    holds = [setup / 'remote-hold-admissions', verification / 'HOLD']
    for path in holds:
        path.write_text('remote MIG reboot: keep held\n')
    original = {p: p.read_bytes() for p in holds}
    launched = []
    monkeypatch.setattr(jobs, 'launch', lambda job, *args: launched.append(job['id']) or 2**30)
    monkeypatch.setattr(jobs, 'source_identity', lambda: {'commit': 'c', 'source_sha256': {}})
    monkeypatch.setattr(jobs, 'gpu_usage', lambda: {})
    monkeypatch.setattr(jobs, 'host_available_gb', lambda: 500)
    monkeypatch.setattr(roles_mod, 'OWNERSHIP_DIR', tmp_path / 'ownership')
    spec = {'hosts': {'local': {'limits': {LOCAL_GPU: {'max_jobs': 3, 'max_mem_gb': 22}}}}, 'jobs': [{**DRM, 'deps': []}]}
    local = setup / 'bx-local-test.yaml'
    local.write_text(yaml.safe_dump(spec))
    jobs.tick(local, tmp_path / 'local-registry', tmp_path / 'runs',
              disk_bytes=(2000 * 2**30, 1000 * 2**30))
    assert launched == [DRM['id']]
    held_queue = verification / 'queue.yaml'
    typed = {'id': 'verify-suite', 'host': 'remote', 'script': 'scripts/run_tests.py', 'args': [],
             'verification_purpose': 'native_suite', 'campaign_eligible': False, 'gpu': REMOTE_GPU,
             'ram_gb': 96, 'mem_gb': 32, 'disk_gb': 1,
             'required_results': ['evidence/tests.json', 'evidence/tests.log', 'evidence/tests.junit.xml']}
    held_queue.write_text(yaml.safe_dump({'hosts': {**spec['hosts'], 'remote': {
        'enabled': True, 'limits': {REMOTE_GPU: {'max_jobs': 1, 'max_mem_gb': 90}}}}, 'jobs': [*spec['jobs'], typed]}))
    launched.clear()

    class NoRemoteCalls:
        def snapshot(self):
            pytest.fail('an adjacent verification HOLD must prevent remote calls')

    jobs.tick(held_queue, tmp_path / 'verification-registry', tmp_path / 'runs',
              disk_bytes=(2000 * 2**30, 1000 * 2**30), remote_backend=NoRemoteCalls())
    assert launched == [] and {p: p.read_bytes() for p in holds} == original


def test_local_admission_and_placement_sources_have_no_remote_holds():
    for name in ('baseline_local_driver.py', 'stage2_local_driver.py'):
        source = driver_source(name)
        assert 'remote-hold-admissions' not in source and 'verification/HOLD' not in source
    source = (REPO / 'ops/jobs/realworld_local.py').read_text()
    assert 'remote-hold-admissions' not in source and 'verification/HOLD' not in source
    assert 'remote-hold-admissions' in driver_source('remote_campaign_driver.py')
    assert '(queue_path.parent / "HOLD").exists()' in (REPO / 'ops/jobs/scheduler.py').read_text()


@pytest.mark.parametrize('source,destination', [('local', 'remote'), ('remote', 'local')])
@pytest.mark.parametrize('domain', ['metaworld', 'realworld', 'droid'])
def test_migration_begin_accepts_any_admitting_destination(tmp_path, source, destination, domain):
    # DROID's export resolver can consume a locally produced run directory without reading remote dataset trees.
    script = 'scripts/export_encoder.py' if domain == 'droid' else 'scripts/train.py'
    role = 'metaworld' if domain == 'metaworld' else domain + '_pretrain'
    job = row('move', script, '--config', f'configs/{domain}/task.yaml', host=source, role=role,
              required_results=['checkpoint.pt'], disk_gb=1)
    queue, registry = sandbox(tmp_path, [job])
    gpu = LOCAL_GPU if source == 'local' else REMOTE_GPU
    jobs.record({'event': 'launched', 'id': 'move', 'host': source, 'attempt': 1, 'gpu': gpu, 'pid': 2**30,
                 'commit': 'c' * 40, 'job_sha256': jobs.job_fingerprint(job)}, registry)
    jobs.record({**STOP, 'id': 'move', 'host': source, 'checkpoint': 'runs/move/checkpoints/latest.pt'}, registry)
    jobs.record({'event': 'exited', 'id': 'move', 'host': source, 'attempt': 1, 'code': 143}, registry)
    paths = migration.Paths(root=tmp_path, queue=queue, registry=registry, ownership=tmp_path / 'ownership',
                            admission_lock=tmp_path / 'admission.lock')
    if domain == 'metaworld' and destination == 'remote':  # 2026-11-12 (user): no Meta-World data on the remote
        with pytest.raises(migration.MigrationError, match='does not admit'):
            migration.begin('move', destination, 'unified pool', paths)
        return
    records = migration.begin('move', destination, 'unified pool', paths)
    assert records[0]['to_host'] == destination and records[0]['state'] == 'migrating'
    if destination == 'remote':
        record = {**records[0], 'manifest': {'sha256': 'a' * 64}}
        roles_mod.write_ownership(record, paths.ownership)
        before = queue.read_bytes(), registry.read_bytes()
        with pytest.raises(migration.MigrationError,
                           match='remote destination switch requires the remote release procedure'):
            migration.switch('move', paths)
        assert (queue.read_bytes(), registry.read_bytes()) == before


def test_remote_kind_preserves_metaworld_release_families_without_executing_driver():
    source = driver_source('remote_campaign_driver.py')
    kind = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'kind')
    gate = ast.unparse(kind.body[1].test)
    assert 'host_roles.job_class(row, by_id).cls' in gate
    assert "'realworld_pretrain'" in gate and "'verification'" in gate and "'metaworld'" not in gate
    assert "host_roles.role_host(row, ROLES, by_id) == 'remote'" in gate
    assert all(repr(k) in ast.unparse(kind) for k in ('pool_pretrain', 'pool_export', 'pool_rl', 'brl'))
