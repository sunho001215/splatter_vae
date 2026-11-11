"""Directive P: synthetic mirrored inputs, pinned argv, placement locks and local FQL admission.

Campaign drivers are inspected only as text. The fake transfer client executes its read-only inventory command against
synthetic directories; it never makes a network connection or imports a campaign module. Queues, registries, locks,
mirrors and closeout records all live under tmp_path.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from ops.jobs import admission_pause, host_roles, migration
from ops.jobs import realworld_local as rw
from ops.jobs import scheduler as jobs

ROLES = host_roles.load_roles()
LOCAL_GPU = jobs.APPROVED_HOST_GPUS["local"][0]
REMOTE_GPU = jobs.APPROVED_HOST_GPUS["remote"][0]
PREFIX = rw.REMOTE_PREFIX
STORE = "learning_inputs/cal/task/store.yaml"
SPLIT = "learning_inputs/cal/task/split_seed0.json"
GATE = "learning_inputs/cal/task/gate_report.json"
EPISODE = "processed/cal/task/train_data/episode"
COMMIT = "a" * 40
SCOPE = {"source_commit": COMMIT, "files": {"scripts/train_realworld_fql.py": "b" * 64}}
IDENTITY = {"commit": "c", "source_sha256": {"scripts/train_realworld_fql.py": "b" * 64}}
REAL_PLACEMENT_SCOPE = rw.placement_scope


@pytest.fixture(autouse=True)
def pinned_scope(monkeypatch):
    """Placement proves the pinned release's execution scope with git reads; synthetic rows use a fixed scope (the
    two tests of the real function call REAL_PLACEMENT_SCOPE)."""
    def scope(job, *, root=rw.REPO):
        if job.get("planned_commit") != COMMIT:
            raise rw.LocalInputError("no pinned release (planned_commit): local placement needs one")
        return copy.deepcopy(SCOPE)
    monkeypatch.setattr(rw, "placement_scope", scope)


def placed(job_id="fql", scope=None):
    """The ownership record ``apply`` writes for a row it placed local."""
    return {"job_id": job_id, "kind": "role_placement", "state": "placed", "owner_host": "local",
            "execution_scope": copy.deepcopy(scope or SCOPE)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def row(job_id="fql", **extra):
    return {"id": job_id, "role": "realworld_fql", "script": "scripts/train_realworld_fql.py", "host": "remote",
            "pin_host": "remote", "gpu": REMOTE_GPU, "mem_gb": 4, "ram_gb": 8, "disk_gb": 1, "priority": 1,
            "args": ["--config", "configs/realworld/fql/base.yaml", "--set", "data.dataset=runs/dataset/dataset.h5",
                     "data.split_manifest=" + PREFIX + "/" + SPLIT, "data.gate_report=" + PREFIX + "/" + GATE,
                     "data.store=" + PREFIX + "/" + STORE, "--name", job_id, "seed=0"],
            "required_results": ["final.json"], "deps": ["dataset"], "planned_commit": COMMIT, **extra}


class Client:
    """Local file operations in a fake remote tree, including the production inventory's actual shell quoting."""

    def __init__(self, root):
        self.root, self.commands, self.downloads = root, [], []

    def checked_ssh(self, command, **kwargs):
        assert kwargs == {"idempotent": True}
        assert command.startswith("cd " + PREFIX + " && ")
        assert all(word not in command for word in ("rsync", "mkdir", "rm ", "python", "ssh "))
        self.commands.append(command)
        local = command.replace("cd " + PREFIX, "cd " + shlex.quote(str(self.root)), 1)
        return subprocess.run(["bash", "-c", local], capture_output=True, text=True, check=True)

    def download_batch(self, remote, local, wanted, name, priority):
        assert name.startswith("realworld-") and priority in ("bulk", "dependency")
        source = self.root / remote.removeprefix(PREFIX + "/")
        self.downloads.append((remote, local, wanted, priority))
        for relative, digest in wanted.items():
            data = (source / relative).read_bytes()
            assert sha(data) == digest
            rw._atomic(local / relative, data)
        return sorted(wanted)


@pytest.fixture
def mirrored(tmp_path):
    remote, mirror, runs = tmp_path / "remote", tmp_path / "mirror", tmp_path / "runs"
    data = {SPLIT: b'{"train": ["episode"], "seed": 0}\n', GATE: b'{"passed": true}\n',
            STORE: ("# retained comment\nkind: compact\npaths:\n  - '" + PREFIX + "/" + EPISODE
                    + "' # retained path comment\nimage_hw: [128, 128]\n").encode(),
            EPISODE + "/_COMPLETE.json": b'{"complete": true}\n', EPISODE + "/episode.h5": b"fake pixels",
            EPISODE + "/extra metadata.json": b"{}\n"}
    for name, payload in data.items():
        path = remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    client = Client(remote)
    result = rw.fetch([SPLIT, GATE, STORE], client, mirror=mirror)
    config = tmp_path / "configs/realworld/fql/base.yaml"  # the trainer configuration the rows name (resolved at launch)
    config.parent.mkdir(parents=True)
    config.write_bytes((rw.REPO / "configs/realworld/fql/base.yaml").read_bytes())
    dataset = runs / "dataset/dataset.h5"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"verified dataset")
    digest = sha(dataset.read_bytes())
    state = {"dataset": {"host": "remote", "status": "done", "sync": {"files": {"dataset.h5": digest}},
                         "remote_verified": {"files": {"dataset.h5": digest}}}}
    return SimpleNamespace(root=tmp_path, mirror=mirror, manifest=mirror / "MANIFEST.json", runs=runs,
                           remote=remote, client=client, result=result, data=data, state=state)


def localize(fixture, job=None, **kwargs):
    job = job or row()
    kwargs.setdefault("ownership", {job["id"]: placed(job["id"])})
    kwargs.setdefault("source_identity", IDENTITY)
    return rw.localize(job, fixture.state, roles=ROLES, mirror=fixture.mirror, manifest=fixture.manifest,
                       runs=fixture.runs, root=fixture.root, **kwargs)


@pytest.mark.parametrize("script,role,local,remote,host", [
    ("scripts/train_realworld_fql.py", "realworld_fql", True, True, "remote"),
    ("scripts/realworld_fql_dataset.py", "realworld_fql", False, True, "remote"),
    ("scripts/realworld_native_depth.py", "realworld_fql", False, True, "remote"),
    ("scripts/train_sincro.py", "realworld_pretrain", False, True, "remote"),
    ("scripts/realworld_preprocess.py", "realworld_preprocess", False, True, "remote"),
    ("scripts/train_rl.py", "metaworld", True, False, "local"),
])
def test_admission_matrix_and_existing_placement(script, role, local, remote, host):
    args = ["--config", "configs/realworld/fql/base.yaml"] if role != "metaworld" else ["--task", "hammer"]
    job = row(script=script, role=role, args=args)
    assert (host_roles.refusal(job, "local", ROLES) is None) == local
    assert (host_roles.refusal(job, "remote", ROLES) is None) == remote
    assert host_roles.role_host(job, ROLES) == host
    if script == "scripts/train_realworld_fql.py":
        assert host_roles.role_hosts(job, ROLES) == ("local", "remote")
        assert host_roles.role_host({**job, "host": "local"}, ROLES) == "local"
        assert host_roles.role_host({**job, "host": "inference"}, ROLES) is None
    if role == "realworld_fql" and not local:
        assert host_roles.refusal(job, "local", ROLES).startswith("job class realworld_fql on local is limited to")


def test_generic_role_replacement_does_not_move_fql(tmp_path):
    queue, registry = queue_files(tmp_path, [row(args=[], deps=[])])
    result = host_roles.replace_to_local(lambda _: False, queue_path=queue, registry_path=registry,
                                        manifest=tmp_path / "pool.json", ownership_dir=tmp_path / "ownership",
                                        admission_lock=tmp_path / "local.lock", root=tmp_path)
    assert result["moved"] == []
    assert jobs.load_queue(queue)["jobs"][0]["host"] == "remote"


def test_inventory_fetch_and_verify(mirrored):
    f = mirrored
    assert f.result["missing"] == [] and f.result["fetched_bytes"] == sum(map(len, f.data.values()))
    assert len(f.client.commands) == 2  # requested files/descriptors, then the validated episode-directory batch
    assert "find " in f.client.commands[1] and "-maxdepth 1 -type f" in f.client.commands[1]
    spec = json.loads(f.manifest.read_bytes())
    assert spec["version"] == 1 and spec["remote_root"] == PREFIX
    assert spec["local_root"] == "runs/realworld-inputs"
    assert (f.mirror / "_remote" / STORE).read_bytes() == f.data[STORE]
    assert (f.mirror / STORE).read_bytes() == f.data[STORE].replace(PREFIX.encode(), str(f.mirror).encode())
    assert set(spec["files"][STORE]["requires"]) == {p for p in f.data if p.startswith(EPISODE + "/")}
    assert spec["files"]["_remote/" + STORE]["kind"] == "copy"
    assert rw.verify(mirror=f.mirror) == [] and rw.verify(False, mirror=f.mirror) == []
    f.client.downloads.clear()
    result = rw.fetch([SPLIT, GATE, STORE], f.client, mirror=f.mirror, priority="dependency")
    assert result["fetched_bytes"] == 0 and not f.client.downloads
    assert len(result["already_present"]) == len(f.data)
    original = f.manifest.read_bytes()
    (f.mirror / GATE).unlink()
    result = rw.fetch([GATE], f.client, mirror=f.mirror, dry_run=True)
    assert result["missing"] == [GATE] and result["fetched_bytes"] == len(f.data[GATE])
    assert f.manifest.read_bytes() == original and not (f.mirror / GATE).exists()
    assert rw.fetch([GATE], f.client, mirror=f.mirror)["missing"] == []


@pytest.mark.parametrize("relative", ["../secret", "/absolute", "a/../../x", "a\nunsafe", "a\runsafe", "a\0b", "name..json"])
def test_remote_plan_rejects_unsafe_paths_without_ssh(relative):
    class NoCalls:
        def checked_ssh(self, *args, **kwargs):
            pytest.fail("invalid paths must fail before a remote call")
    with pytest.raises(rw.LocalInputError):
        rw.remote_plan([relative], NoCalls())


@pytest.mark.parametrize("edit", ["outside", "not_compact", "incomplete", "symlink_parent"])
def test_remote_descriptor_and_episode_fail_closed(mirrored, edit):
    f = mirrored
    path = f.remote / STORE
    if edit == "outside":
        path.write_text("kind: compact\npaths: [/home/compu/elsewhere/episode]\n")
    elif edit == "not_compact":
        path.write_text("kind: other\npaths: []\n")
    elif edit == "incomplete":
        (f.remote / EPISODE / "_COMPLETE.json").unlink()
    else:
        actual = f.remote / "actual"
        (f.remote / "processed").rename(actual)
        (f.remote / "processed").symlink_to(actual, target_is_directory=True)
    with pytest.raises((rw.LocalInputError, subprocess.CalledProcessError)):
        rw.remote_plan([STORE], f.client, mirror=f.mirror)


def test_localize_preserves_every_other_argument_and_fingerprint(mirrored):
    job = row()
    original, fingerprint = copy.deepcopy(job), jobs.job_fingerprint(job)
    effective, record = localize(mirrored, job)
    assert effective == [a.replace(PREFIX, str(mirrored.mirror)) for a in job["args"]]
    assert job == original and jobs.job_fingerprint(job) == fingerprint
    assert record["manifest_sha256"] == sha(mirrored.manifest.read_bytes())
    assert record["inputs"][PREFIX + "/" + SPLIT] == sha(mirrored.data[SPLIT])
    assert record["dependency_inputs"] == {"runs/dataset/dataset.h5": sha(b"verified dataset")}
    assert rw.needed_inputs(job) == {"mirror": sorted([GATE, SPLIT, STORE]),
                                     "dependency_files": [("dataset", "dataset.h5")]}
    args = ["--set", PREFIX + "/" + SPLIT, "data.root=" + PREFIX, "untouched=abc"]
    assert rw.needed_inputs(row(args=args))["mirror"] == [".", SPLIT]
    effective, _ = localize(mirrored, row(args=args[:2]))
    assert effective == ["--set", str(mirrored.mirror / SPLIT)]


@pytest.mark.parametrize("edit,reason", [
    ("missing", "missing mirror file"), ("size", "size/mtime changed"), ("mtime", "size/mtime changed"),
    ("hash", "sha256 mismatch"), ("dataset", "dependency dataset sha256 mismatch"),
    ("remote_verified", "dependency dataset sha256 mismatch"), ("outside", "unmappable"),
    ("symlink", "symlink"), ("parent_symlink", "symlink"), ("episode", "missing mirror file"),
    ("original", "missing mirror file"), ("manifest", "input not mirrored"),
])
def test_localize_refusals(mirrored, edit, reason):
    f, job = mirrored, row()
    path = f.mirror / SPLIT
    if edit == "missing":
        path.unlink()
    elif edit == "size":
        os.chmod(path, 0o644)  # published mirror files are read-only; the corruption is deliberate
        path.write_bytes(path.read_bytes() + b" ")
    elif edit == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    elif edit == "hash":
        stat = path.stat()
        os.chmod(path, 0o644)
        path.write_bytes(b"x" * stat.st_size)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert rw.verify(False, mirror=f.mirror) == []  # quick audit deliberately uses metadata
        assert rw.verify(mirror=f.mirror)
    elif edit == "dataset":
        (f.runs / "dataset/dataset.h5").write_bytes(b"changed dataset")
    elif edit == "remote_verified":
        f.state["dataset"]["remote_verified"]["files"]["dataset.h5"] = "0" * 64
    elif edit == "outside":
        job["args"].append("data.other=/home/compu/not-realworld/input")
    elif edit == "symlink":
        path.unlink()
        path.symlink_to(f.remote / SPLIT)
    elif edit == "parent_symlink":
        actual = f.root / "actual"
        (f.mirror / "learning_inputs").rename(actual)
        (f.mirror / "learning_inputs").symlink_to(actual, target_is_directory=True)
    elif edit == "episode":
        (f.mirror / EPISODE / "episode.h5").unlink()
    elif edit == "original":
        (f.mirror / "_remote" / STORE).unlink()
    else:
        f.manifest.unlink()
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(f, job)


def test_pinned_scope_is_recorded_at_placement_and_compared_at_launch(mirrored, monkeypatch):
    f, job = mirrored, row(planned_commit="a" * 40)
    config = f.root / "configs/realworld/fql/base.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_bytes(b"seed: 0\n")
    digest = sha(config.read_bytes())
    calls = []

    def scope(scripts, commit, root):
        calls.append((scripts, commit, root))
        return {"identical": True, "files": {"scripts/train_realworld_fql.py": "b" * 64},
                "third_party": {"third_party/x@commit": "c" * 40}}

    monkeypatch.setattr(migration, "execution_scope", scope)
    monkeypatch.setattr(migration, "commit_reader", lambda root, commit: (lambda _: b"seed: 0\n", [
        "configs/realworld/fql/base.yaml"]))
    proven = REAL_PLACEMENT_SCOPE(job, root=f.root)
    assert proven == {"source_commit": "a" * 40, "files": {
        "scripts/train_realworld_fql.py": "b" * 64, "third_party/x@commit": "c" * 40,
        "configs/realworld/fql/base.yaml": digest}}
    monkeypatch.setattr(migration, "execution_scope", lambda *a, **kw: pytest.fail("no git scope reads at launch"))
    owned = {job["id"]: placed(job["id"], proven)}
    identity = {"source_sha256": proven["files"]}
    effective, record = rw.localize(job, f.state, roles=ROLES, mirror=f.mirror, runs=f.runs, root=f.root,
                                    ownership=owned, source_identity=identity)
    assert record["execution_scope"] == proven and len(calls) == 1
    identity["source_sha256"] = {**proven["files"], "scripts/train_realworld_fql.py": "e" * 64}
    with pytest.raises(rw.LocalInputError, match="local code differs from the release"):
        rw.localize(job, f.state, roles=ROLES, mirror=f.mirror, runs=f.runs, root=f.root,
                    ownership=owned, source_identity=identity)
    with pytest.raises(rw.LocalInputError, match="local code differs from the release"):
        localize(f, job, source_identity={})


def test_placement_scope_refuses_changed_release_and_argument_file(mirrored, monkeypatch):
    monkeypatch.setattr(migration, "execution_scope", lambda *a: {"identical": False})
    with pytest.raises(rw.LocalInputError, match="local code differs"):
        REAL_PLACEMENT_SCOPE(row(planned_commit="a" * 40), root=mirrored.root)
    monkeypatch.setattr(migration, "execution_scope", lambda *a: {"identical": True, "files": {}})
    monkeypatch.setattr(migration, "commit_reader", lambda *a: (lambda _: b"old", ["config.json"]))
    (mirrored.root / "config.json").write_bytes(b"new")
    with pytest.raises(rw.LocalInputError, match="local code differs"):
        REAL_PLACEMENT_SCOPE(row(planned_commit="a" * 40, args=["config.json"]), root=mirrored.root)
    with pytest.raises(rw.LocalInputError, match="no pinned release"):
        REAL_PLACEMENT_SCOPE({**row(), "planned_commit": None}, root=mirrored.root)


@pytest.mark.parametrize("script,args,override", [
    ("scripts/train_rl.py", ["--task", "hammer", "seed=0"], None),
    ("tests/_job_worker.py", ["0", "--name", "plain"], None),
    ("scripts/train_realworld_fql.py", ["--set", "data.split_manifest=" + PREFIX + "/" + SPLIT], "mapped"),
])
def test_pinned_real_launch_argv_and_wrapper(tmp_path, monkeypatch, script, args, override):
    captured = {}

    def popen(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(jobs.subprocess, "Popen", popen)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "7")
    expected_env = {**os.environ, "CUDA_VISIBLE_DEVICES": LOCAL_GPU, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                    "S4D_HOST_CONFIG": str(jobs.HOST_CONFIG_PATH), "S4D_EXIT_FILE": str(tmp_path / "pinned/exit_code")}
    expected_env.pop("MUJOCO_EGL_DEVICE_ID")
    job = row("pinned", script=script, args=args, host="local")
    effective = [a.replace(PREFIX, str(tmp_path / "mirror")) for a in args] if override else args
    if override:
        jobs.launch(job, LOCAL_GPU, 1, tmp_path, args=effective)
    else:
        jobs.launch(job, LOCAL_GPU, 1, tmp_path)
    assert captured["argv"] == ["setsid", "nohup", "bash", "-c",
                                'echo 0 > /proc/self/oom_score_adj; "$@"; echo $? > "$S4D_EXIT_FILE"', "_",
                                str(jobs.PYTHON), "-I", str(jobs.REPO / script), *effective]
    assert captured["env"] == expected_env
    assert captured["cwd"] == jobs.REPO and captured["stdin"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.STDOUT


def queue_files(root, rows):
    queue, registry = root / "queue.yaml", root / "registry.jsonl"
    queue.write_text("# preserved header\n" + yaml.safe_dump({"hosts": {
        "local": {"limits": {LOCAL_GPU: {"max_jobs": 2, "max_mem_gb": 12}}},
        "remote": {"enabled": False, "limits": {}}}, "jobs": rows}, sort_keys=False))
    registry.touch()
    return queue, registry


def planning(f, rows, **kwargs):
    state = kwargs.pop("state", f.state)
    return rw.plan({"hosts": {"local": {"limits": {LOCAL_GPU: {"max_jobs": 1, "max_mem_gb": 8}}}}, "jobs": rows},
                   state, ROLES, paused_fn=kwargs.pop("paused_fn", lambda _: []), mirror=f.mirror,
                   runs=f.runs, root=f.root, capacity=kwargs.pop("capacity", {"ram_free_bytes": 10 * 2**30}), **kwargs)


def test_placement_plan_rules_and_priority(mirrored):
    f = mirrored
    rows = [row("launched"), row("paused"), row("deps", deps=["missing"]), row("unmirrored", args=[PREFIX + "/missing"]),
            row("second", priority=2), row("first", priority=0),
            row("dataset-row", script="scripts/realworld_fql_dataset.py")]
    state = {**f.state, "launched": {"status": "pending", "attempts": 0}}
    decisions = planning(f, rows, state=state, paused_fn=lambda i: ["hold"] if i == "paused" else [])
    by_id = {d["id"]: d for d in decisions}
    assert [d["id"] for d in decisions][:1] == ["first"]
    assert "dataset-row" not in by_id
    assert by_id["launched"]["reason"] == "launched or started rows never move"
    assert by_id["paused"]["reason"] == "paused"
    assert by_id["deps"]["decision"] == "waits" and "dependencies" in by_id["deps"]["reason"]
    assert by_id["unmirrored"]["decision"] == "waits" and "not mirrored" in by_id["unmirrored"]["reason"]
    assert by_id["first"]["decision"] == "local"
    assert by_id["second"]["reason"] == "local capacity"
    assert planning(f, [row("paused")], paused_fn=lambda _: ["hold"], allow_pauses=["hold"])[0]["decision"] == "local"
    assert planning(f, [row()], capacity={"ram_free_bytes": 0})[0]["reason"] == "local capacity"
    assert planning(f, [row()], capacity={"ram_free_bytes": 0}, policy="all")[0]["decision"] == "local"
    assert planning(f, [row()], ownership={"fql": {}})[0]["reason"] == "ownership record exists"
    (f.runs / "fql").mkdir()
    assert planning(f, [row()])[0]["reason"] == "local output directory exists"


def test_placement_reserves_active_and_already_placed_capacity(mirrored):
    f = mirrored
    active = {**f.state, "running": {"host": "local", "status": "running", "gpu": LOCAL_GPU, "mem_gb": 8}}
    assert planning(f, [row()], state=active)[0]["reason"] == "local capacity"
    placed = row("placed", host="local", pin_host="local", gpu="any", placement="realworld-local yesterday")
    assert planning(f, [placed, row()])[0]["reason"] == "local capacity"
    assert planning(f, [placed, row()], policy="all")[0]["decision"] == "local"
    # paused or dependency-blocked local rows (e.g. a paused Meta-World campaign) reserve no capacity
    assert planning(f, [{**placed, "id": "held"}, row()], paused_fn=lambda i: ["hold"] if i == "held" else [])[0][
        "decision"] == "local"
    assert planning(f, [{**placed, "id": "blocked", "deps": ["missing"]}, row()])[0]["decision"] == "local"
    assert planning(f, [row(args=[PREFIX + "/unknown"])], remote_plan={
        "unknown": {"bytes": 1234, "sha256": "0" * 64, "kind": "copy", "requires": []}})[0]["needs_mirror_bytes"] == 1234


def apply_placement(f, ids, **kwargs):
    return rw.apply(ids, "synthetic directive P", queue_path=f.root / "queue.yaml", registry_path=f.root / "registry.jsonl",
                    manifest=f.root / "pool.json", ownership_dir=f.root / "ownership", admission_lock=f.root / "local.lock",
                    remote_lock=f.root / "remote.lock", mirror=f.mirror, input_manifest=f.manifest, root=f.root,
                    paused_fn=kwargs.pop("paused_fn", lambda _: []), **kwargs)


def test_apply_preserves_science_and_records_ownership(mirrored, monkeypatch):
    f, job = mirrored, row(planned_commit="a" * 40)
    queue, registry = queue_files(f.root, [job, row("dataset", script="scripts/realworld_fql_dataset.py", deps=[], args=[])])
    # A completed dependency is enough for the synthetic registry; its backed-up digest comes from a launched event.
    jobs.record({"event": "launched", "id": "dataset", "attempt": 1, "host": "remote", "gpu": REMOTE_GPU}, registry)
    jobs.record({"event": "results_synced", "id": "dataset", "sync": {
        "final": True, "verified": True, "files": f.state["dataset"]["sync"]["files"]}}, registry)
    jobs.record({"event": "exited", "id": "dataset", "attempt": 1, "code": 0}, registry)
    scope = {"source_commit": "a" * 40, "files": {job["script"]: "b" * 64}}
    monkeypatch.setattr(rw, "placement_scope", lambda *a, **kw: scope)
    before = queue.read_bytes()
    result = apply_placement(f, ["fql"], paused_fn=lambda _: ["hold"], allow_pauses=["hold"])
    assert result["moved"] == ["fql"] and queue.read_text().startswith("# preserved header")
    moved = jobs.load_queue(queue)["jobs"][0]
    assert moved["host"] == moved["pin_host"] == "local" and moved["gpu"] == "any"
    assert moved["placement"].startswith("realworld-local ")
    assert jobs.job_fingerprint(moved) == jobs.job_fingerprint(job)
    assert {k: v for k, v in moved.items() if k not in host_roles.HOST_FIELDS} == {
        k: v for k, v in job.items() if k not in host_roles.HOST_FIELDS}
    pool = json.loads((f.root / "pool.json").read_text())
    move = pool["moves"][0]
    assert move["ids"] == ["fql"] and move["to"] == "local" and move["original"]["fql"] == job
    assert move["queue_sha256_before"] == sha(before) and move["queue_sha256_after"] == sha(queue.read_bytes())
    assert move["allow_pauses"] == ["hold"] and move["execution_scopes"]["fql"] == scope
    assert ROLES["sha256"] in move["reason"]
    owned = host_roles.read_ownership(f.root / "ownership")
    assert owned["fql"]["execution_scope"] == scope and owned["fql"]["history"] == []
    assert host_roles.refusal(moved, "remote", ROLES, owned).startswith("owned by local")
    assert host_roles.refusal(moved, "local", ROLES, owned) is None


@pytest.mark.parametrize("lock_name", ["remote.lock", "local.lock", "registry.lock"])
def test_apply_refuses_busy_locks(mirrored, lock_name):
    f = mirrored
    queue, _ = queue_files(f.root, [row(deps=[])])
    before = queue.read_bytes()
    with (f.root / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert apply_placement(f, ["fql"]) == {"moved": [], "busy": True}
    assert queue.read_bytes() == before


def test_apply_rechecks_launched_since_plan(mirrored):
    f = mirrored
    job = row(deps=[], args=[])
    assert planning(f, [job])[0]["decision"] == "local"
    queue, registry = queue_files(f.root, [job])
    jobs.record({"event": "launching", "id": "fql", "host": "remote", "attempt": 1}, registry)
    result = apply_placement(f, ["fql"])
    assert result["moved"] == [] and result["kept"]["fql"] == "launched or started rows never move"
    assert jobs.load_queue(queue)["jobs"][0] == job


def tick(f, monkeypatch, rows, events=()):
    queue, registry = queue_files(f.root, rows)
    for event in events:
        jobs.record(event, registry)
    launches = []

    def launch(job, gpu, attempt, runs, **kwargs):
        launches.append((job["id"], kwargs))
        return 2**30

    monkeypatch.setattr(jobs, "launch", launch)
    monkeypatch.setattr(jobs, "source_identity", lambda: copy.deepcopy(IDENTITY))
    monkeypatch.setattr(host_roles, "OWNERSHIP_DIR", f.root / "ownership")
    for job in rows:
        if job.get("host") == "local" and job["script"] == "scripts/train_realworld_fql.py" and not job.get("unplaced"):
            host_roles.write_ownership(placed(job["id"]), f.root / "ownership")
    monkeypatch.setattr(host_roles, "EXCEPTIONS_PATH", f.root / "no-exceptions")
    monkeypatch.setattr(rw, "MIRROR", f.mirror)
    monkeypatch.setattr(rw, "MANIFEST", f.manifest)
    messages = jobs.tick(queue, registry, f.runs, usage={}, available_gb=500,
                         disk_bytes=(2000 * 2**30, 1000 * 2**30))
    records = [json.loads(line) for line in registry.read_text().splitlines()]
    return launches, messages, [e for e in records if e["event"] == "launched"]


def test_scheduler_mapped_launch_and_unchanged_metaworld_event(mirrored, monkeypatch):
    f = mirrored
    job = row(host="local", pin_host="local", gpu="any", deps=[], args=["data.split_manifest=" + PREFIX + "/" + SPLIT])
    meta = row("meta", host="local", pin_host="local", gpu="any", role="metaworld", script="scripts/train_rl.py",
               args=["--task", "hammer"], deps=[])
    launches, messages, events = tick(f, monkeypatch, [job, meta])
    assert launches == [("fql", {"args": ["data.split_manifest=" + str(f.mirror / SPLIT)]}), ("meta", {})]
    assert events[0]["inputs"]["inputs"] == {PREFIX + "/" + SPLIT: sha(f.data[SPLIT])}
    assert "inputs" not in events[1] and any("launched fql" in m for m in messages)


def test_scheduler_missing_mirror_is_refused(mirrored, monkeypatch):
    job = row(host="local", pin_host="local", gpu="any", deps=[], args=[PREFIX + "/missing.json"])
    launches, messages, events = tick(mirrored, monkeypatch, [job])
    assert not launches and not events
    assert "fql: not admitted on local: input not mirrored: missing.json" in messages


@pytest.mark.parametrize("changed", ["inputs", "dependency_inputs"])
def test_scheduler_retry_inputs_changed_is_held(mirrored, monkeypatch, changed):
    job = row(host="local", pin_host="local", gpu="any", deps=[], args=[PREFIX + "/" + SPLIT])
    old = {"inputs": {PREFIX + "/" + SPLIT: sha(mirrored.data[SPLIT])}, "dependency_inputs": {}}
    old[changed] = {"changed": "0" * 64}
    events = [{"event": "launched", "id": "fql", "host": "local", "attempt": 1, "gpu": LOCAL_GPU,
               "pid": 2**30, "inputs": old}, {"event": "exited", "id": "fql", "attempt": 1, "code": 1}]
    launches, messages, _ = tick(mirrored, monkeypatch, [job], events)
    assert not launches and "fql: retry inputs changed; admission held" in messages


def test_closeout_is_immutable_and_checks_required_results(tmp_path):
    runs, finished = tmp_path / "runs", tmp_path / "finished"
    result = runs / "fql/final.json"
    result.parent.mkdir(parents=True)
    state = {"fql": {"status": "done", "host": "local", "inputs": {"inputs": {"split": "digest"}},
                     "commit": "c", "gpu": LOCAL_GPU, "attempt": 1}}
    with pytest.raises(rw.LocalInputError, match="missing"):
        rw.closeout("fql", state, row(), finished, runs=runs)
    result.touch()
    with pytest.raises(rw.LocalInputError, match="empty"):
        rw.closeout("fql", state, row(), finished, runs=runs)
    result.write_bytes(b'{"score": 1}\n')
    record = rw.closeout("fql", state, row(), finished, runs=runs)
    assert record["files"] == {"final.json": sha(result.read_bytes())}
    assert record["inputs"] == state["fql"]["inputs"]
    assert json.loads((finished / "fql.json").read_text()) == record
    assert rw.closeout("fql", state, row(), finished, runs=runs) == record
    result.write_bytes(b"changed")
    with pytest.raises(rw.LocalInputError, match="different closeout"):
        rw.closeout("fql", state, row(), finished, runs=runs)


def test_local_driver_source_and_local_rows():
    source = (jobs.REPO / "ops/campaign/baseline_local_driver.py").read_text()
    assert "realworld_local.local_rows(queue['jobs'], host_roles.load_roles())" in source
    assert "host_roles.refusal(j, 'local', _roles, _ownership, _context) is None" in source
    assert "job['script'] == 'scripts/train_realworld_fql.py'" in source
    local = row(host="local")
    assert rw.local_rows([row("remote"), local, row("builder", host="local", script="scripts/realworld_fql_dataset.py")],
                         ROLES) == [local]


def test_active_pause_names_match_existing_paused_behavior(monkeypatch):
    monkeypatch.setattr(admission_pause, "_ACTIVE", [
        {"name": "exact", "ids": ["trainer"]}, {"name": "prefix", "prefixes": ["held-"]},
        {"name": "both", "ids": ["trainer-eval"], "prefixes": ["trainer"]}])
    for job_id in ("trainer", "trainer-eval", "held-job", "held-job-eval", "free", "free-eval"):
        assert bool(admission_pause.active_pauses(job_id)) == admission_pause.paused(job_id)
    assert admission_pause.active_pauses("trainer-eval") == ["exact", "both"]


def test_large_copy_uses_metadata_at_admission_and_full_audit_hashes_it(mirrored, monkeypatch):
    f = mirrored
    monkeypatch.setattr(rw, "SMALL_HASH_BYTES", 1)
    path = f.mirror / EPISODE / "episode.h5"
    stat = path.stat()
    os.chmod(path, 0o644)
    path.write_bytes(b"x" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    localize(f)
    assert rw.verify(False, mirror=f.mirror) == []
    assert any("sha256 mismatch" in p for p in rw.verify(mirror=f.mirror))


def test_missing_byte_estimate_does_not_charge_an_intact_descriptor(mirrored):
    f = mirrored
    path = f.mirror / EPISODE / "episode.h5"
    size = path.stat().st_size
    path.unlink()
    decision = planning(f, [row()])[0]
    assert decision["decision"] == "waits" and decision["needs_mirror_bytes"] == size


def test_localized_descriptor_must_match_its_remote_original(mirrored):
    f = mirrored
    path = f.mirror / STORE
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b"# changed\n")
    spec = json.loads(f.manifest.read_text())
    spec["files"][STORE].update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                                sha256=sha(path.read_bytes()))
    f.manifest.write_text(json.dumps(spec))
    with pytest.raises(rw.LocalInputError, match="localized descriptor differs"):
        localize(f)


def test_scope_difference_is_a_stay_remote_decision(mirrored, monkeypatch):
    def refuse(*a, **kw):
        raise rw.LocalInputError("local code differs from the release the row was placed with")
    monkeypatch.setattr(rw, "placement_scope", refuse)
    decision = planning(mirrored, [row(planned_commit="a" * 40)])[0]
    assert decision["decision"] == "stay_remote" and "local code differs" in decision["reason"]


def test_interrupted_placement_keeps_remote_admission_held(mirrored, monkeypatch):
    f = mirrored
    queue_files(f.root, [row(args=[], deps=[])])

    def killed(*a, **kw):
        raise OSError("killed before queue rewrite")

    monkeypatch.setattr(host_roles, "write_queue", killed)
    with pytest.raises(OSError, match="killed"):
        apply_placement(f, ["fql"])
    owned = host_roles.read_ownership(f.root / "ownership")
    remote_row = jobs.load_queue(f.root / "queue.yaml")["jobs"][0]
    assert remote_row["host"] == "remote"
    assert host_roles.refusal(remote_row, "remote", ROLES, owned).startswith("owned by local")


def test_apply_rereads_pauses_added_since_plan(mirrored, monkeypatch):
    f = mirrored
    job = row(args=[], deps=[])
    queue_files(f.root, [job])
    assert planning(f, [job])[0]["decision"] == "local"
    pause_file = f.root / "pauses.json"
    monkeypatch.setattr(admission_pause, "FILE", pause_file)
    monkeypatch.setattr(admission_pause, "_ACTIVE", [])
    pause_file.write_text(json.dumps({"pauses": [{"name": "new-hold", "active": True, "ids": ["fql"]}]}))
    assert admission_pause.active_pauses("fql", fresh=True) == ["new-hold"]
    result = apply_placement(f, ["fql"], paused_fn=None)
    assert result == {"moved": [], "kept": {"fql": "paused"}}
    result = apply_placement(f, ["fql"], paused_fn=None, allow_pauses=["new-hold"])
    assert result["moved"] == ["fql"]
    assert json.loads(pause_file.read_text())["pauses"][0]["active"] is True


def test_plain_remote_paths_may_contain_equals(mirrored):
    f = mirrored
    relative = "learning_inputs/plain=split.json"
    rw._atomic(f.remote / relative, b"{}")
    rw.fetch([relative], f.client, mirror=f.mirror)
    job = row(args=[PREFIX + "/" + relative], deps=[])
    assert rw.needed_inputs(job)["mirror"] == [relative]
    effective, _ = localize(f, job)
    assert effective == [str(f.mirror / relative)]


def test_plan_reports_unmappable_inputs_without_aborting_other_decisions(mirrored):
    decisions = planning(mirrored, [row("bad", args=["data.other=/home/compu/outside/input"]), row("good")])
    assert decisions[0]["decision"] == "waits" and "unmappable" in decisions[0]["reason"]
    assert decisions[1]["decision"] == "local"


# ---------------------------------------------------------------------------------------------------- review round 1
# (Codex read-only review 2026-11-11, findings 1-6; fixes by Claude)
@pytest.mark.parametrize("arg,reason", [
    ("data.dataset='runs/dataset/dataset.h5'", "quoted or escaped"),  # the trainer yaml-parses override values
    ('data.split_manifest="' + PREFIX + "/" + SPLIT + '"', "quoted or escaped"),
    ("data.other=runs/not-a-dependency/dataset.h5", "not an output of a declared dependency"),
])
def test_inputs_the_trainer_would_read_differently_fail_closed(mirrored, arg, reason):
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(mirrored, row(args=["--set", arg]))


@pytest.mark.parametrize("ownership,job", [
    ({}, None),  # registered or edited onto local without the locked placement command
    ({"fql": {**placed(), "kind": "migration"}}, None),
    ({"fql": {**placed(), "owner_host": "remote"}}, None),
    ({"fql": placed()}, {**row(), "planned_commit": None}),  # no pinned release
    ({"fql": placed(scope={"source_commit": "f" * 40, "files": SCOPE["files"]})}, None),  # placed for another release
])
def test_local_launch_requires_the_explicit_placement_record(mirrored, ownership, job):
    with pytest.raises(rw.LocalInputError, match="placement record|code differs"):
        localize(mirrored, job, ownership=ownership)


def test_rows_without_a_pinned_release_stay_remote_and_unplaced_local_rows_are_refused(mirrored, monkeypatch):
    decision = planning(mirrored, [{**row(), "planned_commit": None}])[0]
    assert decision["decision"] == "stay_remote" and "no pinned release" in decision["reason"]
    job = row(host="local", pin_host="local", gpu="any", deps=[], args=[PREFIX + "/" + SPLIT], unplaced=True)
    launches, messages, _ = tick(mirrored, monkeypatch, [job])
    assert not launches and any("no explicit local placement record" in m for m in messages)


def test_mirror_is_append_only_and_read_only(mirrored):
    f = mirrored
    published = f.mirror / SPLIT
    before = published.read_bytes()
    assert all(not (f.mirror / name).stat().st_mode & 0o222 for name in (SPLIT, STORE, "_remote/" + STORE))
    (f.remote / SPLIT).write_bytes(b'{"train": ["other"]}\n')  # the remote original changed after mirroring
    with pytest.raises(rw.LocalInputError, match="changed since they were mirrored"):
        rw.fetch([SPLIT], f.client, mirror=f.mirror)
    assert published.read_bytes() == before and localize(f, row(args=[PREFIX + "/" + SPLIT], deps=[]))


def test_retry_identity_covers_the_transitive_input_closure(mirrored, monkeypatch):
    f = mirrored
    job = row(host="local", pin_host="local", gpu="any", deps=[], args=["data.store=" + PREFIX + "/" + STORE])
    _, record = rw.localize(job, f.state, roles=ROLES, mirror=f.mirror, manifest=f.manifest, runs=f.runs,
                            ownership={"fql": placed()}, source_identity=IDENTITY)
    assert {EPISODE + "/episode.h5", EPISODE + "/_COMPLETE.json", "_remote/" + STORE} <= set(record["input_closure"])

    def attempt(inputs):
        return [{"event": "launched", "id": "fql", "host": "local", "attempt": 1, "gpu": LOCAL_GPU, "pid": 2**30,
                 "inputs": inputs}, {"event": "exited", "id": "fql", "attempt": 1, "code": 1}]

    launches, _, events = tick(f, monkeypatch, [job], attempt(record))
    assert [i for i, _ in launches] == ["fql"] and events[-1]["inputs"]["input_closure"] == record["input_closure"]
    changed = copy.deepcopy(record)
    changed["input_closure"][EPISODE + "/episode.h5"] = "0" * 64  # an episode refetched under the same descriptor
    (f.root / "registry.jsonl").write_text("")  # a fresh registry for the second scenario
    launches, messages, _ = tick(f, monkeypatch, [job], attempt(changed))
    assert not launches and "fql: retry inputs changed; admission held" in messages


def test_ready_local_retries_reserve_capacity(mirrored):
    f = mirrored
    local = row("retry", host="local", pin_host="local", gpu="any", deps=[])
    crashed = {**f.state, "retry": {"host": "local", "status": "crashed", "failures": 1, "attempts": 1}}
    assert planning(f, [local, row()], state=crashed)[0]["reason"] == "local capacity"
    exhausted = {**f.state, "retry": {"host": "local", "status": "crashed", "failures": 2, "attempts": 2}}
    assert planning(f, [local, row()], state=exhausted)[0]["decision"] == "local"


# ---------------------------------------------------------------------------------------------------- review round 2
# (Codex read-only re-review 2026-11-11: config-resolved inputs, output options, concurrent fetches; fixes by Claude)
@pytest.mark.parametrize("extra,reason", [
    (["data={dataset: runs/dataset/other.h5}"], "not plain checked arguments"),  # a mapping override hides a path
    (["data.extra=runs/elsewhere/file.h5"], "not an output of a declared dependency"),
])
def test_paths_resolved_by_the_trainer_must_be_checked_arguments(mirrored, extra, reason):
    job = row()
    job["args"] = job["args"][:job["args"].index("--name")] + extra + job["args"][job["args"].index("--name"):]
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(mirrored, job)


def test_paths_inside_the_config_file_must_be_checked_arguments(mirrored):
    config = mirrored.root / "configs/realworld/fql/base.yaml"
    config.write_text(config.read_text().replace("dataset: null", "dataset: runs/dataset/dataset.h5", 1))
    job = row()
    job["args"] = [a for a in job["args"] if not a.startswith("data.dataset=")]
    with pytest.raises(rw.LocalInputError, match="not plain checked arguments"):
        localize(mirrored, job)


@pytest.mark.parametrize("extra", [["--output-root", "runs/realworld_fql/custom"], ["--output-root=runs/realworld_fql/x"]])
def test_output_options_are_not_inputs(mirrored, extra):
    job = row()
    job["args"] += extra
    assert rw.needed_inputs(job)["dependency_files"] == [("dataset", "dataset.h5")]
    effective, _ = localize(mirrored, job)
    assert effective[-len(extra):] == extra


@pytest.mark.parametrize("abbreviated", [["--conf", "configs/x.yaml"], ["--se", "train.seed=1"]])
def test_abbreviated_trainer_options_fail_closed(mirrored, abbreviated):
    job = row()
    job["args"] += abbreviated
    with pytest.raises(rw.LocalInputError, match="full spellings"):
        localize(mirrored, job)


def test_fetches_are_serialized(mirrored):
    import threading

    f, done = mirrored, threading.Event()
    with open(f.mirror / ".fetch.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = threading.Thread(target=lambda: (rw.fetch([SPLIT], f.client, mirror=f.mirror), done.set()))
        worker.start()
        assert not done.wait(0.5)  # waits for the lock holder
    worker.join(10)
    assert done.is_set()
