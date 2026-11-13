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
import shutil
import subprocess
import tempfile
from pathlib import Path
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
# ops/jobs/realworld_local.py treats every absolute path under the remote host's home as a REMOTE path (by design: the
# local host has no /home/compu; asserted below). This file's synthetic LOCAL mirrors, runs and queues therefore must not
# live there - but the remote native suite's pytest basetemp does (S4D_PYTEST_TMP under
# /home/compu/kaist/sunho/runtime/verification/...; first remote run of this file: verify-native-37bfd30-cc, 23 failures
# "the effective configuration still names a remote path"). Production checks are unchanged; only the tests' own
# temporary root moves when it would sit inside the remote home.
REMOTE_HOME = "/home/compu/"


def _outside_remote_home(home: str = REMOTE_HOME) -> Path:
    """A fresh writable directory whose path is not under ``home`` (the remote verification container is read-only
    except its mounts: the repository's .cache mount, then /dev/shm, then the system temporary directory)."""
    for base in (rw.REPO / ".cache" / "pytest-realworld-local", Path("/dev/shm"), Path(tempfile.gettempdir())):
        if str(base.resolve()).startswith(home) or str(base).startswith(home):
            continue
        try:
            base.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="rwlocal-", dir=base))
        except OSError:
            continue
    raise RuntimeError(f"no writable temporary directory outside {home}")


@pytest.fixture
def tmp_path(tmp_path):  # noqa: F811 - overrides pytest's tmp_path for this module only
    """pytest's tmp_path, unless it lies inside the remote home (remote native suite): then a fresh directory outside."""
    if not str(tmp_path).startswith(REMOTE_HOME):
        yield tmp_path
        return
    path = _outside_remote_home(REMOTE_HOME)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_production_treats_the_remote_home_as_remote_on_every_host():
    """The premise of the tmp_path override: an unmapped path under REMOTE_HOME is refused as a remote input."""
    with pytest.raises(rw.LocalInputError, match="unmappable remote input"):
        rw._mapped(REMOTE_HOME + "elsewhere/input")
    assert rw._mapped(PREFIX + "/" + SPLIT) == SPLIT and rw._mapped("/srv/elsewhere") is None


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
    ("scripts/train_rl.py", "metaworld", True, False, "local"),  # 2026-11-12 (user): no Meta-World data remotely
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
        assert host_roles.refusal(job, "local", ROLES).startswith(
            "data locality: inputs live on the remote; no local input resolver for")


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
        path.write_text(f"kind: compact\npaths: [{REMOTE_HOME}elsewhere/episode]\n")
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
        job["args"].append(f"data.other={REMOTE_HOME}not-realworld/input")
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
    expected_env.update(dict.fromkeys(jobs.THREAD_ENV, "8"))
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


def test_local_remote_resident_launch_bounds_thread_pools_like_the_container(tmp_path, monkeypatch):
    """2026-11-12: local train_realworld_fql.py runs spanned ~95 cores of the shared host; launches of remote-resident
    rows get OMP / MKL / OpenBLAS = cpu_threads (the remote container's quota); other launches keep their env."""
    captured = []
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda argv, **kw: captured.append(kw["env"]) or SimpleNamespace(pid=1))
    for name in jobs.THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    job = row("bounded", host="local")
    jobs.launch(job, LOCAL_GPU, 1, tmp_path, args=job["args"], cpu_threads=8)
    jobs.launch({**job, "id": "four"}, LOCAL_GPU, 1, tmp_path, args=job["args"], cpu_threads=4)
    jobs.launch({**job, "id": "plain"}, LOCAL_GPU, 1, tmp_path)
    assert [{k: e.get(k) for k in jobs.THREAD_ENV} for e in captured] == [
        dict.fromkeys(jobs.THREAD_ENV, "8"), dict.fromkeys(jobs.THREAD_ENV, "4"), dict.fromkeys(jobs.THREAD_ENV, "8")]
    assert jobs.THREAD_ENV == ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    for bad in (0, -1, 65, 2.0, 8.9, "8", True, None):
        with pytest.raises(ValueError, match="cpu_threads"):
            jobs.launch({**job, "id": "bad"}, LOCAL_GPU, 1, tmp_path, cpu_threads=bad)
    monkeypatch.setenv("OMP_NUM_THREADS", "384")  # an inherited limit is overridden, not kept
    jobs.launch({**job, "id": "inherited"}, LOCAL_GPU, 1, tmp_path, args=job["args"], cpu_threads=64)
    assert captured[-1]["OMP_NUM_THREADS"] == "64"
    jobs.launch({**job, "id": "plain2"}, LOCAL_GPU, 1, tmp_path)  # ... and left alone elsewhere
    assert captured[-1]["OMP_NUM_THREADS"] == "8"
    from ops.containers import launch as container  # the quota this mirrors
    assert (jobs.DEFAULT_CPU_THREADS, jobs.MAX_CPU_THREADS) == (container.DEFAULT_CPU_THREADS, container.MAX_CPU_THREADS)
    assert '("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")' in (
        jobs.REPO / "ops/containers/launch.py").read_text()


@pytest.mark.parametrize("value,expected", [(None, "absent"), (4, 4), (64, 64), (0, "refused"), (65, "refused"),
                                            ("8", "refused"), (True, "refused"), (8.9, "refused")])
def test_scheduler_passes_the_rows_thread_quota_to_local_remote_resident_launches(mirrored, monkeypatch, value,
                                                                                  expected):
    f = mirrored
    job = row(host="local", pin_host="local", gpu="any", deps=[], args=["data.split_manifest=" + PREFIX + "/" + SPLIT])
    if expected != "absent":
        job["cpu_threads"] = value
    launches, messages, _ = tick(f, monkeypatch, [job])
    if expected == "refused":
        assert launches == [] and any("not admitted on local: invalid cpu_threads" in m for m in messages)
    else:
        assert launches == [(job["id"], {"args": ["data.split_manifest=" + str(f.mirror / SPLIT)],
                                         "cpu_threads": 8 if expected == "absent" else expected})]


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
    # the remote-resident FQL row gets the remote container's thread quota; the Meta-World launch is unchanged
    assert launches == [("fql", {"args": ["data.split_manifest=" + str(f.mirror / SPLIT)], "cpu_threads": 8}),
                        ("meta", {})]
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
    assert "host_roles.remote_resident(job, _roles, by_queue)" in source
    local = row(host="local")
    assert rw.local_rows([row("remote"), local, row("builder", host="local", script="scripts/realworld_fql_dataset.py")],
                         ROLES) == [local, row("builder", host="local", script="scripts/realworld_fql_dataset.py")]


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
    decisions = planning(mirrored, [row("bad", args=[f"data.other={REMOTE_HOME}outside/input"]), row("good")])
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


def pretraining_rows(f):
    """The stats/pre/export argv contract of register_ours_realworld_20261111.task_rows (read only as text)."""
    config = f.root / 'configs/realworld/ours/task.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('data: {regime: realworld}\nmodel: {}\n')
    stats = row('stats', role='realworld_pretrain', script='scripts/compute_workspace_stats.py', deps=[], args=[
        '--regime', 'realworld', '--path', PREFIX + '/' + STORE, '--split', PREFIX + '/' + SPLIT,
        '--out', 'runs/stats/workspace_stats.json'])
    pre = row('pre', role='realworld_pretrain', script='scripts/train.py', deps=['stats'], args=[
        '--config', 'configs/realworld/ours/task.yaml', '--name', 'pre', '--output-root', 'runs', '--resume', 'auto',
        '--set', 'data.store=' + PREFIX + '/' + STORE, 'data.split_manifest=' + PREFIX + '/' + SPLIT,
        'data.gate_report=' + PREFIX + '/' + GATE, 'model.anchor_stats=runs/stats/workspace_stats.json'])
    export = row('export', role='realworld_pretrain', script='scripts/export_encoder.py', deps=['pre'], args=[
        '--run', 'runs/pre', '--checkpoint', 'runs/pre/checkpoints/step_0200000.pt', '--out', 'runs/export/encoder.pt'])
    return stats, pre, export


def dependency(f, name, relative, host='local', status='done'):
    path = f.runs / name / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('verified output')
    digest = sha(path.read_bytes())
    state = f.state.setdefault(name, {'host': host, 'status': status, 'sync': {'files': {}},
                                      'remote_verified': {'files': {}}})
    state['sync']['files'][relative] = digest
    state['remote_verified']['files'][relative] = digest
    return path


def test_stats_pretraining_and_export_local_resolvers(mirrored):
    f = mirrored
    stats, pre, export = pretraining_rows(f)
    for job in (stats, pre, export):
        assert rw.applies(job, ROLES) and host_roles.role_hosts(job, ROLES) == ('local', 'remote')
        assert host_roles.role_host(job, ROLES) == 'remote'
    effective, record = localize(f, stats)
    assert effective == [a.replace(PREFIX, str(f.mirror)) for a in stats['args']]
    assert set(record['input_closure']) >= {STORE, EPISODE + '/episode.h5'}
    dependency(f, 'stats', 'workspace_stats.json')
    effective, record = localize(f, pre)
    assert 'model.anchor_stats=runs/stats/workspace_stats.json' in effective
    assert record['dependency_inputs'] == {'runs/stats/workspace_stats.json': sha(b'verified output')}
    dependency(f, 'pre', 'config.yaml')
    dependency(f, 'pre', 'checkpoints/step_0200000.pt')
    effective, record = localize(f, export)
    assert effective == export['args']
    assert set(record['dependency_inputs']) == {'runs/pre/config.yaml', 'runs/pre/checkpoints/step_0200000.pt'}
    assert rw.needed_inputs(export)['dependency_files'] == [('pre', '.'), ('pre', 'checkpoints/step_0200000.pt'),
                                                          ('pre', 'config.yaml')]


@pytest.mark.parametrize('host,status', [('local', 'running'), ('local', 'pending'), ('remote', 'sync_pending'),
                                         (None, 'done')])
def test_dependency_inputs_require_done_and_an_explicit_host(mirrored, host, status):
    f = mirrored
    _, pre, _ = pretraining_rows(f)
    dependency(f, 'stats', 'workspace_stats.json', host, status)
    with pytest.raises(rw.LocalInputError, match='dependency must be done'):
        localize(f, pre)


def test_pretraining_remote_dependency_uses_verified_backup_checksums(mirrored):
    f = mirrored
    _, pre, export = pretraining_rows(f)
    output = dependency(f, 'stats', 'workspace_stats.json', 'remote')
    assert localize(f, pre)[1]['dependency_inputs']
    output.write_text('changed')
    with pytest.raises(rw.LocalInputError, match='unverified backup'):
        localize(f, pre)
    dependency(f, 'pre', 'config.yaml', 'remote')
    dependency(f, 'pre', 'checkpoints/step_0200000.pt', 'remote')
    with pytest.raises(rw.LocalInputError, match='remote dependency run directory'):
        localize(f, export)


@pytest.mark.parametrize('extra', [
    ['--pointworld-root', PREFIX + '/tree'], ['--rlds-root', PREFIX + '/rlds'],
    ['--rlds-index=' + PREFIX + '/index.json'], ['--pa', PREFIX + '/' + STORE],
])
def test_stats_unresolved_options_fail_closed(mirrored, extra):
    stats, _, _ = pretraining_rows(mirrored)
    with pytest.raises(rw.LocalInputError, match='unsupported local input options'):
        localize(mirrored, {**stats, 'args': stats['args'] + extra})


@pytest.mark.parametrize('regime', ['metaworld', 'droid', 'droid_largescale'])
def test_stats_resolver_only_covers_realworld(mirrored, regime):
    stats, _, _ = pretraining_rows(mirrored)
    stats['args'][1] = regime
    assert not rw.has_resolver(stats)
    # Classification is deliberately unchanged; our stats rows declare their pretraining role explicitly.
    assert host_roles.job_class({**stats, 'role': None}).cls != 'realworld_pretrain'


def test_pretraining_merges_all_configs_and_yaml_overrides_exactly(mirrored):
    f = mirrored
    _, pre, _ = pretraining_rows(f)
    dependency(f, 'stats', 'workspace_stats.json')
    second = f.root / 'configs/realworld/ours/second.yaml'
    second.write_text('data: {regime: metaworld, hidden: ' + PREFIX + '/hidden.h5}\n')
    pre['args'].insert(2, 'configs/realworld/ours/second.yaml')
    pre['args'] += ['data.regime=realworld']
    with pytest.raises(rw.LocalInputError, match='not plain checked arguments'):
        localize(f, pre)
    pre['args'] += ['data.hidden=null']
    assert localize(f, pre)[0]
    pre['args'] += ['data={regime: realworld, store: ' + PREFIX + '/hidden.h5}']
    with pytest.raises(rw.LocalInputError, match='not plain checked arguments'):
        localize(f, pre)


@pytest.mark.parametrize('edit,reason', [('resume', 'invalid local resolver'), ('abbreviation', 'invalid local resolver'),
                                        ('regime', 'requires data.regime'), ('quoted', 'quoted or escaped')])
def test_pretraining_fails_closed_for_resume_regime_and_path_spellings(mirrored, edit, reason):
    f = mirrored
    _, pre, _ = pretraining_rows(f)
    dependency(f, 'stats', 'workspace_stats.json')
    if edit == 'resume':
        pre['args'][pre['args'].index('--resume') + 1] = 'runs/other/checkpoint.pt'
    elif edit == 'abbreviation':
        pre['args'][0] = '--conf'
    elif edit == 'regime':
        pre['args'] += ['data.regime=droid']
    else:
        pre['args'] += ['data.store="' + PREFIX + '/' + STORE + '"']
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(f, pre)


def test_export_default_checkpoint_wait_and_local_directory_checks(mirrored):
    f = mirrored
    _, _, export = pretraining_rows(f)
    export['args'] = ['--run=runs/pre', '--out=runs/export/encoder.pt', '--wait']
    dependency(f, 'pre', 'config.yaml')
    checkpoint = dependency(f, 'pre', 'checkpoints/latest.pt')
    assert 'runs/pre/checkpoints/latest.pt' in localize(f, export)[1]['dependency_inputs']
    checkpoint.unlink()
    with pytest.raises(rw.LocalInputError, match='missing mirror file'):
        localize(f, export)


def test_new_resolvers_placement_and_local_rows(mirrored):
    f = mirrored
    stats, pre, export = pretraining_rows(f)
    decisions = planning(f, [stats, pre, export])
    assert [d['decision'] for d in decisions] == ['local', 'waits', 'waits']
    queue_files(f.root, [stats, pre, export])
    assert apply_placement(f, ['stats'])['moved'] == ['stats']
    after = jobs.load_queue(f.root / 'queue.yaml')['jobs']
    assert rw.local_rows(after, ROLES) == [after[0]]
    assert jobs.job_fingerprint(after[0]) == jobs.job_fingerprint(stats)


def test_placement_ignores_remote_holds_and_preserves_them(mirrored):
    f = mirrored
    stats, _, _ = pretraining_rows(f)
    setup, verification = f.runs / 'setup', f.runs / 'remote/verification'
    setup.mkdir(parents=True)
    verification.mkdir(parents=True)
    holds = [setup / 'remote-hold-admissions', verification / 'HOLD']
    for path in holds:
        path.write_text('remote reboot hold\n')
    original = {p: p.read_bytes() for p in holds}
    assert planning(f, [stats], policy='all')[0]['decision'] == 'local'
    queue_files(f.root, [stats])
    assert apply_placement(f, ['stats'])['moved'] == ['stats']
    assert {p: p.read_bytes() for p in holds} == original


def test_realworld_migration_switch_rechecks_mirrored_inputs(mirrored, monkeypatch):
    f = mirrored
    # Migration uses the production mirror location relative to its sandbox root.
    f.mirror = f.runs / 'realworld-inputs'
    rw.fetch([STORE, SPLIT, GATE], f.client, mirror=f.mirror)
    f.manifest = f.mirror / 'MANIFEST.json'
    _, pre, _ = pretraining_rows(f)
    dependency(f, 'stats', 'workspace_stats.json')
    checkpoint = f.runs / 'pre/checkpoints/latest.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'verified paused checkpoint')
    queue, registry = queue_files(f.root, [pre])
    for event in [
        {'event': 'launched', 'id': 'stats', 'host': 'local', 'attempt': 1, 'gpu': LOCAL_GPU, 'pid': 2**30},
        {'event': 'exited', 'id': 'stats', 'host': 'local', 'attempt': 1, 'code': 0},
        {'event': 'launched', 'id': 'pre', 'host': 'remote', 'attempt': 1, 'gpu': REMOTE_GPU, 'commit': COMMIT,
         'job_sha256': jobs.job_fingerprint(pre)},
        {'event': 'checkpoint_paused', 'id': 'pre', 'host': 'remote', 'attempt': 1, 'intentional_stop': True,
         'checkpoint_step': 100, 'checkpoint': 'runs/pre/checkpoints/latest.pt',
         'checkpoint_sha256': sha(checkpoint.read_bytes()), 'reason': 'explicit migration'},
        {'event': 'exited', 'id': 'pre', 'host': 'remote', 'attempt': 1, 'code': 143},
    ]:
        jobs.record(event, registry)
    paths = migration.Paths(root=f.root, queue=queue, registry=registry, ownership=f.root / 'ownership',
                            admission_lock=f.root / 'admission.lock', pool_manifest=f.root / 'migration-pool.json')
    record = migration.begin('pre', 'local', 'explicit pool migration', paths)[0]
    # Model verify's recorded transfer manifest; checkpoint component verification has separate end-to-end tests.
    record['manifest'] = {'sha256': 'a' * 64, 'files': {
        'runs/pre/checkpoints/latest.pt': {'bytes': checkpoint.stat().st_size, 'sha256': sha(checkpoint.read_bytes())}}}
    host_roles.write_ownership(record, paths.ownership)
    original = f.manifest.read_bytes()
    f.manifest.unlink()
    before = queue.read_bytes(), registry.read_bytes()
    with pytest.raises(migration.MigrationError, match='local migration inputs not verified'):
        migration.switch('pre', paths)
    assert (queue.read_bytes(), registry.read_bytes()) == before
    f.manifest.write_bytes(original)
    scope = lambda *args: {'identical': True, 'files': SCOPE['files'], 'third_party': {}}  # noqa: E731
    with monkeypatch.context() as changed:
        def changed_config(*args, **kwargs):
            raise rw.LocalInputError('named config differs from launched release')
        changed.setattr(rw, 'placement_scope', changed_config)
        with pytest.raises(migration.MigrationError, match='argument/code scope differs from launched release'):
            migration.switch('pre', paths, scope_fn=scope)
    assert (queue.read_bytes(), registry.read_bytes()) == before
    switched = migration.switch('pre', paths, scope_fn=scope, identity_fn=lambda _: ('c' * 40, 'local-venv'))
    assert switched[0]['state'] == 'switched' and jobs.read_registry(registry)['pre']['host'] == 'local'
    # Launch accepts migration ownership as an explicit placement and still verifies its recorded scope/inputs.
    local_pre = jobs.load_queue(queue)['jobs'][0]
    assert rw.localize(local_pre, jobs.read_registry(registry), roles=ROLES, root=f.root, mirror=f.mirror,
                       source_identity=IDENTITY, ownership=host_roles.read_ownership(paths.ownership))[1]['inputs']


@pytest.mark.parametrize('which', ['stats', 'pre', 'export'])
def test_scheduler_localizes_every_resolvable_remote_resident_script(mirrored, monkeypatch, which):
    f = mirrored
    rows = dict(zip(('stats', 'pre', 'export'), pretraining_rows(f)))
    job = {**rows[which], 'host': 'local', 'pin_host': 'local', 'gpu': 'any'}
    events = []
    if which == 'pre':
        dependency(f, 'stats', 'workspace_stats.json')
    if which == 'export':
        dependency(f, 'pre', 'config.yaml')
        dependency(f, 'pre', 'checkpoints/step_0200000.pt')
    for dep in job['deps']:
        events += [{'event': 'launched', 'id': dep, 'host': 'local', 'attempt': 1, 'gpu': LOCAL_GPU, 'pid': 2**30},
                   {'event': 'exited', 'id': dep, 'host': 'local', 'attempt': 1, 'code': 0}]
    host_roles.write_ownership(placed(job['id']), f.root / 'ownership')
    original_localize = rw.localize
    monkeypatch.setattr(rw, 'localize', lambda *args, **kwargs: original_localize(*args, **kwargs, root=f.root))
    launches, _, launched = tick(f, monkeypatch, [job], events)
    assert launches == [(job['id'], {'args': [a.replace(PREFIX, str(f.mirror)) for a in job['args']],
                                     'cpu_threads': 8})]  # the remote container's thread quota
    assert launched[-1]['inputs']['execution_scope'] == SCOPE


def test_needs_cli_inventories_each_resolvable_row(mirrored, monkeypatch, capsys):
    import sys

    rows = pretraining_rows(mirrored)
    monkeypatch.setattr(jobs, 'load_queue', lambda: {'jobs': list(rows)})
    monkeypatch.setattr(jobs, 'read_registry', lambda: {})
    monkeypatch.setattr(host_roles, 'read_ownership', lambda: {})
    monkeypatch.setattr(sys, 'argv', ['realworld_local.py', 'needs', 'stats', 'pre', 'export'])
    rw.main()
    inventory = json.loads(capsys.readouterr().out)
    assert inventory['stats']['mirror'] == sorted([STORE, SPLIT])
    assert inventory['pre']['dependency_files'] == [['stats', 'workspace_stats.json']]
    assert inventory['export']['dependency_files'] == [['pre', '.'], ['pre', 'checkpoints/step_0200000.pt'],
                                                      ['pre', 'config.yaml']]


def test_fql_retains_fail_closed_mapping_of_every_remote_path_argument(mirrored):
    # Keep the original FQL launch guard even for an unusual output-root: no unchecked remote-prefix argument
    # is rewritten into the mirror. The new scripts additionally require outputs under their own run id.
    job = row(args=['--output-root', PREFIX + '/unmirrored-output'], deps=[])
    with pytest.raises(rw.LocalInputError, match='input not mirrored: unmirrored-output'):
        localize(mirrored, job)


# ---------------------------------------------------------------------------------------------------- unified pool,
# Codex review 1 fixes (Claude): effective-configuration check, absolute local inputs, dispatch role stamping
def test_unused_set_spelling_cannot_stand_in_for_a_consumed_remote_default(mirrored):
    """The pretraining config's own data.store default is a remote path: dropping the consumed override and adding an
    unused --set that names the same remote file must not pass (the effective configuration still names a remote
    path)."""
    f = mirrored
    _, pre, _ = pretraining_rows(f)
    dependency(f, 'stats', 'workspace_stats.json')
    config = f.root / 'configs/realworld/ours/task.yaml'
    config.write_text('data: {regime: realworld, store: ' + PREFIX + '/' + STORE + '}\nmodel: {}\n')
    store = next(a for a in pre['args'] if a.startswith('data.store='))
    localize(f, pre)  # the consumed override replaces the remote default: accepted
    edited = dict(pre, args=[('audit.unused=' + store.split('=', 1)[1]) if a == store else a for a in pre['args']])
    with pytest.raises(rw.LocalInputError, match='still names a remote path'):
        localize(f, edited)


@pytest.mark.parametrize('which,edit', [
    ('stats', lambda a: [x if not x.endswith('split_seed0.json') else '/home/ws/elsewhere/split_seed0.json' for x in a]),
    ('pre', lambda a: a + ['data.extra_file=/home/ws/ws/splatter4d/runs/other/file.json']),
    ('pre', lambda a: [x if x != 'configs/realworld/ours/task.yaml' else '/tmp/task.yaml' for x in a]),
])
def test_absolute_local_inputs_fail_closed(mirrored, which, edit):
    f = mirrored
    rows = dict(zip(('stats', 'pre', 'export'), pretraining_rows(f)))
    dependency(f, 'stats', 'workspace_stats.json')
    job = rows[which]
    job = dict(job, args=edit(list(job['args'])))
    with pytest.raises(rw.LocalInputError):
        localize(f, job)


def test_dispatch_copy_keeps_a_dependency_derived_class(mirrored):
    """An export classified only through its real-world producer stays remote-resident in a one-row dispatch queue
    once stamped (host_roles.with_role, used by both admission drivers); without the stamp it would be domain-free."""
    f = mirrored
    _, pre, export = pretraining_rows(f)
    bare = {k: v for k, v in export.items() if k != 'role'}
    by_id = {pre['id']: pre, bare['id']: bare}
    assert host_roles.job_class(bare, by_id).cls == 'realworld_pretrain'
    assert host_roles.job_class(bare, {bare['id']: bare}).cls == 'unknown'  # the dispatch queue alone
    stamped = host_roles.with_role(bare, by_id)
    assert stamped['role'] == 'realworld_pretrain' and host_roles.remote_resident(stamped, ROLES, {bare['id']: stamped})
    assert jobs.job_fingerprint(stamped) == jobs.job_fingerprint(bare)
    assert host_roles.with_role(export, by_id) == export


def test_local_driver_stamps_roles_and_remote_driver_honours_queue_hold_and_local_pins():
    local = (rw.REPO / 'ops/campaign/baseline_local_driver.py').read_text()
    assert ("dispatched = [{**host_roles.with_role(j, by_queue), 'gpu': fair['assigned'][j['id']]} "
            "for j in admitted]") in local
    assert "'jobs': json.loads(json.dumps(dispatched))" in local
    assert 'if _paused_now(job' in local  # an intentionally paused Stage 2 row never forces the baselines to yield
    remote = (rw.REPO / 'ops/campaign/remote_campaign_driver.py').read_text()
    assert "or queue['hosts'].get('remote', {}).get('hold', False) is not False)" in remote
    assert "candidates = [j for j in candidates if j.get('release_source') != 'local']" in remote
    assert "if paused(job['id']) or job.get('release_source') == 'local':" in remote
    for text in (local,):
        assert 'remote-hold-admissions' not in text and 'verification/HOLD' not in text


def test_remote_driver_keeps_previous_role_releases_for_retries_only():
    source = (rw.REPO / 'ops/campaign/remote_campaign_driver.py').read_text()
    block = source[source.index("for _previous in ROLE_RELEASE.get('previous', []):"):]
    block = block[:block.index('\nROLE_READY')]
    names = {'ROLE_RELEASE': {'commit': 'b' * 40, 'image': 'i', 'previous': [{'commit': 'a' * 40, 'image': 'i'}]},
             'RELEASES': {('b' * 40, 'i'): 'current'}}
    exec(block, names)  # the mapping block only (no driver pass is run)
    assert names['RELEASES'] == {('b' * 40, 'i'): 'current',
                                 ('a' * 40, 'i'): f"runs/remote/acceptance/{'a' * 7}-cc/manifest.json"}
    release = source[source.index('def job_release(job):'):source.index('\n\n\n', source.index('def job_release(job):'))]
    assert release.index("if previous.get('commit'):") < release.index("if kind(job['id']) == 'role':")


@pytest.mark.parametrize('condition,expected', [('ours', 3), ('dynamo', 3), ('e2e_cnn', 8)])
def test_measured_realworld_fql_ram_is_shared_by_admission_and_running_growth(condition, expected):
    import ast
    from ops.jobs import scheduler
    from ops.jobs.footprints import ram_gb

    row = {'script': 'scripts/train_realworld_fql.py', 'args': ['--set', 'condition=' + condition],
           'fql_condition': condition, 'ram_gb': 64}
    source = (Path(__file__).resolve().parents[1] / 'ops/campaign/baseline_local_driver.py').read_text()
    tree = ast.parse(source)
    names = {}
    for name in ('footprint', 'active_footprint'):
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<driver-functions>', 'exec'), names)
    names['by_queue'] = {'fql': row}
    assert names['footprint'](row) == expected * 2**30
    assert names['active_footprint']('fql', {'ram_gb': 64}) == expected * 2**30
    assert scheduler.reservation({'ram_gb': 64}, row, 'ram_gb') == expected
    assert ram_gb(row) == expected
    assert ram_gb({'script': 'scripts/train_snerl.py', 'ram_gb': 64}) == 64


# ---- baseline pretraining + local dataset construction (pick-and-place image family, 2026-11-13; user directive
# 2026-11-12 local capacity)
OUTCOMES = "learning_inputs/cal/task/outcomes.json"
PLAIN = "learning_inputs/cal/plain/store.yaml"  # what make_learning_inputs writes: kind + paths only


def plain_store(f):
    if not (f.mirror / PLAIN).exists():
        (f.remote / PLAIN).parent.mkdir(parents=True, exist_ok=True)
        (f.remote / PLAIN).write_bytes(("kind: compact\npaths:\n- " + PREFIX + "/" + EPISODE + "\n").encode())
        rw.fetch([PLAIN], f.client, mirror=f.mirror)


def baseline_rows(f, *, source="realworld"):
    plain_store(f)
    for b in ("dynamo", "reviwo"):
        config = f.root / f"configs/realworld/baselines/{b}/task.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("dataset: {hdf5_path: " + PREFIX + "/learning_inputs/old/store.yaml, batch_size: 4}\n"
                          "data_interface: {source: " + source + ", split_manifest: " + PREFIX + "/old/split.json, "
                          "gate_report: " + PREFIX + "/old/gate.json, frame_spacing: 1}\ntrain: {seed: 0}\n")
    sets = ["--set", "dataset.hdf5_path=" + PREFIX + "/" + PLAIN, "data_interface.split_manifest=" + PREFIX + "/" + SPLIT,
            "data_interface.gate_report=" + PREFIX + "/" + GATE]
    dynamo = row("dyn-pre", role="realworld_pretrain", script="scripts/train_dynamo.py", deps=[], args=[
        "--config", "configs/realworld/baselines/dynamo/task.yaml", "--name", "dyn-pre", "--output-root", "runs",
        "--resume", "auto", *sets])
    reviwo = row("rev-pre", role="realworld_pretrain", script="scripts/train_reviwo.py", deps=[], args=[
        "--config", "configs/realworld/baselines/reviwo/task.yaml", "--name", "rev-pre", "--output-root", "runs", *sets])
    return dynamo, reviwo


def dataset_row(f, condition="dynamo", export=True):
    plain_store(f)
    if not (f.mirror / OUTCOMES).exists():
        (f.remote / OUTCOMES).write_bytes(b'{"schema": "rw-outcomes/1"}\n')
        rw.fetch([OUTCOMES], f.client, mirror=f.mirror)
    args = ["build", "--store", PREFIX + "/" + PLAIN, "--condition", condition, "--out", "runs/data-row/dataset.h5",
            "--gate-report", PREFIX + "/" + GATE, "--outcomes", PREFIX + "/" + OUTCOMES]
    args += ["--export", "runs/dyn-pre/encoder.pt", "--device", "cuda"] if export else ["--device", "cpu"]
    return row("data-row", role="realworld_fql", script=rw.FQL_DATASET, deps=["dyn-pre"], args=args)


def test_baseline_pretraining_resolvers_map_inputs_and_pin_outputs(mirrored):
    f = mirrored
    for job in baseline_rows(f):
        assert rw.applies(job, ROLES) and host_roles.role_hosts(job, ROLES) == ("local", "remote")
        effective, record = localize(f, job)
        assert effective == [a.replace(PREFIX, str(f.mirror)) for a in job["args"]]
        assert set(record["input_closure"]) >= {PLAIN, EPISODE + "/episode.h5"} and "raw_map" not in record
        assert rw.needed_inputs(job) == {"mirror": sorted([PLAIN, SPLIT, GATE]), "dependency_files": []}


@pytest.mark.parametrize("edit,reason", [
    ("output_root", "--output-root runs --name"), ("name", "--output-root runs --name"),
    ("reviwo_resume", "unsupported local input options"), ("abbreviation", "invalid local resolver"),
    ("init_export", "unsupported local input options"), ("unset_override", "not plain checked arguments"),
    ("config_outside", "configs/realworld/baselines"), ("source", "data_interface.source realworld"),
    ("relative_store", "dataset.hdf5_path must name a mirrored remote input"),
    ("inline_store", "dataset.hdf5_path must name a mirrored remote input"),
    ("relative_split", "data_interface.split_manifest must name a mirrored remote input"),
    ("pretrained", "no external pretrained weights"), ("descriptor_extra", "plain compact store.yaml"),
])
def test_baseline_pretraining_resolvers_fail_closed(mirrored, edit, reason):
    f = mirrored
    dynamo, reviwo = baseline_rows(f, source="metaworld" if edit == "source" else "realworld")
    job = reviwo if edit == "reviwo_resume" else dynamo
    args = job["args"]
    if edit == "output_root":
        args[args.index("--output-root") + 1] = "runs/pretrain"
    elif edit == "name":
        args[args.index("--name") + 1] = "other"
    elif edit == "reviwo_resume":
        args += ["--resume", "auto"]
    elif edit == "abbreviation":
        args[0] = "--conf"
    elif edit == "init_export":
        args += ["--init-export-only"]
    elif edit == "unset_override":  # the config's own remote gate path would be read unverified
        args.remove("data_interface.gate_report=" + PREFIX + "/" + GATE)
    elif edit == "config_outside":
        (f.root / "configs/realworld/fql/b.yaml").write_text((f.root / args[1]).read_text())
        args[1] = "configs/realworld/fql/b.yaml"
    elif edit == "relative_store":
        args[args.index("dataset.hdf5_path=" + PREFIX + "/" + PLAIN)] = "dataset.hdf5_path=data/store.yaml"
    elif edit == "inline_store":
        args[args.index("dataset.hdf5_path=" + PREFIX + "/" + PLAIN)] = "dataset.hdf5_path={kind: compact, paths: []}"
    elif edit == "relative_split":
        args[args.index("data_interface.split_manifest=" + PREFIX + "/" + SPLIT)] = "data_interface.split_manifest=s.json"
    elif edit == "pretrained":
        args += ["encoder.pretrained=true"]
    elif edit == "descriptor_extra":  # e.g. native-depth overlays: fields the resolvers do not inventory
        args[args.index("dataset.hdf5_path=" + PREFIX + "/" + PLAIN)] = "dataset.hdf5_path=" + PREFIX + "/" + STORE
    if edit == "config_outside":
        assert not rw.has_resolver(job)
        with pytest.raises(rw.LocalInputError, match="no local input resolver|no explicit"):
            localize(f, job)
        return
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(f, job)


def test_local_dataset_construction_maps_inputs_dependency_and_raw_root(mirrored, monkeypatch, tmp_path):
    f = mirrored
    raw_root = tmp_path / "raw-originals"
    raw_root.mkdir()
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", raw_root)
    job = dataset_row(f)
    assert rw.applies(job, ROLES) and host_roles.role_hosts(job, ROLES) == ("local", "remote")
    with pytest.raises(rw.LocalInputError, match="dependency must be done"):
        localize(f, job)
    finished = tmp_path / "finished"
    dependency(f, "dyn-pre", "encoder.pt", host="remote")
    with pytest.raises(rw.LocalInputError, match="needs a LOCAL encoder producer"):
        localize(f, job, finished_dir=finished)
    f.state["dyn-pre"]["host"] = "local"
    with pytest.raises(rw.LocalInputError, match="without a closeout record"):
        localize(f, job, finished_dir=finished)
    rw.closeout("dyn-pre", f.state, {"id": "dyn-pre", "required_results": ["encoder.pt"]}, finished, runs=f.runs)
    (f.runs / "dyn-pre/encoder.pt").write_text("replaced after closeout")
    with pytest.raises(rw.LocalInputError, match="differs from its closeout record"):
        localize(f, job, finished_dir=finished)
    (f.runs / "dyn-pre/encoder.pt").write_text("verified output")
    effective, record = localize(f, job, finished_dir=finished)
    assert effective == [a.replace(PREFIX, str(f.mirror)) for a in job["args"]] + [
        "--raw-root", f"{rw.REMOTE_RAW_ROOT}={raw_root}"]
    assert record["raw_map"] == {"from": rw.REMOTE_RAW_ROOT, "to": str(raw_root)}
    assert record["dependency_inputs"] == {"runs/dyn-pre/encoder.pt": sha(b"verified output")}
    assert set(record["inputs"]) == {PREFIX + "/" + PLAIN, PREFIX + "/" + GATE, PREFIX + "/" + OUTCOMES}
    e2e = dataset_row(f, "e2e_cnn", export=False)
    assert localize(f, e2e, finished_dir=finished)[0][-2:] == ["--raw-root", f"{rw.REMOTE_RAW_ROOT}={raw_root}"]
    link = tmp_path / "raw-link"
    link.symlink_to(raw_root)
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", link)
    with pytest.raises(rw.LocalInputError, match="missing or a symlink"):
        localize(f, job, finished_dir=finished)
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", raw_root)
    raw_root.rmdir()
    with pytest.raises(rw.LocalInputError, match="missing or a symlink"):
        localize(f, job, finished_dir=finished)


@pytest.mark.parametrize("edit,reason", [
    ("smoke", "invalid local resolver|unsupported local input options"),
    ("episodes", "invalid local resolver|unsupported local input options"),
    ("assume", "invalid local resolver|unsupported local input options"),
    ("raw_root", "invalid local resolver|unsupported local input options"),
    ("export_other", "runs/<declared dependency>/encoder.pt"), ("out", "runs/<own id>/"),
    ("out_name", "runs/<own id>/dataset.h5"), ("condition", "supports"),
    ("store_local", "mirrored compact store.yaml"), ("no_outcomes", "--outcomes must name a mirrored remote input"),
    ("descriptor_extra", "plain compact store.yaml"),
    ("native_mirrored_store", "runs/<declared dependency>/store-native.yaml"),
])
def test_local_dataset_construction_fails_closed(mirrored, monkeypatch, tmp_path, edit, reason):
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    dependency(f, "dyn-pre", "encoder.pt")
    job = dataset_row(f)
    args = job["args"]
    if edit == "smoke":
        args += ["--smoke"]
    elif edit == "episodes":
        args += ["--episodes", "a"]
    elif edit == "assume":
        args += ["--assume-final-success"]
    elif edit == "raw_root":
        args += ["--raw-root", "/a=/b"]
    elif edit == "export_other":
        args[args.index("--export") + 1] = "runs/other/encoder.pt"
    elif edit == "out":
        args[args.index("--out") + 1] = "runs/elsewhere/dataset.h5"
    elif edit == "out_name":
        args[args.index("--out") + 1] = "runs/data-row/other.h5"
    elif edit == "condition":
        args[args.index("--condition") + 1] = "unsupported"  # ours now has an accepted local resolver
    elif edit == "descriptor_extra":
        args[args.index("--store") + 1] = PREFIX + "/" + STORE
    elif edit == "store_local":
        args[args.index("--store") + 1] = "runs/x/store.yaml"
    elif edit == "native_mirrored_store":  # a 3D-MVP dataset reads the native descriptor of its native dependency
        args[args.index("--condition") + 1] = "mvp3d_single"
    else:
        i = args.index("--outcomes")
        del args[i:i + 2]
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(f, job)


def _dataset_script():
    import ast

    source = ast.parse((rw.REPO / "scripts/realworld_fql_dataset.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "mapped_raw_recordings")
    names = {"Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<dataset helper>", "exec"), names)
    return SimpleNamespace(mapped_raw_recordings=names["mapped_raw_recordings"])


def test_dataset_builder_raw_root_mapping_verifies_every_file(tmp_path):
    script = _dataset_script()
    local = tmp_path / "local"
    (local / "task").mkdir(parents=True)
    good = local / "task/episode_000001.h5"
    good.write_bytes(b"raw recording")
    want = sha(b"raw recording")
    store = SimpleNamespace(raw_recording=lambda ep: f"/home/compu/kaist/sunho/data/real_world/task/{ep}.h5",
                            raw_sha256=lambda ep: want)
    mapping = f"/home/compu/kaist/sunho/data/real_world={local}"
    assert script.mapped_raw_recordings(store, ["episode_000001"], mapping) == {"episode_000001": str(good)}
    with pytest.raises(SystemExit, match="sha256"):
        script.mapped_raw_recordings(SimpleNamespace(raw_recording=store.raw_recording, raw_sha256=lambda ep: "0" * 64),
                                     ["episode_000001"], mapping)
    with pytest.raises(SystemExit, match="sha256"):  # no recorded identity: never read unverified
        script.mapped_raw_recordings(SimpleNamespace(raw_recording=store.raw_recording, raw_sha256=lambda ep: ""),
                                     ["episode_000001"], mapping)
    with pytest.raises(SystemExit, match="not below"):
        script.mapped_raw_recordings(store, ["episode_000001"], f"/other/root={local}")
    with pytest.raises(SystemExit, match="not a regular file"):
        script.mapped_raw_recordings(store, ["episode_000002"], mapping)
    (local / "task/episode_000003.h5").symlink_to(good)
    with pytest.raises(SystemExit, match="not a regular file"):
        script.mapped_raw_recordings(store, ["episode_000003"], mapping)
    for bad in ("no-separator", "relative=/x", "/x=relative"):
        with pytest.raises(SystemExit, match="RECORDED_ROOT=LOCAL_ROOT"):
            script.mapped_raw_recordings(store, ["episode_000001"], bad)
    (tmp_path / "root-link").symlink_to(local)  # a symlinked root could redirect the whole mapping (Codex review)
    with pytest.raises(SystemExit, match="without symlinks"):
        script.mapped_raw_recordings(store, ["episode_000001"],
                                     f"/home/compu/kaist/sunho/data/real_world={tmp_path}/root-link")


def test_e2e_dataset_waits_for_closeouts_of_its_ordering_dependencies(mirrored, monkeypatch, tmp_path):
    """Codex review: the e2e dataset reads no baseline file, but both baselines must have passed closeout."""
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    finished = tmp_path / "finished"
    e2e = {**dataset_row(f, "e2e_cnn", export=False), "deps": ["dyn-pre", "rev-pre"]}
    for dep in ("dyn-pre", "rev-pre"):
        dependency(f, dep, "encoder.pt")  # local, done (exit 0) - no closeout yet
    with pytest.raises(rw.LocalInputError, match="without a closeout record: dyn-pre"):
        localize(f, e2e, finished_dir=finished)
    rw.closeout("dyn-pre", f.state, {"id": "dyn-pre", "required_results": ["encoder.pt"]}, finished, runs=f.runs)
    with pytest.raises(rw.LocalInputError, match="without a closeout record: rev-pre"):
        localize(f, e2e, finished_dir=finished)
    rw.closeout("rev-pre", f.state, {"id": "rev-pre", "required_results": ["encoder.pt"]}, finished, runs=f.runs)
    assert localize(f, e2e, finished_dir=finished)[1]["dependency_inputs"] == {}
    (finished / "rev-pre.json").write_text('{"job_id": "rev-pre", "files": {}}')
    with pytest.raises(rw.LocalInputError, match="invalid closeout record"):
        localize(f, e2e, finished_dir=finished)


# ---- geometry family (pick-and-place, coordinator 2026-11-13): SinCro / SNeRL / 3D-MVP pretraining, the native-depth
# stage with the sha-verified raw mapping, the native descriptor's overlay inventory, fqldata sincro / snerl / mvp3d_*
RIG = "configs/cal/rig/rig.json"
RIG_REF = "configs/cal/rig/rig.reference.npz"


def geometry_rows(f):
    plain_store(f)
    for b in ("sincro", "snerl", "mvp3d"):
        config = f.root / f"configs/realworld/baselines/{b}/task.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("dataset: {hdf5_path: " + PREFIX + "/learning_inputs/old/store.yaml, batch_size: 4}\n"
                          "data_interface: {source: realworld, split_manifest: " + PREFIX + "/old/split.json, "
                          "gate_report: " + PREFIX + "/old/gate.json}\ntrain: {seed: 0}\n")
    tail = ["data_interface.split_manifest=" + PREFIX + "/" + SPLIT, "data_interface.gate_report=" + PREFIX + "/" + GATE]
    plain = ["--set", "dataset.hdf5_path=" + PREFIX + "/" + PLAIN, *tail]

    def pre(name, b, extra=(), sets=plain, deps=()):
        return row(name, role="realworld_pretrain", script=f"scripts/train_{b}.py", deps=list(deps), args=[
            "--config", f"configs/realworld/baselines/{b}/task.yaml", "--name", name, "--output-root", "runs",
            *extra, *sets])
    return {"sincro": pre("sin-pre", "sincro"), "snerl": pre("sne-pre", "snerl", ("--resume", "auto")),
            "mvp3d": pre("mvp-pre", "mvp3d", ("--resume", "auto"),
                         ["--set", "dataset.hdf5_path=runs/nd/store-native.yaml", *tail], deps=("nd",))}


def native_row(f):
    plain_store(f)
    for name, payload in ((RIG, b'{"format": "rig"}\n'), (RIG_REF, b"reference arrays")):
        if not (f.mirror / name).exists():
            (f.remote / name).parent.mkdir(parents=True, exist_ok=True)
            (f.remote / name).write_bytes(payload)
            rw.fetch([name], f.client, mirror=f.mirror)
    return row("nd", role="realworld_fql", script=rw.NATIVE_DEPTH, deps=[], required_results=["store-native.yaml"],
               args=["build", "--store", PREFIX + "/" + PLAIN, "--rig", PREFIX + "/" + RIG, "--out-dir",
                     "runs/nd/overlays", "--out-store", "runs/nd/store-native.yaml", "--workers", "16"])


def native_outputs(f, *, count=1, episode=None):
    """What the builder publishes locally: overlays under runs/nd/overlays, a descriptor naming the mirror episodes."""
    folder = f.runs / "nd/overlays"
    folder.mkdir(parents=True, exist_ok=True)
    overlays = []
    for i in range(count):
        path = (folder / f"{i:024x}.h5").resolve()
        path.write_bytes(b"overlay %d" % i)
        overlays.append(str(path))
    spec = {"kind": "compact", "paths": [episode or str(f.mirror / EPISODE)] * count, "depth_source": "native_realsense",
            "depth_overlays": overlays, "rig_document_sha256": "d" * 64}
    (f.runs / "nd/store-native.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    f.state["nd"] = {"host": "local", "status": "done"}
    return spec


def test_geometry_pretraining_resolvers_map_inputs_and_resume_semantics(mirrored):
    f = mirrored
    rows = geometry_rows(f)
    for b in ("sincro", "snerl"):
        job = rows[b]
        assert rw.applies(job, ROLES) and host_roles.role_hosts(job, ROLES) == ("local", "remote")
        effective, record = localize(f, job)
        assert effective == [a.replace(PREFIX, str(f.mirror)) for a in job["args"]]
        assert rw.needed_inputs(job) == {"mirror": sorted([PLAIN, SPLIT, GATE]), "dependency_files": []}
    rows["snerl"]["args"][rows["snerl"]["args"].index("auto")] = "never"  # train_snerl.py: auto | never
    localize(f, rows["snerl"])
    for job, extra, reason in ((rows["sincro"], ["--resume", "auto"], "unsupported local input options"),
                               (rows["snerl"], ["--export-init-only"], "unsupported local input options")):
        with pytest.raises(rw.LocalInputError, match=reason):
            localize(f, {**job, "args": job["args"] + extra})
    bad = copy.deepcopy(rows["snerl"])
    bad["args"][bad["args"].index("never")] = "always"
    with pytest.raises(rw.LocalInputError, match="invalid local resolver"):
        localize(f, bad)
    assert rw.RESUME_CHOICES["scripts/train_dynamo.py"] == ("auto",)  # the image family's dynamo is unchanged


def test_native_depth_stage_resolver_maps_store_rig_reference_and_raw_root(mirrored, monkeypatch, tmp_path):
    f = mirrored
    raw_root = tmp_path / "raw-originals"
    raw_root.mkdir()
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", raw_root)
    job = native_row(f)
    assert rw.applies(job, ROLES)
    assert rw.needed_inputs(job) == {"mirror": sorted([PLAIN, RIG, RIG_REF]), "dependency_files": []}
    effective, record = localize(f, job)
    assert effective == [a.replace(PREFIX, str(f.mirror)) for a in job["args"]] + [
        "--raw-root", f"{rw.REMOTE_RAW_ROOT}={raw_root}"]
    assert record["raw_map"] == {"from": rw.REMOTE_RAW_ROOT, "to": str(raw_root)}
    assert set(record["inputs"]) == {PREFIX + "/" + PLAIN, PREFIX + "/" + RIG, PREFIX + "/" + RIG_REF}


@pytest.mark.parametrize("edit,reason", [
    ("episodes", "unsupported local input options|invalid local resolver"),
    ("raw_root", "unsupported local input options|invalid local resolver"),
    ("out_dir", "runs/<own id>/overlays"), ("out_store", "runs/<own id>/store-native.yaml"),
    ("store", "mirrored compact store.yaml"), ("rig", "mirrored rig document"), ("workers", "1..64"),
    ("descriptor_extra", "plain compact store.yaml"),
])
def test_native_depth_stage_resolver_fails_closed(mirrored, monkeypatch, tmp_path, edit, reason):
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    job = native_row(f)
    args = job["args"]
    if edit == "episodes":
        args += ["--episodes", "task/episode"]
    elif edit == "raw_root":
        args += ["--raw-root", "/a=/b"]
    elif edit == "out_dir":
        args[args.index("--out-dir") + 1] = "runs/nd/elsewhere"
    elif edit == "out_store":
        args[args.index("--out-store") + 1] = "runs/nd/store.yaml"
    elif edit == "store":
        args[args.index("--store") + 1] = "runs/x/store.yaml"
    elif edit == "rig":
        args[args.index("--rig") + 1] = PREFIX + "/" + RIG_REF
    elif edit == "workers":
        args[args.index("--workers") + 1] = "0"
    else:
        args[args.index("--store") + 1] = PREFIX + "/" + STORE
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(f, job)


def test_native_closeout_records_every_overlay(mirrored, tmp_path):
    f = mirrored
    job = native_row(f)
    spec = native_outputs(f, count=2)
    finished = tmp_path / "finished"
    record = rw.closeout("nd", f.state, job, finished, runs=f.runs)
    assert set(record["files"]) == {"store-native.yaml"} | {f"overlays/{Path(p).name}" for p in spec["depth_overlays"]}
    with pytest.raises(rw.LocalInputError, match="required result"):
        rw.closeout("nd2", {"nd2": {"host": "local", "status": "done"}}, {**job, "id": "nd2", "required_results": []},
                    finished, runs=f.runs)


def test_mvp3d_pretraining_and_datasets_verify_the_native_descriptor(mirrored, monkeypatch, tmp_path):
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    finished = tmp_path / "finished"
    pre = geometry_rows(f)["mvp3d"]
    with pytest.raises(rw.LocalInputError, match="dependency must be done"):
        localize(f, pre, finished_dir=finished)
    spec = native_outputs(f)
    with pytest.raises(rw.LocalInputError, match="without a closeout record"):
        localize(f, pre, finished_dir=finished)
    rw.closeout("nd", f.state, native_row(f), finished, runs=f.runs)
    effective, record = localize(f, pre, finished_dir=finished)
    overlay = Path(spec["depth_overlays"][0])
    assert set(record["dependency_inputs"]) == {"runs/nd/store-native.yaml", f"runs/nd/overlays/{overlay.name}"}
    assert {EPISODE + "/episode.h5", EPISODE + "/_COMPLETE.json"} <= set(record["input_closure"])
    for condition in ("mvp3d_single", "mvp3d_multiview"):
        data = {**dataset_row(f, condition), "deps": ["mvp-pre", "nd"]}
        data["args"][data["args"].index("--store") + 1] = "runs/nd/store-native.yaml"
        data["args"][data["args"].index("--export") + 1] = "runs/mvp-pre/encoder.pt"
        dependency(f, "mvp-pre", "encoder.pt")
        rw.closeout("mvp-pre", f.state, {"id": "mvp-pre", "required_results": ["encoder.pt"]}, finished, runs=f.runs)
        assert f"runs/nd/overlays/{overlay.name}" in localize(f, data, finished_dir=finished)[1]["dependency_inputs"]
        undeclared = {**data, "deps": ["mvp-pre"]}
        with pytest.raises(rw.LocalInputError, match="declared dependency"):
            localize(f, undeclared, finished_dir=finished)
    overlay.write_bytes(b"tampered")
    with pytest.raises(rw.LocalInputError, match="overlay differs from its closeout record"):
        localize(f, pre, finished_dir=finished)


@pytest.mark.parametrize("edit,reason", [
    ("remote_producer", "LOCAL native depth producer"), ("extra_key", "plain native descriptor"),
    ("not_native", "plain native descriptor"), ("episode_outside", "outside the verified mirror"),
    ("overlay_outside", "outside runs/nd/overlays"), ("plain_store_path", "runs/<declared dependency>/store-native"),
])
def test_native_descriptor_consumers_fail_closed(mirrored, monkeypatch, tmp_path, edit, reason):
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    finished = tmp_path / "finished"
    pre = geometry_rows(f)["mvp3d"]
    spec = native_outputs(f, episode=str(tmp_path / "elsewhere") if edit == "episode_outside" else None)
    if edit == "extra_key":
        spec["notes"] = "x"
    elif edit == "not_native":
        spec["depth_source"] = "foundation_stereo"
    elif edit == "overlay_outside":
        other = (tmp_path / "other.h5").resolve()
        other.write_bytes(b"x")
        spec["depth_overlays"] = [str(other)]
    if edit in ("extra_key", "not_native", "overlay_outside"):
        (f.runs / "nd/store-native.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        f.state["nd"]["status"] = "done"
        digest = sha((f.runs / "nd/store-native.yaml").read_bytes())
        finished.mkdir(parents=True, exist_ok=True)
        (finished / "nd.json").write_text(json.dumps({"job_id": "nd", "files": {"store-native.yaml": digest}}))
    else:
        rw.closeout("nd", f.state, native_row(f), finished, runs=f.runs)
    if edit == "remote_producer":
        f.state["nd"] = {"host": "remote", "status": "done", "sync": {"files": {}}, "remote_verified": {"files": {}}}
    elif edit == "plain_store_path":
        pre["args"][pre["args"].index("dataset.hdf5_path=runs/nd/store-native.yaml")] = "dataset.hdf5_path=runs/nd/s.yaml"
    with pytest.raises(rw.LocalInputError, match=reason):
        localize(f, pre, finished_dir=finished)


def test_geometry_dataset_conditions_sincro_and_snerl(mirrored, monkeypatch, tmp_path):
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    finished = tmp_path / "finished"
    dependency(f, "dyn-pre", "encoder.pt")
    rw.closeout("dyn-pre", f.state, {"id": "dyn-pre", "required_results": ["encoder.pt"]}, finished, runs=f.runs)
    for condition in ("sincro", "snerl"):
        effective, record = localize(f, dataset_row(f, condition), finished_dir=finished)
        assert effective[-2:] == ["--raw-root", f"{rw.REMOTE_RAW_ROOT}={tmp_path}"]
        assert record["dependency_inputs"] == {"runs/dyn-pre/encoder.pt": sha(b"verified output")}
    assert set(rw.LOCAL_DATASET_CONDITIONS) <= set(__import__("s4d.rwlearn.observation",
                                                              fromlist=["CONDITIONS"]).CONDITIONS)


def _snerl_teacher(f, monkeypatch, payload=b"official dino weights", pinned=None):
    """The real SNeRL configuration (dino.pretrained: true) + a synthetic copy of the trainer's pinned constants."""
    cache = f.root / "cache-root"
    weights = cache / "torch/hub/checkpoints/dino_deitsmall8_pretrain.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(payload)
    upstream = f.root / "s4d/baselines/snerl/upstream.py"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text(f'DINO_WEIGHTS_SHA256 = "{pinned or sha(payload)}"\n'
                        'DINO_WEIGHTS_FILE = "dino_deitsmall8_pretrain.pth"\n')
    monkeypatch.setenv("S4D_CACHE_ROOT", str(cache))
    job = geometry_rows(f)["snerl"]
    config = f.root / "configs/realworld/baselines/snerl/real.yaml"
    config.write_bytes((rw.REPO / "configs/realworld/baselines/snerl/pick-and-place.yaml").read_bytes())
    job["args"][job["args"].index("configs/realworld/baselines/snerl/task.yaml")] = "configs/realworld/baselines/snerl/real.yaml"
    return job, weights


def test_snerl_real_config_admits_only_the_pinned_dino_teacher(mirrored, monkeypatch):
    """Codex review21 P1: the real SNeRL config keeps its official DINO teacher; admission verifies the pinned file."""
    f = mirrored
    job, weights = _snerl_teacher(f, monkeypatch)
    effective, record = localize(f, job)
    assert record["pretrained_weights"] == {str(weights): sha(b"official dino weights")}
    with pytest.raises(rw.LocalInputError, match="no dino"):
        localize(f, {**job, "args": job["args"] + ["dino.pretrained=false"]})
    weights.write_bytes(b"other weights")
    with pytest.raises(rw.LocalInputError, match="DINO teacher weights"):
        localize(f, job)
    weights.unlink()
    with pytest.raises(rw.LocalInputError, match="missing mirror file|symlink"):
        localize(f, job)
    for b in ("sincro", "mvp3d"):  # the other geometry baselines still load no pretrained weights
        cfg = yaml.safe_load((rw.REPO / f"configs/realworld/baselines/{b}/pick-and-place.yaml").read_text())
        found, todo = [], [cfg]
        while todo:
            node = todo.pop()
            if isinstance(node, dict):
                found += [v for k, v in node.items() if k == "pretrained"]
                todo += list(node.values())
            elif isinstance(node, list):
                todo += node
        assert all(v is False for v in found), b


@pytest.mark.parametrize("case", ["missing", "altered", "symlink"])
def test_native_depth_rig_reference_is_a_verified_mirror_input(mirrored, monkeypatch, tmp_path, case):
    """Codex review21 P2: the implicit reference arrays are checked like every argument input."""
    f = mirrored
    monkeypatch.setattr(rw, "LOCAL_RAW_ROOT", tmp_path)
    job = native_row(f)
    assert PREFIX + "/" + RIG_REF in localize(f, job)[1]["inputs"]
    path = f.mirror / RIG_REF
    if case == "missing":
        spec = json.loads(f.manifest.read_text())
        del spec["files"][RIG_REF]
        f.manifest.write_text(json.dumps(spec))
    elif case == "altered":
        os.chmod(path, 0o644)
        path.write_bytes(b"other arrays")
    else:
        target = tmp_path / "elsewhere.npz"
        target.write_bytes(path.read_bytes())
        os.chmod(path, 0o644)
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(rw.LocalInputError, match="not mirrored|changed|mismatch|symlink"):
        localize(f, job)


def test_ours_priority_placement_and_blocked_head_cannot_be_backfilled(mirrored):
    f = mirrored
    rows = [row("older-fql", priority=2, ram_gb=1), row("ours-fql", priority=0, ram_gb=1),
            row("pretrain", priority=-1, ram_gb=8)]
    decisions = planning(f, rows)
    assert [d["id"] for d in decisions] == ["pretrain", "ours-fql", "older-fql"]
    assert decisions[0]["decision"] == "local"
    # Six GB cannot fit the eight-GB head; even the tiny lower-priority rows must wait.
    decisions = planning(f, rows, capacity={"ram_free_bytes": 6 * 2**30})
    assert all(d["decision"] == "waits" for d in decisions)
    # A pending (unstarted) lower-priority local row must not reserve ahead of new pretraining.
    placed = row("placed", priority=2, host="local", pin_host="local", gpu="any")
    assert planning(f, [placed, rows[-1]])[0]["decision"] == "local"


def test_local_planner_packs_selected_high_tier_before_pending_lower_tier(mirrored):
    f = mirrored
    rows = [row("pending", host="local", pin_host="local", priority=0, mem_gb=1, ram_gb=1),
            row("pretrain", priority=-1, mem_gb=8, ram_gb=1), row("ours-fql", priority=0, mem_gb=1, ram_gb=1),
            row("old-fql", priority=2, mem_gb=1, ram_gb=1)]
    decisions = rw.plan({"hosts": {"local": {"limits": {LOCAL_GPU: {"max_jobs": 3, "max_mem_gb": 8}}}}, "jobs": rows},
                        f.state, ROLES, paused_fn=lambda _: [], mirror=f.mirror, runs=f.runs, root=f.root,
                        capacity={"ram_free_bytes": 100 * 2**30})
    assert [d["id"] for d in decisions] == ["pretrain", "ours-fql", "old-fql"]
    assert decisions[0]["decision"] == "local" and all(d["decision"] == "waits" for d in decisions[1:])
