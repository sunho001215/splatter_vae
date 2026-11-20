"""CPU-only BH tools contracts, outside the shipped release suite.

Run via run_tests.execute_pytest with --noconftest from this checkout.

Fresh subprocesses isolate the one-task-per-process runtime from pnp default tests.
No GPU guard, dataset, remote host, queue or live admission file is accessed.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "runs/setup/tools"
sys.path[:0] = [str(ROOT), str(TOOLS)]
import bh_isolated_release as isolated  # noqa: E402
import bh_register as registration  # noqa: E402
import bh_release_control as control  # noqa: E402


def child(source):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run([sys.executable, "-B", "-c", source], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


def graph(task):
    return child(f"""
import json
from pathlib import Path
if {task!r} == 'ball-hiding':
    from s4d.rwlearn.ball_hiding import bootstrap
    bootstrap()
from s4d.rwlearn.pnpv2_spec import SPEC, load_spec
from ops.jobs.pnpv2_plan import build_rows
spec=load_spec(SPEC)
scene=dict(lookat=[.4,0,.1],pose_scale=2.5,bounds=[-.5,-.6,-.05,.9,.6,.7])
release=dict(commit='a'*40,image='sha256:'+'b'*64,source='local')
if {task!r} == 'ball-hiding':
    from s4d.rwlearn.bh_training import rows
    output=rows(spec,release,waves=('rgb','geometry'),scene=scene,current=[dict(priority=1337)])
else:
    output=build_rows(spec,release,scene=scene)
print(json.dumps(output))
""")


@pytest.fixture(scope="module")
def graphs():
    return graph("pick-and-place-v2"), graph("ball-hiding")


def test_exact_graph_science_dependencies_and_seeds(graphs):
    pnp, bh = graphs
    assert len(bh) == len(pnp) == len(isolated.IDS) == 90
    assert {r["id"] for r in bh} == isolated.IDS
    scheduling = {"host", "pin_host", "priority", "registration_tool", "evidence_manifest"}

    def substitute(value):
        if isinstance(value, str):
            return (
                value.replace("pnpv2-20261117", "bh-20261120")
                .replace("pick-and-place-v2", "ball-hiding")
                .replace("pnpv2-await-placement-20261117", "bh-await-placement-20261120")
                .replace("scripts/pnpv2_", "runs/setup/tools/bh_")
            )
        if isinstance(value, list):
            return [substitute(x) for x in value]
        if isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        return value

    for before, after in zip(pnp, bh):
        expected = substitute({k: v for k, v in before.items() if k not in scheduling})
        assert expected == {k: v for k, v in after.items() if k not in scheduling}
        assert after["priority"] == 1338
        assert after["host"] == after["pin_host"]
    seeds = [r for r in bh if "-fql-" in r["id"]]
    assert len(seeds) == 30
    for method in isolated.METHODS:
        assert {r["fql_seed"] for r in seeds if r["realworld_method"] == method} == {0, 1, 2}


def test_rgb_has_no_geometry_edges_or_paths(graphs):
    _, bh = graphs
    rgb = {r["id"]: r for r in bh if r["pnpv2_wave"] == "rgb"}
    assert len(rgb) == 33
    for row in rgb.values():
        assert set(row["deps"]) <= set(rgb)
        assert "/geometry/" not in json.dumps(row)
        assert not row["calibration_dependent"]
    assert all(r["host"] == "local" for r in bh if "-gate-" in r["id"])
    # Chains never transfer native descriptors or feature datasets between hosts.
    by_id = {r["id"]: r for r in bh}
    for row in bh:
        for dep in row["deps"]:
            assert by_id[dep]["host"] == row["host"] or "-gate-" in dep


def test_all_configs_exact_task_substitutions():
    paths = sorted((ROOT / "configs/realworld").rglob("*ball-hiding*.yaml"))
    paths = [p for p in paths if "/tasks/" not in str(p) and "/prep/" not in str(p)]
    assert len(paths) == 17
    for path in paths:
        before = path.with_name(path.name.replace("ball-hiding", "pick-and-place-v2")).read_text()
        assert path.read_text() == before.replace("pick-and-place-v2", "ball-hiding").replace(
            "pnpv2-20261117", "bh-20261120"
        )


def test_all_wrappers_guard_then_bootstrap_then_unchanged_entry():
    wrappers = sorted(p for p in TOOLS.glob("bh_*.py") if "s4d.rwlearn.pnpv2_entry import" in p.read_text())
    assert len(wrappers) == 14
    for path in wrappers:
        tree = ast.parse(path.read_text())
        text = path.read_text()
        assert text.index("guard_gpus()") < text.index("bootstrap()") < text.index("from s4d.rwlearn.pnpv2_entry")
        original = ast.parse((ROOT / "scripts" / path.name.replace("bh_", "pnpv2_")).read_text())

        def normalized(tree):
            nodes = []
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == "s4d.rwlearn.ball_hiding":
                    continue
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if isinstance(call.func, ast.Name) and call.func.id == "bootstrap":
                        continue
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "insert":
                        continue  # explicitly checked candidate scripts root below
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        child.value = child.value.replace("ball-hiding", "pick-and-place-v2")
                nodes.append(ast.dump(node))
            return nodes

        assert normalized(tree) == normalized(original)
        assert 'parents[3] / "scripts"' in text
        assert tree.body


def test_every_row_input_resolver_and_role(graphs):
    _, bh = graphs
    from ops.jobs import host_roles, realworld_local

    isolated.install_aliases()
    roles = host_roles.load_roles(ROOT / "configs/hosts/roles.yaml")
    by_id = {r["id"]: r for r in bh}
    for row in bh:
        realworld_local.needed_inputs(row)
        assert host_roles.refusal(row, row["host"], roles, {}, by_id) is None, row["id"]
    foreign = {**bh[0], "id": "bh-20261120-gate-rgb-OTHER-S"}
    with pytest.raises(realworld_local.LocalInputError):
        realworld_local.needed_inputs(foreign)


def test_absent_adapter_does_not_import_driver_or_production(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("production module imported")

    import builtins

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith(("ops.", "s4d.")):
            return boom()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    assert isolated.install(tmp_path / "absent.json") is False


def test_check_row_only_allows_gpu_binding(graphs):
    row = graphs[1][0]
    grant = {"rows": {row["id"]: {"row": row}}}
    isolated.check_row({**row, "gpu": "MIG-synthetic"}, grant)
    for field, value in [("priority", 1), ("planned_commit", "c" * 40), ("host", "remote"), ("args", [])]:
        with pytest.raises(ValueError):
            isolated.check_row({**row, field: value}, grant)


def test_release_pause_exact_ids_and_rollback_preserves_other_holds(graphs):
    ids = {r["id"] for r in graphs[1]}
    foreign = dict(name="existing-remote-held", ids=["old-row"], active=True)
    raw = json.dumps(
        dict(pauses=[foreign, dict(name=registration.PAUSE, ids=sorted(ids), prefixes=[], active=True)])
    ).encode()
    released = json.loads(control.pause_bytes(raw, {graphs[1][0]["id"]}, False))
    assert released["pauses"][0] == foreign
    assert set(released["pauses"][1]["ids"]) == ids - {graphs[1][0]["id"]}
    assert json.loads(control.pause_bytes(json.dumps(released).encode(), {graphs[1][0]["id"]}, True))["pauses"][1]["ids"]


def test_registration_rgb_before_publication_held_and_lowest(tmp_path):
    result = child(f"""
import json
from pathlib import Path
import yaml
from runs.setup.tools import bh_register as reg
reg.setup()
from s4d.rwlearn.pnpv2_spec import SPEC, load_spec
root=Path({str(tmp_path)!r})
(root/'experiments').mkdir()
(root/'runs/setup').mkdir(parents=True)
(root/'experiments/queue.yaml').write_text(yaml.safe_dump(dict(hosts={{}},jobs=[dict(id='old',priority=4321)])))
(root/'experiments/registry.jsonl').write_text('')
(root/'runs/setup/admission-pauses.json').write_text(json.dumps(dict(pauses=[])))
record=reg.prepare(root,load_spec(SPEC),dict(commit='a'*40,image='sha256:'+'b'*64,source='local'),'rgb')
print(json.dumps(dict(rows=record['rows'],hold=json.loads(record['pause_after']),held=record['held'])))
""")
    assert len(result["rows"]) == 33 and result["held"]
    assert {r["priority"] for r in result["rows"]} == {4322}
    assert set(result["hold"]["pauses"][0]["ids"]) == {r["id"] for r in result["rows"]}


def test_nonblocking_locks_and_cas_no_mutation(tmp_path):
    import fcntl

    for name in registration.LOCKS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    with (tmp_path / registration.LOCKS[0]).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with registration.locks(tmp_path):
                raise AssertionError("busy lock admitted")


def test_remote_pin_delegates_foreign_and_refuses_changed_row(monkeypatch, tmp_path, graphs):
    import bh_remote_pin as remote

    from ops.jobs import remote_offload
    from runs.setup.tools import bh_isolated_release as remote_isolated

    row = next(r for r in graphs[1] if r["host"] == "remote")
    path = tmp_path / "grant.json"
    path.write_text("{}")
    digest = registration.sha(path.read_bytes())
    grant = dict(
        commit=row["planned_commit"],
        image=row["planned_image"],
        evidence_manifest="native.json",
        rows={row["id"]: dict(row=row)},
    )
    monkeypatch.setattr(remote, "load", lambda path: (grant, digest))
    monkeypatch.setattr(remote_offload, "admitted_pin", lambda job, ownership: ("delegated", job["id"]))
    alias_calls = []
    original_aliases = remote_isolated.install_aliases

    def aliases():
        alias_calls.append(True)
        original_aliases()

    monkeypatch.setattr(remote_isolated, "install_aliases", aliases)
    monkeypatch.setattr(remote_offload, "legacy_row", lambda job: job)
    assert remote.install(tmp_path)
    assert alias_calls == [True]
    assert remote_offload.legacy_row(row)["script"] == row["script"].replace("runs/setup/tools/bh_", "scripts/")
    assert remote_offload.legacy_row({"id": "old", "script": "old.py"}) == {"id": "old", "script": "old.py"}
    assert remote_offload.admitted_pin({"id": "old"}, {}) == ("delegated", "old")
    assert remote_offload.admitted_pin(row, {}) == ((row["planned_commit"], row["planned_image"]), "native.json")
    assert remote_offload.admitted_pin({**row, "priority": 0}, {}) is None
    path.write_text('{"revoked":true}')
    assert remote_offload.admitted_pin(row, {}) is None


def test_remote_selective_evidence_cannot_replace_native(tmp_path):
    from ops.jobs.remote_backend import RemoteJobError, acceptance_manifest

    # A passing CPU subset has no remote suite, isolation or cross-host acceptance.
    (tmp_path / "subset.json").write_text(
        json.dumps(
            dict(
                commit="a" * 40,
                image_digest="sha256:" + "b" * 64,
                counts=dict(passed=99, failed=0, error=0, skipped=0),
                exit_code=0,
                full_suite_passed=False,
            )
        )
    )
    with pytest.raises((RemoteJobError, KeyError, ValueError)):
        acceptance_manifest(
            dict(commit="a" * 40, image_digest="sha256:" + "b" * 64, evidence_manifest="subset.json"), tmp_path
        )


@pytest.fixture
def valid_grant(tmp_path, graphs):
    row = next(r for r in graphs[1] if r["host"] == "local")
    code = tmp_path / "code"
    code.mkdir()
    source = code / "source.py"
    source.write_text("# sealed source\n")
    source.chmod(0o444)
    code.chmod(0o555)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            dict(
                commit=row["planned_commit"],
                code_root=str(code),
                source_files={"source.py": dict(sha256=registration.sha(source.read_bytes()))},
            )
        )
    )
    report = tmp_path / "results.json"
    report.write_text(json.dumps(dict(exit_code=0, counts=dict(passed=3, failed=0, error=0, skipped=0))))
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            dict(
                commit=row["planned_commit"],
                ids=[row["id"]],
                science_unchanged=True,
                test_report=str(report),
                artifacts_sha256={str(report): registration.sha(report.read_bytes())},
            )
        )
    )
    grant = dict(
        kind="bh-isolated-targeted-v1",
        active=True,
        local_gpus=list(isolated.LOCAL_GPUS),
        commit=row["planned_commit"],
        accepted_by="synthetic owner",
        full_suite_claimed=False,
        rows={row["id"]: dict(row=row, row_sha256=registration.sha(registration.canonical(row)))},
        adapter_sha256=registration.sha(Path(isolated.__file__).read_bytes()),
        worker_sha256=registration.sha((TOOLS / "bh_isolated_worker.py").read_bytes()),
        snapshot=str(snapshot),
        snapshot_sha256=registration.sha(snapshot.read_bytes()),
        targeted_evidence=str(evidence),
        targeted_evidence_sha256=registration.sha(evidence.read_bytes()),
    )
    path = tmp_path / "grant.json"
    path.write_text(json.dumps(grant))
    return path, grant, code


def test_grant_integrity_rejects_changed_source_and_false_suite_claim(valid_grant):
    path, grant, code = valid_grant
    assert isolated.load_authorization(path)[0] == grant
    grant["full_suite_claimed"] = True
    path.write_text(json.dumps(grant))
    with pytest.raises(ValueError):
        isolated.load_authorization(path)
    grant["full_suite_claimed"] = False
    path.write_text(json.dumps(grant))
    source = code / "source.py"
    source.chmod(0o644)
    source.write_text("changed")
    with pytest.raises(ValueError):
        isolated.load_authorization(path)


def test_local_install_delegates_and_preserves_input_policy(monkeypatch, valid_grant):
    path, grant, code = valid_grant
    from ops.jobs import fair_share, local_devices, realworld_local, scheduler

    monkeypatch.setattr(local_devices, "allows", lambda *a, **kw: True)
    monkeypatch.setattr(fair_share, "allows", lambda *a, **kw: True)
    row = next(iter(grant["rows"].values()))["row"]
    called = []
    monkeypatch.setattr(scheduler, "_bh_isolated_installed", set(), raising=False)
    monkeypatch.setattr(realworld_local, "localize", lambda job, state, **kw: ("delegated", job["id"]))
    monkeypatch.setattr(realworld_local, "applies", lambda *a, **kw: True)

    def inputs(*args):
        called.append(args)
        return ["path-only"], {"input_closure": {"verified": "digest"}}

    monkeypatch.setattr(realworld_local, "_inputs", inputs)
    monkeypatch.setattr(scheduler, "launch", lambda job, *a, **kw: (job, kw))
    monkeypatch.setattr(scheduler, "record", lambda event, *a, **kw: event)
    assert isolated.install(path)
    assert all(local_devices.allows(row, g) and fair_share.allows(row, g) for g in isolated.LOCAL_GPUS)
    assert not local_devices.allows(row, "MIG-outside-GPU45")
    assert local_devices.allows({"id": "old"}, "MIG-outside-GPU45")
    with pytest.raises(ValueError, match="eight GPU4/5"):
        scheduler.launch(row, "MIG-outside-GPU45", 1, args=["path-only"])
    assert realworld_local.localize({"id": "old"}, {}, roles={}) == ("delegated", "old")
    effective, record = realworld_local.localize(row, {}, roles={})
    assert effective == ["path-only"] and called[0][6] == code
    assert record["input_closure"] == {"verified": "digest"}
    launch, options = scheduler.launch(row, isolated.LOCAL_GPUS[0], 1, args=effective)
    assert launch["script"] == "runs/setup/tools/bh_isolated_worker.py"
    assert options["args"] == [str(path), row["id"], "path-only"]
    event = scheduler.record(dict(event="launched", id=row["id"], host="local"))
    assert event["commit"] == row["planned_commit"]
    assert scheduler.record(dict(event="launched", id="old", host="local")) == dict(event="launched", id="old", host="local")
    monkeypatch.setattr(realworld_local, "applies", lambda *a, **kw: False)
    with pytest.raises(realworld_local.LocalInputError):
        realworld_local.localize(row, {}, roles={})


def test_worker_rejects_extra_or_scientific_argument_changes_before_writes(monkeypatch, valid_grant, tmp_path):
    path, grant, code = valid_grant
    row = next(iter(grant["rows"].values()))["row"]
    monkeypatch.setattr(isolated, "MAIN", tmp_path)
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("S4D_HOST_CONFIG", str(ROOT / "configs/hosts/local.yaml"))
    monkeypatch.setenv("S4D_CACHE_ROOT", str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="arguments differ"):
        isolated.prepare_worker(path, row["id"], ["--batch-size", "1"])
    assert not (tmp_path / "runs").exists()


def test_control_cas_refuses_changed_queue_without_unholding(tmp_path):
    for name in registration.LOCKS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    queue = tmp_path / "experiments/queue.yaml"
    queue.write_bytes(b"changed")
    pause = tmp_path / "runs/setup/admission-pauses.json"
    pause.write_bytes(b"original hold")
    prepared = dict(queue_path=queue, queue_raw=b"old", pause_path=pause, pause_raw=b"original hold")
    with pytest.raises(ValueError, match="queue changed"):
        control.apply(prepared, tmp_path)
    assert pause.read_bytes() == b"original hold"


@pytest.mark.parametrize("route", ["hybrid", "all-local"])
def test_advancer_releases_local_while_remote_native_is_pending(monkeypatch, tmp_path, route):
    from types import SimpleNamespace

    import bh_advance as advance

    main = tmp_path / "main"
    (main / "runs/setup").mkdir(parents=True)
    output = tmp_path / "ops"
    output.mkdir()
    commit = "a" * 40
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(dict(commit=commit)))
    proof = main / "runs/setup/bh-20261120-rgb.registration.json"
    proof.write_text(
        json.dumps(
            dict(
                release=dict(commit=commit),
                route=route,
                rows=[dict(host="local")] + ([dict(host="remote")] if route == "hybrid" else []),
            )
        )
    )
    monkeypatch.setattr(advance, "MAIN", main)
    calls = []

    def run(argv, **kwargs):
        script = Path(argv[3]).name
        calls.append((script, argv[4:]))
        if script == "bh_register.py":
            return SimpleNamespace(returncode=2)  # geometry not published
        if script == "bh_prepare_release.py":
            if argv[4] == "remote":
                return SimpleNamespace(returncode=2)  # no native acceptance
            Path(argv[argv.index("--out") + 1]).write_text("{}")
        if script == "bh_release_control.py" and "--apply" in argv:
            target = main / "runs/setup/bh-isolated-release.d/local-rgb.json"
            target.parent.mkdir()
            target.write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(advance.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bh_advance",
            "--route",
            route,
            "--snapshot",
            str(snapshot),
            "--binding",
            str(tmp_path / "binding"),
            "--image",
            "sha256:" + "b" * 64,
            "--out",
            str(output),
            "--accepted-by",
            "owner",
            "--once",
            "--apply",
        ],
    )
    assert advance.main() == 0
    assert (main / "runs/setup/bh-isolated-release.d/local-rgb.json").exists()
    assert any(s == "bh_prepare_release.py" and args[0] == "remote" for s, args in calls) == (route == "hybrid")
    assert any(s == "bh_fetch_inputs.py" for s, args in calls)
    assert any(s == "bh_register.py" and "geometry" in args for s, args in calls)


def test_fetch_from_candidate_targets_canonical_main_mirror(monkeypatch, tmp_path, graphs):
    import bh_fetch_inputs as fetcher

    from ops.jobs import realworld_local
    from ops.remote import access

    row = graphs[1][0]
    proof = tmp_path / "registration.json"
    proof.write_text(json.dumps(dict(rows=[row])))
    main = tmp_path / "main"
    (main / "experiments").mkdir(parents=True)
    (main / "experiments/queue.yaml").write_text(json.dumps(dict(jobs=[row])))
    monkeypatch.setattr(fetcher, "MAIN", main)
    monkeypatch.setattr(access, "RemoteClient", lambda **kwargs: ("client", kwargs["repo"]))
    received = []

    def fetch(*args, **kwargs):
        received.append(kwargs)
        return {"verified": True}

    monkeypatch.setattr(realworld_local, "fetch", fetch)
    monkeypatch.setattr(sys, "argv", ["fetch", "--registration", str(proof), "--apply"])
    assert fetcher.main() == 0
    assert received[0]["mirror"] == main / "runs/realworld-inputs"
    assert received[0]["client"] == ("client", main)
    assert received[0]["priority"] == "dependency" and received[0]["dry_run"] is False


def test_99_membership_and_published_view_paths(tmp_path):
    result = child("""
import copy,json
from pathlib import Path
from s4d.rwlearn.ball_hiding import bootstrap
bootstrap()
from s4d.rwlearn.pnpv2_spec import SPEC,load_spec,values
from s4d.rwlearn.pnpv2_publication import manifest_check
from s4d.rwlearn.ball_hiding_membership import amendment,SPLIT
from tests import test_pnpv2_geometry_route as T
import pytest,tempfile
a=amendment(); split=json.loads(SPLIT.read_text())
with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
    m=T.campaign.__wrapped__(Path(tmp),mp).m
    spec=load_spec(SPEC); manifest_check(m,spec)
    assert m['split']=={k:split[k] for k in ('train','validation')}
    assert list(map(len,(split['train'],split['validation'])))==[89,10]
    assert not any('episode_000008' in e for k in ('train','validation') for e in split[k])
    for change in ('excluded','split','episodes'):
        bad=copy.deepcopy(m)
        if change=='split':bad['split']['train'].append(bad['split']['validation'].pop())
        elif change=='excluded':bad['excluded']=[]
        else:bad['episodes']=bad['episodes'][:-1]
        try:manifest_check(bad,spec)
        except ValueError:pass
        else:raise AssertionError(change)
    p=values(spec)
    assert '/bh3/ingest99/' in p['manifest']
    for wave in ('rgb','geometry'):
        assert p[wave+'_manifest'].endswith('/'+wave+'/manifest.json')
        assert p[wave+'_split'].endswith('/'+wave+'/split_seed0.json')
print(json.dumps(dict(episodes=99,train=89,validation=10)))
""")
    assert result == dict(episodes=99, train=89, validation=10)


def test_all_local_route_preserves_science_and_original_roles():
    assert child("""
import json
from s4d.rwlearn.ball_hiding import bootstrap
bootstrap()
from s4d.rwlearn.pnpv2_spec import SPEC,load_spec
from s4d.rwlearn.bh_training import rows
from runs.setup.tools.bh_isolated_release import install_aliases
from ops.jobs import host_roles,realworld_local
from pathlib import Path
spec=load_spec(SPEC); release=dict(commit='a'*40,image='sha256:'+'b'*64,source='local')
kw=dict(waves=('rgb','geometry'),scene=dict(lookat=[.4,0,.1],pose_scale=2.5,bounds=[-.5,-.6,-.05,.9,.6,.7]))
hybrid=rows(spec,release,**kw); local=rows(spec,release,route='all-local',**kw)
assert len(local)==90 and {r['host'] for r in local}=={'local'}
install_aliases(); roles=host_roles.load_roles(Path('configs/hosts/roles.yaml'))
by_id={r['id']:r for r in local}
for before,after in zip(hybrid,local):
    fields=lambda r:{k:v for k,v in r.items() if k not in ('host','pin_host')}
    assert fields(before)==fields(after)
    realworld_local.needed_inputs(after)
    assert host_roles.refusal(after,'local',roles,{},by_id) is None,after['id']
print(json.dumps(True))
""")
