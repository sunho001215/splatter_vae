"""Pure registration and verified local-pool input tests. Never imports a scheduler or campaign driver."""

import copy
import importlib.util
import json
import subprocess
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest
import yaml

from ops.jobs import host_roles
from ops.jobs import realworld_local as rw
from s4d.train.continuation import file_sha

ROOT = Path(__file__).resolve().parents[3]  # runs/setup/tools -> repository root
SPEC = importlib.util.spec_from_file_location("ours_ext_reg", ROOT / "runs/setup/tools/register_ours_ext_20261113.py")
reg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reg)


def queue():
    rows = json.loads((ROOT / "tests/fixtures/ours_ext_templates.json").read_text())
    return {"jobs": rows + [dict(id="other", priority=0), dict(id="other-fql", priority=1, role="realworld_fql")]}


def make_plan():
    cfg = yaml.safe_load((ROOT / reg.BASE).read_text())
    release = dict(commit="a" * 40, image="sha256:" + "b" * 64, evidence_manifest="acceptance.json")
    return reg.build_rows(queue(), release, "c" * 64, file_sha(ROOT / reg.BASE), cfg)


def test_dry_run_rows_priorities_exact_fql_and_independent_branches(tmp_path, capsys, monkeypatch):
    current = queue()
    rows, priorities = make_plan()
    assert len(rows) == len({r["id"] for r in rows}) == 15
    assert priorities["pretraining"] == -1 and priorities["ours_fql"] == 0
    ids = {r["id"]: r for r in rows}
    sources = {r["id"]: r for r in current["jobs"]}
    pre_a, pre_b = reg.PREFIX + "pre300k", reg.PREFIX + "pre400k"
    assert ids[pre_a]["deps"] == [reg.OLD, reg.STATS]
    assert ids[pre_b]["deps"] == [pre_a, reg.STATS]
    assert all(ids[job]["priority"] < min(r.get("priority", 5) for r in current["jobs"]) for job in (pre_a, pre_b))
    for label in ("300k", "400k"):
        dataset = f"ourx-20261113-fqldata{label}-ours-drawer-open"
        assert ids[dataset]["deps"] == [reg.PREFIX + "export" + label]
        for seed in range(3):
            job = f"ourx-20261113-fql{label}-ours-drawer-open-s{seed}"
            source = sources[ids[job]["derived_from"]]
            expected = [
                a.replace(source["id"], job).replace("rwcal-20261111-fqldata-ours-drawer-open", dataset)
                for a in source["args"]
            ]
            assert ids[job]["args"] == expected
            assert ids[job]["fql_budget"] == source["fql_budget"]
            assert ids[job]["deps"] == [dataset]
            assert ids[job]["priority"] < min(
                r.get("priority", 5) for r in current["jobs"] if r.get("role") == "realworld_fql"
            )
        assert pre_b not in ids["ourx-20261113-fql300k-ours-drawer-open-s0"]["deps"]
    for row in rows:
        assert "pin_host" not in row and row["gpu"] == "any"
        assert row["gpu_profile"] == "1g.24gb" and row["cpu_threads"] <= 16
        assert row["mem_gb"] == row["gpu_mem_gb"] < 24
        assert "config.yaml" not in row["required_results"]  # container wrapper reserves this result name
        assert rw.has_resolver(row)
        assert host_roles.job_class(row, {**sources, **ids}).cls in ("realworld_pretrain", "realworld_fql")
        needed = rw.needed_inputs(row)
        if row["script"] in (rw.OURS_CONTINUATION, rw.OURS_EVAL):
            # Every remote/run path hidden in the baseline is a consumed, plain, checked argument.
            checked = {rw.REMOTE_PREFIX + "/" + path for path in needed["mirror"]} | {
                f"runs/{dep}/{path}" for dep, path in needed["dependency_files"]
            }
            assert rw._resolved_paths(row, ROOT) <= checked
    before = yaml.safe_dump(current).encode()
    candidate = reg.queue_candidate(before, rows)
    assert yaml.safe_load(candidate)["jobs"] == current["jobs"] + rows
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/queue.yaml").write_bytes(before)
    (tmp_path / "experiments/registry.jsonl").write_bytes(b"")
    original = tmp_path / "runs" / reg.OLD
    (original / "checkpoints").mkdir(parents=True)
    (original / "config.yaml").write_bytes((ROOT / reg.BASE).read_bytes())
    (original / "checkpoints/step_0200000.pt").write_bytes(b"CPU fixture checkpoint")
    from s4d.config import load_config

    for seed in range(3):
        job = f"rwnq-20261111-fql-ours-drawer-open-s{seed}"
        args = sources[job]["args"]
        saved = load_config([ROOT / args[args.index("--config") + 1]], args[args.index("--set") + 1 :])
        path = tmp_path / "runs" / job / "config.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(saved))
    reg.main(["--repo", str(tmp_path), "--commit", "a" * 40, "--image", "sha256:" + "b" * 64])
    report = json.loads(capsys.readouterr().out)
    assert report["registered"] is False and report["queue_diff"]["added_rows"] == 15
    assert report["queue_diff"]["modified_rows"] == 0
    assert (tmp_path / "experiments/queue.yaml").read_bytes() == before
    assert not (tmp_path / "runs/setup").exists()
    assert 'source' not in report['release'] and 'holds' not in report
    # The local CLI uses the same preparation and science checks, adding only local evidence and an exact hold.
    for relative in {reg.BASE, reg.OVERLAY, reg.TOOL, *[
            row['args'][row['args'].index('--config') + 1] for row in current['jobs']
            if '--config' in row.get('args', [])]}:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/tests.json').write_text('{}')
    release = {**report['release'], 'source': 'local', 'evidence_manifest': reg.LOCAL_EVIDENCE}
    monkeypatch.setattr(reg, 'local_release_pin', lambda repo, commit, image: release)
    reg.main(['--repo', str(tmp_path), '--source-root', str(tmp_path), '--commit', 'a' * 40,
              '--image', 'sha256:' + 'b' * 64, '--release-source', 'local'])
    local = json.loads(capsys.readouterr().out)
    assert local['registered'] is False and local['release'] == release
    assert local['holds'] == [reg.local_hold(local['rows'])]
    assert local['input_sha256'][str(tmp_path / 'docs/tests.json')] == file_sha(tmp_path / 'docs/tests.json')
    assert all(row['host'] == 'remote' and row['release_source'] == 'local' for row in local['rows'])
    assert (tmp_path / 'experiments/queue.yaml').read_bytes() == before
    assert (tmp_path / 'experiments/registry.jsonl').read_bytes() == b''
    assert list((tmp_path / 'runs/setup').iterdir()) == [tmp_path / 'runs/setup/tools']
    with pytest.raises(RuntimeError, match='main sources'):
        reg.prepare(tmp_path, ROOT, 'a' * 40, 'sha256:' + 'b' * 64, 'local')
    with pytest.raises(RuntimeError, match='unknown release source'):
        reg.prepare(tmp_path, ROOT, 'a' * 40, 'sha256:' + 'b' * 64, 'invalid')
    # Rerun against a changed priority field computes a strictly higher new priority.
    current["jobs"].append(dict(id="higher-fql", role="realworld_fql", priority=-10))
    cfg = yaml.safe_load((ROOT / reg.BASE).read_text())
    _, priority = reg.build_rows(current, report["release"], "c" * 64, "d" * 64, cfg)
    assert priority["ours_fql"] == -11 and priority["pretraining"] == -12


def test_execute_refuses_worktree_and_unverified_inputs(tmp_path):
    with pytest.raises(RuntimeError, match="coordinator"):
        reg.execute(tmp_path, {})
    rows, _ = make_plan()
    bad = copy.deepcopy(rows[0])
    bad["args"][bad["args"].index("--source-checkpoint") + 1] = "runs/undeclared/checkpoints/step_0200000.pt"
    with pytest.raises(rw.LocalInputError, match="dependency"):
        rw.needed_inputs(bad)
    bad = copy.deepcopy(rows[0])
    bad["args"].append("loss.chamfer=1")
    with pytest.raises(rw.LocalInputError, match="four verified"):
        rw.needed_inputs(bad)


def test_pure_mig_profile_and_completion_predicates():
    from ops.jobs.continuation import continuation_endpoint, profile_matches

    assert profile_matches({"gpu_profile": "1g.24gb"}, "MIG-one", {"MIG-one"})
    assert not profile_matches({"gpu_profile": "1g.24gb"}, "GPU-physical", {"MIG-one"})
    assert profile_matches({}, "GPU-physical", set())
    with pytest.raises(ValueError):
        profile_matches({"gpu_profile": "7g.192gb"}, "MIG-one", {"MIG-one"})
    completion = dict(status="completed", steps=400000)
    metadata = dict(status="completed", stop_step=300000, schedule_steps=400000)
    assert continuation_endpoint(["--stop-step", "300000"], completion, metadata) == 300000
    with pytest.raises(ValueError):
        continuation_endpoint(["--stop-step", "300000"], dict(completion, steps=300000), metadata)
    with pytest.raises(ValueError):
        continuation_endpoint(["--stop-step", "400000"], completion, metadata)


def test_container_output_paths_for_new_jobs():
    from ops.containers.launch import validate_output_arguments

    for script in ("scripts/train_ours_continuation.py", "scripts/evaluate_ours_length.py"):
        validate_output_arguments(script, ["--name", "own", "--output-root", "runs"], "own")
        with pytest.raises(ValueError, match="artifacts"):
            validate_output_arguments(script, ["--name", "other", "--output-root", "runs"], "own")


def test_actual_legacy_completion_endpoint_is_validated_without_driver_imports():
    from ops.jobs.continuation import pretraining_endpoint

    assert pretraining_endpoint({"steps": 10, "stop_step": 5}) == 5
    assert pretraining_endpoint({"steps": 10, "stop_step": 10**309}) == 10
    assert pretraining_endpoint({"steps": 10, "stop_step": 0}) == 10
    assert pretraining_endpoint({"steps": 10, "stop_step": float("nan")}) is None
    assert pretraining_endpoint({"steps": 10, "stop_step": True}) is None
    assert pretraining_endpoint({"steps": 10, "stop_step": "5"}) is None


def test_explicit_step_overrides_do_not_hide_early_stop():
    from ops.jobs.continuation import resolved_pretraining_endpoint

    assert resolved_pretraining_endpoint(["train.steps=400000", "train.stop_step=300000"], {}) == 300000
    assert resolved_pretraining_endpoint(["train.steps=10"], {"train": {"steps": 5, "stop_step": 3}}) == 3
    assert resolved_pretraining_endpoint([], {"train": {"steps": 200000}}) == 200000
    assert resolved_pretraining_endpoint(["train.steps=10"], {}) == 10


@pytest.mark.parametrize('metadata', [None, dict(status='completed', stop_step=400000, schedule_steps=400000),
                                     dict(status='completed', stop_step=300000, schedule_steps=400000)])
def test_local_closeout_requires_separate_verified_continuation_metadata(tmp_path, metadata):
    row = dict(id='continuation', script=rw.OURS_CONTINUATION, args=['--stop-step', '300000'],
               required_results=['completion.json', 'continuation.json'])
    run = tmp_path / 'runs' / row['id']
    run.mkdir(parents=True)
    (run / 'completion.json').write_text(json.dumps(dict(status='completed', steps=400000)))
    if metadata:
        (run / 'continuation.json').write_text(json.dumps(metadata))
    state = {row['id']: dict(status='done', host='local')}
    if not metadata or metadata['stop_step'] != 300000:
        with pytest.raises(rw.LocalInputError):
            rw.closeout(row['id'], state, row, tmp_path / 'finished', runs=tmp_path / 'runs')
        assert not (tmp_path / 'finished').exists()
    else:
        record = rw.closeout(row['id'], state, row, tmp_path / 'finished', runs=tmp_path / 'runs')
        assert set(record['files']) == {'completion.json', 'continuation.json'}


@pytest.fixture
def local_repo(tmp_path):
    """Small real git checkout; its own bootstrap fingerprints exactly the launch-gate file kinds."""
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 's4d').mkdir()
    (tmp_path / 'runs/setup').mkdir(parents=True)
    (tmp_path / 'docs').mkdir()
    (tmp_path / 's4d/source.py').write_text('VALUE = 1\n')
    (tmp_path / 'scripts/_bootstrap.py').write_text(
        'from pathlib import Path\nimport hashlib\n'
        'def source_fingerprints():\n'
        '    root = Path(__file__).resolve().parents[1]\n'
        '    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()\n'
        '            for folder in ("s4d", "scripts") for p in (root / folder).rglob("*.py")}\n')
    image = 'sha256:' + 'b' * 64
    (tmp_path / 'runs/setup/role-release.json').write_text(json.dumps(dict(commit='a' * 40, image=image)))
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=test', '-c', 'user.email=test@test',
                    'commit', '-qm', 'fixture'], check=True)
    head = subprocess.run(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], check=True,
                          capture_output=True, text=True).stdout.strip()
    evidence = dict(full_suite_passed=True, exit_code=0,
                    counts=dict(passed=5, failed=0, error=0, skipped=0),
                    source_sha256={str(p.relative_to(tmp_path)): file_sha(p)
                                   for folder in ('s4d', 'scripts') for p in (tmp_path / folder).rglob('*.py')})
    (tmp_path / 'docs/tests.json').write_text(json.dumps(evidence))
    return tmp_path, head, image, evidence


def test_local_release_requires_head_and_role_image(local_repo):
    repo, head, image, _ = local_repo
    assert reg.local_release_pin(repo, head, image) == dict(
        commit=head, image=image, source='local', evidence_manifest=reg.LOCAL_EVIDENCE)
    with pytest.raises(RuntimeError, match='HEAD differs'):
        reg.local_release_pin(repo, 'c' * 40, image)
    with pytest.raises(RuntimeError, match='image differs'):
        reg.local_release_pin(repo, head, 'sha256:' + 'd' * 64)


@pytest.mark.parametrize('bad', [dict(full_suite_passed=False), dict(exit_code=1), dict(source_sha256={}),
                               *[dict(counts={**dict(passed=5, failed=0, error=0, skipped=0), **change})
                                 for change in (dict(failed=1), dict(error=1), dict(skipped=1),
                                                dict(passed=0))], dict(counts={})])
def test_local_release_refuses_incomplete_or_stale_suite(local_repo, bad):
    repo, head, image, good = local_repo
    (repo / 'docs/tests.json').write_text(json.dumps({**good, **bad}))
    with pytest.raises(RuntimeError, match='passing full suite|stale'):
        reg.local_release_pin(repo, head, image)


@pytest.mark.parametrize('name', ['s4d/source.py', 's4d/untracked.py'])
def test_local_release_refuses_passing_evidence_from_dirty_sources(local_repo, name):
    repo, head, image, evidence = local_repo
    (repo / name).write_text('VALUE = 2\n')
    evidence['source_sha256'][name] = file_sha(repo / name)
    (repo / 'docs/tests.json').write_text(json.dumps(evidence))
    with pytest.raises(RuntimeError, match='source differs from the local release HEAD'):
        reg.local_release_pin(repo, head, image)


def local_rows():
    cfg = yaml.safe_load((ROOT / reg.BASE).read_text())
    release = dict(commit='a' * 40, image='sha256:' + 'b' * 64, source='local',
                   evidence_manifest=reg.LOCAL_EVIDENCE)
    rows, priorities = reg.build_rows(queue(), release, 'c' * 64, file_sha(ROOT / reg.BASE), cfg)
    return rows, priorities, release


def test_local_rows_preserve_science_priorities_and_remote_host():
    rows, priorities, _ = local_rows()
    remote, remote_priorities = make_plan()
    assert priorities == remote_priorities == dict(pretraining=-1, ours_fql=0,
        semantics='smaller numbers run first; recomputed from the publication queue')
    for row, original in zip(rows, remote):
        assert row['release_source'] == 'local' and row['local_placement_pause'] == reg.LOCAL_PAUSE
        assert row['evidence_manifest'] == reg.LOCAL_EVIDENCE and row['host'] == 'remote'
        assert {k: v for k, v in row.items() if k not in ('release_source', 'local_placement_pause', 'evidence_manifest')} == {
            k: v for k, v in original.items() if k != 'evidence_manifest'}
        assert rw.has_resolver(row)
        rw.needed_inputs(row)
    hold = reg.local_hold(rows)
    assert hold['active'] is True and hold['prefixes'] == []
    assert hold['ids'] == hold['rows'] == sorted(r['id'] for r in rows)


@pytest.mark.parametrize('mutation', ['missing', 'inactive', 'missing-id', 'extra-id', 'prefix', 'duplicate'])
def test_local_publication_requires_exact_active_hold(tmp_path, mutation):
    rows, _, _ = local_rows()
    hold = reg.local_hold(rows)
    if mutation == 'inactive':
        hold['active'] = False
    elif mutation == 'missing-id':
        hold['ids'].pop()
    elif mutation == 'extra-id':
        hold['ids'].append('unrelated')
    elif mutation == 'prefix':
        hold['prefixes'] = ['ourx-']
    path = tmp_path / 'runs/setup/admission-pauses.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(pauses=[] if mutation == 'missing' else
                                   [hold, hold] if mutation == 'duplicate' else [hold])))
    with pytest.raises(RuntimeError, match='ACTIVE admission pause'):
        reg.require_local_hold(tmp_path, dict(rows=rows))
    path.write_text(json.dumps(dict(pauses=[reg.local_hold(rows)])))
    reg.require_local_hold(tmp_path, dict(rows=rows))


@pytest.mark.parametrize('changed', ['hold', 'head', 'evidence', None])
def test_local_transaction_rechecks_under_locks_before_replace(tmp_path, monkeypatch, changed):
    rows, _, release = local_rows()
    setup = tmp_path / 'runs/setup'
    setup.mkdir(parents=True)
    (tmp_path / 'experiments').mkdir()
    queue_path = tmp_path / 'experiments/queue.yaml'
    before = yaml.safe_dump(queue()).encode()
    queue_path.write_bytes(before)
    (tmp_path / 'experiments/registry.jsonl').write_bytes(b'')
    evidence = tmp_path / 'docs/tests.json'
    evidence.parent.mkdir()
    evidence.write_text('{}')
    pause = setup / 'admission-pauses.json'
    pause.write_text(json.dumps(dict(pauses=[reg.local_hold(rows)])))
    candidate = reg.queue_candidate(before, rows)
    record = dict(repo=str(tmp_path), source_root=str(tmp_path), release=release, rows=rows,
                  rows_sha256=reg.json_sha(rows), queue_sha256_before=reg.sha(before),
                  queue_sha256_after=reg.sha(candidate), registry_sha256=reg.sha(b''),
                  input_sha256={str(evidence): file_sha(evidence)})
    monkeypatch.setattr(reg, 'MAIN', tmp_path)
    monkeypatch.setattr(reg, '__file__', str(tmp_path / reg.TOOL))
    locked = False
    @contextmanager
    def locks(repo):
        nonlocal locked
        locked = True
        yield
        locked = False
    monkeypatch.setattr(reg, 'locks', locks)
    checks = []
    def pin(repo, commit, image):
        assert locked
        checks.append(True)
        if changed == 'head' and len(checks) == 2:
            raise RuntimeError('main HEAD differs from --commit')
        return release
    monkeypatch.setattr(reg, 'local_release_pin', pin)
    monkeypatch.setattr(rw, 'placement_scope', lambda row, root: {})
    write = reg.atomic_write
    def atomic(path, raw):
        write(path, raw)
        if path.name.endswith('.pending.json'):
            if changed == 'hold':
                pause.write_text(json.dumps(dict(pauses=[])))
            elif changed == 'evidence':
                evidence.write_text('{"changed": true}')
    monkeypatch.setattr(reg, 'atomic_write', atomic)
    if changed:
        with pytest.raises(RuntimeError, match='ACTIVE admission pause|HEAD differs|source/config changed'):
            reg.execute(tmp_path, record)
        assert queue_path.read_bytes() == before
    else:
        result = reg.execute(tmp_path, record)
        assert result['registered'] is True and queue_path.read_bytes() == candidate
        assert not (setup / f'{reg.STEM}.pending.json').exists()
    assert len(checks) == 2
    assert (tmp_path / 'experiments/registry.jsonl').read_bytes() == b''


@pytest.mark.parametrize('index', range(15))
def test_local_planner_checks_each_rows_mirrors_and_dependencies(tmp_path, monkeypatch, index):
    """Exercise the actual input resolver/planner for all five scripts with synthetic completed local producers."""
    rows, _, _ = local_rows()
    row = rows[index]
    needed = rw.needed_inputs(row)
    mirror, runs, finished = (tmp_path / part for part in ('mirror', 'runs', 'finished'))
    mirror.mkdir()
    raw = tmp_path / 'raw-originals'
    raw.mkdir()
    monkeypatch.setattr(rw, 'LOCAL_RAW_ROOT', raw)
    files = {}
    for name in needed['mirror']:
        path = mirror / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('kind: compact\npaths: []\n' if path.name == 'store.yaml' else '{}')
        stat = path.stat()
        files[name] = dict(kind='copy', sha256=file_sha(path), remote_sha256=file_sha(path),
                           bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, requires=[])
    manifest = mirror / 'MANIFEST.json'
    manifest.write_text(json.dumps(dict(version=1, remote_root=rw.REMOTE_PREFIX,
                                        local_root='runs/realworld-inputs', files=files)))
    state = {dep: dict(status='done', host='local') for dep in row['deps']}
    results = {dep: [] for dep in row['deps']}
    for dep, name in needed['dependency_files']:
        path = runs / dep / name
        if name == '.':
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'verified producer result')
            results[dep].append(name)
    for dep, names in results.items():
        rw.closeout(dep, state, dict(id=dep, required_results=names), finished, runs=runs)
    # The synthetic commit has no git objects; real HEAD scope/input checks are also exercised by the main dry run.
    monkeypatch.setattr(rw, 'placement_scope', lambda job, root: dict(source_commit=job['planned_commit'], files={}))
    monkeypatch.setattr(rw, '_inputs', partial(rw._inputs, finished_dir=finished))
    capacity = dict(ram_total_bytes=128 * 2**30, ram_free_bytes=128 * 2**30, cpu_free_threads=64,
                    disk_total_bytes=1000 * 2**30, disk_free_bytes=1000 * 2**30)
    def plan(allow):
        return rw.plan(dict(jobs=[row]), state, host_roles.load_roles(),
                       paused_fn=lambda job: [reg.LOCAL_PAUSE], allow_pauses=allow,
                       policy='all', capacity=capacity, mirror=mirror, manifest=manifest,
                       runs=runs, root=ROOT)[0]
    assert plan([])['reason'] == 'paused'
    allowed = plan([reg.LOCAL_PAUSE])
    assert allowed['decision'] == 'local', allowed
    effective, inputs = rw._inputs(row, state, mirror, manifest, runs, root=ROOT, finished_dir=finished)
    assert not any(rw.REMOTE_PREFIX in arg for arg in effective)
    assert len(inputs['dependency_inputs']) == sum(name != '.' for _, name in needed['dependency_files'])
    if row['script'] == rw.FQL_DATASET:
        assert effective[-2:] == ['--raw-root', f'{rw.REMOTE_RAW_ROOT}={raw}']
    # A disappearing checked input must make placement wait, even with the hold explicitly allowed.
    if needed['mirror']:
        (mirror / needed['mirror'][0]).unlink()
    else:
        dep, name = next((dep, name) for dep, name in needed['dependency_files'] if name != '.')
        (runs / dep / name).unlink()
    assert plan([reg.LOCAL_PAUSE])['decision'] == 'waits'
