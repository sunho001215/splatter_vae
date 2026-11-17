"""Release gate (scripts/test_gate.py, user approval 2026-11-17 ~23:10): a complete suite, or a selective run proven
equivalent to one by an accepted baseline, complete change-impact coverage, exact-key reuse and no high-risk change.

Synthetic git repositories and report dictionaries only; no suite is run."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_tests as runner
from scripts import test_gate as gate

GIT = ["git", "-c", "user.name=Gate Test", "-c", "user.email=gate@example.invalid", "-c", "commit.gpgsign=false"]
FILES = {
    "s4d/__init__.py": "", "s4d/a.py": "A = 1\n", "s4d/b.py": "B = 1\n",
    "ops/__init__.py": "", "ops/migration.py": "def execution_scope():\n    pass\n",
    "ops/robot_pc/__init__.py": "", "ops/robot_pc/executor.py": "SAFE = True\n",
    "scripts/run_tests.py": "", "configs/x.yaml": "x: 1\n", "uv.lock": "lock\n", "docs/notes.md": "notes\n",
    "runs/setup/tools/one_off.py": "", "tests/conftest.py": "",
    "tests/test_a.py": "import s4d.a\n\ndef test_a():\n    assert s4d.a.A\n",
    "tests/test_b.py": "import s4d.b\n\ndef test_b():\n    assert s4d.b.B\n",
    "tests/test_scope.py": "from ops import migration\n\ndef test_scope():\n    migration.execution_scope()\n",
}
ENV = {"python": "3.10.x", "torch": "2.10.0", "cuda_runtime": "12.9", "host": "local",
       "gpu_mapping": [{"name": "MIG 1g.24gb"}]}


def git(root: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def write(root: Path, name: str, text: str) -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for name, text in FILES.items():
        write(root, name, text)
    git(root, "init", "-q")
    git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    return root


def nodes(files):
    return [f"{f}::{Path(f).stem}" for f in files]


@pytest.fixture
def baseline(repo, tmp_path):
    """An accepted baseline for the base commit (every test file passed)."""
    commit = git(repo, "rev-parse", "HEAD")
    ids = nodes(sorted(f for f in FILES if f.startswith("tests/test_")))
    report = {**ENV, "full_suite_passed": True, "exit_code": 0, "problems": [], "source_changed_during_suite": False,
              "counts": {"passed": len(ids), "error": 0, "failed": 0, "skipped": 0}, "collected_total": len(ids),
              "source_sha256": gate.commit_sources(repo, commit), "reuse": {"reused_tests": 0}}
    store = gate.Baselines(tmp_path / "cache")
    store.record(report, ids, commit, repo, "evidence")
    return SimpleNamespace(commit=commit, store=store, report=report, ids=ids)


def run(repo, baseline, *, commit=True, reused=(), mutate=None, base=None):
    """Commit the candidate, select, and build the selective run's report and verdict."""
    if commit:
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "candidate", "--allow-empty")
    base = base or baseline.commit
    tree = gate.test_impact.Tree(repo)
    selection = {**gate.test_impact.select(gate.test_impact.Graph(tree), gate.test_impact.changed_files(base, repo)),
                 "base": base}
    selected = nodes(selection["files"])
    reused_nodes = [n for n in selected if n.split("::")[0] in reused]
    executed = [n for n in selected if n not in reused_nodes]
    report = {**ENV, "exit_code": 0, "problems": [], "source_changed_during_suite": False, "full_suite_passed": False,
              "counts": {"passed": len(executed), "error": 0, "failed": 0, "skipped": 0},
              "collected_total": len(selected), "source_sha256": {"s4d/a.py": "candidate"},
              "reuse": {"reused_files": [{"file": f, "tests": 1} for f in reused], "error": None}}
    if mutate:
        mutate(report, selection, executed)
    return gate.verdict(report, repo=repo, baselines=baseline.store, current_tests=gate.test_impact.suite_files(tree),
                        base=base, executed=executed, selection=selection)


def test_selective_run_is_accepted_with_executed_reused_and_carried_tests(repo, baseline):
    write(repo, "s4d/a.py", "A = 2\n")
    verdict = run(repo, baseline, reused=("tests/test_scope.py",))
    assert verdict["accepted"] and verdict["mode"] == "selective" and not verdict["reasons"], verdict
    assert verdict["baseline"]["commit"] == baseline.commit
    # test_a ran, the scope guard's pass was reused, test_b's baseline pass was carried (its closure is unchanged)
    assert (verdict["executed_tests"], verdict["reused_tests"], verdict["carried_tests"]) == (1, 1, 1)


def test_inert_documentation_and_deleted_tests_are_covered(repo, baseline):
    write(repo, "docs/notes.md", "changed\n")
    write(repo, "s4d/a.py", "A = 3\n")
    (repo / "tests/test_b.py").unlink()
    verdict = run(repo, baseline)
    assert verdict["accepted"] and verdict["mode"] == "selective", verdict["reasons"]


@pytest.mark.parametrize("path,reason", [
    ("ops/robot_pc/executor.py", "high-risk"), ("scripts/run_tests.py", "high-risk"), ("uv.lock", "high-risk"),
    ("tests/conftest.py", "high-risk"),
    ("s4d/new.py", "unclassified"),              # only a scope guard's whole tree contains it
    ("runs/setup/tools/one_off.py", "unclassified"),
    ("tests/helper.py", "widened"),               # an unreferenced test-side file
    ("configs/x.yaml", "widened"),                # configuration data no test names
])
def test_high_risk_unclassified_or_widened_changes_require_the_complete_suite(repo, baseline, path, reason):
    write(repo, "s4d/a.py", "A = 4\n")
    write(repo, path, "changed = 1\n")
    verdict = run(repo, baseline)
    assert not verdict["accepted"] and verdict["mode"] is None
    assert any(reason in r for r in verdict["reasons"]), verdict["reasons"]


@pytest.mark.parametrize("mutate,reason", [
    (lambda r, s, e: r.update(exit_code=1), "failed"),
    (lambda r, s, e: r["counts"].update(failed=1), "did not all pass"),
    (lambda r, s, e: r.update(source_changed_during_suite=True), "sources changed"),
    (lambda r, s, e: r["reuse"].update(error="OSError: cache"), "reuse failed"),
    (lambda r, s, e: r.update(collected_total=r["collected_total"] + 1), "neither executed nor reused"),
    (lambda r, s, e: r.update(partition="remote"), "unpartitioned local"),
    (lambda r, s, e: r.update(host="remote"), "unpartitioned local"),
    (lambda r, s, e: r.update(torch="2.11.0"), "environment"),
    (lambda r, s, e: s.update(base="other"), "no change-impact selection"),
])
def test_run_failures_and_mismatches_refuse_the_selective_mode(repo, baseline, mutate, reason):
    write(repo, "s4d/a.py", "A = 5\n")
    verdict = run(repo, baseline, mutate=mutate)
    assert not verdict["accepted"] and any(reason in r for r in verdict["reasons"]), verdict["reasons"]


def test_the_baseline_must_be_recorded_untampered_and_an_ancestor(repo, baseline, tmp_path):
    write(repo, "s4d/a.py", "A = 6\n")
    missing = gate.Baselines(tmp_path / "elsewhere")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "candidate")
    assert any("no accepted full-suite baseline" in r for r in
               run(repo, SimpleNamespace(commit=baseline.commit, store=missing), commit=False)["reasons"])
    record = json.loads(baseline.store.path(baseline.commit).read_text())
    baseline.store.path(baseline.commit).write_text(json.dumps({**record, "source_digest": "0" * 64}))
    assert any("does not match the baseline commit" in r for r in run(repo, baseline, commit=False)["reasons"])
    baseline.store.path(baseline.commit).write_text(json.dumps(record))
    # a baseline on a side branch is no ancestor of the candidate
    git(repo, "checkout", "-qb", "side", baseline.commit)
    write(repo, "s4d/b.py", "B = 2\n")
    git(repo, "commit", "-qam", "side")
    side = git(repo, "rev-parse", "HEAD")
    side_report = {**baseline.report, "source_sha256": gate.commit_sources(repo, side)}
    baseline.store.record(side_report, baseline.ids, side, repo)
    git(repo, "checkout", "-q", "main")
    verdict = run(repo, SimpleNamespace(commit=side, store=baseline.store), commit=False, base=side)
    assert any("not an ancestor" in r for r in verdict["reasons"])
    # an unselected test file without a baseline pass cannot be carried
    record["nodes"].pop("tests/test_b.py")
    baseline.store.path(baseline.commit).write_text(json.dumps(record))
    assert any("has no baseline pass" in r for r in run(repo, baseline, commit=False)["reasons"])


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(full_suite_passed=False), lambda r: r["counts"].update(skipped=1),
    lambda r: r.update(partition="local"), lambda r: r.update(host="remote"),
    lambda r: r.update(source_changed_during_suite=None), lambda r: r.update(reuse={"reused_tests": 3}),
    lambda r: r.update(complete_suite_executed=False), lambda r: r.update(source_sha256={"other": "x"}),
])
def test_only_complete_passing_local_runs_of_the_exact_commit_become_baselines(repo, baseline, mutate, tmp_path):
    report = json.loads(json.dumps(baseline.report))
    mutate(report)
    with pytest.raises(ValueError, match="not an accepted baseline"):
        gate.Baselines(tmp_path / "other").record(report, baseline.ids, baseline.commit, repo)
    with pytest.raises(ValueError, match="node ids"):
        gate.Baselines(tmp_path / "other").record(baseline.report, baseline.ids[:1], baseline.commit, repo)


def test_a_run_without_a_baseline_is_judged_as_a_complete_suite(repo):
    for complete in (True, False):
        verdict = gate.verdict({"full_suite_passed": complete}, repo=repo, baselines=None, current_tests=[], base=None,
                               executed=[])
        assert verdict["accepted"] is complete and verdict["mode"] == ("full" if complete else None)


SOURCES = {"s4d/a.py": "1"}


def evidence(**over):
    report = {"full_suite_passed": True, "exit_code": 0, "source_sha256": SOURCES,
              "counts": {"passed": 5, "error": 0, "failed": 0, "skipped": 0}}
    report.update(over)
    return report


@pytest.mark.parametrize("report,expected", [
    (evidence(), (True, "full")),  # before the gate: a complete suite exactly as before
    (evidence(complete_suite_executed=True, release_gate={"accepted": True, "mode": "full"}), (True, "full")),
    (evidence(complete_suite_executed=False, release_gate={"accepted": True, "mode": "selective"}),
     (True, "selective")),
    (evidence(complete_suite_executed=False, release_gate={"accepted": False, "mode": None, "reasons": ["x"]}),
     (False, None)),
    (evidence(complete_suite_executed=True, release_gate={"accepted": True, "mode": "selective"}), (False, None)),
    (evidence(complete_suite_executed=False, release_gate={"accepted": True, "mode": "full"}), (False, None)),
    (evidence(full_suite_passed=False), (False, None)),
    (evidence(counts={"passed": 5, "error": 0, "failed": 1, "skipped": 0}), (False, None)),
    (evidence(source_sha256={"s4d/a.py": "stale"}), (False, None)),
])
def test_acceptance_function_for_adoption_and_release_checks(report, expected):
    ok, mode, reasons = gate.accepted(report, SOURCES)
    assert (ok, mode) == expected and bool(reasons) is not ok


def test_check_command_exit_status(tmp_path):
    for report, code in ((evidence(), 0), (evidence(exit_code=1), 1)):
        (tmp_path / "tests.json").write_text(json.dumps(report))
        assert gate.main(["check", "--evidence", str(tmp_path), "--any-sources"]) == code


def test_runner_gate_errors_never_accept_a_selective_run(monkeypatch, tmp_path):
    import test_gate as runner_gate  # the module object the runner imports

    state = runner.ImpactState(SimpleNamespace(reuse=None, changed="main"), True, tmp_path, lambda m: None)

    def broken(*args, **kwargs):
        raise RuntimeError("graph")

    monkeypatch.setattr(runner_gate, "verdict", broken)
    monkeypatch.setattr(state, "graph", lambda: SimpleNamespace(tree=SimpleNamespace(files=[])))
    verdict = state.gate({"full_suite_passed": False}, [], [], tmp_path)
    assert verdict["accepted"] is False and verdict["mode"] is None and "graph" in verdict["reasons"][0]
    assert state.gate({"full_suite_passed": True}, [], [], tmp_path)["mode"] == "full"
