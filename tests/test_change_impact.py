"""Change-impact selection and per-file result reuse (scripts/test_impact.py, scripts/run_tests.py; 2026-11-17).

Synthetic git repositories only (plus read-only graph queries of this checkout); no suite is run."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_tests as runner
from scripts import test_impact as impact

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=Impact Test", "-c", "user.email=impact@example.invalid", "-c", "commit.gpgsign=false"]

FILES = {
    "s4d/__init__.py": "",
    "s4d/a.py": "from s4d import b\n\ndef lazy():\n    import s4d.c\n    return s4d.c\n",
    "s4d/b.py": "VALUE = 1\n",
    "s4d/c.py": "VALUE = 2\n",
    "s4d/d.py": "VALUE = 3\n",
    "s4d/e.py": 'CONFIG = "configs/x.yaml"\nNOTES = "docs/notes.md"\nOUTPUT = "runs/out.json"\n',
    "configs/x.yaml": "a: 1\n",
    "configs/tree/y.yaml": "b: 2\n",
    "configs/unreferenced.yaml": "c: 3\n",
    "docs/notes.md": "notes\n",
    "runs/out.json": "{}\n",
    "tests/conftest.py": "from s4d import a  # noqa: F401\n",
    "tests/fixtures/f.json": "{}\n",
    "tests/helper_unused.py": "",
    "tests/test_one.py": (
        "import importlib\nimport sys\nfrom pathlib import Path\n\nimport s4d.e\n\n"
        "REPO = Path(__file__).resolve().parents[1]\nSHARDS = {'tests/test_two.py': 'cpu'}\n"
        "sys.path.insert(0, str(REPO / 'docs'))  # an import root, not an input\n\n"
        "def test_one():\n    importlib.import_module('s4d.d')\n    sys.modules.get('tests.test_two')\n"
        "    assert (REPO / 'configs' / 'tree').is_dir() and (Path(__file__).parent / 'fixtures').is_dir()\n"
        "    import s4d.gone  # noqa: F401\n"),
    "tests/test_two.py": "def test_two():\n    pass\n",
    "tests/test_three.py": "import s4d.a\n\ndef test_three():\n    assert s4d.a.lazy().VALUE == 2\n",
    "tests/test_helper_user.py": "from test_three import test_three as helper\n\ndef test_user():\n    helper()\n",
    "tests/test_scope.py": "from ops import migration\n\ndef test_scope():\n    migration.execution_scope\n",
    "tests/test_data.py": "import pytest\n\n@pytest.mark.local_host_data\ndef test_data():\n    pass\n",
    "ops/__init__.py": "",
    "ops/migration.py": "",
}


def git(root: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git(root, "init", "-q")
    git(root, "symbolic-ref", "HEAD", "refs/heads/main")  # (init -b needs git >= 2.28)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    return root


def closure(root: Path, test_file: str) -> set[str]:
    return impact.Graph(impact.Tree(root)).closure(test_file).files


def selected(root: Path, *changed: str) -> dict:
    return impact.select(impact.Graph(impact.Tree(root)), list(changed))


def test_graph_follows_imports_strings_and_test_side_paths_but_not_test_names(repo):
    one = closure(repo, "tests/test_one.py")
    assert {"s4d/e.py", "s4d/d.py", "configs/x.yaml", "configs/tree/y.yaml", "tests/fixtures/f.json",
            "tests/conftest.py", "s4d/a.py", "s4d/b.py", "s4d/__init__.py"} <= one
    # naming a test module (shard tables, sys.modules lookups) is not a dependency, nor is a sys.path entry; library
    # literals naming docs or run records are not inputs; a library's lazy import reached only through the conftest
    # does not run per test
    assert not {"tests/test_two.py", "docs/notes.md", "runs/out.json", "s4d/c.py"} & one
    assert "s4d/c.py" in closure(repo, "tests/test_three.py")  # a test importing the module may call the lazy path
    assert closure(repo, "tests/test_two.py") == {"tests/test_two.py", "tests/conftest.py", "s4d/__init__.py",
                                                  "s4d/a.py", "s4d/b.py"}
    scope = impact.Graph(impact.Tree(repo)).closure("tests/test_scope.py")  # an execution-scope guard: whole trees
    assert scope.scope_guard and {"s4d/c.py", "s4d/d.py", "configs/unreferenced.yaml"} <= scope.guard_files


@pytest.mark.parametrize("changed,expected", [
    (["s4d/b.py"], "all"),                                   # import-time member of the conftest closure
    (["s4d/c.py"], {"tests/test_three.py", "tests/test_helper_user.py", "tests/test_scope.py"}),
    (["configs/tree/new.yaml"], {"tests/test_one.py", "tests/test_scope.py"}),  # a referenced directory
    (["s4d/gone.py"], {"tests/test_one.py", "tests/test_scope.py"}),            # a deleted/missing imported module
    (["tests/test_two.py"], {"tests/test_two.py"}),
    (["tests/test_three.py"], {"tests/test_three.py", "tests/test_helper_user.py"}),  # imported as a helper
    (["docs/notes.md"], set()),                              # unattributed documentation
    (["runs/out.json", "docs/other.md"], set()),
    (["configs/unreferenced.yaml"], "all"),                  # widen: only a scope guard's whole tree names it
    (["configs/new-unreferenced.yaml"], "all"),              # widen: unattributed configuration data
    (["tests/helper_unused.py"], "all"),                     # widen: anything unattributed under tests/
    (["tests/conftest.py"], "all"), (["uv.lock"], "all"), (["third_party/x"], "all"),
    (["ops/robot_pc/safety.py"], "all"), (["ops/containers/launch.py"], "all"),  # protected paths
    (["scripts/test_shards.py"], {"tests/test_scope.py"}),   # not global: scheduling only (scope guards read scripts/)
])
def test_selection_rules(repo, changed, expected):
    plan = selected(repo, *changed)
    tests = set(impact.suite_files(impact.Tree(repo)))
    assert set(plan["files"]) == (tests if expected == "all" else expected)
    assert {p for p in changed if p in tests} <= set(plan["files"])  # a changed test file always selects itself
    assert plan["all"] is (expected == "all")
    if expected == set():
        assert plan["unattributed"] == changed


def test_test_file_changes_select_the_inventory_tests_and_report_deletions(repo):
    (repo / "tests/test_run_tests_sharding.py").write_text("def test_inventory():\n    pass\n")
    plan = selected(repo, "tests/test_two.py", "tests/test_removed.py")
    assert plan["files"] == ["tests/test_run_tests_sharding.py", "tests/test_two.py"]
    assert plan["deleted_tests"] == ["tests/test_removed.py"]
    assert "inventory:tests/test_removed.py" in plan["reasons"]["tests/test_run_tests_sharding.py"]


def test_changed_files_are_the_branch_since_the_merge_base_plus_the_working_tree(repo):
    git(repo, "checkout", "-qb", "feature")
    (repo / "s4d/b.py").write_text("VALUE = 10\n")
    git(repo, "commit", "-qam", "branch change")
    git(repo, "checkout", "-q", "main")
    (repo / "s4d/d.py").write_text("VALUE = 30\n")
    git(repo, "commit", "-qam", "main moved on")  # not on the branch: excluded by BASE...HEAD
    git(repo, "checkout", "-q", "feature")
    (repo / "s4d/c.py").write_text("VALUE = 20\n")                          # unstaged
    (repo / "configs/x.yaml").write_text("a: 2\n")
    git(repo, "add", "configs/x.yaml")                                       # staged
    (repo / "s4d/new.py").write_text("")                                     # untracked
    assert impact.changed_files("main", repo) == ["configs/x.yaml", "s4d/b.py", "s4d/c.py", "s4d/new.py"]
    assert impact.changed_files("main", repo, head="feature", worktree=False) == ["s4d/b.py"]
    committed = impact.CommitTree(repo, "feature")
    assert committed.read("s4d/b.py") == b"VALUE = 10\n" and committed.read("s4d/new.py") is None


def key(root: Path, test_file: str, environment=None) -> str | None:
    graph = impact.Graph(impact.Tree(root))
    return impact.reuse_key(graph, impact.Hasher(graph.tree), test_file, environment or {"python": "x"}, {})[0]


def test_reuse_key_follows_exactly_the_dependencies_and_the_environment(repo, tmp_path):
    before = key(repo, "tests/test_one.py")
    (repo / "docs/notes.md").write_text("changed\n")
    (repo / "s4d/c.py").write_text("VALUE = 22\n")
    assert key(repo, "tests/test_one.py") == before  # neither is a dependency of test_one
    assert key(repo, "tests/test_one.py", {"python": "y"}) != before
    (repo / "configs/tree/y.yaml").write_text("b: 3\n")
    changed = key(repo, "tests/test_one.py")
    assert changed != before
    (repo / "tests/fixtures/new.json").write_text("{}\n")  # an untracked file in a referenced directory
    assert key(repo, "tests/test_one.py") not in (before, changed)
    assert key(repo, "tests/test_data.py") is None  # live-data markers: never reused
    external = tmp_path / "external.json"
    external.write_text("1")
    (repo / "tests/test_two.py").write_text(f"PATH = {str(external)!r}\n\ndef test_two():\n    pass\n")
    first = key(repo, "tests/test_two.py")
    external.write_text("2")
    assert key(repo, "tests/test_two.py") != first


def outcome(node, call="passed", subtests=()):
    return {"nodeid": node, "completed": True, "subtests": [{"when": "call", "outcome": s} for s in subtests],
            "phases": [{"when": "setup", "outcome": "passed"}, {"when": "call", "outcome": call},
                       {"when": "teardown", "outcome": "passed"}]}


def test_only_complete_passes_are_recorded_and_reused_as_subsets(tmp_path):
    cache = impact.ResultCache(tmp_path / "cache")
    nodes = ["tests/a.py::t1", "tests/a.py::t2", "tests/b.py::t1", "tests/c.py::t1", "tests/d.py::t1"]
    outcomes = [outcome(nodes[0]), outcome(nodes[1]), outcome(nodes[2], "failed"), outcome(nodes[3], "skipped"),
                outcome(nodes[4], subtests=["failed"])]
    keys = {f"tests/{n}.py": n * 64 for n in "abcd"}
    assert impact.passed_nodes(outcomes) == nodes[:2]
    assert impact.record_passes(cache, keys, nodes, outcomes, commit="c") == ["tests/a.py"]
    entry = cache.lookup("a" * 64, "tests/a.py")
    assert entry["passed"] == nodes[:2] and entry["commit"] == "c"
    assert cache.lookup("a" * 64, "tests/b.py") is None  # a key names exactly one file
    cache.path("b" * 64).parent.mkdir(parents=True, exist_ok=True)
    cache.path("b" * 64).write_text("{corrupt")
    assert cache.lookup("b" * 64, "tests/b.py") is None
    cache.record("a" * 64, "tests/a.py", ["tests/a.py::t3"])  # a later pass of the same key adds node ids
    assert cache.lookup("a" * 64, "tests/a.py")["passed"] == [*nodes[:2], "tests/a.py::t3"]


def test_plan_reuse_skips_whole_unchanged_files_and_keeps_collection_order(repo, tmp_path):
    graph = impact.Graph(impact.Tree(repo))
    cache = impact.ResultCache(tmp_path / "cache")
    nodes = ["tests/test_two.py::test_two", "tests/test_one.py::test_one", "tests/test_data.py::test_data",
             "tests/test_three.py::test_three"]
    first = impact.plan_reuse(graph, cache, nodes, {"python": "x"}, {})
    assert first["run"] == nodes and not first["reused"]
    assert first["keys"]["tests/test_data.py"] is None and "live-data" in first["ineligible"]["tests/test_data.py"]
    impact.record_passes(cache, first["keys"], nodes, [outcome(n) for n in nodes])
    second = impact.plan_reuse(graph, cache, nodes + ["tests/test_two.py::test_new"], {"python": "x"}, {})
    # a file with a node id not recorded as passed runs whole
    assert second["run"] == ["tests/test_two.py::test_two", "tests/test_data.py::test_data", "tests/test_two.py::test_new"]
    assert [r["file"] for r in second["reused"]] == ["tests/test_one.py", "tests/test_three.py"]
    assert impact.plan_reuse(graph, cache, nodes, {"python": "x"}, {}, lookup=False)["run"] == nodes
    assert impact.plan_reuse(graph, cache, nodes, {"python": "other"}, {})["reused"] == []


@pytest.mark.parametrize("reused,executed,expected", [
    (0, 10, (True, False, True)),    # the complete suite, nothing reused: unchanged acceptance
    (4, 6, (False, True, False)),    # reuse never grants full-suite (or partition) acceptance
    (4, 5, (False, False, False)),   # executed + reused must cover the whole collection
])
def test_reuse_never_grants_full_suite_acceptance(reused, executed, expected):
    counts = {"passed": executed, "failed": 0, "error": 0, "skipped": 0}
    result = runner.suite_acceptance(mode="all", targeted=False, code=0, counts=counts, problems=[],
                                     executed=executed, reused_tests=reused, selected=10, collected=10)
    assert result == expected
    targeted = runner.suite_acceptance(mode="all", targeted=True, code=0, counts=counts, problems=[],
                                       executed=executed, reused_tests=reused, selected=10, collected=10)
    assert targeted == (False, False, False)


def test_reuse_failures_leave_the_run_unchanged(monkeypatch, tmp_path):
    import test_impact as runner_impact  # the module object the runner imports (scripts/ is on its sys.path)

    options = SimpleNamespace(reuse=None, changed=None)
    state = runner.ImpactState(options, True, tmp_path, lambda message: None)

    def broken(*args, **kwargs):
        raise OSError("cache unavailable")

    monkeypatch.setattr(runner_impact, "cache_root", broken)
    nodes = ["tests/test_geometry.py::test_x"]
    assert state.plan(nodes, [], "local") == nodes
    assert "cache unavailable" in state.summary()["error"] and not state.reused
    state.record(nodes, [{"outcomes": [outcome(nodes[0])]}])  # no cache: nothing recorded, nothing raised
    assert state.summary()["recorded_files"] == 0


def test_result_cache_stays_inside_a_checkout_cache(monkeypatch, tmp_path):
    from ops.resources.gpu_guard import GPUIsolationError

    monkeypatch.delenv("S4D_TEST_RESULT_CACHE", raising=False)
    assert impact.cache_root(REPO) == (REPO / ".cache/test-results").resolve()
    monkeypatch.setenv("S4D_TEST_RESULT_CACHE", str(REPO / ".cache/test-results-other"))
    assert impact.cache_root(REPO) == (REPO / ".cache/test-results-other").resolve()
    for outside in (REPO / "docs/test-results", Path("/home/ws/data/test-results")):
        monkeypatch.setenv("S4D_TEST_RESULT_CACHE", str(outside))
        with pytest.raises(GPUIsolationError):
            impact.cache_root(REPO)


def test_runner_subsets_are_exclusive():
    process = subprocess.run([sys.executable, "-I", str(REPO / "scripts/run_tests.py"), "--files", "tests/x.py",
                              "--changed", "HEAD"], capture_output=True, text=True, timeout=120)
    assert process.returncode == 2 and "not allowed with argument" in process.stderr


def test_this_checkout_graph_attributes_known_dependencies():
    graph = impact.Graph(impact.Tree(REPO))
    geometry = graph.closure("tests/test_geometry.py")
    assert {"tests/test_geometry.py", "s4d/geometry.py", "tests/conftest.py", "ops/resources/gpu_guard.py"} <= \
        geometry.files
    assert not geometry.scope_guard and "s4d/rl/agent.py" not in geometry.files
    assert graph.closure("tests/test_memory_admission.py").scope_guard  # an execution-scope guard
    assert graph.closure("tests/test_adoption_guards.py").scope_guard  # imports ops.jobs.closure_impact
    assert graph.closure("tests/test_remote_refusal.py").live_data  # @pytest.mark.local_host_data
    # names inside strings (this file's synthetic sources) neither mark live data nor make a scope guard
    this = graph.closure("tests/test_change_impact.py")
    assert not this.live_data and not this.scope_guard
    note = "docs/never-read-by-a-test.md"  # (an existing path named here would become this file's own input)
    plan = impact.select(graph, ["s4d/geometry.py", note])
    assert "tests/test_geometry.py" in plan["files"] and not plan["all"]
    assert plan["unattributed"] == [note]


def test_no_test_module_shadows_a_test_function():
    """A second definition of a test name silently replaces the first, which then never runs (test audit 2026-11-17
    found one in tests/test_host_roles.py)."""
    shadowed = {}
    for path in sorted((REPO / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        scopes = [tree.body] + [n.body for n in tree.body if isinstance(n, ast.ClassDef)]
        for body in scopes:
            names = Counter(n.name for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and n.name.startswith("test"))
            for name, count in names.items():
                if count > 1:
                    shadowed[f"{path.name}::{name}"] = count
    assert not shadowed, shadowed


def test_dry_run_cli_prints_the_selection(repo, monkeypatch, capsys):
    monkeypatch.setattr(impact, "REPO", repo)
    (repo / "s4d/c.py").write_text("VALUE = 5\n")
    assert impact.main(["--base", "HEAD", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["files"] == ["tests/test_helper_user.py", "tests/test_scope.py", "tests/test_three.py"]
    assert plan["changed"] == ["s4d/c.py"]


def test_subset_runs_refuse_to_start_from_another_checkout(repo, tmp_path):
    """2026-11-18: a --changed run started from the wrong working directory judged and rewrote another worktree's
    evidence. The runner names its checkout and refuses a working directory inside a different one."""
    assert runner.foreign_checkout(REPO, cwd=REPO) is None
    assert runner.foreign_checkout(REPO, cwd=Path(REPO, "tests")) is None
    assert runner.foreign_checkout(REPO, cwd=repo) == repo.resolve()
    assert runner.foreign_checkout(REPO, cwd="/") is None  # outside any checkout: the runner's own tree is unambiguous
    evidence = Path(REPO, "docs", "tests.json")  # (separate segments: no path literal of this checkout's docs tree)
    before = evidence.stat().st_mtime_ns if evidence.exists() else None
    process = subprocess.run([sys.executable, "-I", str(REPO / "scripts/run_tests.py"), "--changed", "HEAD", "--plan"],
                             cwd=repo, capture_output=True, text=True, timeout=120)
    assert process.returncode == 2 and "runs from the checkout" in process.stderr, process.stderr
    assert not list(repo.rglob("tests.json")) and not list(repo.rglob("tests.log"))  # nothing written in either tree
    assert (evidence.stat().st_mtime_ns if evidence.exists() else None) == before
