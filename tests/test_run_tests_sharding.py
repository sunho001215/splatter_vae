"""Sharding/evidence invariants tested with tiny synthetic suites, never the release suite."""
from __future__ import annotations

import cProfile
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts import run_tests as runner
from scripts import test_shards as shards

REPO = Path(__file__).resolve().parents[1]


def outcome(node="test_tiny.py::test_ok", status="passed"):
    return {"nodeid": node, "completed": True,
            "phases": [{"when": phase, "outcome": status if phase == "call" else "passed"}
                       for phase in ("setup", "call", "teardown")]}


def result(records=None, code=0):
    records = [outcome()] if records is None else records
    counts = dict.fromkeys(runner.KINDS, 0)
    for record in records:
        for phase in record["phases"]:
            status, when = phase["outcome"], phase["when"]
            if status == "failed":
                counts["failed" if when == "call" else "error"] += 1
            elif status == "skipped" or when == "call":
                counts[status] += 1
    return {"outcomes": records, "counts": counts, "problems": [], "exit_code": code}


def test_reviewed_classification_covers_every_test_file():
    files = {str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_*.py")}
    assert set(shards.FILE_CLASSES) == files
    assert set(shards.FILE_CLASSES.values()) == {"cpu", "gpu", "serial"}
    assert shards.FILE_CLASSES["tests/test_ddp.py"] == "serial"
    assert shards.FILE_CLASSES["tests/test_rwlearn_trainers_gpu.py"] == "gpu"


def test_unknown_files_default_gpu_and_resource_regex_promotes_serial(tmp_path):
    assert shards.classify("test_new.py", tmp_path) == "gpu"
    (tmp_path / "test_new.py").write_text("import torch.distributed\n")
    assert shards.classify("test_new.py", tmp_path) == "serial"


def test_partition_exact_and_whole_files_with_one_serial_shard():
    nodes = [f"{file}::test_{i}" for file in shards.FILE_CLASSES for i in range(3)]
    plan = shards.partition(nodes, REPO, 10, 3)
    files = [file for shard in plan for file in shard["files"]]
    assert sorted(files) == sorted(shards.FILE_CLASSES)
    assert sum(s["tests"] for s in plan) == len(nodes)
    assert len([s for s in plan if s["class"] == "serial"]) == 1
    assert all(len({shards.classify(f, REPO) for f in s["files"]}) == 1 for s in plan)


def test_lpt_balances_by_durations_and_falls_back_to_counts(tmp_path):
    nodes = [f"test_{n}.py::test_ok" for n in range(5)]
    weights = dict(zip((n.split("::")[0] for n in nodes), (8, 7, 6, 5, 4)))
    plan = shards.partition(nodes, tmp_path, 1, 2, weights)
    assert sorted(s["estimated_seconds"] for s in plan) == [13, 17]
    nodes = [f"test_{n}.py::test_{i}" for n, count in enumerate((8, 7, 6, 5, 4)) for i in range(count)]
    assert sorted(s["tests"] for s in shards.partition(nodes, tmp_path, 1, 2, {})) == [13, 17]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "crash", "exit5", "skip", "failure",
                                      "error", "incomplete", "counts", "exit_mismatch", "phases"])
def test_completeness_fails_closed(mutation):
    payload, code = result(), 0
    if mutation == "missing":
        payload = result([])
    elif mutation == "duplicate":
        payload = result([outcome(), outcome()])
    elif mutation == "extra":
        payload = result([outcome(), outcome("test_other.py::test_extra")])
    elif mutation == "crash":
        payload, code = None, -9
    elif mutation == "exit5":
        payload, code = result(code=5), 5
    elif mutation in ("skip", "failure"):
        payload = result([outcome(status="skipped" if mutation == "skip" else "failed")])
    elif mutation == "error":
        item = outcome()
        item["phases"][0]["outcome"] = "failed"
        payload = result([item])
    elif mutation == "incomplete":
        payload["outcomes"][0]["completed"] = False
    elif mutation == "counts":
        payload["counts"]["passed"] += 1
    elif mutation == "exit_mismatch":
        code = 2
    elif mutation == "phases":
        payload["outcomes"][0]["phases"].pop()
    _, problems = runner.validate_completeness([outcome()["nodeid"]], [(code, payload)])
    assert problems


def test_completeness_accepts_exact_partition_and_rejects_bad_collection():
    counts, problems = runner.validate_completeness([outcome()["nodeid"]], [(0, result())])
    assert not problems and counts["passed"] == 1
    for nodes in ([], [outcome()["nodeid"]] * 2):
        assert runner.validate_completeness(nodes, [(0, result())])[1]


def test_targeted_run_never_passes_full_suite_gate():
    counts = result()["counts"]
    assert runner.full_suite_passed(targeted=False, code=0, counts=counts, problems=[])
    assert not runner.full_suite_passed(targeted=True, code=0, counts=counts, problems=[])


def test_junit_merge_valid_and_complete_and_missing_fails(tmp_path):
    inputs = []
    for i in range(2):
        path = tmp_path / f"{i}.xml"
        path.write_text(f'<testsuites><testsuite tests="1"><testcase name="test_{i}" time="1"/></testsuite></testsuites>')
        inputs.append(path)
    output = tmp_path / "merged.xml"
    assert not runner.merge_junit(inputs, output)
    root = ET.parse(output).getroot()
    assert root.tag == "testsuites" and len(list(root.iter("testcase"))) == 2
    assert {case.get("name") for case in root.iter("testcase")} == {"test_0", "test_1"}
    assert runner.merge_junit([tmp_path / "missing.xml"], output)
    inputs[0].write_text("<broken")
    assert runner.merge_junit(inputs, output)


def test_cprofile_filter_equals_original_python_trace_on_toy_package(tmp_path):
    functions = {}
    for folder in ("s4d", "scripts", "other", "s4d_similar"):
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / "toy.py"
        namespace = {}
        exec(compile("def outer():\n    def inner(): return len([1])\n    return inner()\n", str(path), "exec"), namespace)
        functions[folder] = namespace["outer"]
    old = {}
    def trace(frame, event, arg):
        if event == "call":
            path = frame.f_code.co_filename
            prefix = str(tmp_path) + "/"
            if path.startswith(prefix):
                relative = path[len(prefix):]
                if relative.startswith(("s4d/", "scripts/")):
                    old.setdefault(relative, set()).add(frame.f_code.co_name)
    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        for function in functions.values():
            function()
    finally:
        sys.setprofile(None)
    profile = cProfile.Profile()
    profile.enable()
    for function in functions.values():
        function()
    profile.disable()
    assert runner.profile_functions(profile, tmp_path) == {p: sorted(names) for p, names in old.items()}
    assert set(old) == {"s4d/toy.py", "scripts/toy.py"}
    if isinstance(previous, cProfile.Profile):
        previous.enable()
    else:
        sys.setprofile(previous)


@pytest.mark.parametrize("budget,concurrent,expected", [(32, 14, 2), (8, 4, 2), (1, 14, 1), (32, 1, 32)])
def test_thread_budget_split(monkeypatch, budget, concurrent, expected):
    monkeypatch.delenv("S4D_TEST_CPU_THREADS", raising=False)  # a suite worker inherits its runner's budget
    monkeypatch.setenv("S4D_VERIFICATION_CPU_THREADS", str(budget))
    assert runner.thread_budget() == budget
    assert runner.threads_per_shard(budget, concurrent) == expected


def test_thread_budget_affinity_fallback(monkeypatch):
    monkeypatch.delenv("S4D_TEST_CPU_THREADS", raising=False)  # a suite worker inherits its runner's budget
    monkeypatch.delenv("S4D_VERIFICATION_CPU_THREADS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 2, 3})
    assert runner.thread_budget() == 3


def synthetic_worker(tmp_path, source, collect=False, marker=()):
    test = tmp_path / "test_tiny.py"
    test.write_text(source)
    payload = tmp_path / "worker.json"
    spec = {"pytest_args": [str(test), *marker, "-q", "--basetemp", str(tmp_path / "temp"),
                            "-o", f"cache_dir={tmp_path / 'cache'}"], "json": str(payload)}
    if collect:
        spec["pytest_args"].append("--collect-only")
    else:
        spec["pytest_args"].append(f"--junitxml={tmp_path / 'junit.xml'}")
    code = (f"import sys; sys.path.insert(0, {str(REPO)!r}); from scripts.run_tests import execute_pytest; "
            f"raise SystemExit(execute_pytest({spec!r}, collect={collect!r}))")
    process = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                             cwd=tmp_path, timeout=60)
    return process, json.loads(payload.read_text())


@pytest.mark.parametrize("source,exit_code,kind", [("def test_ok(): assert True\n", 0, "passed"),
    ("def test_bad(): assert False\n", 1, "failed"),
    ("import pytest\ndef test_skip(): pytest.skip('intentional')\n", 0, "skipped"),
    ("import pytest\n@pytest.fixture\ndef bad(): raise RuntimeError('setup')\ndef test_bad(bad): pass\n", 1, "error")])
def test_synthetic_worker_reports_real_outcomes(tmp_path, source, exit_code, kind):
    process, payload = synthetic_worker(tmp_path, source)
    assert process.returncode == exit_code, process.stdout + process.stderr
    assert payload["counts"][kind] == 1
    assert len(payload["outcomes"]) == 1
    assert payload["outcomes"][0]["completed"]
    assert len(list(ET.parse(tmp_path / "junit.xml").iter("testcase"))) == 1


def test_synthetic_collection_and_collection_error(tmp_path):
    process, payload = synthetic_worker(tmp_path, "def test_one(): pass\ndef test_two(): pass\n", collect=True)
    assert process.returncode == 0 and len(payload["nodeids"]) == 2
    process, payload = synthetic_worker(tmp_path, "raise RuntimeError('bad collection')\n", collect=True)
    assert process.returncode != 0 and payload["problems"]


def test_process_timeout_kills_only_its_group(tmp_path):
    code, timed_out = runner.run_process([sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                                         os.environ.copy(), tmp_path / "log", 0.1, cwd=tmp_path)
    assert code == 124 and timed_out


def test_missing_or_corrupt_worker_json_is_not_adopted(tmp_path):
    path = tmp_path / "worker.json"
    assert runner.read_result(path) is None
    path.write_text("{partial")
    assert runner.read_result(path) is None
    path.write_text("[]")
    assert runner.read_result(path) is None


def test_synthetic_worker_exit_five_has_no_acceptance(tmp_path):
    process, payload = synthetic_worker(tmp_path, "# no tests\n")
    assert process.returncode == payload["exit_code"] == 5
    assert runner.validate_completeness(["test_tiny.py::test_expected"], [(5, payload)])[1]


def test_synthetic_worker_crash_leaves_no_json(tmp_path):
    test = tmp_path / "test_crash.py"
    test.write_text("import os\ndef test_crash(): os._exit(19)\n")
    payload = tmp_path / "worker.json"
    spec = {"pytest_args": [str(test), "-q", "-o", f"cache_dir={tmp_path / 'cache'}"], "json": str(payload)}
    code = (f"import sys; sys.path.insert(0, {str(REPO)!r}); from scripts.run_tests import execute_pytest; "
            f"execute_pytest({spec!r})")
    process = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, timeout=60,
                             capture_output=True, text=True)
    assert process.returncode == 19 and not payload.exists()
    assert runner.validate_completeness(["test_crash.py::test_crash"], [(19, runner.read_result(payload))])[1]


def test_cache_paths_are_unique_short_and_honor_short_configured_root():
    configured = Path("/runtime/cache")
    first = runner.shard_cache_root(configured, Path("/output/a"), "shard-0")
    assert configured in first.parents
    assert first != runner.shard_cache_root(configured, Path("/output/b"), "shard-0")
    assert first != runner.shard_cache_root(configured, Path("/output/a"), "shard-1")
    fallback = runner.shard_cache_root(Path("/" + "long" * 40), Path("/output/a"), "shard-0", Path("/repo"))
    assert Path("/repo/.cache") in fallback.parents
    assert len(str(fallback) + "/tmp/pymp-abcdefgh/listener-abcdefgh") < 108


def test_subtests_preserve_legacy_counts_without_duplicate_execution(tmp_path):
    source = ("import unittest\nclass TestNested(unittest.TestCase):\n"
              "    def test_parent(self):\n        for i in range(3):\n"
              "            with self.subTest(i=i): self.assertEqual(i, i)\n")
    process, payload = synthetic_worker(tmp_path, source)
    assert process.returncode == 0, process.stdout + process.stderr
    assert len(payload["outcomes"]) == 1
    record = payload["outcomes"][0]
    assert len(record["phases"]) == 3
    assert len(record["subtests"]) == 3
    assert payload["counts"]["passed"] == 4  # matches the former Evidence callback
    assert not runner.validate_completeness([record["nodeid"]], [(0, payload)])[1]


def test_subtest_skip_still_fails_closed(tmp_path):
    source = ("import unittest\nclass TestNested(unittest.TestCase):\n"
              "    def test_parent(self):\n        with self.subTest(i=0): self.skipTest('intentional')\n")
    process, payload = synthetic_worker(tmp_path, source)
    assert payload["counts"]["skipped"] > 0
    assert runner.validate_completeness([payload["outcomes"][0]["nodeid"]], [(process.returncode, payload)])[1]


def test_seeded_model_cache_is_private(tmp_path):
    source, destination = tmp_path / "shared", tmp_path / "private"
    weights = source / "torch/hub/checkpoints/pinned.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"verified-model")
    runner.seed_cache(source, destination)
    copied = destination / weights.relative_to(source)
    assert copied.read_bytes() == weights.read_bytes()
    copied.write_bytes(b"test-write")
    assert weights.read_bytes() == b"verified-model"


def test_release_suite_partitions_select_exactly_and_complement():
    """The remote excludes the union of both local-data markers; the local partition runs exactly that union."""
    collected = ["a.py::x", "a.py::y", "b.py::z", "c.py::w"]
    marked = ["a.py::y", "b.py::z"]
    mode, marker, selected, deselected, problems = runner.select_partition("remote", collected, marked)
    assert (mode, marker, selected, deselected, problems) == (
        "remote", ("-m", "not (metaworld_data or local_host_data)"),
        ["a.py::x", "c.py::w"], ["a.py::y", "b.py::z"], [])
    mode, marker, selected, deselected, problems = runner.select_partition("local", collected, marked)
    assert (mode, marker, selected, deselected, problems) == (
        "local", ("-m", "metaworld_data or local_host_data"),
        ["a.py::y", "b.py::z"], ["a.py::x", "c.py::w"], [])
    assert runner.select_partition("all", collected, marked)[1:4] == ((), collected, [])
    # nothing marked: the remote suite is the complete suite; an empty local partition fails closed
    assert runner.select_partition("remote", collected, [])[0] == "all"
    assert runner.select_partition("local", collected, [])[4]


def test_the_local_data_markers_are_registered_and_used():
    import subprocess
    import sys
    listed = subprocess.run([sys.executable, "-I", "-m", "pytest", "--markers", "-p", "no:cacheprovider"],
                            cwd=REPO, capture_output=True, text=True, timeout=120).stdout
    assert "@pytest.mark.metaworld_data:" in listed
    assert "@pytest.mark.local_host_data:" in listed
    assert "@pytest.mark.metaworld_data" in (REPO / "tests/test_baselines_snerl.py").read_text()
    assert "@pytest.mark.local_host_data" in (REPO / "tests/test_remote_refusal.py").read_text()


def test_real_pytest_collection_and_workers_apply_both_markers_as_a_union(tmp_path):
    source = ("import pytest\n"
              "def test_portable(): pass\n"
              "@pytest.mark.metaworld_data\ndef test_metaworld(): pass\n"
              "@pytest.mark.local_host_data\ndef test_host(): pass\n"
              "@pytest.mark.metaworld_data\n@pytest.mark.local_host_data\ndef test_both(): pass\n")
    process, collection = synthetic_worker(tmp_path, source, collect=True)
    assert process.returncode == 0, process.stdout + process.stderr
    process, marked = synthetic_worker(tmp_path, source, collect=True, marker=("-m", runner.MARKER))
    assert process.returncode == 0, process.stdout + process.stderr
    assert {n.rsplit("::", 1)[1] for n in marked["nodeids"]} == {"test_metaworld", "test_host", "test_both"}
    expected = {"all": {"test_portable", "test_metaworld", "test_host", "test_both"},
                "remote": {"test_portable"}, "local": {"test_metaworld", "test_host", "test_both"}}
    for mode, names in expected.items():
        _, marker, selected, deselected, problems = runner.select_partition(
            mode, collection["nodeids"], marked["nodeids"])
        assert not problems and {n.rsplit("::", 1)[1] for n in selected} == names
        assert set(selected).isdisjoint(deselected)
        assert set(selected) | set(deselected) == set(collection["nodeids"])
        process, payload = synthetic_worker(tmp_path, source, marker=marker)
        assert process.returncode == 0, process.stdout + process.stderr
        counts, problems = runner.validate_completeness(selected, [(process.returncode, payload)])
        assert not problems and counts["passed"] == len(names)


def test_only_the_local_host_cases_are_marked_in_the_affected_files(tmp_path):
    process = subprocess.run([
        sys.executable, "-I", "-m", "pytest", "tests/test_ops_layout.py", "tests/test_remote_refusal.py",
        "--collect-only", "-q", "-m", "local_host_data", "-p", "no:cacheprovider",
        "--basetemp", str(tmp_path / "collection"),
    ], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    assert {line for line in process.stdout.splitlines() if line.startswith("tests/")} == {
        "tests/test_ops_layout.py::test_argparse_tool_entry_points_keep_their_help[sync_wandb_remote]",
        "tests/test_remote_refusal.py::test_real_registry_replay_unchanged_for_every_job",
    }


def test_terminating_the_runner_kills_and_reaps_its_worker_sessions(tmp_path):
    """Codex review 2026-11-12: a release chain that fails elsewhere terminates the local runner; its workers (own
    sessions) must not be orphaned."""
    import threading
    import time
    done = {}
    thread = threading.Thread(target=lambda: done.update(result=runner.run_process(
        [sys.executable, "-c", "import time; time.sleep(120)"], dict(os.environ), tmp_path / "w.log", 300,
        cwd=tmp_path)))
    thread.start()
    for _ in range(100):
        if runner.ACTIVE_WORKERS:
            break
        time.sleep(0.05)
    (process,) = list(runner.ACTIVE_WORKERS)
    assert runner.terminate_workers() == 1
    thread.join(timeout=30)
    assert not thread.is_alive() and process.poll() is not None and not runner.ACTIVE_WORKERS


def test_thread_budget_prefers_the_passed_quota(monkeypatch):
    monkeypatch.setenv("S4D_TEST_CPU_THREADS", "8")
    monkeypatch.setenv("S4D_VERIFICATION_CPU_THREADS", "32")
    assert runner.thread_budget() == 8
    monkeypatch.delenv("S4D_TEST_CPU_THREADS")
    assert runner.thread_budget() == 32


def test_terminating_the_runner_also_kills_descendants_of_an_exited_leader(tmp_path):
    """Codex review 2026-11-12: a worker leader that exits on SIGTERM must not leave its group's descendants."""
    import threading
    import time
    marker = tmp_path / "child.pid"
    script = ("import os, signal, subprocess, sys, time\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import signal, time; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)'])\n"
              f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
              "signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
              "time.sleep(120)\n")
    thread = threading.Thread(target=lambda: runner.run_process([sys.executable, "-c", script], dict(os.environ),
                                                                tmp_path / "w.log", 300, cwd=tmp_path))
    thread.start()
    for _ in range(200):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.05)
    child = int(marker.read_text())
    runner.terminate_workers()
    thread.join(timeout=30)
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the SIGTERM-immune descendant survived")
