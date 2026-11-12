"""Run the complete guarded suite in isolated, duration-balanced pytest processes."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKER = "metaworld_data"  # tests reading the Meta-World development data (pyproject.toml markers)
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
KINDS = ("passed", "error", "failed", "skipped")


def profile_functions(profile, root=REPO):
    """Use exactly the former Python callback's filename and code-name filter."""
    functions = defaultdict(set)
    prefix = str(root) + "/"
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):  # builtins have no Python code object
            continue
        if code.co_filename.startswith(prefix):
            relative = code.co_filename[len(prefix):]
            if relative.startswith(("s4d/", "scripts/")):
                functions[relative].add(code.co_name)
    return {path: sorted(names) for path, names in sorted(functions.items())}


class Evidence:
    def __init__(self):
        self.counts = Counter()
        self.problems = []
        self.outcomes = []  # A list preserves repeated node IDs (a dict would hide them).

    def pytest_runtest_logstart(self, nodeid, location):
        self.outcomes.append({"nodeid": nodeid, "phases": [], "subtests": [], "completed": False})

    def pytest_runtest_logfinish(self, nodeid, location):
        self.outcomes[-1]["completed"] = True

    def pytest_runtest_logreport(self, report):
        # unittest subTest reports reuse their parent's nodeid. Retain legacy counts,
        # but do not mistake these nested assertions for additional test executions.
        key = "subtests" if hasattr(report, "context") else "phases"
        self.outcomes[-1][key].append({"when": report.when, "outcome": report.outcome})
        if report.failed:
            self.counts["failed" if report.when == "call" else "error"] += 1
        elif report.when == "call" and report.passed:
            self.counts["passed"] += 1
        elif report.skipped:
            self.counts["skipped"] += 1
        if report.failed or report.skipped:
            self.problems.append({"test": report.nodeid, "phase": report.when, "error": str(report.longrepr)})

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.problems.append({"phase": "collection", "test": report.nodeid, "error": str(report.longrepr)})


class Collection:
    def __init__(self):
        self.nodeids = []
        self.problems = []

    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]

    pytest_collectreport = Evidence.pytest_collectreport


def execute_pytest(spec, *, collect=False):
    """Worker implementation, also exercised against synthetic directories in tests."""
    import pytest

    clock = time.perf_counter()
    plugin = Collection() if collect else Evidence()
    profile = cProfile.Profile(subcalls=False, builtins=False)
    profile.enable()
    try:
        code = int(pytest.main(spec["pytest_args"], plugins=[plugin]))
    finally:
        profile.disable()
    result = {"exit_code": code, "problems": plugin.problems, "duration_seconds": time.perf_counter() - clock}
    if collect:
        result["nodeids"] = plugin.nodeids
    else:
        result.update(counts={kind: plugin.counts[kind] for kind in KINDS}, outcomes=plugin.outcomes,
                      executed_functions=profile_functions(profile))
    Path(spec["json"]).write_text(json.dumps(result, indent=2))
    return code


def thread_budget():
    # S4D_TEST_CPU_THREADS: set by the verification wrapper (container CPU quota) and the release chain's local
    # partition; S4D_VERIFICATION_CPU_THREADS: a direct verification run; otherwise the CPU affinity
    value = os.environ.get("S4D_TEST_CPU_THREADS") or os.environ.get("S4D_VERIFICATION_CPU_THREADS")
    budget = int(value) if value is not None else (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    if budget < 1:
        raise ValueError("Thread budget must be positive")
    return budget


def shard_cache_root(cache, output, name, repository=REPO):
    """Keep multiprocessing's filesystem socket names below Linux's 108-byte limit.

    Long external evidence roots use the repository's approved cache mount instead.
    The output-path hash prevents concurrent runs from sharing writable caches.
    """
    token = hashlib.sha256(str(output).encode()).hexdigest()[:8]
    leaf = {"collection": "c", "collection-marked": "m"}.get(name, name.removeprefix("shard-"))
    suffix = Path("ts") / token / leaf
    root = cache / suffix
    socket_suffix = "/tmp/pymp-abcdefgh/listener-abcdefgh"
    if len(os.fsencode(str(root) + socket_suffix)) >= 108:
        root = repository / ".cache" / suffix
    if len(os.fsencode(str(root) + socket_suffix)) >= 108:
        raise ValueError("Repository cache path is too long for multiprocessing Unix sockets")
    return root


def seed_cache(source, destination):
    """Copy pre-provisioned model inputs; every worker still owns its writable cache."""
    checkpoints = source / "torch" / "hub" / "checkpoints"
    if checkpoints.is_dir():
        shutil.copytree(checkpoints, destination / "torch" / "hub" / "checkpoints", dirs_exist_ok=True)


def threads_per_shard(budget, concurrent):
    return max(1, budget // max(1, concurrent))


ACTIVE_WORKERS = set()  # worker sessions (Popen) of this runner, terminated if the runner itself is terminated
_ACTIVE_LOCK = threading.Lock()


def terminate_workers():
    """Kill every live worker session (its whole process group) and reap the leaders."""
    with _ACTIVE_LOCK:
        workers = list(ACTIVE_WORKERS)
    for process in workers:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
    return len(workers)


def _terminated(signum, frame):
    terminate_workers()
    raise SystemExit(128 + signum)


def run_process(argv, env, log, timeout, *, cwd=REPO):
    """Bound the entire process group, including subprocesses left by a timed-out test."""
    with Path(log).open("w") as stream:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        with _ACTIVE_LOCK:
            ACTIVE_WORKERS.add(process)
        try:
            return _wait_bounded(process, timeout)
        finally:
            with _ACTIVE_LOCK:
                ACTIVE_WORKERS.discard(process)


def _wait_bounded(process, timeout):
    try:
        return process.wait(timeout=max(0.01, timeout)), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The group may still have grandchildren even after its leader exits.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return 124, True


def merge_junit(paths, destination):
    root = ET.Element("testsuites")
    problems = []
    for path in paths:
        try:
            child = ET.parse(path).getroot()
            if child.tag == "testsuite":
                root.append(child)
            elif child.tag == "testsuites" and all(s.tag == "testsuite" for s in child):
                root.extend(child)
            else:
                raise ValueError("Expected testsuite(s) root")
        except (OSError, ET.ParseError, ValueError) as exc:
            problems.append({"phase": "evidence", "error": f"Invalid/missing JUnit {path}: {exc}"})
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return problems


def validate_completeness(collected, results):
    """Validate independent collection, worker outcomes, phase accounting and OS exits."""
    problems, seen, counts = [], Counter(), Counter()
    if not collected or len(set(collected)) != len(collected):
        problems.append({"phase": "evidence", "error": "Empty or duplicate authoritative collection"})
    for index, (code, result) in enumerate(results):
        def problem(message, index=index):
            problems.append({"phase": "evidence", "shard": index, "error": message})
        if result is None:
            problem("Missing/invalid shard JSON (worker crashed)")
            continue
        try:
            local = Counter()
            outcomes = result["outcomes"]
            for outcome in outcomes:
                seen[outcome["nodeid"]] += 1
                phases = outcome["phases"]
                if outcome["completed"] is not True:
                    problem(f"Incomplete test {outcome['nodeid']}")
                for phase in [*phases, *outcome.get("subtests", [])]:
                    status, when = phase["outcome"], phase["when"]
                    if status not in ("passed", "failed", "skipped") or when not in ("setup", "call", "teardown"):
                        problem(f"Invalid phase outcome {outcome['nodeid']}")
                    if status == "failed":
                        local["failed" if when == "call" else "error"] += 1
                    elif status == "skipped":
                        local["skipped"] += 1
                    elif status == "passed" and when == "call":
                        local["passed"] += 1
                if not any(p["outcome"] != "passed" for p in phases):
                    if [p["when"] for p in phases] != ["setup", "call", "teardown"]:
                        problem(f"Missing/duplicate test phases {outcome['nodeid']}")
            if result["counts"] != {kind: local[kind] for kind in KINDS}:
                problem("Shard counts do not match phase outcomes")
            counts.update(local)
            problems.extend(result["problems"])
            if result["exit_code"] != code:
                problem("Worker JSON/OS exit codes differ")
            if code != 0:
                problem(f"Shard exit code {code}" + (" (no tests)" if code == 5 else ""))
            if any(local[kind] for kind in ("failed", "error", "skipped")):
                problem("Shard has failures, errors or skips")
        except (KeyError, TypeError, ValueError) as exc:
            problem(f"Invalid shard evidence: {exc}")
    expected = set(collected)
    for label, nodes in (("Missing", expected - seen.keys()), ("Extra", seen.keys() - expected),
                         ("Duplicate", {node for node, count in seen.items() if count != 1})):
        if nodes:
            problems.append({"phase": "evidence", "error": f"{label} nodeids", "tests": sorted(nodes)})
    return {kind: counts[kind] for kind in KINDS}, problems


def select_partition(mode, collected, marked):
    """(mode, pytest marker args, selected, deselected, problems) of a release-suite partition: 'all' runs the complete
    collection; 'remote' every test but the MARKER ones (the complete suite when none is marked); MARKER exactly those."""
    problems = []
    if mode != "all" and not marked:
        if mode == "remote":
            mode = "all"  # nothing needs the Meta-World data: the remote suite is the complete suite
        else:
            problems.append({"phase": "collection", "error": f"No {MARKER} tests to run in this partition"})
    marker = () if mode == "all" else ("-m", f"not {MARKER}") if mode == "remote" else ("-m", MARKER)
    marked_set = set(marked)
    selected = list(collected) if mode == "all" else [n for n in collected if (n in marked_set) == (mode == MARKER)]
    selected_set = set(selected)
    return mode, marker, selected, [n for n in collected if n not in selected_set], problems


def full_suite_passed(*, targeted, code, counts, problems):
    return (not targeted and code == 0 and not problems and counts["passed"] > 0
            and not any(counts[kind] for kind in ("failed", "error", "skipped")))


def read_result(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=os.environ.get("S4D_TEST_EVIDENCE_DIR", str(REPO / "docs")))
    ap.add_argument("--files", nargs="+", help="Development subset; never produces full-suite acceptance")
    ap.add_argument("--partition", choices=("all", "remote", "metaworld_data"),
                    help="default: remote on the remote host (no Meta-World development data there), all elsewhere")
    ap.add_argument("--shards-cpu", type=int)
    ap.add_argument("--shards-gpu", type=int)
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--collect", action="store_true", help=argparse.SUPPRESS)
    options = ap.parse_args()
    from ops.resources.gpu_guard import HOST_NAME, validate_runtime_path
    if not options.worker:
        output = validate_runtime_path(Path(options.output_dir))
        os.environ["S4D_TEST_EVIDENCE_DIR"] = str(output)
        os.environ["S4D_PYTEST_TMP"] = str(output / "tmp")
        if REPO not in output.parents:
            os.environ.setdefault("S4D_CACHE_ROOT", str(output / "cache"))
    from _bootstrap import guard_gpus, source_fingerprints

    gpu_mapping = guard_gpus()  # Parent, collection process and every worker fail closed.
    import torch

    if options.worker:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        return execute_pytest(json.loads(options.worker.read_text()), collect=options.collect)
    from test_shards import partition

    # A terminated runner (e.g. the release chain failing elsewhere) never leaves its worker sessions behind.
    signal.signal(signal.SIGTERM, _terminated)
    signal.signal(signal.SIGINT, _terminated)

    output = validate_runtime_path(Path(options.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    (output / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["S4D_TEST_EVIDENCE_DIR"] = str(output)
    os.environ["S4D_PYTEST_TMP"] = str(output / "tmp")
    cache = validate_runtime_path(Path(os.environ.get("S4D_CACHE_ROOT", str(
        REPO / ".cache" if REPO in output.parents else output / "cache"))), cache=True)
    budget = thread_budget()
    cpu = options.shards_cpu if options.shards_cpu is not None else max(1, min(10, budget // 3))
    gpu = options.shards_gpu if options.shards_gpu is not None else max(1, min(3, budget // 8))
    if cpu < 1 or gpu < 1:
        ap.error("Shard counts must be positive")
    # Leave the verification wrapper time to merge and persist reports before its hard timeout.
    total_timeout = float(os.environ.get("S4D_VERIFICATION_TIMEOUT_SECONDS", "6600"))
    timeout = max(0.01, total_timeout - min(60, total_timeout * 0.05))
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    source_before = source_fingerprints()
    problems = []
    log_path = output / "tests.log"
    log_path.write_text(f"Collecting authoritative suite; CPU budget={budget}, CPU shards={cpu}, GPU shards={gpu}\n")

    def log(message):
        print(message, flush=True)
        with log_path.open("a") as stream:
            stream.write(message + "\n")

    def invoke(name, files, threads, collect=False, marker=()):
        directory = output / name
        # Reusing an evidence directory cannot accidentally adopt a stale worker report.
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "shard.json"
        junit = directory / "tests.junit.xml"
        for path in (result_path, junit):
            path.unlink(missing_ok=True)
        args = [*files, *marker, "-q", "--tb=short", "--basetemp", str(output / "tmp" / name),
                "-o", f"cache_dir={directory / 'pytest-cache'}"]
        args += ["--collect-only"] if collect else [f"--junitxml={junit}"]
        spec = directory / "spec.json"
        spec.write_text(json.dumps({"pytest_args": args, "json": str(result_path)}))
        worker_cache = shard_cache_root(cache, output, name)
        seed_cache(cache, worker_cache)
        env = {**os.environ, "S4D_PYTEST_TMP": str(output / "tmp" / name),
               "S4D_CACHE_ROOT": str(worker_cache),
               **{key: str(threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        argv = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(spec)]
        if collect:
            argv.append("--collect")
        remaining = timeout - (time.perf_counter() - clock)
        try:
            code, timed_out = run_process(argv, env, directory / "tests.log", remaining)
        except OSError as exc:
            (directory / "tests.log").write_text(f"Worker launch failed: {exc}\n")
            code, timed_out = 127, False
        return code, read_result(result_path), args, timed_out

    collection_files = options.files or ["tests/"]
    collect_code, collection, _, timed_out = invoke("collection", collection_files, budget, True)
    collected = collection.get("nodeids", []) if collection else []
    (output / "collected.json").write_text(json.dumps(collected, indent=2))
    if collect_code or not collection or collection.get("problems") or timed_out:
        problems.append({"phase": "collection", "error": f"Collection failed (exit {collect_code})",
                         "details": collection})
    # Release-suite partitions (user 2026-11-12; ops/containers/suite_partition.py): the remote host has no Meta-World
    # development data, so its suite runs every test but the metaworld_data ones; the release chain runs exactly those
    # on the local host. Only the combination of both is a complete suite (full_suite_passed stays false here).
    mode = options.partition or ("remote" if HOST_NAME == "remote" else "all")
    marked = []
    if mode != "all" and not problems:
        marked_code, marked_result, _, marked_timeout = invoke("collection-marked", collection_files, budget, True,
                                                               marker=("-m", MARKER))
        marked = marked_result.get("nodeids", []) if marked_result else []
        if (marked_code not in (0, 5) or not marked_result or marked_result.get("problems") or marked_timeout
                or not set(marked) <= set(collected)):
            problems.append({"phase": "collection", "error": f"Marker collection failed (exit {marked_code})"})
    mode, marker, selected, deselected, selection_problems = select_partition(mode, collected, marked)
    problems.extend(selection_problems)
    shards = partition(selected, REPO, cpu, gpu) if not problems else []
    concurrent = len(shards)
    threads = threads_per_shard(budget, concurrent)
    log(f"Collected {len(collected)} tests; starting {concurrent} shards, {threads} threads each")

    def run(shard):
        index = shard["index"]
        start = time.perf_counter()
        result = invoke(f"shard-{index}", shard["files"], threads, marker=marker)
        log(f"Shard {index} ({shard['class']}): exit={result[0]}, seconds={time.perf_counter() - start:.1f}")
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrent)) as pool:
        executed = list(pool.map(run, shards))
    counts, completeness = validate_completeness(selected, [(r[0], r[1]) for r in executed])
    problems.extend(completeness)
    for shard, result in zip(shards, executed):
        shard["seconds"] = result[1].get("duration_seconds", 0) if result[1] else 0
        shard["exit_code"] = result[0]
        if result[3]:
            problems.append({"phase": "timeout", "shard": shard["index"], "error": "Shard process group timed out"})
    problems.extend(merge_junit([output / f"shard-{s['index']}" / "tests.junit.xml" for s in shards],
                                output / "tests.junit.xml"))
    with log_path.open("a") as stream:
        for name in ["collection", *(f"shard-{s['index']}" for s in shards)]:
            stream.write(f"\n===== {name} =====\n")
            stream.write((output / name / "tests.log").read_text())
    for shard, (_, result, _, _) in zip(shards, executed):
        junit = output / f"shard-{shard['index']}" / "tests.junit.xml"
        if result and junit.is_file():
            try:
                cases = list(ET.parse(junit).iter("testcase"))
                if len(cases) != len(result["outcomes"]):
                    problems.append({"phase": "evidence", "shard": shard["index"],
                                     "error": "JUnit testcase count differs from worker outcomes"})
            except (ET.ParseError, KeyError, TypeError):
                problems.append({"phase": "evidence", "shard": shard["index"], "error": "Invalid shard JUnit/outcomes"})
    functions = defaultdict(set)
    for _, result, _, _ in executed:
        if result:
            for path, names in result.get("executed_functions", {}).items():
                functions[path].update(names)
    fingerprints = source_fingerprints()
    changed = fingerprints != source_before
    if changed:
        problems.append({"phase": "evidence", "error": "Sources changed during the full suite."})
    code = 1 if problems else 0
    report = {
        "started_utc": started, "python": sys.version, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "host": HOST_NAME,
        "expected_commit": os.environ.get("S4D_EXPECTED_COMMIT"),
        "expected_image_digest": os.environ.get("S4D_IMAGE_DIGEST"), "gpu_mapping": gpu_mapping,
        "pytest_args": [r[2] for r in executed], "exit_code": code,
        "duration_seconds": time.perf_counter() - clock, "counts": counts, "total": sum(counts.values()),
        "collected_total": len(collected),
        "full_suite_passed": mode == "all" and full_suite_passed(targeted=options.files is not None, code=code,
                                                                 counts=counts, problems=problems),
        "source_sha256": fingerprints, "source_sha256_before": source_before,
        "source_changed_during_suite": changed,
        "executed_functions": {path: sorted(names) for path, names in sorted(functions.items())},
        "trace_method": "cprofile", "shards": shards,
        "parallel": {"cpu_shards": cpu, "gpu_shards": gpu, "serial_shards": int(any(
            s["class"] == "serial" for s in shards)), "concurrent_shards": concurrent,
            "cpu_threads": budget, "threads_per_shard": threads, "timeout_seconds": timeout},
        "problems": problems,
        **({} if mode == "all" else {
            "partition": mode, "partition_marker": MARKER, "selected": selected, "deselected": deselected,
            "partition_passed": full_suite_passed(targeted=options.files is not None, code=code, counts=counts,
                                                  problems=problems)}),
        "limitations": "Function evidence unions shard processes; nested test subprocesses are not profiled. "
                       "Gloo child results are asserted by tests/test_ddp.py. Synthetic renders are not CUDA acceptance.",
    }
    (output / "tests.json").write_text(json.dumps(report, indent=2))
    log(json.dumps({k: report[k] for k in ("counts", "total", "exit_code", "full_suite_passed")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
