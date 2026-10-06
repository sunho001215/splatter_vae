"""Run every test with a guarded GPU and persist honest pass/error evidence.

Native tests are never skipped. Source/JIT builds remain disabled by the renderer.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, source_fingerprints

GPU_MAPPING = guard_gpus()

import pytest  # noqa: E402
import torch  # noqa: E402


class Evidence:
    def __init__(self):
        self.counts = Counter()
        self.problems = []
        self.functions = defaultdict(set)

    def pytest_runtest_logreport(self, report):
        if report.failed:
            kind = "failed" if report.when == "call" else "error"
            self.counts[kind] += 1
            self.problems.append({"test": report.nodeid, "phase": report.when, "error": str(report.longrepr)})
        elif report.when == "call" and report.passed:
            self.counts["passed"] += 1
        elif report.skipped:
            self.counts["skipped"] += 1

    def trace(self, frame, event, arg):
        if event == "call":
            path = frame.f_code.co_filename
            prefix = str(REPO) + "/"
            if path.startswith(prefix):
                relative = path[len(prefix) :]
                if relative.startswith(("s4d/", "scripts/")):
                    self.functions[relative].add(frame.f_code.co_name)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def main() -> int:
    evidence = Evidence()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    args = ["tests", "-q", "--tb=short", f"--junitxml={REPO / 'docs/tests.junit.xml'}"]
    stdout, stderr = sys.stdout, sys.stderr
    with (REPO / "docs/tests.log").open("w") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        sys.setprofile(evidence.trace)
        try:
            code = int(pytest.main(args, plugins=[evidence]))
        finally:
            sys.setprofile(None)
            sys.stdout, sys.stderr = stdout, stderr
    fingerprints = source_fingerprints()
    report = {
        "started_utc": started,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_mapping": GPU_MAPPING,
        "pytest_args": args,
        "exit_code": code,
        "duration_seconds": time.perf_counter() - clock,
        "counts": {kind: evidence.counts[kind] for kind in ("passed", "error", "failed", "skipped")},
        "total": sum(evidence.counts.values()),
        "full_suite_passed": code == 0 and not evidence.counts["skipped"],
        "source_sha256": fingerprints,
        "executed_functions": {path: sorted(names) for path, names in sorted(evidence.functions.items())},
        "problems": evidence.problems,
        "limitations": (
            "Function trace covers the test parent. Gloo child results are asserted by tests/test_ddp.py. "
            "Synthetic renders are not CUDA acceptance."
        ),
    }
    (REPO / "docs/tests.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("counts", "total", "exit_code", "full_suite_passed")}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
