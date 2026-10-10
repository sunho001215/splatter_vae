"""Run every test with a guarded GPU and persist honest pass/error evidence.

Native tests are never skipped. Source/JIT builds remain disabled by the renderer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=os.environ.get("S4D_TEST_EVIDENCE_DIR", str(REPO / "docs")))
    options = ap.parse_args()
    from s4d.gpu_guard import HOST_NAME, validate_runtime_path

    output_dir = validate_runtime_path(Path(options.output_dir))
    os.environ["S4D_TEST_EVIDENCE_DIR"] = str(output_dir)
    os.environ["S4D_PYTEST_TMP"] = str(output_dir / "tmp")
    if REPO not in output_dir.parents:
        os.environ.setdefault("S4D_CACHE_ROOT", str(output_dir / "cache"))
    from _bootstrap import guard_gpus, source_fingerprints

    gpu_mapping = guard_gpus()
    import pytest
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    source_before = source_fingerprints()
    evidence = Evidence()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    args = [
        str(REPO / "tests"), "-q", "--tb=short", f"--junitxml={output_dir / 'tests.junit.xml'}",
        "-o", f"cache_dir={output_dir / 'pytest-cache'}",
    ]
    stdout, stderr = sys.stdout, sys.stderr
    with (output_dir / "tests.log").open("w") as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        sys.setprofile(evidence.trace)
        try:
            code = int(pytest.main(args, plugins=[evidence]))
        finally:
            sys.setprofile(None)
            sys.stdout, sys.stderr = stdout, stderr
    fingerprints = source_fingerprints()
    source_changed = fingerprints != source_before
    if source_changed:
        code = 1 if code == 0 else code
        evidence.problems.append({"phase": "evidence", "error": "Sources changed during the full suite."})
    report = {
        "started_utc": started,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "host": HOST_NAME,
        "expected_commit": os.environ.get("S4D_EXPECTED_COMMIT"),
        "expected_image_digest": os.environ.get("S4D_IMAGE_DIGEST"),
        "gpu_mapping": gpu_mapping,
        "pytest_args": args,
        "exit_code": code,
        "duration_seconds": time.perf_counter() - clock,
        "counts": {kind: evidence.counts[kind] for kind in ("passed", "error", "failed", "skipped")},
        "total": sum(evidence.counts.values()),
        "full_suite_passed": (
            code == 0 and evidence.counts["passed"] > 0
            and not any(evidence.counts[kind] for kind in ("failed", "error", "skipped"))
        ),
        "source_sha256": fingerprints,
        "source_sha256_before": source_before,
        "source_changed_during_suite": source_changed,
        "executed_functions": {path: sorted(names) for path, names in sorted(evidence.functions.items())},
        "problems": evidence.problems,
        "limitations": (
            "Function trace covers the test parent. Gloo child results are asserted by tests/test_ddp.py. "
            "Synthetic renders are not CUDA acceptance."
        ),
    }
    (output_dir / "tests.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("counts", "total", "exit_code", "full_suite_passed")}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
