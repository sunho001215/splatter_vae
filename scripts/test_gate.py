"""Release gate for scripts/run_tests.py reports: a complete run, or a selective run proven equivalent (2026-11-17).

User approval 2026-11-17 ~23:10: affected-test execution and trustworthy cached passes may satisfy ordinary training-job
release gates, given exact code/dependency/environment fingerprints, valid change-impact evidence and an accepted
full-suite baseline; robot/policy-serving code, GPU guards, container launchers and unknown or high-risk changes keep
the mandatory complete suite.

``verdict`` decides every run; scripts/run_tests.py writes it to tests.json as ``release_gate`` = {"accepted", "mode":
"full"|"selective"|null, "baseline", "candidate", "reasons", ...}. A SELECTIVE run is accepted only when ALL hold:
  (a) an accepted baseline record exists for the --changed BASE commit: a complete passing local run whose source
      fingerprints equal BASE's committed sources (re-derived from git objects now), in this exact environment (python,
      torch, CUDA runtime, host, device names), with BASE an ancestor of HEAD;
  (b) every path changed since BASE (committed, staged, unstaged, untracked; deletions included, renames as delete+add)
      is covered by the change-impact selection (a selected test file depends on it, or it is a test file) or is inert
      documentation/run records (INERT); nothing widened, nothing unattributed and unclassified;
  (c) every selected test executed and passed in this run, or was reused from a pass recorded under its exact key
      (code closure, uv.lock/pyproject/runner files, submodule commits, interpreter/torch/CUDA/host/devices);
  (d) no high-risk path changed (scripts/test_impact.PROTECTED plus GLOBAL_FILES: robot client/executor, policy
      serving, GPU guards and allowlists, container launchers, the runner/selector/gate, conftest, uv.lock ...).
Every unselected test file then still has the baseline's passing node ids for byte-identical dependencies; the
accepted report is the composition (executed + reused + carried), like ops/containers/suite_partition.combine for
the two partitions: ``full_suite_passed`` true (what scripts/_bootstrap.require_passed_tests, the scheduler and the
adoption checks read), ``complete_suite_executed`` false, ``counts`` the composed evidence, ``run_counts`` this run.

Baselines: <result cache>/baselines/<commit>.json, written by run_tests after a complete passing local run on a clean
commit, or registered from an existing evidence directory with ``record-baseline``. ``check`` is the adopter's
acceptance command (exit 0 accepted, 1 refused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import test_impact  # noqa: E402

GATE_VERSION = 1
KINDS = ("passed", "error", "failed", "skipped")
# Inert: documentation and run records outside tests/ and configs/ - never imported, executed or read as test input
# unless a test names them (then they are covered, not inert).
INERT_SUFFIXES = (".md", ".rst", ".txt")
RECORD_ROOTS = ("docs/", "runs/", "experiments/")
CODE_SUFFIXES = (".py", ".sh", ".pyx", ".c", ".cc", ".cpp", ".cu", ".h", ".so", ".toml", ".cfg", ".ini", ".lock")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def source_digest(fingerprints: dict) -> str:
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def environment(report: dict) -> dict:
    """The environment a report ran in, from its own fields (baseline and candidate derive it identically)."""
    return {"python": report.get("python"), "torch": report.get("torch"), "cuda_runtime": report.get("cuda_runtime"),
            "host": report.get("host"), "devices": sorted(str(d.get("name")) for d in report.get("gpu_mapping") or [])}


def commit_sources(repo: Path, commit: str) -> dict:
    from ops.jobs.remote_backend import commit_source_sha256

    return commit_source_sha256(repo, commit)


def high_risk(path: str) -> bool:
    return test_impact.is_global(path) or test_impact.is_protected(path)


def inert(path: str) -> bool:
    if path.startswith(("tests/", "configs/")) or PurePosixPath(path).suffix in CODE_SUFFIXES:
        return False
    return PurePosixPath(path).suffix in INERT_SUFFIXES or path.startswith(RECORD_ROOTS)


def complete_local_evidence(report: dict) -> list[str]:
    """Reasons a report is not a complete passing local suite (empty when it is)."""
    counts = report.get("counts") or {}
    reasons = []
    if report.get("full_suite_passed") is not True or report.get("complete_suite_executed", True) is not True:
        reasons.append("not a complete executed suite")
    if report.get("exit_code") != 0 or report.get("problems"):
        reasons.append("exit code or problems")
    if any(counts.get(kind, 1) for kind in ("failed", "error", "skipped")) or counts.get("passed", 0) < 1:
        reasons.append("failed, errored, skipped or no passed tests")
    if report.get("source_changed_during_suite") is not False:
        reasons.append("sources changed during the suite")
    if report.get("partition", "all") != "all" or report.get("host") != "local":
        reasons.append("not an unpartitioned local suite")
    if (report.get("reuse") or {}).get("reused_tests"):
        reasons.append("reused results")
    return reasons


# ---------------------------------------------------------------------------------------------------- baselines
class Baselines:
    """Accepted complete-suite records: ``<root>/baselines/<commit>.json``."""

    def __init__(self, root: Path):
        self.root = Path(root) / "baselines"

    def path(self, commit: str) -> Path:
        return self.root / f"{commit}.json"

    def load(self, commit: str) -> dict | None:
        try:
            record = json.loads(self.path(commit).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or record.get("version") != GATE_VERSION or record.get("commit") != commit:
            return None
        return record

    def record(self, report: dict, nodeids: list[str], commit: str, repo: Path, evidence: str = "") -> dict:
        """Validate ``report`` as complete passing local evidence of exactly ``commit`` and store it."""
        reasons = complete_local_evidence(report)
        if not nodeids or len(nodeids) != report.get("collected_total") or len(set(nodeids)) != len(nodeids):
            reasons.append("collected node ids do not match the report")
        if report.get("source_sha256") != commit_sources(repo, commit):
            reasons.append(f"report sources differ from commit {commit}")
        if reasons:
            raise ValueError("not an accepted baseline: " + "; ".join(reasons))
        record = {"version": GATE_VERSION, "commit": commit, "source_digest": source_digest(report["source_sha256"]),
                  "environment": environment(report), "collected_total": len(nodeids),
                  "nodes": test_impact.by_file(sorted(nodeids)), "evidence": evidence,
                  "recorded_utc": datetime.now(timezone.utc).isoformat()}
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.root, delete=False, suffix=".tmp") as stream:
            json.dump(record, stream)
            temporary = Path(stream.name)
        os.replace(temporary, self.path(commit))
        return record


# ---------------------------------------------------------------------------------------------------- verdict
def verdict(report: dict, *, repo: Path, baselines: Baselines | None, current_tests: list[str], base: str | None,
            executed: list[str], selection: dict | None = None, passed: list[str] | None = None) -> dict:
    """The release-gate verdict of a finished run (``report`` as run_tests builds it, before the gate fields).

    ``executed`` are the node ids this run executed (all reported passing when the run passed), ``current_tests`` the
    checkout's test files, ``base`` the --changed BASE (None for a run without one), ``selection`` the complete
    (untruncated) test_impact.select result of this run, ``passed`` the node ids whose setup, call and teardown passed
    with no failed or skipped subtest (test_impact.passed_nodes of the worker outcomes). Tests are counted by node id:
    ``counts`` also counts subtest reports, so it never equals the number of executed nodes (2026-11-18)."""
    reasons: list[str] = []
    candidate = {"head": _head(repo), "source_digest": source_digest(report.get("source_sha256") or {})}
    if base is None:
        accepted = report.get("full_suite_passed") is True
        return {"version": GATE_VERSION, "accepted": accepted, "mode": "full" if accepted else None, "baseline": None,
                "candidate": candidate, "reasons": [] if accepted else ["complete suite did not pass"]}
    selection = selection or {}
    reuse = report.get("reuse") or {}
    out = {"version": GATE_VERSION, "accepted": False, "mode": None, "baseline": None, "candidate": candidate}
    # (a) the baseline
    try:
        commit = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        commit = None
        reasons.append(f"baseline {base!r} is not a commit")
    record = baselines.load(commit) if commit and baselines is not None else None
    if commit and record is None:
        reasons.append(f"no accepted full-suite baseline record for {commit}")
    if record is not None:
        out["baseline"] = {"commit": commit, "source_digest": record["source_digest"], "evidence": record.get("evidence")}
        try:
            if source_digest(commit_sources(repo, commit)) != record["source_digest"]:
                reasons.append("baseline record does not match the baseline commit's sources")
        except Exception as exc:  # noqa: BLE001 - an unreadable baseline commit refuses
            reasons.append(f"baseline commit sources unreadable ({type(exc).__name__})")
        if record.get("environment") != environment(report):
            reasons.append("baseline environment (python/torch/CUDA/host/devices) differs from this run")
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
            reasons.append("baseline commit is not an ancestor of HEAD")
    # (b) + (d) the change set
    if selection.get("base") != base or "changed" not in selection:
        reasons.append("no change-impact selection for this baseline")
    risky = [p for p in selection.get("changed", []) if high_risk(p)]
    if risky:
        reasons.append("high-risk paths changed (complete suite required): " + ", ".join(risky[:5]))
    if selection.get("widened"):
        reasons.append("unattributed test/config inputs widened the selection: " + ", ".join(selection["widened"][:5]))
    unclassified = [p for p in [*selection.get("unattributed", []), *selection.get("guard_only", [])] if not inert(p)]
    if unclassified:
        reasons.append("unclassified changed paths (complete suite required): " + ", ".join(unclassified[:5]))
    # (c) the selected tests
    if report.get("partition", "all") != "all" or report.get("host") != "local":
        reasons.append("selective acceptance is for unpartitioned local runs only")
    if report.get("exit_code") != 0 or report.get("problems") or report.get("source_changed_during_suite") is not False:
        reasons.append("the run failed, had problems or its sources changed")
    if reuse.get("error"):
        reasons.append(f"result reuse failed ({reuse['error']})")
    reused = {item["file"]: item["tests"] for item in reuse.get("reused_files", [])}
    counts = report.get("counts") or {}
    not_passed = sorted(set(executed) - set(passed or []))
    if not_passed or any(counts.get(kind) for kind in ("failed", "error", "skipped")):
        reasons.append(f"executed tests did not all pass ({len(not_passed)} of {len(executed)} node ids without a "
                       "complete pass)" + (f", e.g. {not_passed[0]}" if not_passed else ""))
    if len(executed) + sum(reused.values()) != report.get("collected_total"):
        reasons.append("selected tests are neither executed nor reused")
    selected = set(selection.get("files", []))
    carried = 0
    if record is not None:
        for test_file in current_tests:
            if test_file in selected:
                continue
            nodes = record["nodes"].get(test_file)
            if not nodes:
                reasons.append(f"unselected {test_file} has no baseline pass")
                break
            carried += len(nodes)
    out.update(accepted=not reasons, mode="selective" if not reasons else None, reasons=reasons,
               executed_tests=len(executed), reused_tests=sum(reused.values()), carried_tests=carried,
               selected_files=len(selected), total_files=len(current_tests))
    return out


def _head(repo: Path) -> str | None:
    try:
        return _git(repo, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None


def accepted(report: dict, sources: dict | None = None,
             checkout: Path | None = None) -> tuple[bool, str | None, list[str]]:
    """The acceptance function for adoption and release checks: (accepted, "full"|"selective", reasons).

    A report without ``release_gate`` (before 2026-11-17) is accepted as a complete suite exactly as before. A report
    naming its checkout (``checkout``, since 2026-11-18) must name ``checkout`` when one is given; a selective report's
    counts must be its executed + reused + carried node ids."""
    counts = report.get("counts") or {}
    reasons = []
    if checkout is not None and report.get("checkout") is not None and \
            Path(report["checkout"]).resolve() != Path(checkout).resolve():
        reasons.append(f"evidence of another checkout ({report['checkout']})")
    if report.get("full_suite_passed") is not True or report.get("exit_code") != 0:
        reasons.append("no accepted suite")
    if any(counts.get(kind, 1) for kind in ("failed", "error", "skipped")) or counts.get("passed", 0) < 1:
        reasons.append("failed, errored, skipped or no passed tests")
    gate = report.get("release_gate")
    mode = "full"
    if gate is not None:
        mode = gate.get("mode")
        if gate.get("accepted") is not True or mode not in ("full", "selective"):
            reasons.append("release gate refused: " + "; ".join(gate.get("reasons") or ["no verdict"]))
        if mode == "selective" and (report.get("complete_suite_executed") is not False or counts.get("passed") != sum(
                gate.get(name, 0) for name in ("executed_tests", "reused_tests", "carried_tests"))):
            reasons.append("inconsistent selective report")
        if mode == "full" and report.get("complete_suite_executed", True) is not True:
            reasons.append("inconsistent complete report")
    if sources is not None and report.get("source_sha256") != sources:
        reasons.append("evidence is stale for these sources")
    return not reasons, (mode if not reasons else None), reasons


def adoption_check(evidence: dict, worktree: Path, commit: str, *, baselines: Baselines | None = None,
                   candidate_check: bool = True) -> list[str]:
    """Reasons an ACCEPTED SELECTIVE report cannot be adopted (empty = adoptable); the adoption tool's counterpart of
    its shard-coverage check for complete runs (2026-11-18).

    Re-verified with the ADOPTING checkout's gate and selector code, not only the candidate's: the report is accepted
    (``accepted``: verdict, flags, composed counts) and names ``worktree``; it was produced at ``commit``; its baseline is
    an ancestor of ``commit`` with an untampered record (re-derived from git objects) in the report's environment; the
    change set baseline..commit, re-selected from the committed trees, holds no high-risk, widened or unclassified path,
    and every re-selected test file was selected by the run; finally the candidate's own ``scripts/test_gate.py check``
    accepts the evidence for its sources. Git objects are read through ``worktree`` (worktrees share them)."""
    worktree = Path(worktree).resolve()
    gate = evidence.get("release_gate") or {}
    if gate.get("mode") != "selective" or gate.get("accepted") is not True:
        return ["not an accepted selective verdict"]
    ok, mode, reasons = accepted(evidence, None, worktree)
    reasons = list(reasons)
    if not ok or mode != "selective":
        reasons.append("the report is not accepted as selective")
    if evidence.get("head") != commit or (gate.get("candidate") or {}).get("head") != commit:
        reasons.append("the evidence was not produced at the candidate commit")
    if evidence.get("checkout") is None:
        reasons.append("the evidence does not name its checkout")
    base = (gate.get("baseline") or {}).get("commit")
    if not base:
        return reasons + ["no baseline in the verdict"]
    if subprocess.run(["git", "-C", str(worktree), "merge-base", "--is-ancestor", base, commit]).returncode != 0:
        reasons.append("the baseline is not an ancestor of the candidate")
    store = baselines or Baselines(test_impact.cache_root(REPO))
    record = store.load(base)
    if record is None:
        reasons.append(f"no accepted baseline record for {base}")
    else:
        digest = source_digest(commit_sources(worktree, base))
        if record.get("source_digest") != digest or (gate.get("baseline") or {}).get("source_digest") != digest:
            reasons.append("the baseline record does not match the baseline commit's sources")
        if record.get("environment") != environment(evidence):
            reasons.append("the baseline environment differs from the evidence")
    tree = test_impact.CommitTree(worktree, commit)
    selection = test_impact.select(test_impact.Graph(tree),
                                   test_impact.changed_files(base, worktree, head=commit, worktree=False))
    risky = [p for p in selection["changed"] if high_risk(p)]
    if risky:
        reasons.append("high-risk paths changed (complete suite required): " + ", ".join(risky[:5]))
    if selection["widened"]:
        reasons.append("unattributed test/config inputs: " + ", ".join(selection["widened"][:5]))
    unclassified = [p for p in [*selection["unattributed"], *selection.get("guard_only", [])] if not inert(p)]
    if unclassified:
        reasons.append("unclassified changed paths: " + ", ".join(unclassified[:5]))
    uncovered = sorted(set(selection["files"]) - set((evidence.get("selection") or {}).get("files") or []))
    if uncovered:
        reasons.append("test files the change set selects were not covered by the run: " + ", ".join(uncovered[:5]))
    if candidate_check:
        script = worktree / "scripts/test_gate.py"
        process = subprocess.run([sys.executable, "-I", str(script), "check", "--evidence", str(worktree / "docs")],
                                 cwd=worktree, capture_output=True, text=True, timeout=600)
        if process.returncode != 0:
            reasons.append(f"the candidate's test_gate check refuses (exit {process.returncode}): "
                           + (process.stdout + process.stderr).strip()[-300:])
    return reasons


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="accept or refuse an evidence directory's tests.json (exit 0 = accepted)")
    check.add_argument("--evidence", type=Path, default=REPO / "docs")
    check.add_argument("--any-sources", action="store_true", help="do not compare with this checkout's sources")
    base = sub.add_parser("record-baseline", help="register a complete passing local run as a baseline")
    base.add_argument("--evidence", type=Path, required=True, help="directory with tests.json and collected.json")
    base.add_argument("--commit", default="HEAD")
    options = ap.parse_args(argv)
    if options.command == "check":
        path = (options.evidence / "tests.json").resolve()
        report = json.loads(path.read_text())
        sources = None
        if not options.any_sources:
            from _bootstrap import source_fingerprints

            sources = source_fingerprints()
        ok, mode, reasons = accepted(report, sources, REPO)
        gate = report.get("release_gate") or {}
        print(json.dumps({"accepted": ok, "mode": mode, "reasons": reasons, "evidence": str(path),
                          "checkout": str(REPO), "report_checkout": report.get("checkout"),
                          "report_gate": {k: gate.get(k) for k in ("accepted", "mode", "reasons")}}, indent=1))
        return 0 if ok else 1
    commit = _git(REPO, "rev-parse", "--verify", f"{options.commit}^{{commit}}")
    report = json.loads((options.evidence / "tests.json").read_text())
    nodeids = json.loads((options.evidence / "collected.json").read_text())
    record = Baselines(test_impact.cache_root(REPO)).record(report, nodeids, commit, REPO, str(options.evidence))
    print(json.dumps({k: record[k] for k in ("commit", "source_digest", "collected_total")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
