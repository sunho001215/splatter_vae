"""Release-suite partitions (user 2026-11-12): the remote native suite deselects the metaworld_data tests, the release
chain runs exactly those on the local host, and only their exact combination is a complete suite."""

from __future__ import annotations

import copy

import pytest

from ops.containers import suite_partition as sp
from ops.containers.verification import validate_report

SOURCE = {"s4d/a.py": "1" * 64, "tests/test_x.py": "2" * 64}
COMMIT, IMAGE = "c" * 40, "sha256:" + "d" * 64
ALL = [f"tests/test_x.py::t{i}" for i in range(6)]
LOCAL_IDS = ["tests/test_x.py::t5"]


def part(partition, selected, host, **changes):
    deselected = [n for n in ALL if n not in selected]
    report = {"partition": partition, "selected": list(selected), "deselected": deselected, "collected_total": len(ALL),
              "partition_passed": True, "full_suite_passed": False, "exit_code": 0,
              "counts": {"passed": len(selected), "error": 0, "failed": 0, "skipped": 0}, "host": host,
              "source_sha256": dict(SOURCE), "source_sha256_before": dict(SOURCE), "source_changed_during_suite": False,
              "problems": [], "expected_commit": COMMIT,
              "expected_image_digest": IMAGE if host == "remote" else None}
    report.update(changes)
    return report


def remote(**changes):
    return part(sp.REMOTE, [n for n in ALL if n not in LOCAL_IDS], "remote", **changes)


def local(**changes):
    return part(sp.LOCAL, LOCAL_IDS, "local", **changes)


def test_exact_partitions_combine_into_a_complete_suite():
    combined = sp.combine(remote(), local())
    assert combined["full_suite_passed"] is True and combined["partition"] == sp.COMBINED
    assert combined["counts"] == {"passed": 6, "error": 0, "failed": 0, "skipped": 0} and combined["total"] == 6
    assert combined["host"] == "remote" and combined["partitions"][sp.LOCAL]["host"] == "local"
    sp.check_combined(combined)
    validate_report("native_suite", combined, SOURCE, COMMIT, IMAGE, "")
    validate_report("native_suite", remote(), SOURCE, COMMIT, IMAGE, "")  # the verification job's own report


@pytest.mark.parametrize("which,changes", [
    ("remote", {"partition_passed": False}), ("remote", {"exit_code": 1}),
    ("remote", {"counts": {"passed": 5, "error": 0, "failed": 0, "skipped": 1}}),
    ("remote", {"full_suite_passed": True}), ("remote", {"source_changed_during_suite": True}),
    ("remote", {"problems": [{"error": "x"}]}), ("remote", {"host": "local"}),
    ("local", {"host": "remote"}), ("local", {"counts": {"passed": 0, "error": 0, "failed": 0, "skipped": 0}}),
    ("local", {"source_sha256": {**SOURCE, "s4d/a.py": "9" * 64}, "source_sha256_before": {**SOURCE, "s4d/a.py": "9" * 64}}),
    ("local", {"expected_commit": "e" * 40}), ("remote", {"collected_total": 7}),
])
def test_failing_stale_or_mismatched_partitions_never_combine(which, changes):
    parts = {"remote": remote(), "local": local()}
    parts[which].update(changes)
    with pytest.raises(ValueError):
        sp.combine(parts["remote"], parts["local"])


def test_coverage_must_be_exact():
    with pytest.raises(ValueError):  # the local partition ran a test the remote also ran
        sp.combine(remote(), part(sp.LOCAL, LOCAL_IDS + ["tests/test_x.py::t0"], "local"))
    with pytest.raises(ValueError):  # a test ran nowhere
        sp.combine(part(sp.REMOTE, ALL[:4], "remote"), local())
    with pytest.raises(ValueError):  # duplicates
        duplicated = remote()
        duplicated["selected"] = duplicated["selected"] + duplicated["selected"][:1]
        sp.combine(duplicated, local())
    with pytest.raises(ValueError):  # the remote partition must deselect something
        sp.combine(part(sp.REMOTE, ALL, "remote"), local())


def test_a_tampered_combined_report_is_rejected_everywhere():
    combined = sp.combine(remote(), local())
    for mutate in (lambda c: c.update(total=7), lambda c: c["counts"].update(passed=7),
                   lambda c: c["partitions"][sp.LOCAL].update(selected=[]), lambda c: c.pop("partitions")):
        bad = copy.deepcopy(combined)
        mutate(bad)
        with pytest.raises(ValueError):
            sp.check_combined(bad)
        with pytest.raises(ValueError):
            validate_report("native_suite", bad, SOURCE, COMMIT, IMAGE, "")


def test_a_local_partition_alone_is_never_a_full_suite():
    with pytest.raises(ValueError):
        validate_report("native_suite", local(), SOURCE, COMMIT, IMAGE, "")
    with pytest.raises(ValueError):  # wrong sources or commit for the release
        validate_report("native_suite", remote(), {**SOURCE, "x": "0" * 64}, COMMIT, IMAGE, "")
    with pytest.raises(ValueError):
        validate_report("native_suite", remote(), SOURCE, "f" * 40, IMAGE, "")


def test_every_gate_consumer_re_derives_a_combined_report(tmp_path, monkeypatch):
    """Codex review 2026-11-12: docs/tests.json consumers (scheduler.tested_source, _bootstrap.require_passed_tests)
    and the acceptance manifest never trust a combined report's aggregate fields alone."""
    import json
    import sys

    from ops.jobs import scheduler
    combined = sp.combine(remote(), local())
    tampered = copy.deepcopy(combined)
    tampered["partitions"][sp.LOCAL]["counts"]["skipped"] = 1  # aggregate fields left passing
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(combined))
    bad.write_text(json.dumps(tampered))
    assert scheduler.tested_source(good) == SOURCE
    assert scheduler.tested_source(bad) is None

    sys.path.insert(0, str(sp.__file__).rsplit("/ops/", 1)[0] + "/scripts")
    import _bootstrap
    monkeypatch.setattr(_bootstrap, "source_fingerprints", lambda: SOURCE)
    for report, ok in ((combined, True), (tampered, False)):
        evidence = tmp_path / "evidence"
        evidence.mkdir(exist_ok=True)
        (evidence / "tests.json").write_text(json.dumps(report))
        monkeypatch.setattr(_bootstrap, "test_evidence_directory", lambda directory=evidence: directory)
        if ok:
            _bootstrap.require_passed_tests()
        else:
            with pytest.raises(RuntimeError, match="partitions are inconsistent"):
                _bootstrap.require_passed_tests()
    from pathlib import Path
    source = Path(sp.__file__).resolve().parents[2] / "ops/jobs/remote_backend.py"
    assert "suite_partition.check_combined(suite)" in source.read_text()  # acceptance_manifest applies it too


def test_adoption_provenance_is_the_only_extra_field_a_combined_report_may_carry(tmp_path):
    """Codex review 2026-11-12: adopt_remote_suite_evidence adds adopted_evidence; nothing else may be added."""
    import json

    from ops.jobs import scheduler
    adopted = {**sp.combine(remote(), local()), "adopted_evidence": {"job": "verify-native-x"}}
    sp.check_combined(adopted)
    path = tmp_path / "tests.json"
    path.write_text(json.dumps(adopted))
    assert scheduler.tested_source(path) == SOURCE
    with pytest.raises(ValueError):
        sp.check_combined({**adopted, "unexpected": 1})
