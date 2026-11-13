"""Release-suite partitions: tests reading Meta-World development data or local campaign-host files run on the
local host; the remote native suite runs every other test.

scripts/run_tests.py writes, for a partition run, ``partition`` ('remote' or 'local'), the nodeids it ran
(``selected``), the nodeids of the complete collection it did not run (``deselected``), the size of the complete
collection (``collected_total``) and ``partition_passed``; ``full_suite_passed`` stays true only for a complete run.

``combine`` turns a passing remote partition and a passing local partition of the SAME sources into the
complete-suite report the gate consumers expect (``full_suite_passed`` true, summed counts, both parts embedded), and
only when: both partitions passed with no failure, error or skip; the local partition ran exactly the nodeids the
remote one deselected and nothing else, so the two are disjoint and together equal the remote's complete collection;
both report the same source fingerprints (before = after, unchanged) and expected commit. Anything else raises
ValueError. The reference collection is the REMOTE host's, as for the unpartitioned gate: collection is host-dependent
by design (e.g. tests/test_dataset_downloads.py adds [zstd] variants where zstd exists, tests/test_mujoco_guard.py
parametrizes over the host's GPUs), so the local host's own complete collection may differ.
"""

from __future__ import annotations

MARKER = "metaworld_data or local_host_data"
# provenance a consumer may add to a combined report (adopt_remote_suite_evidence: docs/tests.json)
PROVENANCE_KEYS = frozenset({"adopted_evidence"})
REMOTE, LOCAL, COMBINED = "remote", "local", "combined"
LOCAL_EVIDENCE_DIR = "local-suite"
KINDS = ("passed", "error", "failed", "skipped")


def check_partition(report: dict, partition: str) -> None:
    """A passing partition report of ``partition`` (raises ValueError otherwise)."""
    if not isinstance(report, dict) or report.get("partition") != partition:
        raise ValueError(f"not a {partition} partition report")
    counts = report.get("counts")
    selected, deselected = report.get("selected"), report.get("deselected")
    if (report.get("partition_marker") != MARKER
            or report.get("partition_passed") is not True or report.get("full_suite_passed") is not False
            or type(report.get("exit_code")) is not int or report["exit_code"] != 0 or not isinstance(counts, dict)
            or any(type(counts.get(kind)) is not int for kind in KINDS)
            or any(counts[kind] != 0 for kind in ("failed", "error", "skipped")) or counts["passed"] < 1
            or not isinstance(selected, list) or not isinstance(deselected, list)
            or not all(isinstance(item, str) and item for item in selected + deselected)
            or len(set(selected)) != len(selected) or len(set(deselected)) != len(deselected)
            or set(selected) & set(deselected)
            or type(report.get("collected_total")) is not int
            or len(selected) + len(deselected) != report["collected_total"]
            or report.get("source_changed_during_suite") is not False
            or not isinstance(report.get("source_sha256"), dict) or not report["source_sha256"]
            or report.get("source_sha256_before") != report["source_sha256"]
            or report.get("problems")):
        raise ValueError(f"{partition} partition report is incomplete or failing")
    if partition == REMOTE and (report.get("host") != "remote" or not deselected):
        raise ValueError("the remote partition must run on the remote host and deselect the local-data tests")
    if partition == LOCAL and report.get("host") != "local":
        raise ValueError("the local-data partition runs on the local host")


def combine(remote: dict, local: dict) -> dict:
    """The complete-suite report of a remote partition plus its local-data complement."""
    check_partition(remote, REMOTE)
    check_partition(local, LOCAL)
    if remote["source_sha256"] != local["source_sha256"]:
        raise ValueError("the partitions ran different sources")
    if remote.get("expected_commit") is None or remote.get("expected_commit") != local.get("expected_commit"):
        raise ValueError("the partitions ran different commits")
    if set(local["selected"]) != set(remote["deselected"]):
        raise ValueError("the local partition did not run exactly the remote partition's deselected tests")
    counts = {kind: remote["counts"][kind] + local["counts"][kind] for kind in KINDS}
    return {**{k: v for k, v in remote.items() if k not in ("selected", "deselected")},
            "partition": COMBINED, "full_suite_passed": True, "partition_passed": True, "counts": counts,
            "total": sum(counts.values()), "collected_total": remote["collected_total"],
            "partitions": {REMOTE: remote, LOCAL: local}}


def check_combined(report: dict) -> None:
    """A combined report whose embedded partitions really combine to it (raises ValueError otherwise)."""
    parts = report.get("partitions") if isinstance(report, dict) else None
    if report.get("partition") != COMBINED or not isinstance(parts, dict) or set(parts) != {REMOTE, LOCAL}:
        raise ValueError("not a combined suite report")
    expected = combine(parts[REMOTE], parts[LOCAL])
    if set(report) - set(expected) - PROVENANCE_KEYS or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("combined suite report differs from its partitions")


def release_support(read) -> dict:
    """Read-only capability preflight of a release tree; acceptance still requires both actual passing reports.

    A historical same-science MIG variant may apply cleanly while carrying a runner
    from before suite partitions. Refuse that route before reserving or launching work.
    """
    required = {
        'scripts/run_tests.py': (b'--partition',),
        'scripts/test_shards.py': (),
        'ops/containers/suite_partition.py': (b'def combine(', MARKER.encode()),
        'ops/containers/verification.py': (b'from ops.containers import suite_partition',),
    }
    blockers = []
    for name, tokens in required.items():
        data = read(name)
        if data is None or any(token not in data for token in tokens):
            blockers.append(f'{name}: missing partitioned release-gate support')
    return {'compatible': not blockers, 'blockers': blockers}
