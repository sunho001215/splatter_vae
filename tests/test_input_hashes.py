"""Digest reuse must preserve SHA checks and admission refusals."""

import hashlib
import os
import time
from pathlib import Path

import pytest

from ops.jobs import host_roles
from ops.jobs import local_phase_a as a
from ops.jobs import local_phase_a_hybrid as hybrid
from ops.jobs import realworld_local as rw
from ops.jobs import scheduler as hashes


def uncached(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_hits_hardlinks_nested_passes_and_pass_expiry(tmp_path, monkeypatch):
    p = tmp_path / "file"
    p.write_bytes(b"content")
    alias = tmp_path / "alias"
    os.link(p, alias)
    opened = []
    real_open = Path.open

    def tracked(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked)
    with hashes.input_verification_pass():
        first = hashes.input_sha256(p)
        with hashes.input_verification_pass():
            assert hashes.input_sha256(alias) == hashes.input_sha256(p) == first
        assert len(opened) == 1
    with hashes.input_verification_pass():
        assert hashes.input_sha256(p) == first
    assert len(opened) == 2
    hashes.input_sha256(p)
    hashes.input_sha256(p)
    assert len(opened) == 4


@pytest.mark.parametrize("change", ["mtime", "size", "inode", "ctime"])
def test_stat_changes_invalidate(tmp_path, monkeypatch, change):
    p = tmp_path / "file"
    p.write_bytes(b"original")
    with hashes.input_verification_pass():
        hashes.input_sha256(p)
        old = p.stat()
        if change == "mtime":
            os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
        elif change == "size":
            p.write_bytes(b"longer content")
        elif change == "inode":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"replaced")
            os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
            replacement.replace(p)
            assert p.stat().st_ino != old.st_ino and p.stat().st_size == old.st_size
        else:
            # ctime updates can share a filesystem clock tick with creation.
            time.sleep(0.02)
            p.write_bytes(b"modified")
            os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns))
            assert p.stat().st_ctime_ns != old.st_ctime_ns
        expected = uncached(p)
        calls = []
        real_open = Path.open

        def tracked(path, *args, **kwargs):
            calls.append(path)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", tracked)
        assert hashes.input_sha256(p) == expected
        assert calls == [p]


@pytest.mark.parametrize("replace", [False, True])
def test_mutation_during_hash_fails_closed_and_is_not_cached(tmp_path, monkeypatch, replace):
    p = tmp_path / "file"
    p.write_bytes(b"original")
    real_sha = hashlib.sha256

    class MutatingDigest:
        def __init__(self):
            self.digest = real_sha()

        def update(self, block):
            self.digest.update(block)
            if replace:
                other = tmp_path / "other"
                other.write_bytes(b"changed")
                other.replace(p)
            else:
                p.write_bytes(b"changed")

        def hexdigest(self):
            return self.digest.hexdigest()

    with hashes.input_verification_pass():
        monkeypatch.setattr(hashes.hashlib, "sha256", MutatingDigest)
        with pytest.raises(ValueError, match="input (changed|replaced) during hashing"):
            hashes.input_sha256(p)
        assert not hashes._input_hash_cache.get()
        monkeypatch.setattr(hashes.hashlib, "sha256", real_sha)
        assert hashes.input_sha256(p) == uncached(p)


def test_exception_discards_pass():
    with pytest.raises(RuntimeError), hashes.input_verification_pass():
        raise RuntimeError("abort")
    assert hashes._input_hash_cache.get() is None


def test_adapter_restores_pinned_resolvers_and_preserves_test_doubles(monkeypatch):
    original_a, original_rw = a.sha, rw._hash
    with pytest.raises(RuntimeError), hashes.input_verification_pass():
        assert a.sha is rw._hash is hashes.input_sha256
        raise RuntimeError("abort")
    assert a.sha is original_a and rw._hash is original_rw
    fake = lambda path: "test double"
    monkeypatch.setattr(a, "sha", fake)
    with hashes.input_verification_pass():
        assert a.sha is fake
        assert rw._hash is hashes.input_sha256
    assert a.sha is fake and rw._hash is original_rw


def phase_row(tmp_path, monkeypatch):
    from ops.jobs.local_phase_a_gate import TOLERANCES

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    weight = frozen / "weights"
    weight.write_bytes(b"weights")
    environment = tmp_path / "environment"
    environment.write_bytes(b"env")
    a.atomic(
        tmp_path / "envelope.json",
        dict(
            frozen=str(frozen),
            files=a.inventory(frozen),
            commit="pin",
            upstream_pins={},
            environment_files={str(environment): a.sha(environment)},
        ),
    )
    a.readonly(frozen)
    monkeypatch.setattr(a, "git", lambda *args: "pin")
    evidence = tmp_path / "evidence"
    evidence.write_bytes(b"processed qualification data")
    reference = tmp_path / "reference"
    reference.write_bytes(b"reference")
    a.atomic(
        tmp_path / hybrid.REPORT,
        dict(
            schema=hybrid.MODE,
            qualified=True,
            stages=list(hybrid.STAGES),
            coverage="two-episodes-three-stages",
            envelope_sha256=a.sha(tmp_path / "envelope.json"),
            tolerances=TOLERANCES,
            reference_report=str(reference),
            reference_report_sha256=a.sha(reference),
            episodes=[
                dict(
                    task=t,
                    passed=True,
                    label="success",
                    stages={s: dict(passed=True, files={"payload": "sha"}) for s in hybrid.STAGES},
                    evidence_files={str(evidence): a.sha(evidence)},
                )
                for t in ("covering", "pick-and-place")
            ],
            footprints=dict(gpu_declaration_gib=8, ram_declaration_gib=16, measured=True),
            measurement_files={str(environment): a.sha(environment)},
        ),
    )
    a.atomic(tmp_path / "dedup-audit.json", {})
    raw = tmp_path / "raw"
    raw.write_bytes(b"raw episode")
    ep = "drawer-open-v2-success-episode_000001"
    payload = dict(
        episode=ep,
        raw=str(raw),
        raw_sha256=a.sha(raw),
        envelope_sha256=a.sha(tmp_path / "envelope.json"),
        dedup_audit_sha256=a.sha(tmp_path / "dedup-audit.json"),
        mode=hybrid.MODE,
        stages=list(hybrid.STAGES),
        remote_remaining_stages=["sam3_masks"],
        pipeline=dict(
            output_root=str(a.PROCESSED.joinpath(*a.identity(ep))), adapter=dict(config=dict(episodes={ep: str(raw)}))
        ),
    )
    jobfile = tmp_path / "jobs" / f"{ep}.json"
    a.atomic(jobfile, payload)
    row = dict(
        id=f"rwv2-20261113-prep-ours-{ep}-hybridA",
        script=a.SCRIPT,
        role="realworld_preprocess",
        host="local",
        planned_commit="a" * 40,
        registration_tool="runs/setup/tools/register_local_phase_a_20261115.py",
        args=[
            "--bundle",
            str(tmp_path),
            "--job",
            str(jobfile),
            "--job-sha256",
            a.sha(jobfile),
            "--gate-sha256",
            a.sha(tmp_path / hybrid.REPORT),
        ],
    )
    scope = dict(source_commit=row["planned_commit"], files={"scripts/example.py": "sha"})
    ownership = {row["id"]: dict(kind="role_placement", state="placed", owner_host="local", execution_scope=scope)}

    def decision():
        try:
            return rw.localize(
                row,
                {},
                roles=host_roles.load_roles(),
                ownership=ownership,
                source_identity=dict(source_sha256=scope["files"]),
                root=tmp_path,
            )
        except rw.LocalInputError as exc:
            return str(exc)

    return decision, dict(
        weight=weight,
        environment=environment,
        evidence=evidence,
        raw=raw,
        job=jobfile,
        gate=tmp_path / hybrid.REPORT,
        frozen=frozen,
    )


@pytest.mark.parametrize("changed", ["weight", "environment", "evidence", "raw", "job", "gate", "inventory"])
def test_admission_identical_and_changes_never_bypass_verification(tmp_path, monkeypatch, changed):
    decision, files = phase_row(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(a, "sha", uncached)
        before = decision()
    assert isinstance(before, tuple)
    with hashes.input_verification_pass():
        assert decision() == decision() == before
        if changed == "inventory":
            files["frozen"].chmod(0o755)
            (files["frozen"] / "extra").write_bytes(b"new unlisted file")
        else:
            path = files[changed]
            path.chmod(0o644)
            path.write_bytes(b"changed")
            path.chmod(0o444)
        refused = decision()
        assert isinstance(refused, str)
        with monkeypatch.context() as patch:
            patch.setattr(a, "sha", uncached)
            assert decision() == refused
