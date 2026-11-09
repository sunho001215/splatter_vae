"""Shared operational primitives are tested without importing campaign command modules."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import time
import zipfile
from pathlib import Path

import pytest

from ops.jobs import checkpoint_pause
from ops.jobs.arguments import option
from ops.provenance import hashing
from ops.resources import processes, reservations


@pytest.mark.parametrize("state,expected", [("S", True), ("R", True), ("Z", False), ("X", True)])
def test_liveness_checks_the_stat_state_after_the_signal_probe(monkeypatch, state, expected):
    probes = []
    monkeypatch.setattr(processes.os, "kill", lambda pid, sig: probes.append((pid, sig)))
    monkeypatch.setattr(Path, "read_text", lambda path: f"123 (name with ) parentheses) {state} 1 2 3")
    assert processes.pid_alive(123) is expected
    assert probes == [(123, 0)]


@pytest.mark.parametrize("error,expected", [(ProcessLookupError, False), (PermissionError, True)])
def test_liveness_signal_errors_do_not_read_stat(monkeypatch, error, expected):
    def kill(pid, sig):
        raise error()

    def read(path):
        raise AssertionError("stat must not be read after a failed signal probe")

    monkeypatch.setattr(processes.os, "kill", kill)
    monkeypatch.setattr(Path, "read_text", read)
    assert processes.pid_alive(123) is expected


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_liveness_handles_exit_between_probe_and_stat(monkeypatch, error):
    monkeypatch.setattr(processes.os, "kill", lambda pid, sig: None)

    def read(path):
        raise error()

    monkeypatch.setattr(Path, "read_text", read)
    assert processes.pid_alive(123) is False


@pytest.mark.parametrize("args,flag,expected", [
    ([], "--name", None), (["--name"], "--name", None),
    (["--name", 12, "--name", 34], "--name", "12"),
    (["--name=x"], "--name", None), ([1, 2], "1", "2"),
    (["--name", "--other"], "--name", "--other"),
])
def test_option_preserves_first_occurrence_string_conversion_and_trailing_flag(args, flag, expected):
    assert option(args, flag) == expected


@pytest.mark.parametrize("expires,expected", [(100.01, 123), (100, 0), (99, 0)])
def test_reservation_expiry_and_fixed_path(monkeypatch, tmp_path, expires, expected):
    reservation = tmp_path / "reservation.json"
    reservation.write_text(json.dumps({"ram_bytes": "123", "expires_unix": str(expires)}))

    def path(value):
        assert value == "/home/ws/ws/splatter4d/runs/setup/local-validation-reservation.json"
        return reservation

    monkeypatch.setattr(reservations, "Path", path)
    monkeypatch.setattr(time, "time", lambda: 100)
    assert reservations.validation_reservation() == expected


def test_reservation_missing_is_zero_but_malformed_is_not_ignored(monkeypatch, tmp_path):
    reservation = tmp_path / "reservation.json"
    monkeypatch.setattr(reservations, "Path", lambda path: reservation)
    assert reservations.validation_reservation() == 0
    reservation.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        reservations.validation_reservation()
    reservation.write_text('{}')
    with pytest.raises(KeyError, match="expires_unix"):
        reservations.validation_reservation()


def test_hashing_uses_eight_mib_reads_and_closes_the_stream():
    payload = b"x" * (8 * 1024 * 1024 + 3)
    reads = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    stream = Stream(payload)

    class File:
        def open(self, mode):
            assert mode == "rb"
            return stream

    assert hashing.sha256_file(File()) == hashlib.sha256(payload).hexdigest()
    assert reads == [8 * 1024 * 1024] * 3 and stream.closed


def test_canonical_json_preserves_encoding_order_and_nan_refusal():
    assert hashing.json_bytes({"z": "한", "a": 1}) == b'{"a":1,"z":"\\ud55c"}'
    assert hashing.json_bytes(7) == b"7"
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        hashing.json_bytes({"a": float("nan")})


def test_pause_identity_parses_after_the_last_parenthesis(monkeypatch):
    fields = ["S", "1", "222", "333", *(["0"] * 15), "444"]

    def read(path):
        assert str(path) == "/proc/123/stat"
        return "123 (a name with ) parentheses) " + " ".join(fields)

    monkeypatch.setattr(Path, "read_text", read)
    assert checkpoint_pause.identity(123) == {
        "pid": 123, "group": 222, "session": 333, "start_ticks": 444, "state": "S"}


@pytest.mark.parametrize("step", [1, 50000, 100000])
def test_pause_checkpoint_reader_without_loading_pickle(tmp_path, step):
    path = tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", pickle.dumps({"step": step, "other": None}, protocol=2))
    assert checkpoint_pause.checkpoint_step(path) == step


@pytest.mark.parametrize("payload", [{"other": 1, "step": 2}, {"step": "invalid", "other": None}])
def test_pause_checkpoint_reader_preserves_refusal_message(tmp_path, payload):
    path = tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", pickle.dumps(payload, protocol=2))
    with pytest.raises(RuntimeError, match="^unexpected checkpoint step metadata$"):
        checkpoint_pause.checkpoint_step(path)


def test_pause_session_scan_uses_the_supplied_identity_and_skips_vanished_processes(monkeypatch):
    entries = [Path(name) for name in ("self", "1", "2", "3", "4", "5", "6", "7")]
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(entries))
    calls = []

    def identity(pid):
        calls.append(pid)
        if pid <= 3:
            raise (FileNotFoundError, ProcessLookupError, IndexError)[pid - 1]()
        return {"session": 100 if pid != 6 else 101, "state": {4: "Z", 5: "X", 6: "S", 7: "S"}[pid]}

    assert checkpoint_pause.session_alive(100, identity) is True
    assert calls == list(range(1, 8))
    assert checkpoint_pause.session_alive(999, identity) is False


def test_pause_registry_writer_uses_explicit_path_appends_locks_and_fsyncs(monkeypatch, tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text('{"old": true}\n')
    monkeypatch.setattr(checkpoint_pause.time, "time", lambda: 12.5)
    locks, flushed = [], []
    monkeypatch.setattr(checkpoint_pause.fcntl, "flock", lambda lock, mode: locks.append((lock, mode)))
    monkeypatch.setattr(checkpoint_pause.os, "fsync", lambda fd: flushed.append(registry.read_text()))
    event = {"event": "checkpoint_paused", "time": -1}
    assert checkpoint_pause.record(event, registry) is None
    expected = '{"old": true}\n{"event": "checkpoint_paused", "time": 12.5}\n'
    assert registry.read_text() == expected and flushed == [expected]
    assert len(locks) == 1 and locks[0][1] == checkpoint_pause.fcntl.LOCK_EX
    assert locks[0][0].name == str(registry.with_suffix(".lock")) and locks[0][0].closed
    assert event["time"] == -1
