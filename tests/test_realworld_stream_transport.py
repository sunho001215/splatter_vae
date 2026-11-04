"""CPU-only framing, RAM guards, worker sessions, identity checks, and transport recovery."""

from __future__ import annotations

import dataclasses
import io
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ops.inference_server import stream_client
from ops.inference_server.stream_client import RemoteWorkerPool, StreamSession, TransportError
from s4d.realworld.workers import PROTOCOL, WORKER_DIR, WorkerError, WorkerSpec, call_worker
from s4d.realworld.workers._framing import FrameError, pack_arrays, read_frame, unpack_arrays, write_frame
from s4d.realworld.workers._protocol import _mount_for, git_commit, ram_backed, resources, serve_stream

REPO = Path(__file__).resolve().parents[1]
MOCK = WORKER_DIR / "mock_worker.py"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # The repository conftest checks its allowed MIG device before collection. Workers use no CUDA devices.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    # conftest has imported/initialized torch for its guard; in-process protocol tests must not query CUDA.
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def arrays():
    return {"flows": np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4),
            "visconf": np.ones((2, 2, 3, 4), np.float32)}


@pytest.fixture
def config(tmp_path, monkeypatch):
    if not Path("/dev/shm").is_dir():
        pytest.skip("pool requires a real RAM filesystem")
    monkeypatch.setenv("TMPDIR", "/dev/shm")
    monkeypatch.delenv("S4D_WORKER_ALLOW_DISK", raising=False)
    return {"local_argv": True, "workdir": "/dev/shm", "source_root": str(REPO),
            "workers": {"mock": {"python": sys.executable}}, "expected_source_commit": git_commit(str(REPO)),
            "retries": 2, "backoff_s": [0, 0], "call_timeout_s": 15, "hello_timeout_s": 30}


@pytest.fixture
def pool(config, tmp_path):
    instance = RemoteWorkerPool(config, log_dir=tmp_path / "logs")
    try:
        yield instance
    finally:
        instance.close()


def frame(header=None, payload=b""):
    stream = io.BytesIO()
    write_frame(stream, header or {"type": "test"}, payload)
    return stream.getvalue()


def request(seq=1, chunk_id="chunk", task="echo_tracks"):
    return {"type": "request", "seq": seq, "chunk_id": chunk_id, "task": task, "params": {}}


def test_frames_round_trip_and_empty_payload():
    stream = io.BytesIO()
    header = {"type": "hello", "unicode": "안녕"}
    write_frame(stream, header)
    write_frame(stream, {"type": "request"}, b"some bytes")
    stream.seek(0)
    assert read_frame(stream) == (header, b"")
    received, payload = read_frame(stream)
    assert payload == b"some bytes" and len(received["payload_sha256"]) == 64
    assert "payload_sha256" not in header  # write_frame does not mutate the caller's header
    with pytest.raises(EOFError):
        read_frame(stream)


@pytest.mark.parametrize("offset,value", [(0, ord("X")), (4, 2), (-1, 0)])
def test_frame_corruption(offset, value):
    encoded = bytearray(frame(payload=b"abcdef"))
    encoded[offset] = value
    with pytest.raises(FrameError):
        read_frame(io.BytesIO(encoded))


def test_all_truncations_are_errors():
    encoded = frame(payload=b"abcdef")
    for length in range(1, len(encoded)):
        with pytest.raises(FrameError, match="truncated"):
            read_frame(io.BytesIO(encoded[:length]))


def test_oversize_refused_before_reading_body():
    with pytest.raises(FrameError, match="header length"):
        read_frame(io.BytesIO(b"S4DF\x01" + struct.pack(">I", 1024)), max_header=16)
    with pytest.raises(FrameError, match="payload length"):
        read_frame(io.BytesIO(frame(payload=b"1234")[:-4]), max_payload=3)


@pytest.mark.parametrize("header", [b"[]", b"{bad", b'"string"', b"\xff"])
def test_invalid_headers(header):
    with pytest.raises(FrameError):
        read_frame(io.BytesIO(b"S4DF\x01" + struct.pack(">I", len(header)) + header + struct.pack(">Q", 0)))


def test_checksum_is_required_for_nonempty_payload():
    with pytest.raises(FrameError, match="sha256"):
        read_frame(io.BytesIO(frame()[:-8] + struct.pack(">Q", 1) + b"x"))


def test_partial_reads_and_writes():
    class SlowIO(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 3))

        def write(self, data):
            return super().write(data[:3])

    stream = SlowIO()
    write_frame(stream, {"a": 1}, b"0123456789")
    stream.seek(0)
    assert read_frame(stream)[1] == b"0123456789"


def test_pack_unpack_arrays_no_pickle():
    arrays = {"byte": np.arange(12, dtype=np.uint8).reshape(3, 4),
              "depth": np.arange(6, dtype=np.uint16).reshape(1, 2, 3),
              "flow": np.linspace(-1, 1, 20, dtype=np.float32).reshape(2, 5, 2),
              "mask": np.array([[True, False]], bool), "empty": np.empty((0, 3), np.float32)}
    back = unpack_arrays(pack_arrays(arrays))
    for key, value in arrays.items():
        assert back[key].dtype == value.dtype and back[key].shape == value.shape
        np.testing.assert_array_equal(back[key], value)
    assert unpack_arrays(pack_arrays({})) == {}
    with pytest.raises(ValueError, match="pickle"):
        pack_arrays({"objects": np.array([{}], object)})
    unsafe = io.BytesIO()
    np.savez(unsafe, objects=np.array([{}], object))
    with pytest.raises(ValueError, match="allow_pickle=False"):
        unpack_arrays(unsafe.getvalue())


def test_pool_two_calls_match_file_runner(pool, arrays, tmp_path):
    hello = pool.session_info("mock")
    assert hello["protocol"] == PROTOCOL and hello["transport"] == "stream/1" and hello["worker"] == "mock"
    assert hello["pid"] > 0 and hello["source_commit"] == git_commit(str(REPO))
    assert hello["ram_backed"]["ok"] is True and hello["resources"]["ru_maxrss"] > 0
    assert hello.get("gpu", {}).get("device_count", 0) == 0
    spec = WorkerSpec("mock", sys.executable)
    file_response, expected = call_worker(spec, "echo_tracks", {}, arrays, tmp_path / "file", allow_mock=True)
    for seq in (1, 2):
        response, outputs = pool.call_worker(spec, "echo_tracks", {}, arrays, tmp_path / "unused", True,
                                             chunk_id=f"chunk{seq}")
        assert response["protocol"] == file_response["protocol"] and response["model"] == file_response["model"]
        assert response["spec"] == dataclasses.asdict(spec)
        assert response["meta"]["oracle"] and response["meta"]["resources"]["ru_maxrss"] > 0
        assert response["transport"]["seq"] == seq and response["transport"]["kind"] == "stream/1"
        assert response["transport"]["session"]["pid"] == hello["pid"]
        assert response["transport"]["elapsed_s"] >= 0
        for key in expected:
            np.testing.assert_array_equal(outputs[key], expected[key])
        pool.ack("mock", seq)
    assert pool.session_info("mock")["pid"] == hello["pid"]


@pytest.mark.parametrize("reason", ["mock", "commit", "handler"])
def test_worker_errors_are_not_retried(pool, arrays, monkeypatch, reason):
    starts = []
    original = StreamSession.__init__

    def counted(self, *args, **kwargs):
        starts.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(StreamSession, "__init__", counted)
    spec = WorkerSpec("mock", sys.executable, expected_commit="0" * 40 if reason == "commit" else None)
    match = {"mock": "mock output refused", "commit": "pinned", "handler": "unknown task"}[reason]
    with pytest.raises(WorkerError, match=match):
        pool.call_worker(spec, "bad_task" if reason == "handler" else "echo_tracks", {}, arrays,
                         allow_mock=reason != "mock", chunk_id="bad")
    assert len(starts) == 1
    assert "mock" not in pool._sessions


def test_no_next_call_before_ack(pool, arrays):
    spec = WorkerSpec("mock", sys.executable)
    pool.call_worker(spec, "echo_tracks", {}, arrays, allow_mock=True, chunk_id="one")
    with pytest.raises(WorkerError, match="ack seq 1"):
        pool.call_worker(spec, "echo_tracks", {}, arrays, allow_mock=True, chunk_id="two")
    pool.ack("mock", 1)
    response, _ = pool.call_worker(spec, "echo_tracks", {}, arrays, allow_mock=True, chunk_id="two")
    assert response["seq"] == 2
    pool.ack("mock", 2)


def test_server_request_before_ack_exits_three(config, tmp_path, arrays):
    session = StreamSession([sys.executable, "-I", str(MOCK), "--stream"], name="mock", log_path=None,
                            call_timeout_s=15, hello_timeout_s=30, cwd=Path(config["workdir"]))
    try:
        session.hello()
        session.call("echo_tracks", {}, arrays, chunk_id="one")
        write_frame(session.process.stdin, request(seq=2), pack_arrays(arrays))
        header, payload = read_frame(session.process.stdout)
        assert header["type"] == "error" and "expected ack for seq 1" in header["error"] and not payload
        assert session.process.wait(timeout=10) == 3
    finally:
        session.close()


def test_pool_reconnects_between_calls(pool, arrays):
    spec = WorkerSpec("mock", sys.executable)
    first, expected = pool.call_worker(spec, "echo_tracks", {}, arrays, allow_mock=True, chunk_id="one")
    pool.ack("mock", first["seq"])
    proc = pool._sessions["mock"].process
    proc.kill()
    proc.wait(timeout=10)
    response, actual = pool.call_worker(spec, "echo_tracks", {}, arrays, allow_mock=True, chunk_id="two")
    assert response["transport"]["session"]["pid"] != first["transport"]["session"]["pid"]
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    pool.ack("mock", response["seq"])


def test_pool_reconnects_during_call(pool, arrays, monkeypatch):
    pool.session_info("mock")
    victim = pool._sessions["mock"]
    original = stream_client.read_frame
    killed = []

    def interrupted(stream, *args, **kwargs):
        if stream is victim.process.stdout and not killed:
            killed.append(True)
            victim.process.kill()
            victim.process.wait(timeout=10)
            raise EOFError("injected mid-response disconnect")
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(stream_client, "read_frame", interrupted)
    response, output = pool.call_worker(WorkerSpec("mock", sys.executable), "echo_tracks", {}, arrays,
                                       allow_mock=True, chunk_id="same-after-interruption")
    assert killed and response["chunk_id"] == "same-after-interruption"
    np.testing.assert_array_equal(output["flows"], arrays["flows"])
    pool.ack("mock", response["seq"])


def test_retries_exhausted_are_bounded_and_logged(pool, arrays, tmp_path, monkeypatch):
    attempts = []

    def fail(self, *args, **kwargs):
        attempts.append(1)
        self.process.kill()
        raise TransportError("injected disconnect")

    monkeypatch.setattr(StreamSession, "call", fail)
    with pytest.raises(TransportError, match="injected disconnect"):
        pool.call_worker(WorkerSpec("mock", sys.executable), "echo_tracks", {}, arrays,
                         allow_mock=True, chunk_id="fails")
    assert len(attempts) == 3
    log = (tmp_path / "logs/mock.log").read_text()
    assert "retry 1/2" in log and "retry 2/2" in log


def test_source_commit_refused(config, tmp_path):
    config["expected_source_commit"] = "0" * 40
    instance = RemoteWorkerPool(config, log_dir=tmp_path / "logs")
    try:
        with pytest.raises(WorkerError, match="source commit"):
            instance.session_info("mock")
    finally:
        instance.close()


def test_ram_guard_refuses_disk_and_pool_never_falls_back(config, tmp_path, monkeypatch):
    monkeypatch.delenv("S4D_WORKER_ALLOW_DISK", raising=False)
    assert _mount_for(str(tmp_path), Path("/proc/self/mountinfo").read_text())["ok"] is False
    proc = subprocess.run([sys.executable, "-I", str(MOCK), "--stream"], cwd=tmp_path,
                          input=frame({"type": "hello"}), capture_output=True, timeout=15)
    header, _ = read_frame(io.BytesIO(proc.stdout))
    assert proc.returncode == 4 and header["type"] == "error" and not header["ram_backed"]["ok"]
    config.update(workdir=str(tmp_path), retries=0)
    instance = RemoteWorkerPool(config, log_dir=tmp_path / "logs")
    try:
        with pytest.raises((WorkerError, TransportError)):
            instance.session_info("mock")
        assert not instance._sessions
    finally:
        instance.close()


def test_ram_override_is_reported_but_refused_by_pool(config, tmp_path, monkeypatch):
    monkeypatch.setenv("S4D_WORKER_ALLOW_DISK", "1")
    session = StreamSession([sys.executable, "-I", str(MOCK), "--stream"], name="mock", log_path=None,
                            call_timeout_s=15, hello_timeout_s=30, cwd=tmp_path)
    try:
        ram = session.hello()["ram_backed"]
        assert ram["ok"] is False and ram["override"] is True
    finally:
        session.close()
    config["workdir"] = str(tmp_path)
    instance = RemoteWorkerPool(config, log_dir=tmp_path / "logs")
    try:
        with pytest.raises(WorkerError, match="RAM-backed"):
            instance.session_info("mock")
    finally:
        instance.close()


def test_ram_longest_mount_prefix_and_tmpdir(monkeypatch):
    mounts = ("1 0 0:1 / / rw - overlay overlay rw\n"
              "2 1 0:2 / /ram rw - tmpfs tmpfs rw\n"
              "3 2 0:3 / /ram/disk rw - ext4 /dev/x rw\n"
              "4 1 0:4 / /space\\040mount rw - ramfs ramfs rw\n")
    assert _mount_for("/ram/work", mounts)["ok"]
    assert not _mount_for("/ram/disk/episode", mounts)["ok"]
    assert not _mount_for("/ram2", mounts)["ok"]
    assert _mount_for("/space mount/work", mounts)["ok"]
    if Path("/dev/shm").is_dir():
        monkeypatch.chdir("/dev/shm")
        monkeypatch.setenv("TMPDIR", str(REPO))
        assert ram_backed()["cwd"]["ok"] and not ram_backed()["ok"]
        monkeypatch.setenv("TMPDIR", "/dev/shm/s4d-nonexistent-tmpdir/child")
        assert not ram_backed()["ok"]  # tempfile would otherwise fall back to a disk directory


@pytest.mark.parametrize("layout", ["sha", "loose", "packed", "gitdir", "worktree", "missing", "bad"])
def test_git_commit_fallback(tmp_path, monkeypatch, layout):
    monkeypatch.setenv("PATH", "")
    commit = "abcdef0123456789" * 2 + "abcdef01"
    dotgit = tmp_path / ".git"
    if layout in ("gitdir", "worktree"):
        gitdir = tmp_path / "metadata" / "worktrees" / "worker"
        gitdir.mkdir(parents=True)
        dotgit.write_text("gitdir: metadata/worktrees/worker\n")
    else:
        gitdir = dotgit
        gitdir.mkdir()
    if layout == "sha":
        (gitdir / "HEAD").write_text(commit + "\n")
    elif layout == "bad":
        (gitdir / "HEAD").write_text("not a commit\n")
    else:
        (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
        root = gitdir
        if layout == "worktree":
            (gitdir / "commondir").write_text("../..\n")
            root = gitdir.parent.parent
        if layout in ("packed", "worktree"):
            (root / "packed-refs").write_text(f"# pack-refs with: peeled\n{commit} refs/heads/main\n")
        elif layout != "missing":
            ref = root / "refs/heads/main"
            ref.parent.mkdir(parents=True)
            ref.write_text(commit + "\n")
    assert git_commit(str(tmp_path)) == (None if layout in ("missing", "bad") else commit)


def test_stream_handler_errors_prints_and_resources(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("S4D_WORKER_ALLOW_DISK", "1")
    monkeypatch.chdir(tmp_path)
    seen = []

    def handler(request, inputs):
        seen.append(request)
        print("handler progress on stderr")
        if request["task"] == "fail":
            raise ValueError("handler failure")
        return {"is_mock": True}, inputs, {"handled": True}

    incoming = io.BytesIO()
    write_frame(incoming, request(task="fail"), pack_arrays({}))
    write_frame(incoming, {"type": "ack", "seq": 1})
    write_frame(incoming, request(seq=2), pack_arrays({"x": np.arange(2)}))
    write_frame(incoming, {"type": "ack", "seq": 2})
    write_frame(incoming, {"type": "bye"})
    incoming.seek(0)
    outgoing = io.BytesIO()
    assert serve_stream(handler, incoming, outgoing, worker_name="unit") == 0
    captured = capsys.readouterr()
    assert not captured.out and "handler progress on stderr" in captured.err
    outgoing.seek(0)
    failed, payload = read_frame(outgoing)
    assert not failed["ok"] and "Traceback" in failed["error"] and not payload
    success, payload = read_frame(outgoing)
    assert success["ok"] and success["meta"]["resources"]["ru_maxrss"] > 0
    np.testing.assert_array_equal(unpack_arrays(payload)["x"], [0, 1])
    assert set(seen[0]) == {"protocol", "task", "params", "inputs", "outputs", "repo_dir", "checkpoint",
                            "expected_commit"}
    assert resources()["ru_maxrss"] > 0


@pytest.mark.parametrize("bad", [b"garbage", frame({"type": "ack", "seq": 1}), frame(request())[:-1]])
def test_stream_framing_errors_exit_three(monkeypatch, bad):
    monkeypatch.setenv("S4D_WORKER_ALLOW_DISK", "1")
    outgoing = io.BytesIO()
    assert serve_stream(lambda *_: None, io.BytesIO(bad), outgoing, worker_name="unit") == 3
    outgoing.seek(0)
    assert read_frame(outgoing)[0]["type"] == "error"
    assert serve_stream(lambda *_: None, io.BytesIO(), io.BytesIO(), worker_name="unit") == 0


def test_transport_timeout_kills_process(tmp_path):
    session = StreamSession([sys.executable, "-I", "-c", "import time; time.sleep(30)"], name="sleep",
                            log_path=None, call_timeout_s=0.05, hello_timeout_s=0.05)
    try:
        with pytest.raises(TransportError, match="timeout"):
            session.hello()
        assert session.process.poll() is not None
    finally:
        session.close()


@pytest.mark.parametrize("failure", ["frame", "seq", "chunk"])
def test_invalid_responses_kill_session(pool, arrays, monkeypatch, failure):
    pool.session_info("mock")
    session = pool._sessions["mock"]
    original = stream_client.read_frame

    def invalid(*args, **kwargs):
        header, payload = original(*args, **kwargs)
        if failure == "frame":
            raise FrameError("bad checksum")
        header["seq" if failure == "seq" else "chunk_id"] = -1
        return header, payload

    monkeypatch.setattr(stream_client, "read_frame", invalid)
    with pytest.raises(TransportError):
        session.call("echo_tracks", {}, arrays, chunk_id="invalid")
    assert session.process.poll() is not None


def test_stderr_is_drained_to_append_log(tmp_path):
    script = "import sys; sys.stderr.write('noise' * 100000); sys.stderr.flush()"
    log = tmp_path / "worker.log"
    log.write_bytes(b"previous\n")
    session = StreamSession([sys.executable, "-I", "-c", script], name="stderr", log_path=log,
                            call_timeout_s=10, hello_timeout_s=10)
    try:
        session.process.wait(timeout=10)
    finally:
        session.close()
    assert log.read_bytes() == b"previous\n" + b"noise" * 100000


def test_deadline_covers_blocked_request_write(tmp_path, arrays):
    session = StreamSession([sys.executable, "-I", "-c", "import time; time.sleep(30)"], name="blocked",
                            log_path=None, call_timeout_s=0.05, hello_timeout_s=0.05)
    try:
        with pytest.raises(TransportError, match="timeout"):
            session.call("echo_tracks", {}, {"large": np.zeros(1 << 20, np.uint8)}, chunk_id="blocked")
        assert session.process.poll() is not None
    finally:
        session.close()


def test_remote_argv_is_built_from_config_without_spawning(monkeypatch, tmp_path):
    captured = []

    class FakeSession:
        def __init__(self, argv, **kwargs):
            captured.append((argv, kwargs))

        def hello(self):
            return {"type": "hello", "protocol": PROTOCOL, "transport": "stream/1", "worker": "mock",
                    "ram_backed": {"ok": True}, "source_commit": "a" * 40}

        def close(self):
            pass

    monkeypatch.setattr(stream_client, "StreamSession", FakeSession)
    config = {"ssh": ["ssh", "-o", "BatchMode=yes", "test-host"], "container": "test-container",
              "workdir": "/ram/test", "source_root": "/opt/test-source", "expected_source_commit": "a" * 40,
              "workers": {"mock": {"python": "/opt/test-env/bin/python"},
                          "custom": {"python": "/opt/other/bin/python", "script": "/opt/custom.py"}}}
    instance = RemoteWorkerPool(config, log_dir=tmp_path / "logs")
    try:
        instance.session_info("mock")
        instance.session_info("mock")
        instance.session_info("custom")
        assert len(captured) == 2
        assert captured[0][0] == ["ssh", "-o", "BatchMode=yes", "test-host", "docker", "exec", "-i", "-w",
                                   "/ram/test", "test-container", "/opt/test-env/bin/python", "-I",
                                   "/opt/test-source/s4d/realworld/workers/mock_worker.py", "--stream"]
        assert captured[0][1]["cwd"] is None
        assert captured[1][0][-4:] == ["/opt/other/bin/python", "-I", "/opt/custom.py", "--stream"]
    finally:
        instance.close()
