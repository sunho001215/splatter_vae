from __future__ import annotations

import importlib.util
import json
import os
import select
import subprocess
import sys
import uuid
from pathlib import Path
from queue import Queue
from threading import Thread

import pytest

REPO = Path(__file__).resolve().parents[1]
HELD_COMMAND = [
    sys.executable, "-I", "-c",
    "import os,signal;os.closerange(3,65536);print('READY',flush=True);signal.pause()",
]


def load_query():
    spec = importlib.util.spec_from_file_location("query_under_test", REPO / "scripts/_nvidia_query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(query, pid):
    return {"pid": pid, "start_ticks": query._process_info(pid)[0], "boot_id": query._boot_id()}


def read_ready(descriptor):
    assert select.select([descriptor], [], [], 5)[0], "CPU child did not publish its readiness"
    value = os.read(descriptor, 1024)
    assert value, "CPU child exited before its readiness notification"
    return value


def test_cpu_query_success_and_failed_commands(tmp_path):
    query = load_query()
    assert query.query_output([sys.executable, "-I", "-c", "print('first')"], tmp_path) == "first\n"
    first = json.loads((tmp_path / query.STATE_NAME).read_text())
    assert not query._state_is_live(first)
    assert query.query_output([sys.executable, "-I", "-c", "print('second')"], tmp_path) == "second\n"
    second = json.loads((tmp_path / query.STATE_NAME).read_text())
    assert second["pid"] != first["pid"] and not query._OWNED_CHILDREN
    command = [sys.executable, "-I", "-c", "import sys;print('bad',file=sys.stderr);sys.exit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as failed:
        query.query_output(command, tmp_path)
    assert failed.value.returncode == 7 and failed.value.cmd == command and failed.value.stderr == "bad\n"
    with pytest.raises(subprocess.CalledProcessError):
        query.query_output([str(tmp_path / "nonexistent-command")], tmp_path)


def test_timeout_defers_reap_and_fresh_instance_uses_durable_identity(tmp_path, monkeypatch):
    query = load_query()
    real_popen = subprocess.Popen
    created, killed, communicated = [], [], []

    def launch(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append((proc, proc.kill, proc.wait))
        communicate = proc.communicate

        def bounded_communicate(*args, **kwargs):
            communicated.append(proc.pid)
            return communicate(*args, **kwargs)

        proc.communicate = bounded_communicate
        proc.kill = lambda: killed.append(proc.pid)
        proc.wait = lambda *args, **kwargs: pytest.fail("timeout cleanup must not wait")
        return proc

    monkeypatch.setattr(query.subprocess, "Popen", launch)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as timed_out:
            query.query_output(HELD_COMMAND, tmp_path, timeout=1)
        proc = created[0][0]
        assert b"READY" in timed_out.value.output
        assert communicated == [proc.pid] and killed == [proc.pid]
        assert proc.poll() is None and proc.stdout.closed and proc.stderr.closed
        state = json.loads((tmp_path / query.STATE_NAME).read_text())
        assert state == identity(query, proc.pid)
        fresh = load_query()
        with pytest.raises(fresh.QueryUnavailable, match="durable query process is still alive"):
            fresh.query_output(HELD_COMMAND, tmp_path, timeout=1)
        assert len(created) == 1 and communicated == [proc.pid] and killed == [proc.pid]
        created[0][1]()
        created[0][2](timeout=5)
        monkeypatch.setattr(query.subprocess, "Popen", real_popen)
        assert query.query_output([sys.executable, "-I", "-c", "print('recovered')"], tmp_path) == "recovered\n"
        assert not query._OWNED_CHILDREN
    finally:
        for proc, kill, wait in created:
            if proc.poll() is None:
                kill()
            wait(timeout=5)


def test_actual_timeout_kills_only_owned_child(tmp_path, monkeypatch):
    query = load_query()
    real_popen, real_kill = subprocess.Popen, os.kill
    children, signals = [], []

    def launch(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        children.append(proc)
        return proc

    def signal(pid, sig):
        signals.append(pid)
        return real_kill(pid, sig)

    monkeypatch.setattr(query.subprocess, "Popen", launch)
    monkeypatch.setattr(query.os, "kill", signal)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            query.query_output(HELD_COMMAND, tmp_path, timeout=1)
        assert signals == [children[0].pid]
        children[0].wait(timeout=5)
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


def test_cross_caller_flock_prevents_spawn_without_state(tmp_path, monkeypatch):
    query = load_query()
    holder = subprocess.Popen(
        [sys.executable, "-I", "-c",
         "import fcntl,os,sys;fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600);"
         "fcntl.flock(fd,fcntl.LOCK_EX);print('LOCKED',flush=True);sys.stdin.buffer.read(1)",
         str(tmp_path / query.LOCK_NAME)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    try:
        assert read_ready(holder.stdout.fileno()) == b"LOCKED\n"

        def forbidden(*args, **kwargs):
            pytest.fail("another caller's real lock must prevent spawning")

        monkeypatch.setattr(query.subprocess, "Popen", forbidden)
        for caller in (query, load_query()):
            with pytest.raises(caller.QueryUnavailable, match="shared lock"):
                caller.query_output(HELD_COMMAND, tmp_path)
        assert not (tmp_path / query.STATE_NAME).exists()
    finally:
        holder.communicate(input=b"x", timeout=5)


def test_inherited_lock_covers_parent_crash_before_publication_and_fd_closing_exec(tmp_path, monkeypatch):
    query = load_query()
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    wrapper = tmp_path / "publication_barrier.py"
    wrapper.write_text(
        "import os,runpy,sys\n"
        "namespace=runpy.run_path(sys.argv[1])\n"
        "sys.argv=sys.argv[1:]\n"
        "main=namespace['main']\n"
        "publish=main.__globals__['_publish_identity']\n"
        "def held_publish(state_dir):\n"
        f"    os.write({ready_write},str(os.getpid()).encode())\n"
        f"    os.read({release_read},1)\n"
        f"    os.close({ready_write})\n"
        f"    os.close({release_read})\n"
        "    publish(state_dir)\n"
        "main.__globals__['_publish_identity']=held_publish\n"
        "main()\n"
    )
    created = []
    real_popen = subprocess.Popen

    class CallerCrash(BaseException):
        pass

    def launch_then_crash(args, **kwargs):
        kwargs["pass_fds"] += (ready_write, release_read)
        proc = real_popen([*args[:3], str(wrapper), *args[3:]], **kwargs)
        created.append(proc)
        raise CallerCrash

    def forbidden(*args, **kwargs):
        pytest.fail("inherited lock or durable identity must prevent spawning")

    monkeypatch.setattr(query.subprocess, "Popen", launch_then_crash)
    try:
        with pytest.raises(CallerCrash):
            query.query_output(HELD_COMMAND, tmp_path)
        assert int(read_ready(ready_read)) == created[0].pid
        assert not (tmp_path / query.STATE_NAME).exists()
        monkeypatch.setattr(query.subprocess, "Popen", forbidden)
        fresh = load_query()
        with pytest.raises(fresh.QueryUnavailable, match="shared lock"):
            fresh.query_output(HELD_COMMAND, tmp_path)
        os.write(release_write, b"x")
        assert read_ready(created[0].stdout.fileno()) == b"READY\n"
        assert json.loads((tmp_path / query.STATE_NAME).read_text()) == identity(query, created[0].pid)
        fresh = load_query()
        with pytest.raises(fresh.QueryUnavailable, match="durable query process is still alive"):
            fresh.query_output(HELD_COMMAND, tmp_path)
        assert len(created) == 1 and not query._OWNED_CHILDREN
    finally:
        for descriptor in (ready_read, ready_write, release_read, release_write):
            os.close(descriptor)
        for proc in created:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            proc.stdout.close()
            proc.stderr.close()


@pytest.mark.parametrize("stale_kind", ["reused_pid", "changed_boot", "exited", "zombie"])
def test_stale_identity_clears_without_signalling_recorded_pid(tmp_path, monkeypatch, stale_kind):
    query = load_query()
    holder = subprocess.Popen(HELD_COMMAND, stdout=subprocess.PIPE)
    signals = []
    real_kill = os.kill
    try:
        assert read_ready(holder.stdout.fileno()) == b"READY\n"
        state = identity(query, holder.pid)
        if stale_kind == "reused_pid":
            state["start_ticks"] += 1
        elif stale_kind == "changed_boot":
            state["boot_id"] = str(uuid.uuid4())
        else:
            exits = Queue()

            def observe_exit():
                exits.put(os.waitid(os.P_PID, holder.pid, os.WEXITED | os.WNOWAIT))

            Thread(target=observe_exit, daemon=True).start()
            holder.kill()
            assert exits.get(timeout=5).si_pid == holder.pid
            if stale_kind == "zombie":
                assert query._process_info(holder.pid)[1] == "Z"
            else:
                holder.wait(timeout=5)
                assert query._process_info(holder.pid) is None
        (tmp_path / query.STATE_NAME).write_text(json.dumps(state))

        def signal(pid, sig):
            signals.append(pid)
            return real_kill(pid, sig)

        monkeypatch.setattr(query.os, "kill", signal)
        assert query.query_output([sys.executable, "-I", "-c", "print('new')"], tmp_path) == "new\n"
        assert not signals
        assert json.loads((tmp_path / query.STATE_NAME).read_text())["pid"] != holder.pid
        if stale_kind in ("reused_pid", "changed_boot"):
            assert holder.poll() is None
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=5)
        holder.stdout.close()


@pytest.mark.parametrize("state_text", [
    "{", "[]", "null", "{}",
    json.dumps({"pid": True, "start_ticks": 1, "boot_id": str(uuid.uuid4())}),
    json.dumps({"pid": -1, "start_ticks": 1, "boot_id": str(uuid.uuid4())}),
    json.dumps({"pid": 1, "start_ticks": -1, "boot_id": str(uuid.uuid4())}),
    json.dumps({"pid": 1, "start_ticks": True, "boot_id": str(uuid.uuid4())}),
    json.dumps({"pid": 1, "start_ticks": 1, "boot_id": "invalid"}),
])
def test_malformed_state_fails_unavailable_without_spawn_or_signal(tmp_path, monkeypatch, state_text):
    query = load_query()
    path = tmp_path / query.STATE_NAME
    path.write_text(state_text)

    def forbidden(*args, **kwargs):
        pytest.fail("malformed durable state must not spawn or signal")

    monkeypatch.setattr(query.subprocess, "Popen", forbidden)
    monkeypatch.setattr(query.os, "kill", forbidden)
    with pytest.raises(query.QueryUnavailable):
        query.query_output(HELD_COMMAND, tmp_path)
    assert path.read_text() == state_text


@pytest.mark.parametrize("failure", [
    "state_permission", "state_encoding", "proc_permission", "proc_missing_fields", "proc_wrong_pid",
    "proc_negative_ticks", "proc_bad_state", "boot_permission", "boot_invalid",
])
def test_unreadable_or_malformed_identity_information_fails_closed(tmp_path, monkeypatch, failure):
    query = load_query()
    state_path = tmp_path / query.STATE_NAME
    state_path.write_text(json.dumps(identity(query, os.getpid())))
    if failure == "state_encoding":
        state_path.write_bytes(b"\xff")
    read_text = Path.read_text
    proc_path = Path(f"/proc/{os.getpid()}/stat")
    boot_path = Path("/proc/sys/kernel/random/boot_id")

    def faulty_read(path, *args, **kwargs):
        if (failure == "state_permission" and path == state_path
            or failure == "proc_permission" and path == proc_path
            or failure == "boot_permission" and path == boot_path):
            raise PermissionError("test-controlled unreadable identity information")
        if path == proc_path:
            if failure == "proc_missing_fields":
                return "malformed"
            if failure == "proc_wrong_pid":
                return read_text(path).replace(str(os.getpid()), "999999", 1)
            if failure in ("proc_negative_ticks", "proc_bad_state"):
                prefix, _, suffix = read_text(path).rpartition(")")
                fields = suffix.split()
                fields[19 if failure == "proc_negative_ticks" else 0] = "-1" if failure == "proc_negative_ticks" else "?"
                return prefix + ") " + " ".join(fields)
        if failure == "boot_invalid" and path == boot_path:
            return "invalid"
        return read_text(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable identity information must not spawn or signal")

    monkeypatch.setattr(Path, "read_text", faulty_read)
    monkeypatch.setattr(query.subprocess, "Popen", forbidden)
    monkeypatch.setattr(query.os, "kill", forbidden)
    with pytest.raises(query.QueryUnavailable):
        query.query_output(HELD_COMMAND, tmp_path)
    assert state_path.exists()
