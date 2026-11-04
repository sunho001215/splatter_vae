"""The scheduler daemon re-executes itself only when its sources changed to the suite-tested fingerprint (2026-11-04)."""
import json
import os

from ops.jobs import scheduler as jobs


def test_reexec_decision():
    a, b, c = {"x": "1"}, {"x": "2"}, {"x": "3"}
    assert not jobs.daemon_reexec_needed(a, a, a)  # unchanged
    assert not jobs.daemon_reexec_needed(a, b, None)  # changed, no passing suite evidence
    assert not jobs.daemon_reexec_needed(a, b, c)  # changed, but not to the tested state
    assert jobs.daemon_reexec_needed(a, b, b)  # changed to exactly the tested state
    assert not jobs.daemon_reexec_needed(b, b, b)  # already running the tested state


def test_tested_source(tmp_path):
    path = tmp_path / "tests.json"
    assert jobs.tested_source(path) is None  # missing
    path.write_text("not json")
    assert jobs.tested_source(path) is None
    path.write_text(json.dumps({"source_sha256": {"a": "1"}, "full_suite_passed": False}))
    assert jobs.tested_source(path) is None
    path.write_text(json.dumps({"source_sha256": {"a": "1"}, "full_suite_passed": True}))
    assert jobs.tested_source(path) == {"a": "1"}


def test_pid_conflict(tmp_path):
    pidfile = tmp_path / "daemon.pid"
    assert jobs.daemon_pid_conflict(pidfile, os.getpid()) is None  # no file
    pidfile.write_text(str(os.getpid()))
    assert jobs.daemon_pid_conflict(pidfile, os.getpid()) is None  # a re-executed daemon keeps its pid
    pidfile.write_text(str(os.getppid()))
    assert jobs.daemon_pid_conflict(pidfile, os.getpid()) == os.getppid()  # another live process
    pidfile.write_text("999999999")
    assert jobs.daemon_pid_conflict(pidfile, os.getpid()) is None  # dead pid


def _loop(identities, tested, ticks):
    calls = {"tick": 0, "exec": 0, "sleep": 0}
    seq = iter(identities)

    def tick_fn():
        calls["tick"] += 1
        return []

    def exec_fn():
        calls["exec"] += 1

    jobs.run_daemon(0, tick_fn=tick_fn, identity_fn=lambda: next(seq), tested_fn=lambda: tested,
                    exec_fn=exec_fn, sleep_fn=lambda _: calls.__setitem__("sleep", calls["sleep"] + 1), max_ticks=ticks)
    return calls


def test_loop_reexecs_on_tested_change(capsys):
    n = jobs.DAEMON_SOURCE_CHECK_TICKS
    calls = _loop([{"v": 1}, {"v": 2}], {"v": 2}, 3 * n)
    assert calls["exec"] == 1 and calls["tick"] == n  # re-executed at the first check, no further tick
    assert "re-executing" in capsys.readouterr().out


def test_loop_keeps_code_on_untested_change(capsys):
    n = jobs.DAEMON_SOURCE_CHECK_TICKS
    calls = _loop([{"v": 1}] + [{"v": 2}] * 3, {"v": 1}, 3 * n)
    assert calls["exec"] == 0 and calls["tick"] == 3 * n
    out = capsys.readouterr().out
    assert out.count("keeping the loaded code") == 1  # warned once per new fingerprint


def test_loop_no_change_no_exec():
    n = jobs.DAEMON_SOURCE_CHECK_TICKS
    calls = _loop([{"v": 1}] * 4, {"v": 1}, 3 * n)
    assert calls["exec"] == 0 and calls["tick"] == 3 * n and calls["sleep"] == 3 * n
