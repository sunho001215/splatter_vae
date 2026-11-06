"""P4 differential oracles: frozen parent commands, synthetic files, no process or network operations."""

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import importlib.util
import inspect
import io
import itertools
import json
import pickle
import pickletools
import signal
import struct
import sys
import types
import zipfile
from pathlib import Path, PurePosixPath

import pytest
import yaml

from ops.jobs import _checkpoint_support

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/checkpoint_pause"
COMMANDS = ("pause_pretrain_at_checkpoint", "pause_rl_at_checkpoint", "pause_remote_pretrain_at_checkpoint")
PRIMITIVES = {"identity", "record", "checkpoint_step", "session_alive"}


def source(command, side):
    path = FIXTURES / f"{command}.py.txt" if side == "before" else REPO / "ops/campaign" / f"{command}.py"
    return path.read_text()


def checkpoint(payload=None, member="archive/data.pkl"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
        archive.writestr(info, pickle.dumps({"step": 50000, "fixture": 0}, protocol=2) if payload is None else payload)
    return stream.getvalue()


def outcome(call):
    try:
        value = call()
        return {"type": type(value).__name__, "value": value}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc)}


class Sandbox:
    """Closed in-memory filesystem; only injected fakes can access process/transport operations."""

    def __init__(self, scenario="success"):
        self.events = []
        self.files = {}
        self.scenario = scenario
        self.dying = {}
        self.identities = {}
        self.stat_reads = {}
        self.ticks = 0
        self.sleeps = 0
        sandbox = self

        class FakePath:
            def __init__(self, value):
                self.path = PurePosixPath(str(value))

            def __str__(self):
                return str(self.path)

            def __truediv__(self, part):
                return FakePath(self.path / part)

            @property
            def name(self):
                return self.path.name

            def with_suffix(self, suffix):
                return FakePath(self.path.with_suffix(suffix))

            def relative_to(self, root):
                return self.path.relative_to(str(root))

            def read_bytes(self):
                name = str(self)
                sandbox.events.append(["read_bytes", name])
                if name.startswith("/proc/") and name.endswith("/stat"):
                    pid = int(self.path.parts[2])
                    sandbox.stat_reads[pid] = sandbox.stat_reads.get(pid, 0) + 1
                    if sandbox.scenario == "identity_changed" and pid == 41 and sandbox.stat_reads[pid] > 1:
                        return sandbox.stat(pid, start=9999).encode()
                    if pid in sandbox.identities:
                        result = sandbox.identities[pid]
                        if isinstance(result, Exception):
                            raise result
                        return result.encode()
                    if pid in sandbox.dying:
                        sandbox.dying[pid] -= 1
                        return sandbox.stat(pid, "S" if sandbox.dying[pid] > 0 else "Z").encode()
                if name not in sandbox.files:
                    raise FileNotFoundError(name)
                return sandbox.files[name]

            def read_text(self):
                return self.read_bytes().decode()

            def exists(self):
                sandbox.events.append(["exists", str(self)])
                return str(self) in sandbox.files

            def is_file(self):
                sandbox.events.append(["is_file", str(self)])
                return str(self) in sandbox.files

            def stat(self):
                sandbox.events.append(["stat", str(self)])
                return types.SimpleNamespace(st_size=len(sandbox.files[str(self)]), st_mtime=1700000000.0)

            def iterdir(self):
                assert str(self) == "/proc", f"unexpected enumeration: {self}"
                sandbox.events.append(["iterdir", str(self)])
                return iter(FakePath("/proc/" + name) for name in sandbox.proc_entries)

            def open(self, mode):
                name = str(self)
                sandbox.events.append(["open", name, mode])
                if sandbox.scenario == "lock_open_error" and name.endswith(".lock"):
                    raise PermissionError("synthetic lock open")
                if sandbox.scenario == "registry_open_error" and name.endswith("registry.jsonl"):
                    raise PermissionError("synthetic registry open")
                if mode == "x" and name in sandbox.files:
                    raise FileExistsError(name)
                if mode in {"a", "x"}:
                    sandbox.files.setdefault(name, b"")
                return Stream(self, mode)

        class Stream:
            def __init__(self, path, mode):
                self.path, self.mode = path, mode
                self.data = io.BytesIO(sandbox.files.get(str(path), b""))
                if mode == "a":
                    self.data.seek(0, 2)

            def __enter__(self):
                return self

            def __exit__(self, kind, value, traceback):
                sandbox.events.append(["close", str(self.path)])

            def seek(self, offset):
                sandbox.events.append(["seek", str(self.path), offset])
                return self.data.seek(offset)

            def read(self):
                value = self.data.read()
                sandbox.events.append(["read", str(self.path)])
                return value if "b" in self.mode else value.decode()

            def write(self, text):
                sandbox.events.append(["write", str(self.path), text])
                if sandbox.scenario == "write_error":
                    raise OSError("synthetic write")
                value = text.encode() if isinstance(text, str) else text
                self.data.write(value)
                sandbox.files[str(self.path)] = self.data.getvalue()
                return len(text)

            def flush(self):
                sandbox.events.append(["flush", str(self.path)])
                if sandbox.scenario == "flush_error":
                    raise OSError("synthetic flush")

            def fileno(self):
                sandbox.events.append(["fileno", str(self.path)])
                return 37

        self.Path = FakePath
        self.proc_entries = ["self", "90", "91", "92", "93", "41", "42"]
        self.identities = {
            90: FileNotFoundError("vanished"),
            91: IndexError("short stat"),
            92: self.stat(92, "X"),
            93: self.stat(93, "S", session=999),
        }
        self.clock = types.SimpleNamespace(time=self.time, monotonic=self.monotonic, sleep=self.sleep)
        self.os = types.SimpleNamespace(fsync=self.fsync, killpg=self.killpg)
        self.fcntl = types.SimpleNamespace(LOCK_EX=2, flock=self.flock)
        self.zipfile = types.SimpleNamespace(ZipFile=self.archive)

    @staticmethod
    def stat(pid, state="S", session=None, start=1234):
        fields = [state, "1", str(pid), str(pid if session is None else session)] + ["0"] * 15 + [str(start)]
        return f"{pid} (name with ) parentheses) " + " ".join(fields)

    def time(self):
        self.events.append(["time"])
        if self.scenario == "time_error":
            raise RuntimeError("synthetic time")
        return 1700000000.0

    def monotonic(self):
        self.events.append(["monotonic"])
        self.ticks += 1
        if self.scenario == "deadline" and self.ticks > 1:
            return 1e9
        if self.scenario == "wait_timeout" and self.ticks > 4:
            return 1e9
        return self.ticks / 100

    def sleep(self, seconds):
        self.events.append(["sleep", seconds])
        self.sleeps += 1
        if self.sleeps >= 12:
            raise RuntimeError("fixture failed to terminate")
        if self.scenario == "missed_boundary":
            self.files["/fixture/runs/pretrain/job/log.txt"] = b"[x] step 150 loss 1\n"
            self.files["/fixture/runs/pretrain/job/metrics.jsonl"] = b'{"step": 200, "val/loss": 1}\n'
            data = checkpoint(pickle.dumps({"step": 200, "fixture": 0}, protocol=2))
            self.files["/fixture/runs/pretrain/job/checkpoints/step_0000200.pt"] = data
            self.files["/fixture/runs/pretrain/job/checkpoints/latest.pt"] = data
        if self.scenario == "rl_missed_boundary":
            self.files["/fixture/runs/job/train.jsonl"] = b'{"step": 100000, "fixture": 0}\n'
            self.files["/fixture/runs/job/checkpoints/latest.pt"] = checkpoint(
                pickle.dumps({"step": 100000, "fixture": 0}, protocol=2)
            )

    def flock(self, stream, flag):
        self.events.append(["flock", str(stream.path), flag])
        if self.scenario == "flock_error":
            raise OSError("synthetic flock")

    def fsync(self, fd):
        self.events.append(["fsync", fd])
        if self.scenario == "fsync_error":
            raise OSError("synthetic fsync")

    def killpg(self, pid, sig):
        self.events.append(["killpg", pid, int(sig)])
        if self.scenario == "signal_error":
            raise ProcessLookupError("synthetic missing process")
        if self.scenario != "wait_timeout":
            self.dying[pid] = 2
            self.files.pop(f"/proc/{pid}/stat", None)

    def archive(self, path):
        self.events.append(["ZipFile", str(path)])
        return zipfile.ZipFile(io.BytesIO(path.read_bytes()))

    def namespace(self, command, side):
        def forbidden(*args, **kwargs):
            raise RuntimeError("sandbox blocked a real import or filesystem/code operation")

        safe_builtins = {**vars(builtins), **dict.fromkeys(("open", "__import__", "exec", "eval", "compile"), forbidden)}
        ns = {
            "__builtins__": safe_builtins,
            "__name__": "ops.campaign." + command,
            "__file__": str(REPO / "ops/campaign" / f"{command}.py"),
            "_checkpoint_support": _checkpoint_support,
            "Path": self.Path,
            "ROOT": self.Path("/fixture"),
            "REGISTRY": self.Path("/fixture/experiments/registry.jsonl"),
            "BOOT": "fixture-boot",
            "fcntl": self.fcntl,
            "os": self.os,
            "time": self.clock,
            "zipfile": self.zipfile,
            "json": json,
            "yaml": yaml,
            "hashlib": hashlib,
            "itertools": itertools,
            "pickletools": pickletools,
            "signal": signal,
            "re": __import__("re"),
            "shlex": __import__("shlex"),
            "sys": types.SimpleNamespace(argv=[command + ".py"], path=[]),
        }
        tree = ast.parse(source(command, side))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), command, "exec"), ns)
        return ns

    def snapshot(self, result, stdout=""):
        return {
            "result": result,
            "stdout": stdout,
            "events": self.events,
            "files": {
                k: v.decode()
                for k, v in sorted(self.files.items())
                if k.endswith(".json") or k.endswith(".jsonl") or k.endswith(".lock")
            },
        }


def primitive_capture(side):
    results = {}
    for command in COMMANDS:
        cases = {}
        for scenario in (
            "success",
            "lock_open_error",
            "flock_error",
            "registry_open_error",
            "time_error",
            "write_error",
            "flush_error",
            "fsync_error",
        ):
            box = Sandbox(scenario)
            ns = box.namespace(command, side)
            box.files[str(ns["REGISTRY"])] = b'{"prior": true}\n'
            event = {"event": "checkpoint_paused", "id": "fixture", "time": -1, "reason": "\u2603", "attempt": 3}
            result = outcome(lambda ns=ns, event=event: ns["record"](event))
            cases["record/" + scenario] = {**box.snapshot(result), "input_after": event}
        if command == COMMANDS[-1]:
            results[command] = cases
            continue
        payloads = {
            "BININT1": b"\x80\x02}q\x00(X\x04\x00\x00\x00stepq\x01K\xffs.",
            "BININT2": b"\x80\x02}q\x00(X\x04\x00\x00\x00stepq\x01M\xff\xffs.",
            "BININT": b"\x80\x02}q\x00(X\x04\x00\x00\x00stepq\x01J" + struct.pack("<i", -123) + b"s.",
            "wrong_key": pickle.dumps({"oops": 3}, protocol=2),
            "long_integer": pickle.dumps({"step": 2**40}, protocol=2),
            "boolean": pickle.dumps({"step": True}, protocol=2),
            "float": pickle.dumps({"step": 2.0}, protocol=2),
            "wrong_first_seven": pickle.dumps({"other": 0, "step": 3}, protocol=2),
            "short_ops": b"N.",
            "truncated": b"\x80\x02}q\x00(X\x04\x00",
            "empty_pickle": b"",
        }
        for label, payload in payloads.items():
            box = Sandbox()
            ns = box.namespace(command, side)
            box.files["/checkpoint.pt"] = checkpoint(payload)
            cases["checkpoint/" + label] = box.snapshot(
                outcome(lambda ns=ns, box=box: ns["checkpoint_step"](box.Path("/checkpoint.pt")))
            )
        for label, value in (
            ("missing_member", checkpoint(member="data.pkl")),
            ("bad_zip", b"bad zip"),
            ("truncated_zip", checkpoint()[:15]),
        ):
            box = Sandbox()
            ns = box.namespace(command, side)
            box.files["/checkpoint.pt"] = value
            cases["checkpoint/" + label] = box.snapshot(
                outcome(lambda ns=ns, box=box: ns["checkpoint_step"](box.Path("/checkpoint.pt")))
            )
        for label, value in (
            ("valid", Sandbox.stat(41)),
            ("vanished", None),
            ("malformed", "short"),
            ("invalid_integer", Sandbox.stat(41).replace(" 1234", " nope")),
        ):
            box = Sandbox()
            ns = box.namespace(command, side)
            if value is not None:
                box.files["/proc/41/stat"] = value.encode()
            cases["identity/" + label] = box.snapshot(outcome(lambda ns=ns: ns["identity"](41)))
        for label, value in (("alive", "S"), ("zombie", "Z"), ("dead", "X"), ("vanished", None)):
            box = Sandbox()
            box.proc_entries = ["self", "90", "91", "92", "93", "41"]
            ns = box.namespace(command, side)
            if value:
                box.files["/proc/41/stat"] = box.stat(41, value).encode()
            cases["session/" + label] = box.snapshot(outcome(lambda ns=ns: ns["session_alive"](41)))
        for label, error in (
            ("lookup", ProcessLookupError("gone")),
            ("index", IndexError("short")),
            ("permission", PermissionError("denied")),
            ("value", ValueError("invalid")),
        ):
            box = Sandbox()
            ns = box.namespace(command, side)

            def patched_identity(pid, error=error, box=box):
                box.events.append(["patched_identity", pid])
                if pid == 41:
                    return {"session": 41, "state": "S"}
                raise error

            ns["identity"] = patched_identity
            cases["patched_session/" + label] = box.snapshot(outcome(lambda ns=ns: ns["session_alive"](41)))
        results[command] = cases
    return results


def command_capture(command, side, scenario="success", argv=None):
    box = Sandbox(scenario)
    ns = box.namespace(command, side)
    job = "job"
    event = {"event": "launched", "id": job, "host": "local", "pid": 41, "attempt": 2}
    evaluator = {**event, "id": "job-eval", "pid": 42}
    box.files["/fixture/experiments/HOLD"] = b"hold"
    box.files[str(ns["REGISTRY"])] = (json.dumps(event) + "\n" + json.dumps(evaluator) + "\n").encode()
    box.files["/sys/fs/cgroup/memory.current"] = b"123456"
    for pid, script, arguments in (
        (41, "train.py" if command == COMMANDS[0] else "train_rl.py", "job"),
        (42, "eval_rl.py", "--train-job\0job\0"),
    ):
        box.files[f"/proc/{pid}/stat"] = box.stat(pid).encode()
        box.files[f"/proc/{pid}/cmdline"] = f"python\0scripts/{script}\0{arguments}".encode()
    cfg = {"train": {"save_every": 100, "eval_every": 100, "log_every": 10, "steps": 1000}}
    box.files["/fixture/runs/pretrain/job/config.yaml"] = yaml.safe_dump(cfg).encode()
    box.files["/fixture/runs/pretrain/job/log.txt"] = b"[x] step 50 loss 1\n"
    box.files["/fixture/runs/pretrain/job/metrics.jsonl"] = b'broken\n{"step": 100, "val/loss": 1}\n'
    data = checkpoint(pickle.dumps({"step": 100, "fixture": 0}, protocol=2))
    box.files["/fixture/runs/pretrain/job/checkpoints/step_0000100.pt"] = data
    box.files["/fixture/runs/pretrain/job/checkpoints/latest.pt"] = data
    box.files["/fixture/runs/job/train.jsonl"] = b'{"step": 49999}\n'
    box.files["/fixture/runs/job/eval.jsonl"] = b'{"step": 50000, "fixture": 0}\n'
    box.files["/fixture/runs/job/checkpoints/latest.pt"] = checkpoint()
    box.files["/fixture/runs/job/replay/ram_snapshot.npz"] = b"synthetic replay"
    box.files["/fixture/runs/job/config.yaml"] = yaml.safe_dump(cfg).encode()
    box.files["/fixture/runs/job/checkpoint_pause.json"] = b'{"checkpoint_step": 50000, "no_steps_lost": true}\n'
    if scenario != "evaluators":
        box.files.pop("/fixture/runs/job/checkpoint_pause.json")
    if scenario == "missed_boundary":
        # The first read arms 100; the second sees a later logged step and advances to 200.
        original = ns["logged_step"]
        calls = iter([50, 150])

        def logged(run):
            return next(calls, None) or original(run)

        ns["logged_step"] = logged
    if scenario == "rl_missed_boundary":
        original = ns["last_step"]
        calls = iter([49999, 50001])

        def last(path):
            return next(calls, None) or original(path)

        ns["last_step"] = last
        box.files["/fixture/runs/job/eval.jsonl"] = b'{"step": 100000, "fixture": 0}\n'
    if scenario == "bad_owner":
        box.files["/proc/41/cmdline"] = b"unowned"
    if scenario == "missing_hold":
        box.files.pop("/fixture/experiments/HOLD")
    if scenario == "proof_exists":
        name = "checkpoint_pause-fixture.json" if command == COMMANDS[-1] else "checkpoint_pause.json"
        box.files["/fixture/runs/job/" + name] = b"existing proof\n"
    if scenario == "bad_checkpoint":
        for path in list(box.files):
            if path.endswith(".pt"):
                box.files[path] = b"bad zip"
    ns["sys"].argv += (
        argv
        if argv is not None
        else (
            ["--reason", "fixture", job]
            if command == COMMANDS[-1]
            else ["--evaluators", "job-eval=50000"]
            if scenario == "evaluators"
            else [job]
        )
    )
    if command == COMMANDS[-1]:
        current = {"host": "remote", "status": "running", "attempt": 2, "container": "s4d-job-a2", "container_id": "a" * 64}
        if scenario == "bad_owner":
            current["container"] = "unowned"

        def read_registry():
            box.events.append(["read_registry"])
            return {job: current}

        ns["jobs"] = types.SimpleNamespace(read_registry=read_registry)

        class Client:
            def __init__(self):
                box.events.append(["RemoteClient"])

            def remote_path(self, path):
                box.events.append(["remote_path", path])
                return "/remote/" + path

            def ssh(self, text, **kwargs):
                box.events.append(["ssh", text, kwargs])
                if scenario == "transport_error":
                    raise RuntimeError("synthetic transport")
                lines = [
                    "ARMED 50 100",
                    "SIGNALLED 100 100 1700000000.0",
                    "EXITED 143",
                    "AFTER 100",
                    "STEP checkpoints/step_0000100.pt 100",
                    "STEP checkpoints/latest.pt 100",
                    "SHA " + "b" * 64 + "  checkpoints/step_0000100.pt",
                    "SHA " + "b" * 64 + "  checkpoints/latest.pt",
                ]
                if scenario == "bad_remote_result":
                    lines[3] = "AFTER 999"
                if scenario == "missing_remote_result":
                    lines = ["ARMED 50 100"]
                return types.SimpleNamespace(
                    returncode=7 if scenario == "remote_nonzero" else 0,
                    stdout="\n".join(lines) + "\n",
                    stderr="warning" * 100,
                )

        class Thread:
            def __init__(self, target, args):
                self.target, self.args = target, args
                box.events.append(["Thread", list(args[:2])])

            def start(self):
                box.events.append(["start"])
                self.target(*self.args)

            def join(self):
                box.events.append(["join"])

        ns["RemoteClient"] = Client
        ns["threading"] = types.SimpleNamespace(Thread=Thread)
        ns["__name__"] = "__main__"
    tree = ast.parse(source(command, side))
    ignored = {"ROOT", "REGISTRY", "BOOT", "_spec", "_checkpoint_support"}
    body = [
        n
        for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ignored for t in n.targets))
        and not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "_spec.loader.exec_module"
        )
    ]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = outcome(lambda: exec(compile(ast.Module(body=body, type_ignores=[]), command, "exec"), ns))
    return box.snapshot(result, output.getvalue())


def all_command_captures(side):
    results = {}
    scenarios = (
        "success",
        "bad_owner",
        "identity_changed",
        "signal_error",
        "wait_timeout",
        "deadline",
        "missing_hold",
        "bad_checkpoint",
        "proof_exists",
    )
    for command in COMMANDS[:2]:
        for scenario in scenarios + (
            ("missed_boundary",) if command == COMMANDS[0] else ("evaluators", "rl_missed_boundary")
        ):
            results[command + "/" + scenario] = command_capture(command, side, scenario)
    for scenario in (
        "success",
        "bad_owner",
        "remote_nonzero",
        "transport_error",
        "bad_remote_result",
        "missing_remote_result",
        "proof_exists",
    ):
        results[COMMANDS[-1] + "/" + scenario] = command_capture(COMMANDS[-1], side, scenario)
    for command in COMMANDS:
        for label, args in (
            ("empty", []),
            ("missing_reason", ["--reason"]),
            ("help", ["--help"]),
            ("missing_id", ["absent"]),
            ("bad_evaluator", ["--evaluators", "bad"]),
        ):
            results[command + "/cli_" + label] = command_capture(command, side, argv=args)
    return results


@pytest.mark.parametrize("command", COMMANDS)
def test_primitive_equivalence(command):
    assert primitive_capture("before")[command] == primitive_capture("after")[command]


@pytest.fixture(scope="module")
def command_captures():
    return {side: all_command_captures(side) for side in ("before", "after")}


@pytest.mark.parametrize("name", sorted(all_command_captures("before")))
def test_complete_pause_transcript(name, command_captures):
    before, after = command_captures["before"][name], command_captures["after"][name]
    assert after == before
    if (
        name.endswith("/success")
        or name.endswith("/evaluators")
        or name.endswith("/missed_boundary")
        or name.endswith("/rl_missed_boundary")
    ):
        assert before["result"] == {"type": "NoneType", "value": None}
        assert "checkpoint_paused" in before["stdout"]


@pytest.mark.parametrize("command", COMMANDS)
def test_original_api_and_orchestration_are_preserved(command):
    old, new = (ast.parse(source(command, side)) for side in ("before", "after"))
    old_defs = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_defs = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    assert old_defs.keys() == new_defs.keys()
    for name, node in old_defs.items():
        actual = new_defs[name]
        assert ast.dump(node.args) == ast.dump(actual.args)
        assert (ast.dump(node.returns) if node.returns else None) == (ast.dump(actual.returns) if actual.returns else None)
        if name not in PRIMITIVES:
            assert ast.dump(node) == ast.dump(actual)
        else:
            old_ns, new_ns = Sandbox().namespace(command, "before"), Sandbox().namespace(command, "after")
            for attr in ("__module__", "__qualname__", "__annotations__", "__doc__"):
                assert getattr(old_ns[name], attr) == getattr(new_ns[name], attr)
            assert inspect.signature(old_ns[name]) == inspect.signature(new_ns[name])
    # The only new import/global names are private loader/support bindings.
    added_imports = [
        n
        for n in new.body
        if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.dump(n) not in {ast.dump(x) for x in old.body}
    ]
    assert [ast.unparse(n) for n in added_imports] == ["import importlib.util as _importlib_util"]

    def orchestration(tree):
        return [
            ast.dump(n)
            for n in tree.body
            if not isinstance(n, ast.FunctionDef)
            and "_importlib_util" not in ast.unparse(n)
            and "_checkpoint_support" not in ast.unparse(n)
            and "_spec.loader" not in ast.unparse(n)
        ]

    assert orchestration(old) == orchestration(new)


def test_private_support_import_is_inert_and_has_no_dependencies(capsys):
    path = REPO / "ops/jobs/_checkpoint_support.py"
    tree = ast.parse(path.read_text())
    assert all(isinstance(n, (ast.FunctionDef, ast.Expr)) for n in tree.body)
    spec = importlib.util.spec_from_file_location("_isolated_p4_support", path)
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    spec.loader.exec_module(module)
    assert sys.path == before
    assert capsys.readouterr() == ("", "")
    assert {name for name in vars(module) if not name.startswith("_")} == PRIMITIVES


@pytest.mark.parametrize("command", COMMANDS)
def test_support_bootstrap_is_bound_to_command_checkout(command):
    tree = ast.parse(source(command, "after"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Import)
        and n.names[0].asname == "_importlib_util"
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in {"_spec", "_checkpoint_support"} for t in n.targets)
        or isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "_spec.loader.exec_module"
    ]
    ns = {"__file__": str(REPO / "ops/campaign" / f"{command}.py"), "Path": Path}
    before = list(sys.path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), command, "exec"), ns)
    assert Path(ns["_checkpoint_support"].__file__) == REPO / "ops/jobs/_checkpoint_support.py"
    assert sys.path == before


def test_independent_primitive_expectations():
    cases = primitive_capture("after")[COMMANDS[0]]
    for opcode, step in (("BININT1", 255), ("BININT2", 65535), ("BININT", -123)):
        assert cases["checkpoint/" + opcode]["result"] == {"type": "int", "value": step}
    assert cases["checkpoint/wrong_first_seven"]["result"] == {
        "exception": "RuntimeError",
        "message": "unexpected checkpoint step metadata",
    }
    assert cases["session/alive"]["result"] == {"type": "bool", "value": True}
    for state in ("zombie", "dead", "vanished"):
        assert cases["session/" + state]["result"] == {"type": "bool", "value": False}
    record = cases["record/success"]
    assert record["result"] == {"type": "NoneType", "value": None}
    assert [e[0] for e in record["events"]] == [
        "open",
        "flock",
        "open",
        "time",
        "write",
        "flush",
        "fileno",
        "fsync",
        "close",
        "close",
    ]
    assert record["input_after"]["time"] == -1
