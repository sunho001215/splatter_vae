"""Local orchestration of RAM-only worker streams over a configured subprocess command."""

from __future__ import annotations

import logging
import queue
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from s4d.realworld.workers import PROTOCOL, WorkerError, WorkerSpec
from s4d.realworld.workers._framing import MAX_PAYLOAD, pack_arrays, read_frame, unpack_arrays, write_frame

LOG = logging.getLogger(__name__)


class TransportError(RuntimeError):
    """A session failed and must be replaced before retrying a pure worker call."""


class StreamSession:
    """One serial worker connection, with at most one frame in flight.

    Callers must durably save outputs BEFORE ``ack(seq)``: the pipeline order is
    outputs -> fsync'd file + manifest -> ack. No next call is allowed until that ACK.
    Deadlines cover both writing the request and reading/verifying its full response.
    """

    def __init__(self, argv: list[str], *, name: str, log_path: Path | None,
                 call_timeout_s: float, hello_timeout_s: float, cwd: Path | None = None,
                 max_payload: int = MAX_PAYLOAD):
        self.name = name
        self.call_timeout_s = call_timeout_s
        self.hello_timeout_s = hello_timeout_s
        self.max_payload = max_payload
        self.seq = 0
        self.pending_seq: int | None = None
        self._closed = False
        self._hello: dict | None = None
        log = None
        try:
            if log_path is not None:
                log_path = Path(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("ab", buffering=0)
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, bufsize=0, cwd=cwd)
        except OSError as exc:
            if log is not None:
                log.close()
            raise TransportError(f"{name}: cannot start worker: {exc}") from exc

        def drain() -> None:
            sink = log
            try:
                while block := self.process.stderr.read(65536):
                    if sink is not None:
                        try:
                            sink.write(block)
                        except OSError:
                            # Keep draining even if local logging fails, so stderr cannot block the worker.
                            sink.close()
                            sink = None
            except (OSError, ValueError):
                pass
            finally:
                if sink is not None:
                    sink.close()
                self.process.stderr.close()

        self._stderr_thread = threading.Thread(target=drain, name=f"{name}-stderr", daemon=True)
        self._stderr_thread.start()

    def _abort(self) -> None:
        self._closed = True
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()

    def _run(self, operation: Callable, timeout: float):
        if self._closed:
            raise TransportError(f"{self.name}: session is closed")
        result = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                result.put((True, operation()))
            except Exception as exc:
                result.put((False, exc))

        # Only this operation reads stdout. It exits after one frame; no background read-ahead.
        reader = threading.Thread(target=run, name=f"{self.name}-exchange", daemon=True)
        reader.start()
        try:
            ok, value = result.get(timeout=timeout)
        except queue.Empty as exc:
            self._abort()
            raise TransportError(f"{self.name}: timeout after {timeout}s") from exc
        if not ok:
            self._abort()
            raise TransportError(f"{self.name}: {type(value).__name__}: {value}") from value
        return value

    def _exchange(self, header: dict, payload: bytes, timeout: float) -> tuple[dict, bytes]:
        if len(payload) > self.max_payload:
            self._abort()
            raise TransportError(f"{self.name}: request exceeds payload limit {self.max_payload}")

        def exchange():
            write_frame(self.process.stdin, header, payload)
            return read_frame(self.process.stdout, max_payload=self.max_payload)

        return self._run(exchange, timeout)

    def hello(self) -> dict:
        """Handshake once; return the server's provenance and resource report."""
        if self._hello is None:
            header, payload = self._exchange({"type": "hello"}, b"", self.hello_timeout_s)
            if payload:
                self._abort()
                raise TransportError(f"{self.name}: unexpected hello payload")
            self._hello = header
        return self._hello

    def call(self, task: str, params: dict, arrays: dict, *, chunk_id: str, repo_dir: str | None = None,
             checkpoint: str | None = None, expected_commit: str | None = None) -> tuple[dict, dict]:
        """Return a fully verified response; leave its ACK to the caller after durable saving."""
        if self.pending_seq is not None:
            raise WorkerError(f"{self.name}: ack seq {self.pending_seq} before the next call")
        self.seq += 1
        header = {"type": "request", "protocol": PROTOCOL, "seq": self.seq, "chunk_id": chunk_id, "task": task,
                  "params": params, "repo_dir": repo_dir, "checkpoint": checkpoint, "expected_commit": expected_commit}
        response, payload = self._exchange(header, pack_arrays(arrays), self.call_timeout_s)
        if (response.get("type") != "response" or type(response.get("seq")) is not int
                or response["seq"] != self.seq or response.get("chunk_id") != chunk_id):
            self._abort()
            raise TransportError(f"{self.name}: response type/seq/chunk mismatch: {response}")
        try:
            outputs = unpack_arrays(payload) if payload else {}
        except Exception as exc:
            self._abort()
            raise TransportError(f"{self.name}: invalid output archive: {exc}") from exc
        self.pending_seq = self.seq
        return response, outputs

    def ack(self, seq: int) -> None:
        """Acknowledge ONLY after outputs and their manifest have been durably saved."""
        if type(seq) is not int or seq != self.pending_seq:
            self._abort()
            raise TransportError(f"{self.name}: ack seq {seq} != pending {self.pending_seq}")
        self._run(lambda: write_frame(self.process.stdin, {"type": "ack", "seq": seq}), self.call_timeout_s)
        self.pending_seq = None

    def close(self) -> None:
        if self._closed:
            self._stderr_thread.join(timeout=1.0)
            return
        try:
            self._run(lambda: write_frame(self.process.stdin, {"type": "bye"}), min(2.0, self.call_timeout_s))
            self.process.wait(timeout=2.0)
        except (TransportError, subprocess.TimeoutExpired):
            pass
        finally:
            self._abort()
            self._stderr_thread.join(timeout=1.0)


class RemoteWorkerPool:
    """Lazily keep one serial session per worker; retry only failed transports.

    ``retries`` counts retries after the first attempt. A rejected worker response
    closes its session, so no rejected output is ACKed. A successful response needs
    an explicit ``pool.ack(name, response['transport']['seq'])`` after durable saving.
    ``workdir`` in ``call_worker`` is accepted for compatibility with the file runner;
    the session's server cwd is configured once via ``config['workdir']``.
    """

    def __init__(self, config: dict, *, log_dir: Path):
        self.config = config
        self.log_dir = Path(log_dir)
        self._sessions: dict[str, StreamSession] = {}
        self._info: dict[str, dict] = {}

    def _session(self, name: str) -> StreamSession:
        if name in self._sessions:
            return self._sessions[name]
        worker = self.config["workers"][name]
        script = worker.get("script") or f"{self.config['source_root']}/s4d/realworld/workers/{name}_worker.py"
        argv = [worker["python"], "-I", "-B", script, "--stream"]  # -B: -I ignores PYTHONDONTWRITEBYTECODE
        local = self.config.get("local_argv", False)
        workdir = self.config.get("workdir", "/ram/work")
        if not local:  # ssh joins its arguments into one remote shell command: quote everything after the host
            env = {**self.config.get("env", {}), **worker.get("env", {})}
            remote = ["docker", "exec", "-i", "-w", workdir, *[f"--env={k}={v}" for k, v in sorted(env.items())],
                      self.config["container"], *argv]
            argv = [*self.config["ssh"], " ".join(shlex.quote(str(a)) for a in remote)]
        session = StreamSession(argv, name=name, log_path=self.log_dir / f"{name}.log",
                                call_timeout_s=self.config.get("call_timeout_s", 3600),
                                hello_timeout_s=self.config.get("hello_timeout_s", 120),
                                cwd=Path(workdir) if local else None,
                                max_payload=self.config.get("max_payload", MAX_PAYLOAD))
        try:
            hello = session.hello()
            if hello.get("protocol") != PROTOCOL:
                raise WorkerError(f"{name}: hello protocol {hello.get('protocol')!r} != {PROTOCOL!r}")
            if hello.get("type") != "hello" or hello.get("transport") != "stream/1":
                raise WorkerError(f"{name}: hello refused: {hello.get('error', 'invalid stream greeting')}")
            if (hello.get("ram_backed") or {}).get("ok") is not True:
                raise WorkerError(f"{name}: RAM-backed worker required: {hello.get('ram_backed')}")
            expected = self.config.get("expected_source_commit")
            if expected and hello.get("source_commit") != expected:
                raise WorkerError(f"{name}: source commit {hello.get('source_commit')} != pinned {expected}")
        except Exception:
            session.close()
            raise
        self._sessions[name] = session
        self._info[name] = hello
        return session

    def session_info(self, name: str) -> dict:
        """Return the current session's hello for provenance, starting it if needed."""
        self._session(name)
        return self._info[name]

    def _discard(self, name: str) -> None:
        session = self._sessions.pop(name, None)
        self._info.pop(name, None)
        if session is not None:
            session.close()

    def call_worker(self, spec: WorkerSpec, task: str, params: dict, arrays: dict, workdir: Path | None = None,
                    allow_mock: bool = False, *, chunk_id: str) -> tuple[dict, dict]:
        """File-runner identity checks and return shape, plus stream provenance (no automatic ACK)."""
        retries = self.config.get("retries", 3)
        backoff = self.config.get("backoff_s", [5, 30, 120])
        for attempt in range(retries + 1):
            try:
                session = self._session(spec.name)
                response, outputs = session.call(task, params, arrays, chunk_id=chunk_id, repo_dir=spec.repo_dir,
                                                  checkpoint=spec.checkpoint, expected_commit=spec.expected_commit)
                break
            except TransportError as exc:
                self._discard(spec.name)
                if attempt == retries:
                    raise
                delay = backoff[min(attempt, len(backoff) - 1)] if backoff else 0
                message = f"{spec.name}: transport failure; retry {attempt + 1}/{retries} after {delay}s: {exc}"
                LOG.warning(message)
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / f"{spec.name}.log").open("a") as log:
                    log.write(message + "\n")
                time.sleep(delay)
        try:
            if response.get("protocol") != PROTOCOL:
                raise WorkerError(f"{spec.name}: protocol {response.get('protocol')!r} != {PROTOCOL!r}")
            if not response.get("ok"):
                raise WorkerError(f"{spec.name}: {response.get('error')}")
            model = response.get("model") or {}
            if model.get("is_mock") and not allow_mock:
                raise WorkerError(f"{spec.name}: mock output refused (allow_mock=False)")
            if spec.expected_commit and model.get("commit") != spec.expected_commit:
                raise WorkerError(f"{spec.name}: upstream commit {model.get('commit')} != pinned {spec.expected_commit}")
        except WorkerError:
            self._discard(spec.name)
            raise
        response["spec"] = asdict(spec)
        hello = self._info[spec.name]
        response["transport"] = {"kind": "stream/1", "session": {key: hello[key] for key in
                                 ("worker", "pid", "source_commit", "resources", "ram_backed", "gpu") if key in hello},
                                 "seq": response["seq"], "elapsed_s": response["elapsed_s"]}
        return response, outputs

    def ack(self, name: str, seq: int) -> None:
        """Forward a durable-save ACK to the same session that produced the response."""
        if name not in self._sessions:
            raise TransportError(f"{name}: no live session to acknowledge")
        try:
            self._sessions[name].ack(seq)
        except TransportError:
            self._discard(name)
            raise

    def close(self) -> None:
        for name in list(self._sessions):
            self._discard(name)
