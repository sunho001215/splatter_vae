"""Bound NVIDIA process queries without duplicating an unreapable child across restarts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

LOCK_NAME = "local_gpu_query.lock"
STATE_NAME = "local_gpu_query.json"
_OWNED_CHILDREN: dict[int, subprocess.Popen] = {}


class QueryUnavailable(subprocess.TimeoutExpired):
    def __init__(self, command: list[str], timeout: float, reason: str = "query could not be safely started"):
        super().__init__(command, timeout)
        self.reason = reason

    def __str__(self) -> str:
        return f"GPU query unavailable: {self.reason}"


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    uuid.UUID(value)
    return value


def _process_info(pid: int) -> tuple[int, str] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    prefix, separator, suffix = stat.rpartition(")")
    fields = suffix.split()
    if (
        not separator or not prefix.startswith(f"{pid} (")
        or len(fields) < 20 or fields[0] not in ("R", "S", "D", "Z", "T", "t", "X", "x", "K", "W", "P", "I")
        or not fields[19].isascii() or not fields[19].isdecimal()
    ):
        raise ValueError("Malformed process stat")
    return int(fields[19]), fields[0]


def _read_state(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if (
        not isinstance(value, dict)
        or type(value.get("pid")) is not int or value["pid"] <= 0
        or type(value.get("start_ticks")) is not int or value["start_ticks"] < 0
        or not isinstance(value.get("boot_id"), str)
    ):
        raise ValueError("Malformed durable GPU query identity")
    uuid.UUID(value["boot_id"])
    return value


def _state_is_live(identity: dict) -> bool:
    if identity["boot_id"] != _boot_id():
        return False
    info = _process_info(identity["pid"])
    return info is not None and info[0] == identity["start_ticks"] and info[1] not in ("Z", "X", "x")


def _publish_identity(state_dir: Path) -> None:
    info = _process_info(os.getpid())
    if info is None:
        raise RuntimeError("Cannot inspect query wrapper identity")
    identity = {"pid": os.getpid(), "start_ticks": info[0], "boot_id": _boot_id()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=state_dir, prefix="local_gpu_query-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(identity, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_dir / STATE_NAME)
        descriptor = os.open(state_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _close_pipes(proc: subprocess.Popen) -> None:
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()


def _reap_finished_owned_children() -> None:
    for pid, proc in list(_OWNED_CHILDREN.items()):
        if proc.poll() is not None:
            _close_pipes(proc)
            _OWNED_CHILDREN.pop(pid, None)


def query_output(command: list[str], state_dir: Path, timeout: float = 60) -> str:
    _reap_finished_owned_children()
    descriptor = None
    try:
        state_dir = Path(state_dir).resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(state_dir / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueryUnavailable(command, timeout, "another caller or query child holds the shared lock") from exc
        state_path = state_dir / STATE_NAME
        identity = _read_state(state_path)
        if identity is not None:
            if _state_is_live(identity):
                raise QueryUnavailable(command, timeout, "the durable query process is still alive")
            state_path.unlink()
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child", "--lock-fd", str(descriptor),
             "--state-dir", str(state_dir), "--", *command],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(descriptor,), text=True,
        )
        _OWNED_CHILDREN[proc.pid] = proc
        try:
            output, error = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Closing our FD does not unlock the child's inherited open-file description.
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            _close_pipes(proc)
            raise
        _OWNED_CHILDREN.pop(proc.pid, None)
        _close_pipes(proc)
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command, output=output, stderr=error)
        return output
    except (OSError, ValueError, IndexError) as exc:
        raise QueryUnavailable(
            command, timeout, f"cannot safely inspect or publish query state ({type(exc).__name__})",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a query command is required")
    descriptor_stat = os.fstat(args.lock_fd)
    lock_stat = (args.state_dir / LOCK_NAME).stat()
    if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
        raise RuntimeError("Query wrapper did not inherit the shared lock file")
    os.set_inheritable(args.lock_fd, True)
    _publish_identity(args.state_dir)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
