"""Persistent download state (atomic JSON), the single-owner lock and owner liveness.

Item status: ``pending`` -> ``downloading`` -> ``done`` (plain file published) or ``downloaded`` (verified archive kept
for extraction) -> ``extracted`` (content verified and published, archive deleted). ``failed`` after the bounded
attempts, ``blocked`` when a gated source has no token. A restarted owner turns ``downloading`` back into ``pending``
and continues the partial temp file.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import threading
import time
from pathlib import Path

STATUSES = ("pending", "downloading", "downloaded", "extracted", "done", "failed", "blocked")
FINISHED = ("done", "extracted")


def write_json_atomic(path: Path, payload, indent: int | None = None) -> None:
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temp, "w") as out:
        json.dump(payload, out, indent=indent, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


class StateStore:
    """Per-item records plus free-form sections (streams, controller); thread-safe; saves are throttled."""

    def __init__(self, path: Path, item_ids, min_interval: float = 5.0):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.min_interval = min_interval
        self._saved_at = 0.0
        self._dirty = False
        data = read_json(self.path, {}) or {}
        self.items = data.get("items", {})
        self.sections = data.get("sections", {})
        for item_id in item_ids:
            self.items.setdefault(item_id, {"status": "pending", "attempts": 0})

    def get(self, item_id: str) -> dict:
        with self._lock:
            return dict(self.items[item_id])

    def update(self, item_id: str, **fields) -> dict:
        with self._lock:
            record = self.items[item_id]
            record.update(fields)
            record["updated"] = time.time()
            self._dirty = True
            return dict(record)

    def section(self, name: str) -> dict:
        with self._lock:
            return self.sections.setdefault(name, {})

    def set_section(self, name: str, value: dict) -> None:
        with self._lock:
            self.sections[name] = value
            self._dirty = True

    def by_status(self, *statuses) -> list[str]:
        with self._lock:
            return [k for k, v in self.items.items() if v["status"] in statuses]

    def save(self, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and (not self._dirty or now - self._saved_at < self.min_interval):
                return
            payload = {"format": "s4d_dataset_download_state_v1", "saved": time.time(), "items": self.items,
                       "sections": self.sections}
            write_json_atomic(self.path, payload)
            self._saved_at, self._dirty = now, False


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def process_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[19])  # field 22 (starttime) counted after the comm field


class OwnerLock:
    """Exclusive ``flock`` held for the owner's lifetime plus ``owner.json`` (pid, start ticks, boot id, tools sha)."""

    def __init__(self, state_dir: Path):
        self.lock_path = Path(state_dir) / "owner.lock"
        self.owner_path = Path(state_dir) / "owner.json"
        self._fd = None

    def acquire(self, **extra) -> bool:
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._fd = fd
        pid = os.getpid()
        write_json_atomic(self.owner_path, {"pid": pid, "start_ticks": process_start_ticks(pid), "boot_id": boot_id(),
                                            "host": socket.gethostname(), "started": time.time(), **extra})
        return True

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def held_by_other(self) -> bool:
        """True while some process holds the lock (probe without keeping it)."""
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def owner(self) -> dict | None:
        return read_json(self.owner_path)

    def live_owner(self) -> dict | None:
        """The recorded owner if it is this boot's same process (pid reuse rejected by start ticks) and the lock is
        held; otherwise None (a stale record after a crash or a reboot)."""
        record = self.owner()
        if not record or not self.held_by_other():
            return None
        if record.get("boot_id") != boot_id() or process_start_ticks(int(record["pid"])) != record.get("start_ticks"):
            return None
        return record
