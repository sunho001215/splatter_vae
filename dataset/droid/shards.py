from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INDEX_SCHEMA_VERSION = 1
SHARD_COMPLETION_SCHEMA_VERSION = 1


def sha256_path(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


@dataclass(frozen=True)
class TarIndexEntry:
    offset: int
    size: int
    sha256: str


def shard_sidecars(tar_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    tar = Path(tar_path)
    if tar.suffix != ".tar":
        raise ValueError(f"Indexed shard must end in .tar: {tar}")
    return tar.with_suffix(".idx.json"), tar.with_suffix(".complete.json")


def quarantine_incomplete_shard(
    tar_path: str | os.PathLike[str],
    *,
    recovery_root: str | os.PathLike[str],
    schema_signature: str | None = None,
) -> tuple[Path, ...]:
    """Move only incomplete artifacts aside so a deterministic shard can resume.

    A verified complete shard is never touched. Moving instead of unlinking keeps
    interrupted bytes available for diagnosis and makes restart recovery
    reversible.
    """

    tar = Path(tar_path)
    if shard_is_complete(
        tar,
        schema_signature=schema_signature,
        verify_checksums=True,
    ):
        raise ValueError(f"Refusing to quarantine a complete shard: {tar}")
    index, completion = shard_sidecars(tar)
    candidates = (
        tar,
        index,
        completion,
        tar.with_suffix(tar.suffix + ".partial"),
        index.with_suffix(index.suffix + ".partial"),
        completion.with_suffix(completion.suffix + ".partial"),
    )
    existing = tuple(path for path in candidates if path.exists())
    if not existing:
        return ()
    destination = Path(recovery_root) / (f"{tar.stem}-{time.time_ns()}-{os.getpid()}")
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in existing:
        target = destination / path.name
        os.replace(path, target)
        moved.append(target)
    return tuple(moved)


class IndexedTarWriter:
    """Deterministic uncompressed TAR writer with byte-offset sidecar indexes."""

    def __init__(
        self,
        tar_path: str | os.PathLike[str],
        *,
        stage: str,
        shard_id: int,
        schema_signature: str,
    ) -> None:
        self.tar_path = Path(tar_path)
        self.index_path, self.complete_path = shard_sidecars(self.tar_path)
        self.partial_tar = self.tar_path.with_suffix(self.tar_path.suffix + ".partial")
        self.partial_index = self.index_path.with_suffix(
            self.index_path.suffix + ".partial"
        )
        self.tar_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            self.tar_path.exists()
            or self.index_path.exists()
            or self.complete_path.exists()
            or self.partial_tar.exists()
            or self.partial_index.exists()
        ):
            raise FileExistsError(
                f"Shard artifacts already exist for {self.tar_path}; completed shards "
                "must be skipped and incomplete artifacts inspected before recovery."
            )
        self.stage = str(stage)
        self.shard_id = int(shard_id)
        self.schema_signature = str(schema_signature)
        self._tar = tarfile.open(  # noqa: SIM115 - writer owns the long-lived handle
            self.partial_tar,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        )
        self._entries: dict[str, TarIndexEntry] = {}
        self._sample_keys: set[str] = set()
        self._finished = False

    def add(self, sample_key: str, suffix: str, payload: bytes) -> str:
        if self._finished:
            raise RuntimeError("Cannot add to a finished shard.")
        if not sample_key or "/" in sample_key or ".." in sample_key:
            raise ValueError(f"Unsafe sample key {sample_key!r}.")
        if not suffix or "/" in suffix or suffix.startswith("."):
            raise ValueError(f"Unsafe sample suffix {suffix!r}.")
        name = f"{sample_key}.{suffix}"
        if name in self._entries:
            raise KeyError(f"Duplicate TAR member {name!r}.")
        value = bytes(payload)
        info = tarfile.TarInfo(name=name)
        info.size = len(value)
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        import io

        header_offset = int(self._tar.offset)
        self._tar.addfile(info, io.BytesIO(value))
        self._entries[name] = TarIndexEntry(
            # USTAR emits exactly one 512-byte header for our short ASCII names.
            offset=header_offset + tarfile.BLOCKSIZE,
            size=len(value),
            sha256=hashlib.sha256(value).hexdigest(),
        )
        self._sample_keys.add(sample_key)
        return name

    def abort(self) -> None:
        if not self._finished:
            self._tar.close()
            self._finished = True

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("Shard writer was already finished.")
        self._tar.close()
        self._finished = True
        with self.partial_tar.open("rb") as stream:
            os.fsync(stream.fileno())
        index_payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "tar": self.tar_path.name,
            "stage": self.stage,
            "shard_id": self.shard_id,
            "schema_signature": self.schema_signature,
            "entries": {
                name: asdict(entry) for name, entry in sorted(self._entries.items())
            },
        }
        with self.partial_index.open("w", encoding="utf-8") as stream:
            json.dump(index_payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        _validate_offsets(self.partial_tar, index_payload)
        tar_sha256 = sha256_path(self.partial_tar)
        index_sha256 = sha256_path(self.partial_index)
        completion = {
            "schema_version": SHARD_COMPLETION_SCHEMA_VERSION,
            "stage": self.stage,
            "shard_id": self.shard_id,
            "schema_signature": self.schema_signature,
            "tar": self.tar_path.name,
            "tar_bytes": self.partial_tar.stat().st_size,
            "tar_sha256": tar_sha256,
            "index": self.index_path.name,
            "index_bytes": self.partial_index.stat().st_size,
            "index_sha256": index_sha256,
            "entry_count": len(self._entries),
            "sample_count": len(self._sample_keys),
        }
        os.replace(self.partial_tar, self.tar_path)
        os.replace(self.partial_index, self.index_path)
        write_json_atomic(self.complete_path, completion)
        return completion

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None and not self._finished:
            self.finish()
        elif not self._finished:
            self.abort()


def _validate_offsets(tar_path: Path, index: Mapping[str, Any]) -> None:
    size = tar_path.stat().st_size
    fd = os.open(tar_path, os.O_RDONLY)
    try:
        for name, raw in index["entries"].items():
            entry = TarIndexEntry(**raw)
            if entry.offset < 512 or entry.size < 0 or entry.offset + entry.size > size:
                raise ValueError(f"Indexed member {name!r} lies outside {tar_path}.")
            header = os.pread(fd, 512, entry.offset - 512)
            stored_name = header[:100].split(b"\0", 1)[0].decode("utf-8")
            if stored_name != name:
                raise ValueError(
                    f"TAR header/index mismatch at {entry.offset}: {stored_name!r} != {name!r}."
                )
    finally:
        os.close(fd)


class IndexedTarReader:
    def __init__(
        self,
        tar_path: str | os.PathLike[str],
        *,
        verify_completion: bool = True,
    ) -> None:
        self.tar_path = Path(tar_path)
        self.index_path, self.complete_path = shard_sidecars(self.tar_path)
        if verify_completion and not shard_is_complete(self.tar_path):
            raise ValueError(f"Shard is not verified complete: {self.tar_path}")
        with self.index_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if int(payload.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shard index schema in {self.index_path}.")
        self.metadata = {
            key: value for key, value in payload.items() if key != "entries"
        }
        self.entries = {
            name: TarIndexEntry(**entry) for name, entry in payload["entries"].items()
        }
        self._fd: int | None = None

    def _ensure_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.tar_path, os.O_RDONLY)
        return self._fd

    def read(self, sample_key: str, suffix: str, *, verify: bool = False) -> bytes:
        name = f"{sample_key}.{suffix}"
        try:
            entry = self.entries[name]
        except KeyError as exc:
            raise KeyError(
                f"Shard {self.tar_path.name} has no member {name!r}."
            ) from exc
        payload = os.pread(self._ensure_fd(), entry.size, entry.offset)
        if len(payload) != entry.size:
            raise OSError(f"Short read for {name!r} in {self.tar_path}.")
        if verify and hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise ValueError(f"Checksum mismatch for {name!r} in {self.tar_path}.")
        return payload

    def contains(self, sample_key: str, suffix: str) -> bool:
        return f"{sample_key}.{suffix}" in self.entries

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fd"] = None
        return state

    def __del__(self) -> None:
        self.close()


def shard_is_complete(
    tar_path: str | os.PathLike[str],
    *,
    schema_signature: str | None = None,
    verify_checksums: bool = False,
) -> bool:
    tar = Path(tar_path)
    index, completion_path = shard_sidecars(tar)
    if not tar.is_file() or not index.is_file() or not completion_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if int(completion["schema_version"]) != SHARD_COMPLETION_SCHEMA_VERSION:
            return False
        if (
            schema_signature is not None
            and completion["schema_signature"] != schema_signature
        ):
            return False
        if int(completion["tar_bytes"]) != tar.stat().st_size:
            return False
        if int(completion["index_bytes"]) != index.stat().st_size:
            return False
        if verify_checksums:
            if sha256_path(tar) != completion["tar_sha256"]:
                return False
            if sha256_path(index) != completion["index_sha256"]:
                return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def validate_indexed_shard(
    tar_path: str | os.PathLike[str], *, deep: bool = False
) -> dict[str, Any]:
    tar = Path(tar_path)
    if not shard_is_complete(tar, verify_checksums=True):
        raise ValueError(f"Shard completion checksum validation failed: {tar}")
    reader = IndexedTarReader(tar)
    with reader.index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    _validate_offsets(tar, index)
    if deep:
        for name, entry in reader.entries.items():
            payload = os.pread(reader._ensure_fd(), entry.size, entry.offset)
            if hashlib.sha256(payload).hexdigest() != entry.sha256:
                raise ValueError(f"Member checksum mismatch: {tar.name}:{name}")
    completion = json.loads(reader.complete_path.read_text(encoding="utf-8"))
    reader.close()
    return completion
