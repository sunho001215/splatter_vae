"""Explicit local placement and verified inputs for real-world offline FQL (directive P, 2026-11-11 ~18:10).

Only never-launched training rows move. Preprocessing and dataset construction still execute remotely, where their
source data lives. The queue's arguments and scientific fingerprint never change: the scheduler maps input paths at
launch and records their content identities. Compact descriptors retain their byte-identical remote originals beside
the localized copies; every referenced episode file is inventoried. Large files use the verified transfer's size and
mtime at admission, small files are hashed again. ``verify`` can audit every byte independently of admission.

Placement compares the pinned release's execution scope once, under the placement locks, and records the digests in
both ownership and pool records. Launch compares that scope against the tick's source identity without git reads.
No campaign driver is imported here. Network access is confined to the explicitly requested plan/fetch operations;
remote commands only read files, and downloads use RemoteClient's shared budget and verified atomic placement.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.jobs import host_roles  # noqa: E402
from s4d.rwlearn.configs import REMOTE_REALWORLD_ROOT  # noqa: E402

REMOTE_PREFIX = REMOTE_REALWORLD_ROOT
MIRROR = REPO / "runs/realworld-inputs"
MANIFEST = MIRROR / "MANIFEST.json"
FINISHED_DIR = REPO / "runs/setup/realworld-local-finished"
SMALL_HASH_BYTES = 64 * 2**20
REMOTE_LOCK = REPO / "runs/setup/remote-driver.lock"


class LocalInputError(ValueError):
    """An input or release identity cannot be proved locally; admission must wait."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in value or any(c in value for c in "\n\r\0")
            or "\\" in value):
        raise LocalInputError(f"unsafe relative input path: {value!r}")
    return str(path)


def _mapped(value: str, remote_prefix: str = REMOTE_PREFIX) -> str | None:
    if value == remote_prefix or value.startswith(remote_prefix + "/"):
        return _rel(value[len(remote_prefix):].lstrip("/") or ".")
    if value.startswith("/home/compu/"):
        raise LocalInputError(f"unmappable remote input: {value}")
    return None


def _value(arg) -> str:
    text = str(arg)
    key, separator, value = text.partition("=")
    return value if separator and "/" not in key else text


def applies(job: dict, roles: dict, by_id: dict | None = None) -> bool:
    """Pure policy check: this FQL script is admitted locally, independent of ownership and placement."""
    return (host_roles.job_class(job, by_id).cls == "realworld_fql"
            and host_roles.refusal({**job, "host": "local"}, "local", roles, {}, by_id) is None)


def needed_inputs(job: dict, *, remote_prefix: str = REMOTE_PREFIX) -> dict:
    mirror, dependencies = set(), set()
    for arg in job.get("args", []):
        value = _value(arg)
        relative = _mapped(value, remote_prefix)
        if relative is not None:
            mirror.add(relative)
        parts = PurePosixPath(value).parts
        if len(parts) >= 3 and parts[0] == "runs" and parts[1] in job.get("deps", []):
            dependencies.add((parts[1], _rel("/".join(parts[2:]))))
    return {"mirror": sorted(mirror), "dependency_files": sorted(dependencies)}


def _manifest(path: Path, remote_prefix: str = REMOTE_PREFIX, *, raw: bytes | None = None) -> dict:
    if raw is None and not path.exists():
        return {"version": 1, "remote_root": remote_prefix, "local_root": "runs/realworld-inputs", "files": {}}
    spec = json.loads(path.read_bytes() if raw is None else raw)
    if (spec.get("version") != 1 or spec.get("remote_root") != remote_prefix
            or spec.get("local_root") != "runs/realworld-inputs" or not isinstance(spec.get("files"), dict)):
        raise LocalInputError("invalid real-world mirror manifest")
    for name, item in spec["files"].items():
        if _rel(name) != name or item.get("kind") not in ("copy", "localized"):
            raise LocalInputError(f"invalid mirror entry: {name}")
    return spec


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def _json(path: Path, spec: dict) -> None:
    _atomic(path, (json.dumps(spec, indent=1, sort_keys=True) + "\n").encode())


def _regular(path: Path) -> None:
    # Reject symlinked parents too: checking only the leaf would allow a mirror to escape its declared root.
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise LocalInputError(f"symlink input: {path}")
    if not path.is_file():
        raise LocalInputError(f"missing mirror file: {path}")


def _descriptor(data: bytes, *, mirror: Path = MIRROR, remote_prefix: str = REMOTE_PREFIX) -> tuple[bytes, list[str]]:
    """Rewrite only scalar text inside ``paths``; preserve all other descriptor bytes and YAML formatting.

    YAML node marks give the original scalar spans, so quoting, comments, ordering and every unrelated field survive.
    Requiring the prefix to occur literally also fails closed for an escaped or folded spelling we cannot substitute.
    """
    text = data.decode()
    spec = yaml.safe_load(text)
    if (not isinstance(spec, dict) or spec.get("kind") != "compact" or not isinstance(spec.get("paths"), list)
            or not spec["paths"]):
        raise LocalInputError("store.yaml must be a compact descriptor with paths")
    tree = yaml.compose(text)
    nodes = [value for key, value in tree.value if key.value == "paths"]
    if len(nodes) != 1 or not isinstance(nodes[0], yaml.SequenceNode):
        raise LocalInputError("invalid compact descriptor paths")
    changes, directories = [], []
    for node in nodes[0].value:
        value = node.value
        if not isinstance(node, yaml.ScalarNode) or not isinstance(value, str):
            raise LocalInputError("invalid compact descriptor path")
        relative = _mapped(value, remote_prefix)
        if relative is None or relative == ".":
            raise LocalInputError(f"compact episode outside remote root: {value}")
        raw = text[node.start_mark.index:node.end_mark.index]
        if remote_prefix not in raw:
            raise LocalInputError("compact path prefix must be literal")
        changes.append((node.start_mark.index, node.end_mark.index, raw.replace(remote_prefix, str(mirror), 1)))
        directories.append(relative)
    for start, end, replacement in reversed(changes):
        text = text[:start] + replacement + text[end:]
    return text.encode(), directories


def _remote_batch(rels: list[str], directories: list[str], client, remote_prefix: str) -> tuple[dict, dict]:
    """One read-only SSH per batch, with NUL-framed records (filenames and descriptor newlines stay unambiguous)."""
    files = " ".join(shlex.quote(_rel(r)) for r in rels)
    dirs = " ".join(shlex.quote(_rel(r)) for r in directories)
    # The path walk rejects symlinked parents before stat, cat or find. No remote file is created or changed.
    guard = ('check_path() { p="$1"; [ ! -L . ]; while [ "$p" != . ]; do '
             '[ ! -L "$p" ] || return 1; case "$p" in */*) p="${p%/*}";; *) p=.;; esac; done; }; ')
    emit = ('emit() { f="${1#./}"; check_path "$f"; [ -f "$f" ]; '
            'printf "file\\0%s\\0" "$f"; stat -c "%s" -- "$f"; sha256sum -- "$f"; '
            'printf "\\0"; case "$f" in */store.yaml|store.yaml) '
            'printf "store\\0%s\\0" "$f"; cat -- "$f"; printf "\\0";; esac; }; ')
    command = "set -euo pipefail; " + guard + emit
    command += f"for f in {files}; do emit \"$f\"; done; " if files else ""
    # find's output is NUL-delimited, including arbitrary spaces. Bash read -d is used explicitly via bash -c below.
    if dirs:
        command += (f'for d in {dirs}; do check_path "$d"; [ -d "$d" ]; '
                    'find "$d" -maxdepth 1 -type f -print0 | '
                    'while IFS= read -r -d "" f; do emit "$f"; done; done; ')
    output = client.checked_ssh("cd " + shlex.quote(remote_prefix) + " && bash -c " + shlex.quote(command),
                                idempotent=True).stdout
    parts = output.decode().split("\0") if isinstance(output, bytes) else output.split("\0")
    found, descriptors = {}, {}
    while parts and parts[-1] == "":
        parts.pop()
    if len(parts) % 3:
        raise LocalInputError("invalid remote input inventory")
    for i in range(0, len(parts), 3):
        kind, name, body = parts[i:i + 3]
        name = _rel(name)
        if kind == "file":
            size, checksum = body.split("\n", 1)
            digest = checksum[:64]
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or int(size) < 0:
                raise LocalInputError("invalid remote checksum")
            found[name] = {"bytes": int(size), "sha256": digest, "kind": "copy", "requires": []}
        elif kind == "store":
            descriptors[name] = body.encode()
        else:
            raise LocalInputError("invalid remote inventory record")
    return found, descriptors


def remote_plan(rels, client, *, remote_prefix: str = REMOTE_PREFIX, mirror: Path = MIRROR) -> dict:
    """Inventory requested files, then compact episodes in a second batch after parsing descriptors locally.

    Each batch uses one idempotent checked_ssh (stat/sha256sum, plus cat/find for compact stores). YAML is parsed only
    locally; no remote interpreter or descriptor-derived shell expression is executed. Episode directories must
    contain both the completion marker and episode.h5; every regular top-level file is included in ``requires``.
    """
    wanted = sorted({_rel(r) for r in rels})
    if not wanted:
        return {}
    found, descriptors = _remote_batch(wanted, [], client, remote_prefix)
    if not set(wanted) <= found.keys():
        raise LocalInputError("remote inputs are missing")
    directories = {}
    for name in wanted:
        if PurePosixPath(name).name != "store.yaml":
            continue
        data = descriptors.get(name)
        if data is None or _sha(data) != found[name]["sha256"] or len(data) != found[name]["bytes"]:
            raise LocalInputError(f"remote descriptor changed during inventory: {name}")
        _, directories[name] = _descriptor(data, mirror=mirror, remote_prefix=remote_prefix)
    episodes, _ = _remote_batch([], sorted({d for ds in directories.values() for d in ds}), client, remote_prefix) \
        if directories else ({}, {})
    for name, dirs in directories.items():
        requires = sorted(r for r in episodes if str(PurePosixPath(r).parent) in dirs)
        for directory in dirs:
            if not {directory + "/_COMPLETE.json", directory + "/episode.h5"} <= set(requires):
                raise LocalInputError(f"incomplete compact episode: {directory}")
        found[name].update(kind="localized", requires=requires)
    found.update(episodes)
    return found


def _fetch_client():
    from ops.remote.access import RemoteClient
    return RemoteClient(repo=REPO)


def fetch(rels, client=None, priority="bulk", dry_run=False, *, mirror: Path = MIRROR,
          manifest: Path | None = None, remote_prefix: str = REMOTE_PREFIX) -> dict:
    """Fetch mismatches through the shared download budget; publish only fully hashed files and descriptors.

    A descriptor's original lands under _remote rather than overwriting its localized sibling. A failed transfer
    leaves its manifest entry unpublished; the next fetch can resume RemoteClient's staging files.
    """
    client = client or _fetch_client()
    mirror, manifest = Path(mirror), Path(manifest or Path(mirror) / "MANIFEST.json")
    spec = _manifest(manifest, remote_prefix)
    inventory = remote_plan(rels, client, remote_prefix=remote_prefix, mirror=mirror)
    groups, present, pending = {}, [], []
    for relative, item in inventory.items():
        target = mirror / ("_remote" if item["kind"] == "localized" else "") / relative
        try:
            _regular(target)
            matches = target.stat().st_size == item["bytes"] and _hash(target) == item["sha256"]
        except LocalInputError:
            matches = False
        if matches:
            present.append(relative)
        else:
            pending.append(relative)
            parent = str(PurePosixPath(relative).parent)
            groups.setdefault((parent, item["kind"] == "localized"), {})[PurePosixPath(relative).name] = item["sha256"]
    result = {"fetched_bytes": 0, "already_present": sorted(present), "missing": sorted(pending)}
    if dry_run:
        result["fetched_bytes"] = sum(inventory[r]["bytes"] for r in pending)
        return result
    for (directory, original), wanted in sorted(groups.items()):
        target = mirror / ("_remote" if original else "") / directory
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise LocalInputError(f"symlink mirror directory: {target}")
        client.download_batch(remote_prefix + "/" + directory, target, wanted,
                              name="realworld-" + _sha(str(target).encode())[:24], priority=priority)
    stamp, missing = host_roles.now(), []
    for relative, item in inventory.items():
        original = "_remote/" + relative if item["kind"] == "localized" else relative
        target = mirror / original
        try:
            _regular(target)
            if target.stat().st_size != item["bytes"] or _hash(target) != item["sha256"]:
                raise LocalInputError(f"download checksum mismatch: {relative}")
        except LocalInputError:
            missing.append(relative)
            continue
        stat = target.stat()
        entry = {"kind": "copy", "sha256": item["sha256"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "remote_sha256": item["sha256"], "remote_bytes": item["bytes"], "fetched_at": stamp, "requires": []}
        spec["files"][original] = entry
        if relative in pending:
            result["fetched_bytes"] += stat.st_size
    # Publish localized files only when the entire episode closure is available, including previously fetched files.
    for relative, item in inventory.items():
        if item["kind"] != "localized" or relative in missing or set(item["requires"]) & set(missing):
            continue
        data, _ = _descriptor((mirror / "_remote" / relative).read_bytes(), mirror=mirror, remote_prefix=remote_prefix)
        target = mirror / relative
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise LocalInputError(f"symlink descriptor: {target}")
        if not target.is_file() or target.read_bytes() != data:
            _atomic(target, data)
        stat = target.stat()
        spec["files"][relative] = {"kind": "localized", "sha256": _hash(target), "bytes": stat.st_size,
                                  "mtime_ns": stat.st_mtime_ns, "remote_sha256": item["sha256"],
                                  "remote_bytes": item["bytes"], "fetched_at": stamp, "requires": item["requires"]}
    _json(manifest, spec)
    result["missing"] = sorted(missing)
    return result


def _check(relative: str, files: dict, mirror: Path, seen: set, *, full=False, quick=False,
           remote_prefix: str = REMOTE_PREFIX) -> str:
    relative = _rel(relative)
    item = files.get(relative)
    if item is None:
        raise LocalInputError(f"input not mirrored: {relative}")
    if relative in seen:
        return item["sha256"]
    path = mirror / relative
    _regular(path)
    stat = path.stat()
    if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
        raise LocalInputError(f"mirror size/mtime changed: {relative}")
    if not quick and (full or stat.st_size <= SMALL_HASH_BYTES) and _hash(path) != item["sha256"]:
        raise LocalInputError(f"mirror sha256 mismatch: {relative}")
    seen.add(relative)
    if item["kind"] == "copy" and item["sha256"] != item["remote_sha256"]:
        raise LocalInputError(f"copy differs from remote checksum: {relative}")
    for required in item.get("requires", []):
        _check(required, files, mirror, seen, full=full, quick=quick, remote_prefix=remote_prefix)
    if item["kind"] == "localized":
        original = "_remote/" + relative
        _check(original, files, mirror, seen, full=full, quick=quick, remote_prefix=remote_prefix)
        data = (mirror / original).read_bytes()
        localized, dirs = _descriptor(data, mirror=mirror, remote_prefix=remote_prefix)
        if (_sha(data) != item["remote_sha256"] or len(data) != item["remote_bytes"]
                or localized != path.read_bytes()):
            raise LocalInputError(f"localized descriptor differs from original: {relative}")
        required = set(item.get("requires", []))
        for directory in dirs:
            if not {directory + "/_COMPLETE.json", directory + "/episode.h5"} <= required:
                raise LocalInputError(f"incomplete descriptor requirements: {relative}")
    return item["sha256"]


def verify(full=True, *, mirror: Path = MIRROR, manifest: Path | None = None,
           remote_prefix: str = REMOTE_PREFIX) -> list[str]:
    problems = []
    try:
        files = _manifest(Path(manifest or Path(mirror) / "MANIFEST.json"), remote_prefix)["files"]
        if not Path(manifest or Path(mirror) / "MANIFEST.json").is_file():
            return ["missing real-world mirror manifest"]
        seen = set()
        for relative in sorted(files):
            try:
                # Quick verification uses metadata only; descriptor originals are still compared textually.
                _check(relative, files, Path(mirror), seen, full=full, quick=not full, remote_prefix=remote_prefix)
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                problems.append(str(exc))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        problems.append(str(exc))
    return problems


def _inputs(job: dict, state: dict, mirror: Path, manifest: Path, runs: Path,
            remote_prefix: str = REMOTE_PREFIX) -> tuple[list, dict]:
    needed = needed_inputs(job, remote_prefix=remote_prefix)
    raw = manifest.read_bytes() if manifest.is_file() else None
    files, inputs, dependencies = _manifest(manifest, remote_prefix, raw=raw)["files"], {}, {}
    for arg in job.get("args", []):
        value = _value(arg)
        relative = _mapped(value, remote_prefix)
        if relative is not None:
            inputs[value] = _check(relative, files, mirror, set(), remote_prefix=remote_prefix)
    for dep, relative in needed["dependency_files"]:
        current = state.get(dep, {})
        if current.get("host") != "remote":
            continue
        name, path = f"runs/{dep}/{relative}", runs / dep / relative
        _regular(path)
        expected = current.get("sync", {}).get("files", {}).get(relative)
        digest = _hash(path)
        remote_files = current.get("remote_verified", {}).get("files")
        if (current.get("status") != "done" or not expected or digest != expected
                or (remote_files is not None and remote_files.get(relative) != digest)):
            raise LocalInputError(f"dependency dataset sha256 mismatch or unverified backup: {name}")
        dependencies[name] = digest
    effective = []
    for arg in job.get("args", []):
        value = _value(arg)
        relative = _mapped(value, remote_prefix)
        effective.append(str(arg)[:len(str(arg)) - len(value)] + str(mirror) + value[len(remote_prefix):]
                         if relative is not None else arg)
    return effective, {"input_map": {"from": remote_prefix, "to": str(mirror)}, "manifest_sha256": _sha(raw or b""),
                       "inputs": inputs, "dependency_inputs": dependencies}


def placement_scope(job: dict, *, root: Path = REPO) -> dict:
    """Prove a pinned release's import/config identity during placement, including repository files in arguments."""
    commit = job.get("planned_commit")
    if not commit:
        return {"source_commit": None, "files": {}}
    from ops.jobs import migration
    scope = migration.execution_scope([job["script"]], commit, root)
    if not scope["identical"]:
        raise LocalInputError("local code differs from the release the row was placed with")
    files = {**scope["files"], **scope.get("third_party", {})}
    read, listed = migration.commit_reader(root, commit)
    for arg in job.get("args", []):
        value = _value(arg)
        path = Path(value)
        if path.is_absolute():
            try:
                value = str(path.relative_to(root))
            except ValueError:
                continue
        if ".." in Path(value).parts:
            raise LocalInputError(f"unsafe repository input: {value}")
        if value in listed or (root / value).is_file() or value.startswith(("configs/", "scripts/", "s4d/", "ops/")):
            # Runtime datasets and checkpoints are verified as data, never as repository code.
            if Path(value).parts[0] == "runs":
                continue
            before = read(value)
            if before is None or not (root / value).is_file() or _sha(before) != _hash(root / value):
                raise LocalInputError("local code differs from the release the row was placed with")
            files[value] = _sha(before)
    return {"source_commit": commit, "files": files}


def localize(job: dict, state: dict, *, roles: dict, by_id=None, runs: Path | None = None,
             mirror: Path | None = None, manifest: Path | None = None, source_identity: dict | None = None,
             ownership: dict | None = None, root: Path = REPO, remote_prefix: str = REMOTE_PREFIX) -> tuple[list, dict]:
    """Validate the complete input closure and placed scope, returning effective args without modifying the row."""
    try:
        if not applies(job, roles, by_id):
            raise LocalInputError("script is not admitted for local real-world FQL")
        manifest = Path(manifest or (Path(mirror) / "MANIFEST.json" if mirror is not None else MANIFEST))
        mirror = Path(mirror or MIRROR)
        effective, record = _inputs(job, state, mirror, manifest,
                                    Path(runs or root / "runs"), remote_prefix)
        owned = ownership if ownership is not None else host_roles.read_ownership()
        scope = (owned.get(job["id"]) or {}).get("execution_scope")
        if job.get("planned_commit"):
            found = (source_identity or {}).get("source_sha256", {})
            if not scope or scope.get("source_commit") != job["planned_commit"] or not scope.get("files"):
                raise LocalInputError("local code differs from the release the row was placed with")
            for name, digest in scope["files"].items():
                current = found.get(name)
                if current is None and "@" not in name and (root / name).is_file():
                    current = _hash(root / name)
                if current != digest:
                    raise LocalInputError("local code differs from the release the row was placed with")
        record["execution_scope"] = scope or {"source_commit": None, "files": {}}
        return effective, record
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise LocalInputError(str(exc)) from exc


def _pause_names(paused_fn, job_id: str) -> list[str]:
    result = paused_fn(job_id)
    return [""] if result is True else list(result or [])


def _packing(queue: dict, state: dict, paused_fn=None) -> tuple[dict, float]:
    """Local capacity left for placement: running / launching local jobs, then never-launched local rows that the
    local drivers could admit now (no active pause, every dependency done). Paused or dependency-blocked local rows
    (e.g. intentionally paused Meta-World campaigns) reserve nothing: they cannot start before a later pass."""
    limits = queue.get("hosts", {}).get("local", {}).get("limits", {})
    slots = {gpu: {"jobs": 0, "mem_gb": 0.0, **limit} for gpu, limit in limits.items()}
    queued_ram = 0.0
    by_id = {j["id"]: j for j in queue["jobs"]}
    for job_id, current in state.items():
        if current.get("host", "local") == "local" and current.get("status") in ("running", "launching"):
            job = by_id.get(job_id, {})
            gpu = current.get("gpu")
            if gpu in slots:
                slots[gpu]["jobs"] += 1
                slots[gpu]["mem_gb"] += max(float(current.get("mem_gb", 0)), float(job.get("mem_gb", 0)))
    for job in queue["jobs"]:
        if (job.get("host", "local") == "local" and job["id"] not in state
                and not (paused_fn is not None and _pause_names(paused_fn, job["id"]))
                and all(state.get(dep, {}).get("status") == "done" for dep in job.get("deps", []))):
            queued_ram += float(job.get("ram_gb", 0)) * 2**30
            if _pack(job, slots) is None:
                # An existing waiting reservation that cannot pack consumes the remaining placement capacity.
                for slot in slots.values():
                    slot["jobs"] = int(slot["max_jobs"])
    return slots, queued_ram


def _pack(job: dict, slots: dict) -> str | None:
    for gpu, slot in slots.items():
        if (job.get("gpu", "any") in ("any", gpu) and slot["jobs"] < int(slot["max_jobs"])
                and slot["mem_gb"] + float(job.get("mem_gb", 0)) <= float(slot["max_mem_gb"])):
            slot["jobs"] += 1
            slot["mem_gb"] += float(job.get("mem_gb", 0))
            return gpu
    return None


def _missing_bytes(job: dict, mirror: Path, manifest: Path, inventory: dict | None,
                   remote_prefix: str = REMOTE_PREFIX) -> int:
    """Transfer bytes still needed, counting each original once; localization itself needs no download."""
    files = _manifest(manifest, remote_prefix)["files"]
    pending, seen, total = list(needed_inputs(job, remote_prefix=remote_prefix)["mirror"]), set(), 0
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        item = (inventory or {}).get(relative) or files.get(relative, {})
        pending += item.get("requires", [])
        original = "_remote/" + relative if item.get("kind") == "localized" else relative
        known, path = files.get(original, {}), mirror / original
        try:
            _regular(path)
            stat = path.stat()
            expected = item.get("remote_sha256", item.get("sha256"))
            if (not known or stat.st_size != known.get("bytes") or stat.st_mtime_ns != known.get("mtime_ns")
                    or known.get("sha256") != expected
                    or (stat.st_size <= SMALL_HASH_BYTES and _hash(path) != expected)):
                raise LocalInputError("input copy changed")
        except (OSError, ValueError):
            total += int(item.get("remote_bytes", item.get("bytes", 0)))
    return total


def plan(queue: dict, state: dict, roles: dict, *, paused_fn, allow_pauses=(), policy="fit", capacity=None,
         mirror: Path = MIRROR, manifest: Path | None = None, ownership=None, runs: Path | None = None,
         root: Path = REPO, remote_plan: dict | None = None, remote_prefix: str = REMOTE_PREFIX) -> list[dict]:
    """Read-only decisions in priority order. ``fit`` reserves per-MIG slots and remaining local RAM; ``all`` queues.

    The caller supplies RAM remaining after measured budget usage and active-job growth. Reservations for rows already
    placed but not started, and for rows selected earlier in this plan, are charged here. A named pause may be allowed
    for placement only; admission remains paused until the coordinator lifts it.
    """
    if policy not in ("fit", "all"):
        raise ValueError("unknown local placement policy")
    mirror, manifest, runs = Path(mirror), Path(manifest or Path(mirror) / "MANIFEST.json"), Path(runs or root / "runs")
    by_id, decisions = {j["id"]: j for j in queue["jobs"]}, []
    slots, queued_ram = _packing(queue, state, paused_fn)
    ram = float((capacity or {}).get("ram_free_bytes", 0)) - queued_ram
    for _, job in sorted(enumerate(queue["jobs"]), key=lambda pair: (pair[1].get("priority", 5), pair[0])):
        if job.get("host", "local") != "remote" or not applies(job, roles, by_id):
            continue
        job_id, reason, decision, scope = job["id"], "ready", "local", None
        needs = 0
        if job_id in state:
            decision, reason = "stay_remote", "launched or started rows never move"
        elif (runs / job_id).exists() or (runs / job_id).is_symlink():
            decision, reason = "stay_remote", "local output directory exists"
        elif job_id in (ownership or {}):
            decision, reason = "stay_remote", "ownership record exists"
        elif set(_pause_names(paused_fn, job_id)) - set(allow_pauses):
            decision, reason = "stay_remote", "paused"
        elif any(state.get(dep, {}).get("status") != "done" for dep in job.get("deps", [])):
            decision, reason = "waits", "dependencies not done"
        else:
            try:
                _inputs(job, state, mirror, manifest, runs, remote_prefix)
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                decision, reason = "waits", str(exc)
                try:
                    needs = _missing_bytes(job, mirror, manifest, remote_plan, remote_prefix)
                except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
                    needs = 0  # the refusal above remains authoritative when a size cannot be inventoried
            if decision == "local":
                try:
                    scope = placement_scope(job, root=root)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    decision, reason = "stay_remote", str(exc)
            if decision == "local" and policy == "fit":
                if ram < float(job.get("ram_gb", 0)) * 2**30:
                    decision, reason = "waits", "local capacity"
                elif _pack(host_roles.destination_row(job, "local", ""), slots) is None:
                    decision, reason = "waits", "local capacity"
                else:
                    ram -= float(job.get("ram_gb", 0)) * 2**30
        item = {"id": job_id, "decision": decision, "reason": reason, "mem_gb": float(job.get("mem_gb", 0)),
                "ram_gb": float(job.get("ram_gb", 0)), "needs_mirror_bytes": needs}
        if scope is not None:
            item["execution_scope"] = scope
        decisions.append(item)
    return decisions


def apply(ids, reason, *, allow_pauses=(), queue_path=None, registry_path=None,
          manifest: Path = host_roles.POOL_MANIFEST, ownership_dir: Path = host_roles.OWNERSHIP_DIR,
          admission_lock: Path = host_roles.ADMISSION_LOCK, remote_lock: Path = REMOTE_LOCK, blocking=False,
          mirror: Path = MIRROR, input_manifest: Path | None = None, roles_path: Path = host_roles.ROLES_PATH,
          root: Path = REPO, paused_fn=None) -> dict:
    """Recheck all non-capacity rules under remote -> local-admission -> registry locks, then move host fields only.

    Capacity is a planning policy, not a permanent ownership condition: rows placed by ``all`` can wait in the local
    queue. Every move retains the original row, fingerprints and release scope. Allowed pauses are recorded and are
    never lifted here. A busy driver lock returns immediately unless the caller explicitly requested blocking.
    """
    from ops.jobs import scheduler
    from ops.jobs.admission_pause import active_pauses
    queue_path, registry_path = Path(queue_path or scheduler.QUEUE), Path(registry_path or scheduler.REGISTRY)
    with contextlib.ExitStack() as stack:
        for path in (Path(remote_lock), Path(admission_lock), registry_path.with_suffix(".lock")):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                return {"moved": [], "busy": True}
        roles = host_roles.load_roles(roles_path)
        queue, state, ownership = scheduler.load_queue(queue_path), scheduler.read_registry(registry_path), \
            host_roles.read_ownership(ownership_dir)
        requested = set(ids)
        pause_check = paused_fn or (lambda job_id: active_pauses(job_id, fresh=True))
        decisions = plan(queue, state, roles, paused_fn=pause_check, allow_pauses=allow_pauses,
                         policy="all", mirror=mirror, manifest=input_manifest, ownership=ownership, root=root)
        selected = [d for d in decisions if d["id"] in requested and d["decision"] == "local"]
        kept = {d["id"]: d["reason"] for d in decisions if d["id"] in requested and d["decision"] != "local"}
        kept.update({i: "not a remote local-admitted FQL training row" for i in requested - {d["id"] for d in decisions}})
        if not selected:
            return {"moved": [], "kept": kept}
        by_id = {j["id"]: j for j in queue["jobs"]}
        stamp = host_roles.now()
        placement = f"realworld-local {stamp} (local; configs/hosts/roles.yaml; runs/setup/pool-placement.json)"
        effective = {d["id"]: host_roles.destination_row(by_id[d["id"]], "local", placement) for d in selected}
        for job_id, new in effective.items():
            old = by_id[job_id]
            assert scheduler.job_fingerprint(new) == scheduler.job_fingerprint(old)
            assert {k: v for k, v in new.items() if k not in host_roles.HOST_FIELDS} == {
                k: v for k, v in old.items() if k not in host_roles.HOST_FIELDS}
        # Record ownership before the queue rewrite: an interrupted move fails closed on both drivers. A later
        # coordinator can inspect the original rows/scopes in the pool record; no row can launch on both hosts.
        scopes = {d["id"]: d["execution_scope"] for d in selected}
        before = queue_path.read_bytes()
        Path(manifest).parent.mkdir(parents=True, exist_ok=True)
        for job_id in effective:
            previous = ownership.get(job_id)
            host_roles.write_ownership({
                "job_id": job_id, "kind": "role_placement", "state": "placed", "owner_host": "local",
                "from_host": "remote", "to_host": "local", "attempts": 0, "time": stamp, "reason": reason,
                "roles_sha256": roles["sha256"], "execution_scope": scopes[job_id], "allow_pauses": list(allow_pauses),
                "history": (previous.get("history", []) + [{k: v for k, v in previous.items() if k != "history"}]
                            if previous else [])}, ownership_dir)
        host_roles.append_pool_record(Path(manifest), {
            "time": stamp, "to": "local", "reason": f"{reason} (configs/hosts/roles.yaml sha256 {roles['sha256']})",
            "ids": list(effective), "original": {i: by_id[i] for i in effective}, "execution_scopes": scopes,
            "allow_pauses": list(allow_pauses), "queue_sha256_before": _sha(before),
            "queue_sha256_after": None}, effective)
        _, after = host_roles.write_queue(queue_path, [effective.get(j["id"], j) for j in queue["jobs"]],
                                          scheduler.load_queue)
        pool = json.loads(Path(manifest).read_text())
        pool["moves"][-1]["queue_sha256_after"] = _sha(after)
        _json(Path(manifest), pool)
        return {"moved": list(effective), "kept": kept}


def local_rows(rows: list[dict], roles: dict) -> list[dict]:
    by_id = {j["id"]: j for j in rows}
    return [j for j in rows if j.get("host", "local") == "local" and applies(j, roles, by_id)]


def closeout(job_id: str, state: dict, job: dict, finished_dir: Path = FINISHED_DIR, *, runs: Path | None = None) -> dict:
    """Publish immutable evidence for a completed local row; a different existing closeout is never overwritten."""
    current = state.get(job_id, {})
    if current.get("status") != "done" or current.get("host") != "local":
        raise LocalInputError(f"{job_id}: not completed locally")
    files, runs = {}, Path(runs or REPO / "runs")
    for name in job.get("required_results", []):
        path = runs / job_id / _rel(name)
        _regular(path)
        if path.stat().st_size == 0:
            raise LocalInputError(f"empty required result: {name}")
        files[name] = _hash(path)
    if not files:
        raise LocalInputError("no required results")
    record = {"job_id": job_id, "files": files, "inputs": current.get("inputs"), "commit": current.get("commit"),
              "gpu": current.get("gpu"), "attempt": current.get("attempt")}
    path = Path(finished_dir) / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # link gives atomic create-if-absent semantics, including concurrent closeout invocations.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
    _json(tmp, record)
    try:
        os.link(tmp, path)
    except FileExistsError:
        if json.loads(path.read_bytes()) != record:
            raise LocalInputError(f"different closeout already exists: {job_id}") from None
    finally:
        tmp.unlink(missing_ok=True)
    return record


def _capacity(queue: dict, state: dict) -> dict:
    from ops.resources import memory_budget as mb
    pss = mb.session_nonfile_pss()
    by_id = {j["id"]: j for j in queue["jobs"]}
    growth = sum(max(0, max(float(s.get("ram_gb", 0)), float(by_id.get(i, {}).get("ram_gb", 0))) * 2**30
                     - pss.get(int(s.get("pid") or -1), 0)) for i, s in state.items()
                 if s.get("host", "local") == "local" and s.get("status") in ("running", "launching", "sync_pending"))
    return {"ram_free_bytes": max(0, mb.CEILING - mb.TRANSIENT_OVERHEAD - mb.snapshot()["budget_usage"] - growth)}


def main() -> None:
    from ops.jobs import scheduler
    from ops.jobs.admission_pause import active_pauses
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "place"):
        command = sub.add_parser(name)
        command.add_argument("--policy", choices=("fit", "all"), default="fit")
        command.add_argument("--allow-pause", action="append", default=[])
        if name == "plan":
            command.add_argument("--remote", action="store_true")
        else:
            command.add_argument("ids", nargs="+")
            command.add_argument("--dry-run", action="store_true")
    command = sub.add_parser("fetch")
    command.add_argument("ids", nargs="+")
    command.add_argument("--priority", choices=("bulk", "dependency"), default="bulk")
    command.add_argument("--dry-run", action="store_true")
    command = sub.add_parser("verify")
    command.add_argument("--quick", action="store_true")
    command = sub.add_parser("closeout")
    command.add_argument("ids", nargs="*")
    command.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.command == "closeout" and (args.all == bool(args.ids)):
        parser.error("closeout requires either --all or job ids")
    if args.command == "verify":
        problems = verify(full=not args.quick)
        print(json.dumps(problems, indent=1))
        raise SystemExit(bool(problems))
    queue, state = scheduler.load_queue(), scheduler.read_registry()
    roles, owned = host_roles.load_roles(), host_roles.read_ownership()
    by_id = {j["id"]: j for j in queue["jobs"]}
    if args.command == "fetch":
        result = fetch(sorted({r for i in args.ids for r in needed_inputs(by_id[i])["mirror"]}),
                       priority=args.priority, dry_run=args.dry_run)
    elif args.command == "closeout":
        ids = [j["id"] for j in local_rows(queue["jobs"], roles) if state.get(j["id"], {}).get("status") == "done"] \
            if args.all else args.ids
        result = [closeout(i, state, by_id[i]) for i in ids]
    else:
        inventory = None
        if args.command == "plan" and args.remote:
            files = _manifest(MANIFEST)["files"]
            missing = set()
            for job in queue["jobs"]:
                if job.get("host") != "remote" or not applies(job, roles):
                    continue
                for relative in needed_inputs(job)["mirror"]:
                    try:
                        _check(relative, files, MIRROR, set())
                    except (OSError, ValueError, KeyError, TypeError):
                        missing.add(relative)
            if missing:
                inventory = remote_plan(sorted(missing), _fetch_client())
        decisions = plan(queue, state, roles, paused_fn=active_pauses, allow_pauses=args.allow_pause,
                         policy=args.policy, capacity=_capacity(queue, state), ownership=owned, remote_plan=inventory)
        if args.command == "plan":
            result = decisions
        else:
            chosen = [d["id"] for d in decisions if d["id"] in args.ids and d["decision"] == "local"]
            result = {"would_move": chosen} if args.dry_run else apply(
                chosen, "directive P: explicit local FQL placement", allow_pauses=args.allow_pause)
            result["decisions"] = [d for d in decisions if d["id"] in args.ids]
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
