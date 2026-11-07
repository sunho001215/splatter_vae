"""Resumable real-world episode ingestion + transfer worker (user directive 2026-11-06, item 3).

Discovers ``episode_NNNNNN.h5`` recordings in /home/ws/data/real_world/{drawer-open,mug-hanging}/ (READ-ONLY: never
modified, moved or renamed), validates them (episode_validator.py), detects duplicates/conflicts, and uploads valid
episodes one at a time to the remote campaign root under data/real_world/<task>/:

  rsync --partial --append-verify --bwlimit=<upload-budget.json, or a live upload-boost lease> -> .incoming/<name>.part
  remote sha256(part) == local sha256  ->  chmod 0444 + ln (atomic, never clobbers) + rm part  ->  sha256(final)

State (manifest.json, events.jsonl, status.json, worker.lock/pid) lives in /home/ws/ws/realworld-ops/ingest/; see the
README.md there for the schema. Every pass is idempotent; the worker can be killed at any time.

  python -I ingest_worker.py --once [--dry-run] [--no-transfer] [--max-transfers N] [--state-dir DIR]
  python -I ingest_worker.py --loop --interval 120          (detached: see README.md)
  python -I ingest_worker.py --status
  python -I ingest_worker.py set-stage KEY STAGE JSON        (preprocessing agents: record a stage result)
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import episode_validator as V  # noqa: E402

REPO = Path("/home/ws/ws/splatter4d")
STATE_DIR = Path("/home/ws/ws/realworld-ops/ingest")
RAW_ROOT = Path("/home/ws/data/real_world")
REMOTE_DATA_REL = "data/real_world"
TASKS = {"drawer-open": {"drawer-opening"}, "mug-hanging": {"mug-hanging"}}  # folder -> accepted task_name attrs
TARGET_PER_TASK = 100
MANIFEST_SCHEMA = "rw-ingest-manifest"
MANIFEST_SCHEMA_VERSION = 2   # 1 -> 2 (2026-11-06 03:00): explicit schema_version, versioned preprocessing stages
PREPROCESS_SCHEMA = "rw-preprocess"
PREPROCESS_SCHEMA_VERSION = 1
STAGE_STATUSES = ("pending", "running", "done", "failed", "skipped")
STAGE_FIELDS_REQUIRED = ("status", "code_version")
STAGE_FIELDS = {"status", "code_version", "stage_schema_version", "code_commit", "config_sha256", "input_sha256",
                "output_format", "outputs", "started_at", "finished_at", "host", "failure", "attempt", "notes",
                "metrics"}
WORKER_VERSION = "rw-ingest-worker/3"  # /3: reads the upload-boost lease; budget increases apply at the next transfer
HEARTBEAT_S = 3600.0          # quiet loop: a pass summary is logged on change, else at most hourly
SCAN_DURING_TRANSFER_S = 300.0  # new/completed raw files are checked every 5 min even during a long upload
DEFAULT_KIB = 600          # authorized initial upload budget (KiB/s)
MAX_KIB = 600              # never exceeded unless upload-budget.json carries "coordinator_override": true
# Temporary dataset-upload boost (user decision 2026-11-06 ~11:20, item 6), granted by upload_boost.py: ONE aggregate
# budget for this worker's dataset uploads. It applies only while upload-boost/policy.json is a live lease that is valid
# in every respect (boost_level); anything else (missing, stale, malformed, base budget changed) = the base budget.
BOOST_POLICY = "upload-boost/policy.json"
BOOST_SCHEMA = "rw-upload-boost/1"
BOOST_LEVELS_KIB = (2048, 4096, 8192)  # 2, 4, 8 MiB/s; 8 MiB/s is the ceiling
BOOST_MAX_LEASE_S = 600.0              # a lease reaching further into the future than this is invalid
GRACE_S = 120.0            # files modified (mtime/ctime) more recently are not looked at
INVALID_AFTER_S = 3 * 3600  # a finished-looking file that keeps failing, unchanged, this long -> "invalid"
RECHECK_FAILED_S = 600.0
RECHECK_INVALID_S = 6 * 3600.0
TRANSFER_WINDOW_S = 1800.0  # per pass; then discovery/validation runs again
REMOTE_MIN_FREE_FRACTION = 0.15
MAX_SHA_MISMATCHES = 3
BLOCKING_FLAGS = {"duplicate_content", "index_conflict", "source_missing", "source_changed_after_transfer",
                  "remote_conflict", "transfer_failed_repeatedly"}
DERIVED_FLAGS = {"duplicate_content", "index_conflict", "source_missing"}  # recomputed every pass
UPLOAD_TEMP_RE = re.compile(r"^\.episode_\d{6}\.h5\.[A-Za-z0-9]{6}$")  # rsync's in-flight name
PART_RE = re.compile(r"/data/real_world/[a-z-]+/\.incoming/episode_\d{6}\.h5\.part$")

_STOP = False


def _on_signal(signum, _frame):
    global _STOP
    _STOP = True
    log(f"signal {signum}: stopping after the current step")


def now_iso(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).astimezone().isoformat(
        timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def base_budget_kib(state_dir: Path) -> int:
    """The coordinator's upload budget (upload-budget.json); missing = DEFAULT_KIB, unreadable/invalid = 0 (pause)."""
    path = state_dir / "upload-budget.json"
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        return DEFAULT_KIB
    except (OSError, ValueError) as exc:
        log(f"upload-budget.json unreadable ({exc}): transfers paused")
        return 0
    value = doc.get("kib_per_s") if isinstance(doc, dict) else None
    if type(value) is not int or value < 0:
        log(f"upload-budget.json kib_per_s {value!r} invalid: transfers paused")
        return 0
    if value > MAX_KIB and doc.get("coordinator_override") is not True:
        log(f"upload-budget.json asks {value} KiB/s > {MAX_KIB} without coordinator_override: using {MAX_KIB}")
        return MAX_KIB
    return value


def boost_level(state_dir: Path, base_kib: int, now: float | None = None) -> tuple[int | None, str]:
    """(aggregate KiB/s, why) of a live upload-boost lease, or (None, why not) -> the base budget applies (fail safe)."""
    try:
        return _boost_level(state_dir, base_kib, now)
    except Exception as exc:  # noqa: BLE001 - whatever is wrong with the record, the original budget applies
        return None, f"boost check failed ({type(exc).__name__})"


def _boost_level(state_dir: Path, base_kib: int, now: float | None) -> tuple[int | None, str]:
    try:
        doc = json.loads((state_dir / BOOST_POLICY).read_text())
    except FileNotFoundError:
        return None, "no boost policy"
    except (OSError, ValueError) as exc:
        return None, f"boost policy unreadable ({type(exc).__name__})"
    now = time.time() if now is None else now
    if not isinstance(doc, dict) or doc.get("schema") != BOOST_SCHEMA:
        return None, "unknown boost policy schema"
    if doc.get("state") != "active":
        return None, f"boost {doc.get('state')!r}"
    level = doc.get("aggregate_kib_per_s")
    if type(level) is not int or level not in BOOST_LEVELS_KIB:
        return None, f"boost level {level!r} not allowed"
    if base_kib <= 0 or doc.get("original_kib_per_s") != base_kib:
        return None, f"boost granted for base {doc.get('original_kib_per_s')!r}, base is {base_kib}"
    lease = doc.get("lease")
    expires = lease.get("expires_at_unix") if isinstance(lease, dict) else None
    if type(expires) not in (int, float) or not math.isfinite(expires):
        return None, "boost lease invalid"
    if expires <= now:
        return None, "boost lease expired"
    if expires - now > BOOST_MAX_LEASE_S:
        return None, "boost lease too long"
    return level, "boost lease valid"


def atomic_write_json(path: Path, doc) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as out:
        json.dump(doc, out, indent=1, sort_keys=True, default=str)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


# ----------------------------------------------------------------------------------------------- state / manifest


class State:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = state_dir / "manifest.json"
        self.events_path = state_dir / "events.jsonl"
        self.lock_path = state_dir / "manifest.lock"
        self.doc = self._load()

    def _load(self) -> dict:
        if self.manifest_path.exists():
            return migrate(json.loads(self.manifest_path.read_text()))
        return {"schema": MANIFEST_SCHEMA, "schema_version": MANIFEST_SCHEMA_VERSION, "created_at": now_iso(),
                "episodes": {}, "unrecognized_files": {}, "migrations": []}

    @property
    def episodes(self) -> dict:
        return self.doc["episodes"]

    def event(self, kind: str, key: str | None = None, quiet: bool = False, **fields) -> None:
        """Append to events.jsonl; also log the line unless quiet (routine transitions stay out of the log)."""
        rec = {"t": now_iso(), "event": kind, **({"key": key} if key else {}), **fields}
        line = json.dumps(rec, sort_keys=True, default=str) + "\n"
        fd = os.open(self.events_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
        if not quiet:
            log(f"{kind} {key or ''} {json.dumps(fields, default=str)[:400]}")

    def save(self) -> None:
        """Atomic write under manifest.lock. Fields owned by other agents ("preprocessing") are taken from disk."""
        with open(self.lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.manifest_path.exists():
                disk = migrate(json.loads(self.manifest_path.read_text())).get("episodes", {})
                for key, entry in self.episodes.items():
                    if key in disk and "preprocessing" in disk[key]:
                        entry["preprocessing"] = disk[key]["preprocessing"]
            self.doc["updated_at"] = now_iso()
            atomic_write_json(self.manifest_path, self.doc)


def new_entry(task: str, name: str, path: Path, now: float) -> dict:
    m = V.NAME_RE.match(name)
    return {
        "key": f"{task}/{name}", "task": task, "filename": name, "file_index": int(m.group(1)),
        "episode_index": None, "episode_id": None, "sha256": None, "size": None,
        "source": {"path": str(path), "present": True, "first_seen": now_iso(now), "first_seen_unix": now},
        "attrs": {}, "flags": {},
        "validation": {"status": "pending", "reasons": [], "warnings": [], "warning_codes": [], "checks": 0,
                       "failed_checks": 0},
        "transfer": {"status": "pending", "attempts": 0, "failures": 0, "sha_mismatches": 0},
        "preprocessing": new_preprocessing(),
    }


def new_preprocessing() -> dict:
    return {"schema": PREPROCESS_SCHEMA, "schema_version": PREPROCESS_SCHEMA_VERSION, "status": "pending",
            "stages": {}}


def migrate(doc: dict) -> dict:
    """Bring an older manifest to MANIFEST_SCHEMA_VERSION in memory (written back by the next save)."""
    if doc.get("schema") == "rw-ingest-manifest/1" and "schema_version" not in doc:
        doc["schema"], doc["schema_version"] = MANIFEST_SCHEMA, 2
        doc.setdefault("migrations", []).append({"from": "rw-ingest-manifest/1", "to": 2, "at": now_iso()})
    for e in doc.get("episodes", {}).values():  # verification snapshots for copies verified before they existed
        t = e.get("transfer") or {}
        if t.get("status") == "verified" and "verified_size" not in t:
            same = t.get("remote_sha256") and t.get("remote_sha256") == e.get("sha256")
            t["verified_size"] = e.get("size") if same else None
            t["verified_episode_index"] = e.get("episode_index") if same else None
    for e in doc.get("episodes", {}).values():  # per-entry: v1 preprocessing blocks (also inside early v2 files)
        old = e.get("preprocessing") or {}
        if old.get("schema") in (None, "rw-preprocess-stage/1"):
            pre = new_preprocessing()
            for name, rec in (old.get("stages") or {}).items():  # none were recorded under v1
                pre["stages"][name] = {"stage": name, "stage_schema_version": 0, "code_version": rec.get("version"),
                                       **{k: v for k, v in rec.items() if k != "version"}}
            pre["status"] = old.get("status", "pending")
            e["preprocessing"] = pre
    if doc.get("schema") != MANIFEST_SCHEMA or doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema {doc.get('schema')!r} v{doc.get('schema_version')!r}")
    return doc


# ------------------------------------------------------------------------------------------------ remote access


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _child_setup():
    """preexec_fn factory: rsync must not outlive (and race) a killed worker."""
    parent = os.getpid()

    def setup():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_DFL)  # not the worker's "stop later" handler
        if _LIBC.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
            os._exit(97)
        if os.getppid() != parent:  # the worker died before PDEATHSIG was armed
            os._exit(98)
        os.setpriority(os.PRIO_PROCESS, 0, 15)
    return setup


class RemoteError(RuntimeError):
    pass


class Unreachable(RuntimeError):
    pass


class RemoteOps:
    """Remote side through ops.remote.access.RemoteClient (credentials stay in its environment, never in args)."""

    def __init__(self):
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from ops.remote import access

        self._access = access
        self.client = access.RemoteClient()
        self.target = self.client.target

    # paths (RemoteClient.remote_path refuses anything outside the campaign root)
    def final_path(self, task: str, name: str) -> str:
        return self.client.remote_path(f"{REMOTE_DATA_REL}/{task}/{name}")

    def part_path(self, task: str, name: str) -> str:
        return self.client.remote_path(f"{REMOTE_DATA_REL}/{task}/.incoming/{name}.part")

    def _ssh(self, command: str, idempotent: bool = True, timeout: float = 300) -> str:
        try:
            r = self.client.ssh(command, wait=False, idempotent=idempotent, timeout=timeout)
        except self._access.RemoteUnreachable as exc:
            raise Unreachable(str(exc)) from None
        if r.returncode:
            raise RemoteError(f"remote command exited {r.returncode}: {r.stderr.strip()[-500:]}")
        return r.stdout

    def reachable(self) -> None:
        try:
            self.client.check_tunnel(wait=False)
        except self._access.RemoteUnreachable as exc:
            raise Unreachable(str(exc)) from None

    def ensure_dirs(self, task: str, create: bool = True) -> None:
        """RemoteClient.ensure_remote_path: every component of <root>/data/real_world/<task>/.incoming must be a
        non-symlink inside the campaign root (realpath check); create=False only checks (dry run)."""
        try:
            self.client.ensure_remote_path(f"{REMOTE_DATA_REL}/{task}/.incoming", create_parent=create,
                                           directory=True, wait=False)
        except self._access.RemoteUnreachable as exc:
            raise Unreachable(str(exc)) from None
        except self._access.RemoteCommandError as exc:
            raise RemoteError(str(exc)) from None

    def probe(self, task: str, name: str) -> dict:
        f, p = self.final_path(task, name), self.part_path(task, name)
        d = self.client.remote_path(f"{REMOTE_DATA_REL}/{task}")
        q = shlex.quote
        rw = self.client.remote_path(REMOTE_DATA_REL)
        script = (f"set -u; f={q(f)}; p={q(p)}; d={q(d)}; rw={q(rw)}; "
                  'fs=$(stat -c %s -- "$f" 2>/dev/null || echo -); ps=$(stat -c %s -- "$p" 2>/dev/null || echo -); '
                  'fl=-; [ -L "$f" ] && fl=L; [ -L "$p" ] && fl=L; '
                  'while [ ! -d "$d" ]; do d=$(dirname -- "$d"); done; '
                  'echo "$fs $ps $fl $(df -B1 --output=size,avail -- "$d" | tail -1)"; '
                  'if [ -d "$rw" ]; then du -sb -- "$rw" | cut -f1; else echo 0; fi')
        lines = self._ssh(script).split("\n")
        tok = lines[0].split()
        if len(tok) != 5 or not lines[1].strip().isdigit():
            raise RemoteError(f"unexpected probe output {lines[:2]!r}")
        num = lambda x: None if x == "-" else int(x)  # noqa: E731
        return {"final_size": num(tok[0]), "part_size": num(tok[1]), "symlink": tok[2] == "L",
                "fs_size": int(tok[3]), "fs_avail": int(tok[4]), "rw_used": int(lines[1])}

    def sha256(self, path: str) -> str | None:
        q = shlex.quote(path)
        out = self._ssh(f"if [ -f {q} ] && [ ! -L {q} ]; then nice -n 10 sha256sum -- {q}; fi", timeout=900)
        if not out.strip():
            return None
        if not re.match(r"^[0-9a-f]{64}  ", out):
            raise RemoteError("invalid remote sha256 output")
        return out[:64]

    def remove_part(self, path: str) -> None:
        if not PART_RE.search(path):
            raise ValueError(f"refusing to remove a non-part path {path}")
        self._ssh(f"rm -f -- {shlex.quote(path)}")

    def publish(self, part: str, final: str) -> str:
        """Atomic no-clobber publish: hard link (fails if the final name exists) then drop the part name."""
        if not PART_RE.search(part):
            raise ValueError(f"not a part path {part}")
        q = shlex.quote
        script = (f"set -eu; p={q(part)}; f={q(final)}; "
                  'if [ -L "$f" ] || [ -e "$f" ]; then echo EXISTS; exit 0; fi; '
                  'if [ ! -f "$p" ] || [ -L "$p" ]; then echo NOPART; exit 0; fi; '
                  'chmod 0444 -- "$p"; ln -T -- "$p" "$f"; rm -f -- "$p"; echo PUBLISHED')
        out = self._ssh(script, idempotent=False).strip().split("\n")[-1]
        if out not in ("EXISTS", "NOPART", "PUBLISHED"):
            raise RemoteError(f"unexpected publish output {out!r}")
        return out

    def rsync_command(self, local: Path, remote: str, kib: int, append: bool = True) -> tuple[list[str], dict]:
        transport = []
        args = self.client.ssh_args()
        i = 0
        while i < len(args):  # dedicated TCP connection: no ControlMaster sharing with the campaign's command mux
            if args[i] == "-o" and args[i + 1].split("=")[0] in ("ControlMaster", "ControlPersist", "ControlPath"):
                i += 2
                continue
            transport.append(args[i])
            i += 1
        transport[transport.index("ssh") + 1:transport.index("ssh") + 1] = [
            "-o", "ControlMaster=no", "-o", "ControlPath=none"]
        mode = ["--partial", "--append-verify"] if append else ["--checksum"]
        cmd = ["rsync", "-t", *mode, "--protect-args", "--chmod=F0644",
               f"--bwlimit={int(kib)}", "--timeout=300", "-e", shlex.join(transport), "--", str(local),
               f"{self.target}:{remote}"]
        env = self.client._environment()
        if any(self.client.credentials.password in a for a in cmd):
            raise ValueError("credential in transfer arguments")
        return cmd, env

    def redact(self, text: str) -> str:
        return self.client._redact(text)

    def upload_small(self, local: Path, rel_final: str, kib: int) -> None:
        """Small informational file (remote manifest mirror): temp name + mv -T (this file may be replaced)."""
        final = self.client.remote_path(rel_final)
        tmp = final + ".tmp"
        cmd, env = self.rsync_command(local, tmp, kib, append=False)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600,
                           preexec_fn=_child_setup())
        if r.returncode:
            raise RemoteError(f"mirror upload exited {r.returncode}: {self.redact(r.stderr)[-300:]}")
        self._ssh(f"mv -T -- {shlex.quote(tmp)} {shlex.quote(final)}", idempotent=False)


# --------------------------------------------------------------------------------------------------- the worker


class Worker:
    def __init__(self, state_dir: Path, raw_root: Path = RAW_ROOT, tasks: dict = None, remote=None,
                 grace_s: float = GRACE_S, invalid_after_s: float = INVALID_AFTER_S, dry_run: bool = False,
                 transfer: bool = True, max_transfers: int | None = None, window_s: float = TRANSFER_WINDOW_S,
                 run_adapter: bool = True, mirror: bool = True):
        self.state = State(state_dir)
        self.raw_root = raw_root
        self.tasks = tasks or TASKS
        self.remote = remote
        self.grace_s = grace_s
        self.invalid_after_s = invalid_after_s
        self.dry_run = dry_run
        self.transfer_enabled = transfer
        self.max_transfers = max_transfers
        self.window_s = window_s
        self.run_adapter = run_adapter
        self.mirror_enabled = mirror
        self.transfers_done = 0
        self.remote_unreachable_since = None
        self._last_kib = None
        self._boost_why = None
        self._inflight = None  # {"proc", "kib", "reduced_to"} while _run_rsync supervises an upload
        self._mirror_dirty = True
        self.last_scan = 0.0
        self.remote_disk = None
        self._last_summary = None
        self._last_summary_at = 0.0

    # ------------------------------------------------------------------ configuration files
    def upload_kib(self) -> int:
        return self.upload_budget()[0]

    def upload_budget(self) -> tuple[int, str]:
        """(KiB/s, "boost"|"base"): the boost lease's aggregate budget while it is valid for the current base budget,
        else the base budget (upload-budget.json). A change of the boost's applicability is logged once."""
        base = self.base_kib()
        level, why = boost_level(self.state.dir, base)
        if why != self._boost_why:
            self.state.event("upload_boost", applied=level is not None, kib_per_s=level or base, base_kib_per_s=base,
                             reason=why, quiet=self._boost_why is None and level is None)
            self._boost_why = why
        return (level, "boost") if level else (base, "base")

    def base_kib(self) -> int:
        return base_budget_kib(self.state.dir)

    def overrides(self) -> dict:
        path = self.state.dir / "overrides.json"
        try:
            doc = json.loads(path.read_text())
            return doc if isinstance(doc, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log(f"overrides.json unreadable ({exc}); ignored")
            return {}

    def stop_requested(self) -> bool:
        return _STOP or (self.state.dir / "STOP").exists()

    # ------------------------------------------------------------------ discovery
    def discover(self, now: float) -> None:
        unrecognized = {}
        seen = set()
        for task in self.tasks:
            folder = self.raw_root / task
            try:
                names = sorted(os.listdir(folder))
            except FileNotFoundError:
                continue
            for name in names:
                path = folder / name
                if not V.NAME_RE.match(name) or os.path.islink(path) or not os.path.isfile(path):
                    try:
                        st = os.lstat(path)
                        unrecognized[f"{task}/{name}"] = {"size": st.st_size, "mtime": now_iso(st.st_mtime),
                                                          "symlink": os.path.islink(path)}
                    except FileNotFoundError:
                        pass
                    continue
                key = f"{task}/{name}"
                seen.add(key)
                entry = self.state.episodes.get(key)
                if entry is None:
                    entry = self.state.episodes[key] = new_entry(task, name, path, now)
                    self.state.event("discovered", key, size=os.path.getsize(path))
                src = entry["source"]
                if not src.get("present", True):
                    self.state.event("source_reappeared", key)
                    src.pop("missing_since", None)
                src["present"] = True
                src["last_seen"] = now_iso(now)
        for key, entry in self.state.episodes.items():
            if key not in seen and entry["source"].get("present", True) and key.split("/")[0] in self.tasks:
                entry["source"]["present"] = False
                entry["source"]["missing_since"] = now_iso(now)
                self.state.event("source_missing", key, transfer=entry["transfer"]["status"])
        old = set(self.state.doc.get("unrecognized_files", {}))
        for key in sorted(set(unrecognized) - old):  # uploader temp files (".episode_N.h5.XXXXXX") are routine
            self.state.event("unrecognized_file", key, quiet=bool(UPLOAD_TEMP_RE.match(key.split("/", 1)[1])),
                             **unrecognized[key])
        self.state.doc["unrecognized_files"] = unrecognized

    # ------------------------------------------------------------------ validation
    def needs_validation(self, entry: dict, now: float) -> bool:
        if not entry["source"].get("present", True):
            return False
        v = entry["validation"]
        try:
            st = V.stat_fingerprint(Path(entry["source"]["path"]))
        except FileNotFoundError:
            return False
        if v["status"] in ("pending", "waiting", "incomplete"):
            return True
        if v.get("checked_stat") != st or v.get("code_digest") != V.CODE_DIGEST:
            return True
        last = v.get("checked_at_unix", 0)
        if v["status"] == "retrying":
            return now - last >= RECHECK_FAILED_S
        if v["status"] == "invalid":
            return now - last >= RECHECK_INVALID_S
        return False

    def validate_entry(self, entry: dict, now: float) -> None:
        key = entry["key"]
        path = Path(entry["source"]["path"])
        r = V.validate(path, grace_s=self.grace_s, now=now, task_names=self.tasks[entry["task"]],
                       run_adapter=self.run_adapter)
        v = entry["validation"]
        prev_status, prev_sha = v["status"], entry.get("sha256")
        if r["status"] == "missing":
            return
        v["checks"] = v.get("checks", 0) + 1
        v.update(validator_version=r["validator_version"], code_digest=r["code_digest"],
                 checked_at=now_iso(r["checked_at"]), checked_at_unix=r["checked_at"], checked_stat=r.get("stat"),
                 reasons=r["reasons"][:20], warnings=r["warnings"][:40], warning_codes=r.get("warning_codes", []),
                 elapsed_s=r.get("elapsed_s"), age_s=r.get("age_s"))
        st = r.get("stat") or {}
        entry["size"] = st.get("size", entry.get("size"))
        entry["source"].update({k: st[k] for k in ("size", "mtime_ns", "ctime_ns", "ino") if k in st})
        if r["status"] == "waiting":
            v["status"] = "waiting"
        elif r["status"] == "incomplete":
            v["status"] = "incomplete"
            v.setdefault("incomplete_since", now_iso(now))
        elif r["status"] == "failed":
            same = v.get("failed_stat") == st and prev_status in ("retrying", "invalid")
            if not same:
                v["first_failed_at"], v["first_failed_unix"], v["failed_stat"] = now_iso(now), now, st
            v["failed_checks"] = v.get("failed_checks", 0) + 1
            age = now - max(st.get("mtime_ns", 0), st.get("ctime_ns", 0)) / 1e9
            long_failed = now - v["first_failed_unix"] >= self.invalid_after_s and age >= self.invalid_after_s
            v["status"] = "invalid" if long_failed else "retrying"
        elif r["status"] == "valid":
            v["status"] = "valid"
            v["valid_at"] = now_iso(now)
            for k in ("incomplete_since", "first_failed_at", "first_failed_unix", "failed_stat"):
                v.pop(k, None)
            v["adapter"] = r.get("adapter")
            entry["sha256"] = r["sha256"]
            entry["episode_index"] = r.get("episode_index")
            entry["episode_id"] = f"{entry['task']}/{r['episode_index']:06d}"
            entry["attrs"] = r["summary"]
            t = entry["transfer"]
            if prev_sha and prev_sha != r["sha256"]:
                if t["status"] == "verified":
                    entry["flags"]["source_changed_after_transfer"] = (
                        f"local sha256 {r['sha256'][:12]} != transferred {prev_sha[:12]}; remote copy kept")
                    self.state.event("source_changed_after_transfer", key, old=prev_sha, new=r["sha256"])
                else:
                    self.state.event("source_changed", key, old=prev_sha, new=r["sha256"])
                    t.update(status="pending", sha_mismatches=0)
        if v["status"] != prev_status or r["status"] == "valid" and prev_sha != entry.get("sha256"):
            self.state.event("validation", key, status=v["status"], previous=prev_status,
                             reasons=v["reasons"][:3], warning_codes=v.get("warning_codes"),
                             quiet=v["status"] == "waiting" and prev_status in ("pending", "waiting"))
            self._mirror_dirty = True

    # ------------------------------------------------------------------ duplicates / conflicts / overrides
    def detect_conflicts(self) -> None:
        eps = self.state.episodes
        for e in eps.values():
            for f in DERIVED_FLAGS:
                e["flags"].pop(f, None)
            if not e["source"].get("present", True):
                e["flags"]["source_missing"] = f"file gone since {e['source'].get('missing_since')}"
        live = [e for e in eps.values() if e["validation"]["status"] == "valid" and e.get("sha256")
                and e["source"].get("present", True)]
        order = sorted(live, key=lambda e: (e["source"].get("first_seen_unix", 0), e["key"]))
        by_sha: dict = {}
        for e in order:
            by_sha.setdefault(e["sha256"], []).append(e)
        for group in by_sha.values():
            for e in group[1:]:
                e["flags"]["duplicate_content"] = f"same sha256 as {group[0]['key']}"
        by_index: dict = {}
        for e in order:
            if "duplicate_content" not in e["flags"]:
                by_index.setdefault((e["task"], e["episode_index"]), []).append(e)
        for (_task, idx), group in by_index.items():
            if len(group) > 1:
                for e in group:
                    others = [o["key"] for o in group if o is not e]
                    e["flags"]["index_conflict"] = f"episode_index {idx} also in {others} with different content"
        for e in eps.values():
            before = e.get("flags_reported", [])
            now_flags = sorted(e["flags"])
            if now_flags != before:
                self.state.event("flags", e["key"], flags=now_flags, previous=before)
                e["flags_reported"] = now_flags
                self._mirror_dirty = True

    def decision(self, entry: dict, overrides: dict) -> str | None:
        o = overrides.get(entry["key"])
        return o.get("decision") if isinstance(o, dict) else None

    def blocked(self, entry: dict, overrides: dict) -> list[str]:
        d = self.decision(entry, overrides)
        if d == "exclude":
            return ["excluded_by_override"]
        flags = [f for f in entry["flags"] if f in BLOCKING_FLAGS]
        if d == "accept":
            flags = [f for f in flags if f == "source_missing"]
        return flags

    # ------------------------------------------------------------------ counts
    def counted(self, e: dict, overrides: dict) -> bool:
        """The single predicate for 'counts toward the 100': present, valid, not blocked; and, for 'transferred',
        the verified remote copy is this content."""
        return (e["source"].get("present", True) and e["validation"]["status"] == "valid"
                and not self.blocked(e, overrides)
                and (e["transfer"]["status"] != "verified" or e["transfer"].get("remote_sha256") == e.get("sha256")))

    def counts(self, overrides: dict) -> dict:
        out = {}
        for task in self.tasks:
            es = [e for e in self.state.episodes.values() if e["task"] == task]
            present = [e for e in es if e["source"].get("present", True)]
            valid = [e for e in present if self.counted(e, overrides)]
            c = {
                "target": TARGET_PER_TASK,
                "received": len(present),
                "waiting": sum(e["validation"]["status"] in ("pending", "waiting") for e in present),
                "incomplete": sum(e["validation"]["status"] == "incomplete" for e in present),
                "retrying": sum(e["validation"]["status"] == "retrying" for e in present),
                "invalid": sum(e["validation"]["status"] == "invalid" for e in present),
                "valid": len(valid),
                "valid_with_warnings": sum(bool(e["validation"].get("warnings")) for e in valid),
                "adapter_ready": sum(bool((e["validation"].get("adapter") or {}).get("ok")) for e in valid),
                "transferred": sum(e["transfer"]["status"] == "verified" for e in valid),
                "transfer_partial": sum(e["transfer"]["status"] == "partial" for e in valid),
                "blocked": sorted(e["key"] for e in present if e["validation"]["status"] == "valid"
                                  and self.blocked(e, overrides)),
                "missing_after_seen": sorted(e["key"] for e in es if not e["source"].get("present", True)),
                "unique_episode_indices": len({e["episode_index"] for e in valid}),
                "bytes_valid": sum(e.get("size") or 0 for e in valid),
                "bytes_transferred": sum(e.get("size") or 0 for e in valid if e["transfer"]["status"] == "verified"),
            }
            c["remaining_to_target"] = max(0, TARGET_PER_TASK - c["valid"])
            c["complete"] = c["valid"] >= TARGET_PER_TASK and c["transferred"] >= TARGET_PER_TASK
            out[task] = c
        return out

    # ------------------------------------------------------------------ transfer
    def eligible(self, e: dict, overrides: dict) -> bool:
        return (e["validation"]["status"] == "valid" and e["source"].get("present", True)
                and not self.blocked(e, overrides) and e["transfer"]["status"] != "verified")

    def transfer_candidates(self, overrides: dict) -> list[dict]:
        now = time.time()
        out = []
        for e in self.state.episodes.values():
            t = e["transfer"]
            ident = json.dumps(overrides.get(e["key"]), sort_keys=True)
            if self.decision(e, overrides) == "retry" and "transfer_failed_repeatedly" in e["flags"] and \
                    t.get("retry_override_consumed") != ident:  # each distinct override record acts once
                e["flags"].pop("transfer_failed_repeatedly")
                t.update(sha_mismatches=0, status="pending", consecutive_failures=0, next_attempt_unix=0,
                         retry_override_consumed=ident)
                self.state.event("transfer_retry_override", e["key"])
            if self.eligible(e, overrides) and t.get("next_attempt_unix", 0) <= now:
                out.append(e)
        # resumable partials first, then arrival order
        return sorted(out, key=lambda e: (e["transfer"]["status"] != "partial",
                                          e["source"].get("mtime_ns", 0), e["key"]))

    def _hold(self, entry: dict, reason: str, retry_s: float | None = None) -> None:
        t = entry["transfer"]
        if t.get("hold_reason") != reason:
            self.state.event("transfer_hold", entry["key"], reason=reason)
        t["hold_reason"] = reason
        if retry_s:
            t["next_attempt_unix"] = time.time() + retry_s
            t["next_attempt_at"] = now_iso(t["next_attempt_unix"])

    def _failure(self, entry: dict, message: str) -> None:
        t = entry["transfer"]
        t["failures"] = t.get("failures", 0) + 1
        t["consecutive_failures"] = t.get("consecutive_failures", 0) + 1
        t["last_error"] = message[-500:]
        t["last_error_at"] = now_iso()
        backoff = min(60 * 2 ** (t["consecutive_failures"] - 1), 3600)
        t["next_attempt_unix"] = time.time() + backoff
        t["next_attempt_at"] = now_iso(t["next_attempt_unix"])
        self.state.event("transfer_error", entry["key"], error=message[-300:], retry_in_s=backoff)

    def transfer_one(self, entry: dict) -> bool:
        """Returns True when the episode is verified on the remote. Raises Unreachable."""
        key, task, name = entry["key"], entry["task"], entry["filename"]
        t = entry["transfer"]
        local = Path(entry["source"]["path"])
        st = V.stat_fingerprint(local)
        if st != entry["validation"].get("checked_stat"):
            entry["validation"]["status"] = "pending"  # changed since validation: re-validate first
            self.state.event("source_changed_before_transfer", key)
            return False
        final, part = self.remote.final_path(task, name), self.remote.part_path(task, name)
        t.update(remote_path=final, remote_part=part)
        self.remote.ensure_dirs(task, create=not self.dry_run)  # containment (no symlinked ancestors) first
        probe = self.remote.probe(task, name)
        t["remote_probe"] = {k: probe[k] for k in ("final_size", "part_size")}
        if probe["symlink"]:
            entry["flags"]["remote_conflict"] = "remote final/part path is a symlink"
            self.state.event("remote_conflict", key, reason="symlink")
            return False
        sha = entry["sha256"]
        if probe["final_size"] is not None:
            remote_sha = self.remote.sha256(final)
            if remote_sha == sha:
                if probe["part_size"] is not None and not self.dry_run:
                    self.remote.remove_part(part)
                self._verified(entry, remote_sha, final, already=True)
                return True
            entry["flags"]["remote_conflict"] = (f"remote {final} exists with sha256 {str(remote_sha)[:12]} != local "
                                                 f"{sha[:12]}; not overwritten")
            t["status"] = "conflict"
            self.state.event("remote_conflict", key, remote_sha=remote_sha, local_sha=sha)
            return False
        need = entry["size"] - (probe["part_size"] or 0)
        self.remote_disk = {"checked_at": now_iso(), "fs_size": probe["fs_size"], "fs_avail": probe["fs_avail"],
                            "free_fraction": round(probe["fs_avail"] / probe["fs_size"], 4),
                            "data_real_world_bytes": probe["rw_used"]}
        if (probe["fs_avail"] - need) / probe["fs_size"] < REMOTE_MIN_FREE_FRACTION:
            self._hold(entry, f"remote SSD free space would drop below {REMOTE_MIN_FREE_FRACTION:.0%}", 1800)
            raise _PauseTransfers("remote disk")
        kib, budget = self.upload_budget()
        if kib <= 0:
            self._hold(entry, "upload budget 0 (paused by upload-budget.json)")
            raise _PauseTransfers("budget 0")
        if self.dry_run:
            log(f"DRY-RUN would upload {key} ({entry['size'] / 1e6:.1f} MB, part {probe['part_size']}) "
                f"at {kib} KiB/s -> {part} -> {final}")
            return False
        self.remote.ensure_dirs(task)
        t.pop("hold_reason", None)
        t["attempts"] = t.get("attempts", 0) + 1
        t["status"] = "partial"
        t["started_at"] = now_iso()
        t["resumed_from_bytes"] = probe["part_size"] or 0
        t["kib_per_s"] = kib  # lowered by _run_rsync when the budget drops mid-transfer
        self.state.save()
        self.state.event("transfer_start", key, size=entry["size"], resume_from=probe["part_size"], kib_per_s=kib,
                         budget=budget)
        started = time.time()
        rc, err = self._run_rsync(local, part, kib, entry)
        t["rsync_s"] = round(time.time() - started, 1)
        if rc is None:  # stopped / paused
            return False
        if rc != 0:
            if rc in (10, 12, 20, 30, 35, 255):
                self._failure(entry, f"rsync exit {rc} (transport/interrupted; partial kept): {err}")
                raise Unreachable(f"rsync exit {rc}")
            if rc == 23 and probe["part_size"] is not None:  # e.g. a stale read-only part of older content
                self.remote.remove_part(part)
                err += " (stale part removed)"
            self._failure(entry, f"rsync exit {rc}: {err}")
            return False
        if V.stat_fingerprint(local) != st:
            entry["validation"]["status"] = "pending"
            self.state.event("source_changed_during_transfer", key)
            return False
        part_sha = self.remote.sha256(part)
        if part_sha != sha:
            t["sha_mismatches"] = t.get("sha_mismatches", 0) + 1
            self.remote.remove_part(part)
            msg = f"remote part sha256 {str(part_sha)[:12]} != local {sha[:12]} (part removed, full retransfer)"
            if t["sha_mismatches"] >= MAX_SHA_MISMATCHES:
                entry["flags"]["transfer_failed_repeatedly"] = msg
                t["status"] = "failed"
            self._failure(entry, msg)
            return False
        if not self.eligible(entry, self.overrides()):  # flagged (e.g. index conflict) by a scan during the upload
            self._hold(entry, f"blocked before publish by {sorted(entry['flags']) or 'override'}; verified part kept")
            return False
        outcome = self.remote.publish(part, final)
        if outcome == "NOPART":  # the verified part vanished before publish: transfer again
            self._failure(entry, "part disappeared before publish")
            return False
        final_sha = self.remote.sha256(final)
        if final_sha != sha:
            entry["flags"]["remote_conflict"] = f"after publish ({outcome}) remote sha256 {str(final_sha)[:12]} != {sha[:12]}"
            t["status"] = "conflict"
            self.state.event("remote_conflict", key, outcome=outcome, remote_sha=final_sha)
            return False
        t["duration_s"] = round(time.time() - started, 1)  # rsync + remote sha256 + publish + sha256
        sent = entry["size"] - (probe["part_size"] or 0)
        t["effective_kib_per_s"] = round(sent / 1024 / max(t["duration_s"], 1e-3), 1)
        t["rsync_kib_per_s"] = round(sent / 1024 / max(t["rsync_s"], 1e-3), 1)
        t["sent_bytes"] = sent
        self._verified(entry, final_sha, final, already=False, outcome=outcome)
        return True

    def _verified(self, entry: dict, remote_sha: str, final: str, already: bool, outcome: str = "") -> None:
        t = entry["transfer"]
        t.update(status="verified", remote_sha256=remote_sha, remote_path=final, verified_at=now_iso(),
                 consecutive_failures=0, verified_size=entry["size"], verified_episode_index=entry["episode_index"])
        for k in ("hold_reason", "next_attempt_unix", "next_attempt_at"):
            t.pop(k, None)
        self.transfers_done += 0 if already else 1
        self._mirror_dirty = True
        self.state.event("transfer_verified", entry["key"], remote_path=final, sha256=remote_sha,
                         already_present=already, outcome=outcome, duration_s=t.get("duration_s"),
                         effective_kib_per_s=t.get("effective_kib_per_s"),
                         **({} if already else {k: t.get(k) for k in ("rsync_s", "rsync_kib_per_s", "sent_bytes",
                                                                        "kib_per_s")}))

    def _run_rsync(self, local: Path, part: str, kib: int, entry: dict) -> tuple[int | None, str]:
        """rsync with budget polling: a budget reduction (incl. a lapsed boost lease) restarts it at once with the
        lower limit (the part is resumed); an increase applies from the next transfer, so a valid in-flight transfer
        is not interrupted. STOP/signal/0 ends it (part kept). A watchdog thread enforces reductions even while this
        loop is busy (e.g. validating new arrivals); whatever happens, the rsync never outlives this call."""
        while True:
            _kill_stray_uploads(part)
            cmd, env = self.remote.rsync_command(local, part, kib)
            size = entry["size"]
            deadline = time.time() + size / (max(kib, 1) * 1024) * 2 + 900
            errfile = self.state.dir / "rsync-current.err"
            with open(errfile, "w") as err:
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err,
                                        preexec_fn=_child_setup())
            restart_with = None
            self._inflight = cur = {"proc": proc, "kib": kib, "reduced_to": None}
            watch_stop = threading.Event()
            watcher = threading.Thread(target=self._budget_watchdog, args=(cur, watch_stop), daemon=True)
            try:
                (self.state.dir / "rsync-current.json").write_text(json.dumps(
                    {"pid": proc.pid, "key": entry["key"], "kib_per_s": kib, "started_at": now_iso()}))
                watcher.start()
                while True:
                    if cur["reduced_to"] is None:
                        if proc.poll() is not None:
                            break  # rsync ended by itself
                        time.sleep(2.0)
                    if time.time() - self.last_scan >= SCAN_DURING_TRANSFER_S and cur["reduced_to"] is None:
                        try:
                            self.scan(skip_key=entry["key"])
                            self.write_status(self.state.doc["counts"], {"remote": "ok",
                                                                         "transferring": entry["key"]})
                        except Exception:  # noqa: BLE001 - a scan problem must not break the running upload
                            log("scan during transfer failed:\n" + traceback.format_exc())
                            self.last_scan = time.time()
                    if self.stop_requested():
                        _terminate(proc)
                        self.state.event("transfer_interrupted", entry["key"], reason="stop requested; partial kept")
                        return None, ""
                    new = self.upload_kib()
                    if cur["reduced_to"] is not None:  # latched by the watchdog: never restart above it
                        new = min(new, cur["reduced_to"])
                    if new < kib:
                        _terminate(proc)
                        if new <= 0:
                            self._hold(entry, "upload budget 0 (paused by upload-budget.json)")
                            self.state.event("transfer_interrupted", entry["key"], reason="budget 0; partial kept")
                            return None, ""
                        restart_with = new
                        break
                    if time.time() > deadline:
                        _terminate(proc)
                        return 30, "local transfer deadline exceeded"
                watch_stop.set()  # settle the watchdog before deciding: a late latch is a reduction, not a failure
                if watcher.is_alive():
                    watcher.join(timeout=45)
                if cur["reduced_to"] is not None:
                    _terminate(proc)
                    low = min(self.upload_kib(), cur["reduced_to"])
                    restart_with = low if restart_with is None else min(restart_with, low)
                    if restart_with <= 0:
                        self._hold(entry, "upload budget 0 (paused by upload-budget.json)")
                        self.state.event("transfer_interrupted", entry["key"], reason="budget 0; partial kept")
                        return None, ""
            finally:  # every exit (normal, restart, stop, pause, deadline, exception): no rsync left running
                watch_stop.set()
                _terminate(proc)
                self._inflight = None
                with contextlib.suppress(OSError):
                    (self.state.dir / "rsync-current.json").unlink()
            rc = proc.returncode
            err_text = self.remote.redact(errfile.read_text(errors="replace"))[-500:].strip()
            if restart_with is not None:
                self.state.event("budget_changed_mid_transfer", entry["key"], old=kib, new=restart_with)
                kib = restart_with
                entry["transfer"]["kib_per_s"] = kib
                continue
            return rc, err_text

    def _budget_watchdog(self, cur: dict, stop: threading.Event) -> None:
        """Thread: every 2 s, a budget below the running limit terminates the rsync and latches the lowest budget seen
        (read-only check: no events, no state changes besides cur)."""
        while not stop.wait(2.0):
            if cur["proc"].poll() is not None:
                return
            base = base_budget_kib(self.state.dir)
            eff = boost_level(self.state.dir, base)[0] or base
            if eff < cur["kib"]:
                cur["reduced_to"] = eff
                _terminate(cur["proc"])
                return

    # ------------------------------------------------------------------ remote mirror
    def mirror(self, counts: dict) -> None:
        overrides = self.overrides()
        if not (self.mirror_enabled and self._mirror_dirty and self.remote and not self.dry_run):
            return
        doc = {"schema": "rw-ingest-remote-mirror/1", "updated_at": now_iso(), "authoritative":
               "local manifest /home/ws/ws/realworld-ops/ingest/manifest.json (this is an informational copy)",
               "counts": {k: {x: v[x] for x in ("target", "received", "valid", "transferred", "adapter_ready")}
                          for k, v in counts.items()},
               "episodes": {}}
        for e in self.state.episodes.values():
            if e["transfer"]["status"] == "verified":
                t = e["transfer"]
                doc["episodes"][e["key"]] = {  # what was verified on the remote, not the current local file
                    "sha256": t["remote_sha256"], "size": t.get("verified_size"),
                    "episode_index": t.get("verified_episode_index"),
                    "remote_path": t["remote_path"], "verified_at": t.get("verified_at"),
                    "validation": e["validation"]["status"],
                    "warning_codes": e["validation"].get("warning_codes", []),
                    "adapter_ok": (e["validation"].get("adapter") or {}).get("ok"),
                    "flags": sorted(e["flags"]), "counted": self.counted(e, overrides)}
        local = self.state.dir / "remote-mirror.json"
        atomic_write_json(local, doc)
        kib = self.base_kib() or 1  # informational file: never boosted
        self.remote.ensure_dirs(next(iter(self.tasks)))
        self.remote.upload_small(local, f"{REMOTE_DATA_REL}/ingest-manifest.json", kib)
        self._mirror_dirty = False

    # ------------------------------------------------------------------ one pass
    def scan(self, skip_key: str | None = None) -> int:
        """Discovery + (re)validation + conflicts + counts; also runs every 5 min during a long upload (the entry
        being uploaded is skipped). Returns the number of files validated."""
        overrides = self.overrides()
        self.discover(time.time())
        validated = 0
        for key in sorted(self.state.episodes):
            if self.stop_requested():
                break
            e = self.state.episodes[key]
            if key != skip_key and key.split("/")[0] in self.tasks and self.needs_validation(e, time.time()):
                self.validate_entry(e, time.time())
                validated += 1
                if validated % 10 == 0:
                    self.state.save()
        self.detect_conflicts()
        for e in self.state.episodes.values():
            e["override"] = overrides.get(e["key"])
        self.state.doc["counts"] = self.counts(overrides)
        self.state.doc["worker"] = {"version": WORKER_VERSION, "validator": V.VALIDATOR_VERSION,
                                    "code_digest": V.CODE_DIGEST, "pid": os.getpid(), "host": socket.gethostname()}
        self.state.save()
        self.last_scan = time.time()
        return validated

    def run_pass(self) -> dict:
        t0 = time.time()
        validated = self.scan()
        overrides = self.overrides()
        result = {"validated": validated, "transferred": 0, "remote": "skipped", "pending_transfers": 0}
        cands = self.transfer_candidates(overrides)
        result["pending_transfers"] = len(cands)
        kib = self.upload_kib()
        if kib != self._last_kib:
            self.state.event("upload_budget", kib_per_s=kib)
            self._last_kib = kib
        if self.transfer_enabled and self.remote is not None and not self.stop_requested() and \
                (cands or self._mirror_dirty):
            try:
                self.remote.reachable()
                if self.remote_unreachable_since:
                    self.state.event("remote_reachable", outage_s=round(time.time() - self.remote_unreachable_since))
                    self.remote_unreachable_since = None
                result["remote"] = "ok"
                window_end = time.time() + self.window_s
                for e in cands:
                    if self.stop_requested() or time.time() > window_end:
                        break
                    if self.max_transfers is not None and self.transfers_done >= self.max_transfers:
                        break
                    overrides = self.overrides()  # a scan during the previous upload may have changed flags
                    if not self.eligible(e, overrides) or e["transfer"].get("next_attempt_unix", 0) > time.time():
                        continue
                    try:
                        if self.transfer_one(e):
                            result["transferred"] += 1
                            e["transfer"]["consecutive_failures"] = 0
                    except RemoteError as exc:
                        self._failure(e, f"remote error: {exc}")
                    finally:
                        self.state.doc["counts"] = self.counts(overrides)
                        self.state.save()
                self.mirror(self.state.doc["counts"])
            except _PauseTransfers as exc:
                result["remote"] = f"paused ({exc})"
            except Unreachable as exc:
                result["remote"] = "unreachable"
                if not self.remote_unreachable_since:
                    self.remote_unreachable_since = time.time()
                    self.state.event("remote_unreachable", error=str(exc)[:300])
            except RemoteError as exc:
                result["remote"] = f"error: {exc}"
                self.state.event("remote_error", error=str(exc)[:300])
        counts = self.counts(overrides)
        self.state.doc["counts"] = counts
        self.state.doc["last_pass"] = {"at": now_iso(), "duration_s": round(time.time() - t0, 1), **result}
        result["pending_transfers"] = len(self.transfer_candidates(overrides))
        self.state.doc["last_pass"]["pending_transfers"] = result["pending_transfers"]
        self.state.save()
        self.write_status(counts, result)
        return result

    def write_status(self, counts: dict, result: dict) -> None:
        issues = {
            "invalid": {}, "retrying": {}, "incomplete_stale": {}, "flags": {}, "warnings": {},
            "unrecognized_files": self.state.doc.get("unrecognized_files", {}),
        }
        for e in self.state.episodes.values():
            v = e["validation"]
            if v["status"] in ("invalid", "retrying"):
                issues[v["status"]][e["key"]] = v["reasons"][:5]
            if v["status"] == "incomplete" and (v.get("age_s") or 0) > 3600:
                issues["incomplete_stale"][e["key"]] = v["reasons"][:2]
            if e["flags"]:
                issues["flags"][e["key"]] = e["flags"]
            if v["status"] == "valid" and v.get("warning_codes"):
                issues["warnings"][e["key"]] = v["warning_codes"]
        kib, budget = self.upload_budget()
        doc = {"updated_at": now_iso(), "counts": counts, "last_pass": self.state.doc.get("last_pass"),
               "upload_kib_per_s": kib, "upload_budget": {"source": budget, "base_kib_per_s": self.base_kib(),
                                                          "boost": self._boost_why},
               "issues": issues, "remote_disk": self.remote_disk,
               "manifest_schema": [MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION],
               "remote_unreachable_since": now_iso(self.remote_unreachable_since)
               if self.remote_unreachable_since else None}
        atomic_write_json(self.state.dir / "status.json", doc)
        line = " | ".join(f"{t}: recv {c['received']} valid {c['valid']} xfer {c['transferred']}/{c['target']}"
                          f" (wait {c['waiting']}, incompl {c['incomplete']}, retry {c['retrying']}, "
                          f"invalid {c['invalid']}, blocked {len(c['blocked'])})" for t, c in counts.items())
        summary = f"{line} | remote {result.get('remote')}"
        # quiet: only when the state changed, otherwise an hourly heartbeat
        if summary != self._last_summary or time.time() - self._last_summary_at >= HEARTBEAT_S:
            tag = "state" if summary != self._last_summary else "heartbeat"
            log(f"{tag}: {summary} | pending uploads {result.get('pending_transfers', '?')}")
            self._last_summary, self._last_summary_at = summary, time.time()


class _PauseTransfers(Exception):
    pass


def _terminate(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _kill_stray_uploads(part: str) -> None:
    """A previous worker's rsync still writing the same part (e.g. after SIGKILL) would race this one."""
    me = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            argv = Path("/proc", name, "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and os.path.basename(argv[0].decode(errors="replace")) == "rsync" and \
                any(a.endswith((":" + part).encode()) for a in argv):
            log(f"terminating stray upload pid {name} for {part}")
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(name), sig)
                for _ in range(wait):
                    if not Path("/proc", name).exists() or _is_zombie(name):
                        break
                    time.sleep(1)
                else:
                    continue
                break
            else:
                raise RemoteError(f"stray upload pid {name} for {part} does not exit; transfer refused")


def _is_zombie(pid: str) -> bool:
    try:
        return Path("/proc", pid, "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return True


# ------------------------------------------------------------------------------------------------ preprocessing


def set_stage(state_dir: Path, key: str, stage: str, record: dict) -> dict:
    """Record one preprocessing stage of one episode (preprocessing agents; the worker never changes these).

    Every stage record carries: stage (name), stage_schema_version (int, the record layout of that stage), code_version
    (str, required), status (pending|running|done|failed|skipped), output_format (str, e.g. "rw-train-crop128/1";
    required when done), timestamps (started_at/finished_at; updated_at added here) and failure ({message, at, ...} or
    null). Optional: code_commit, config_sha256, input_sha256 (must equal the episode sha256), outputs {name: remote
    path}, host, attempt, notes, metrics."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", stage):
        raise ValueError("stage names are lowercase [a-z0-9_.-]")
    unknown = set(record) - STAGE_FIELDS
    if unknown:
        raise ValueError(f"unknown stage fields {sorted(unknown)} (allowed {sorted(STAGE_FIELDS)})")
    missing = [k for k in STAGE_FIELDS_REQUIRED if not record.get(k)]
    if missing:
        raise ValueError(f"stage record needs {missing}")
    if record["status"] not in STAGE_STATUSES:
        raise ValueError(f"status must be one of {STAGE_STATUSES}")
    if record["status"] == "done" and not record.get("output_format"):
        raise ValueError("a done stage must name its output_format (format name/version)")
    if record["status"] == "failed" and not record.get("failure"):
        raise ValueError("a failed stage must describe its failure")
    state_dir = Path(state_dir)
    lock_path, manifest_path = state_dir / "manifest.lock", state_dir / "manifest.json"
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = migrate(json.loads(manifest_path.read_text()))
        entry = doc["episodes"][key]
        if record.get("input_sha256") and record["input_sha256"] != entry.get("sha256"):
            raise ValueError("input_sha256 does not match the episode's sha256")
        pre = entry.setdefault("preprocessing", new_preprocessing())
        rec = {"stage": stage, "stage_schema_version": int(record.get("stage_schema_version", 1)), "failure": None,
               **record, "updated_at": now_iso()}
        pre["stages"][stage] = rec
        statuses = [x.get("status") for x in pre["stages"].values()]
        pre["status"] = ("failed" if "failed" in statuses else "running" if "running" in statuses else
                         "done" if statuses and all(x in ("done", "skipped") for x in statuses) else "pending")
        doc["updated_at"] = now_iso()
        atomic_write_json(manifest_path, doc)
    ev = State.__new__(State)
    ev.events_path = state_dir / "events.jsonl"
    ev.event("preprocessing_stage", key, stage=stage, status=record["status"],
             code_version=record["code_version"], output_format=record.get("output_format"))
    return rec


# --------------------------------------------------------------------------------------------------------- main


@contextlib.contextmanager
def worker_lock(state_dir: Path):
    path = state_dir / "worker.lock"
    fd = open(path, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        pid = (state_dir / "worker.pid").read_text() if (state_dir / "worker.pid").exists() else "?"
        raise SystemExit(f"another ingest worker holds {path} ({pid.strip()[:200]})") from None
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _repo_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="real-world episode ingestion + transfer worker")
    ap.add_argument("command", nargs="?", default="run", choices=("run", "set-stage"))
    ap.add_argument("args", nargs="*")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--state-dir", type=Path, default=STATE_DIR)
    ap.add_argument("--dry-run", action="store_true", help="validate + probe the remote read-only; no upload")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--max-transfers", type=int)
    ap.add_argument("--window", type=float, default=TRANSFER_WINDOW_S)
    ap.add_argument("--grace", type=float, default=GRACE_S)
    a = ap.parse_args(argv)
    if a.command == "set-stage":
        key, stage, record = a.args
        set_stage(a.state_dir, key, stage, json.loads(record))
        return 0
    if a.status:
        print((a.state_dir / "status.json").read_text())
        return 0
    if not (a.once or a.loop):
        ap.error("one of --once / --loop / --status")
    os.setpriority(os.PRIO_PROCESS, 0, max(15, os.getpriority(os.PRIO_PROCESS, 0)))
    with contextlib.suppress(Exception):
        subprocess.run(["ionice", "-c2", "-n7", "-p", str(os.getpid())], check=False, capture_output=True)
    a.state_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with worker_lock(a.state_dir):
        remote = None if a.no_transfer else RemoteOps()
        w = Worker(a.state_dir, remote=remote, dry_run=a.dry_run, transfer=not a.no_transfer,
                   max_transfers=a.max_transfers, window_s=a.window, grace_s=a.grace, run_adapter=not a.no_adapter)
        pidinfo = {"pid": os.getpid(), "started_at": now_iso(), "argv": sys.argv, "python": sys.executable,
                   "host": socket.gethostname(), "worker": WORKER_VERSION, "validator": V.VALIDATOR_VERSION,
                   "code_digest": V.CODE_DIGEST, "repo_head": _repo_head(), "mode": "loop" if a.loop else "once"}
        if not a.dry_run:
            atomic_write_json(a.state_dir / "worker.pid", pidinfo)
        w.state.event("worker_start", **{k: pidinfo[k] for k in ("pid", "mode", "code_digest", "repo_head")},
                      dry_run=a.dry_run)
        errors = 0
        try:
            while True:
                try:
                    res = w.run_pass()
                    errors = 0
                except Exception as exc:  # noqa: BLE001 - one bad pass must not kill the loop
                    errors += 1
                    res = {}
                    log("pass failed:\n" + traceback.format_exc())
                    w.state.event("pass_error", error=f"{type(exc).__name__}: {str(exc)[:300]}", consecutive=errors)
                    if a.once:
                        return 1
                if a.once or _STOP or w.stop_requested():
                    break
                if errors:
                    delay = min(a.interval * 2 ** errors, 1800)
                elif res.get("remote") == "unreachable":
                    outage = time.time() - (w.remote_unreachable_since or time.time())
                    delay = min(max(a.interval, outage / 4), 900)
                elif res.get("pending_transfers") and res.get("remote") == "ok" and \
                        (a.max_transfers is None or w.transfers_done < a.max_transfers):
                    delay = 5
                else:
                    delay = a.interval
                end = time.time() + delay
                while time.time() < end and not (_STOP or w.stop_requested()):
                    time.sleep(1.0)
                if _STOP or w.stop_requested():
                    break
        finally:
            w.state.event("worker_stop", pid=os.getpid(), stop_file=(a.state_dir / "STOP").exists())
            if not a.dry_run:
                with contextlib.suppress(OSError):
                    pidinfo.update(stopped_at=now_iso(), running=False)
                    atomic_write_json(a.state_dir / "worker.pid", pidinfo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
