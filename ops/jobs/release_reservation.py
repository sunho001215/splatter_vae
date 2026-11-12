"""Release-verification resource reservation (user 2026-11-12: release verification has the highest scheduling
priority over admitting new training or preprocessing work; never preempt running jobs; no idle GPU is reserved without
a pending release).

While a remote release is verified (ops/campaign/release_chain.py), runs/setup/release-reservation.json declares the
remote resources its verification jobs need: per GPU a job slot, GPU memory and load units, plus host RAM, CPU threads
and disk. Every remote admission path charges it before placing any other job, so newly admitted work never consumes
it; the release's own verification jobs are admitted against it:

- the scheduler tick (ops/jobs/scheduler.py ``_tick``): campaign rows see phantom running entries on the reserved GPUs
  (slot + memory), host RAM, disk and CPU threads reduced by the reservation; the release's verification jobs are
  admitted without these charges (every other admission rule, e.g. the disk floor the remote launch guard enforces,
  is unchanged). Every remote launch goes through the tick under the registry lock, and the chain publishes a new
  reservation content under that lock, so no tick admits work against a stale view;
- the remote campaign driver's placement and pool moves (ops/campaign/remote_campaign_driver.py): load units and
  memory per GPU and the RAM cap are charged before any candidate is placed;
- the real-world preprocessing dispatcher reads ``reserved_gpus``.

Running jobs are never touched: if capacity is already in use the release waits for it to free, and no new work takes
it meanwhile. The reservation is valid only while the process that wrote it is alive, before its expiry (renewed by
the chain's heartbeat thread every 60 s; a crashed chain's reservation lapses within EXPIRY_SECONDS) and within
MAX_LIFETIME_SECONDS of its creation. One release at a time: a live reservation of another release or writer is never
overwritten (ReservationConflict); writes and clears are serialized by a lock file. A missing file means no
reservation; an unreadable or malformed file (incl. non-finite or future timestamps) raises ValueError so that callers
fail closed (no admission).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "runs/setup/release-reservation.json"
EXPIRY_SECONDS = 900  # a reservation not renewed within this window lapses (the chain's heartbeat renews every 60 s)
MAX_LIFETIME_SECONDS = 4 * 3600  # never longer than this after its creation (suite timeout 2 h + isolation + slack)
CLOCK_SLACK_SECONDS = 120
SCHEMA = 1


class ReservationConflict(ValueError):
    """Another live release already holds the reservation."""
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate(doc: dict) -> dict:
    """The reservation document, checked; raises ValueError when malformed."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("release reservation: unknown schema")
    if not isinstance(doc.get("commit"), str) or not _HEX40.fullmatch(doc["commit"]):
        raise ValueError("release reservation: commit must be a full sha")
    if type(doc.get("pid")) is not int or doc["pid"] <= 0:
        raise ValueError("release reservation: owner pid")
    for key in ("created", "expires"):
        if type(doc.get(key)) not in (int, float) or not math.isfinite(doc[key]):
            raise ValueError(f"release reservation: {key}")
    if not 0 < doc["expires"] - doc["created"] <= MAX_LIFETIME_SECONDS + EXPIRY_SECONDS:
        raise ValueError("release reservation: lifetime")
    jobs = doc.get("jobs")
    if not isinstance(jobs, list) or not jobs or not all(isinstance(i, str) and i.startswith("verify-") for i in jobs):
        raise ValueError("release reservation: jobs must be verify-* ids")
    gpus = doc.get("gpus")
    if not isinstance(gpus, dict) or not gpus:
        raise ValueError("release reservation: gpus")
    for uuid, charge in gpus.items():
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or not isinstance(charge, dict):
            raise ValueError("release reservation: gpu entry")
        for key in ("mem_gb", "load"):
            value = charge.get(key)
            if type(value) not in (int, float) or not 0 <= value <= 96:
                raise ValueError(f"release reservation: gpu {key}")
    for key in ("ram_gb", "cpu_threads", "disk_gb"):
        value = doc.get(key)
        if type(value) not in (int, float) or not 0 <= value <= 1024:
            raise ValueError(f"release reservation: {key}")
    return doc


def load(path: Path | None = None, now: float | None = None, alive=_alive) -> dict | None:
    """The active reservation, or None (no file, expired, or its owner process is gone). Malformed: ValueError."""
    path = PATH if path is None else path
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"release reservation unreadable ({type(exc).__name__})") from exc
    try:
        doc = validate(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError("release reservation is not JSON") from exc
    now = time.time() if now is None else now
    if doc["expires"] > now + EXPIRY_SECONDS + CLOCK_SLACK_SECONDS or doc["created"] > now + CLOCK_SLACK_SECONDS:
        raise ValueError("release reservation: timestamps in the future")
    if now > doc["expires"] or now - doc["created"] > MAX_LIFETIME_SECONDS or not alive(doc["pid"]):
        return None
    return doc


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def write(doc: dict, path: Path | None = None, alive=_alive) -> None:
    """Publish or renew ``doc``. Refuses (ReservationConflict) while another release or writer holds a live
    reservation; an expired, orphaned or malformed one is replaced."""
    path = PATH if path is None else path
    validate(doc)
    with _locked(path):
        try:
            current = load(path, alive=alive)
        except ValueError:
            current = None
        if current is not None and (current["commit"], current["pid"]) != (doc["commit"], doc["pid"]):
            raise ReservationConflict(f"release {current['short']} (pid {current['pid']}) holds the reservation")
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)


def clear(commit: str, path: Path | None = None, pid: int | None = None) -> bool:
    """Remove the reservation of ``commit`` written by ``pid`` (default: this process). Another release's or another
    writer's reservation is left in place."""
    path = PATH if path is None else path
    with _locked(path):
        try:
            doc = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if doc.get("commit") != commit or doc.get("pid") != (os.getpid() if pid is None else pid):
            return False
        path.unlink(missing_ok=True)
        return True


def covers(doc: dict | None, job: dict) -> bool:
    """True for a verification job of the reserved release (its id is listed and it declares a verification purpose)."""
    return bool(doc) and job.get("id") in doc["jobs"] and "verification_purpose" in job


def charges(doc: dict | None) -> dict:
    """{'gpus': {uuid: {'mem_gb', 'load'}}, 'ram_gb', 'cpu_threads', 'disk_gb'} to charge against other admissions."""
    if not doc:
        return {"gpus": {}, "ram_gb": 0.0, "cpu_threads": 0, "disk_gb": 0.0}
    return {"gpus": {uuid: {"mem_gb": float(c["mem_gb"]), "load": float(c["load"])} for uuid, c in doc["gpus"].items()},
            "ram_gb": float(doc["ram_gb"]), "cpu_threads": int(doc["cpu_threads"]), "disk_gb": float(doc["disk_gb"])}


def reserved_gpus(path: Path | None = None) -> set[str]:
    """GPUs reserved for a pending release (for dispatchers outside the scheduler). Malformed: ValueError."""
    doc = load(path)
    return set(doc["gpus"]) if doc else set()


def plan(commit: str, short: str, stage: str, native_gpu: str, gpus: list[str], *, native: dict, isolation: dict,
         pid: int | None = None, now: float | None = None, created: float | None = None) -> dict:
    """The reservation document for a release at ``stage`` ('native' or 'isolation'). The native stage reserves the
    native suite's resources on its GPU AND the isolation checks' small share on every GPU, so the isolation stage
    launches without waiting; the isolation stage keeps only the isolation share."""
    if stage not in ("native", "isolation"):
        raise ValueError("stage")
    now = time.time() if now is None else now
    share = {uuid: {"mem_gb": float(isolation["mem_gb"]), "load": float(isolation["load"])} for uuid in gpus}
    jobs = [f"verify-isolation-{i}-{short}" for i in range(len(gpus))]
    ram, cpu, disk = (float(isolation["ram_gb"]) * len(gpus), int(isolation["cpu_threads"]) * len(gpus),
                      float(isolation["disk_gb"]) * len(gpus))
    if stage == "native":
        native_share = share.setdefault(native_gpu, {"mem_gb": 0.0, "load": 0.0})
        native_share["mem_gb"] += float(native["mem_gb"])
        native_share["load"] += float(native["load"])
        jobs = [f"verify-native-{short}"] + jobs
        ram += float(native["ram_gb"])
        cpu += int(native["cpu_threads"])
        disk += float(native["disk_gb"])
    return {"schema": SCHEMA, "commit": commit, "short": short, "stage": stage, "pid": os.getpid() if pid is None else pid,
            "created": now if created is None else created, "expires": now + EXPIRY_SECONDS, "jobs": jobs, "gpus": share,
            "ram_gb": ram, "cpu_threads": cpu, "disk_gb": disk}
