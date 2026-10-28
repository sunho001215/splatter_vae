"""Host-side replay deletion of completed RL runs (user directive 2026-10-28), standard library only.

The same code runs on both hosts: in-process on the main server (``scripts/cleanup_replay.py --host local``) and on the
remote host (python3 there has no PyTorch outside the job containers), where it is sent in one SSH command per batch
together with s4d/remote_artifact_check.py, whose inert unpickler it uses. The controller (s4d/lifecycle_cleanup.py)
decides eligibility from the queue, the registry and the admission pauses; this module re-checks every on-host
condition of one run immediately before deleting:

- the run directory and its replay/ are real directories (no symlink, realpath identity under the runs root);
- final.json is a small regular file reporting ``completed: true`` at the configured budget (and, when the controller
  passes them, final.json has the checksum verified in the registry and config.yaml the checksum of the local copy
  that defined the budget);
- the final inference policy snapshots/step_<steps:07d>.pt is an intact zip archive (``testzip`` CRC of every member)
  whose pickle, read with inert stand-ins (no global is ever resolved, so nothing executes), references only the
  globals a weights-only load accepts, holds ``step`` = budget and the trainer's policy modules with tensor storage
  consistent with the archive records; on the main server the controller adds a real ``torch.load(weights_only=True)``
  on the CPU (as scripts/eval_rl.py loads snapshots);
- no writer remains (the caller's check, run last: on the remote no container labelled with the run id in a Docker
  state other than exited / dead; locally no live process of the job's session, none referencing the run directory
  except read-only evaluators, none holding a replay file open or mapped).

Only regular files directly inside replay/ whose names match the trainer's allowlist are deleted: never a directory,
never through a symlink, each file's realpath must be its own path in replay/ and its inode the one inspected. An
execution holds flock on the replay directory, appends an intent record and, after the deletions, a completion record
to replay/.replay-cleanup.jsonl (not an allowlisted name, so it stays) and reports every record, so a retry after a
lost SSH response still yields a complete ledger; an intent left without completion by an interrupted execution is
closed by the next one. A run without remaining allowlisted files is a no-op.

    python3 -I -c <loader> <runs root> <base64 JSON {"dry_run": bool, "runs": [spec, ...]}>   (remote, one JSON line)
"""
import base64
import fcntl
import fnmatch
import hashlib
import json
import os
import pickle
import re
import stat
import subprocess
import sys
import time
import uuid
import zipfile

if "inspect_archive" not in globals():  # on the remote host the loader runs remote_artifact_check.py in this namespace
    from s4d.remote_artifact_check import inspect_archive

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
PATTERN = re.compile(r"[A-Za-z0-9_.*?-]{1,64}\Z")  # a file-name glob: no path separator, no character class
MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RECEIPTS = ".replay-cleanup.jsonl"
SMALL_LIMIT = 1 << 20
# the globals a weights-only load accepts (scripts/jobs.py WEIGHTS_ONLY_GLOBALS)
WEIGHTS_ONLY = re.compile(r"(?:collections\.OrderedDict|torch\._utils\._rebuild_tensor_v2|torch\.[A-Za-z0-9]+Storage)\Z")
# Docker states in which a container may still write the bind-mounted run directory
LIVE_STATES = ("created", "restarting", "running", "paused", "removing")


class Refused(Exception):
    pass


def validate_spec(spec):
    """One run's request: {"run_id", "steps", "patterns", "policy_keys"[, "config_sha256"][, "final_sha256"]}."""
    if (not isinstance(spec, dict) or not isinstance(spec.get("run_id"), str) or not RUN_ID.match(spec["run_id"])
            or type(spec.get("steps")) is not int or spec["steps"] <= 0
            or not isinstance(spec.get("patterns"), list) or not spec["patterns"]
            or any(not isinstance(p, str) or not PATTERN.match(p) or p.strip("*?") == "" for p in spec["patterns"])
            or not isinstance(spec.get("policy_keys"), list)
            or any(not isinstance(k, str) or not MODULE.match(k) for k in spec["policy_keys"])
            or any(key in spec and not (isinstance(spec[key], str) and SHA256.match(spec[key]))
                   for key in ("config_sha256", "final_sha256"))
            or set(spec) - {"run_id", "steps", "patterns", "policy_keys", "config_sha256", "final_sha256"}):
        raise ValueError("Invalid replay cleanup request")
    return spec


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _regular(run_dir, name):
    """The run file ``name`` as a regular, non-empty file reached without any symlink."""
    path = os.path.join(run_dir, name)
    parent = os.path.dirname(path)
    if (os.path.realpath(path) != path or os.path.islink(path) or os.path.islink(parent)
            or not os.path.isfile(path) or os.path.getsize(path) == 0):
        raise Refused(name + " is missing, empty, not a regular file or behind a symlink")
    return path


def _small(run_dir, name):
    path = _regular(run_dir, name)
    if os.path.getsize(path) > SMALL_LIMIT:
        raise Refused(name + " is unexpectedly large")
    with open(path, "rb") as stream:
        return stream.read()


def _final(run_dir, spec):
    data = _small(run_dir, "final.json")
    try:
        record = json.loads(data.decode())
    except ValueError:
        raise Refused("final.json is not JSON") from None
    if (not isinstance(record, dict) or record.get("completed") is not True or type(record.get("steps")) is not int
            or record["steps"] != spec["steps"]):
        raise Refused("final.json does not report the configured budget completed")
    digest = hashlib.sha256(data).hexdigest()
    if spec.get("final_sha256") and digest != spec["final_sha256"]:
        raise Refused("final.json differs from the checksum verified in the registry")
    return {"sha256": digest, "steps": record["steps"], "completed": True}


def _snapshot(run_dir, spec, verify_policy):
    name = f"snapshots/step_{spec['steps']:07d}.pt"
    path = _regular(run_dir, name)
    try:
        archive = inspect_archive(path)
    except (zipfile.BadZipFile, pickle.UnpicklingError, EOFError, ValueError, KeyError, TypeError, AttributeError,
            IndexError, OSError) as exc:
        raise Refused("final policy archive is unreadable (" + type(exc).__name__ + ")") from None
    found = archive.get("globals") or []
    policy = archive.get("policy")
    if archive.get("zip_ok") is not True:
        raise Refused("final policy archive failed its integrity check")
    if any(not WEIGHTS_ONLY.match(value) for value in found):
        raise Refused("final policy references globals a weights-only load rejects")
    if archive.get("step") != spec["steps"]:
        raise Refused("final policy step differs from the budget")
    if (not isinstance(policy, dict) or not policy or archive.get("policy_storage_problems")
            or any(not isinstance(policy.get(key), dict) for key in spec["policy_keys"])):
        raise Refused("final policy modules are missing or their tensor storage is inconsistent")
    tensors = sum(len(value) for value in policy.values() if isinstance(value, dict))
    if tensors == 0:
        raise Refused("final policy holds no tensors")
    evidence = {"name": name, "bytes": os.path.getsize(path), "sha256": _digest(path), "step": archive["step"],
                "zip_ok": True, "globals": sorted(found), "modules": sorted(policy), "tensors": tensors,
                "check": "zip testzip + inert unpickle (no global resolved)"}
    if verify_policy is not None:
        try:
            evidence.update(verify_policy(path, spec["steps"], spec["policy_keys"]))
        except Exception as exc:  # any load failure means "not verified loadable"
            raise Refused("final policy does not load (" + type(exc).__name__ + ": " + str(exc)[:200] + ")") from None
    return evidence


def any_match(name, patterns):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _scan(replay, patterns):
    """(allowlisted regular files directly in replay/ with their identity, other entry names); no recursion, no
    symlink is followed or deleted."""
    found, kept = [], []
    for entry in os.scandir(replay):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False) or not any_match(entry.name, patterns):
            if entry.name != RECEIPTS:
                kept.append(entry.name)
            continue
        info = entry.stat(follow_symlinks=False)
        path = os.path.join(replay, entry.name)
        if os.path.realpath(path) != path or os.path.dirname(os.path.realpath(path)) != replay:
            kept.append(entry.name)
            continue
        found.append({"name": entry.name, "bytes": info.st_size, "allocated_bytes": info.st_blocks * 512,
                      "identity": [info.st_dev, info.st_ino]})
    return sorted(found, key=lambda item: item["name"]), sorted(kept)


def _public(item):
    return {key: value for key, value in item.items() if key != "identity"}


def _receipts(replay):
    path = os.path.join(replay, RECEIPTS)
    if os.path.islink(path) or not os.path.isfile(path) or os.path.getsize(path) > SMALL_LIMIT:
        return []
    records = []
    with open(path) as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("receipt_id"), str):
                records.append(record)
    return records


def _append(replay, record):
    descriptor = os.open(os.path.join(replay, RECEIPTS), os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unclosed(records):
    done = {record["receipt_id"] for record in records if record.get("state") == "done"}
    return [record for record in records if record.get("state") == "intent" and record["receipt_id"] not in done]


def _close_interrupted(replay, run_id):
    """Complete the intent records of executions that ended before their completion record (the caller holds the
    replay lock, so none of them still runs): each listed file is recorded as absent or still present now."""
    for record in _unclosed(_receipts(replay)):
        files = [{**_public(item), "removed": None, "absent_now": not os.path.lexists(os.path.join(replay, item["name"]))}
                 for item in record.get("files", []) if isinstance(item, dict) and isinstance(item.get("name"), str)
                 and "/" not in item["name"]]
        _append(replay, {"receipt_id": record["receipt_id"], "run_id": run_id, "state": "done", "time": time.time(),
                         "interrupted": True, "files": files})


def _unlink(replay, item):
    path = os.path.join(replay, item["name"])
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return {**_public(item), "removed": False, "absent_now": True}
    if (not stat.S_ISREG(info.st_mode) or [info.st_dev, info.st_ino] != item["identity"]
            or os.path.realpath(path) != path):
        return {**_public(item), "removed": False, "absent_now": False, "note": "changed since inspection; kept"}
    try:
        os.unlink(path)
    except FileNotFoundError:
        return {**_public(item), "removed": False, "absent_now": True}
    return {**_public(item), "removed": True}


def clean_run(runs_root, spec, dry_run=True, writers=None, verify_policy=None):
    """Re-check and (unless ``dry_run``) delete one run's allowlisted replay files. ``writers(run_id, run_dir)`` returns
    the reasons a writer may remain (empty: none); ``verify_policy(path, steps, policy_keys)`` adds a real load.
    Returns {"run_id", "status": noop | would_delete | deleted | refused | busy | error, "reason", "files", "kept",
    "evidence", "receipts"}."""
    report = {"run_id": spec.get("run_id") if isinstance(spec, dict) else None, "status": "refused", "reason": "",
              "files": [], "kept": [], "evidence": {}, "receipts": []}
    descriptor = None
    try:
        validate_spec(spec)
        run_id = spec["run_id"]
        root = os.path.realpath(runs_root)
        run_dir = os.path.join(root, run_id)
        replay = os.path.join(run_dir, "replay")
        if not os.path.lexists(run_dir):
            report.update(status="noop", reason="no run directory")
            return report
        if os.path.islink(run_dir) or os.path.realpath(run_dir) != run_dir or not os.path.isdir(run_dir):
            raise Refused("run directory is a symlink or not a directory")
        if not os.path.lexists(replay):
            report.update(status="noop", reason="no replay directory")
            return report
        if os.path.islink(replay) or os.path.realpath(replay) != replay or not os.path.isdir(replay):
            raise Refused("replay is a symlink or not a directory")
        candidates, report["kept"] = _scan(replay, spec["patterns"])
        report["receipts"] = _receipts(replay)
        if not candidates and (dry_run or not _unclosed(report["receipts"])):
            report.update(status="noop", reason="no allowlisted replay files remain")
            return report
        evidence = None
        if candidates:
            evidence = {"final": _final(run_dir, spec)}
            if spec.get("config_sha256"):
                if _digest(_regular(run_dir, "config.yaml")) != spec["config_sha256"]:
                    raise Refused("config.yaml differs from the copy that defined the budget")
                evidence["config_sha256"] = spec["config_sha256"]
            evidence["snapshot"] = _snapshot(run_dir, spec, verify_policy)
            report["evidence"] = evidence
        if not dry_run:
            descriptor = os.open(replay, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report.update(status="busy", reason="another cleanup execution holds this replay directory")
                return report
            _close_interrupted(replay, run_id)  # also when nothing remains: the ledger learns what it deleted
            candidates, report["kept"] = _scan(replay, spec["patterns"])  # again, under the lock
            if not candidates:  # an earlier execution deleted them meanwhile
                report.update(status="noop", reason="no allowlisted replay files remain", receipts=_receipts(replay))
                return report
            if evidence is None:
                raise Refused("replay files appeared during the inspection")
        found = writers(run_id, run_dir) if writers is not None else []  # last check before deleting
        if found:
            raise Refused("a writer may remain: " + "; ".join(found)[:500])
        evidence["writers"] = "none"
        if dry_run:
            report.update(status="would_delete", files=[_public(item) for item in candidates])
            return report
        receipt = uuid.uuid4().hex
        _append(replay, {"receipt_id": receipt, "run_id": run_id, "state": "intent", "time": time.time(),
                         "files": [_public(item) for item in candidates], "evidence": evidence})
        files = [_unlink(replay, item) for item in candidates]
        _append(replay, {"receipt_id": receipt, "run_id": run_id, "state": "done", "time": time.time(), "files": files})
        report.update(status="deleted", receipt_id=receipt, files=files, receipts=_receipts(replay))
    except Refused as exc:
        report.update(status="refused", reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report.update(status="error", reason=type(exc).__name__ + ": " + str(exc)[:300])
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return report


def docker_writers(run_id, run_dir):
    """Remote writer check: containers labelled with this run id in a state in which they may still write the
    bind-mounted run directory (created, restarting, running, paused, removing). Docker failing is a refusal."""
    command = ["docker", "ps", "-a", "-q", "--no-trunc", "--filter", "label=s4d.run_id=" + run_id]
    for state in LIVE_STATES:
        command += ["--filter", "status=" + state]
    # PIPE / universal_newlines: the remote host's python3 may predate 3.7 (capture_output, text)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  # noqa: UP022
                            universal_newlines=True, timeout=120)  # noqa: UP021
    if result.returncode:
        return [f"Docker could not be queried (exit {result.returncode})"]
    return [f"container {line[:12]} labelled s4d.run_id={run_id} is not exited" for line in result.stdout.split()]


def cleanup_main(argv):
    request = json.loads(base64.b64decode(argv[2]).decode())
    dry_run = request.get("dry_run") is not False  # anything but an explicit false only inspects
    reports = [clean_run(argv[1], spec, dry_run=dry_run, writers=docker_writers) for spec in request["runs"]]
    sys.stdout.write(json.dumps({"root": os.path.realpath(argv[1]), "dry_run": dry_run, "reports": reports},
                                sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cleanup_main(sys.argv))
