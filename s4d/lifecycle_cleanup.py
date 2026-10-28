"""Lifecycle cleanup of completed RL replay storage (user directive 2026-10-28; supersedes "completed replays retained;
no automatic runtime-computed deletion").

A completed RL run's replay is deleted on either host once (1) its full training budget has finished, (2) its final
inference policy is saved and verified loadable, (3) no writer remains and (4) no continuation depends on the replay.
Evaluation still pending on the saved policies does not delay it (scripts/eval_rl.py reads only config.yaml,
final.json and snapshots/step_*.pt). Final policies, evaluation snapshots, encoder exports, configurations, provenance,
metrics and logs are kept: only allowlisted regular files directly inside <run>/replay/ are deleted. Replay of
running, paused, failed or resumable work is never eligible.

This module holds the controller's pure decisions (queue + registry + admission pauses) and the main server's writer
and loadability checks; s4d/replay_cleanup_host.py re-checks every on-host condition right before deleting, on both
hosts; scripts/cleanup_replay.py is the entry point (not scheduled here: the user schedules it).

Eligibility of a queue row (any trainer: a row whose ``--run-dir`` is ``runs/<row id>``, other than the read-only
evaluators):
- it executed on this host (local: registry host local, or migrated from local) and its registry record is complete:
  local ``done``; remote ``done``, or compute-complete ``sync_pending`` whose final attempt exited 0 and was verified in
  place (``remote_verified``; the replay is never part of the local backup, so the backup does not gate it). Running,
  launching, crashed / paused (an intentional checkpoint stop exits non-zero), failed, pending (reset) and unknown
  records are refused;
- no active admission pause covers it (runs/setup/admission-pauses.json, runs/setup/tools/admission_pause.py semantics;
  an unreadable pause file fails the whole pass closed);
- no other queue row continues from it: a reference (in any field, e.g. ``args`` or ``continues``) to its checkpoints/
  or replay/, or to the run directory itself from anything but a read-only evaluator;
- its configured budget is known: the row's ``train.num_train_steps=`` override, else the run's config.yaml (for a
  remote run, the local backup copy, whose checksum the remote check compares); both must agree when both exist.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from s4d import replay_cleanup_host as host_check

REMOTE_ROOT = "/home/compu/kaist/sunho"
DONE = "done"
# Regular files directly inside replay/ that may be deleted: DrM's disk memmaps of end-to-end (CNN) encoders
# (atom, proprio, action, reward, discount, state_id, transition_id, episode_id, episode_step, meta .memmap) and the RAM
# replay snapshot of frozen encoders (ram_snapshot.npz and its temporary ram_snapshot.tmp.npz). A trainer whose replay
# uses other names (e.g. *.npz chunk files of a replay_dir) registers its own allowlist in TRAINERS.
DEFAULT_REPLAY_PATTERNS = ("*.memmap", "ram_snapshot*.npz")


@dataclass(frozen=True)
class Trainer:
    budget_key: str = "train.num_train_steps"
    replay_patterns: tuple[str, ...] = DEFAULT_REPLAY_PATTERNS
    policy_keys: tuple[str, ...] = ()  # modules the final snapshot's "policy" must hold (state dicts)


# Per-script settings (by file name); unlisted trainers use the defaults. New downstream trainers (e.g. train_sac.py,
# train_ppo.py, train_dreamer.py) add an entry when their budget key, replay names or policy modules differ.
TRAINERS = {"train_rl.py": Trainer(policy_keys=("encoder", "actor")),
            "train_sac.py": Trainer(policy_keys=("actor",))}  # SNeRL-code SAC snapshots hold the actor (2026-10-28)
DEFAULT_TRAINER = Trainer()
# Companion jobs that only read a run's saved policies; their rows and processes never block replay cleanup.
READ_ONLY_CONSUMERS = frozenset({"eval_rl.py"})
# Run-relative top-level entries a continuation would resume from
TRAINING_STATE = frozenset({"checkpoints", "replay"})


@dataclass
class Decision:
    run_id: str
    host: str
    eligible: bool
    reason: str = ""
    spec: dict | None = None  # host-side request (s4d.replay_cleanup_host.validate_spec)
    evidence: dict = field(default_factory=dict)
    session: int | None = None  # local: the recorded process-session leader of the job


def option(args: list, flag: str) -> str | None:
    args = list(map(str, args))
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def override(args: list, key: str) -> str | None:
    values = [str(value).split("=", 1)[1] for value in args if str(value).startswith(key + "=")]
    return values[-1] if values else None


def load_pauses(path: Path) -> list[dict]:
    """Active admission pauses. A missing file has none; an unreadable or malformed one raises (fail closed)."""
    if not path.exists():
        return []
    spec = json.loads(path.read_text())
    pauses = spec["pauses"]
    if not isinstance(pauses, list) or not all(isinstance(pause, dict) for pause in pauses):
        raise ValueError("Malformed admission pause file")
    return [pause for pause in pauses if pause.get("active") is True]


def paused(job_id: str, active: list[dict]) -> bool:
    """runs/setup/tools/admission_pause.py ``paused``: by id or id prefix, for the job and its trainer id."""
    base = job_id.removesuffix("-eval")
    return any(i in pause.get("ids", []) or any(i.startswith(prefix) for prefix in pause.get("prefixes", []))
               for pause in active for i in (job_id, base))


def references(text: str, run_id: str) -> list[str]:
    """Run-relative suffixes ("" for the run directory itself, else "/<path>") of every path reference to
    ``runs/<run_id>`` in ``text`` (relative or absolute, bare or as ``key=value``); ``runs/<run_id>-eval`` and similar
    longer names are other runs."""
    pattern = (r"(?:^|(?<=[\s=:,/'\"]))runs/" + re.escape(run_id) + r"(?P<rest>/[^\s,;:'\"]*)?(?![A-Za-z0-9_.-])")
    return [match.group("rest") or "" for match in re.finditer(pattern, text)]


def _strings(value) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _blocking(suffix: str, consumer: bool) -> bool:
    parts = [part for part in suffix.split("/") if part]
    if ".." in parts or "." in parts:  # unresolved traversal or aliases (snapshots/../replay) are never harmless
        return True
    top = parts[0] if parts else ""
    return top in TRAINING_STATE or (top == "" and not consumer)


def continuations(run_id: str, rows: list[dict]) -> list[str]:
    """Other queue rows whose fields reference this run's training state (checkpoints/, replay/) or its run directory
    (except read-only evaluators reading its policies): a continuation, resume or second writer."""
    found = []
    for row in rows:
        if row.get("id") == run_id:
            continue
        consumer = Path(str(row.get("script", ""))).name in READ_ONLY_CONSUMERS
        values = [value for key, value in row.items() if key != "id"]
        needle = f"runs/{run_id}"
        for text in _strings(values):
            if needle not in text:  # cheap prefilter: every reference (relative or absolute) contains it
                continue
            if any(_blocking(suffix, consumer) for suffix in references(text, run_id)):
                found.append(f"{row.get('id')}: {text[:160]}")
                break
    return found


def _nested(config: dict, dotted: str):
    value = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def budget(row: dict, trainer: Trainer, run_dir: Path) -> tuple[int | None, str | None, str]:
    """(configured training steps, sha256 of the config.yaml that defined them or None, source or refusal reason)."""
    import yaml

    raw = override(row.get("args", []), trainer.budget_key)
    try:
        queued = int(raw) if raw is not None else None
    except ValueError:
        return None, None, "budget override is not an integer"
    configured, digest, path = None, None, run_dir / "config.yaml"
    if path.is_file() and not path.is_symlink():
        data = path.read_bytes()
        value = _nested(yaml.safe_load(data) or {}, trainer.budget_key)
        configured = value if type(value) is int else None
        digest = hashlib.sha256(data).hexdigest()
    if queued is not None:
        if configured is not None and configured != queued:
            return None, None, "config.yaml budget differs from the queued override"
        return (queued, None, "queue override") if queued > 0 else (None, None, "budget override is not positive")
    if configured is None or configured <= 0:
        return None, None, "configured budget unknown (no override; no readable config.yaml here)"
    return configured, digest, "config.yaml"


def trainer_of(row: dict) -> Trainer | None:
    """The row's trainer settings, or None for rows that are not run-directory trainers."""
    name = Path(str(row.get("script", ""))).name
    if name in READ_ONLY_CONSUMERS or option(row.get("args", []), "--run-dir") != f"runs/{row.get('id')}":
        return None
    return TRAINERS.get(name, DEFAULT_TRAINER)


def remote_complete(current: dict) -> bool:
    """Remote compute complete: done (exit 0), or sync_pending after an exit 0 whose final attempt was verified in
    place (2026-10-27). Verification jobs are never campaign runs."""
    if current.get("verification_purpose") is not None or current.get("campaign_eligible") is False:
        return False
    verified = current.get("remote_verified") or {}
    return current.get("code") == 0 and (current.get("status") == DONE or (
        current.get("status") == "sync_pending" and verified.get("attempt") == current.get("attempts")
        and current.get("attempt", current.get("attempts")) == current.get("attempts")))


def decide(row: dict, state: dict, rows: list[dict], pauses: list[dict], host: str, runs: Path) -> Decision | None:
    """One row's decision on ``host`` (None: not a trainer row executed there)."""
    trainer = trainer_of(row)
    if trainer is None:
        return None
    run_id, current = row["id"], state.get(row["id"]) or {}
    here = current.get("host") == host or (host == "local" and (current.get("migrated_from") or {}).get("host") == "local")
    if not here:
        return None
    status = current.get("status")
    if current.get("unknown"):
        return Decision(run_id, host, False, "registry state unknown")
    if host == "local" and (status != DONE or current.get("code") != 0):
        return Decision(run_id, host, False, f"registry status {status}")
    if host == "remote" and not remote_complete(current):
        return Decision(run_id, host, False, f"registry status {status} (not compute-complete and verified)")
    if paused(run_id, pauses):
        return Decision(run_id, host, False, "admission pause active")
    found = continuations(run_id, rows)
    if found:
        return Decision(run_id, host, False, "continuation references the run: " + "; ".join(found)[:300])
    steps, digest, source = budget(row, trainer, runs / run_id)
    if steps is None:
        return Decision(run_id, host, False, source)
    spec = {"run_id": run_id, "steps": steps, "patterns": list(trainer.replay_patterns),
            "policy_keys": list(trainer.policy_keys)}
    evidence = {"registry": {"status": status, "attempt": current.get("attempt"), "attempts": current.get("attempts"),
                             "code": current.get("code")},
                "budget": {"steps": steps, "source": source}, "admission_pause": "none", "continuations": "none"}
    if host == "remote":
        if digest is not None:
            spec["config_sha256"] = digest
        verified = current.get("remote_verified") or {}
        files = (verified.get("files") if verified.get("attempt") == current.get("attempts")
                 else (current.get("sync") or {}).get("files")) or {}
        if isinstance(files.get("final.json"), str):
            spec["final_sha256"] = files["final.json"]
        evidence["registry"]["remote_verified_attempt"] = verified.get("attempt")
    host_check.validate_spec(spec)
    session = current.get("pid") if host == "local" and type(current.get("pid")) is int else None
    return Decision(run_id, host, True, "", spec, evidence, session)


def plan(rows: list[dict], state: dict, pauses: list[dict], host: str, runs: Path) -> list[Decision]:
    if host not in ("local", "remote"):
        raise ValueError("Unknown host")
    decisions = [decide(row, state, rows, pauses, host, runs) for row in rows]
    return [decision for decision in decisions if decision is not None]


# ---- main-server checks ------------------------------------------------------------------------------------------------
def local_writers(run_dir: Path, run_id: str, session: int | None = None, proc: Path = Path("/proc")) -> list[str]:
    """Live processes that may still write a local run's replay: any process of the job's recorded session; any process
    holding a file under its replay/ open or mapped (memmaps stay mapped after their descriptor closes); and any process
    whose command line references the run directory, except read-only evaluators (eval_rl.py) not referencing its
    checkpoints/ or replay/. Run right before deletion; this process itself is skipped."""
    replay = str(Path(os.path.realpath(run_dir)) / "replay")
    found = []
    for name in os.listdir(proc):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        base = proc / name
        try:
            argv = [part.decode(errors="replace") for part in (base / "cmdline").read_bytes().split(b"\0") if part]
            fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue  # exited meanwhile (or a kernel thread)
        if fields[0] in {"Z", "X"}:
            continue
        label = f"pid {name} ({' '.join(argv)[:120]})"
        if session is not None and fields[3] == str(session):
            found.append(f"{label} is in the job's session {session}")
            continue
        targets = []
        try:
            for descriptor in os.listdir(base / "fd"):
                try:
                    targets.append(os.readlink(base / "fd" / descriptor))
                except OSError:
                    continue
            targets += [line.split(None, 5)[5] for line in (base / "maps").read_text().splitlines()
                        if len(line.split(None, 5)) == 6]
        except FileNotFoundError:
            continue  # exited meanwhile
        except OSError as exc:  # fail closed: a process of ours that cannot be inspected may be a writer
            try:
                owner = base.stat().st_uid
            except FileNotFoundError:
                continue
            if isinstance(exc, PermissionError) and owner != os.geteuid():
                continue  # another user's process: the replay files are ours (mode 0644), it cannot write them
            found.append(f"{label} cannot be inspected ({type(exc).__name__})")
            continue
        if any(target == replay or target.startswith(replay + "/") for target in targets):
            found.append(f"{label} holds a replay file open or mapped")
            continue
        consumer = any(Path(arg).name in READ_ONLY_CONSUMERS for arg in argv)
        suffixes = [suffix for arg in argv for suffix in references(arg, run_id)]
        absolute = str(Path(os.path.realpath(run_dir)))
        suffixes += ["/" + arg[len(absolute) + 1:] if arg.startswith(absolute + "/") else "" for arg in argv
                     if arg == absolute or arg.startswith(absolute + "/")]
        if any(_blocking(suffix, consumer) for suffix in suffixes):
            found.append(f"{label} references the run directory")
    return found


def torch_policy_check(path: str, steps: int, policy_keys: list[str]) -> dict:
    """The final snapshot loads as scripts/eval_rl.py loads it (``torch.load(weights_only=True)``, here on the CPU), with
    step = budget and every required policy module a state dict; at least one tensor."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = payload.get("policy") if isinstance(payload, dict) else None
    if not isinstance(policy, dict) or int(payload.get("step", -1)) != steps:
        raise ValueError("snapshot step or policy differs")
    if any(not isinstance(policy.get(key), dict) for key in policy_keys):
        raise ValueError("snapshot policy module missing")
    tensors = sum(isinstance(value, torch.Tensor) for module in policy.values() if isinstance(module, dict)
                  for value in module.values())
    if not tensors:
        raise ValueError("snapshot policy holds no tensors")
    return {"torch_weights_only_load": True, "torch_tensors": tensors}


# ---- remote ------------------------------------------------------------------------------------------------------------
def remote_command(root: str, specs: list[dict], dry_run: bool) -> str:
    """One SSH command for a batch: python3 on the remote host runs s4d/remote_artifact_check.py and
    s4d/replay_cleanup_host.py (standard library only) in one namespace and prints one JSON report."""
    if root != REMOTE_ROOT:
        raise ValueError("Remote cleanup root is not authorized")
    if not specs or len({spec["run_id"] for spec in specs}) != len(specs):
        raise ValueError("A batch names each run once")
    for spec in specs:
        host_check.validate_spec(spec)
    here = Path(__file__).resolve().parent
    loader = "import base64,sys;g={'__name__':'s4d_replay_cleanup'};"
    for name in ("remote_artifact_check", "replay_cleanup_host"):
        source = base64.b64encode((here / f"{name}.py").read_bytes()).decode()
        loader += f"exec(compile(base64.b64decode('{source}'),'{name}','exec'),g);"
    loader += "sys.exit(g['cleanup_main'](sys.argv))"
    payload = base64.b64encode(json.dumps({"dry_run": dry_run is not False, "runs": specs}, sort_keys=True).encode())
    runs = shlex.quote(f"{root}/runs")
    return (f"set -eu; test -d {runs}; test ! -L {runs}; exec python3 -I -c {shlex.quote(loader)} {runs} "
            + shlex.quote(payload.decode()))


def parse_remote(output: str, specs: list[dict], dry_run: bool) -> list[dict]:
    """The batch's reports, checked against the request (one per run, known statuses, allowlisted names only)."""
    payload = json.loads(output)
    payload = payload if isinstance(payload, dict) else {}
    reports, wanted = payload.get("reports"), {spec["run_id"]: spec for spec in specs}
    if (payload.get("root") != f"{REMOTE_ROOT}/runs" or payload.get("dry_run") is not (dry_run is not False)
            or not isinstance(reports, list) or not all(isinstance(r, dict) for r in reports)
            or [r.get("run_id") for r in reports] != list(wanted)):
        raise ValueError("Remote replay cleanup returned an unexpected report")
    for report in reports:
        patterns = wanted[report["run_id"]]["patterns"]
        names = [item.get("name") for item in report.get("files", [])]
        if (report.get("status") not in {"noop", "would_delete", "deleted", "refused", "busy", "error"}
                or not all(isinstance(n, str) and "/" not in n and host_check.any_match(n, patterns) for n in names)):
            raise ValueError("Remote replay cleanup report is outside its request")
    return reports


# ---- ledger ------------------------------------------------------------------------------------------------------------
def ledger_entries(report: dict, host: str, decision: Decision, recorded: set[str], now: float) -> list[dict]:
    """One ledger line per completed deletion receipt of this run not yet recorded (a deletion of this pass, or of an
    earlier execution whose response was lost)."""
    receipts = report.get("receipts") or []
    intents = {r["receipt_id"]: r for r in receipts if r.get("state") == "intent"}
    entries = []
    for done in receipts:
        identity = done.get("receipt_id")
        if done.get("state") != "done" or identity in recorded:
            continue
        files = [item for item in done.get("files", []) if isinstance(item, dict)]
        removed = [item for item in files if item.get("removed") is True]
        # an interrupted execution's files that are gone now: deleted by it (unconfirmed), recorded separately
        absent = [item for item in files if item.get("removed") is None and item.get("absent_now") is True]
        entries.append({
            "time": now, "host": host, "run_id": decision.run_id, "receipt_id": identity,
            "deleted_at": done.get("time"), "interrupted": bool(done.get("interrupted")),
            "files": files, "removed_files": len(removed), "absent_after_interruption": len(absent),
            "freed_bytes": sum(int(item.get("allocated_bytes", 0)) for item in removed + absent),
            "evidence": {"host": (intents.get(identity) or {}).get("evidence"), "controller": decision.evidence},
            "tool": "scripts/cleanup_replay.py"})
        recorded.add(identity)
    return entries


def recorded_receipts(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    found = set()
    for line in path.read_text().splitlines():
        try:
            found.add(json.loads(line)["receipt_id"])
        except (ValueError, KeyError, TypeError):
            continue
    return found


def append_ledger(path: Path, entries: list[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
