"""Explicit migration of a started, checkpoint-paused run to the host its role belongs to (user directive 2026-11-06
item 2: remote Meta-World runs continue on local from their full state, same code / configuration / seeds, with
explicit ownership records and no duplicates). Generalises the 2026-10-24 local -> remote migration
(runs/setup/tools/migrate_cnn_to_remote_20261024.py) to remote -> local, reusing the scheduler's ``migrated`` registry
event unchanged: the run keeps its job id, arguments, seed and fingerprint and resumes from the recorded intentional
checkpoint pause of its latest attempt.

A migration unit is a run and its evaluator companion (``<id>-eval``, launched or not). Each phase runs under the local
admission lock and the scheduler's registry lock and updates the unit's ownership records
(runs/setup/ownership/<job id>.json, ops/jobs/host_roles.py):

  begin    the run is not active and its latest attempt ended in a recorded intentional checkpoint pause (a launched
           evaluator too), the destination is the role host of its class, no other migration owns it -> state
           ``migrating`` (owner: the source host). From here on neither host admits any member of the unit.
  verify   after the transfer (resumable, checksum-verified, under the transfer limits; not done here): every file of
           the source run directories' checksum listing exists locally with the same sha256; the paused checkpoint
           matches the pause record; the checkpoint's components are present and inventoried with their digests
           (DrM: networks, optimizers, dormant / awaken schedule state, replay counters, python / numpy / torch / CUDA /
           camera RNG; pretraining: encoder, decoder, optimizer, LR scheduler, RNG) together with the replay files;
           consumed encoder exports equal their producer's verified checksum -> transferred-state manifest recorded.
  switch   the execution scope (the scripts' static import closure inside the repository except three listed
           operational files, configs/metaworld|rl|baselines, uv.lock, pyproject.toml, third-party pins) of the source
           release commit equals the destination
           working tree (``sources_identical`` of the ``migrated`` event); the source host's paused containers are
           already removed by the scheduler cleanup -> queue rows change host fields only (recorded in
           pool-placement.json), one ``migrated`` event per launched member, state ``switched`` (owner: destination).
           The scheduler then admits the run on the destination only while those scope files are unchanged.
  confirm  the destination attempt resumed at the paused step (DrM events.jsonl ``resumed``; pretraining log.txt) and
           continued past it -> state ``verified``: the source copies may be retired by the cleanup.
  abort    before ``switch`` only -> state ``aborted`` (owner: the source host; nothing else changed).

    python -I ops/jobs/migration.py begin   --to local --reason TEXT <run id>
    python -I ops/jobs/migration.py verify  --listing LISTING.json <run id>   ({member id: {run-relative path: sha256}})
    python -I ops/jobs/migration.py switch  <run id>
    python -I ops/jobs/migration.py confirm <run id>
    python -I ops/jobs/migration.py abort   --reason TEXT <run id>
    python -I ops/jobs/migration.py status  [<run id>]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.jobs import host_roles  # noqa: E402

CONFIG_SCOPE = ("configs/metaworld/", "configs/rl/", "configs/baselines/")
ENVIRONMENT_SCOPE = ("uv.lock", "pyproject.toml")
# Operational files a run imports that may differ between the source release and the destination without making the
# continuation non-exact, each recorded with the comparison: the scheduler and the role module (admission bookkeeping;
# a job imports the scheduler only for the tested-source check of scripts/_bootstrap.py) and the GPU guard (approved
# host tables and output-path authorization). Every other imported operational file (run identity / provenance,
# evaluator registry access, remote access, ...) must be identical like the research code.
OPERATIONAL_ALLOWED = frozenset({"ops/jobs/scheduler.py", "ops/jobs/host_roles.py", "ops/resources/gpu_guard.py"})
IMPORT_ROOTS = ("s4d", "ops", "scripts")
# Checkpoint components a resume restores (scripts/train_rl.py, s4d/train/checkpoint.py), by script.
COMPONENTS = {
    "train_rl.py": {"top": ("step", "episode", "agent", "replay_meta", "rolling_success", "rolling_return",
                            "train_reset_count", "camera_rng", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"),
                    "agent": ("encoder", "actor", "critic", "critic_target", "value_predictor", "actor_opt",
                              "critic_opt", "predictor_opt", "dormant_ratio", "awaken_step"),
                    "replay": "replay/"},
    "train.py": {"top": ("step", "encoder", "decoder", "optimizer", "scheduler", "config", "rng"), "agent": (),
                 "replay": None},
}
REFUSED_EXTRAS = ("final.json", "exit_code", "completion.json")  # stale completion records must not precede a resume


class MigrationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------- paths / state
class Paths:
    """Where one migration reads and writes (defaults: the live checkout; tests pass a sandbox)."""

    def __init__(self, root: Path = REPO, queue: Path | None = None, registry: Path | None = None,
                 ownership: Path | None = None, pool_manifest: Path | None = None, admission_lock: Path | None = None,
                 roles: Path | None = None):
        self.root = Path(root)
        self.queue = Path(queue or self.root / "experiments/queue.yaml")
        self.registry = Path(registry or self.root / "experiments/registry.jsonl")
        self.ownership = Path(ownership or self.root / "runs/setup/ownership")
        self.pool_manifest = Path(pool_manifest or self.root / "runs/setup/pool-placement.json")
        self.admission_lock = Path(admission_lock or self.root / "runs/setup/stage2-driver.lock")
        self.roles = Path(roles or host_roles.ROLES_PATH)


@contextlib.contextmanager
def locked(paths: Paths):
    with open(paths.admission_lock, "a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with open(paths.registry.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield


def _scheduler():
    from ops.jobs import scheduler

    return scheduler


def _load(paths: Paths) -> tuple[dict, dict, dict]:
    jobs = _scheduler()
    queue = jobs.load_queue(paths.queue)
    return queue, {j["id"]: j for j in queue["jobs"]}, jobs.read_registry(paths.registry)


def unit_of(run_id: str, by_id: dict) -> list[str]:
    if run_id not in by_id or run_id.endswith("-eval"):
        raise MigrationError(f"{run_id}: name the run (not its evaluator) of a queued row")
    return [run_id] + ([run_id + "-eval"] if run_id + "-eval" in by_id else [])


def run_dir(job: dict) -> str:
    """The run-relative output directory of a member (trainer --run-dir, evaluator --output-dir, pretraining
    --output-root/--name, else runs/<id>)."""
    args = [str(a) for a in job.get("args", [])]
    if job.get("script") == "scripts/eval_rl.py" and "--output-dir" in args:
        return args[args.index("--output-dir") + 1]
    if "--run-dir" in args:
        return args[args.index("--run-dir") + 1]
    if "--output-root" in args and "--name" in args:
        return f"{args[args.index('--output-root') + 1]}/{args[args.index('--name') + 1]}"
    return f"runs/{job['id']}"


def _paused_stop(job_id: str, current: dict) -> dict:
    jobs = _scheduler()
    if current.get("status") in jobs.ACTIVE or current.get("unknown"):
        raise MigrationError(f"{job_id}: still active or unknown ({current.get('status')})")
    stops = [s for s in current.get("intentional_stops", []) if s.get("attempt") == current.get("attempt")]
    if not stops or not isinstance(stops[-1].get("checkpoint_step"), int):
        raise MigrationError(f"{job_id}: its latest attempt has no recorded intentional checkpoint pause")
    return stops[-1]


def _records(paths: Paths, unit: list[str]) -> list[dict]:
    found = host_roles.read_ownership(paths.ownership)
    return [found.get(i) for i in unit]


def _save(paths: Paths, record: dict, state: str, note: str, **changes) -> dict:
    previous = {k: v for k, v in record.items() if k != "history"}
    new = {**record, **changes, "state": state, "time": host_roles.now(),
           "history": record.get("history", []) + [{"state": previous["state"], "time": previous["time"],
                                                     "note": note}]}
    host_roles.write_ownership(new, paths.ownership)
    return new


# ---------------------------------------------------------------------------------------------------- begin
def begin(run_id: str, to_host: str, reason: str, paths: Paths | None = None) -> list[dict]:
    paths = paths or Paths()
    roles = host_roles.load_roles(paths.roles)
    jobs = _scheduler()
    with locked(paths):
        queue, by_id, state = _load(paths)
        unit = unit_of(run_id, by_id)
        trainer = state.get(run_id)
        if not trainer or not trainer.get("attempts"):
            raise MigrationError(f"{run_id}: never launched; a role re-placement moves it (host_roles.replace_to_local)")
        from_host = trainer.get("host", "local")
        if from_host == to_host or to_host not in host_roles.SCHEDULER_HOSTS:
            raise MigrationError(f"{run_id}: invalid destination {to_host} (runs on {from_host})")
        if any(by_id[i].get("host", "local") != from_host for i in unit):
            raise MigrationError(f"{run_id}: queue rows of the unit are not all on {from_host}")
        if any(host_roles.role_host(by_id[i], roles, by_id) != to_host for i in unit):
            raise MigrationError(f"{run_id}: {to_host} is not the role host of every member's job class")
        if any(by_id[i].get("stage") == 2 for i in unit):
            raise MigrationError(f"{run_id}: Stage 2 rows are placed by the Stage 2 driver's own overlay; not supported")
        existing = [r for r in _records(paths, unit) if r and r["state"] in ("migrating", "switched")]
        if existing:
            raise MigrationError(f"{run_id}: a migration already owns it ({existing[0]['state']})")
        members = {}
        for i in unit:
            current = state.get(i)
            if current is None:  # an evaluator that never launched moves with the queue rows only
                members[i] = {"launched": False}
                continue
            if current.get("host", "local") != from_host:
                raise MigrationError(f"{i}: registry host {current.get('host')} differs from {from_host}")
            stop = _paused_stop(i, current)
            if current.get("job_sha256") != jobs.job_fingerprint(by_id[i]):
                raise MigrationError(f"{i}: the queue row differs from the run launched on {from_host} (fingerprint)")
            members[i] = {"launched": True, "attempt": current["attempt"], "stop": stop,
                          "commit": current.get("commit"), "image_digest": current.get("image_digest"),
                          "job_sha256": current.get("job_sha256"), "run_dir": run_dir(by_id[i]),
                          "container_id": (current.get("attempt_containers", {}).get(str(current["attempt"])) or {}
                                           ).get("container_id")}
        stamp = host_roles.now()
        out = []
        for i in unit:
            previous = host_roles.read_ownership(paths.ownership).get(i)
            record = {"job_id": i, "kind": "migration", "unit": unit, "state": "migrating", "owner_host": from_host,
                      "original_row": by_id[i],
                      "from_host": from_host, "to_host": to_host, "reason": reason, "time": stamp,
                      "roles_sha256": roles["sha256"], "member": members[i],
                      "checkpoint_step": members[run_id]["stop"]["checkpoint_step"],
                      "history": (previous.get("history", []) + [{k: v for k, v in previous.items() if k != "history"}]
                                  if previous else [])}
            host_roles.write_ownership(record, paths.ownership)
            out.append(record)
        return out


# ---------------------------------------------------------------------------------------------------- verify
def canonical_digest(value) -> str:
    """Order-independent digest of a checkpoint component (tensors: dtype, shape, bytes; arrays likewise)."""
    digest = hashlib.sha256()

    def walk(item):
        import numpy as np

        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        if torch is not None and isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"T{tensor.dtype}{tuple(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")
        elif isinstance(item, np.ndarray):
            digest.update(f"A{item.dtype}{item.shape}".encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                walk(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"L{len(item)}".encode())
            for element in item:
                walk(element)
        else:
            digest.update(f"S{type(item).__name__}:{item!r}".encode())

    walk(value)
    return digest.hexdigest()


def checkpoint_components(path: Path, script: str, step: int) -> dict:
    """Inventory of a paused checkpoint's resume state with per-component digests."""
    import torch

    spec = COMPONENTS.get(Path(script).name)
    if spec is None:
        raise MigrationError(f"no checkpoint component specification for {script}")
    state = torch.load(path, map_location="cpu", weights_only=False)  # our own checkpoint, checksum-verified first
    missing = [key for key in spec["top"] if key not in state]
    agent = state.get("agent", {}) if spec["agent"] else {}
    missing += [f"agent.{key}" for key in spec["agent"] if key not in agent]
    if missing:
        raise MigrationError(f"{path.name}: resume components missing: {missing}")
    if int(state["step"]) != step:
        raise MigrationError(f"{path.name}: holds step {state['step']}, the pause recorded {step}")
    out = {key: canonical_digest(state[key]) for key in spec["top"] if key != "agent"}
    out.update({f"agent.{key}": canonical_digest(agent[key]) for key in spec["agent"]})
    if "encoder_opt" in agent:  # end-to-end CNN encoders
        out["agent.encoder_opt"] = canonical_digest(agent["encoder_opt"])
    return {"step": int(state["step"]), "digests": out,
            "groups": {"optimizer": sorted(k for k in out if k.endswith(("_opt", "optimizer"))),
                       "scheduler": sorted(k for k in out if k.endswith(("dormant_ratio", "awaken_step", "scheduler"))),
                       "rng": sorted(k for k in out if "rng" in k),
                       "replay": sorted(k for k in out if k.startswith("replay"))}}


def _producer_digest(export: str, by_id: dict, state: dict) -> str | None:
    """The verified checksum of an encoder export written by a queued export row (``--out``)."""
    for job in by_id.values():
        args = [str(a) for a in job.get("args", [])]
        if job.get("script") == "scripts/export_encoder.py" and "--out" in args and args[args.index("--out") + 1] == export:
            produced = state.get(job["id"], {})
            relative = export.removeprefix(f"runs/{job['id']}/")
            verified = (produced.get("remote_verified") or {}).get("files", {}).get(relative)
            synced = (produced.get("sync") or {}).get("files", {}).get(relative)
            return verified or synced
    return None


def verify(run_id: str, listing: dict, paths: Paths | None = None,
           components: Callable[[Path, str, int], dict] = checkpoint_components) -> list[dict]:
    """Check the transferred state against the source listing ({member id: {run-relative path: sha256}}) and record
    the transferred-state manifest."""
    paths = paths or Paths()
    with locked(paths):
        queue, by_id, state = _load(paths)
        unit = unit_of(run_id, by_id)
        records = _records(paths, unit)
        if any(r is None or r["state"] != "migrating" for r in records):
            raise MigrationError(f"{run_id}: not in a migration (begin first)")
        manifest = {"files": {}, "components": None, "inputs": {}, "extras": {}}
        for i, record in zip(unit, records):
            member = record["member"]
            if not member["launched"]:
                continue
            current = state.get(i, {})
            if current.get("attempt") != member["attempt"] or _paused_stop(i, current) != member["stop"]:
                raise MigrationError(f"{i}: registry changed since begin")
            files = listing.get(i)
            if not isinstance(files, dict) or not files:
                raise MigrationError(f"{i}: no source checksum listing")
            base = paths.root / member["run_dir"]
            for relative, digest in sorted(files.items()):
                local = (base / relative).resolve()
                if base.resolve() not in local.parents or not local.is_file():
                    raise MigrationError(f"{i}: {relative} missing locally")
                if sha256(local) != digest:
                    raise MigrationError(f"{i}: {relative} differs from the source checksum")
                stat = local.stat()
                manifest["files"][f"{member['run_dir']}/{relative}"] = {"sha256": digest, "bytes": stat.st_size,
                                                                        "mtime_ns": stat.st_mtime_ns}
            present = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
            extras = sorted(present - set(files))
            if any(Path(e).name in REFUSED_EXTRAS for e in extras):
                raise MigrationError(f"{i}: stale completion records beside the transferred state: {extras}")
            manifest["extras"][i] = extras
            stop, script = member["stop"], by_id[i]["script"]
            if i == run_id:
                latest = "checkpoints/latest.pt"
                if latest not in files:
                    raise MigrationError(f"{i}: the listing lacks {latest}")
                paused = stop.get("checkpoint", f"{member['run_dir']}/{latest}").removeprefix(f"{member['run_dir']}/")
                if files.get(paused) != stop.get("checkpoint_sha256"):
                    raise MigrationError(f"{i}: {paused} differs from the pause record's checksum")
                if stop.get("latest_sha256") and files[latest] != stop["latest_sha256"]:
                    raise MigrationError(f"{i}: latest.pt differs from the pause record")
                spec = COMPONENTS.get(Path(script).name)
                if spec and spec["replay"] and not any(r.startswith(spec["replay"]) for r in files):
                    raise MigrationError(f"{i}: the listing has no replay files")
                manifest["components"] = components(base / latest, script, stop["checkpoint_step"])
                manifest["components"]["replay_files"] = {r: files[r] for r in sorted(files)
                                                          if spec and spec["replay"] and r.startswith(spec["replay"])}
            for arg in by_id[i].get("args", []):
                if str(arg).startswith("vision.export_path="):
                    export = str(arg).split("=", 1)[1]
                    expected = _producer_digest(export, by_id, state)
                    if expected is None or not (paths.root / export).is_file() or sha256(paths.root / export) != expected:
                        raise MigrationError(f"{i}: input {export} is not present with its producer's verified checksum")
                    manifest["inputs"][export] = expected
        manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        return [_save(paths, record, "migrating", "transferred state verified", manifest=manifest)
                for record in records]


def remote_listing(client, run_rel: str) -> dict[str, str]:
    """{run-relative path: sha256} of every file of a remote run directory (one bounded SSH command; read-only)."""
    from ops.jobs.remote_backend import ROOT as REMOTE_ROOT

    if not re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*", run_rel):
        raise MigrationError(f"unsafe run path {run_rel}")
    out = client.checked_ssh(f"cd {REMOTE_ROOT}/{run_rel} && find . -type f -print0 | sort -z | xargs -0 -r sha256sum",
                             wait=False, idempotent=True, timeout=3600).stdout
    found = {}
    for line in out.splitlines():
        digest, name = line.split(None, 1)
        found[name.strip().removeprefix("./")] = digest
    return found


# ---------------------------------------------------------------------------------------------------- source scope
def _module_files(name: str, importer: str) -> list[str]:
    parts = name.split(".")
    if parts[0] in IMPORT_ROOTS:
        out = ["/".join(parts[:k]) + "/__init__.py" for k in range(1, len(parts) + 1)]
        return out + ["/".join(parts) + ".py"]
    # a sibling module on sys.path (scripts/_bootstrap.py) or a repository-root module (mujoco_mig_setup.py)
    return [str(Path(importer).parent / (parts[0] + ".py")), parts[0] + ".py"]


def import_closure(scripts: list[str], read: Callable[[str], bytes | None]) -> set[str]:
    """Repository files a script statically imports (also inside functions, and s4d/ops/scripts module names given as
    strings to dynamic imports), transitively."""
    todo, seen = list(scripts), set()
    while todo:
        relative = todo.pop()
        if relative in seen:
            continue
        data = read(relative)
        if data is None:
            continue
        seen.add(relative)
        for node in ast.walk(ast.parse(data)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = Path(relative).parents[node.level - 1]
                    base = ".".join(package.parts + ((node.module,) if node.module else ()))
                else:
                    base = node.module or ""
                names = [base] + [f"{base}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(
                    r"(?:s4d|ops|scripts)(?:\.\w+)+", node.value):
                names = [node.value]
            for name in names:
                todo += [f for f in _module_files(name, relative) if f not in seen]
    return seen


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def commit_reader(root: Path, commit: str) -> tuple[Callable[[str], bytes | None], list[str]]:
    listed = set(_git(root, "ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")) - {""}

    def read(relative: str) -> bytes | None:
        return _git(root, "cat-file", "blob", f"{commit}:{relative}") if relative in listed else None

    return read, sorted(listed)


def tree_reader(root: Path) -> tuple[Callable[[str], bytes | None], list[str]]:
    def read(relative: str) -> bytes | None:
        path = root / relative
        return path.read_bytes() if path.is_file() else None

    listed = [str(p.relative_to(root)) for folder in CONFIG_SCOPE if (root / folder).is_dir()
              for p in sorted((root / folder).rglob("*")) if p.is_file()]
    return read, listed


def execution_scope(scripts: list[str], source_commit: str, root: Path = REPO) -> dict:
    """Compare the execution scope of ``scripts`` between the source release commit and the destination working
    tree: {"identical", "files": {path: sha256}, "third_party": {...}, "differences": [...]}."""
    from s4d.third_party import commit_identity, working_tree_identity

    source_read, source_listed = commit_reader(root, source_commit)
    target_read, target_listed = tree_reader(root)
    paths = import_closure(scripts, source_read) | import_closure(scripts, target_read)
    paths |= {p for p in source_listed if p.startswith(CONFIG_SCOPE)} | set(target_listed) | set(ENVIRONMENT_SCOPE)
    files, differences, operational = {}, [], []
    for relative in sorted(paths):
        before, after = source_read(relative), target_read(relative)
        if relative in OPERATIONAL_ALLOWED:  # recorded, not compared (OPERATIONAL_ALLOWED)
            operational += [relative] if before != after else []
            continue
        if before != after:
            differences.append(relative)
        if after is not None:
            files[relative] = hashlib.sha256(after).hexdigest()
    source_tp, target_tp = commit_identity(root, source_commit), working_tree_identity(root)
    differences += [key for key in sorted(set(source_tp) | set(target_tp)) if source_tp.get(key) != target_tp.get(key)]
    return {"identical": not differences, "files": files, "third_party": target_tp, "differences": differences,
            "operational_differences": operational, "scripts": scripts, "source_commit": source_commit}


def destination_identity(root: Path = REPO) -> tuple[str, str]:
    """(commit, environment identity) of the local destination: the checked-out commit and the uv.lock the local
    environment is synchronised from (the local host runs no container image)."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    return commit, f"local-venv:uv.lock-sha256:{sha256(root / 'uv.lock')}"


# ---------------------------------------------------------------------------------------------------- switch
def _validated_events(paths: Paths, events: list[dict]) -> None:
    """Replay the unit's registry history plus the new events with the scheduler's own reducer (raises on an invalid
    migration) before anything is appended to the live registry."""
    jobs = _scheduler()
    ids = {event["id"] for event in events}
    with tempfile.TemporaryDirectory() as folder:
        trial = Path(folder) / "registry.jsonl"
        with open(paths.registry) as source, open(trial, "w") as target:
            for line in source:
                if any(f'"{i}"' in line for i in ids) and json.loads(line)["id"] in ids:
                    target.write(line)
        for event in events:
            jobs.record(event, trial)
        replay = jobs.read_registry(trial)
        for event in events:
            if replay[event["id"]]["host"] != event["host"] or replay[event["id"]]["status"] != jobs.PENDING:
                raise MigrationError(f"{event['id']}: the migrated event does not move the run")


def _provenance_transition(paths: Paths, launched: list[tuple[str, dict]], from_host: str) -> dict[str, str]:
    """Preserve every source-host provenance record of each launched member's run directory (provenance.json, and the
    evaluators' <suite>_provenance/provenance.json) under provenance.<host>-a<attempt>.json in the same folder, so the
    destination attempt records its own provenance (ops/jobs/run_identity.py refuses to overwrite a campaign record of
    another host). Idempotent; returns {repository-relative original: preserved path} of the moved records."""
    renamed = {}
    for _, member in launched:
        base = paths.root / member["run_dir"]
        preserved = f"provenance.{from_host}-a{member['attempt']}.json"
        sources = {p for p in base.rglob("provenance.json") if p.is_file()} | {
            p.with_name("provenance.json") for p in base.rglob(preserved) if p.is_file()}
        for source in sorted(sources):
            target = source.with_name(preserved)
            if source.is_file() and target.is_file():
                if sha256(source) != sha256(target):
                    raise MigrationError(f"{source} and its preserved copy {target.name} differ")
                source.unlink()
            elif source.is_file():
                if json.loads(source.read_text()).get("host") != from_host:
                    continue  # not a record of the source host (nothing to preserve)
                source.rename(target)
            if target.is_file():
                renamed[str(source.relative_to(paths.root))] = str(target.relative_to(paths.root))
    return renamed


def switch(run_id: str, paths: Paths | None = None, scope_fn: Callable = execution_scope,
           identity_fn: Callable = destination_identity) -> list[dict]:
    paths = paths or Paths()
    jobs = _scheduler()
    with locked(paths):
        queue, by_id, state = _load(paths)
        unit = unit_of(run_id, by_id)
        records = _records(paths, unit)
        if any(r is None or r["state"] not in ("migrating", "switched") or not (r.get("manifest") or {}).get("sha256")
               for r in records):
            raise MigrationError(f"{run_id}: transferred state not verified (verify first)")
        if all(r["state"] == "switched" for r in records):
            return records
        to_host, from_host = records[0]["to_host"], records[0]["from_host"]
        if to_host != "local":
            raise MigrationError("this procedure migrates to local (local -> remote: the 2026-10-24 tool)")
        launched = [(i, r["member"]) for i, r in zip(unit, records) if r["member"]["launched"]]
        for i, member in launched:
            current = state.get(i, {})
            if current.get("attempt") != member["attempt"] or _paused_stop(i, current) != member["stop"]:
                raise MigrationError(f"{i}: registry changed since begin")
            if member.get("job_sha256") != jobs.job_fingerprint(by_id[i]):
                raise MigrationError(f"{i}: the queue row differs from the run launched on {from_host} (fingerprint)")
            settled = set(current.get("containers_removed", [])) | set(current.get("containers_absent", []))
            if from_host == "remote" and member.get("container_id") and member["container_id"] not in settled:
                raise MigrationError(f"{i}: its paused remote container is not removed yet (scheduler cleanup)")
        commits = {member["commit"] for _, member in launched}
        if len(commits) != 1 or not all(commits):
            raise MigrationError(f"{run_id}: the unit's source commits differ or are unknown: {commits}")
        source_commit = commits.pop()
        scripts = sorted({by_id[i]["script"] for i in unit})
        scope = scope_fn(scripts, source_commit, paths.root)
        if not scope["identical"]:
            raise MigrationError(f"{run_id}: execution scope differs from release {source_commit[:7]}: "
                                 f"{scope['differences'][:10]} (a non-exact continuation needs a decision)")
        # the verified transferred state, checked again in full right before the switch (provenance records may
        # already be preserved under their new name by an interrupted earlier switch)
        manifest = records[0]["manifest"]
        for name, item in sorted(manifest["files"].items()):
            path = paths.root / name
            if not path.is_file() and Path(name).name == "provenance.json":
                path = path.with_name(f"provenance.{from_host}-a" + next(
                    str(m["attempt"]) for _, m in launched if name.startswith(m["run_dir"] + "/")) + ".json")
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise MigrationError(f"{name} changed after its verification; verify again")
        for export, digest in manifest.get("inputs", {}).items():
            if not (paths.root / export).is_file() or sha256(paths.root / export) != digest:
                raise MigrationError(f"input {export} changed after its verification")
        commit, environment = identity_fn(paths.root)
        stamp = host_roles.now()
        # Re-runnable: a switch interrupted after its queue rewrite, some registry events or some ownership records
        # completes the remaining steps; until the trainer's record (written last) is switched it is not admitted.
        hosts = {by_id[i].get("host", "local") for i in unit}
        if hosts == {to_host}:
            effective = {i: by_id[i] for i in unit}
        elif hosts == {from_host}:
            effective = {i: host_roles.destination_row(
                by_id[i], to_host, f"migration {stamp} ({from_host} -> {to_host}; runs/setup/ownership/{i}.json; "
                "runs/setup/pool-placement.json)") for i in unit}
            if any(jobs.job_fingerprint(effective[i]) != jobs.job_fingerprint(by_id[i]) for i in unit):
                raise MigrationError(f"{run_id}: the placement would change a job fingerprint")
        else:
            raise MigrationError(f"{run_id}: the unit's queue rows are on {sorted(hosts)}")
        done = {i for i, member in launched if state.get(i, {}).get("host") == to_host
                and (state[i].get("migrated_from") or {}).get("checkpoint_step") == member["stop"]["checkpoint_step"]}
        events = [{"event": "migrated", "id": i, "from_host": from_host, "host": to_host,
                   "checkpoint_step": member["stop"]["checkpoint_step"],
                   "checkpoint_sha256": member["stop"].get("checkpoint_sha256"), "from_commit": member["commit"],
                   "from_image_digest": member.get("image_digest"), "commit": commit, "image_digest": environment,
                   "sources_identical": True,
                   "source_comparison": (f"execution scope of {scripts} ({len(scope['files'])} files + third-party "
                                         f"pins) equal between {source_commit[:7]} and the local working tree; "
                                         f"operational differences {scope.get('operational_differences', [])}"),
                   "ownership": f"runs/setup/ownership/{i}.json",
                   "manifest_sha256": records[unit.index(i)]["manifest"]["sha256"],
                   "reason": records[0]["reason"]} for i, member in launched if i not in done]
        if events:
            _validated_events(paths, events)
        renamed = _provenance_transition(paths, launched, from_host)
        if hosts == {to_host}:
            before = after = paths.queue.read_bytes()
        else:
            before, after = host_roles.write_queue(paths.queue, [effective.get(j["id"], j) for j in queue["jobs"]],
                                                   jobs.load_queue)
        for event in events:
            jobs.record(event, paths.registry)
        placed = json.loads(paths.pool_manifest.read_text()).get("effective_jobs", {}) if paths.pool_manifest.is_file() \
            else {}
        if any(placed.get(i) != effective[i] for i in unit):
            host_roles.append_pool_record(paths.pool_manifest, {
                "time": stamp, "to": to_host, "reason": f"migration of a checkpoint-paused run "
                f"({records[0]['reason']}); registry migrated events, resume from latest.pt", "ids": unit,
                "rewritten_exports": [], "original": {i: r["original_row"] for i, r in zip(unit, records)},
                "queue_sha256_before": hashlib.sha256(before).hexdigest(),
                "queue_sha256_after": hashlib.sha256(after).hexdigest()}, effective)
        destination = {"commit": commit, "image_digest": environment, "scope_files": scope["files"],
                       "scope_third_party": scope["third_party"], "scripts": scripts, "source_commit": source_commit,
                       "operational_differences": scope.get("operational_differences", [])}
        saved = {}
        for i, record in reversed(list(zip(unit, records))):  # companions first, the run itself last
            saved[i] = record if record["state"] == "switched" else _save(
                paths, record, "switched", f"queue and registry moved to {to_host}", owner_host=to_host,
                destination=destination, provenance_transition={"renamed": renamed})
        return [saved[i] for i in unit]


# ---------------------------------------------------------------------------------------------------- confirm / abort
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _resumed(paths: Paths, job: dict, step: int, since: float) -> bool:
    """The destination attempt (launched at ``since``) resumed at the paused step and made progress past it."""
    base = paths.root / run_dir(job)
    script = Path(job["script"]).name
    if script == "train_rl.py":
        return (any(e.get("event") == "resumed" and e.get("step") == step and float(e.get("time", 0)) >= since - 60
                    for e in _jsonl(base / "events.jsonl"))
                and any(int(r.get("step", 0)) > step for r in _jsonl(base / "train.jsonl")))
    if script == "train.py":
        log = (base / "log.txt").read_text() if (base / "log.txt").is_file() else ""
        return (re.search(rf"resumed from \S+ at step {step}\b", log) is not None
                and any(int(r.get("step", 0)) > step for r in _jsonl(base / "metrics.jsonl")))
    raise MigrationError(f"no resume check for {script}")


def confirm(run_id: str, paths: Paths | None = None) -> list[dict]:
    """Every member runs (or finished) on the destination in a new attempt; the run itself resumed at the paused step
    and progressed past it in that attempt."""
    paths = paths or Paths()
    jobs = _scheduler()
    with locked(paths):
        queue, by_id, state = _load(paths)
        unit = unit_of(run_id, by_id)
        records = _records(paths, unit)
        if any(r is None or r["state"] != "switched" for r in records):
            raise MigrationError(f"{run_id}: not switched")
        attempts = {}
        for i, record in zip(unit, records):
            current, member = state.get(i, {}), record["member"]
            first = member["attempt"] + 1 if member["launched"] else 1
            if (current.get("host") != record["to_host"] or current.get("attempts", 0) < first
                    or current.get("status") not in jobs.ACTIVE | {jobs.DONE}):
                raise MigrationError(f"{i}: no running or finished destination attempt yet ({current.get('status')})")
            attempts[i] = current["attempts"]
        trainer = records[0]["member"]
        if not _resumed(paths, by_id[run_id], trainer["stop"]["checkpoint_step"],
                        float(state[run_id].get("attempt_since", 0))):
            raise MigrationError(f"{run_id}: no verified resume at step {trainer['stop']['checkpoint_step']} in "
                                 f"attempt {attempts[run_id]}")
        return [_save(paths, record, "verified", f"resumed on {record['to_host']} attempt {attempts[i]}",
                      resumed_attempt=attempts[i]) for i, record in zip(unit, records)]


def abort(run_id: str, reason: str, paths: Paths | None = None) -> list[dict]:
    paths = paths or Paths()
    with locked(paths):
        queue, by_id, state = _load(paths)
        unit = unit_of(run_id, by_id)
        records = _records(paths, unit)
        if any(r is None or r["state"] != "migrating" for r in records):
            raise MigrationError(f"{run_id}: only a migration before its switch can be aborted")
        return [_save(paths, record, "aborted", f"aborted: {reason}", owner_host=record["from_host"])
                for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("begin")
    start.add_argument("--to", required=True, choices=host_roles.SCHEDULER_HOSTS)
    start.add_argument("--reason", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--listing", required=True, type=Path)
    stop = sub.add_parser("abort")
    stop.add_argument("--reason", required=True)
    for command in ("begin", "verify", "abort"):
        sub.choices[command].add_argument("run_id")
    for command in ("switch", "confirm"):
        sub.add_parser(command).add_argument("run_id")
    sub.add_parser("status").add_argument("run_id", nargs="?")
    args = parser.parse_args()
    if args.command == "begin":
        result = begin(args.run_id, args.to, args.reason)
    elif args.command == "verify":
        result = verify(args.run_id, json.loads(args.listing.read_text()))
    elif args.command == "switch":
        result = switch(args.run_id)
    elif args.command == "confirm":
        result = confirm(args.run_id)
    elif args.command == "abort":
        result = abort(args.run_id, args.reason)
    else:
        found = host_roles.read_ownership()
        result = [found[args.run_id]] if args.run_id else [
            {k: r.get(k) for k in ("job_id", "kind", "state", "owner_host", "from_host", "to_host", "time")}
            for r in found.values()]
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
