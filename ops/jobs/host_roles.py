"""Unified pool roles (user directive 2026-11-11 ~20:15 KST): which job classes each host may admit.

Both scheduler hosts admit all training families (directive 2026-11-11 ~20:15). Remote-resident classes require a
local input resolver before local admission or placement. Preprocessing and typed verification stay remote; the RTX
5090 inference server never schedules. Roles are read at every decision. Shared classes keep their existing host;
started runs move only through explicit migration and its ownership records.

A row's class (``job_class``) is its explicit ``role:`` field (validated: it must not contradict the row's script or
configuration) or is derived from decreasing-strength evidence, the first tier with any evidence deciding:
  1. the script (Meta-World simulation scripts; real-world / DROID scripts by name),
  2. configuration files (configs/metaworld|rl|realworld|droid/, baseline configs by name or Meta-World task) and the
     data regime (``--regime``, ``data.regime=``),
  3. data roots (/data/metaworld, /data/real_world, /data/droid) and Meta-World task flags (``--task``, ``--tasks``),
  4. any other path or name containing metaworld / realworld / droid,
  5. the class shared by all of the row's dependencies (encoder exports, evaluations).
Several domains within the deciding tier make the row ``ambiguous`` (refused on every host until it declares
``role:``). Real-world / DROID rows take their activity (preprocess, pretrain, fql) from the script.

    python -I ops/jobs/host_roles.py status                 (class counts and refusals of the queue, read-only)
    python -I ops/jobs/host_roles.py classify <job id> [...] (read-only)
    python -I ops/jobs/host_roles.py replace [--dry-run]     (re-place never-launched rows admitted exclusively by local)
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.jobs.retired import retired_reason  # noqa: E402
from s4d.rwlearn.tasks import TARGETS  # noqa: E402

ROLES_PATH = REPO / "configs/hosts/roles.yaml"
OWNERSHIP_DIR = REPO / "runs/setup/ownership"
POOL_MANIFEST = REPO / "runs/setup/pool-placement.json"
EXCEPTIONS_PATH = REPO / "runs/setup/role-exceptions.json"
ADMISSION_LOCK = REPO / "runs/setup/stage2-driver.lock"  # held by both local admission drivers for their whole pass
SCHEDULER_HOSTS = ("local", "remote")
ROLE_HOSTS = ("local", "remote", "inference")
DOMAIN = {"metaworld": "metaworld", "realworld_preprocess": "realworld", "droid_preprocess": "droid",
          "realworld_pretrain": "realworld", "droid_pretrain": "droid", "realworld_fql": "realworld",
          "verification": None}
CLASSES = tuple(DOMAIN)
OWNERSHIP_STATES = ("placed", "migrating", "switched", "verified", "aborted")
# Scripts that simulate Meta-World or read / write only Meta-World data, whatever their arguments.
METAWORLD_SCRIPTS = frozenset({
    "train_rl.py", "eval_rl.py", "train_sac.py", "train_ppo.py", "train_dreamer.py", "train_flashsac.py",
    "collect_metaworld.py", "check_metaworld.py", "split_metaworld.py", "check_metaworld_equivalence.py",
    "verify_metaworld_tasks.py", "render_heldout_sets.py", "screen_tasks.py", "render_snerl.py",
})
PRETRAIN_SCRIPTS = frozenset({"train_ours_continuation.py", "evaluate_ours_length.py", "train.py", "export_encoder.py", "train_snerl.py", "train_dynamo.py", "train_mvp3d.py",
                              "train_sincro.py", "train_reviwo.py"})
PREPROCESS_WORDS = frozenset({
    "preprocess", "preprocessing", "preproc", "ingest", "ingestion", "calib", "calibrate", "calibration", "stereo",
    "foundationstereo", "depth", "track", "tracks", "tracking", "trackeverything", "extract", "manifest", "index",
    "validate", "validation", "convert", "stats", "sam", "sam3", "dino", "dinov2", "features",
})
HOST_FIELDS = ("host", "placement", "pin_host", "gpu")  # what a role re-placement may change in a row


class Classification(NamedTuple):
    cls: str  # a class tag, or "unknown", "ambiguous", "invalid"
    domain: str | None  # metaworld / realworld / droid, or None
    evidence: str


# ---------------------------------------------------------------------------------------------------- configuration
def load_roles(path: Path | None = None) -> dict:
    """The validated role declaration (fails closed: any inconsistency raises)."""
    path = Path(path or ROLES_PATH)
    raw = path.read_bytes()
    roles = yaml.safe_load(raw)
    if not isinstance(roles, dict) or type(roles.get("version")) is not int or roles.get("version") != 2:
        raise ValueError("roles: unsupported role declaration")
    if not isinstance(roles.get("classes"), dict) or tuple(roles["classes"]) != CLASSES:
        raise ValueError(f"roles: the declared classes must be exactly {CLASSES}")
    hosts = roles.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != set(ROLE_HOSTS):
        raise ValueError(f"roles: hosts must be exactly {ROLE_HOSTS}")
    for name, spec in hosts.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("admit"), list):
            raise ValueError(f"roles: host {name} needs an admit list")
        if any(cls not in CLASSES for cls in spec["admit"]) or spec.get("unknown") not in ("allow", "deny"):
            raise ValueError(f"roles: host {name} admits an unknown class or has no unknown policy")
        if spec.get("scheduler") is not (name in SCHEDULER_HOSTS):
            raise ValueError(f"roles: only {SCHEDULER_HOSTS} are scheduler hosts")
        if name not in SCHEDULER_HOSTS and (spec["admit"] or spec["unknown"] != "deny"):
            raise ValueError(f"roles: {name} is not a scheduler host and admits nothing")
        if not isinstance(spec.get("drain_evaluators", False), bool) or (spec.get("drain_evaluators")
                                                                        and name != "remote"):
            raise ValueError("roles: drain_evaluators is a boolean of the remote host")
        if "scripts" in spec:
            raise ValueError("roles: scripts restrictions were replaced by data_locality in version 2")
    if {"realworld_preprocess", "droid_preprocess"} & set(hosts["local"]["admit"]):
        raise ValueError("roles: real-world / DROID preprocessing executes on the remote only")
    if "verification" in hosts["local"]["admit"]:
        raise ValueError("roles: typed verification executes on the remote only")
    for cls in CLASSES:
        admitted = [spec for name, spec in hosts.items() if name in SCHEDULER_HOSTS and cls in spec["admit"]]
        if not admitted:
            raise ValueError("roles: every class needs at least one scheduler host")
    locality = roles.get("data_locality")
    if (not isinstance(locality, dict) or set(locality) != {"remote_resident"}
            or not isinstance(locality["remote_resident"], list)
            or any(cls not in CLASSES for cls in locality["remote_resident"])):
        raise ValueError("roles: data_locality must name only known classes")
    if not {"realworld_preprocess", "droid_preprocess", "realworld_pretrain", "droid_pretrain", "realworld_fql"} <= set(
            locality["remote_resident"]):
        raise ValueError("roles: remote-resident data classes must retain their locality checks")
    return {**roles, "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path)}


def read_ownership(directory: Path | None = None) -> dict[str, dict]:
    """{job id: ownership record} (fails closed on an unreadable or inconsistent record)."""
    directory, records = Path(directory or OWNERSHIP_DIR), {}
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"ownership location {directory} is not a directory")
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.suffix == ".json"):
            record = json.loads(path.read_text())
            if (record.get("job_id") != path.stem or record.get("state") not in OWNERSHIP_STATES
                    or record.get("owner_host") not in SCHEDULER_HOSTS):
                raise ValueError(f"invalid ownership record {path.name}")
            records[path.stem] = record
    return records


def write_ownership(record: dict, directory: Path | None = None) -> Path:
    """Atomically write one ownership record (the previous state is kept in its ``history``)."""
    if record.get("state") not in OWNERSHIP_STATES or record.get("owner_host") not in SCHEDULER_HOSTS:
        raise ValueError("invalid ownership record")
    directory = Path(directory or OWNERSHIP_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['job_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    with open(tmp, "rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return path


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------- classification
def _metaworld_tasks() -> frozenset[str]:
    """Task names of the Meta-World task configs and of the RL task table (screened reserve tasks included)."""
    folder, table = REPO / "configs/metaworld/tasks", REPO / "configs/rl/tasks.yaml"
    names = {path.stem for path in folder.glob("*.yaml")} if folder.is_dir() else set()
    return frozenset(names | set((yaml.safe_load(table.read_text()) or {}) if table.is_file() else {}))


METAWORLD_TASKS = _metaworld_tasks()
# Real-world campaign tasks (user directive 2026-11-06 item 3; covering and pick-and-place 2026-11-09), read from the
# learning registry s4d/rwlearn/tasks.py; drawer-open is also a Meta-World task, so a task flag alone cannot place a
# row that names it.
REALWORLD_TASKS = frozenset(TARGETS)


def _named_domains(text: str) -> set[str]:
    text = text.lower()
    found = set()
    if re.search(r"meta[-_]?world", text):
        found.add("metaworld")
    if re.search(r"real[-_]?world", text):
        found.add("realworld")
    if "droid" in text:
        found.add("droid")
    return found


def _regime_domain(value: str) -> set[str]:
    value = value.lower()
    return {"metaworld"} if value.startswith("metaworld") else {"droid"} if value.startswith("droid") else (
        {"realworld"} if re.match(r"real[-_]?world", value) else set())


def _normalized(args: list) -> list[str]:
    """Arguments with ``--flag=value`` split into ``--flag value`` (both spellings parse alike)."""
    out = []
    for arg in map(str, args):
        out += arg.split("=", 1) if arg.startswith("--") and "=" in arg else [arg]
    return out


def _script_domains(script: str) -> set[str]:
    name = Path(script).name
    return {"metaworld"} if name in METAWORLD_SCRIPTS else _named_domains(name)


def _config_domains(args: list[str]) -> set[str]:
    found = set()
    for i, arg in enumerate(args):
        match = re.search(r"(?:^|/)configs/([^/\s]+)/(\S*)", arg)
        if match:
            folder, rest = match.groups()
            if folder in ("metaworld", "rl"):
                found.add("metaworld")
            elif folder in ("realworld", "droid"):
                found.add(folder)
            elif folder == "baselines":
                named = _named_domains(rest)
                found |= named or ({"metaworld"} if Path(rest).stem in METAWORLD_TASKS else set())
        if arg == "--regime" and i + 1 < len(args):
            found |= _regime_domain(args[i + 1])
        if arg.startswith("data.regime="):
            found |= _regime_domain(arg.split("=", 1)[1])
    return found


def _task_domains(args: list[str]) -> set[str]:
    """Task flags: Meta-World task names; a name the real-world campaign shares (drawer-open) names both domains."""
    found = set()
    for flag in ("--task", "--tasks"):
        if flag in args:
            values = []
            for value in args[args.index(flag) + 1:]:
                if value.startswith("--"):
                    break
                values.append(value)
                if flag == "--task":
                    break
            if values and all(value in METAWORLD_TASKS for value in values):
                found.add("metaworld")
            if any(value in REALWORLD_TASKS for value in values):
                found.add("realworld")
    return found


def _tokens(job: dict) -> list[str]:
    extra = []
    for key in ("inputs", "baseline_export"):
        value = job.get(key)
        extra += [str(v) for v in value] if isinstance(value, list) else ([str(value)] if value else [])
    return [str(job.get("script", ""))] + _normalized(job.get("args", [])) + extra


def _activity(script: str) -> str | None:
    name = Path(script).name
    words = set(re.split(r"[_\-.]", Path(name).stem.lower()))
    if "fql" in words:
        return "fql"
    if name in PRETRAIN_SCRIPTS or words & {"pretrain", "pretraining", "export"}:
        return "pretrain"
    if words & PREPROCESS_WORDS:
        return "preprocess"
    return None


def _domain_class(domain: str, script: str, evidence: str) -> Classification:
    if domain == "metaworld":
        return Classification("metaworld", domain, evidence)
    activity = _activity(script)
    cls = f"{domain}_{activity}" if activity else None
    if cls in DOMAIN:
        return Classification(cls, domain, evidence)
    return Classification("unknown", domain, f"{evidence}; no {domain} activity recognised from the script name")


def _derived(job: dict, by_id: dict | None, seen: frozenset) -> tuple[Classification, set[str]]:
    """The class derived from the row's evidence (ignoring any declared role) and the domains its deciding tier names."""
    script = str(job.get("script", ""))
    args = _normalized(job.get("args", []))
    tiers = (("script", _script_domains(script)), ("configuration", _config_domains(args)),
             ("data / name / task", set().union(*map(_named_domains, _tokens(job)[1:])) | _task_domains(args)))
    for tier, domains in tiers:
        if len(domains) > 1:
            return Classification("ambiguous", None, f"{tier} evidence names {sorted(domains)}; declare role:"), domains
        if domains:
            return _domain_class(next(iter(domains)), script, f"{tier} evidence"), domains
    deps = [d for d in job.get("deps", []) if by_id and d in by_id and d not in seen]
    if deps and len(deps) == len(job.get("deps", [])):
        classes = {job_class(by_id[d], by_id, seen | {job["id"]}).cls for d in deps}
        if len(classes) == 1 and next(iter(classes)) in DOMAIN:
            cls = next(iter(classes))
            return Classification(cls, DOMAIN[cls], "dependency class"), {DOMAIN[cls]} - {None}
    return Classification("unknown", None, "no role evidence"), set()


def job_class(job: dict, by_id: dict | None = None, _seen: frozenset = frozenset()) -> Classification:
    """The row's job class (module docstring). A declared ``role:`` is valid only if the deciding evidence tier names
    no domain or names the role's domain (it may resolve an ambiguity, never contradict the evidence)."""
    role = job.get("role")
    if role is not None:
        if role not in DOMAIN:
            return Classification("invalid", None, f"unknown role {role!r} (configs/hosts/roles.yaml classes)")
        if (role == "verification") != bool(job.get("verification_purpose")):
            return Classification("invalid", None, "role verification is exactly the typed verification rows")
        derived, domains = _derived(job, by_id, _seen)
        if role != "verification" and domains and DOMAIN[role] not in domains:
            return Classification("invalid", None, f"declared role {role} contradicts the {derived.evidence} "
                                                   f"({sorted(domains)})")
        return Classification(role, DOMAIN[role], "declared role")
    if job.get("verification_purpose"):
        return Classification("verification", None, "typed verification row")
    return _derived(job, by_id, _seen)[0]


# ---------------------------------------------------------------------------------------------------- admission
def ownership_refusal(job_id: str, host: str, ownership: dict | None) -> str | None:
    record = (ownership or {}).get(job_id)
    if record is None:
        return None
    # a migration holds its whole unit (the run and its evaluator) until every member's record is switched
    if any(((ownership or {}).get(member) or {}).get("state") == "migrating" for member in record.get("unit") or [job_id]):
        return (f"migration {record.get('from_host')} -> {record.get('to_host')} in progress "
                f"(runs/setup/ownership/{job_id}.json)")
    if record["owner_host"] != host:
        return f"owned by {record['owner_host']} (ownership record, state {record['state']})"
    return None


def load_exceptions(path: Path | None = None) -> dict[str, dict]:
    """{job id: exception} of the coordinator's explicit role exceptions (runs/setup/role-exceptions.json, outside the
    tested sources so it needs no release; absent = none). The only scope is ``retry_in_place``: a listed run that
    already launched on ``host`` may be retried there from its own checkpoint (e.g. a near-complete remote Meta-World
    run interrupted by a remote outage), although its class is not a role of that host. A grant names the interrupted
    ``attempt`` (which must have executed there: a recorded container / process) and the run's ``job_sha256``; it covers
    the retry of exactly that attempt, never a row that has not run there, never another host, and never overrides an
    ownership record. Fails closed."""
    path = Path(path or EXCEPTIONS_PATH)
    if not path.is_file():
        return {}
    found = {}
    for item in json.loads(path.read_text()).get("exceptions", []):
        if (not isinstance(item, dict) or item.get("scope") != "retry_in_place" or item.get("host") not in SCHEDULER_HOSTS
                or not isinstance(item.get("id"), str) or not item.get("reason") or not item.get("granted")
                or type(item.get("attempt")) is not int or item["attempt"] < 1
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("job_sha256")))):
            raise ValueError(f"invalid role exception in {path.name}")
        if item.get("active", True) is True:
            found[item["id"]] = item
    return found


def retry_in_place(job_id: str, host: str, current: dict | None, exceptions: dict | None) -> bool:
    """The grant (load_exceptions) covers this retry: the run's latest attempt is the granted one, it executed on
    ``host`` (a launched container or process is recorded for it) and the run identity is the granted one."""
    grant, current = (exceptions or {}).get(job_id), current or {}
    if grant is None or grant["host"] != host or current.get("host") != host:
        return False
    executed = (current.get("attempt_containers", {}).get(str(grant["attempt"])) or {}).get("container_id") \
        if host == "remote" else current.get("pid")
    return (current.get("attempt") == grant["attempt"] == current.get("attempts") and bool(executed)
            and current.get("job_sha256") == grant["job_sha256"])


def drain_evaluator(job: dict, host: str, roles: dict, trainer: dict | None) -> bool:
    """The drain exception (configs/hosts/roles.yaml ``drain_evaluators``): ``job`` is the evaluator companion of its
    trainer (``<trainer>-eval`` running scripts/eval_rl.py on ``--run-dir runs/<trainer> --train-job <trainer>`` over
    all snapshots of suite v1, i.e. not a re-evaluation) and the trainer's registry state (``trainer``) shows it
    confirmed running (a recorded container / process) or completed on ``host``: the evaluator reads that trainer's
    snapshots, which exist only there."""
    run_id = job["id"].removesuffix("-eval")
    args = _normalized(job.get("args", []))

    def option(flag):
        return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None

    trainer = trainer or {}
    executed = trainer.get("status") in ("sync_pending", "done") or (
        trainer.get("status") == "running" and bool(
            (trainer.get("attempt_containers", {}).get(str(trainer.get("attempt"))) or {}).get("container_id")
            if host == "remote" else trainer.get("pid")))
    return (roles["hosts"][host].get("drain_evaluators") is True and job["id"].endswith("-eval")
            and job.get("script") == "scripts/eval_rl.py" and not job.get("reevaluates")
            and option("--train-job") == run_id and option("--run-dir") == f"runs/{run_id}"
            and option("--snapshots") in (None, "all") and option("--suites") in (None, "v1")
            and trainer.get("host") == host and not trainer.get("unknown") and executed)


def refusal(job: dict, host: str, roles: dict, ownership: dict | None = None, by_id: dict | None = None,
            current: dict | None = None, exceptions: dict | None = None, trainer: dict | None = None) -> str | None:
    """None if ``host`` may admit ``job`` under the roles, the ownership records, the explicit role exceptions
    (``current``: the job's registry state) and the drain exception (``trainer``: the registry state of the trainer an
    evaluator companion belongs to; pass it for admission only, never for a placement), else the reason."""
    reason = retired_reason(job, by_id)
    if reason is not None:
        return reason
    spec = roles["hosts"].get(host)
    if spec is None or spec.get("scheduler") is not True:
        return f"{host} is not a scheduler host (configs/hosts/roles.yaml)"
    found = job_class(job, by_id)
    if found.cls in ("invalid", "ambiguous"):
        return f"{found.cls} job class: {found.evidence}"
    allowed = found.cls in spec["admit"] or (found.cls == "unknown" and found.domain is None
                                              and spec["unknown"] == "allow")
    # Dataset reset readiness rows only read verified prep outputs (bulk workers
    # remain under their established host policy). The user authorized either host.
    if (not allowed and host == "local" and found.cls == "realworld_preprocess"
            and job.get("script") == "scripts/realworld_dsreset_check.py"
            and str(job.get("id", "")).startswith("rwv2-20261113-")):
        allowed = True
    if not allowed and retry_in_place(job["id"], host, current, exceptions):
        allowed = True  # an explicit retry in place of a run that executed here (load_exceptions)
    if not allowed and found.cls == "metaworld" and drain_evaluator(job, host, roles, trainer):
        allowed = True  # the evaluator companion of a trainer that ran here (drain exception)
    if not allowed:
        return f"job class {found.cls} ({found.evidence}) is not a {host} role (configs/hosts/roles.yaml)"
    reason = locality_refusal(job, host, roles, by_id)
    if reason is not None:
        return reason
    return ownership_refusal(job["id"], host, ownership)


def migrated_state_refusal(job: dict, ownership: dict | None, current: dict | None, fingerprint: str,
                           root: Path = REPO) -> str | None:
    """A run switched here by a migration (ops/jobs/migration.py) launches only with the queue row it was launched with
    on the source host (job fingerprint) and, until its first destination attempt, with its verified transferred state
    unchanged (every manifest file present with its recorded size and modification time; the resume checkpoint with
    its checksum)."""
    record = (ownership or {}).get(job["id"])
    if record is None or record.get("kind") != "migration" or record["state"] not in ("switched", "verified"):
        return None
    member = record.get("member") or {}
    if member.get("job_sha256") and fingerprint != member["job_sha256"]:
        return "the queue row differs from the run launched on the source host (job fingerprint)"
    if record["state"] != "switched" or not member.get("launched") or (current or {}).get("attempts") != member.get(
            "attempt"):
        return None
    prefix = f"{member['run_dir']}/"
    renamed = (record.get("provenance_transition") or {}).get("renamed", {})
    for name, item in sorted((record.get("manifest") or {}).get("files", {}).items()):
        if not name.startswith(prefix):
            continue
        path = root / renamed.get(name, name)
        try:
            stat = path.stat()
        except OSError:
            return f"transferred state {name} is missing"
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item.get("mtime_ns", stat.st_mtime_ns):
            return f"transferred state {name} changed after its verification"
        if name == f"{prefix}checkpoints/latest.pt" and hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            return "the resume checkpoint differs from its verified checksum"
    return None


def migrated_source_refusal(job_id: str, ownership: dict | None, source_sha256: dict, root: Path = REPO) -> str | None:
    """A run migrated here (ownership state switched / verified) resumes only on the code it was migrated with: every
    file and third-party pin of the recorded execution scope (ops/jobs/migration.py switch) must be unchanged."""
    record = (ownership or {}).get(job_id)
    if record is None or record.get("kind") != "migration" or record["state"] not in ("switched", "verified"):
        return None
    destination = record.get("destination") or {}
    expected = {**destination.get("scope_files", {}), **destination.get("scope_third_party", {})}
    if not expected:
        return "migrated run without a recorded execution scope"
    for key, digest in sorted(expected.items()):
        found = source_sha256.get(key)
        if found is None and "@" not in key and (root / key).is_file():
            found = hashlib.sha256((root / key).read_bytes()).hexdigest()
        if found != digest:
            return f"{key} differs from the execution scope it was migrated with (same pinned code required)"
    return None


def with_role(job: dict, by_id: dict | None = None) -> dict:
    """A dispatch copy of ``job`` carrying its class as an explicit ``role`` (unchanged when it already declares one or
    its class is not a role class). A dispatch queue holds only the admitted rows, so a class derived from dependencies
    (an encoder export classified by its producer) must travel with the row. The job fingerprint is unchanged."""
    found = job_class(job, by_id)
    return {**copy.deepcopy(job), "role": found.cls} if "role" not in job and found.cls in DOMAIN else copy.deepcopy(job)


def remote_resident(job: dict, roles: dict, by_id: dict | None = None) -> bool:
    return job_class(job, by_id).cls in roles["data_locality"]["remote_resident"]


def locality_refusal(job: dict, host: str, roles: dict, by_id: dict | None = None) -> str | None:
    if host == "local" and remote_resident(job, roles, by_id):
        # Lazy import: the resolver table is the single source of truth; realworld_local also uses host_roles.
        from ops.jobs.realworld_local import has_resolver

        if not has_resolver(job):
            return f"data locality: inputs live on the remote; no local input resolver for {job.get('script', '')}"
    return None


def role_hosts(job: dict, roles: dict, by_id: dict | None = None) -> tuple[str, ...]:
    """Admitting scheduler hosts, including locality, independent of ownership and current placement."""
    found = job_class(job, by_id)
    return tuple(name for name in SCHEDULER_HOSTS
                 if (found.cls in roles["hosts"][name]["admit"]
                     or (found.cls == "unknown" and found.domain is None and roles["hosts"][name]["unknown"] == "allow"))
                 and locality_refusal(job, name, roles, by_id) is None)


def role_host(job: dict, roles: dict, by_id: dict | None = None) -> str | None:
    """The unique role host, or the row's existing host when its class and script are admitted on both hosts.

    Shared admission never moves a row implicitly; placement is explicit.
    """
    hosts = role_hosts(job, roles, by_id)
    current = job.get("host", "local")
    return current if current in hosts else hosts[0] if len(hosts) == 1 else None


# ---------------------------------------------------------------------------------------------------- re-placement
def _partner(job_id: str) -> str:
    return job_id.removesuffix("-eval") if job_id.endswith("-eval") else job_id + "-eval"


def _output_dirs(job: dict, root: Path) -> list[Path]:
    args = [str(a) for a in job.get("args", [])]
    out = [root / "runs" / job["id"]]
    for flag in ("--run-dir", "--output-dir"):
        if flag in args and args.index(flag) + 1 < len(args):
            out.append(root / args[args.index(flag) + 1])
    return out


def replaceable(job: dict, state: dict, by_id: dict, roles: dict, root: Path) -> str | None:
    """None if a remote row may be re-placed to local by the roles, else why not. Only rows that never launched (no
    registry entry, no local output directory, an RL partner equally untouched) move; started runs need the migration
    procedure (ops/jobs/migration.py)."""
    if job.get("host", "local") != "remote":
        return "not a remote row"
    if role_host(job, roles, by_id) != "local":
        return "its class is not a local role"
    if job.get("verification_purpose") or job.get("continues") or job.get("stage") == 2:
        return "verification, continuation or Stage 2 row (own placement procedure)"  # Stage 2: its driver's overlay
    members = [job] + ([by_id[_partner(job["id"])]] if _partner(job["id"]) in by_id else [])
    if job.get("script") in ("scripts/train_rl.py", "scripts/eval_rl.py") and len(members) != 2:
        return "RL row without its partner"
    for member in members:
        if member["id"] in state:
            return f"{member['id']} has registry state (started runs migrate explicitly)"
        if any(path.exists() for path in _output_dirs(member, root)):
            return f"{member['id']} has a local output directory"
        if member.get("host", "local") != "remote" or role_host(member, roles, by_id) != "local":
            return f"partner {member['id']} is not a remote local-role row"
    return None


def destination_row(job: dict, host: str, placement: str) -> dict:
    """``job`` placed on ``host``: host fields only (host, placement, a pin superseded by the role, an explicit GPU
    selection the destination does not authorize becomes "any"); the job fingerprint is unchanged."""
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS

    new = copy.deepcopy(job)
    new["host"] = host
    new["placement"] = placement
    if "pin_host" in new:
        new["pin_host"] = host
    if new.get("gpu", "any") != "any" and new["gpu"] not in APPROVED_HOST_GPUS[host]:
        new["gpu"] = "any"
    return new


def _placed_row(job: dict, stamp: str) -> dict:
    new = destination_row(job, "local", f"role {stamp} (local; configs/hosts/roles.yaml; runs/setup/pool-placement.json)")
    assert {k: v for k, v in new.items() if k not in HOST_FIELDS} == {k: v for k, v in job.items()
                                                                       if k not in HOST_FIELDS}, job["id"]
    return new


def write_queue(queue_path: Path, new_jobs: list[dict], load_queue: Callable) -> tuple[bytes, bytes]:
    """Replace the queue's job list in place (the rest of the file byte for byte), validated before an atomic rename;
    returns (before, after) bytes (the pool placement's queue rewrite)."""
    before = Path(queue_path).read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == "jobs")
    queue = load_queue(queue_path)
    staged = Path(queue_path).with_name(Path(queue_path).name + ".roles.tmp")
    staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False)
                      + text[node.end_mark.index:])
    check = load_queue(staged)
    if check["jobs"] != new_jobs or check["hosts"] != queue["hosts"]:
        staged.unlink()
        raise ValueError("staged queue differs from the intended rows")
    os.replace(staged, queue_path)
    return before, Path(queue_path).read_bytes()


def append_pool_record(manifest: Path, move: dict, effective: dict) -> None:
    record = json.loads(Path(manifest).read_text()) if Path(manifest).is_file() else {
        "purpose": "Host-neutral pool placement (runs/setup/tools/pool_placement.py)", "moves": [],
        "effective_jobs": {}}
    record["moves"].append(move)
    record["effective_jobs"].update(effective)
    tmp = Path(manifest).with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, manifest)


def replace_to_local(paused: Callable[[str], bool], reason: str = "server roles: destination admits this class exclusively",
                     *, dry_run: bool = False, blocking: bool = False, queue_path: Path | None = None,
                     registry_path: Path | None = None, manifest: Path = POOL_MANIFEST,
                     ownership_dir: Path = OWNERSHIP_DIR, admission_lock: Path = ADMISSION_LOCK,
                     roles_path: Path = ROLES_PATH, root: Path = REPO) -> dict:
    """Re-place every never-launched remote row whose class is admitted exclusively on local to local: host fields only,
    under the local admission lock and the scheduler's registry lock, recorded in pool-placement.json (move entry and
    effective rows, which the local drivers apply as their overlay) and one ownership record per row. Rows under an
    intentional admission pause are never moved (runs/setup/admission-pauses.json). Returns {"moved": [...],
    "kept": {id: reason}}; {"moved": [], "busy": True} when the admission lock is held and not ``blocking``."""
    from ops.jobs import scheduler as jobs

    queue_path = Path(queue_path or jobs.QUEUE)
    registry_path = Path(registry_path or jobs.REGISTRY)
    roles = load_roles(roles_path)
    with open(admission_lock, "a") as admission:
        try:
            fcntl.flock(admission, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            return {"moved": [], "busy": True}
        with open(registry_path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = jobs.load_queue(queue_path)
            by_id = {j["id"]: j for j in queue["jobs"]}
            state = jobs.read_registry(registry_path)
            ownership = read_ownership(ownership_dir)
            chosen, kept = [], {}
            for job in queue["jobs"]:
                if job.get("host", "local") != "remote" or role_host(job, roles, by_id) != "local":
                    continue
                why = "intentional admission pause" if paused(job["id"]) else replaceable(job, state, by_id, roles, root)
                if why is None and job["id"] in ownership and ownership[job["id"]]["state"] in ("migrating", "switched"):
                    why = "an explicit migration owns it"
                if why is None:
                    chosen.append(job["id"])
                else:
                    kept[job["id"]] = why
            for job_id in list(chosen):  # a trainer / evaluator pair moves together or not at all
                partner = _partner(job_id)
                if partner in by_id and partner not in chosen:
                    chosen.remove(job_id)
                    kept[job_id] = f"partner {partner} stays ({kept.get(partner, 'not re-placeable')})"
            # a re-placement interrupted after its queue rewrite (before its records) is completed here
            placed = json.loads(Path(manifest).read_text()).get("effective_jobs", {}) if Path(manifest).is_file() else {}
            unrecorded = [j["id"] for j in queue["jobs"] if str(j.get("placement", "")).startswith("role ")
                          and j.get("host", "local") == "local" and placed.get(j["id"]) != j]
            if dry_run or not (chosen or unrecorded):
                return {"moved": [], "would_move": chosen if dry_run else [], "kept": kept, "unrecorded": unrecorded}

            def records(ids, original, effective, before, after, note):
                stamp = now()
                append_pool_record(manifest, {
                    "time": stamp, "to": "local", "reason": f"{note} (configs/hosts/roles.yaml sha256 {roles['sha256']})",
                    "ids": ids, "rewritten_exports": [], "original": original,
                    "queue_sha256_before": hashlib.sha256(before).hexdigest(),
                    "queue_sha256_after": hashlib.sha256(after).hexdigest()}, effective)
                for i in ids:
                    previous = ownership.get(i)
                    write_ownership({
                        "job_id": i, "kind": "role_placement", "state": "placed", "owner_host": "local",
                        "from_host": "remote", "to_host": "local", "attempts": 0, "time": stamp, "reason": note,
                        "roles_sha256": roles["sha256"], "job_class": job_class(by_id[i], by_id).cls,
                        "queue_sha256_after": hashlib.sha256(after).hexdigest(),
                        "history": (previous.get("history", []) + [{k: v for k, v in previous.items() if k != "history"}]
                                    if previous else [])}, ownership_dir)

            if unrecorded:
                current = Path(queue_path).read_bytes()
                records(unrecorded, {i: None for i in unrecorded}, {i: by_id[i] for i in unrecorded}, current, current,
                        f"{reason} (records completed after an interrupted re-placement; original rows not recorded)")
            if not chosen:
                return {"moved": [], "repaired": unrecorded, "kept": kept}
            stamp = now()
            effective = {i: _placed_row(by_id[i], stamp) for i in chosen}
            for i in chosen:
                assert jobs.job_fingerprint(effective[i]) == jobs.job_fingerprint(by_id[i]), i
            before, after = write_queue(queue_path, [effective.get(j["id"], j) for j in queue["jobs"]], jobs.load_queue)
            records(chosen, {i: by_id[i] for i in chosen}, effective, before, after, reason)
            return {"moved": chosen, "repaired": unrecorded, "kept": kept}


def placement_repair_needed(rows: list[dict], manifest: Path | None = None) -> bool:
    """A role placement of the queue has no matching pool-placement record (an interrupted re-placement)."""
    manifest = Path(manifest or POOL_MANIFEST)
    placed = json.loads(manifest.read_text()).get("effective_jobs", {}) if manifest.is_file() else {}
    return any(str(j.get("placement", "")).startswith("role ") and placed.get(j["id"]) != j for j in rows)


# ---------------------------------------------------------------------------------------------------- CLI
def main() -> None:
    from ops.jobs import scheduler as jobs
    from ops.jobs.admission_pause import paused

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    classify = sub.add_parser("classify")
    classify.add_argument("ids", nargs="+")
    replace = sub.add_parser("replace")
    replace.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    roles = load_roles()
    if args.command == "replace":
        print(json.dumps(replace_to_local(paused, dry_run=args.dry_run, blocking=True), indent=1))
        return
    queue = jobs.load_queue()
    by_id = {j["id"]: j for j in queue["jobs"]}
    ownership = read_ownership()
    if args.command == "classify":
        for job_id in args.ids:
            job = by_id.get(job_id)
            if job is None:
                print(json.dumps({"id": job_id, "error": "not in the queue"}))
                continue
            print(json.dumps({"id": job_id, "host": job.get("host", "local"), **job_class(job, by_id)._asdict(),
                              "refusal": {h: refusal(job, h, roles, ownership, by_id) for h in SCHEDULER_HOSTS}}))
        return
    counts: dict = {}
    for job in queue["jobs"]:
        host = job.get("host", "local")
        found = job_class(job, by_id)
        key = f"{host}/{found.cls}" + ("" if refusal(job, host, roles, ownership, by_id) is None else " (refused)")
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"roles_sha256": roles["sha256"], "rows": counts, "ownership_records": len(ownership)}, indent=1))


if __name__ == "__main__":
    main()
