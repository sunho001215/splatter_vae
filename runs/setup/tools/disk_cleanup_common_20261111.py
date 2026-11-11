"""Shared, read-only inputs of the 2026-11-11 cleanup tools (user directive 2026-11-11 ~17:30; memory
disk-cleanup-directive-20261111.md). Used by the dated copies of the reviewed 2026-10-25 tools:
disk_cleanup_stage0_20261111.py, disk_cleanup_local_models_20261111.py, disk_cleanup_remote_20261111.py,
remote_container_cleanup_20261111.py.

Differences to the 2026-10-25 tools, all of them tightening:
- the scheduler is NOT imported or executed (directive: never import/execute ops.campaign or scripts/jobs.py):
  ``read_registry`` below is a verbatim copy of the pure registry replay ops/jobs/scheduler.py:read_registry (it only
  reads experiments/registry.jsonl) and the queue is read with yaml.safe_load;
- active admission pauses (runs/setup/admission-pauses.json; ops/jobs/lifecycle.py ``paused`` semantics) protect every
  matching run; an unreadable pause file fails closed;
- protected families (PROTECT_PREFIXES): the just-completed four-arm study and its named references (under analysis),
  every real-world / FQL / serving run, paused or stopped families whose pause record keeps their checkpoints;
- every run named under /home/ws/ws/realworld-ops (serving POLICY-INVENTORY / checkpoint manifests, robot-trial
  inventory, diagnostics) is protected as a whole;
- reference scan (the 2026-11-06 refs_v2 rules): exact ``runs/...{pt,npz,pth,ckpt}`` paths, templated paths, and
  (run, step) pairs named together with a checkpoint word, over the campaign records, sources, reports, artifacts,
  paper, presentation, real-world operations, validation notes, memory and every run's config.yaml; candidate
  manifests of cleanups are not references;
- every queue row that references a run (any field) must be done (this covers evaluators, camera re-evaluations,
  continuations and dependents).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
RUNS = ROOT / 'runs'
REGISTRY = ROOT / 'experiments/registry.jsonl'
QUEUE = ROOT / 'experiments/queue.yaml'
PAUSES = ROOT / 'runs/setup/admission-pauses.json'
OUT = Path('/home/ws/ws/realworld-ops/cleanup-20261111')
RECORD = ROOT / 'runs/setup/disk-cleanup-20261111'
REMOTE_ROOT = '/home/compu/kaist/sunho'

# ---------------------------------------------------------------------------------------------------------------
# registry replay: verbatim copy of ops/jobs/scheduler.py read_registry (pure function of the registry file)
DONE, FAILED, RUNNING, PENDING = "done", "failed", "running", "pending"
ACTIVE = {RUNNING, "launching", "sync_pending"}


def read_registry(path: Path = REGISTRY) -> dict[str, dict]:
    state: dict[str, dict] = {}
    executed = set()
    if path.is_file():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            kind = event["event"]
            job = state.setdefault(event["id"], {"status": PENDING, "attempts": 0, "failures": 0,
                                                "failed_attempts": [], "infrastructure_failures": []})
            if kind in {"launched", "exited", "remote_exited", "results_synced", "remote_verified",
                        "remote_verification_failed", "checkpoint_paused", "intentional_stopped", "migrated"}:
                executed.add(event["id"])
            if kind in {"launching", "launched"}:
                if job["status"] not in ACTIVE or event["attempt"] != job.get("attempt"):
                    job["attempt_since"] = event["time"]
                job.update({key: value for key, value in event.items() if key not in {"event", "id", "time", "attempt"}})
                job.update(status="launching" if kind == "launching" else RUNNING,
                           host=event.get("host", "local"), attempt=event["attempt"],
                           attempts=max(job["attempts"], event["attempt"]), since=event["time"], unknown=False)
                job.pop("sync", None)
                job.pop("remote_verified", None)
                job.pop("remote_verification_failed", None)
                if kind == "launched" and event.get("container_id"):
                    job.setdefault("attempt_containers", {})[str(event["attempt"])] = {
                        "container": event.get("container"), "container_id": event["container_id"]}
            elif kind == "unknown":
                job.update(unknown=True, reason=event.get("reason", "unknown remote state"))
            elif kind == "remote_exited":
                job.update(status="sync_pending", code=event["code"], unknown=False)
            elif kind == "results_synced":
                job.update(sync=event["sync"], unknown=False)
            elif kind == "remote_verified":
                job["remote_verified"] = {key: event[key] for key in ("attempt", "files", "checks") if key in event}
                job["unknown"] = False
                job.pop("remote_verification_failed", None)
            elif kind == "remote_verification_failed":
                job["remote_verification_failed"] = {"attempt": event.get("attempt"), "time": event["time"],
                                                     "reason": event.get("reason", "")}
            elif kind == "container_removed":
                job.setdefault("containers_removed", []).append(event.get("container_id"))
            elif kind == "container_kept":
                job.setdefault("containers_kept", []).append(event.get("container_id"))
            elif kind == "container_kept_cleared":
                job["containers_kept"] = [i for i in job.get("containers_kept", []) if i != event.get("container_id")]
            elif kind == "container_absent":
                job.setdefault("containers_absent", []).append(event.get("container_id"))
            elif kind == "exited":
                synced = job.get("sync", {})
                verified = job.get("host", "local") != "remote" or (synced.get("final") is True
                                                                              and synced.get("verified") is True)
                job.update(status=(DONE if verified else "sync_pending") if event["code"] == 0 else "crashed",
                           code=event["code"], unknown=False)
                attempt = event.get("attempt", job.get("attempt", 0))
                job.setdefault("attempt_codes", {})[str(attempt)] = event["code"]
                if event["code"] != 0 and any(stop["attempt"] == attempt for stop in job.get("intentional_stops", [])):
                    pass
                elif event["code"] != 0:
                    failure = {"attempt": attempt,
                               "launch_time": job.get("attempt_since", event["time"]),
                               "time": event["time"], "code": event["code"]}
                    if not any(item["attempt"] == failure["attempt"] and item["launch_time"] == failure["launch_time"]
                               for item in job["failed_attempts"]):
                        job["failed_attempts"].append(failure)
                        job["failures"] += 1
            elif kind in {"checkpoint_paused", "intentional_stopped"} and event.get("intentional_stop") is True:
                stop = {key: event[key] for key in ("attempt", "checkpoint_step", "checkpoint_sha256", "reason",
                                                    "checkpoint", "container_id", "container_exit_code")
                        if key in event}
                if stop not in job.setdefault("intentional_stops", []):
                    job["intentional_stops"].append(stop)
                    matching = [failure for failure in job["failed_attempts"] if failure["attempt"] == stop["attempt"]]
                    job["failed_attempts"] = [failure for failure in job["failed_attempts"]
                                              if failure["attempt"] != stop["attempt"]]
                    job["failures"] -= len(matching)
                    if matching and job["status"] == FAILED:
                        job["status"] = "crashed"
            elif kind == "infrastructure_failure":
                start, end = event.get("incident_started"), event.get("incident_ended")
                if (event.get("reason") not in {"cuda_init_unavailable", "nvml_init_unavailable"}
                        or type(event.get("attempt")) is not int or event["attempt"] < 1
                        or type(start) not in {int, float} or not math.isfinite(start) or start < 0
                        or (end is not None and (type(end) not in {int, float} or not math.isfinite(end) or end < start))):
                    raise ValueError(f"invalid infrastructure failure event for {event['id']}")
                matching = [item for item in job["failed_attempts"] if item["attempt"] == event["attempt"]
                            and item["time"] >= start and (end is None or item["launch_time"] <= end)
                            and ("failure_time" not in event or item["time"] == event["failure_time"])]
                if len(matching) != 1:
                    raise ValueError(f"infrastructure failure must identify one observed outage failure for {event['id']}")
                failure = matching[0]
                if not any(item["attempt"] == failure["attempt"] and item["failure_time"] == failure["time"]
                           for item in job["infrastructure_failures"]):
                    job["infrastructure_failures"].append({**event, "failure_time": failure["time"]})
                    job["failures"] -= 1
                    if job["status"] == FAILED:
                        job["status"] = "crashed"
            elif kind == "failed":
                job.update(status=FAILED)
            elif kind == "reset":
                reconciliation = event.get("reconciliation", {})
                if reconciliation.get("conclusion") == "launch refused before docker run; deterministic container never existed":
                    checks = reconciliation.get("checks", [])
                    if (job["status"] != "launching" or job.get("host") != "remote"
                            or reconciliation.get("attempt") != job.get("attempt") or len(checks) != 2
                            or any(check.get("containers_with_name") != 0 or check.get("remote_output") != "absent"
                                   for check in checks)):
                        raise ValueError(f"invalid refused-launch reconciliation for {event['id']}")
                    if event["id"] not in executed:
                        job.pop("job_sha256", None)
                        job.pop("commit", None)
                        job.pop("image_digest", None)
                job.update(status=PENDING)
            elif kind == "migrated":
                stops = [stop for stop in job.get("intentional_stops", []) if stop.get("attempt") == job.get("attempt")]
                if (job["status"] in ACTIVE or job.get("host", "local") != event.get("from_host") or not stops
                        or stops[-1].get("checkpoint_step") != event.get("checkpoint_step")
                        or event.get("host") not in {"local", "remote"} or event.get("host") == event.get("from_host")
                        or event.get("sources_identical") is not True
                        or not all(isinstance(event.get(key), str) and event[key] for key in ("commit", "image_digest"))):
                    raise ValueError(f"invalid migration event for {event['id']}")
                job.update(status=PENDING, host=event["host"], commit=event["commit"], image_digest=event["image_digest"],
                           migrated_from={"host": event["from_host"], "attempt": job["attempt"],
                                          "checkpoint_step": event["checkpoint_step"]})
            elif kind == "host_unreachable":
                job.update(status="unreachable", host=event["host"], outage_started=event["time"])
            elif kind == "host_reconnected":
                job.update(status="reachable", host=event["host"])
    return state


def load_queue_rows() -> list[dict]:
    rows = (yaml.safe_load(QUEUE.read_text()) or {}).get('jobs', [])
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate job ids in queue')
    return rows


# ---------------------------------------------------------------------------------------------------------------
# admission pauses (ops/jobs/lifecycle.py semantics; fail closed)
def active_pauses() -> list[dict]:
    spec = json.loads(PAUSES.read_text())
    pauses = spec['pauses']
    if not isinstance(pauses, list) or not all(isinstance(p, dict) for p in pauses):
        raise ValueError('Malformed admission pause file')
    return [p for p in pauses if p.get('active') is True]


def paused(job_id: str, active: list[dict]) -> list[str]:
    base = job_id.removesuffix('-eval')
    return [p['name'] for p in active
            if any(i in p.get('ids', []) or any(i.startswith(prefix) for prefix in (p.get('prefixes') or []))
                   for i in (job_id, base))]


# ---------------------------------------------------------------------------------------------------------------
# explicit protections (directive 2026-11-11: Arm4 just completed and under analysis; FQL diagnostics; CNN no-crop;
# real-world preprocessing / training / serving / robot trials; paused families whose pause keeps checkpoints)
PROTECT_PREFIXES = {
    'ocsw4-': 'Arm4 arm A (completed 11-11, under analysis)',
    'ochf-': 'Arm4 arm B (completed 11-11, under analysis)',
    'c50dg-': 'Arm4 arm C (completed 11-11, under analysis)',
    'c50vis-': 'Arm4 arm D (completed 11-11, under analysis)',
    'ana-arm4': 'Arm4 analysis rows',
    'arm4': 'Arm4 rows',
    'cst50-': 'Arm4 named reference (cst50; C/D compared against it)',
    'cst75-': 'Arm4 analysis ranking member (cst75), kept until the Arm4 analysis is final',
    'chl-': 'Arm4 named reference (chl)',
    's3-': 'Arm4 named reference (Original = s3-*)',
    'i2-m3d': 'Arm4 named reference checkpoints (i2-m3d)',
    'rw': 'real-world run (preprocessing / pretraining / FQL / diagnostics / serving)',
    'realworld': 'real-world run',
    'droid': 'DROID run',
    'fql': 'FQL run',
    'e2e': 'end-to-end CNN run',
    'ds-': 'stopped/paused downstream family (pause record keeps checkpoints, replay, logs)',
    'enc-abl-': 'paused paper-ablation family',
    'sr-csw4-': 'stopped self-render family (pause keeps outputs)',
    's3spec-': 'held Stage 3 specification',
    'timing': 'benchmark leftovers (explicit approval required)',
}


# baselines under the 2026-11-11 robot-policy failure diagnosis (directive M/N: "existing checkpoints incl. earlier
# epochs", DynaMo representation study) and the CNN / FQL / no-crop work: protected by name anywhere in the run id
PROTECT_SUBSTRINGS = {
    'dynamo': 'DynaMo baseline (failure diagnosis uses existing checkpoints incl. earlier epochs)',
    'reviwo': 'ReViWo baseline (failure diagnosis)', 'sincro': 'SinCro baseline (failure diagnosis)',
    'snerl': 'SNeRL baseline', 'mvp3d': '3D-MVP baseline', 'fql': 'FQL', 'e2e': 'end-to-end CNN', 'cnn': 'CNN baseline',
    'noaug': 'CNN no-crop', 'rwcal': 'real-world', 'rwng': 'real-world', 'arm4': 'Arm4',
}


def protected_prefix(run_id: str) -> str | None:
    for prefix, why in PROTECT_PREFIXES.items():
        if run_id.startswith(prefix):
            return why
    low = run_id.lower()
    for sub, why in PROTECT_SUBSTRINGS.items():
        if sub in low:
            return why
    return None


RUN_TOKEN = re.compile(r'runs/(?:pretrain/)?([A-Za-z0-9][A-Za-z0-9_.-]*)')


def realworld_ops_runs() -> dict[str, str]:
    """Every run id named anywhere under /home/ws/ws/realworld-ops (serving inventory, checkpoint manifests, robot-trial
    inventory, diagnostics, briefs), except this cleanup's own manifests and the 11-06 cleanup records."""
    found = {}
    base = Path('/home/ws/ws/realworld-ops')
    skip = (str(base / 'cleanup'), str(OUT))
    for d, dirs, files in os.walk(base):
        if any(d == s or d.startswith(s + '/') for s in skip):
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if x not in {'.git', '__pycache__', 'node_modules', 'env', '.venv'}]
        for f in files:
            p = os.path.join(d, f)
            # Codex session transcripts contain directory listings of runs/ (e.g. diag/codex/audit1.log lists every
            # run); they protect exact paths and (run, step) pairs through reference_scan, not whole runs
            if f.endswith('.log') and ('/codex' in d or f.startswith('codex')):
                continue
            text = _text(p)
            if text is None:
                continue
            for m in RUN_TOKEN.finditer(text):
                found.setdefault(m.group(1).rstrip('.'), p)
    return found


# ---------------------------------------------------------------------------------------------------------------
# reference scan (2026-11-06 refs_v2 rules + run configs + validation notes)
REF = re.compile(r'runs/[A-Za-z0-9_.\-/{}:]+?\.(?:pt|npz|pth|ckpt)\b')
STEPFILE = re.compile(r'step_(\d{7})\.pt')
STEPTOK = re.compile(r'(?:step[_ =:"]*(\d{4,7})\b|step[_ ]?(\d{1,4})k\b|\b(\d{1,4})k\b)', re.I)
WORDS = re.compile(r'checkpoint|ckpt|snapshot|\.pt\b|policy|weights|encoder|export', re.I)
REF_BASES = [ROOT / 'runs/setup', ROOT / 'docs', ROOT / 'reports', ROOT / 'analysis', ROOT / 'scripts', ROOT / 's4d',
             ROOT / 'ops', ROOT / 'configs', ROOT / 'experiments', Path('/home/ws/ws/presentation-assets-20261105'),
             Path('/home/ws/ws/presentation-assets-tools'), Path('/home/ws/ws/rss2027-stage1-paper'),
             Path('/home/ws/ws/realworld-ops'), Path('/home/ws/ws/validation-20261031'),
             Path('/root/.claude/projects/-home-ws/memory')]
SKIP_DIRS = {'.git', 'env', 'envs', '__pycache__', 'node_modules', '.venv', 'downloads', 'site-packages'}
# candidate manifests / deletion ledgers list files; they are not consumers (10-25 incident, 11-06 rule)
SKIP_PATHS = (str(Path('/home/ws/ws/realworld-ops/cleanup')), str(OUT), str(ROOT / 'runs/setup/disk-cleanup-20261019'),
              str(ROOT / 'runs/setup/disk-cleanup-20261025'), str(RECORD))
SKIP_FILES = {str(ROOT / 'runs/setup/replay-cleanup-ledger.jsonl')}


def _text(path: str, limit: float = 200e6, errors: list | None = None) -> str | None:
    """Text of a file, None for symlinks, files > limit and binary files (NUL in the first 8 KiB). An unreadable file
    is appended to ``errors`` when given (reference scans fail closed on it); a file that vanished is skipped."""
    try:
        if os.path.islink(path) or os.path.getsize(path) > limit:
            return None
        with open(path, 'rb') as fh:
            head = fh.read(8192)
            if b'\0' in head:
                return None
            return (head + fh.read()).decode(errors='ignore')
    except FileNotFoundError:
        return None
    except OSError as exc:
        if errors is not None:
            errors.append(f'{path}: {type(exc).__name__}')
        return None


def reference_scan(run_ids: set[str]) -> dict:
    run_re = (re.compile(r'(?<![A-Za-z0-9_.-])(' + '|'.join(sorted(map(re.escape, run_ids), key=len, reverse=True))
                         + r')(?![A-Za-z0-9_.-])') if run_ids else None)
    paths, pairs, nfiles = set(), set(), 0
    sources = []
    for base in REF_BASES:
        if not base.exists():
            continue
        def walk_error(err, base=base):  # fail closed on unreadable directories (Codex re-review finding 4)
            raise RuntimeError(f'reference scan: cannot enumerate {err.filename}: {err}')
        for d, dirs, files in os.walk(base, onerror=walk_error):
            if any(d == s or d.startswith(s + '/') for s in SKIP_PATHS):
                dirs[:] = []
                continue
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            sources += [os.path.join(d, f) for f in files]
    # every run's configuration (a consumer of encoder / checkpoint paths)
    for base in (RUNS, RUNS / 'pretrain'):
        for entry in os.scandir(base):
            if entry.is_dir(follow_symlinks=False):
                for name in ('config.yaml', 'resolved_config.yaml', 'provenance.json'):
                    p = os.path.join(entry.path, name)
                    if os.path.isfile(p):
                        sources.append(p)
    errors = []
    for p in sources:
        if p in SKIP_FILES:
            continue
        text = _text(p, errors=errors)
        if text is None:
            continue
        nfiles += 1
        for m in REF.finditer(text):
            paths.add(m.group(0)[m.group(0).index('runs/'):])
        if run_re is None:
            continue
        for line in text.splitlines():
            line = line[:20000]
            found = set(run_re.findall(line))
            if not found:
                continue
            steps = {int(s) for s in STEPFILE.findall(line)}
            if WORDS.search(line):
                for a, b, c in STEPTOK.findall(line):
                    if a:
                        steps.add(int(a))
                    if b:
                        steps.add(int(b) * 1000)
                    if c:
                        steps.add(int(c) * 1000)
            for r in found:
                for s in steps:
                    pairs.add((r, s))
    if errors:  # fail closed (Codex review 2026-11-11 finding 5)
        raise RuntimeError(f'reference scan: {len(errors)} unreadable sources, e.g. {errors[:3]}')
    # both spellings of a pretraining run directory: runs/<id>/... and runs/pretrain/<id>/...
    paths |= {p.replace('runs/pretrain/', 'runs/', 1) for p in paths if p.startswith('runs/pretrain/')}
    paths |= {'runs/pretrain/' + p[len('runs/'):] for p in paths if not p.startswith('runs/pretrain/')}
    templates = [re.compile('[^/]+'.join(re.escape(x) for x in re.split(r'\{[^}]*\}', r))) for r in paths if '{' in r]
    return {'files_scanned': nfiles, 'paths': paths, 'pairs': pairs, 'templates': templates}


def referenced(rel: str, run: str, step: int | None, refs: dict) -> str | None:
    # both spellings: runs/<id>/... and runs/pretrain/<id>/... (the scan adds both spellings of every reference too)
    alts = {rel, rel.replace('runs/pretrain/', 'runs/', 1),
            'runs/pretrain/' + rel[len('runs/'):] if not rel.startswith('runs/pretrain/') else rel}
    if any(a in refs['paths'] for a in alts):
        return 'path referenced'
    if any(t.fullmatch(a) for t in refs['templates'] for a in alts):
        return 'templated path referenced'
    if step is not None and (run, step) in refs['pairs']:
        return '(run, step) referenced'
    return None


_BLOBS: dict = {}


def referencing_rows(run: str, rows: list[dict]) -> list[str]:
    """Ids of other queue rows naming runs/<run> (or runs/pretrain/<run>) in any field."""
    key = id(rows)
    if _BLOBS.get('key') != key:
        _BLOBS.clear()
        _BLOBS.update(key=key, rows=rows, blobs=[(r['id'], json.dumps({k: v for k, v in r.items() if k != 'id'}))
                                                for r in rows])
    pat = re.compile(r'(?:^|(?<=[\s=:,/\'"]))runs/(?:pretrain/)?' + re.escape(run) + r'(?![A-Za-z0-9_.-])')
    return [rid for rid, blob in _BLOBS['blobs'] if rid != run and run in blob and pat.search(blob)]


def proc_scan(strict: bool = False) -> tuple[dict, dict, set]:
    """(pid -> cmdline, target -> [pids holding it], cwds) of every local process. A process or descriptor that
    vanished meanwhile is skipped; with ``strict`` any other inspection failure (e.g. permission) raises, so holders can
    never be missed silently (Codex re-review 2026-11-11 finding 3; this tool runs as root locally)."""
    import errno
    gone = (errno.ENOENT, errno.ESRCH)
    cmd, held, cwds = {}, {}, set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd[pid] = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            cwds.add(os.readlink(f'/proc/{pid}/cwd'))
            for fd in os.listdir(f'/proc/{pid}/fd'):
                try:
                    held.setdefault(os.readlink(f'/proc/{pid}/fd/{fd}'), []).append(pid)
                except OSError as exc:
                    if strict and exc.errno not in gone:
                        raise
        except OSError as exc:
            if strict and exc.errno not in gone:
                raise
            continue
    return cmd, held, cwds


def mount_points() -> list[str]:
    """Every mount point visible in this mount namespace (/proc/self/mountinfo field 5, octal escapes decoded)."""
    out = []
    with open('/proc/self/mountinfo') as fh:
        for line in fh:
            mp = line.split()[4]
            out.append(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), mp))
    return out


def mount_inside(path: str) -> str | None:
    """A mount point at or below ``path`` (bind mounts on the same filesystem included), else None."""
    path = path.rstrip('/')
    return next((m for m in mount_points() if m == path or m.startswith(path + '/')), None)


COMMIT40 = re.compile(r'\b[0-9a-f]{40}\b')


def release_pins(state: dict, rows: list[dict]) -> dict[str, str]:
    """7-char release commits that must be treated as in use (Codex review 2026-11-11 finding 3), with the first
    reason found: any registry job not done/failed pinned to it; the planned_commit of every row not done; every
    commit named in the queue; the remote campaign driver's release constants and release files
    (ops/campaign/*.py, ops/jobs/*.py, runs/setup/*release*.json, pool-placement.json, admission-pauses.json); the five
    most recent remote acceptances (runs/remote/acceptance, newest first)."""
    pins: dict[str, str] = {}
    for job, s in state.items():
        if s.get('commit') and (s['status'] not in ('done', 'failed') or s.get('unknown')):
            pins.setdefault(str(s['commit'])[:7], f'job {job} {s["status"]}')
    for r in rows:
        if r.get('planned_commit') and state.get(r['id'], {}).get('status') != 'done':
            pins.setdefault(str(r['planned_commit'])[:7], f'planned_commit of {r["id"]}')
    files = [QUEUE, PAUSES, ROOT / 'runs/setup/pool-placement.json'] + sorted((ROOT / 'ops/campaign').glob('*.py')) \
        + sorted((ROOT / 'ops/jobs').glob('*.py')) + sorted((ROOT / 'runs/setup').glob('*release*.json'))
    for f in files:
        try:
            text = f.read_text()
        except FileNotFoundError:
            continue
        for m in COMMIT40.findall(text):
            pins.setdefault(m[:7], f'named in {f.relative_to(ROOT)}')
    accepted = sorted((ROOT / 'runs/remote/acceptance').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in accepted[:5]:
        pins.setdefault(p.name[:7], 'one of the five most recent remote acceptances')
    return pins


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')
