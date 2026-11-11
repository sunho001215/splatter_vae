"""Remote container inventory + verified cleanup (user instruction 2026-11-11 ~19:20, memory
remote-mig-reboot-directive-20261111.md "container simplification" (1)). Companion of remote_container_cleanup_20261111.py
(same archive-first removal script of ops/jobs/remote_backend.py). The scheduler and ops.campaign are not imported; the
fail-closed MIG hold is not touched; nothing is launched. Other users' containers are listed by name and state only.
Revised after the Codex review 2026-11-11 (codex/review3-containers.log.last.md: phase-A rwprep kept; behavioural
exact-id binding; publication proof for every terminal status; fresh per-batch re-inventory + reclassification +
record checks under the PRE / ING / result-sync locks; stop at the first per-container failure; fail-closed verification).

Classes (all: exited, read-only root filesystem, exact id pinned in the removal script, labels and deterministic name;
docker log + inspect + diff + sha256 receipt archived before `docker rm <id>`; no force, no volumes, no prune):
  rwprep (release-* profiles only)   containers of the real-world preprocessing dispatcher (/home/ws/ws/realworld-ops/pre):
      labels s4d.role=rwprep, s4d.owner=kaist-sunho, s4d.project=splatter4d, s4d.rwprep.job/.episode/s4d.attempt equal
      to EXACTLY one attempt record in pre/state.json naming this container; terminal attempt (ok -> exit 0, failed ->
      exit != 0; the dispatcher reconciles only running/launched/launching attempts and adopts only containers missing
      from its records); episode `done` with QA done and every stage record of state.json equal in the ING manifest
      (sync_stage_records semantics); exact-id binding (the dispatcher records no container id): the job's own
      result.json (written inside the container) names this container, job and attempt with the attempt's status, the
      container's StartedAt / FinishedAt equal the result's started_utc / finished_utc (+-5 s), it was created within
      the attempt's launch window and its argv is exactly rwprep_job.py <jobs>/<job>/job-a<N>.json; job-a<N>.json names
      the same job / attempt; pipeline.log exists; ok attempts: the shared outputs root's qa.json exists.
      Phase-A attempts (calrel-6c3b9ec-prep) are KEPT: their episodes have no QA yet (phase B held by the MIG hold).
  verification   scheduler verification containers: registry done / failed (not unknown), registry launch record of
      exactly this id + attempt, recorded exit code == container exit code; evidence reports present (native:
      tests.json + tests.log; isolation: gpu_isolation.json). Acceptance reads only local report copies.
  job_failed     failed / crashed attempts of scheduler jobs (exit code + id recorded, no intentional-stop record): the
      scheduler never removes or reads them again (a retry is a new attempt / container); archive in
      runs/<run_id>/containers/ under the scheduler's per-run result-sync lock (taken without blocking);
      run dir must hold console.log + provenance.json.
Kept: live containers, scheduler jobs running / launching / sync_pending / unknown / done (the scheduler removes done
ones), rwprep of non-done or phase-A episodes, ambiguous identity (e267ba8-20261011), other agents' study containers
(s4d-rwx-*), anything not named s4d-*.

    nice -n 19 .venv/bin/python -I runs/setup/tools/remote_container_inventory_cleanup_20261111.py           # plan
    nice -n 19 .venv/bin/python -I runs/setup/tools/remote_container_inventory_cleanup_20261111.py --apply   # reviewed
"""
import base64
import contextlib
import fcntl
import json
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_cleanup_common_20261111 as C  # noqa: E402

sys.path.insert(0, str(C.ROOT))
from ops.jobs.remote_backend import _INSPECT_IDENTITY, _REMOVAL_SCRIPT, acceptance_manifest  # noqa: E402
from ops.remote.access import RemoteClient  # noqa: E402

REMOTE, RW = C.REMOTE_ROOT, C.REMOTE_ROOT + '/realworld'
PRE = Path('/home/ws/ws/realworld-ops/pre')
ING_DIR = Path('/home/ws/ws/realworld-ops/ingest')
OUT = C.OUT / 'remote-container-inventory.json'
REVIEWED = C.OUT / 'remote-container-inventory.reviewed.json'
APPLY = '--apply' in sys.argv
NEVER = {'s4d-verify-native-e267ba8-20261011-a1'}
RWPREP_FORMAT = ('{{.Id}}|{{.State.Status}}|{{.State.ExitCode}}|{{index .Config.Labels "s4d.role"}}|'
                 '{{index .Config.Labels "s4d.rwprep.job"}}|{{index .Config.Labels "s4d.attempt"}}|'
                 '{{index .Config.Labels "s4d.rwprep.episode"}}|{{index .Config.Labels "s4d.owner"}}|'
                 '{{index .Config.Labels "s4d.project"}}|{{.Name}}|{{.HostConfig.ReadonlyRootfs}}')
INSPECT = ('{{.Id}}|{{.Name}}|{{.State.Status}}|{{.State.ExitCode}}|{{.Created}}|{{.State.StartedAt}}|'
           '{{.State.FinishedAt}}|{{.HostConfig.ReadonlyRootfs}}|{{json .Args}}|{{json .Config.Labels}}')
TOL = 5.0
CLIENT = RemoteClient()
ADOPTED = {}  # filled at start (adoption_records)


def ssh(command, timeout=900, idempotent=True):
    time.sleep(2)
    return CLIENT.ssh(command, wait=True, idempotent=idempotent, timeout=timeout)


def ts(value):
    """Seconds since the epoch of an ISO time (Docker nanoseconds / 'Z' / offsets); None if unparsable."""
    if not value:
        return None
    v = re.sub(r'(\.\d{6})\d+', r'\1', str(value)).replace('Z', '+00:00')
    try:
        t = datetime.fromisoformat(v)
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def inspect(ids):
    details = {}
    fmt = shlex.quote(INSPECT)
    for i in range(0, len(ids), 40):
        out = ssh(f"for id in {' '.join(ids[i:i + 40])}; do docker inspect --format {fmt} \"$id\" || echo \"__MISSING__ $id\"; "
                  "done; echo __END__")
        if out.returncode or not out.stdout.rstrip().endswith('__END__'):
            raise SystemExit('inspect failed')
        for line in out.stdout.split('__END__')[0].strip().splitlines():
            if line.startswith('__MISSING__'):
                continue
            p = line.split('|', 9)
            if len(p) != 10 or not re.fullmatch(r'[0-9a-f]{64}', p[0]):
                continue  # vanished / unparsable (e.g. removed meanwhile): never eligible
            # Args JSON never contains '|' for our containers (paths only); labels JSON is the last field
            details[p[0]] = {'id': p[0], 'name': p[1].lstrip('/'), 'state': p[2], 'code': int(p[3]), 'created': p[4],
                             'started': p[5], 'finished': p[6], 'ro': p[7], 'args': json.loads(p[8]),
                             'labels': json.loads(p[9])}
    return details


def inventory():
    r = ssh("docker ps -a --no-trunc --format '{{.ID}}|{{.Names}}|{{.State}}' && echo __END__")
    if r.returncode or not r.stdout.rstrip().endswith('__END__'):
        raise SystemExit('inventory listing failed')
    rows = [line.split('|') for line in r.stdout.split('__END__')[0].strip().splitlines()]
    return rows, inspect([x[0] for x in rows if x[1].startswith('s4d-')])


@contextlib.contextmanager
def shared_locks():
    """PRE state.lock (dispatcher) and ING manifest.lock (ingest worker / sync_stage_records), shared."""
    with open(PRE / 'state.lock', 'a') as a, open(ING_DIR / 'manifest.lock', 'a') as b:
        fcntl.flock(a, fcntl.LOCK_SH)
        fcntl.flock(b, fcntl.LOCK_SH)
        yield json.loads((PRE / 'state.json').read_text()), json.loads((ING_DIR / 'manifest.json').read_text())


def comparable(rec):  # sync_stage_records.py semantics
    out = {k: v for k, v in rec.items() if k not in ('stage', 'updated_at')}
    out.setdefault('stage_schema_version', 1)
    out.setdefault('failure', None)
    return out


FACTS = r'''
import json, os, sys
for p in json.load(sys.stdin):
    ok = os.path.isfile(p) and not os.path.islink(p) and os.path.getsize(p) > 0
    doc = None
    if ok and p.endswith('.json'):
        try:
            d = json.load(open(p))
            import hashlib
            sha = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            if isinstance(d, dict) and d:
                doc = {k: d.get(k) for k in ('status', 'job_id', 'attempt', 'container', 'started_utc', 'finished_utc',
                                             'passed', 'full_suite_passed', 'fingerprint', 'build_id', 'stage')}
                doc['__keys__'] = sorted(d)[:200]
                doc['__sha256__'] = sha
                doc['__canonical__'] = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            else:
                doc = {'unreadable': True, 'not_object': True}
        except Exception:
            doc = {'unreadable': True}
    print(json.dumps([p, ok, doc]))
print('__END__')
'''


def remote_facts(paths):
    facts = {}
    paths = sorted(set(paths))
    for i in range(0, len(paths), 400):
        payload = base64.b64encode(json.dumps(paths[i:i + 400]).encode()).decode()
        r = ssh(f"printf %s {payload} | base64 -d | python3 -I -c {shlex.quote(FACTS)}", timeout=600)
        if r.returncode or not r.stdout.rstrip().endswith('__END__'):
            raise SystemExit('remote fact check failed')
        for line in r.stdout.splitlines():
            if line.startswith('['):
                p, ok, doc = json.loads(line)
                facts[p] = (ok, doc)
    return facts


def registry_ids():
    ids = {}
    for line in C.REGISTRY.read_text().splitlines():
        e = json.loads(line)
        if e.get('container_id') and e['event'] in ('launched', 'remote_exited') and e.get('attempt') is not None:
            ids.setdefault(e['container_id'], set()).add((e['id'], int(e['attempt'])))
    return ids


def canonical(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def adoption_records():
    """{commit7: {commit, image, adopted_at, native_sha, isolation_shas}} of ADOPTED releases (user 2026-11-11 ~20:05;
    tightened after Codex review5): runs/setup/merge-adopt-<commit7>-<YYYYMMDD>.log with exactly one line
    'ADOPTED verify-native-<commit7>-cc <digest12> {passed>0, failed/error/skipped 0}' and an 'adopted at HH:MM:SS'
    line (KST, the file's date), AND exactly one local acceptance manifest runs/remote/acceptance/<commit7>*/manifest.json
    with the full campaign evidence (commit = 40 hex starting with commit7, image digest, full_suite, 4 gpu_isolation
    references, data, cross_host). Anything unparsable or ambiguous is not adopted."""
    import ast
    from datetime import timedelta
    out = {}
    for log in sorted((C.ROOT / 'runs/setup').glob('merge-adopt-*.log')):
        m = re.fullmatch(r'merge-adopt-([0-9a-f]{7})-(\d{8})\.log', log.name)
        if not m:
            continue
        c7, day = m.groups()
        lines = log.read_text(errors='ignore').splitlines()
        adopted = [re.fullmatch(r'ADOPTED (verify-native-([0-9a-f]{7})-cc) ([0-9a-f]{12}) (\{.*\})', x) for x in lines]
        adopted = [x for x in adopted if x]
        when = [re.fullmatch(r'adopted at (\d\d):(\d\d):(\d\d)', x) for x in lines]
        when = [x for x in when if x]
        if len(adopted) != 1 or adopted[0].group(2) != c7 or len(when) != 1:
            continue
        try:
            counts = ast.literal_eval(adopted[0].group(4))
        except (ValueError, SyntaxError):
            continue
        if not (isinstance(counts, dict) and counts.get('passed', 0) > 0
                and all(counts.get(k) == 0 for k in ('failed', 'error', 'skipped'))):
            continue
        hh, mm, ss = (int(x) for x in when[0].groups())
        if not (hh < 24 and mm < 60 and ss < 60):  # Codex review6: no normalisation of invalid clock values
            continue
        at = (datetime.strptime(day, '%Y%m%d').replace(tzinfo=timezone(timedelta(hours=9)))
              + timedelta(hours=hh, minutes=mm, seconds=ss)).timestamp()
        mans = sorted((C.ROOT / 'runs/remote/acceptance').glob(f'{c7}*/manifest.json'))
        if len(mans) != 1:
            continue
        man = json.loads(mans[0].read_text())
        iso = man.get('gpu_isolation') or {}
        if (not re.fullmatch(r'[0-9a-f]{40}', str(man.get('commit', ''))) or not man['commit'].startswith(c7)
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', str(man.get('image_digest', '')))
                or man.get('campaign_eligible', True) is not True or 'verification_purpose' in man
                or not all(isinstance(man.get(k), dict) and re.fullmatch(r'[0-9a-f]{64}', str(man[k].get('sha256', '')))
                           for k in ('full_suite', 'data', 'cross_host'))
                or len(iso) != 4 or not all(re.fullmatch(r'[0-9a-f]{64}', str(v.get('sha256', ''))) for v in iso.values())):
            continue
        try:  # the scheduler's own full acceptance validator (suite, 4 approved GPUs, data, cross-host, source identity)
            acceptance_manifest({'commit': man['commit'], 'image_digest': man['image_digest'],
                                 'evidence_manifest': str(mans[0].relative_to(C.ROOT))}, C.ROOT)
        except Exception:  # RemoteJobError or unreadable artifacts: not adopted
            continue
        # the accepted artifacts name their source job and embed its report: {commit, image_digest, job, report}
        jobs, ok = {}, True
        for ref in [man['full_suite'], *iso.values()]:
            f = C.ROOT / str(ref.get('path', ''))
            import hashlib
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != ref['sha256']:
                ok = False
                break
            art = json.loads(f.read_text())
            if (art.get('commit') != man['commit'] or art.get('image_digest') != man['image_digest']
                    or not isinstance(art.get('report'), dict) or not re.fullmatch(r'verify-[A-Za-z0-9_.-]+', str(art.get('job')))):
                ok = False
                break
            jobs.setdefault(art['job'], set()).add(canonical(art['report']))
        if not ok:
            continue
        out[c7] = {'commit': man['commit'], 'image': man['image_digest'], 'adopted_at': at,
                   'jobs': {k: sorted(v) for k, v in jobs.items()},
                   'adopt_log': log.name, 'manifest': str(mans[0].relative_to(C.ROOT))}
    return out


def superseded(full_commit, failed_at, adopted):
    """Adopted releases (commit7) whose commit descends from ``full_commit`` and that were adopted after the failure."""
    import subprocess
    hits = []
    for c7, rec in adopted.items():
        full, when = rec['commit'], rec['adopted_at']
        if failed_at and when > failed_at and subprocess.run(
                ['git', '-C', str(C.ROOT), 'merge-base', '--is-ancestor', full_commit, full],
                capture_output=True).returncode == 0:
            hits.append(c7)
    return sorted(hits)


def classify(details, state, pre, ing, recorded):
    """(candidates with their required remote records, kept). Record contents are checked by ``confirm``."""
    by_container = {}
    for key, ep in pre['episodes'].items():
        for a in ep['attempts']:
            if a.get('container'):
                by_container.setdefault(a['container'], []).append((key, a))
    plan, kept = [], []
    for cid, d in sorted(details.items(), key=lambda x: x[1]['name']):
        name, l = d['name'], d['labels']
        row = {'id': cid, 'name': name, 'state': d['state'], 'code': d['code'], 'created': d['created'],
               'started': d['started'], 'finished': d['finished'], 'ro': d['ro'],
               'labels': {k: v for k, v in l.items() if k.startswith('s4d.')}}

        def keep(cls, why):
            kept.append({**row, 'class': cls, 'kept': why})
        if d['state'] != 'exited':
            keep('live', f"state {d['state']}")
            continue
        if name in NEVER:
            keep('ambiguous', 'kept since 2026-10-25 (no labels / registry container id mismatch)')
            continue
        if d['ro'] != 'true':
            keep('ambiguous', 'root filesystem not read-only')
            continue
        if l.get('s4d.role') == 'rwprep':
            recs = by_container.get(name, [])
            if len(recs) != 1:
                keep('rwprep', f'{len(recs)} dispatcher attempt records name this container')
                continue
            key, a = recs[0]
            ep = pre['episodes'][key]
            job, n = a.get('job_id'), a.get('attempt')
            if (l.get('s4d.rwprep.job') != job or l.get('s4d.attempt') != str(n) or l.get('s4d.rwprep.episode') != key
                    or l.get('s4d.owner') != 'kaist-sunho' or l.get('s4d.project') != 'splatter4d'
                    or name != f's4d-rwprep-{job}-a{n}'):
                keep('rwprep', 'labels / name differ from the dispatcher record')
                continue
            if ep['status'] != 'done':
                keep('rwprep', f"episode {ep['status']} (dispatcher may still need its attempts)")
                continue
            if a.get('profile') == 'calrel-6c3b9ec-prep':
                keep('rwprep', 'phase-A attempt: episode has no QA yet (phase B held by the MIG hold); Codex review3')
                continue
            if not str(a.get('profile', '')).startswith('release-'):
                keep('rwprep', f"profile {a.get('profile')} without a removal rule")
                continue
            if not ((a['status'] == 'ok' and d['code'] == 0) or (a['status'] == 'failed' and d['code'] != 0)):
                keep('rwprep', f"attempt {a['status']} with exit {d['code']} (not terminal and consistent)")
                continue
            ing_stages = ((ing['episodes'].get(key) or {}).get('preprocessing') or {}).get('stages') or {}
            raw = ep.get('stages') or {}
            st_stages = {k: v for k, v in raw.items() if not k.startswith('_') and isinstance(v, dict)}
            if (raw.get('_gate_missing') or (st_stages.get('qa') or {}).get('status') != 'done'
                    or (ing_stages.get('qa') or {}).get('status') != 'done'
                    or any(comparable(st_stages[s]) != comparable(ing_stages.get(s) or {}) for s in st_stages)):
                keep('rwprep', 'episode not fully processed + published (QA done, state.json == ING manifest)')
                continue
            if not (ts(a.get('launched_at')) and ts(d['created']) and
                    ts(a['launched_at']) - TOL <= ts(d['created']) <= ts(a['launched_at']) + 600):
                keep('rwprep', 'container not created within the attempt launch window')
                continue
            if d['args'] != ['-I', '-B', '/opt/rwprep/rwprep_job.py', f'{RW}/jobs/{job}/job-a{n}.json']:
                keep('rwprep', 'container argv does not name this attempt\'s job file')
                continue
            jobdir = f'{RW}/jobs/{job}'
            req = {f'{jobdir}/job-a{n}.json': {'job_id': job, 'attempt': n},
                   f'{jobdir}/a{n}/result.json': {'status': a['status'], 'job_id': job, 'attempt': n, 'container': name},
                   f'{jobdir}/a{n}/pipeline.log': None}
            if a['status'] == 'ok':  # the historical shared outputs root of the release-* attempts
                req[f"{RW}/processed/release-4fff862/{ep['task']}/{ep['episode_id']}/qa.json"] = {'__keys__': ['passed']}
            # the episode's PUBLISHED records must still match the remote completion records (Codex review4)
            eid, bad = ep['episode_id'], None
            for stage, rec in ing_stages.items():
                out_dir, met = rec.get('outputs') or '', rec.get('metrics') or {}
                if stage == 'qa':
                    if not out_dir.endswith('/qa.json'):
                        bad = 'published QA has no qa.json output'
                        break
                    req[out_dir] = {'passed': True}
                    continue
                if not (met.get('fingerprint') and met.get('build_id')):
                    continue  # records without a completion record of their own (e.g. validation markers)
                if stage in ('train_data', 'vis_geometry'):
                    comp = f'{out_dir}/_COMPLETE.json'
                elif stage == 'normalized':
                    comp = f'{out_dir}/normalized/_COMPLETE.json'
                else:
                    comp = f'{out_dir}/derived/{stage}/_COMPLETE.json'
                if not out_dir.startswith(f'{RW}/processed/') or eid not in out_dir:
                    bad = f'published {stage} output outside the processed roots'
                    break
                req[comp] = {'fingerprint': met['fingerprint'], 'build_id': met['build_id']}
            if bad:
                keep('rwprep', bad)
                continue
            sub = 'rwprep-superseded-ok' if a['status'] == 'ok' else 'rwprep-failed-retried'
            plan.append({**row, 'class': 'rwprep', 'subclass': sub, 'episode': key, 'job': job, 'attempt': n,
                         'attempt_status': a['status'], 'profile': a.get('profile'), 'required': req,
                         'keep_dir': f'{REMOTE}/runtime/container-archive/rwprep/{name}', 'format': RWPREP_FORMAT,
                         'expected': f"{cid}|exited|{d['code']}|rwprep|{job}|{n}|{key}|kaist-sunho|splatter4d|/{name}|true"})
            continue
        if l.get('s4d.managed') == 'true':
            run_id, att = l.get('s4d.run_id', ''), l.get('s4d.attempt', '')
            s = state.get(run_id, {})
            ident = (f"{cid}|exited|{d['code']}|true|{l.get('s4d.host', '')}|{run_id}|{att}|/{name}|true|"
                     f"{l.get('s4d.verification_purpose', '')}|{l.get('s4d.campaign_eligible', '')}")
            cls = 'verification' if l.get('s4d.verification_purpose') else 'job'
            if (l.get('s4d.host') != 'remote' or not att.isdigit() or name != f's4d-{run_id}-a{att}'
                    or (run_id, int(att)) not in recorded.get(cid, set())):
                keep(cls, 'labels / name / registry container id differ')
                continue
            if s.get('unknown') or s.get('status') in C.ACTIVE or s.get('status') in (None, C.PENDING):
                keep(cls, f"registry {s.get('status')}{' unknown' if s.get('unknown') else ''}")
                continue
            if str((s.get('attempt_codes') or {}).get(att)) != str(d['code']):
                keep(cls, 'exit code not recorded in the registry')
                continue
            if cls == 'verification':
                if s['status'] not in ('done', 'failed'):
                    keep(cls, f"verification {s['status']}")
                    continue
                m7 = re.match(r'verify-(?:native|isolation-\d+)-([0-9a-f]{7})(?:-cc)?$', run_id)
                c7 = m7.group(1) if m7 else None
                if s['status'] == 'done' and c7 not in ADOPTED:
                    keep(cls, f'release {c7} not adopted (no acceptance manifest + ADOPTED record): kept')
                    continue
                if s['status'] == 'failed':
                    hits = superseded(s.get('commit') or '', ts(d['finished']), ADOPTED) if s.get('commit') else []
                    if not hits:
                        keep(cls, 'failed suite not provably superseded by a later adopted descendant release')
                        continue
                    row['superseded_by_adopted'] = hits
                ev = f'{REMOTE}/runtime/verification/{run_id}/a{att}/evidence'
                req = ({f'{ev}/gpu_isolation.json': {'__keys__': ['passed', 'results']}}
                       if l.get('s4d.verification_purpose') == 'gpu_isolation'
                       else {f'{ev}/tests.json': {'__keys__': ['counts', 'exit_code', 'full_suite_passed']},
                             f'{ev}/tests.log': None})
                row['release'] = c7
                if s['status'] == 'done':  # bind this container's own evidence to the ADOPTED artifact (Codex review5)
                    rec = ADOPTED[c7]
                    if s.get('commit') != rec['commit']:
                        keep(cls, 'registry commit differs from the adopted acceptance manifest')
                        continue
                    if run_id not in rec['jobs']:
                        keep(cls, 'this verification job is not a source of the adopted acceptance artifacts')
                        continue
                    key = 'gpu_isolation.json' if l.get('s4d.verification_purpose') == 'gpu_isolation' else 'tests.json'
                    path = f'{ev}/{key}'
                    req[path] = {**req[path], '__canonical_in__': rec['jobs'][run_id]}
                    row['adoption'] = {k: rec[k] for k in ('commit', 'adopted_at', 'adopt_log', 'manifest')}
                plan.append({**row, 'class': 'verification', 'subclass': ('verification-adopted' if s['status'] == 'done'
                                                                            else 'verification-failed-superseded'),
                             'run_id': run_id, 'attempt': int(att), 'registry_status': s['status'], 'required': req,
                             'keep_dir': f'{REMOTE}/runtime/container-archive/{name}', 'format': _INSPECT_IDENTITY,
                             'expected': ident})
            else:
                if s['status'] == 'done':
                    keep(cls, 'done job: removed by the scheduler itself (its _cleanup_targets)')
                    continue
                if s['status'] not in ('failed', 'crashed') or d['code'] == 0:
                    keep(cls, f"job {s['status']} exit {d['code']}")
                    continue
                if any(stop.get('attempt') == int(att) for stop in s.get('intentional_stops', [])):
                    keep(cls, 'intentional checkpoint pause (scheduler handles)')
                    continue
                if C.protected_prefix(run_id) and not run_id.startswith('rwng-'):
                    keep(cls, f'protected family: {C.protected_prefix(run_id)}')
                    continue
                plan.append({**row, 'class': 'job_failed', 'subclass': 'job_failed', 'run_id': run_id,
                             'attempt': int(att), 'registry_status': s['status'],
                             'required': {f'{REMOTE}/runs/{run_id}/provenance.json': {'__keys__': []},
                                          f'{REMOTE}/runs/{run_id}/console.log': None},
                             'keep_dir': f'{REMOTE}/runs/{run_id}/containers', 'format': _INSPECT_IDENTITY,
                             'expected': ident})
            continue
        keep('other-ours', f"not scheduler-managed nor rwprep (role {l.get('s4d.role')}, agent {l.get('s4d.agent')}); "
                           'its owner agent has not acknowledged its outputs')
    return plan, kept


def confirm(plan, details, facts):
    """Split ``plan`` by the remote record contents (existence, JSON fields) and the time binding of rwprep."""
    ok, bad = [], []
    for p in plan:
        why = None
        for f, want in p['required'].items():
            exists, doc = facts.get(f, (False, None))
            if not exists:
                why = f'missing {f}'
            elif f.endswith('.json') and (doc or {}).get('unreadable'):
                why = f'unreadable / not a JSON object: {f}'
            elif want and not set(want.get('__keys__', ())) <= set((doc or {}).get('__keys__', ())):
                why = f'record lacks required keys: {f}'
            elif want and '__canonical_in__' in want and (doc or {}).get('__canonical__') not in want['__canonical_in__']:
                why = f'evidence {f} is not the report embedded in the adopted artifact'
            elif want and any((doc or {}).get(k) != v for k, v in want.items() if not k.startswith('__')):
                why = f'record differs: {f} { {k: (doc or {}).get(k) for k in want if not k.startswith("__")} }'
            if why:
                break
        if why is None and p['class'] == 'rwprep':
            res = facts[f"{RW}/jobs/{p['job']}/a{p['attempt']}/result.json"][1]
            d = details[p['id']]
            s0, s1, f0, f1 = ts(d['started']), ts(res.get('started_utc')), ts(d['finished']), ts(res.get('finished_utc'))
            if None in (s0, s1, f0, f1) or abs(s0 - s1) > TOL or abs(f0 - f1) > TOL:
                why = f"time binding fails (container {d['started']}..{d['finished']} vs result " \
                      f"{res.get('started_utc')}..{res.get('finished_utc')})"
            else:
                p['binding'] = {'result_container': res.get('container'), 'started_delta_s': round(s0 - s1, 2),
                                'finished_delta_s': round(f0 - f1, 2), 'argv_job_file': d['args'][-1]}
        (bad if why else ok).append((p, why))
    return [p for p, _ in ok], [{**{k: v for k, v in p.items() if k not in ('format', 'expected')}, 'kept': w}
                                for p, w in bad]


def removal_script(row):
    q = shlex.quote
    name, keep = row['name'], row['keep_dir']
    log, record, diff = f'{keep}/{name}.docker.log', f'{keep}/{name}.inspect.json', f'{keep}/{name}.diff.txt'
    receipt = f'{keep}/{name}.removal-receipt'
    checks = ''.join(f"test -s {q(p)} || {{ printf 'missing %s\\n' {q(p)} >&2; exit 4; }}\n" for p in sorted(row['required']))
    script = _REMOVAL_SCRIPT.replace('test -d @RUN_DIR@ || exit 4\n@REQUIRED@\n', checks)
    script = script.replace("id=@ID@; tmp_log=; tmp_record=", "id=@ID@; tmp_log=; tmp_record=; tmp_diff=")
    script = script.replace("""trap 'rm -f -- "$tmp_log" "$tmp_record"' EXIT""",
                            """trap 'rm -f -- "$tmp_log" "$tmp_record" "$tmp_diff"' EXIT""")
    script = script.replace('docker inspect "$id" > "$tmp_record" || exit 5\n',
                            'docker inspect "$id" > "$tmp_record" || exit 5\n'
                            'tmp_diff="$(mktemp @DIFF_TMP@)" || exit 5\ndocker diff "$id" > "$tmp_diff" || exit 5\n'
                            'digest "$tmp_diff"; diffsum="$d"\nmv -f -- "$tmp_diff" @DIFF@ || exit 5\ntmp_diff=\n')
    script = script.replace("""printf '%s %s %s\\n' "$id" "$logsum" "$inspectsum" > @RECEIPT_TMP@""",
                            """printf '%s %s %s %s\\n' "$id" "$logsum" "$inspectsum" "$diffsum" > @RECEIPT_TMP@""")
    script = script.replace('read -r rid rlog rinspect < @RECEIPT@', 'read -r rid rlog rinspect rdiff < @RECEIPT@')
    values = {'@ID@': q(row['id']), '@FORMAT@': q(row['format']), '@EXPECTED@': q(row['expected']),
              '@KEEP@': q(keep), '@LOCK@': q(f'{keep}/.{name}.lock'), '@LOG@': q(log), '@RECORD@': q(record),
              '@DIFF@': q(diff), '@LOG_TMP@': q(log + '.tmp.') + 'XXXXXX', '@RECORD_TMP@': q(record + '.tmp.') + 'XXXXXX',
              '@DIFF_TMP@': q(diff + '.tmp.') + 'XXXXXX', '@RECEIPT_TMP@': q(receipt + '.tmp'), '@RECEIPT@': q(receipt)}
    for k, v in values.items():
        script = script.replace(k, v)
    assert '@' not in re.sub(r"'[^']*'", '', script), 'unfilled placeholder'
    for needle in ('docker rm "$id"', 'check\ndocker rm', '"$diffsum" > ', 'rinspect rdiff', 'docker diff "$id"', 'test -s '):
        assert needle in script, needle
    assert ' -f "$id"' not in script and '--force' not in script and ' -v ' not in script and 'prune' not in script
    return script


def plan_all():
    rows, details = inventory()
    state, recorded = C.read_registry(), registry_ids()
    with shared_locks() as (pre, ing):
        plan, kept = classify(details, state, pre, ing, recorded)
    plan, more = confirm(plan, details, remote_facts([f for p in plan for f in p['required']]))
    return rows, details, plan, kept + more


def summarize(rows, details, plan, kept):
    others = {r[1]: r[2] for r in rows if not r[1].startswith('s4d-')}
    by = {}
    for p in plan:
        by.setdefault(p.get('subclass') or p['class'], {'remove': 0, 'keep': {}})['remove'] += 1
    for k in kept:
        b = by.setdefault(k.get('subclass') or k['class'], {'remove': 0, 'keep': {}})
        why = k['kept'] if len(k['kept']) < 120 else k['kept'][:117] + '...'
        b['keep'][why] = b['keep'].get(why, 0) + 1
    return {'containers_total': len(rows), 'ours_s4d': len(details), 'other_users_listed_only': others, 'by_class': by}


def apply(plan):
    allow = {p['id'] for p in json.loads(REVIEWED.read_text())['plan']}
    ids = [p['id'] for p in plan if p['id'] in allow]
    results, stop, log = [], None, (C.OUT / 'executed-actions.jsonl').open('a')
    recorded = registry_ids()
    for i in range(0, len(ids), 8):
        if stop:
            break
        batch_ids, batch_results = ids[i:i + 8], []
        details = inspect(batch_ids)  # fresh identity, state and times
        state = C.read_registry()
        with shared_locks() as (pre, ing):  # no dispatcher tick / ING write while this batch is decided and removed
            fresh, _ = classify(details, state, pre, ing, recorded)
            fresh, _ = confirm(fresh, details, remote_facts([f for p in fresh for f in p['required']]))
            fresh = [p for p in fresh if p['id'] in allow]
            for cid in batch_ids:
                if cid not in {p['id'] for p in fresh}:
                    batch_results.append({'id': cid, 'skipped': 'fresh recheck failed / not eligible now'})
            locks = {}
            try:
                todo = []
                for p in fresh:
                    if p['class'] == 'job_failed':  # the scheduler's per-run result-sync lock, never blocking; one per run
                        if p['run_id'] not in locks:
                            fh = (CLIENT.state_dir / f"results-{p['run_id']}.lock").open('a')
                            try:
                                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                fh.close()
                                fh = None
                            locks[p['run_id']] = fh
                        if locks[p['run_id']] is None:
                            batch_results.append({'id': p['id'], 'name': p['name'], 'skipped': 'result sync in progress'})
                            continue
                    todo.append(p)
                if todo:
                    body = ''.join(f"( {removal_script(p)} ); rc=$?; echo \"__RC__ {p['id']} $rc\"; "
                                   f"[ $rc -eq 0 ] || exit 90\n" for p in todo)
                    r = ssh(body, timeout=900, idempotent=False)
                    rcs = {}
                    payload = {}
                    last = None
                    for line in r.stdout.splitlines():
                        if line.startswith('{'):
                            last = line
                        elif line.startswith('__RC__ '):
                            _, cid, rc = line.split()
                            rcs[cid] = rc
                            payload[cid] = json.loads(last) if last else None
                            last = None
                    for p in todo:
                        res = {'id': p['id'], 'name': p['name'], 'class': p['subclass'], 'rc': rcs.get(p['id']),
                               'result': payload.get(p['id']), 'archive': p['keep_dir'], 'binding': p.get('binding')}
                        if rcs.get(p['id']) not in ('0', None):
                            res['stderr'] = r.stderr[-300:]
                        batch_results.append(res)
                    if r.returncode or any(rcs.get(p['id']) != '0' for p in todo):
                        stop = f'batch exit {r.returncode}; rcs {rcs}'
            finally:
                for fh in locks.values():
                    if fh is not None:
                        fh.close()
        for res in batch_results:
            log.write(json.dumps({'time': time.time(), 'host': 'remote',
                                  'action': 'container_rm' if res.get('rc') == '0' else 'container_skip', **res}) + '\n')
        log.flush()
        results += batch_results
    log.close()
    after = ssh("docker ps -a --no-trunc --format '{{.ID}}|{{.Names}}|{{.State}}' && echo __END__")
    if after.returncode or not after.stdout.rstrip().endswith('__END__'):
        raise SystemExit('final listing failed; results unverified')
    remaining = {line.split('|')[0] for line in after.stdout.split('__END__')[0].strip().splitlines()}
    contradictions = []
    for res in results:
        if res.get('rc') == '0':
            res['verified_absent'] = res['id'] not in remaining
            if not res['verified_absent'] or not (res.get('result') or {}).get('removed', (res.get('result') or {}).get('absent')):
                contradictions.append(res['id'])
    return results, stop, remaining, contradictions


ADOPTED.update(adoption_records())
rows, details, plan, kept = plan_all()
record = {'recorded_local': datetime.now().astimezone().isoformat(), 'tool': __file__,
          'summary': summarize(rows, details, plan, kept), 'plan': plan, 'kept': kept}
print(json.dumps(record['summary'], indent=1))
for p in plan:
    removal_script(p)  # every script is built and checked before anything is removed
if not APPLY:
    OUT.write_text(json.dumps(record, indent=1) + '\n')
    raise SystemExit(0)
results, stop, remaining, contradictions = apply(plan)
record.update(applied_local=datetime.now().astimezone().isoformat(), results=results, stopped=stop,
              removed_verified=sum(1 for r in results if r.get('verified_absent')), containers_after=len(remaining),
              contradictions=contradictions)
OUT.with_name('remote-container-inventory.applied.json').write_text(json.dumps(record, indent=1) + '\n')
print(f"removed (verified absent) {record['removed_verified']}; stopped: {stop}; contradictions: {len(contradictions)}; "
      f"containers now {len(remaining)}")
if stop or contradictions:
    raise SystemExit(2)
