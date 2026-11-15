"""Qualified local three-stage phase A and hash-bound SAM3-only handoff.

The four-stage gate remains unchanged. This separate authorization never permits
local SAM3. Published stage build IDs survive relocation and feed remote SAM3/B.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import yaml

from ops.jobs import local_phase_a as a

MODE = 'qualified-local-three/1'
STAGES = a.STAGES[:3]
SUFFIX = 'hybridA'
REPORT = 'hybrid-equivalence.json'
LEDGERS = (a.MAIN/'runs/setup/remote-ledger/latest.json',
           a.MAIN/'runs/setup/remote-ledger/pre-outage-ledger.json',
           Path('/home/ws/ws/realworld-ops/pre/state.json'))
FOOTPRINTS = dict(measured=True, gpu_declaration_gib=8, ram_declaration_gib=16,
                  cpu_threads=4, disk_declaration_gib=32)


def make_job(row, bundle):
    return {**a.make_job(row, bundle), 'mode': MODE, 'stages': list(STAGES),
            'remote_remaining_stages': ['sam3_masks'],
            'dedup_audit_sha256': a.sha(Path(bundle)/'dedup-audit.json')}


def is_hybrid(job):
    if 'mode' not in job:
        return False
    a.require(job['mode'] == MODE and job.get('stages') == list(STAGES)
              and job.get('remote_remaining_stages') == ['sam3_masks'], 'invalid hybrid stage contract')
    return True


def suffix(job):
    return SUFFIX if is_hybrid(job) else 'localA'


def stage_evidence(row):
    return {p: h for p, h in row['evidence_files'].items()
            if '/normalized/' in p or any('/derived/'+s+'/' in p for s in STAGES[1:])}


def qualify(bundle, reference):
    """Reuse measured comparisons, never reinterpret SAM failure as a full pass."""
    from ops.jobs import local_phase_a_gate as g
    bundle, reference = Path(bundle), Path(reference).resolve()
    old = json.loads(reference.read_text())
    a.require(old['coverage'] == 'full-two-episodes' and old['tolerances'] == g.TOLERANCES,
              'qualification coverage/tolerances differ')
    a.require(len(old['episodes']) == 2 and {r['task'] for r in old['episodes']} == {'covering', 'pick-and-place'},
              'both reference episodes required')
    original = reference.parent
    oldenv = json.loads((original/'envelope.json').read_text())
    a.require(a.sha(original/'envelope.json') == old['envelope_sha256'], 'reference envelope drift')
    env = json.loads((bundle/'envelope.json').read_text())
    # Computation, adapters and workers must be the actually qualified bytes.
    science = {p: h for p, h in oldenv['files'].items() if p.startswith('source/s4d/')}
    a.require(science and all(env['files'].get(p) == h for p, h in science.items()), 'qualified science source differs')
    def comparator(path):
        return [ast.dump(n) for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name in
                ('metrics_depth', 'metrics_masks', 'compare_instances', 'exact_h5', 'compare_episode')]
    a.require(comparator(Path(oldenv['frozen'])/'source/ops/jobs/local_phase_a_gate.py') == comparator(g.__file__),
              'qualification comparator differs')
    rows = []
    for row in old['episodes']:
        a.require(row['label'] == 'success' and all(row['stages'][s]['passed'] is True and
                  row['stages'][s]['files'] for s in STAGES), 'unqualified local stage')
        evidence = stage_evidence(row)
        a.require(evidence and all(a.sha(p) == h for p, h in evidence.items()), 'qualified evidence drift')
        rows.append(dict(task=row['task'], episode=row['episode'], label='success', passed=True,
                         stages={s: row['stages'][s] for s in STAGES}, evidence_files=evidence))
    # The original three-stage run (not the later SAM-only resume) measured FS.
    measurement_files = {}
    measured = []
    for row in old['episodes']:
        path = Path(row['candidate']).parents[1]/'local-phase-a.json'
        proof = json.loads(path.read_text())
        if proof.get('stages') == ['sam3_masks']:
            # Stage comparisons contain the original FS completion record. Its root
            # retains the original three-stage resource proof in the initial bundle.
            files = row['evidence_files']
            paths = [Path(p) for p in files if '/derived/foundation_stereo_depth/' in p and p.endswith('_COMPLETE.json')]
            rec = json.loads(paths[-1].read_text())
            checkpoint = Path(rec['config']['worker_identity']['checkpoint'])
            path = checkpoint.parents[3].parent/'gate'/row['task']/'local-phase-a.json'
            proof = json.loads(path.read_text())
        fp = proof['footprints']
        a.require(fp['measured'] and fp['gpu_peak_reserved_gib'] <= 8 and fp['rss_peak_gib'] < 16,
                  'local stage footprint exceeds declaration')
        measurement_files[str(path)] = a.sha(path)
        measured.append(fp)
    result = dict(schema=MODE, qualified=True, coverage='two-episodes-three-stages', stages=list(STAGES),
                  envelope_sha256=a.sha(bundle/'envelope.json'), reference_report=str(reference),
                  reference_report_sha256=a.sha(reference), reference_qualified=old['qualified'],
                  tolerances=g.TOLERANCES, episodes=rows, measurement_files=measurement_files,
                  footprints={**FOOTPRINTS,
                    'gpu_peak_reserved_gib': max(f['gpu_peak_reserved_gib'] for f in measured),
                    'rss_peak_gib': max(f['rss_peak_gib'] for f in measured)})
    a.atomic(bundle/REPORT, result)
    return result


def require_gate(bundle, expected):
    from ops.jobs.local_phase_a_gate import TOLERANCES
    bundle = Path(bundle)
    path = bundle/REPORT
    a.require(a.sha(path) == expected, 'hybrid gate drift')
    d = json.loads(path.read_text())
    a.require(d.get('schema') == MODE and d.get('qualified') is True and d.get('stages') == list(STAGES)
              and d.get('coverage') == 'two-episodes-three-stages', 'three-stage gate required')
    a.require(d['envelope_sha256'] == a.sha(bundle/'envelope.json') and d['tolerances'] == TOLERANCES,
              'hybrid envelope/tolerances differ')
    a.require(a.sha(d['reference_report']) == d['reference_report_sha256'], 'reference report drift')
    a.require(len(d['episodes']) == 2 and {r['task'] for r in d['episodes']} == {'covering', 'pick-and-place'},
              'both qualified episodes required')
    for row in d['episodes']:
        a.require(row['passed'] is True and row['label'] == 'success' and set(row['stages']) == set(STAGES),
                  'invalid hybrid qualification')
        a.require(all(s['passed'] is True and s['files'] for s in row['stages'].values()), 'local stage failed')
        a.require(row['evidence_files'] and all(a.sha(p) == h for p, h in row['evidence_files'].items()),
                  'hybrid evidence drift')
    a.require(d['footprints']['gpu_declaration_gib'] == 8 and d['footprints']['ram_declaration_gib'] == 16,
              'hybrid resource declarations differ')
    a.require(d['measurement_files'] and all(a.sha(p) == h for p, h in d['measurement_files'].items()),
              'measurement drift')
    a.require(d['footprints']['measured'] is True, 'measured local footprints required')
    return d


def audit_plans(plans, ledgers=LEDGERS):
    """Explicit reuse wins; completed or unresolved matching raw SHA never reruns."""
    rows, reused, seen = [], [], set()
    evidence = {}
    records = []
    for path in ledgers:
        path = Path(path)
        a.require(path.is_file(), f'missing dedup ledger: {path}')
        evidence[str(path)] = a.sha(path)
        d = json.loads(path.read_text())
        def collect(value, key=''):
            if isinstance(value, dict):
                records.append({**value, '_ledger_key': key})
                for name, child in value.items():
                    if isinstance(child, (dict, list)):
                        collect(child, str(name))
            elif isinstance(value, list):
                for child in value:
                    collect(child, key)
        collect(d.get('jobs', d.get('episodes', {})))
    for plan in plans:
        plan = Path(plan).resolve()
        evidence[str(plan)] = a.sha(plan)
        for row in json.loads(plan.read_text()):
            a.identity(row['episode'])
            a.require(row['episode'] not in seen, 'duplicate episode in canonical plans')
            seen.add(row['episode'])
            a.require('raw_sha256' in row, 'unbound raw identity')
            if 'config' not in row:
                a.require(row.get('reuse'), 'missing explicit success reuse')
                reused.append(row)
                continue
            matches = [r for r in records if row['raw_sha256'] in
                       (r.get('raw_sha256'), r.get('input_sha256'), (r.get('inputs') or {}).get('raw_sha256'))
                       or row['episode'] == r.get('episode') or row['episode'] in r.get('_ledger_key', '')
                       or row['raw_sha256'] in (r.get('inputs') or {}).values()]
            a.require(not matches, f"remote ledger already owns/completed {row['episode']}; reconcile before staging")
            evidence[row['config']] = a.sha(row['config'])
            rows.append(row)
    return rows, dict(schema=MODE, eligible=len(rows), reused=reused, evidence_files=evidence,
                      note='pre-outage snapshots; no remote reachability or freshness claim')


def finish(root, job):
    """Durable SHA closure written only after all three ordinary records exist."""
    root = Path(root)
    ep = job['episode']
    for s in STAGES:
        tail = Path('normalized') if s == 'ingest' else Path('derived')/s
        rec = root/'episodes'/ep/tail/'_COMPLETE.json'
        a.require(rec.is_file(), f'missing qualified completion: {s}')
    a.require(not (root/'episodes'/ep/'derived/sam3_masks').exists(), 'hybrid output contains local SAM3')
    files = a.inventory(root/'episodes'/ep)
    result = dict(schema=MODE, episode=ep, raw_sha256=job['raw_sha256'], stages=list(STAGES),
                  remote_remaining_stages=['sam3_masks'], envelope_sha256=job['envelope_sha256'],
                  files={f'episodes/{ep}/'+p: h for p, h in files.items()})
    a.atomic(root/'hybrid-stage-manifest.json', result)
    return result


def verify_output(root, episode):
    root = Path(root)
    d = json.loads((root/'hybrid-stage-manifest.json').read_text())
    a.require(d['schema'] == MODE and d['episode'] == episode and d['stages'] == list(STAGES)
              and d['remote_remaining_stages'] == ['sam3_masks'], 'invalid hybrid import')
    actual = {f'episodes/{episode}/'+p: h for p, h in a.inventory(root/'episodes'/episode).items()}
    a.require(actual == d['files'], 'hybrid payload drift/inventory differs')
    return d


def remote_pipeline(job):
    from ops.jobs.rwprep_remote import rewrite, RW, RAW
    cfg = a.execution_pipeline(job)
    for src, dst in ((str(a.PROCESSED), a.REMOTE), ('/home/ws/data/real_world', RAW),
                     (str(a.ASSETS), RW+'/assets/validation-20261031')):
        cfg = rewrite(cfg, src, dst)
    return cfg


def remote_jobs(plans, out, processed, raw):
    """Compile on the recovered reference AFTER verified import; only SAM3 runs."""
    from scripts.realworld_prep_register import compile_plans
    for plan in plans:
        for row in json.loads(Path(plan).read_text()):
            if 'config' in row:
                root = Path(processed).joinpath(*a.identity(row['episode']))
                manifest = verify_output(root, row['episode'])
                a.require(manifest['raw_sha256'] == row['raw_sha256'], 'import/raw mismatch')
    jobs = compile_plans(plans, out, processed, raw)
    for job in jobs:
        root = Path(processed).joinpath(*a.identity(job['episode']))
        job['id'] = job['id'].removesuffix('-phaseA')+'-remoteSAM3'
        job['steps'][0]['argv'][-1] = 'sam3_masks'
        manifest = root/'hybrid-stage-manifest.json'
        job['inputs'][str(manifest)] = a.sha(manifest)
        # Existing admission verifies every imported payload, including completion
        # records and normalized contract, before launch (not merely a marker).
        job['inputs'].update({str(root/p): h for p, h in verify_output(root, job['episode'])['files'].items()})
    a.atomic(Path(out)/'jobs.json', jobs)
    return jobs


def portable_payloads(source, episode):
    """Relocate contract paths, retain original records/build IDs and SHA provenance.

    Remote jobs explicitly request SAM3 only. A later full phase-A command would
    intentionally invalidate path-bound ingest/worker fingerprints; do not use it.
    """
    from ops.jobs.rwprep_remote import rewrite, RW, RAW
    source = Path(source)
    original = verify_output(source, episode)
    payloads = {p: source/p for p in a.inventory(source) if p not in ('.local-phase-a.lock', '.publication.json')}
    def relocate(value):
        for src, dst in ((str(a.PROCESSED), a.REMOTE), ('/home/ws/data/real_world', RAW),
                         (str(a.ASSETS), RW+'/assets/validation-20261031')):
            value = rewrite(value, src, dst)
        return value
    contract = f'episodes/{episode}/normalized/episode.json'
    data = relocate(json.loads((source/contract).read_text()))
    payloads[contract] = (json.dumps(data, indent=2, sort_keys=True)+'\n').encode()
    cfg = relocate(yaml.safe_load((source/'pipeline.yaml').read_text()))
    payloads['pipeline.yaml'] = yaml.safe_dump(cfg, sort_keys=False).encode()
    import hashlib
    digest = lambda v: hashlib.sha256(v).hexdigest() if isinstance(v, bytes) else a.sha(v)
    manifest = copy.deepcopy(original)
    manifest['files'] = {p: digest(v) for p, v in payloads.items() if p.startswith(f'episodes/{episode}/')}
    manifest['relocation'] = dict(original_manifest_sha256=a.sha(source/'hybrid-stage-manifest.json'),
                                  original_files=original['files'], fields='normalized contract paths; pipeline paths',
                                  completion_records='unchanged, including build IDs')
    payloads['hybrid-stage-manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    return payloads, {p: digest(v) for p, v in payloads.items()}
