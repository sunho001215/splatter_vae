"""Preview local phase-A scheduler rows. --stage writes new jobs; --execute appends held rows.

The coordinator uses the established realworld_local place helper, then releases
only these exact IDs. Neither mode launches work or changes any existing row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 's4d').is_dir())
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from ops.jobs import local_phase_a as a  # noqa: E402
import register_ours_ext_20261113 as reviewed  # noqa: E402

PAUSE = 'dsreset-local-phaseA-await-placement-20261115'
TOOL = 'runs/setup/tools/register_local_phase_a_20261115.py'


def jobs(plans, bundle):
    seen = set()
    for plan in plans:
        for row in json.loads(Path(plan).read_text()):
            if 'config' not in row:
                continue
            a.require(row['episode'] not in seen, 'duplicate episode in plans')
            seen.add(row['episode'])
            yield a.make_job(row, bundle)


def build_rows(payloads, bundle, release, gate_sha, footprints):
    rows = []
    for payload in payloads:
        ep = payload['episode']
        identity = f'rwv2-20261113-prep-ours-{ep}-localA'
        path = Path(bundle) / 'jobs' / f'{ep}.json'
        serialized = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        import hashlib
        checksum = hashlib.sha256(serialized).hexdigest()
        rows.append(dict(id=identity, host='remote', gpu='any', gpu_profile='1g.24gb', role='realworld_preprocess',
                         priority=-4, script=a.SCRIPT, args=['--bundle', str(bundle), '--job', str(path),
                         '--job-sha256', checksum, '--gate-sha256', gate_sha], deps=[], max_restarts=0,
                         mem_gb=footprints.get('gpu_declaration_gib', 16), gpu_mem_gb=footprints.get('gpu_declaration_gib', 16),
                         ram_gb=footprints.get('ram_declaration_gib', 48), cpu_threads=4, disk_gb=32,
                         measured_footprints=footprints, planned_commit=release['commit'], planned_image=release['image'],
                         evidence_manifest=release['evidence_manifest'], release_source='local',
                         local_placement_pause=PAUSE, registration_tool=TOOL,
                         required_results=['completion.json'], realworld_target=a.identity(ep)[0]))
    return rows


def publish(root, rows, release, bundle, gate_sha):
    a.require(root.resolve() == ROOT == a.MAIN and Path.cwd().resolve() == a.MAIN, 'execute only from adopted main')
    a.require(a.git(root, 'symbolic-ref', '--short', 'HEAD') == 'main', 'execute requires main')
    with reviewed.locks(root):
        a.require(reviewed.local_release_pin(root, release['commit'], release['image']) == release, 'release changed')
        a.require_gate(bundle, gate_sha)
        from ops.jobs import host_roles, realworld_local
        from ops.jobs.scheduler import read_registry
        current = reviewed.queue_document((root / 'experiments/queue.yaml').read_bytes())
        existing = {r['id'] for r in current['jobs']}
        registry = read_registry(root / 'experiments/registry.jsonl')
        owned = host_roles.read_ownership()
        remote = root/'runs/dsreset-phaseA-remote-20261114/state'
        a.require((remote/'STOP').exists(), 'drain R1 and disable its automatic preparation before local registration')
        statepath = remote/'state.json'
        if statepath.exists():
            state = json.loads(statepath.read_text())
            a.require(not any(v.get('status') in ('running', 'launching', 'unknown', 'uncertain')
                              for v in state.get('jobs', state).values() if isinstance(v, dict)),
                      'R1 has unresolved execution; reconcile before local fallback')
        for row in rows:
            a.require(row['id'] not in existing | set(registry) | set(owned), 'row already registered/launched/owned')
            a.require(not (root / 'runs' / row['id']).exists(), 'row output already exists')
            a.scheduler_inputs(row, root)
            realworld_local.placement_scope(row, root=root)
            from ops.jobs.migration import execution_scope
            snapshot = json.loads((Path(bundle)/'envelope.json').read_text())['commit']
            a.require(execution_scope([a.SCRIPT], snapshot, root)['identical'], 'snapshot execution scope differs from adopted release')
        pausepath = root / 'runs/setup/admission-pauses.json'
        pauses = json.loads(pausepath.read_text())
        a.require(not any(h['name'] == PAUSE for h in pauses['pauses']), 'placement hold exists; inspect prior transaction')
        hold = reviewed.local_hold(rows, PAUSE)
        before = (root / 'experiments/queue.yaml').read_bytes()
        path = root / 'runs/setup/rw-registration-local-phase-a-20261115.json'
        a.require(not path.exists(), 'registration proof already exists')
        # Durable exact-ID hold first; a crash cannot accidentally admit new remote rows.
        pauses['pauses'].append(hold)
        a.atomic(pausepath, pauses)
        candidate = reviewed.queue_candidate(before, rows)
        reviewed.atomic_write(root / 'experiments/queue.yaml', candidate)
        a.atomic(path, {'rows': rows, 'release': release, 'equivalence_sha256': gate_sha,
                        'hold': hold, 'queue_before_sha256': reviewed.sha(before),
                        'queue_after_sha256': a.sha(root / 'experiments/queue.yaml'), 'launched': False})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--plan', action='append', type=Path, required=True)
    p.add_argument('--stage', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    bundle = args.bundle.resolve()
    envelope = a.verify_envelope(bundle, a.sha(bundle / 'envelope.json'))
    gatepath = bundle / 'equivalence.json'
    gate_sha = a.sha(gatepath) if gatepath.exists() else 'GATE-NOT-RUN'
    qualified = False
    footprints = {}
    try:
        gate = a.require_gate(bundle, gate_sha)
        qualified = True
        footprints = gate['footprints']
    except (OSError, ValueError, KeyError):
        pass
    payloads = list(jobs(args.plan, bundle))
    role = json.loads((a.MAIN / 'runs/setup/role-release.json').read_text())
    release = dict(commit=a.git(a.MAIN, 'rev-parse', 'HEAD') if args.execute else envelope['commit'], image=role['image'], source='local', evidence_manifest='docs/tests.json (local evidence)')
    rows = build_rows(payloads, bundle, release, gate_sha, footprints)
    if args.stage:
        for payload in payloads:
            path = bundle / 'jobs' / f"{payload['episode']}.json"
            a.require(not path.exists(), 'job exists; do not replace immutable execution inputs')
            a.atomic(path, payload)
            path.chmod(0o444)
    if args.execute:
        a.require(qualified and footprints.get('measured') is True, 'full gate and measured footprints required')
        publish(ROOT, rows, release, bundle, gate_sha)
    print(json.dumps({'qualified': qualified, 'rows': rows, 'registered': len(rows) if args.execute else 0,
                      'staged': args.stage, 'launched': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
