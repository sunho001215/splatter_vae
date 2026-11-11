"""Combine the per-tool dry-run manifests of the 2026-11-11 cleanup into the directive's two manifests
/home/ws/ws/realworld-ops/cleanup-20261111/{local,remote}-manifest.json (path, bytes, owner uid / label, category,
reason, evidence per entry) and summary.json. Reads only the tool outputs; deletes nothing.

    .venv/bin/python -I runs/setup/tools/disk_cleanup_manifests_20261111.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_cleanup_common_20261111 as C  # noqa: E402

O = C.OUT
stage0 = json.loads((O / 'local-stage0-manifest.json').read_text())
lmodels = json.loads((O / 'local-models-manifest.json').read_text())
rmodels = json.loads((O / 'remote-models-manifest.json').read_text())
conts = json.loads((O / 'remote-containers.json').read_text())


def entry(c, host):
    return {'host': host, 'path': c['path'], 'bytes': c['bytes'], 'owner_uid': c.get('owner_uid'),
            'category': c['category'], 'run': c.get('run'), 'reason': c['reason'], 'evidence': c.get('evidence', {})}


local = [entry(c, 'local') for c in stage0['candidates'] + lmodels['candidates']]
remote = [entry(c, 'remote') for c in rmodels['candidates']]
for p in conts['plan']:
    remote.append({'host': 'remote', 'path': f"docker container {p['name']} ({p['id'][:12]})", 'bytes': 32768,
                   'owner_uid': None, 'owner_label': {k: v for k, v in p['labels'].items() if k.startswith('s4d.')},
                   'category': 'container_stopped_ours',
                   'reason': f"exited {p['kind']} container of {p['run_id']} (registry {p['registry_status']}); "
                             'docker log + inspect + diff archived with a receipt first, then docker rm <id>',
                   'evidence': {'identity': p['identity'], 'registry_status': p['registry_status'],
                                'finished': p.get('finished'), 'archive_dir': p['keep_dir'],
                                'writable_layer': '~31 kB (read-only rootfs; 11-06 measurement)'}})


def totals(rows):
    out = {}
    for r in rows:
        t = out.setdefault(r['category'], {'items': 0, 'bytes': 0})
        t['items'] += 1
        t['bytes'] += r['bytes']
    return out


for name, rows, extra in (('local', local, {'free_bytes_before': stage0['free_bytes_before']}),
                          ('remote', remote, {'free_bytes_before': rmodels['free_bytes_before']})):
    (O / f'{name}-manifest.json').write_text(json.dumps({'recorded_local': C.now_iso(), 'host': name, **extra,
                                                         'dry_run': True, 'totals': totals(rows), 'entries': rows},
                                                        indent=1) + '\n')


def kept(m):
    return {k: {'files': v['files'], 'GB': round(v['bytes'] / 1e9, 2)} for k, v in
            sorted(m['kept_by_reason'].items(), key=lambda x: -x[1]['bytes'])}


summary = {
    'recorded_local': C.now_iso(), 'directive': 'user 2026-11-11 ~17:30 (memory disk-cleanup-directive-20261111.md)',
    'tools': ['runs/setup/tools/disk_cleanup_common_20261111.py', 'runs/setup/tools/disk_cleanup_stage0_20261111.py',
              'runs/setup/tools/disk_cleanup_local_models_20261111.py', 'runs/setup/tools/disk_cleanup_remote_20261111.py',
              'runs/setup/tools/remote_container_cleanup_20261111.py'],
    'local': {'free_bytes_before': stage0['free_bytes_before'], 'candidates': totals(local),
              'retained_partials': stage0['retained_partials_by_reason'], 'retained_pytest_trees': stage0['retained_trees'],
              'models_kept_by_reason': kept(lmodels)},
    'remote': {'free_bytes_before': rmodels['free_bytes_before'], 'candidates': totals(remote),
               'release_pins': rmodels['release_pins'], 'models_kept_by_reason': kept(rmodels),
               'verification_tmp_kept': [{'rel': t['rel'], 'GB': round(t['bytes'] / 1e9, 2), 'why': t['kept']}
                                         for t in rmodels['verification_tmp_kept']],
               'containers': {'total_on_host': conts['containers_total'], 'ours_stopped_inspected':
                              conts['ours_stopped_inspected'], 'running_ours': conts['running_ours'],
                              'plan': len(conts['plan']), 'kept': {}, 'not_ours_listed_only': conts['not_ours_listed_only']}},
    'not_in_scope_or_uncertain_untouched': [
        'replay buffers (owned by the scripts/cleanup_replay.py loop; its remaining files belong to paused/running/crashed runs)',
        'Arm4 (ocsw4/ochf/c50dg/c50vis/ana-arm4) and its named references cst50/cst75/chl/s3-/i2-m3d: under analysis',
        'paused / stopped families (ds-*, enc-abl-, sr-csw4-, s3spec-): pause records keep checkpoints/replay',
        'all real-world runs, FQL diagnostics, CNN / no-crop, serving / robot-trial checkpoints, rwprep containers, /data_8T',
        'local copies of remote runs\' intermediate checkpoints (may be the only copies after 10-25 / 11-06)',
        'worktree .cache copies (other agents\' worktrees)', 'W&B directories (local and remote)',
        'remote releases/, realworld/processed, data/, runtime/cache/tmp', 'validation-20261031, realworld-20261104',
        'RL 100k-multiple snapshots and pretraining 100k milestones / finals / latest.pt / exports'],
}
for k in conts['kept']:
    summary['remote']['containers']['kept'][k['kept']] = summary['remote']['containers']['kept'].get(k['kept'], 0) + 1
(O / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps({'local': summary['local']['candidates'], 'remote': summary['remote']['candidates']}, indent=1))
