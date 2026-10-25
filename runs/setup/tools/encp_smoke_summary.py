"""Summary of the encoder-pilot smoke runs on both hosts (user directive 2026-10-25: smoke runs per arm on both hosts'
images, s/it and peak memory). Local: runs/setup/encp-smoke-local-20261025/<arm>.json (enc_smoke_local.py, the main
.venv, one MIG 1g.24gb slice each, three slices in parallel). Remote: runs/encsmoke-<arm>-hammer (queue rows on release
d281258 / derived image, full GPUs shared with other campaign containers; the jobs on the same GPU at launch are
listed). Median logged s/it over steps > 100 (train/step_time, every 50 steps), peak torch.cuda.max_memory_allocated.
Writes runs/setup/encp-smoke-20261025.json.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

ARMS = ('chlrm', 'hn', 'cs', 'h2', 'w4', 'w8')
events = [json.loads(line) for line in jobs.REGISTRY.read_text().splitlines() if line.strip()]
out = {'local': {}, 'remote': {}}
for arm in ARMS:
    p = ROOT / f'runs/setup/encp-smoke-local-20261025/{arm}.json'
    if p.is_file():
        d = json.loads(p.read_text())
        out['local'][arm] = {k: d[k] for k in ('exit_code', 'median_step_time_s', 'median_gpu_time_s', 'peak_gpu_mem_gb',
                                               'peak_session_pss_bytes', 'finite', 'gpu')}
    jid = f'encsmoke-{arm}-hammer'
    run = ROOT / 'runs' / jid
    if not (run / 'metrics.jsonl').is_file():
        continue
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    train = [r for r in rows if 'train/step_time' in r and r.get('step', 0) > 100]
    launch = [e for e in events if e.get('id') == jid and e.get('event') == 'launched'][-1]
    t = launch['time']
    active, ends = {}, {}
    for e in events:  # jobs launched on the same GPU before this launch and not yet exited then
        if e.get('host') == 'remote' and e.get('gpu') == launch['gpu'] and e.get('event') == 'launched' and e['time'] < t:
            active[e['id']] = e['time']
        if e.get('event') in ('remote_exited', 'exited', 'results_synced') and e.get('id') in active and e['time'] < t:
            ends[e['id']] = e['time']
    out['remote'][jid] = {
        'gpu': launch['gpu'], 'commit': launch.get('commit'), 'image_digest': launch.get('image_digest'),
        'median_step_time_s': statistics.median(r['train/step_time'] for r in train) if train else None,
        'median_gpu_time_s': statistics.median(r['train/gpu_time'] for r in train if 'train/gpu_time' in r) if train else None,
        'peak_gpu_mem_gb': max((r['train/gpu_mem_gb'] for r in rows if 'train/gpu_mem_gb' in r), default=None),
        'finite': all(v == v for r in train for k, v in r.items() if k.startswith('loss/')),
        'completion': json.loads((run / 'completion.json').read_text()) if (run / 'completion.json').is_file() else None,
        'same_gpu_at_launch': sorted(i for i in active if i not in ends and i != jid),
    }
(ROOT / 'runs/setup/encp-smoke-20261025.json').write_text(json.dumps(out, indent=1) + '\n')
for host, recs in out.items():
    for k, v in recs.items():
        print(host, k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()
                        if x in ('median_step_time_s', 'peak_gpu_mem_gb', 'finite', 'gpu', 'exit_code')},
              len(v.get('same_gpu_at_launch', [])) if host == 'remote' else '')
