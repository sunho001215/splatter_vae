"""Device provenance of every local launch in experiments/registry.jsonl: host, MIG instance and parent physical GPU.

The MIG -> parent mapping is the union of every local layout the campaign used: the 2026-10-16 layout (MIG
instances of GPUs 4/5/6) and the 2026-10-19 allowlist (GPUs 4/5 only; runs/setup/gpu-allowlist-20261019.json).
Pre-MIG launches name a physical GPU UUID, which is its own parent. Launch events written after the 2026-10-19 merge
carry `gpu_parent` themselves; this index is checked against it. Output: runs/setup/local-device-provenance.json.

    python -I local_device_provenance.py
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
manifest = json.loads((ROOT / 'runs/setup/gpu-allowlist-20261019.json').read_text())
PARENT = {mig: parent for layout in (manifest['authorized_local'], manifest['revoked_local'])
          for parent, migs in layout.items() for mig in migs}
AUTHORIZED = {mig for migs in manifest['authorized_local'].values() for mig in migs}
CHANGE = datetime.fromisoformat(manifest['recorded_local']).timestamp()

runs, unknown = {}, []
for line in (ROOT / 'experiments/registry.jsonl').read_text().splitlines():
    event = json.loads(line)
    if event['event'] != 'launched' or event.get('host', 'local') != 'local':
        continue
    gpu = event.get('gpu')
    parent = PARENT.get(gpu, gpu if str(gpu).startswith('GPU-') else None)
    if parent is None:
        unknown.append(event['id'])
    if 'gpu_parent' in event:
        assert event['gpu_parent'] == parent, event['id']
    if event['time'] >= CHANGE:
        assert gpu in AUTHORIZED, f"{event['id']} launched on {gpu} after the 2026-10-19 allowlist"
    runs.setdefault(event['id'], []).append({'attempt': event.get('attempt'), 'time': event['time'], 'host': 'local',
                                             'mig_uuid': gpu if str(gpu).startswith('MIG-') else None,
                                             'parent_gpu_uuid': parent})
out = ROOT / 'runs/setup/local-device-provenance.json'
out.write_text(json.dumps({'recorded_local': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
                           'mig_parent_mapping': PARENT, 'runs': runs, 'unmapped': unknown}, indent=1) + '\n')
print('runs', len(runs), 'launches', sum(map(len, runs.values())), 'unmapped', len(unknown),
      'by parent', Counter(a['parent_gpu_uuid'] for v in runs.values() for a in v).most_common())
