"""Upload one new Meta-World task's training data to the remote (user directive 2026-10-28, new-task batches).

The four files a remote pretraining needs (remote_campaign_driver.py ``ready``): <task>.hdf5, splits/<task>_seed0.json,
heldout_sets/<task>.hdf5 and workspace_stats/<task>.json, uploaded and sha256-verified by ``RemoteClient.sync_data``
(manifest under runs/remote/data/<identity>.json). Appends ``MANIFEST <identity> <task> <seconds>`` to
runs/setup/remote-data-sync-ds-20261028.log, one of the driver's DATA_LOGS, so the files become remote targets.

    python -I upload_task_data_20261028.py <task>
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from s4d.remote_access import RemoteClient  # noqa: E402

DATA = Path('/home/ws/data/metaworld/splatter4d_v1')
LOG = ROOT / 'runs/setup/remote-data-sync-ds-20261028.log'
task = sys.argv[1]
files = [DATA / f'{task}.hdf5', DATA / f'splits/{task}_seed0.json', DATA / f'heldout_sets/{task}.hdf5',
         DATA / f'workspace_stats/{task}.json']
missing = [str(f) for f in files if not f.is_file()]
assert not missing, missing
assert json.loads((ROOT / 'docs/task_verification/summary.json').read_text())[task]['passed'] is True
start = time.time()
result = RemoteClient().sync_data({'files': [{'source': str(f)} for f in files]})
seconds = time.time() - start
with LOG.open('a') as stream:
    stream.write(f"MANIFEST {result['identity_sha256']} {task} {seconds:.0f}\n")
print(json.dumps({'task': task, 'identity': result['identity_sha256'], 'seconds': round(seconds),
                  'bytes': sum(f['bytes'] for f in result['files'])}))
