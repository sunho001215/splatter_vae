import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs

path = ROOT / 'runs/setup/tools/remote_campaign_driver.py'
source = path.read_text()
tree = ast.parse(source)
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
admission = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.iter, ast.Name)
                 and n.iter.id == 'candidates')
queue = jobs.load_queue()
registry = jobs.read_registry()
by_id = {j['id']: j for j in queue['jobs']}
freeze = json.loads((ROOT / 'runs/setup/method-frozen-v1.json').read_text())
proof = json.loads((ROOT / 'runs/setup/stage3-final-registration-20261013.json').read_text())
remote = [j for j in queue['jobs'] if j.get('host') == 'remote']
gpus = tuple(queue['hosts']['remote']['limits'])
ns = {'Path': Path, 'json': json, 'hashlib': hashlib, 'copy': copy, 'jobs': jobs,
      'REPO': ROOT, 'FINISHED': ROOT / 'runs/remote/finished', 'freeze': freeze,
      'by_id': by_id, 'state': copy.deepcopy(registry), 'remote': remote,
      'GPUS': gpus, 'GPU_LOAD_LIMIT': 2.8, 'MAX_REVIWO': 8, 'CAP': 1000000000000,
      'RANK': {'proxy': 0, 'final_export': 0, 'baseline': 1, 'final_pretrain': 1, 'brl': 2, 'orl': 2, 'stage3': 3}}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
checks = []

def checked(name, condition):
    assert condition, name
    checks.append(name)

checked('registration counts 6+6+24+24', len(proof['jobs']) == 60 and
        sum(j['script'] == 'scripts/train.py' for j in proof['jobs']) == 6 and
        sum(j['script'] == 'scripts/export_encoder.py' for j in proof['jobs']) == 6 and
        sum(j['script'] == 'scripts/train_rl.py' for j in proof['jobs']) == 24 and
        sum(j['script'] == 'scripts/eval_rl.py' for j in proof['jobs']) == 24)
checked('all registered jobs preserve frozen metadata', all(
        j['freeze_manifest_sha256'] == proof['freeze_manifest_sha256'] and
        j['frozen_method'] == 'method-frozen-v1' for j in proof['jobs']))
checked('all final pretraining schedules are exactly 200k', all(
        ns['pretrain_target'](j) == 200000 and j['args'][j['args'].index('--set') + 1:] == freeze['overrides']
        for j in proof['jobs'] if j['script'] == 'scripts/train.py'))
checked('pretraining final checkpoint is mandatory synced evidence', all(
        'checkpoints/step_0200000.pt' in j['required_results']
        for j in proof['jobs'] if j['script'] == 'scripts/train.py'))
checked('RL seeds and agent steps unchanged', {j['args'][j['args'].index('--seed') + 1]
        for j in proof['jobs'] if j['script'] == 'scripts/train_rl.py'} == {'2000', '2001', '2002'} and
        all('train.num_train_steps=1000000' in j['args']
            for j in proof['jobs'] if j['script'] == 'scripts/train_rl.py'))
checked('exactly two development encoders reused', sum(e['reused'] for e in proof['encoders'].values()) == 2)
ns['uploaded'] = set()
for marker in (ROOT / 'runs/remote/exports').glob('*.json'):
    value = json.loads(marker.read_text())
    checked('uploaded export marker ' + value['export'], value['sha256'] ==
            hashlib.sha256((ROOT / value['export']).read_bytes()).hexdigest())
    ns['uploaded'].add(value['export'])
ns['targets'] = set()
for manifest in (ROOT / 'runs/remote/data').glob('*.json'):
    value = json.loads(manifest.read_text())
    ns['targets'] |= {f['target'].split('/splatter4d_v1/', 1)[1] for f in value['files']}
for job in proof['jobs']:
    k = ns['kind'](job['id'])
    if k == 'final_pretrain':
        checked('verified data ready ' + job['id'], ns['ready'](job))
    elif k == 'final_export':
        checked('export waits for verified pretrain closeout ' + job['id'], not ns['ready'](job))
    elif k == 'orl':
        task = by_id[job['id'].removesuffix('-eval')]['args'][1]
        checked('RL export gate ' + job['id'], ns['ready'](job) == proof['encoders'][task]['reused'])
for job in queue['jobs']:
    if job['id'].startswith('s3spec-'):
        checked('speculative 300k admission suppressed ' + job['id'], not ns['ready'](job))
        checked('speculative schedule preserved ' + job['id'], ns['pretrain_target'](job) == 300000)

trainer = by_id['s3-drm-splatter4d-hammer-s2000']
evaluator = by_id[trainer['id'] + '-eval']
compiled_admission = compile(ast.Module(body=[admission], type_ignores=[]), str(path), 'exec')

def admit(candidates, loads, memories=None, reserved=0):
    ns.update(candidates=candidates, gpu_load=dict(zip(gpus, loads)),
              gpu_mem=dict(zip(gpus, memories or [0, 0, 0, 0])), chosen=[], taken=set(),
              total=reserved, reviwo=0)
    exec(compiled_admission, ns)
    return [j['id'] for j in ns['chosen']]

checked('trainer admitted even when evaluator does not fit',
        admit([trainer, evaluator], [2.25, 2.8, 2.8, 2.8]) == [trainer['id']])
checked('trainer RAM is reserved alone', ns['total'] == 8 * 2**30)
ns['state'][trainer['id']] = {'status': jobs.RUNNING}
checked('evaluator admitted independently on a later pass',
        admit([evaluator], [2.45, 2.8, 2.8, 2.8]) == [evaluator['id']])
ns['state'][trainer['id']] = {'status': jobs.PENDING}
checked('evaluator never launched before its trainer', admit([evaluator], [0, 0, 0, 0]) == [])
checked('memory-full least-loaded GPU does not hide another fitting GPU',
        admit([trainer], [0, 1, 2.8, 2.8], [90, 0, 0, 0]) == [trainer['id']] and
        ns['chosen'][0]['gpu'] == gpus[1])
heavy = next(j for j in proof['jobs'] if ns['kind'](j['id']) == 'final_pretrain')
small = next(j for j in proof['jobs'] if ns['kind'](j['id']) == 'final_export')
checked('large RAM job cannot block later fitting export',
        admit([heavy, small], [0, 0, 0, 0], reserved=ns['CAP'] - 5 * 2**30) == [small['id']])
ns['state'][trainer['id']] = {'status': jobs.RUNNING}
other = by_id['s4-drm-reviwo-hammer-s2001']
checked('active evaluators first and seeds interleaved across methods',
        [j['id'] for j in sorted([other, trainer, evaluator], key=ns['dispatch_key'])] ==
        [evaluator['id'], trainer['id'], other['id']])
checked('GPU load ceiling never relaxed for final jobs',
        admit([trainer], [2.4, 2.4, 2.4, 2.4]) == [])
checked('no replay deletion inside the driver (lifecycle cleanup is scripts/cleanup_replay.py, user 2026-10-28)',
        'rm -rf' not in source and "remote_replay='lifecycle-cleanup'" in source)
checked('parent script compiles without executing it', bool(compile(source, str(path), 'exec')))
print(json.dumps({'passed': len(checks), 'checks': checks, 'script_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'scope': 'scheduler/configuration fixtures only; no native component substitution or new upstream execution',
                  'full_native_acceptance': False}, indent=2))
