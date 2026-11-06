"""Four-arm campaign (arm4, user directive 2026-11-06): configuration audit before registration (CPU, read-only except
the output record). Writes runs/setup/arm4-config-audit-20261106.json.

1. Reference rows: arguments (queue), launch commits (registry), stored resolved configs and their config_sha256.
2. YAML drift: each reference row's arguments resolved under the release (this checkout) must equal the run's stored
   config.yaml (minus run.*).
3. Arms: per task the arm's resolved pretraining config vs the reference run's STORED resolved config; the flattened
   difference must equal the arm's intended keys exactly (arm4_common.ARMS[...]['diff']); for hammer / pick-place the
   Original reference additionally holds train.stop_step 200000 (its 100k + 100k resume under one 200k schedule; with
   train.steps 200000 the end step is the same, recorded explicitly).
4. Cross-checks: A's encoder block = cst50's except the explicit mask_mode 'tube' (EncoderConfig default 'tube'); B's
   Chamfer keys = cst50's and chl's; C's depth_grad and D's visibility = Original's; tube keep / motion quota.
5. Predicted config_sha256 per pretraining row (scripts/train.py's provenance hash).
6. Matched initialisation (full size, CPU): seed_everything(0) + build_model -> parameter hash per arm; A = C = D =
   cst50, B = Original.
7. Code defaults (AST) at every reference commit vs the release: literal defaults of cfg.get / get(cfg, ...), dataclass
   field defaults, function argument defaults and module constants in the training-path files; changed or removed
   entries are drift candidates.
8. Cross-commit training-path fingerprints (arm4_xcommit_fingerprint_20261106.py) for the Original config at 226d8ae /
   10b3116 / 3fba521 and the cst50 config at fd5e950 / 7643e9d vs the release.
9. DrM: the resolved RL config of a new trainer row (reference export path substituted) vs the stored reference DrM
   config (minus run.* and vision.export_path); RL code drift per reference commit.

    nice -n 15 .venv/bin/python -I runs/setup/tools/arm4_config_audit_20261106.py [--release <checkout>] [--skip-xcommit]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm4_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--release', default=str(C.ROOT), help='checkout whose s4d / configs the new rows run (default: main)')
ap.add_argument('--skip-xcommit', action='store_true')
ap.add_argument('--out', default=str(C.ROOT / 'runs/setup/arm4-config-audit-20261106.json'))
args = ap.parse_args()
REL = Path(args.release)
sys.path.insert(0, str(REL))
os.environ.setdefault('OMP_NUM_THREADS', '8')
REG = C.ROOT / 'experiments/registry.jsonl'
queue = yaml.safe_load((C.ROOT / 'experiments/queue.yaml').read_text())
by_queue = {j['id']: j for j in queue['jobs']}


def git(*a, root=REL):
    return subprocess.run(['git', '-C', str(root), *a], capture_output=True, text=True, check=True).stdout


release_commit = git('rev-parse', 'HEAD').strip()
launches = {}
for line in REG.open():
    if '"launched"' not in line:
        continue
    e = json.loads(line)
    launches.setdefault(e['id'], []).append({k: e.get(k) for k in ('host', 'attempt', 'commit', 'image_digest', 'gpu')})


def stored(path: Path) -> dict:
    return yaml.safe_load((path / 'config.yaml').read_text())


def row_stack(rid: str) -> tuple[list[str], list[str]]:
    a = by_queue[rid]['args']
    return a[a.index('--config') + 1:a.index('--name')], a[a.index('--set') + 1:]


out = {'recorded_at': datetime.now().astimezone().isoformat(), 'release_checkout': str(REL),
       'release_commit': release_commit, 'tasks': list(C.TASKS), 'arms': {}, 'references': {}}
problems = []

# ---- 1 + 2: references ------------------------------------------------------------------------------------------------
for task in C.TASKS:
    ref = {}
    odir = C.original_pretrain_dir(task)
    o_rows = C.original_rows(task)
    files, sets = row_stack(o_rows[-1])
    s = stored(odir)
    drift = C.keydiff(s, C.load(files, sets, REL))
    ref['original'] = {'rows': o_rows, 'row_args_last': {'files': files, 'set': sets},
                       'launches': {r: launches.get(r, []) for r in o_rows}, 'run_dir': str(odir.relative_to(C.ROOT)),
                       'checkpoint_200k': str((odir / 'checkpoints/step_0200000.pt').relative_to(C.ROOT)),
                       'checkpoint_200k_exists': (odir / 'checkpoints/step_0200000.pt').is_file(),
                       'export': str(C.original_export(task).relative_to(C.ROOT)), 'export_exists': C.original_export(task).is_file(),
                       'stored_config_sha256': s['run']['config_sha256'], 'stored_git_commit': s['run']['git_commit'],
                       'yaml_drift_vs_release': {k: list(v) for k, v in drift.items()}}
    if drift:
        problems.append(f'original {task}: reference row resolves differently under the release: {drift}')
    cdir = C.cst50_pretrain_dir(task)
    files, sets = row_stack(f'cst50-seed0-{task}-200k')
    s = stored(cdir)
    drift = C.keydiff(s, C.load(files, sets, REL))
    ref['cst50'] = {'rows': [f'cst50-seed0-{task}-200k'], 'row_args': {'files': files, 'set': sets},
                    'launches': launches.get(f'cst50-seed0-{task}-200k', [])[-1:], 'run_dir': str(cdir.relative_to(C.ROOT)),
                    'checkpoint_200k': str((cdir / 'checkpoints/step_0200000.pt').relative_to(C.ROOT)),
                    'checkpoint_200k_exists': (cdir / 'checkpoints/step_0200000.pt').is_file(),
                    'export': str(C.cst50_export(task).relative_to(C.ROOT)), 'export_exists': C.cst50_export(task).is_file(),
                    'stored_config_sha256': s['run']['config_sha256'], 'stored_git_commit': s['run']['git_commit'],
                    'yaml_drift_vs_release': {k: list(v) for k, v in drift.items()}}
    if drift:
        problems.append(f'cst50 {task}: reference row resolves differently under the release: {drift}')
    assert (files, sets) == C.reference_stack('cst50', task), (task, files, sets)
    out['references'][task] = ref

# ---- 3 + 4 + 5: arms ----------------------------------------------------------------------------------------------
from s4d.model.encoder import EncoderConfig, visible_count  # noqa: E402

for arm, spec in C.ARMS.items():
    rec = {'prefix': spec['prefix'], 'label': spec['label'], 'base': spec['base'], 'per_task': {}}
    for task in C.TASKS:
        files, sets = C.pretrain_stack(arm, task)
        cfg = C.load(files, sets, REL)
        ref_dir = C.reference_pretrain_dir(spec['base'], task)
        ref_cfg = stored(ref_dir)
        d = C.keydiff(ref_cfg, cfg)
        expected = dict(spec['diff'])
        if spec['base'] == 'original' and task in ('hammer', 'pick-place'):
            expected['train.stop_step'] = (200000, C.ABSENT)
        if d != expected:
            problems.append(f'{arm} {task}: resolved diff {d} != expected {expected}')
        # the same diff against the reference rows' stack resolved under the release (what tests/test_arm4_configs.py checks)
        d_release = C.keydiff(C.load(*C.reference_stack(spec['base'], task), root=REL), cfg)
        if d_release != spec['diff']:
            problems.append(f'{arm} {task}: diff vs the release-resolved reference stack {d_release} != {spec["diff"]}')
        enc = cfg['model']['encoder']
        rec['per_task'][task] = {'files': files, 'set': sets, 'config_sha256': C.config_sha256(cfg),
                                 'reference_run': str(ref_dir.relative_to(C.ROOT)),
                                 'reference_config_sha256': ref_cfg['run']['config_sha256'],
                                 'resolved_diff_vs_reference_run': {k: list(v) for k, v in d.items()},
                                 'tube': {'mask_mode': enc.get('mask_mode', EncoderConfig().mask_mode),
                                          'mask_ratio': enc['mask_ratio'],
                                          'visible_positions': visible_count(64, enc['mask_ratio']),
                                          'motion_quota': visible_count(64, enc['mask_ratio']) // 2}}
    out['arms'][arm] = rec

cross = {}
for task in C.TASKS:
    a = C.load(*C.pretrain_stack('A', task), root=REL)['model']['encoder']
    c = stored(C.cst50_pretrain_dir(task))['model']['encoder']
    cross.setdefault('A_encoder_vs_cst50_encoder', {})[task] = {k: list(v) for k, v in C.keydiff({'e': c}, {'e': a}).items()}
    b = C.load(*C.pretrain_stack('B', task), root=REL)['loss']
    chl_dir = next((d for d in (C.ROOT / f'runs/pretrain/chl-seed0-{task}-200k', C.ROOT / f'runs/chl-seed0-{task}-200k')
                    if (d / 'config.yaml').is_file()), None)
    chl = stored(chl_dir)['loss'] if chl_dir else {}
    cst = stored(C.cst50_pretrain_dir(task))['loss']
    keys = ('chamfer', 'chamfer_points', 'chamfer_voxel_m')
    cross.setdefault('B_chamfer_vs_cst50_and_chl', {})[task] = {k: [b.get(k), cst.get(k), chl.get(k)] for k in keys}
    if not all(b.get(k) == cst.get(k) == chl.get(k) for k in keys):
        problems.append(f'B {task}: chamfer settings differ from cst50 / chl')
    o = stored(C.original_pretrain_dir(task))['loss']
    cdg = C.load(*C.pretrain_stack('C', task), root=REL)['loss']
    dvis = C.load(*C.pretrain_stack('D', task), root=REL)['loss']
    cross.setdefault('C_depth_grad_D_visibility_vs_original', {})[task] = {
        'depth_grad': [cdg['depth_grad'], o['depth_grad']], 'visibility': [dvis['visibility'], o['visibility']],
        'C_hard_visibility': [cdg['depth_hard'], cdg['visibility']], 'D_hard_grad': [dvis['depth_hard'], dvis['depth_grad']]}
    if cdg['depth_grad'] != o['depth_grad'] or dvis['visibility'] != o['visibility'] or \
            (cdg['depth_hard'], cdg['visibility'], dvis['depth_hard'], dvis['depth_grad']) != (0.0, 0.0, 0.0, 0.0):
        problems.append(f'C/D {task}: restored coefficient or disabled terms wrong')
    if cross['A_encoder_vs_cst50_encoder'][task] != {'e.mask_mode': ['tube', C.ABSENT]}:
        problems.append(f'A {task}: encoder differs from cst50 beyond the explicit mask_mode')
cross['encoder_default_mask_mode'] = EncoderConfig().mask_mode
cross['encoder_default_stem'] = EncoderConfig().stem
if EncoderConfig().mask_mode != 'tube':
    problems.append('EncoderConfig default mask_mode is not tube')
out['cross_checks'] = cross

# ---- 6: matched initialisation (full size) --------------------------------------------------------------------------
import torch  # noqa: E402

torch.set_num_threads(8)
from s4d.train.loop import build_model, seed_everything  # noqa: E402


def init_hash(cfg):
    seed_everything(int(cfg['train']['seed']))
    m = build_model(cfg)
    h = hashlib.sha256()
    for name, p in m.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes() if p.dtype != torch.bool else p.numpy().tobytes())
    return h.hexdigest(), sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.encoder.parameters())


inits = {}
task = 'hammer'
for fam in ('original', 'cst50'):
    inits[fam] = init_hash(C.load(*C.reference_stack(fam, task), root=REL))
for arm in C.ARMS:
    inits[arm] = init_hash(C.load(*C.pretrain_stack(arm, task), root=REL))
out['matched_initialisation'] = {
    'protocol': 'train.seed 0, s4d.train.loop.seed_everything(seed + rank 0) then build_model (backbone scratch, no warm '
                'start); data order from the loader generator seeded with train.seed; the full 200k schedule',
    'task': task, 'hashes': {k: {'state_sha256': v[0], 'parameters': v[1], 'encoder_parameters': v[2]} for k, v in inits.items()},
    'equal_to_matched_reference': {arm: inits[arm][0] == inits[C.MATCHED_INIT[arm]][0] for arm in C.ARMS}}
if not all(out['matched_initialisation']['equal_to_matched_reference'].values()):
    problems.append('matched initialisation differs')

# ---- 7: code defaults (AST) -------------------------------------------------------------------------------------------
FILES = ['s4d/train/loop.py', 's4d/train/augment.py', 's4d/train/evaluate.py', 's4d/model/encoder.py',
         's4d/model/decoder.py', 's4d/model/render.py', 's4d/model/gaussians.py', 's4d/geometry.py', 's4d/config.py',
         's4d/losses/chamfer.py', 's4d/losses/depth.py', 's4d/losses/invariance.py', 's4d/losses/motion.py',
         's4d/losses/motion3d.py', 's4d/losses/occlusion.py', 's4d/losses/regularizers.py', 's4d/losses/rgb.py',
         's4d/data/contract.py', 's4d/data/metaworld/dataset.py', 'scripts/train.py', 'scripts/export_encoder.py']


def defaults(source: str, fname: str) -> dict:
    tree = ast.parse(source)
    found: dict[str, set] = {}

    def add(key, node):
        found.setdefault(f'{fname}:{key}', set()).add(ast.unparse(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == 'get' and len(node.args) == 2 and \
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                add(f'.get({node.args[0].value!r})', node.args[1])
            if isinstance(f, ast.Name) and f.id == 'get' and len(node.args) == 3 and \
                    isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                add(f'get(cfg,{node.args[1].value!r})', node.args[2])
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.value is not None and isinstance(item.target, ast.Name):
                    add(f'{node.name}.{item.target.id}', item.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pos = node.args.posonlyargs + node.args.args
            for a, dflt in zip(pos[len(pos) - len(node.args.defaults):], node.args.defaults):
                add(f'{node.name}({a.arg}=)', dflt)
            for a, dflt in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if dflt is not None:
                    add(f'{node.name}({a.arg}=)', dflt)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id.isupper():
            add(node.targets[0].id, node.value)
    return {k: sorted(v) for k, v in found.items()}


def tree_defaults(commit: str) -> dict:
    allk = {}
    for f in FILES:
        r = subprocess.run(['git', '-C', str(REL), 'show', f'{commit}:{f}'], capture_output=True, text=True)
        if r.returncode == 0:
            allk.update(defaults(r.stdout, f))
    return allk


REF_COMMITS = {'226d8ae': 'Original pretraining + export + DrM (6 remote tasks)', '10b3116': 'Original hammer / pick-place 0-100k',
               '3fba521': 'Original hammer / pick-place 100k-200k + exports', 'fd5e950': 'cst50 remote (5 tasks)',
               '7643e9d': 'cst50 local pretraining (3 tasks)', 'cec4b00': 'cst50 local exports + DrM'}
head = tree_defaults('HEAD')
ast_out = {'release_entries': len(head), 'files': FILES, 'per_reference_commit': {}}
for c, what in REF_COMMITS.items():
    old = tree_defaults(c)
    changed = {k: {'reference': old[k], 'release': head[k]} for k in old if k in head and old[k] != head[k]}
    removed = sorted(k for k in old if k not in head)
    added = sorted(k for k in head if k not in old)
    ast_out['per_reference_commit'][c] = {'what': what, 'entries': len(old), 'changed': changed, 'removed': removed,
                                          'added_count': len(added), 'added': added}
out['code_defaults_ast'] = ast_out

# ---- 8: cross-commit fingerprints ---------------------------------------------------------------------------------
if not args.skip_xcommit:
    harness = HERE / 'arm4_xcommit_fingerprint_20261106.py'
    py = str(C.ROOT / '.venv/bin/python')
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfgs = {'original': C.load(*C.reference_stack('original', 'bin-picking'), root=REL),
                'cst50': C.load(*C.reference_stack('cst50', 'bin-picking'), root=REL)}
        for k, v in cfgs.items():
            (tmp / f'{k}.json').write_text(json.dumps(v))
        plan = {'original': ['226d8ae', '10b3116', '3fba521'], 'cst50': ['fd5e950', '7643e9d']}
        envx = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        for fam, commits in plan.items():
            size = ['--size', '32'] if fam == 'cst50' else []
            for c in ['HEAD'] + commits:
                d = tmp / f'tree-{c}'
                d.mkdir()
                subprocess.run(f'git -C {REL} archive {c} s4d | tar -x -C {d}', shell=True, check=True)
                subprocess.run(['nice', '-n', '15', py, '-I', str(harness), str(d), str(tmp / f'{fam}.json'),
                                str(tmp / f'fp-{fam}-{c}.json'), *size], check=True, env=envx, capture_output=True)
                subprocess.run(['rm', '-rf', str(d)], check=True)
            head_fp = json.loads((tmp / f'fp-{fam}-HEAD.json').read_text())
            head_fp.pop('source_root')
            for c in commits:
                fp = json.loads((tmp / f'fp-{fam}-{c}.json').read_text())
                fp.pop('source_root')
                res[f'{fam}@{c}'] = {'bitwise_identical_to_release': fp == head_fp,
                                     'loss_terms': sorted({k.rsplit('_t', 1)[0] for k in fp['steps'][0]['losses']}),
                                     'steps': len(fp['steps'])}
                if fp != head_fp:
                    problems.append(f'cross-commit fingerprint {fam}@{c} differs from the release')
    out['cross_commit_fingerprints'] = {'harness': str(harness.relative_to(C.ROOT)),
                                        'harness_sha256': hashlib.sha256(harness.read_bytes()).hexdigest(),
                                        'config_task': 'bin-picking (task-independent at test size)', 'results': res}

# ---- 9: DrM ----------------------------------------------------------------------------------------------------------
from s4d.rl.protocol import resolve_config  # noqa: E402

drm = {}
for fam in ('original', 'cst50'):
    for task in C.TASKS:
        rid = C.reference_drm(fam, task, 2000)
        path = C.ROOT / 'runs' / rid / 'config.yaml'
        if not path.is_file():
            drm[f'{fam}/{task}'] = 'reference config missing'
            continue
        ref = yaml.safe_load(path.read_text())
        export = C.original_export(task) if fam == 'original' else C.cst50_export(task)
        cfg = resolve_config(task, 'splatter4d', 2000, ['train.num_train_steps=1000000',
                                                         f'vision.export_path={export.relative_to(C.ROOT)}'])
        d = C.keydiff(ref, cfg, ignore=('run.', 'vision.export_path'))
        drm[f'{fam}/{task}'] = {k: list(map(str, v)) for k, v in d.items()}
        if d:
            problems.append(f'DrM {fam} {task}: resolved RL config differs from the stored reference: {d}')
rl_files = ['s4d/rl', 'configs/rl', 'scripts/train_rl.py', 'scripts/eval_rl.py']
out['drm'] = {'resolved_config_diff_vs_reference_s2000': drm,
              'rl_code_changed_files': {c: git('diff', '--name-only', c, 'HEAD', '--', *rl_files).split()
                                        for c in ('226d8ae', 'fd5e950', 'cec4b00')}}
out['problems'] = problems
Path(args.out).write_text(json.dumps(out, indent=1, default=str) + '\n')
print(json.dumps({'problems': problems, 'init_equal': out['matched_initialisation']['equal_to_matched_reference'],
                  'xcommit': {k: v['bitwise_identical_to_release'] for k, v in out.get('cross_commit_fingerprints', {}).get('results', {}).items()},
                  'ast_changed': {c: len(v['changed']) for c, v in ast_out['per_reference_commit'].items()},
                  'ast_removed': {c: len(v['removed']) for c, v in ast_out['per_reference_commit'].items()}}, indent=1))
