"""Four-arm local Meta-World campaign (arm4; user directive 2026-11-06 ~12:00 KST): the single definition of the arms,
their reference runs and row naming, shared by the audit (arm4_config_audit_20261106.py), the smoke
(arm4_smoke_local_20261106.py), the registration (register_arm4_eight_task_20261106.py) and the render-analysis entry
(reports/results-artifact/src/run_arm4_render_20261106.py). Read-only helpers; nothing here launches or writes.

Arms (each derived from a versioned reference's RESOLVED configuration; tests/test_arm4_configs.py and the audit assert
the exact resolved differences):
  A ocsw4  Original (C + D0 + M3D, method-frozen-v1 row arguments) with the encoder switched to CSW4 (enc_cs.yaml +
           enc_w4.yaml: patch-local conv stem, width 256, depth 4, 4 heads); everything else Original's.
  B ochf   Original + the bidirectional Chamfer loss at its registered settings (chamfer 5.0, chamfer_points 4096,
           chamfer_voxel_m 0.01; chamfer_lean recipe); hard depth, depth gradient and visibility stay at Original's values.
  C c50dg  cst50 (CSW4 + 50 % motion-aware tube, chamfer_lean recipe) + Original's multi-scale depth-gradient loss
           (loss.depth_grad 0.5); hard depth and visibility stay off.
  D c50vis cst50 + Original's visibility regulariser (loss.visibility 1.0); hard depth and depth gradient stay off.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
TASKS = ('hammer', 'pick-place', 'bin-picking', 'coffee-push', 'door-open', 'peg-insert-side', 'peg-unplug-side',
         'stick-push')
SEEDS = (2000, 2001, 2002)
V = 'configs/metaworld/variants/'
COMMON_SET = ['train.steps=200000', 'train.seed=0']
# method-frozen-v1's four registered overrides (the s3-m3d-seed0-<task>-200k rows; chamfer_lean.yaml repeats them)
ORIGINAL_SET = ['aug.synth_views=2', 'aug.synth_render=true', 'loss.depth_valid_far=true', 'loss.motion_space=gaussian']
CST50_FILES = [V + 'chamfer_lean.yaml', V + 'chamfer_lean_randmask.yaml', V + 'enc_cs.yaml', V + 'enc_w4.yaml']
CST50_SET = ['model.encoder.mask_mode=tube', 'model.encoder.mask_ratio=0.5']
CHAMFER_SET = ['loss.chamfer=5.0', 'loss.chamfer_points=4096', 'loss.chamfer_voxel_m=0.01']
ABSENT = '<absent>'  # a key missing from a resolved configuration (distinct from an explicit null)

ARMS = {
    'A': {'prefix': 'ocsw4', 'label': 'Original + CSW4 encoder', 'base': 'original',
          'files': [V + 'enc_cs.yaml', V + 'enc_w4.yaml'], 'set': [],
          'variant': 'original_csw4_encoder',
          # resolved difference vs the Original reference (flattened key: (reference, arm)); ABSENT = key missing
          'diff': {'model.encoder.stem': (ABSENT, 'patch_cnn'), 'model.encoder.width': (128, 256),
                   'model.encoder.depth': (8, 4), 'model.encoder.heads': (8, 4)}},
    'B': {'prefix': 'ochf', 'label': 'Original + Chamfer', 'base': 'original', 'files': [], 'set': CHAMFER_SET,
          'variant': 'original_chamfer',
          'diff': {'loss.chamfer': (ABSENT, 5.0), 'loss.chamfer_points': (ABSENT, 4096),
                   'loss.chamfer_voxel_m': (ABSENT, 0.01)}},
    'C': {'prefix': 'c50dg', 'label': 'cst50 + depth gradient', 'base': 'cst50', 'files': [], 'set': ['loss.depth_grad=0.5'],
          'variant': 'csw4_tube50_depthgrad', 'diff': {'loss.depth_grad': (0.0, 0.5)}},
    'D': {'prefix': 'c50vis', 'label': 'cst50 + visibility', 'base': 'cst50', 'files': [], 'set': ['loss.visibility=1.0'],
          'variant': 'csw4_tube50_visibility', 'diff': {'loss.visibility': (0.0, 1.0)}},
}
PREFIXES = tuple(a['prefix'] + '-' for a in ARMS.values())
# the comparison each arm answers (definition "comparisons"); the reference arm's run family
REFERENCE_OF = {'A': 'original', 'B': 'original', 'C': 'cst50', 'D': 'cst50'}
# matched initialisation (same seed and architecture -> bit-identical initial parameters, checked by the audit)
MATCHED_INIT = {'A': 'cst50', 'B': 'original', 'C': 'cst50', 'D': 'cst50'}
RENDER_MODELS = {'ocsw4': 'A', 'ochf': 'B', 'c50dg': 'C', 'c50vis': 'D', 'orig': 'original', 'cst50': 'cst50'}


def task_file(task: str) -> str:
    return f'configs/metaworld/tasks/{task}.yaml'


def pretrain_stack(arm: str, task: str) -> tuple[list[str], list[str]]:
    """(config files, --set overrides) of an arm's pretraining row, in row order."""
    spec = ARMS[arm]
    if spec['base'] == 'original':
        files = ['configs/metaworld/base.yaml', task_file(task)] + spec['files']
        sets = COMMON_SET + ORIGINAL_SET + spec['set']
    else:
        files = ['configs/metaworld/base.yaml', task_file(task)] + CST50_FILES + spec['files']
        sets = COMMON_SET + CST50_SET + spec['set']
    return files, sets


def reference_stack(family: str, task: str) -> tuple[list[str], list[str]]:
    """The reference rows' arguments as one 200k-from-scratch row (Original: the s3-m3d row form for every task)."""
    if family == 'original':
        return ['configs/metaworld/base.yaml', task_file(task)], COMMON_SET + ORIGINAL_SET
    return ['configs/metaworld/base.yaml', task_file(task)] + CST50_FILES, COMMON_SET + CST50_SET


def load(files: list[str], sets: list[str], root: Path = ROOT) -> dict:
    import sys
    sys.path.insert(0, str(root))
    from s4d.config import load_config
    return load_config([root / f for f in files], list(sets))


def flat(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f'{prefix}{k}.') if isinstance(v, dict) else {f'{prefix}{k}': v})
    return out


def keydiff(ref: dict, new: dict, ignore: tuple[str, ...] = ('run.',)) -> dict:
    fa, fb = flat(ref), flat(new)
    return {k: (fa.get(k, ABSENT), fb.get(k, ABSENT)) for k in sorted(set(fa) | set(fb))
            if not k.startswith(ignore) and fa.get(k, ABSENT) != fb.get(k, ABSENT)}


def config_sha256(cfg: dict) -> str:
    """scripts/train.py's provenance hash (ops/jobs/run_identity.numeric_config + canonical JSON)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from ops.jobs.run_identity import numeric_config
    cfg = copy.deepcopy(cfg)
    cfg.pop('run', None)
    return hashlib.sha256(json.dumps(numeric_config(cfg), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


# ---- reference runs --------------------------------------------------------------------------------------------------
def original_pretrain_dir(task: str) -> Path:
    """method-frozen-v1 pretraining that produced the Original RL encoders: hammer / pick-place i2-m3d-seed0-<task>
    (0-100k job i2-m3d-seed0-<task>, resumed 100k-200k by ls-a200-m3d-seed0-<task>, one 200k schedule), other tasks
    s3-m3d-seed0-<task>-200k."""
    if task in ('hammer', 'pick-place'):
        return ROOT / f'runs/pretrain/i2-m3d-seed0-{task}'
    return ROOT / f'runs/s3-m3d-seed0-{task}-200k'


def original_rows(task: str) -> list[str]:
    return [f'i2-m3d-seed0-{task}', f'ls-a200-m3d-seed0-{task}'] if task in ('hammer', 'pick-place') else \
        [f's3-m3d-seed0-{task}-200k']


def original_export(task: str) -> Path:
    if task in ('hammer', 'pick-place'):
        return ROOT / f'runs/exports/ls-a200-m3d-seed0-{task}-200k.pt'
    return ROOT / f'runs/s3-export-m3d-seed0-{task}-200k/encoder.pt'


def cst50_pretrain_dir(task: str) -> Path:
    local = ROOT / f'runs/pretrain/cst50-seed0-{task}-200k'
    return local if (local / 'config.yaml').is_file() else ROOT / f'runs/cst50-seed0-{task}-200k'


def cst50_export(task: str) -> Path:
    return ROOT / f'runs/cst50-export-seed0-{task}-200k/encoder.pt'


def reference_pretrain_dir(family: str, task: str) -> Path:
    return original_pretrain_dir(task) if family == 'original' else cst50_pretrain_dir(task)


def reference_drm(family: str, task: str, seed: int) -> str:
    return f's3-drm-splatter4d-{task}-s{seed}' if family == 'original' else f'cst50-drm-{task}-s{seed}'


def matched_init_reference(arm: str, task: str) -> Path:
    return reference_pretrain_dir(MATCHED_INIT[arm], task)


# ---- campaign row ids ------------------------------------------------------------------------------------------------
def ids(arm: str, task: str) -> dict:
    p = ARMS[arm]['prefix']
    return {'pretrain': f'{p}-seed0-{task}-200k', 'export': f'{p}-export-seed0-{task}-200k',
            'drm': [f'{p}-drm-{task}-s{s}' for s in SEEDS], 'eval': [f'{p}-drm-{task}-s{s}-eval' for s in SEEDS]}


def render_row_id(model: str, task: str) -> str:
    return f'ana-arm4-render-{model}-{task}'
