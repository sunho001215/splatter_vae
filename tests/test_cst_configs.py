"""CSW4 tube-ratio arms (user directive 2026-11-04): the resolved pretraining configurations of cst50 and cst75 differ
only in model.encoder.mask_ratio; both are the CSW4 configuration with the inherited random masking overridden to the
existing motion-aware tube mode; the architecture is CSW4's (patch-local conv stem, width 256, depth 4, 4 heads)."""
from pathlib import Path

import torch

from s4d.config import load_config
from s4d.model.encoder import EncoderConfig, sample_tube_mask

REPO = Path(__file__).resolve().parents[1]
STACK = ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/hammer.yaml', 'configs/metaworld/variants/chamfer_lean.yaml',
         'configs/metaworld/variants/chamfer_lean_randmask.yaml', 'configs/metaworld/variants/enc_cs.yaml',
         'configs/metaworld/variants/enc_w4.yaml']
COMMON = ['train.steps=200000', 'train.seed=0']


def resolve(extra, stack=STACK):
    return load_config([REPO / p for p in stack], COMMON + extra)


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flat(v, key + '.'))
        else:
            out[key] = v
    return out


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


ARM = {0.5: ['model.encoder.mask_mode=tube', 'model.encoder.mask_ratio=0.5'],
       0.75: ['model.encoder.mask_mode=tube', 'model.encoder.mask_ratio=0.75']}


def test_arms_differ_only_in_mask_ratio():
    a, b = resolve(ARM[0.5]), resolve(ARM[0.75])
    assert diff(a, b) == {'model.encoder.mask_ratio'}
    assert a['model']['encoder']['mask_ratio'] == 0.5 and b['model']['encoder']['mask_ratio'] == 0.75
    assert a['model']['encoder']['mask_mode'] == b['model']['encoder']['mask_mode'] == 'tube'


def test_arms_are_csw4_with_tube_masking():
    csw4 = resolve([])
    assert csw4['model']['encoder']['mask_mode'] == 'random' and csw4['model']['encoder']['mask_ratio'] == 0.75
    assert diff(resolve(ARM[0.5]), csw4) == {'model.encoder.mask_mode', 'model.encoder.mask_ratio'}
    assert diff(resolve(ARM[0.75]), csw4) == {'model.encoder.mask_mode'}
    # arm A equals CSW4 without the random-masking file (the base tube setting at 0.5)
    no_rm = [p for p in STACK if 'randmask' not in p]
    plain = resolve([], no_rm)
    assert diff(resolve(ARM[0.5]), plain) == {'model.encoder.mask_mode'}  # explicit 'tube' vs the dataclass default
    assert 'mask_mode' not in plain['model']['encoder'] and EncoderConfig().mask_mode == 'tube'
    enc = resolve(ARM[0.75])['model']['encoder']
    assert (enc['stem'], enc['width'], enc['depth'], enc['heads']) == ('patch_cnn', 256, 4, 4)
    assert resolve(ARM[0.75])['loss']['chamfer'] == 5.0  # the chamfer_lean recipe


def test_tube_policy_quota_scales_with_visible_count():
    n = 64
    scores = torch.zeros(4, n)
    scores[:, :40] = torch.linspace(1.0, 0.5, 40)  # 40 moving patches above any threshold
    for ratio, keep in ((0.5, 32), (0.75, 16)):
        torch.manual_seed(0)
        visible = sample_tube_mask(scores, ratio, threshold=0.05)
        assert (visible.sum(1) == keep).all()  # shared spatial positions: (B, N), the same in every frame
        # the motion-priority quota is half of the visible positions: the keep // 2 highest-scoring moving patches
        assert visible[:, :keep // 2].all()
