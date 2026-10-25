"""Behaviour preservation of a release for chlrm with every new flag off (encoder pilots, user directive 2026-10-25):
the resolved chlrm configuration and the outputs on a fixed real batch, computed by the code of ``--source`` (a clean
checkout or archive of one commit). Run once per source and compare the JSON outputs (--compare a.json b.json).

The batch: hammer training windows (fixed indices, B = 16) of the pretraining dataset; seed 0; the real CUDA renderer;
bf16 as in pretraining. Recorded: the resolved config (sha256 of its canonical JSON), the sample tensors, the step-0
forward (every loss, the states, the Gaussians), the gradients, and the forward / parameters after two AdamW steps.
GPU rasterisation backward uses atomics, so gradients and later steps can differ between two runs of the SAME code;
--repeat measures that run-to-run spread so a difference can be told from it.

    python -I enc_identity_check.py --source <root> --out <json> [--repeat]
    python -I enc_identity_check.py --compare <a.json> <b.json>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source')
ap.add_argument('--out')
ap.add_argument('--compare', nargs=2)
args = ap.parse_args()

if args.compare:
    a, b = (json.loads(Path(p).read_text()) for p in args.compare)
    keys = sorted(set(a) | set(b))
    report = {k: a.get(k) == b.get(k) for k in keys if k not in ('source', 'commit')}
    print(json.dumps(report, indent=1))
    raise SystemExit(0)

root = Path(args.source).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))
from s4d.gpu_guard import enforce_allowed_gpus  # noqa: E402

enforce_allowed_gpus()
import torch  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.data.contract import collate  # noqa: E402
from s4d.data.metaworld.dataset import MetaworldWindowDataset, split_episodes  # noqa: E402
from s4d.train.loop import build_model, build_optimizer, forward_losses, move_batch, seed_everything  # noqa: E402


def sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()).hexdigest()


files = [root / 'configs/metaworld/base.yaml', root / 'configs/metaworld/tasks/hammer.yaml',
         root / 'configs/metaworld/variants/chamfer_lean.yaml', root / 'configs/metaworld/variants/chamfer_lean_randmask.yaml']
cfg = load_config(files, ['train.steps=200000', 'train.seed=0'])
out = {'source': str(root), 'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
       'config': cfg}
path = Path(cfg['data']['root']) / 'hammer.hdf5'
train, _ = split_episodes(path, float(cfg['data']['train_ratio']), 0)
ds = MetaworldWindowDataset(path, train, strides=tuple(cfg['data']['strides']), seed=0)
picks = [i * 997 % len(ds) for i in range(16)]
batch = collate([ds[i] for i in picks])
out['batch'] = {k: sha(v) for k, v in sorted(batch.items()) if torch.is_tensor(v)}
device = torch.device('cuda')
batch = move_batch(batch, device)
seed_everything(0)
model = build_model(cfg).to(device)
model.decoder.set_anchor_statistics(json.loads(Path(cfg['model']['anchor_stats']).read_text()))
model.train()
optimizer, scheduler = build_optimizer(model, cfg)
out['init_params'] = sha(torch.cat([p.detach().flatten() for p in model.parameters()]))
for step in range(3):
    res = forward_losses(model, batch, cfg, step)
    out[f'step{step}_losses'] = {k: float(v).hex() for k, v in sorted(res['losses'].items())}
    out[f'step{step}_slots'] = sha(res['slots'])
    out[f'step{step}_gaussians'] = sha(torch.cat([res['gs'].xyz.flatten(), res['gs'].opacity.flatten(), res['gs'].rgb.flatten()]))
    if step == 2:
        break
    optimizer.zero_grad(set_to_none=True)
    res['total'].backward()
    out[f'step{step}_grads'] = sha(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['train']['grad_clip']))
    optimizer.step()
    scheduler.step()
    out[f'step{step + 1}_params'] = sha(torch.cat([p.detach().flatten() for p in model.parameters()]))
Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
print('wrote', args.out)
