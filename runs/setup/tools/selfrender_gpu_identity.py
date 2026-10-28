"""One csw4 training forward/backward on a real hammer batch with the real gsplat renderer (bf16, CUDA), from the
source tree given as argv[1]; prints loss bit patterns and parameter-gradient hashes."""
import hashlib, json, sys
root = sys.argv[1]
sys.path.insert(0, root)
import torch
from s4d.config import load_config
from s4d.data.contract import collate
from s4d.data.metaworld.dataset import MetaworldWindowDataset, split_episodes
from s4d.train.loop import build_model, forward_losses
files = [f"{root}/configs/metaworld/{p}" for p in ("base.yaml", "tasks/hammer.yaml", "variants/chamfer_lean.yaml",
         "variants/chamfer_lean_randmask.yaml", "variants/enc_cs.yaml", "variants/enc_w4.yaml")]
cfg = load_config(files)
path = "/home/ws/data/metaworld/splatter4d_v1/hammer.hdf5"
train, _ = split_episodes(path, 0.96, 0)
ds = MetaworldWindowDataset(path, train[:20], strides=(2, 4, 6), seed=0)
batch = collate([ds[i] for i in (0, 37, 101, 250)])
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
torch.manual_seed(0)
model = build_model(cfg).cuda().train()
torch.manual_seed(1); torch.cuda.manual_seed_all(1)
out = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1, 2, 3], device="cuda"))
out["total"].backward()
g = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
print(json.dumps({"losses": {k: float(v).hex() for k, v in sorted(out["losses"].items())},
                  "metrics": {k: float(v).hex() for k, v in sorted(out["metrics"].items())},
                  "grad_sha": hashlib.sha256(g.cpu().numpy().tobytes()).hexdigest(),
                  "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}))
