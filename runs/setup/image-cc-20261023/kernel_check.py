"""Functional check of the derived image: the Triton Chamfer kernel builds and runs, and agrees with the exact fallback
at the training shapes (48 clouds x 8192 centres vs 4096 target points)."""
import json, shutil, statistics, sys, time
sys.path.insert(0, '/workspace/splatter4d')
import torch
import triton
from s4d.losses import chamfer as cm

dev = torch.device('cuda', 0)
out = {'cc': shutil.which('gcc'), 'triton': triton.__version__, 'torch': torch.__version__,
       'gpu': torch.cuda.get_device_name(0), 'fused_available': cm.fused_available(dev)}
g = torch.Generator(device=dev).manual_seed(0)
scale = torch.tensor([0.8, 0.8, 0.4], device=dev)
c = (torch.rand(16, 3, 8192, 3, device=dev, generator=g) - 0.5) * scale
t = (torch.rand(16, 3, 4096, 3, device=dev, generator=g) - 0.5) * scale
v = torch.ones(16, 3, dtype=torch.bool, device=dev)
res = {}
for name, fused in (('fused', True), ('fallback', False)):
    cm._FUSED_OK[str(dev)] = fused
    times = []
    for _ in range(8):
        x = c.clone().requires_grad_(True)
        torch.cuda.synchronize(); s = time.time()
        loss = cm.chamfer_loss(x, t, v)['loss']
        loss.sum().backward()
        torch.cuda.synchronize(); times.append(time.time() - s)
    d, i = cm.nearest_neighbours(c.flatten(0, 1), t.flatten(0, 1), fused=fused)
    res[name] = (loss.detach(), x.grad.detach(), d, i, statistics.median(times[2:]))
a, b = res['fused'], res['fallback']
out.update(index_equal_fraction=float((a[3] == b[3]).float().mean()), max_abs_d2_diff=float((a[2] - b[2]).abs().max()),
           loss_max_rel_diff=float(((a[0] - b[0]).abs() / a[0].abs()).max()),
           grad_rel_diff=float((a[1] - b[1]).norm() / a[1].norm()),
           chamfer_fwd_bwd_s={'fused': a[4], 'fallback': b[4]})
print(json.dumps(out))
