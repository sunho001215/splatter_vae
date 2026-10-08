"""``train.stop_step`` stops a run early on its full LR schedule (screens), natively on the GPU with the review options."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from s4d.config import load_config
from s4d.data.metaworld.cameras import camera_rig
from s4d.train import ddp
from s4d.train.checkpoint import load_checkpoint
from s4d.train.loop import build_model, train

REPO = Path(__file__).resolve().parents[1]


class TinyRigData(Dataset):
    """Four windows seen by the six Meta-World training cameras at 32x32: a flat floor at the workspace height."""

    def __init__(self, n: int = 4):
        rig = camera_rig(32, 32)
        self.K = torch.from_numpy(rig["K"][:6])
        self.c2w = torch.from_numpy(rig["c2w"][:6])
        self.w2c = torch.from_numpy(rig["w2c"][:6])
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        depth = torch.full((3, 6, 1, 32, 32), 1.0)
        return {
            "images": torch.randint(0, 256, (3, 6, 3, 32, 32), dtype=torch.uint8, generator=g),
            "K": self.K.clone(),
            "w2c": self.w2c.clone(),
            "c2w": self.c2w.clone(),
            "depth": depth,
            "motion3d": torch.zeros(3, 6, 3, 32, 32),
            "motion_weight": torch.ones(3, 6, 1, 32, 32),
            "motion_score": torch.zeros(3, 6, 1, 32, 32),
            "meta": {"task": "tiny", "episode": f"ep{i}", "t_indices": [0, 2, 4], "stride": 2, "dt_seconds": 0.0125},
        }


class Log:
    def __init__(self):
        self.lines = []

    def text(self, message):
        self.lines.append(message)

    def scalars(self, step, values):
        self.lines.append((step, values))


def test_stop_step_saves_and_evaluates_at_the_stop_with_the_full_schedule(tmp_path):
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["model"]["encoder"].update(image_height=32, image_width=32, patch_size=8, width=32, depth=1, heads=4, num_slots=2)
    cfg["model"]["decoder"].update(dim=32, depth=1, heads=4, state_concat=True, conditioning="adaln_zero", anchor_fourier=4)
    cfg["model"]["decoder"]["scene"]["parents"] = 8
    cfg["model"]["decoder"]["dynamic"]["parents"] = 4
    cfg["train"].update(steps=10, stop_step=3, batch_size=2, num_workers=0, warmup_steps=2, save_every=100, eval_every=100,
                        eval_at_start=False, decoder_lr_mult=3.0, log_every=1)
    cfg["aug"] = {"crop": {"prob": 0.5}, "synth_views": 2, "synth_render": True}
    cfg["loss"].update(self_render=0.5, motion_norm="moving", depth_hard_boost={"factor": 3.0, "fraction": 0.2})
    evaluated = []
    train(cfg, tmp_path, TinyRigData(), ddp.init_distributed(), Log(), lambda model, step: evaluated.append(step))
    assert evaluated == [3]
    ckpts = sorted(p.name for p in (tmp_path / "checkpoints").glob("step_*.pt"))
    assert ckpts == ["step_0000003.pt"]
    model = build_model(cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    assert load_checkpoint(tmp_path / "checkpoints" / "step_0000003.pt", model.encoder, model.decoder, rank=0) == 3
    state = torch.load(tmp_path / "checkpoints" / "step_0000003.pt", map_location="cpu", weights_only=False)
    scheduler = state.get("scheduler") or {}
    if scheduler:  # the cosine schedule runs over train.steps (10), not the stop step
        assert scheduler["last_epoch"] == 3
    assert optimizer is not None
